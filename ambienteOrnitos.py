import time
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from collections import deque

xml_path = "Modelos XML/Current_Model.xml"

# AMBIENTE
class OrnitoEnv(gym.Env):
    def __init__(self, num_steps = 1e3):
        super(OrnitoEnv, self).__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.num_steps = num_steps

        self.cont_fis = 0

        # Atuadores (5 motores)
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        
        # Observação: ((12 sensores + 5 ações) * 25 frames) + (30 pontos futuro * 3 coords) = 515
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(515,), dtype=np.float32)

        # Filtro e Frequências
        self.dt_ia = 0.0195

        # O relógio da FÍSICA idêntico ao do XML
        self.dt_sim = self.model.opt.timestep # (0.0015)(666.6Hz) 
        self.alpha_sim = (2*np.pi*7.0*self.dt_sim)/(2*np.pi*7.0*self.dt_sim + 1) # 7.o é a frequência de corte em Hz

        self.passos_de_fisica_por_ia = int(self.dt_ia / self.dt_sim) # 13 passos

        self.last_policy_action = np.zeros(5)
        self.current_motor_target = np.zeros(5) # Guarda a posição suave atual dos servos

        self.history = deque(maxlen=25)

        self.limiter_deltas = self._calculate_max_deltas() # Calcula os limites de variação para os motores com base no XML

        self.render_mode = False

        self.vel_target = 3.8

        #### Parâmetros iniciais para o currículo: ####

        self.enable_curriculo = False  # Ativa o currículo de dificuldade progressiva durante o reset do ambiente

        self.global_step = 0  # Inicia o relógio zerado

        # Configuração dos limites da rampa
        self.passo_final_rampa = self.num_steps - 5e3  # A rampa dura 3M steps, e valores ficam constantes depois disso
        
        # Gravidade: Começa na Lua (-1.62) e termina na Terra (-9.81)
        self.g_inicial = -9.81
        self.g_final = -1.62

        self.bonus_sobrevivencia = 0.0

        # Velocidade da aeronave no começo do episódio
        self.vel_inicial = self.vel_target/4.0
        self.vel_final = self.vel_target
        self.vel_atual = self.vel_inicial

        # Penalidade de energia, para incentivar comportamentos mais frenéticos no começo e mais econômicos no final
        self.ener_inicial = -0.001
        self.ener_final = -0.05
        self.energia_atual = self.ener_final

        self.pos_bonification = 0.5 # 0.5 no artigo

        # Kutta Lift: Começa super alto (8.0) e termina no valor real do artigo (3.14) 
        self.ck_inicial = 8.0
        self.ck_final = 3.14

        self.cd_blunt_inicial = 0.4
        self.cd_blunt_final = 0.2

        self.cd_slender_inicial = 0.01
        self.cd_slender_final = 0.6

        # self.assign_coefs_to_surfs([self.cd_blunt_inicial, self.cd_slender_inicial, 1.5, self.ck_final, 1.0, 0, 0, 0, 0, 0, 0, 0])

    def set_global_step(self, step_atual):
        # O Callback chama isso a cada frame apenas para atualizar o número de steps atual
        self.global_step = step_atual

    def set_render_mode(self, mode:bool):
        self.render_mode = mode
        if mode:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        elif not mode:
            self.viewer = None

    def set_curriculo(self):
        if self.global_step <= self.passo_final_rampa:
            # Calcula o fator de interpolação linear (alfa vai de 0.0 a 1.0)
            alfa = min(self.global_step / self.passo_final_rampa, 1.0)

            # Equação da rampa: valor = inicial + alfa * (final - inicial)
            # self.model.opt.gravity[2] = self.g_inicial + alfa * (self.g_final - self.g_inicial)
            # self.vel_atual = self.vel_inicial + alfa * (self.vel_final - self.vel_inicial)
            self.energia_atual = self.ener_inicial + alfa * (self.ener_final - self.ener_inicial)
            # ck_atual = self.ck_inicial + alfa * (self.ck_final - self.ck_inicial)
            cd_blunt_atual = self.cd_blunt_inicial + alfa * (self.cd_blunt_final - self.cd_blunt_inicial)
            cd_slender_atual = self.cd_slender_inicial + alfa * (self.cd_slender_final - self.cd_slender_inicial)

            # self.assign_coefs_to_surfs([cd_blunt_atual, cd_slender_atual, 1.5, self.ck_final, 1.0, 0, 0, 0, 0, 0, 0, 0])

            # if self.global_step % 50_000 == 0:
            #     print(f"[CURRÍCULO] Passo: {self.global_step} | Gravidade Z: {self.model.opt.gravity[2]:.2f} | C_K: {ck_atual:.2f} | Velocidade Inicial: {self.vel_atual:.2f} | Penalt Energia: {self.energia_atual:.5f}")

    def _calculate_max_deltas(self):
        # A ordem aqui DEVE ser a mesma que a sua IA devolve na variável 'action'
        motores = ['motor_q1', 'motor_q2', 'motor_q3', 'motor_q4', 'motor_q5']
        deltas = np.zeros(5)
        
        v_max_rad_s = np.deg2rad(2400) # Converte 600*4 graus/s para rad/s
        max_rad_per_step = v_max_rad_s*self.dt_ia

        for i, nome in enumerate(motores):
            try:
                a_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, nome)
                ctrl_range = self.model.actuator_ctrlrange[a_id] # Retorna [min, max] em radianos
                
                # Amplitude física total do motor (ex: max - min)
                amplitude_rad = np.deg2rad(ctrl_range[1] - ctrl_range[0])
            
                # Regra de 3: Converte o limite físico para a escala normalizada da ação
                deltas[i] = (max_rad_per_step / amplitude_rad) * 2.0
            except:
                print(f"Erro: Atuador '{nome}' não encontrado no modelo.")
        
        # print(f"Delta motores calculados | {deltas}")
        return deltas

    def assign_coefs_to_surfs(self, coefs:list):
        # IDs das geoms (certifique-se que os nomes batem com o seu XML)
        asas = ['wing_l', 'wing_r', 'tail'] # nomes das geoms

        # Coeficientes: [Sustentação, Arrasto, Momento, Kutta_Lift, Escala_Força, ...]
        # MuJoCo usa 12 slots internos. Vamos preencher os 5 primeiros.
        novos_coefs = coefs

        for nome in asas:
            try:
                g_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, nome)
                self.model.geom_fluid[g_id] = novos_coefs
                # print(".cf")
                # print(f"Sucesso: Coeficientes de {nome} atualizados manualmente.")
            except:
                print(f"Erro: Geom '{nome}' não encontrada no modelo.")

        # Verifique o coeficiente de arrasto da primeira asa (geom id ou nome)
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'wing_l')
        # print(f"Coeficientes do fluido da asa: {self.model.geom_fluid[geom_id]}")

        mujoco.mj_resetData(self.model, self.data)


    def _get_obs(self):
        # Extrai a matriz de rotação Global -> Local (Transposta do torso)
        mat_global_to_local = self.data.body('torso').xmat.reshape(3, 3).T

        # Pega a gravidade global e projeta para o referencial do pássaro (3 valores)
        gravity_global = np.array([0.0, 0.0, -1.0])
        g_local = mat_global_to_local @ gravity_global

        pqr = self.data.qvel[3:6]  # Vel Angular
        qj = self.data.qpos[7:12]  # Juntas
        
        vel_global = self.data.qvel[:3]
        vel_local = mat_global_to_local @ vel_global
        vx = np.array([vel_local[0]]) 
        
        # Concatena g_local(3) + pqr(3) + qj(5) + vx(1) + last_action(5) = 17 valores
        current_obs = np.concatenate([g_local, pqr, qj, vx, self.last_policy_action])
        self.history.append(current_obs)
        
        # Trajetória Futura (Look-ahead: 30 pontos)
        lim_fut = 30
        future_traj = []
        for i in range(1, lim_fut + 1):
            t_futuro = self.data.time + (i * self.dt_ia)

            # Vetor de erro no referencial GLOBAL (Alvo voando em X)
            pos_f_global = np.array([self.vel_target * t_futuro, 0.0, 12.0]) - self.data.qpos[:3]
            # pos_f_global = np.array([0.0, self.vel_target * t_futuro, 12.0]) - self.data.qpos[:3]

            # Projeta o erro para o referencial LOCAL da aeronave
            pos_f_local = mat_global_to_local @ pos_f_global
            future_traj.append(pos_f_local)
        
        return np.concatenate([np.array(self.history).flatten(), np.array(future_traj).flatten()])

    def step(self, action):

        # 1. Rate Limiter (A 50Hz): Limita o quão brusca a política pode ser de um passo pro outro
        target_action = np.clip(action, self.last_policy_action - self.limiter_deltas, self.last_policy_action + self.limiter_deltas)
        self.last_policy_action = target_action

        # 2. Loop da Física (A 666Hz)
        for _ in range(self.passos_de_fisica_por_ia):
            # Filtro Passa-Baixa INTERPOLADO a cada passo do MuJoCo
            self.current_motor_target = self.alpha_sim * target_action + (1 - self.alpha_sim) * self.current_motor_target
            
            # Envia a curva suave para os motores
            self.data.ctrl[:] = self.current_motor_target

            # Atualiza alvo no MuJoCo para visualização
            self.data.mocap_pos[0] = [self.vel_target * self.data.time, 0.0, 12.0]
            # self.data.mocap_pos[0] = [0.0, self.vel_target * self.data.time, 12.0]
            # self.data.mocap_pos[1] = [0.0, self.vel_target * self.data.time, 12.0] # Alteração da posição da visualização do raio de morte
            mujoco.mj_step(self.model, self.data)

            if np.any(np.abs(self.data.qvel) > 150) or np.any(np.isnan(self.data.qpos)):
                # self.data.qpos[2] = 0.0
                print("Ops... Um erro infinito/Nan aconteceu.")
                obs = self.reset()[0] # Força um reset seguro para limpar os NaNs
                return obs, -100.0, True, False, {} # Retorna punição máxima e encerra

            # Sincroniza o visualizador a cada passo de física se estiver ativo

            if self.render_mode and self.viewer.is_running():
                self.viewer.sync()
                self.viewer.cam.lookat = self.data.body('torso').xpos
                self.viewer.cam.distance = 10.0
                # self.cont_fis += 1
                # if self.cont_fis % 100 == 0:
                #     print(f"Tempo: {self.data.time}, step_fis: {self.cont_fis}")
                time.sleep(0.00195 * 1.5)

        obs = self._get_obs()
        reward = self._compute_reward(target_action)
        
        lim_area = 3.0

        dist = np.linalg.norm(self.data.qpos[:3] - self.data.mocap_pos[0])
        bateu_no_chao = self.data.qpos[2] < 1.0
        terminated = bool(dist > lim_area or bateu_no_chao) # Fora da área

        if terminated:
            reward -= 30.0 # Punição fixa por morte

        truncated = False
        
        return obs, reward, terminated, truncated, {}

    def _compute_reward(self, action):
        # 1. Rastreamento de Posição (Gaussiana)
        # O uso do exp() garante que o erro máximo seja assintótico a 0, 
        # evitando o "suicídio" da IA por punições infinitas (-dist^2).
        dist = np.linalg.norm(self.data.qpos[:3] - self.data.mocap_pos[0])
        r_pos = np.exp(-0.2 * (dist**2)) # Retorna 1 se perfeito, cai suavemente para 0 se longe

        # 2. Estabilidade Angular (Penaliza apenas Roll e Pitch, permite Yaw)
        # Usamos a matriz de rotação do torso para extrair os ângulos reais de forma limpa
        mat = self.data.body('torso').xmat.reshape(3, 3)
        pitch = np.arcsin(np.clip(-mat[2, 0], -1.0, 1.0))
        roll = np.arctan2(mat[2, 1], mat[2, 2])
        r_att = np.exp(-2.0 * (pitch**2 + roll**2)) # 1 quando nivelado, decai se inclinar

        # 3. Taxa de Rotação do Corpo (Suavidade)
        r_omega = np.exp(-0.1 * np.sum(np.square(self.data.qvel[3:6])))

        # Alternativa de Energia: penalizar a variação da ação em relação ao step anterior
        delta_action = action - self.last_policy_action 
        r_en = self.energia_atual*np.sum(np.square(delta_action))

        # Retorna o somatório com os pesos baseados no artigo
        return (self.pos_bonification * r_pos) + (0.2 * r_att) + (0.1 * r_omega) + r_en + self.bonus_sobrevivencia

    def reset(self, seed = None, options = None):

        self.cont_fis = 0

        if self.enable_curriculo:
            self.set_curriculo()

        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[2] = 12.0  # Altitude de spawn

        lim_hist = 25
        num_sensors = 17 # 12 sensores + 5 ações anteriores

        # self.data.qpos[3:7] = [0.7071068, 0.0, 0.0, 0.7071068] # Rotação em +90° em Z
        # self.data.qpos[3:7] = [0.0, 0.0, 0.0, 1.0]

        # Dá empulso na direção +X
        self.data.qvel[0] = self.vel_atual
        # self.data.qvel[1] = self.vel_atual
        # self.data.qvel[2] = self.vel_atual

        self.history.clear()

        # Preenche com zeros se o histórico ainda não estiver cheio
        while len(self.history) < lim_hist:
            self.history.append(np.zeros(num_sensors))

        self.last_policy_action = np.zeros(5)
        self.current_motor_target = np.zeros(5)

        # ATUALIZA A FÍSICA CINEMÁTICA ANTES DO PASSO 1 (Evita os erros NaN)
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def close(self):
        # Boa prática: fechar o visualizador ao encerrar o script
        if self.viewer is not None:
            self.viewer.close()


# TESTAR ASPECTOS INDIVIDUAIS DO AMBIENTE
if __name__ == "__main__":
    import numpy as np
    import time
    
    # 1. Instancia o ambiente apenas para teste local
    print("Iniciando ambiente em modo de teste...")
    env = OrnitoEnv() 
    obs, info = env.reset()

    print("\n--- Verificação do Setup ---")
    print(f"Limites de Delta Calculados (q1 a q5): {env.limiter_deltas}")
    
    # 2. Criamos uma ação de teste isolada
    # Exemplo: Comanda ação máxima (1.0) APENAS no índice 0 (esperado: motor_q1)
    acao_teste = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    print("\n--- Injetando Ação Teste ---")
    print(f"Ação enviada pelo 'cérebro': {acao_teste}")

    # Roda alguns steps para o filtro passa-baixa e o integrador processarem
    for _ in range(10): 
        obs, reward, terminated, truncated, info = env.step(acao_teste)
        
    # 3. Verifica o que chegou de fato nos atuadores do MuJoCo
    print(f"Buffer de controle do MuJoCo (data.ctrl): {env.data.ctrl[:5]}")
    
    # Verifica o estado da física (posição das juntas)
    # Os índices exatos dependem do seu XML, mas qpos geralmente guarda as posições
    print(f"Velocidades resultantes (data.qvel): {env.data.qvel[:10]}") 
    
    env.close()