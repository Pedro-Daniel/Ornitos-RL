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
    def __init__(self, num_steps = 5e3):
        super(OrnitoEnv, self).__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.num_steps = num_steps

        # Atuadores (5 motores)
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        
        # Observação: ((13 sensores + 5 ações) * 25 frames) + (30 pontos futuro * 3 coords) = 540
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(540,), dtype=np.float32)

        # Variáveis para gravação de dados para simular comportamento no ESP32
        self.buffer_sensores = []
        self.gravacao_concluida = False

        # Contador de episódios para carrossel de direções em torno de Z
        self.ep_counter = 0
        self.multidir_counter = 0

        # Variável para armazenar a direção do voo no episódio atual
        self.heading_angle = 0.0

        # Filtro e Frequências
        self.dt_ia = 0.0195

        # O relógio da FÍSICA idêntico ao do XML
        self.dt_sim = self.model.opt.timestep # (0.0015)(666.6Hz) 

        self.alpha_sim = 1.0 - np.exp(-2 * np.pi * 7.0 * self.dt_sim) # 7.0 é a frequência de corte em Hz. Discretizado por ZOH.
        # self.alpha_sim = (2 * np.pi * 7.0 * self.dt_sim) / (2 * np.pi * 7.0 * self.dt_sim + 1)

        self.passos_de_fisica_por_ia = int(self.dt_ia / self.dt_sim) # 13 passos

        self.last_policy_action = np.zeros(5)
        self.current_motor_target = np.zeros(5) # Guarda a posição suave atual dos servos

        self.history = deque(maxlen=25)

        self.limiter_deltas = self._calculate_max_deltas() # Calcula os limites de variação para os motores com base no XML

        self.render_mode = False
        self.viewer = None

        self.vel_target = 3.8



        #### Parâmetros iniciais para currículo: ####

        self.enable_curriculo = False  # Ativa o currículo de dificuldade progressiva durante o reset do ambiente

        self.global_step = 0  # Inicia o relógio zerado

        # Configuração dos limites da rampa
        self.passo_final_rampa = self.num_steps - 5e3  # A rampa dura 3M steps, e valores ficam constantes depois disso

        # Velocidade da aeronave no começo do episódio
        self.vel_inicial = self.vel_target



        # Variáveis de Navegação 3D
        self.ultimo_tempo_comando = 0.0
        self.target_heading = 0.0 # Direção XY (Yaw)
        self.target_gamma = 0.0 # Ângulo de Subida/Descida (Pitch)

        self.lim_fut = 30
        
        # BUFFER DE TRAJETÓRIA (1 atual + 30 futuros)
        self.target_buffer = deque(maxlen=self.lim_fut+1)
        self.pos_alvo_gerador = np.array([0.0, 0.0, 50.0]) # A "ponta" que desenha o caminho

        self.ep_step = 0

    ################ MÉTODOS DO AMBIENTE ################


    def _exportar_amostras_c(self, obs, max_amostras=100, nome_arquivo="sensores_voo.h"):
        # Não faz nada se já gravou o arquivo neste episódio/execução
        if self.gravacao_concluida:
            return

        self.buffer_sensores.append(obs)

        # Após atingir o limite de amostras
        if len(self.buffer_sensores) >= max_amostras:
            with open(nome_arquivo, 'w') as f:
                f.write("#ifndef SENSORES_VOO_H\n#define SENSORES_VOO_H\n\n")
                f.write(f"// Look-up table gerada automaticamente pelo MuJoCo\n")
                f.write(f"// Duração: ~{max_amostras * self.dt_ia:.2f} segundos ({max_amostras} amostras)\n\n")
                
                f.write(f"const int NUM_AMOSTRAS = {max_amostras};\n")
                f.write(f"const int NUM_SENSORES = {len(obs)};\n\n")
                
                f.write(f"const float sensores_voo[{max_amostras}][{len(obs)}] = {{\n")
                
                for amostra in self.buffer_sensores:
                    # Converte cada valor para float com 6 casas decimais e formata para C++
                    linha = ", ".join([f"{val:.6f}f" for val in amostra])
                    f.write(f"    {{{linha}}},\n")
                
                f.write("};\n\n#endif // SENSORES_VOO_H\n")
            
            print(f"\n[EXPORTAÇÃO] Matriz de {max_amostras}x{len(obs)} salva com sucesso em '{nome_arquivo}'!")
            self.gravacao_concluida = True



    def set_global_step(self, step_atual):
        # O Callback chama isso a cada frame apenas para atualizar o número de steps atual
        self.global_step = step_atual


    def set_curriculo(self):
        if self.global_step <= self.passo_final_rampa:
            # Calcula o fator de interpolação linear (alfa vai de 0.0 a 1.0)
            alfa = min(self.global_step / self.passo_final_rampa, 1.0)

            # Expande o leque de arfagem conforme o treinamento avança
            self.pitch_max_atual = np.deg2rad(15.0) + alfa * (np.deg2rad(45.0) - np.deg2rad(15.0))
            self.pitch_min_atual = np.deg2rad(-15.0) + alfa * (np.deg2rad(-60.0) - np.deg2rad(-15.0))

            # if self.ep_step % 50_000 == 0:
            #     print(f"[CURRÍCULO] Passo: {self.ep_step} | Gravidade Z: {self.model.opt.gravity[2]:.2f}")


    def set_render_mode(self, mode:bool):
        self.render_mode = mode
        if mode:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        elif not mode:
            self.viewer = None

    # Esta função será reciclada para a randomização de domínio!
    # def assign_coefs_to_surfs(self, coefs:list):
    #     # IDs das geoms (certifique-se que os nomes batem com o seu XML)
    #     asas = ['wing_l', 'wing_r', 'tail'] # nomes das geoms

    #     # Coeficientes: [Sustentação, Arrasto, Momento, Kutta_Lift, Escala_Força, ...]
    #     # MuJoCo usa 12 slots internos. Vamos preencher os 5 primeiros.
    #     novos_coefs = coefs

    #     for nome in asas:
    #         try:
    #             g_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, nome)
    #             self.model.geom_fluid[g_id] = novos_coefs
    #             # print(".cf")
    #             # print(f"Sucesso: Coeficientes de {nome} atualizados manualmente.")
    #         except:
    #             print(f"Erro: Geom '{nome}' não encontrada no modelo.")

    #     # Verifique o coeficiente de arrasto da primeira asa (geom id ou nome)
    #     geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'wing_l')
    #     # print(f"Coeficientes do fluido da asa: {self.model.geom_fluid[geom_id]}")

    #     mujoco.mj_resetData(self.model, self.data)


    def _sortear_novo_comando(self, flag="protocolar"):
        # Apenas ajusta a matemática da IA. O gerador desenhará isso no futuro.
        self.ultimo_tempo_comando = self.data.time
        
        if flag == "protocolar":
            cronograma = [
                (6.0, 15.0),
                (12.0, 30.0)
                # (18.0, -15.0),
                # (24.0, 0.0),
                # (30.0, 15.0),
                # (36.0, 0.0),
                # (42.0, -15.0),
                # (48.0, 0.0),
                # (54.0, 15.0),
                # (60.0, 0.0),
                # (66.0, -15.0),
                # (72.0, 0.0),
                # (78.0, 15.0),
                # (84.0, -15.0),
                # (90.0, 0.0)
            ]

            # Encontra em qual estágio do tempo estamos
            novo_angulo_deg = 0.0
            for tempo_transicao, angulo in reversed(cronograma):
                if self.data.time >= tempo_transicao:
                    novo_angulo_deg = angulo
                    break
                    
            self.target_gamma = np.deg2rad(novo_angulo_deg)

        elif flag == "aleatorio":
            limite_pitch_absoluto_max = np.deg2rad(45.0)
            limite_pitch_absoluto_min = np.deg2rad(-60.0)
            limite_delta = np.deg2rad(15.0)

            gamma_candidato = np.random.uniform(limite_pitch_absoluto_min, limite_pitch_absoluto_max)
            self.target_gamma = np.clip(gamma_candidato, self.target_gamma - limite_delta, self.target_gamma + limite_delta)
        
        # NOVO CÁLCULO DA VELOCIDADE VARIÁVEL
        if self.target_gamma < 0:
            # Mergulho ganha velocidade com a gravidade. 
            # Em -60° (sin(-60) = -0.866), usamos fator 0.45 para gerar ~5.3 m/s.
            fator_gravidade = 1.0 - (0.45 * np.sin(self.target_gamma))
        else:
            # Subida mantém a velocidade cinética nominal constante (fator 1.0)
            fator_gravidade = 1.0
        
        self.vel_dinamica_target = self.vel_target * fator_gravidade


    def _avancar_gerador_alvo(self):
        # Cria apenas 1 ponto novo no futuro (dt_ia) e adiciona na ponta do vetor
        v_xy = self.vel_dinamica_target * np.cos(self.target_gamma)
        v_z = self.vel_dinamica_target * np.sin(self.target_gamma)
        
        delta_x = v_xy * self.dt_ia * np.cos(self.target_heading)
        delta_y = v_xy * self.dt_ia * np.sin(self.target_heading)
        delta_z = v_z * self.dt_ia
        
        self.pos_alvo_gerador += np.array([delta_x, delta_y, delta_z])


    def _calculate_max_deltas(self):
        # A ordem aqui DEVE ser a mesma que a sua IA devolve na variável 'action'
        motores = ['motor_q1', 'motor_q2', 'motor_q3', 'motor_q4', 'motor_q5']
        deltas = np.zeros(5)
        
        v_max_rad_s = np.deg2rad(600) # Converte 600 graus/s para rad/s
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


    def _compute_reward(self, action):
        # 1. Rastreamento de Posição (Gaussiana)
        # O uso do exp() garante que o erro máximo seja assintótico a 0, 
        # evitando o "suicídio" da IA por punições infinitas (-dist^2).
        dist = np.linalg.norm(self.data.qpos[:3] - self.data.mocap_pos[0])
        r_pos = np.exp(-0.2 * (dist**2)) # Retorna 1 se perfeito, cai suavemente para 0 se longe

        # # 2. Estabilidade Angular (Média Móvel do Buffer Preditivo)
        # # Converte o deque para array NumPy para cálculos vetorizados rápidos
        # buffer_np = np.array(self.target_buffer)
        # # Calcula os deltas (dx, dy, dz) entre cada um dos 30 pontos consecutivos
        # diffs = buffer_np[1:] - buffer_np[:-1] 
        # # Extrai os 30 ângulos de inclinação (pitch) do futuro
        # angulos_futuros = np.arctan2(diffs[:, 2], diffs[:, 0]) 
        # # A média gradativa
        # gamma_gradual = np.mean(angulos_futuros) 
        # # Matriz de rotação e ângulos reais do ornitóptero
        # mat = self.data.body('torso').xmat.reshape(3, 3)
        # pitch = np.arcsin(np.clip(-mat[2, 0], -1.0, 1.0))
        # roll = np.arctan2(mat[2, 1], mat[2, 2])
        # # O erro agora contra uma rampa suave
        # erro_pitch = pitch - gamma_gradual
        # r_att = np.exp(-2.0 * (erro_pitch**2 + roll**2))

        # 2. Estabilidade Angular (Antecipação Extrema / Preditiva)
        mat = self.data.body('torso').xmat.reshape(3, 3)
        pitch = np.arcsin(np.clip(mat[2, 0], -1.0, 1.0))
        roll = np.arctan2(mat[2, 1], mat[2, 2])
        
        # O pássaro é cobrado para alinhar sua atitude IMEDIATAMENTE com 
        # a intenção futura do gerador (target_gamma), 0.6s antes da ladeira chegar.
        erro_pitch = pitch - self.target_gamma
        r_att = np.exp(-2.0 * (erro_pitch**2 + roll**2))

        # 3. Taxa de Rotação do Corpo (Suavidade)
        r_omega = np.exp(-0.1 * np.sum(np.square(self.data.qvel[3:6])))

        # Alternativa de Energia: penalizar a variação da ação em relação ao step anterior
        delta_action = action - self.last_policy_action
        r_en = np.sum(np.square(delta_action))

        # Retorna o somatório com os pesos baseados no artigo
        return 0.5*r_pos + 0.1*r_omega + 0.2*r_att - 0.05*r_en


    def _get_obs(self):
        # Extrai a matriz de rotação Global -> Local (Transposta do torso)
        mat_global_to_local = self.data.body('torso').xmat.reshape(3, 3).T

        # Pega a gravidade global e projeta para o referencial do pássaro (3 valores)
        # gravity_global = np.array([0.0, 0.0, -1.0])
        # g_local = mat_global_to_local @ gravity_global

        # Extrai o quatérnio [w, x, y, z] diretamente do MuJoCo
        quat = self.data.qpos[3:7]

        pqr = self.data.qvel[3:6]  # Vel Angular
        qj = self.data.qpos[7:12]  # Juntas
        
        vel_global = self.data.qvel[:3]
        vel_local = mat_global_to_local @ vel_global
        vx = np.array([vel_local[0]]) 
        
        # Concatena quat(4) + pqr(3) + qj(5) + vx(1) + last_action(5) = 18 valores
        current_obs = np.concatenate([quat, pqr, qj, vx, self.last_policy_action])
        self.history.append(current_obs)
        
        # Trajetória Futura (Look-ahead: 30 pontos)
        future_traj = []
        for i in range(1, self.lim_fut + 1):
            # i=0 é a posição atual, i=1 a 30 são os futuros
            pos_f_global = self.target_buffer[i] - self.data.qpos[:3]
            pos_f_local = mat_global_to_local @ pos_f_global
            future_traj.append(pos_f_local)
        
        return np.concatenate([np.array(self.history).flatten(), np.array(future_traj).flatten()])

    def step(self, action):

        # 1. Rate Limiter (A 50Hz): Limita o quão brusca a política pode ser de um passo pro outro
        target_action = np.clip(action, self.last_policy_action - self.limiter_deltas, self.last_policy_action + self.limiter_deltas)
        self.last_policy_action = target_action

        self.ep_step += 1

        # 2. Lógica do Alvo e do Buffer (Atualiza a 50Hz)
        # if (self.data.time - self.ultimo_tempo_comando) >= 6.0:
        #     # Só permite curvas após 6 segundos de voo estabilizado
        #     if self.data.time >= 6.0:
        #         self._sortear_novo_comando("protocolar")
        #     else:
        #         # Renova o tempo, mas mantém a trajetória reta (gamma = 0)
        #         self.ultimo_tempo_comando = self.data.time

        # Remove a bola antiga e gera uma nova na ponta do corredor
        self.target_buffer.popleft()
        self._avancar_gerador_alvo()
        self.target_buffer.append(np.copy(self.pos_alvo_gerador))

        # Atualiza a visualização das esferas para você (0 é a vermelha atual, 1-5 as laranjas)
        self.data.mocap_pos[0] = self.target_buffer[0]
        # # self.data.mocap_pos[6] = self.target_buffer[0] # Mockup área de morte
        self.data.mocap_pos[1] = self.target_buffer[6]
        self.data.mocap_pos[2] = self.target_buffer[12]
        self.data.mocap_pos[3] = self.target_buffer[18]
        self.data.mocap_pos[4] = self.target_buffer[24]
        self.data.mocap_pos[5] = self.target_buffer[30]

        # 2. Loop da Física (A 666Hz)
        for _ in range(self.passos_de_fisica_por_ia):
            # Filtro Passa-Baixa INTERPOLADO a cada passo do MuJoCo
            self.current_motor_target = self.alpha_sim * target_action + (1 - self.alpha_sim) * self.current_motor_target
            
            # Envia a curva suave para os motores
            self.data.ctrl[:] = self.current_motor_target

            mujoco.mj_step(self.model, self.data)

            if np.any(np.abs(self.data.qvel) > 150) or np.any(np.isnan(self.data.qpos)):
                # self.data.qpos[2] = 0.0
                print("Ops... Um erro infinito/Nan aconteceu.")
                obs = self.reset()[0] # Força um reset seguro para limpar os NaNs
                return obs, -100.0, True, False, {} # Retorna punição máxima e encerra

            # Sincroniza o visualizador a cada passo de física se estiver ativo
            if self.render_mode and self.viewer.is_running():
                self.viewer.sync()
                self.viewer.cam.lookat = self.data.body('target').xpos
                self.viewer.cam.distance = 6.0
                time.sleep((self.dt_ia/13.0)/1.0)
            
            # Mostra o pitch atual a cada momento.
            # mat = self.data.body('torso').xmat.reshape(3, 3)
            # pitch = np.arcsin(np.clip(mat[2, 0]))
            # print(f"Pitch: {np.rad2deg(pitch):.1f}°")

        obs = self._get_obs()
        reward = self._compute_reward(target_action)
        
        # Descomente a linha abaixo para gravar os primeiros 100 steps do voo em um arquivo .h
        # self._exportar_amostras_c(obs, max_amostras=100)

        lim_area = 3.0

        # Extrai a orientação atual no step
        mat = self.data.body('torso').xmat.reshape(3, 3)
        pitch = np.arcsin(np.clip(mat[2, 0], -1.0, 1.0))
        roll = np.arctan2(mat[2, 1], mat[2, 2])
        
        # Limite de 90 graus (pi/2 radianos)
        morte_orientacao = abs(pitch) > (np.pi/2.0) or abs(roll) > (np.pi/2.0)

        dist = np.linalg.norm(self.data.qpos[:3] - self.data.mocap_pos[0])
        
        # O episódio termina se sair da área OU se capotar/empinar 90°
        # terminated = bool(dist > lim_area)
        terminated = bool(dist > lim_area or morte_orientacao)

        if terminated:
            reward -= 30.0 # Punição fixa por morte

        truncated = bool(self.ep_step >= 5000)
        
        if truncated:
            print("Episódio concluído com sucesso!")
            reward += 50.0 # Bônus de conclusão do episódio máximo

        return obs, reward, terminated, truncated, {}


    def reset(self, seed = None, options = None):
        self.ep_step = 0

        if self.enable_curriculo:
            self.set_curriculo()

        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[2] = 50.0  # Altitude de spawn

        # --- LÓGICA DO CARROSSEL MULTIDIRECIONAL ---

        if self.ep_counter % 10 == 0: # 1 episódio fixo + 9 aleatórios
            # Distribui as 9 direções ortogonais/diagonais.
            angulos_fixos = [
                0.0, 0.25*np.pi, 0.5*np.pi, 0.75*np.pi, np.pi,
                -np.pi, -0.75*np.pi, -0.5*np.pi, -0.25*np.pi
            ]
            self.heading_angle = angulos_fixos[self.multidir_counter]
            self.multidir_counter = (self.multidir_counter + 1) % 9
        else:
            # 9 episódios completamente aleatórios
            self.heading_angle = np.random.uniform(-np.pi, np.pi)
        
        self.ep_counter += 1
        
        # print(f"Ángulo do episódio (graus): {np.rad2deg(self.heading_angle)}")
        self.heading_angle = 0.0
        self.target_heading = self.heading_angle
        # -------------------------------------------
        
        # Converte o ângulo Z em um Quatérnio [w, x, y, z] para rotacionar o pássaro
        half_angle = self.heading_angle / 2.0
        self.data.qpos[3:7] = [np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)]

        # Dá empuxo decomposto na direção do novo rumo
        self.data.qvel[0] = self.vel_inicial * np.cos(self.heading_angle)
        self.data.qvel[1] = self.vel_inicial * np.sin(self.heading_angle)

        lim_hist = 25
        num_sensors = 18 # 13 sensores + 5 ações anteriores
        self.history.clear()

        # Preenche com zeros se o histórico ainda não estiver cheio
        while len(self.history) < lim_hist:
            self.history.append(np.zeros(num_sensors))

        self.last_policy_action = np.zeros(5)
        self.current_motor_target = np.zeros(5)

        # ATUALIZA A FÍSICA CINEMÁTICA ANTES DO PASSO 1 (Evita os erros NaN)
        mujoco.mj_forward(self.model, self.data)

        # Define a âncora EXATAMENTE onde o pássaro nasceu
        self.pos_ancora_alvo = [0.0, 0.0, 50.0]

        # # --- LÓGICA DA PAREDE ---
        # # A ladeira de -15° começa aos 6 segundos. A 3.8m/s, isso dá 22.8 metros de distância.
        # # Colocamos a parede aos 25.5 metros (quase 3 metros APÓS a quina da descida).
        # dist_parede = (19 * self.vel_inicial)
        
        # x_parede = dist_parede * np.cos(self.heading_angle)
        # y_parede = dist_parede * np.sin(self.heading_angle)
        
        # # O Centro da parede fica em 51.6m. 
        # # Como ela tem 2.0m de "raio" vertical, a borda inferior fica exatamente em 49.6m.
        # # Se o pássaro voar reto (50.0m), ele bate. Se ele descer a -15°, ele passa por baixo (aprox 49.2m).
        # z_parede = 51.75
        
        # # Quatérnio para girar a parede e deixá-la de frente para o pássaro
        # quat_w = np.cos(self.heading_angle / 2.0)
        # quat_z = np.sin(self.heading_angle / 2.0)
        
        # # O MuJoCo mapeia os mocaps na ordem do XML. O obstacle_wall será o índice 6.
        # self.data.mocap_pos[6] = [x_parede, y_parede, z_parede]
        # self.data.mocap_quat[6] = [quat_w, 0.0, 0.0, quat_z]
        # # -----------------------------------

        self.ultimo_tempo_comando = self.data.time
        
        self.target_gamma = 0.0
        self.vel_dinamica_target = self.vel_target

        # Define a âncora EXATAMENTE onde o pássaro nasceu
        self.pos_alvo_gerador = np.array([0.0, 0.0, 50.0])
        self.target_buffer.clear()

        # Preenche o buffer com a posição atual e os 30 passos futuros iniciais
        self.target_buffer.append(np.copy(self.pos_alvo_gerador))
        for _ in range(30):
            self._avancar_gerador_alvo()
            self.target_buffer.append(np.copy(self.pos_alvo_gerador))

        # Posiciona todas as esferas corretamente no instante t=0
        self.data.mocap_pos[0] = self.target_buffer[0]
        self.data.mocap_pos[1] = self.target_buffer[6]
        self.data.mocap_pos[2] = self.target_buffer[12]
        self.data.mocap_pos[3] = self.target_buffer[18]
        self.data.mocap_pos[4] = self.target_buffer[24]
        self.data.mocap_pos[5] = self.target_buffer[30]

        return self._get_obs(), {}

    def close(self):
        # Boa prática: fechar o visualizador ao encerrar o script
        if self.viewer is not None:
            self.viewer.close()


# TESTAR ASPECTOS INDIVIDUAIS DO AMBIENTE
if __name__ == "__main__":    
    # 1. Instancia o ambiente apenas para teste local
    print("Você clicou no botão Run errado! Rode '(main) treinamentoOrnitos.py' ou 'visualizacaoOrnitos.py'para treinar ou visualizar a IA.")