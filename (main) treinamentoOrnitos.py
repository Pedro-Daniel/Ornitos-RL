import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv 
from stable_baselines3.common.monitor import Monitor

from ambienteOrnitos import OrnitoEnv

class RelogioGlobalCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RelogioGlobalCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Apenas injeta o tempo global como uma variável dentro do ambiente
        self.training_env.env_method("set_global_step", self.num_timesteps)
        return True

def obter_proximo_nome_run(base_name, dir_models, dir_logs):
    nome_teste = base_name
    contador = 1
    
    while os.path.exists(os.path.join(dir_models, nome_teste)) or os.path.exists(os.path.join(dir_logs, nome_teste)):
        nome_teste = f"{base_name}_{contador}"
        contador += 1
    return nome_teste

def criar_ambiente(num_steps):
    def _init():
        # Passa os mesmos passos que você já tinha
        env = OrnitoEnv(num_steps)
        # IMPORTANTE! NUNCA HABILITE A VISUALIZAÇÃO CASO ESTIVER FAZENDO TREINAMENTO COM MAIS DE UMA INSTÂNCIA! ISSO PROVAVELMENTE VAI TRAVAR O COMPUTADOR! SE FIZER PARE IMEDIATAMENTE!
        env.set_render_mode(True)  # Desative a renderização para o treinamento
        env = Monitor(env)  # Envolve o ambiente com Monitor para registrar estatísticas do episódio
        env = gym.wrappers.TimeLimit(env, max_episode_steps=5000)
        return env
    return _init

# SCRIPT DE TREINAMENTO
if __name__ == "__main__":
    models_dir = "C:/Users/pedro/Desktop/Ornitos/models"
    logdir = "C:/Users/pedro/Desktop/Ornitos/logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    NUM_STEPS = 6e6 # Total de steps para o treinamento

    LEARN_RATE = 3e-4

# --- ANOTAÇÕES DO EXPERIMENTO ---
    NOTAS_EXPERIMENTO = """
    Objetivo: Verificar possibilidades de flapeio
    Mudanças: 
    - Modelo XML alterado para ter margem de fase de 15%, verificar se o PPO é capaz de gerar um flapeio similar ao modelo de margem estática de 70%.
    - A posição da asa foi alterada diretamente no XML e a angulação no tempo foi avaliada pelo algoritmo de testePlaneio.py
    - 6 milhões de steps
    - O modelo antigo acabou voando como um 14 bis... Essa é uma segunda tentativa de voo com os mesmos parâmetros.
    """

    # --- CHAVES DE CONTROLE DE TREINAMENTO---
    CONTINUAR_TREINO = True  # True = Continua o treino de onde parou (na mesma pasta ou em outra), False = Inicia do zero 
    MESMA_PASTA = False  # False = Transfusão de consciência (zera os steps, nova pasta, mantém o cérebro)

    NOME_BASE = "PPO_Voo_Reto_novo" # Nome para a pasta nova, usada se CONTINUAR_TREINO = False.
    NOME_RUN_ANTIGA = "PPO_Voo_Reto_novo_23" # Nome da pasta do modelo que eu quero continuar treinando (MESMA_PASTA = True) ou fazer a transfusão de consciência (MESMA_PASTA = False).

    # Usada apenas quando eu quiser fazer transfusão de consciência.
    path_do_ultimo_checkpoint = f"{models_dir}/PPO_Voo_Reto_novo_23/PPO_Voo_Reto_novo_23_4199832_steps.zip" # Caminho dentro da pasta de modelos com o nome do modelo que eu quero carregar.

    NUM_ENVS = 1
    print(f"Iniciando treinamento com {NUM_ENVS} pássaros simultâneos. Agora vai!")
    env = SubprocVecEnv([criar_ambiente(NUM_STEPS) for _ in range(NUM_ENVS)])

    # --- LÓGICA DE INSTANCIAÇÃO DO MODELO ---
    if CONTINUAR_TREINO and os.path.exists(path_do_ultimo_checkpoint):
        
        if MESMA_PASTA:
            print(f"Cérebro carregado! Continuando o treino na MESMA pasta ({NOME_RUN_ANTIGA})...")
            model = PPO.load(path_do_ultimo_checkpoint, env=env, learning_rate=LEARN_RATE)
            reset_timesteps = False
            nome_run_final = NOME_RUN_ANTIGA
            
        else:
            print(f"Transfusão de Consciência! Cérebro carregado. Iniciando NOVA pasta ({NOME_BASE}) e zerando os steps...")
            # Carrega os pesos, mas define o diretório de log para ele criar a nova run
            model = PPO.load(path_do_ultimo_checkpoint, env=env, tensorboard_log=logdir, learning_rate=LEARN_RATE)
            reset_timesteps = True
            nome_run_final = obter_proximo_nome_run(NOME_BASE, models_dir, logdir)
            print(f"--> Auto-incremento detectou nova run. Usando o nome: {nome_run_final}")

    else:
        if CONTINUAR_TREINO:
            print("Aviso: Checkpoint não encontrado. Iniciando um modelo do ZERO.")
        else:
            print("Iniciando um novo modelo do ZERO por escolha.")
        
        # Expansão das hidden layers de 64 (standard) para 256, visto que o modelo tem 540 entradas e 4 saídas.
        size_rede = dict(net_arch = dict(pi = [256, 256], vf = [256, 256]))

        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=size_rede,
            verbose=1,
            learning_rate=LEARN_RATE,
            n_steps=2048,
            batch_size=64
        )
        reset_timesteps = True
        nome_run_final = obter_proximo_nome_run(NOME_BASE, models_dir, logdir)

    # Cria uma subpasta dentro de 'models' com o mesmo nome da run
    pasta_checkpoints = f"{models_dir}/{nome_run_final}"
    os.makedirs(pasta_checkpoints, exist_ok=True)

# --- SALVAMENTO DAS NOTAS CIENTÍFICAS ---
    caminho_notas = f"{pasta_checkpoints}/notas_experimento_{nome_run_final}.txt"
    with open(caminho_notas, "w", encoding="utf-8") as f:
        f.write(f"=== REGISTRO DO EXPERIMENTO: {nome_run_final} ===\n")
        f.write("Data/Hora: " + os.popen("date /t").read().strip() + " " + os.popen("time /t").read().strip() + "\n\n")
        
        f.write("--- NOTAS DO PESQUISADOR ---\n")
        f.write(NOTAS_EXPERIMENTO.strip() + "\n\n")
        
        f.write("--- HIPERPARÂMETROS AUTOMÁTICOS ---\n")
        f.write(f"Total Steps: {NUM_STEPS}\n")
        f.write(f"Continuar Treino: {CONTINUAR_TREINO}\n")
        # Se você tiver variáveis do ambiente aqui, pode logar também:
        f.write(f"Learning Rate:{model.learning_rate}\n")
        f.write(f"Batch Size:{model.batch_size}\n")
    # ----------------------------------------

    caminho_log_exato = f"{logdir}/{nome_run_final}"
    novo_logger = configure(caminho_log_exato, ["stdout", "tensorboard"])
    model.set_logger(novo_logger)

    # Configura Callback dos logs
    checkpoint_callback = CheckpointCallback(
        save_freq=100000//NUM_ENVS,  # Salva a cada 100k steps (ajustado para o número de ambientes)
        save_path=pasta_checkpoints,
        name_prefix=nome_run_final
    )

    callback_relogio = RelogioGlobalCallback()
    meus_callbacks = [checkpoint_callback, callback_relogio]
    # meus_callbacks = [checkpoint_callback]

    # --- INÍCIO DO TREINAMENTO ---
    print("Iniciando Treinamento!")
    model.learn(
        total_timesteps = NUM_STEPS,
        reset_num_timesteps = reset_timesteps,
        callback = meus_callbacks  # Para realizar treinamento de currículo
    )

    # --- SALVAMENTO FINAL ---
    if CONTINUAR_TREINO:
        print("Treinamento de continuação/transfusão finalizado. Salvando o progresso final...")
        model.save(f"{models_dir}/ornito_model_novo_final_continuacao")
    else:
        print("Treinamento do zero finalizado. Salvando modelo...")
        model.save(f"{models_dir}/final_model_ornito_novo")
        model.save(f"{pasta_checkpoints}/{nome_run_final}_final")

    env.close()

# cd C:/Users/pedro/Desktop/Ornitos
# python -m tensorboard.main --logdir=logs