import gymnasium as gym
from stable_baselines3 import PPO

from ambienteOrnitos import OrnitoEnv

env = OrnitoEnv()
env.set_render_mode(True)  # Ativa a renderização para visualização
env = gym.wrappers.TimeLimit(env, max_episode_steps=5000)

# 2. Carregar o modelo treinado
model_path = "C:/Users/pedro/Desktop/Ornitos/models/PPO_Sobe_Desce_9/PPO_Sobe_Desce_9_1299948_steps.zip" #Melhor exemplo voo reto
# model_path = "C:/Users/pedro/Desktop/Ornitos/models/PPO_Voo_Reto_novo_32 (15pc)c/PPO_Voo_Reto_novo_32 (15pc)c_1099956_steps.zip" # Treinado 600°/s
# model_path = "C:/Users/pedro/Desktop/Ornitos/models/PPO_Sobe_Desce_1/PPO_Sobe_Desce_1_1999920_steps"
model = PPO.load(model_path, env=env, learning_rate=1e-4)

# 3. Loop de Execução (Sem treino, apenas ação)
obs, info = env.reset()

while True: 
    # A IA decide a ação com base no que "vê" (obs)
    # deterministic=True garante que ela use a melhor jogada, sem aleatoriedade
    action, _states = model.predict(obs, deterministic=True)
    
    # Executa a ação no simulador
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()