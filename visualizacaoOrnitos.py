import gymnasium as gym
from stable_baselines3 import PPO

from ambienteOrnitos import OrnitoEnv

env = OrnitoEnv()
env.set_render_mode(True)  # Ativa a renderização para visualização
env = gym.wrappers.TimeLimit(env, max_episode_steps=5000)

# 2. Carregar o modelo treinado
model_path = "C:/Users/pedro/Desktop/Ornitos/models/PPO_Voo_Reto_novo_25/PPO_Voo_Reto_novo_25_3899844_steps.zip"
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