import os
import time
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from stable_baselines3 import PPO

# DEFINIÇÃO DO MODELO XML
xml_model = """
<mujoco model="Ornithopter_UnB_Final">
    <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
    
    <statistic center="0 0 12" extent="2"/>
    
    <option timestep="0.0015" integrator="RK4" gravity="0 0 -9.81" density="1.225" viscosity="0.000018">
        <flag energy="enable"/>
    </option>

    <asset>
        <material name="carbon_fiber" rgba="0.1 0.1 0.1 0.4"/>
        <material name="wing_mylar" rgba="0 0.5 0.8 0.6"/>
        <material name="joint_visual" rgba="1 0 0 1"/>
        <material name="motor_internal" rgba="0.5 0 0 1"/>
    </asset>

    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="100 100 0.1" rgba=".9 .9 .9 1"/>

        <body name="target" mocap="true" pos="0 0 12">
            <geom type="sphere" size="0.1" rgba="1 0 0 0.5" contype="0" conaffinity="0"/>
        </body>

        <!-- CORPO PRINCIPAL -->
        <body name="torso" pos="0 0 12">
            <freejoint name="root"/>

            <!-- Fuselagem Principal (GeoGebra): 255g total, motores inclusos aqui -->
            <geom name="fuselagem" type="ellipsoid" fluidshape="ellipsoid" pos="0.12995 0 0" size="0.1565 0.035 0.028" mass="0.210" material="carbon_fiber"/>

            <!-- Massas dos Motores (9g cada) posicionadas internamente para ajuste de CG -->
            <geom name="m_flap_l_mass" type="sphere" pos="0.09 0.02 0" size="0.01" mass="0.009" material="motor_internal"/>
            <geom name="m_flap_r_mass" type="sphere" pos="0.09 -0.02 0" size="0.01" mass="0.009" material="motor_internal"/>
            <geom name="m_pitch_l_mass" type="sphere" pos="0.06 0.02 0" size="0.01" mass="0.009" material="motor_internal"/>
            <geom name="m_pitch_r_mass" type="sphere" pos="0.06 -0.02 0" size="0.01" mass="0.009" material="motor_internal"/>
            <geom name="m_tail_mass" type="sphere" pos="0.03 0 0" size="0.01" mass="0.009" material="motor_internal"/>

            <!-- Tubo de Cauda (4g) -->
            <geom name="tail_tube" type="cylinder" fromto="-0.02197 0 0 -0.37657 0 0" size="0.006" mass="0.004" material="carbon_fiber"/>

            <!-- ASA ESQUERDA -->
            <body name="asa_esq_flap" pos="0 0.01951 0">
                <joint name="q1_flap_esq" type="hinge" axis="1 0 0" range="-45 45" damping="0.1"/>
                <!-- Cilindro de placeholder para satisfazer mjMINVAL -->
                <geom type="cylinder" size="0.005 0.005" mass="0.0001" material="joint_visual"/>
                
                <body name="asa_esq_pitch" pos="0 0.00249 0">
                    <joint name="q2_pitch_esq" type="hinge" axis="0 1 0" range="-25 25" damping="0.05"/>
                    <geom type="cylinder" size="0.005 0.005" mass="0.0001" material="joint_visual" euler="90 0 0"/>
                    
                    <body name="asa_esq">
                        <geom name="wing_l" type="ellipsoid" fluidshape="ellipsoid" pos="0 0.23625 0" size="0.1139 0.2362 0.005" mass="0.020" material="wing_mylar" fluidcoef="0.2 0.12 1.5 3.14 1.0"/>
                    </body>
                </body>
            </body>

            <!-- ASA DIREITA -->
            <body name="asa_dir_flap" pos="0 -0.01951 0">
                <joint name="q3_flap_dir" type="hinge" axis="1 0 0" range="-45 45" damping="0.1"/>
                <geom type="cylinder" size="0.005 0.005" mass="0.0001" material="joint_visual"/>
                
                <body name="asa_dir_pitch" pos="0 -0.00249 0">
                    <joint name="q4_pitch_dir" type="hinge" axis="0 1 0" range="-25 25" damping="0.05"/>
                    <geom type="cylinder" size="0.005 0.005" mass="0.0001" material="joint_visual" euler="90 0 0"/>
                    
                    <body name="asa_dir">
                        <geom name="wing_r" type="ellipsoid" fluidshape="ellipsoid" pos="0 -0.23625 0" size="0.1139 0.2362 0.005" mass="0.020" material="wing_mylar" fluidcoef="0.2 0.12 1.5 3.14 1.0"/>
                    </body>
                </body>
            </body>

            <!-- CAUDA -->
            <body name="cauda_pitch" pos="-0.37657 0 0">
                <joint name="q5_pitch_tail" type="hinge" axis="0 1 0" range="-35 35" damping="0.1"/>
                <geom type="cylinder" size="0.005 0.005" mass="0.0001" material="joint_visual" euler="90 0 0"/>
                
                <body name="cauda">
                    <geom name="tail_geom" type="ellipsoid" fluidshape="ellipsoid" pos="-0.11333 0 0" size="0.1133 0.17 0.005" mass="0.009" material="wing_mylar" fluidcoef="0.2 0.12 1.5 3.14 1.0"/>
                </body>
            </body>
        </body>
    </worldbody>

    <actuator>
        <position name="motor_q1" joint="q1_flap_esq" kp="10" ctrllimited="true" ctrlrange="-45 45"/>
        <position name="motor_q2" joint="q2_pitch_esq" kp="5" ctrllimited="true" ctrlrange="-25 25"/>
        <position name="motor_q3" joint="q3_flap_dir" kp="10" ctrllimited="true" ctrlrange="-45 45"/>
        <position name="motor_q4" joint="q4_pitch_dir" kp="5" ctrllimited="true" ctrlrange="-25 25"/>
        <position name="motor_q5" joint="q5_pitch_tail" kp="5" ctrllimited="true" ctrlrange="-35 35"/>
    </actuator>
</mujoco>
"""

# CLASSE DO AMBIENT COM GYMNASIUM
class OrnitoEnv(gym.Env):
    def __init__(self):
        super(OrnitoEnv, self).__init__()
        self.model = mujoco.MjModel.from_xml_string(xml_model)
        self.data = mujoco.MjData(self.model)
        
        # Atuadores (5 motores)
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        
        # Observação: (13 sensores atuais * 25 frames histórico) + (30 pontos futuro * 3 coords) = 415
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(540,), dtype=np.float32)

        # Filtro e Frequências
        self.dt_ia = 0.02
        self.alpha = (2*np.pi*self.dt_ia*7.0)/(2*np.pi*self.dt_ia*7.0 + 1)
        self.last_action = np.zeros(5)
        self.history = deque(maxlen=25)

        self.render_mode = True
        
        # Se o render for True, abrimos a janela de visualização do MuJoCo
        if self.render_mode:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer = None


    def _get_obs(self):

# Sensores Atuais (13) + Ações Anteriores (5)
        eta = self.data.qpos[3:7]  # Quatérnio
        pqr = self.data.qvel[3:6]  # Vel Angular
        qj = self.data.qpos[7:12]  # Juntas
        vx = np.array([self.data.qvel[0]]) # Pitot
        current_obs = np.concatenate([eta, pqr, qj, vx, self.last_action])
        
        self.history.append(current_obs)
        
        # Trajetória Futura (Look-ahead: 30 pontos)
        lim_fut = 30
        vel_target = 5.0
        future_traj = []
        for i in range(1, lim_fut + 1):
            t_futuro = self.data.time + (i * self.dt_ia)
            # Posição alvo futura relativa ao robô (Eixo X linear)
            pos_f = np.array([vel_target * t_futuro, 0.0, 12.0]) - self.data.qpos[:3]
            future_traj.append(pos_f)
            
        return np.concatenate([np.array(self.history).flatten(), np.array(future_traj).flatten()])

    def step(self, action):
        # Filtro Passa-Baixa nos comandos
        filtered_action = self.alpha * action + (1 - self.alpha) * self.last_action
        self.last_action = filtered_action
        
        self.data.ctrl[:] = filtered_action

        vel_target = 5.0
        
        # Sub-stepping: 5 passos de física (0.004s * 5 = 0.02s)
        for _ in range(5):
            # Atualiza alvo no MuJoCo para visualização
            self.data.mocap_pos[0] = [vel_target * self.data.time, 0.0, 12.0]
            mujoco.mj_step(self.model, self.data)

            # Sincroniza o visualizador a cada passo de física se estiver ativo
            if self.render_mode and self.viewer.is_running():
                self.viewer.sync()
                # time.sleep(0.002)

        # Cálculos de Fim de Step
        obs = self._get_obs()
        reward = self._compute_reward(filtered_action)
        reward += 1 * self.data.qvel[0] ### FORÇAR IR PRA FRENTE
        reward += 2 * np.exp(-1 * (self.data.qpos[2] - 12)**2) ### FORÇAR ALTITUDE NIVELADA
        reward += 2 * np.exp(-1 * (self.data.qpos[1] - 0)**2) ### FORÇAR ESTABILIDADE LATERAL
        
        lim_area = 4.0

        dist = np.linalg.norm(self.data.qpos[:3] - self.data.mocap_pos[0])
        terminated = bool(dist > lim_area) # Fora da área

        if terminated:
            reward -= 20.0 # Punição fixa por morrer

        truncated = False
        
        return obs, reward, terminated, truncated, {}

    def _compute_reward(self, action):
        dist = np.linalg.norm(self.data.qpos[:3] - self.data.mocap_pos[0])
        r_pos = np.exp(-2.0 * (dist**2))
        r_omega = np.exp(-0.1 * np.sum(np.square(self.data.qvel[3:6])))
        r_att = self.data.qpos[3]**2 # Componente W do quatérnio
        r_en = -0.05 * np.sum(np.square(action))
        
        return (0.5 * r_pos) + (0.1 * r_omega) + (0.2 * r_att) + r_en

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.history.clear()

        lim_hist = 25
        num_sensors = 18 #13 sensores + 5 actions

        # Preencher com zeros se o histórico ainda não estiver cheio
        while len(self.history) < lim_hist:
            self.history.append(np.zeros(num_sensors))

        self.last_action = np.zeros(5)
        return self._get_obs(), {}
    
    def close(self):
        # Boa prática: fechar o visualizador ao encerrar o script
        if self.viewer is not None:
            self.viewer.close()

env = OrnitoEnv()
env = gym.wrappers.TimeLimit(env, max_episode_steps=5000)

# Carregar o modelo treinado
model_path = "C:/Users/pedro/Desktop/Ornitos/models/final_model_ornito_novo.zip"
model = PPO.load(model_path, env=env, learning_rate=1e-4)

# Visualização apenas, sem treino
obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    
    # Executa as ações no simulador
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Delay para visualização
    time.sleep(0.02)

    if terminated or truncated:
        obs, info = env.reset()