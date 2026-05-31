from xml.parsers.expat import model

import mujoco
import mujoco.viewer
import numpy as np
import time
from pynput import keyboard

# Configurações de controle
xml_path = "Current_Model copy.xml"
TAXA_MUDANCA = 1.0  # Velocidade da cauda (unidades por segundo)
ctrl_cauda = 0.475
# Gerenciamento de estado do teclado
keys_pressed = set()

def on_press(key):
    try:
        if hasattr(key, 'char'):
            keys_pressed.add(key.char.lower())
    except AttributeError:
        pass

def on_release(key):
    try:
        if hasattr(key, 'char'):
            keys_pressed.discard(key.char.lower())
    except AttributeError:
        pass

# Inicia o Listener em background
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

def test_gliding(v_initial):
    global ctrl_cauda
    model = mujoco.MjModel.from_xml_path(xml_path)

    data = mujoco.MjData(model)

    # IDs das geoms (certifique-se que os nomes batem com o seu XML)
    asas = ['wing_l', 'wing_r', 'tail'] # nomes das geoms

    # Coeficientes: [Sustentação, Arrasto, Momento, Escala_Angular, Escala_Força, ...]
    # MuJoCo usa 12 slots internos. Vamos preencher os 5 primeiros.
    novos_coefs = [1.5, 0.01, 1.0, 3.14, 1.0, 0, 0, 0, 0, 0, 0, 0]

    for nome in asas:
        try:
            g_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, nome)
            model.geom_fluid[g_id] = novos_coefs
            print(f"Sucesso: Coeficientes de {nome} atualizados manualmente.")
        except:
            print(f"Erro: Geom '{nome}' não encontrada no modelo.")

    # Verifique o coeficiente de arrasto da primeira asa (geom id ou nome)
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'wing_l')
    print(f"Coeficientes do fluido da asa: {model.geom_fluid[geom_id]}")
    
    # Resetar o estado e aplicar velocidade inicial
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 12.0
    data.qvel[0] = v_initial
    ctrl_cauda = 0.475
    
    try:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'motor_q5')
    except:
        actuator_id = 4 

    print(f"\n--- Teste Iniciado: V0 = {v_initial} m/s | SEGURE W/S para controlar ---")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:

        viewer.cam.lookat = data.body('torso').xpos
        viewer.cam.distance = 4.0
        time.sleep(60)

        while viewer.is_running() and data.qpos[2] > 0.05:
            step_start = time.time()
            
            # Lógica de "Segurar": Incrementa baseado no timestep da simulação
            if 'w' in keys_pressed:
                ctrl_cauda += TAXA_MUDANCA * model.opt.timestep
            if 's' in keys_pressed:
                ctrl_cauda -= TAXA_MUDANCA * model.opt.timestep
            
            # Limitar o comando (opcional, para evitar extrapolar os limites do servo)
            ctrl_cauda = np.clip(ctrl_cauda, -1.0, 1.0)
            
            # Aplicar o comando
            data.ctrl[actuator_id] = ctrl_cauda
            
            mujoco.mj_step(model, data)
            
            # Feedback no console (Reduzi a frequência para não poluir)www
            if int(data.time * 10) % 10 == 0:
                mat = data.body('torso').xmat.reshape(3, 3)
                pitch_rad = np.arcsin(mat[2, 0])
                pitch = np.degrees(pitch_rad)
                print(f"T: {data.time:.1f}s | Alt: {data.qpos[2]:.2f}m | Dist: {data.qpos[0]:.2f}m | VelX: {data.qvel[0]:.1f}m/s | Pitch: {pitch:.1f}° | Cntrl Cauda: {ctrl_cauda:.3f}")
            
            # Câmera de Perseguição
            viewer.cam.lookat = data.body('torso').xpos
            viewer.cam.distance = 5.0
            
            viewer.sync()
            
            # Controle de tempo real
            SLOW_DOWN = 0.8
            time_until_next_step = (model.opt.timestep/SLOW_DOWN) - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

# Bateria de testes
for v in [20]:
    test_gliding(v)