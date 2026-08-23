from xml.parsers.expat import model

import mujoco
import mujoco.viewer
import numpy as np
import time
from pynput import keyboard

# Configurações de controle
xml_path = "Current_Model.xml"

TAXA_MUDANCA = 1.0  # Velocidade da mudança da cauda (unidades por segundo)

ctrl_cauda = 1.0 #Trimagem, maior distância = 55.25m

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
listener = keyboard.Listener(on_press = on_press, on_release = on_release)
listener.start()

def test_gliding(v_initial):
    global ctrl_cauda
    model = mujoco.MjModel.from_xml_path(xml_path)

    data = mujoco.MjData(model)
    
    # Resetar o estado e aplicar velocidade inicial
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 12.0
    data.qvel[0] = v_initial
    
    try:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'motor_q5')
    except:
        actuator_id = 4 

    print(f"\n--- Teste Iniciado: V0 = {v_initial} m/s | SEGURE S/X para controlar ---")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:

        viewer.cam.lookat = data.body('torso').xpos
        viewer.cam.distance = 4.0
        time.sleep(3)

        while viewer.is_running() and data.qpos[2] > 0.05:
            step_start = time.time()
            
            # Lógica de "Segurar": Incrementa baseado no timestep da simulação
            if 's' in keys_pressed:
                ctrl_cauda += TAXA_MUDANCA * model.opt.timestep
            if 'x' in keys_pressed:
                ctrl_cauda -= TAXA_MUDANCA * model.opt.timestep
            
            # Limitar o comando (opcional, para evitar extrapolar os limites do servo)
            ctrl_cauda = np.clip(ctrl_cauda, -1.0, 1.0)
            
            # Aplicar o comando
            data.ctrl[actuator_id] = ctrl_cauda
            
            mujoco.mj_step(model, data)
            
            # Print de feedback no terminal
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

test_gliding(5.0)