import mujoco
import mujoco.viewer
import numpy as np
import time
from pynput import keyboard

# --- CONFIGURAÇÕES DO BATIMENTO (FLAPPING) ---
FREQUENCIA = 4.0    # Hz (Batimentos por segundo)
AMPLITUDE = 0.8     # Fração do range total da junta (0.0 a 1.0)
TAXA_CAUDA = 0.5     # Velocidade de resposta da cauda

xml_path = "Modelos XML/Current_Model.xml"
keys_pressed = set()
# ctrl_cauda = 0.2341 #Trimado 3Hz
ctrl_cauda = 0.1296 #Trimado 4Hz
# ctrl_cauda = 0.0882 #Trimado 5Hz

def on_press(key):
    try:
        if hasattr(key, 'char'): keys_pressed.add(key.char.lower())
    except AttributeError: pass

def on_release(key):
    try:
        if hasattr(key, 'char'): keys_pressed.discard(key.char.lower())
    except AttributeError: pass

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

def test_flapping_flight(v_initial):
    global ctrl_cauda
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 25.0
    data.qvel[0] = v_initial
    
    # IDs dos Atuadores
    f_esq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'motor_q1')
    f_dir = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'motor_q3')
    cauda = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'motor_q5')

    print(f"\n--- DECOLAGEM: V0 = {v_initial} m/s | Freq = {FREQUENCIA}Hz ---")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:

        viewer.cam.lookat = data.body('torso').xpos
        viewer.cam.distance = 4.0
        time.sleep(3)

        while viewer.is_running() and data.qpos[2] > 0.3:
            step_start = time.time()
            
            # 1. CÁLCULO DA SENOIDE (Flapping)
            # q(t) = A * sin(2 * pi * f * t)
            sinal_asa = AMPLITUDE * np.sin(2 * np.pi * FREQUENCIA * data.time)
            
            data.ctrl[f_esq] = -sinal_asa
            data.ctrl[f_dir] = sinal_asa # Asas batem em fase
            
            # 2. CONTROLE MANUAL DA CAUDA (W/S)
            if 's' in keys_pressed: ctrl_cauda += TAXA_CAUDA * model.opt.timestep
            if 'x' in keys_pressed: ctrl_cauda -= TAXA_CAUDA * model.opt.timestep
            ctrl_cauda = np.clip(ctrl_cauda, -1.0, 1.0)
            data.ctrl[cauda] = ctrl_cauda
            
            mujoco.mj_step(model, data)
            
            # 3. PITCH CORRIGIDO (Positivo = Nariz para Cima)
            mat = data.body('torso').xmat.reshape(3, 3)
            pitch = np.degrees(np.arcsin(mat[2, 0])) # Inverti o sinal aqui
            
            if int(data.time * 100) % 50 == 0:
                print(f"T: {data.time:.1f}s | Alt: {data.qpos[2]:.2f}m | Dist: {data.qpos[0]:.2f}m | VelX: {data.qvel[0]:.1f}m/s | Pitch: {pitch:.1f}° | Cntrl Cauda: {ctrl_cauda:.3f}")

            viewer.cam.lookat = data.body('torso').xpos
            viewer.cam.distance = 4.0
            viewer.sync()
            
            time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))

test_flapping_flight(5.0)