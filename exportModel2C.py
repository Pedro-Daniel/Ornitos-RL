import torch
import numpy as np
from stable_baselines3 import PPO

def exportar_pesos_para_c(caminho_modelo, arquivo_saida="policy_pesos.h"):
    print(f"Carregando modelo: {caminho_modelo}")
    model = PPO.load(caminho_modelo)
    
    # Extrai o dicionário de estados (pesos e vieses) do modelo PyTorch
    state_dict = model.policy.state_dict()
    
    # Filtra APENAS a rede do Actor (que gera as ações determinísticas)
    # A action_net é a camada de saída final.
    actor_keys = [k for k in state_dict.keys() if 'mlp_extractor.policy_net' in k or 'action_net' in k]
    
    with open(arquivo_saida, 'w') as f:
        f.write("#ifndef POLICY_PESOS_H\n#define POLICY_PESOS_H\n\n")
        f.write("// --- Pesos do Controlador do Ornitoptero (PPO) ---\n\n")
        
        for key in actor_keys:
            # Converte o tensor do PyTorch para um array NumPy
            tensor = state_dict[key].cpu().numpy()
            
            # Formata o nome da variável para ser aceito em C++
            var_name = key.replace('.', '_')
            
            if len(tensor.shape) == 2:  # É uma matriz de Pesos (Weights)
                out_dim, in_dim = tensor.shape
                f.write(f"const float {var_name}[{out_dim}][{in_dim}] = {{\n")
                
                # Itera sobre as linhas da matriz
                for row in tensor:
                    row_str = ", ".join([f"{val:.6f}f" for val in row])
                    f.write(f"    {{{row_str}}},\n")
                f.write("};\n\n")
                
            elif len(tensor.shape) == 1:  # É um vetor de Vieses (Biases)
                dim = tensor.shape[0]
                f.write(f"const float {var_name}[{dim}] = {{\n    ")
                
                val_str = ", ".join([f"{val:.6f}f" for val in tensor])
                f.write(f"{val_str}\n}};\n\n")
                
        f.write("#endif // POLICY_PESOS_H\n")
        
    print(f"Sucesso! Pesos formatados e salvos em: {arquivo_saida}")

# Substitua pelo caminho do seu modelo de gabarito
caminho_do_zip = "C:/Users/pedro/Desktop/Ornitos/models/PPO_Voo_Reto_Multidirecional/PPO_Voo_Reto_Multidirecional_1399944_steps.zip"
exportar_pesos_para_c(caminho_do_zip)