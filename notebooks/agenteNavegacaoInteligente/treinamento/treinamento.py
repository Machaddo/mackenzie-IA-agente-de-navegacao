import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from ambiente.dqn import DQN
from ambiente.replayBuffer import ReplayBuffer
from ambiente.gridNavigationEnv import GridNavigationWithMetricsEnv

# ------------- Hiperparâmetros -------------
GRID_SIZE = 20              
SEED = 42                   
EPISODES = 10000              

BATCH_SIZE = 64             
GAMMA = 0.99                
LR = 1e-3                   
BUFFER_SIZE = 20000         
TARGET_UPDATE = 5           

EPSILON_START = 1.0        
EPSILON_MIN = 0.05          
EPSILON_DECAY = 0.9997  

MAX_TRAIN_STEPS = None      

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelos_dir = os.path.join(parent_dir, "modelos")
os.makedirs(modelos_dir, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(modelos_dir, f"modelo_grid_{GRID_SIZE}.pth")

# Criando o ambiente
env = GridNavigationWithMetricsEnv(
    grid_size=GRID_SIZE,
    seed=SEED
)

state_dim = 4
action_dim = 4

# Inicializando reprodutibilidade
np.random.seed(SEED)
torch.manual_seed(SEED)

# Criando redes DQN
policy_net = DQN(state_dim, action_dim).to(DEVICE)
target_net = DQN(state_dim, action_dim).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()   

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
criterion = nn.MSELoss()  

buffer = ReplayBuffer(BUFFER_SIZE)

epsilon = EPSILON_START
success_count = 0  

# Arquivo de LOG para Excel
modelos_dir = os.path.join(parent_dir, "treinamento")
os.makedirs(modelos_dir, exist_ok=True)
LOG_PATH = os.path.join(modelos_dir, f"treino_results__grid_{GRID_SIZE}.txt")

with open(LOG_PATH, "w") as f:
    f.write("episodio,recompensa_total,epsilon,chegou_ao_objetivo,eficiencia_episodio,passos_totais\n")

# Loop principal de treinamento
for ep in range(EPISODES):

    obs, info = env.reset()
    state = obs["agent_0"]
    done = False
    total_reward = 0.0
    episode_success = False

    while not done:

        if np.random.rand() < epsilon:
            action = np.random.randint(action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                action = policy_net(state_tensor).argmax().item()

        obs_next, rewards, terminations, truncations, infos = env.step({"agent_0": action})
        next_state = obs_next["agent_0"]
        reward = rewards["agent_0"]
        terminated = terminations["agent_0"]
        truncated = truncations.get("agent_0", False)

        buffer.push(state, action, reward, next_state, float(terminated or truncated))

        state = next_state
        total_reward += reward

        if len(buffer) >= BATCH_SIZE:

            states_b, actions_b, rewards_b, next_states_b, dones_b = buffer.sample(BATCH_SIZE)

            states_b = states_b.to(DEVICE)
            actions_b = actions_b.to(DEVICE)
            rewards_b = rewards_b.to(DEVICE)
            next_states_b = next_states_b.to(DEVICE)
            dones_b = dones_b.to(DEVICE)

            q_values = policy_net(states_b).gather(1, actions_b.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                q_next = target_net(next_states_b).max(1)[0]
                q_target = rewards_b + GAMMA * q_next * (1.0 - dones_b)

            loss = criterion(q_values, q_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if terminated or truncated:
            episode_success = infos["agent_0"].get("success", False)
            break

    steps_taken = infos["agent_0"]["steps_taken"]
    min_steps = infos["agent_0"]["min_steps"]
    reached_goal = infos["agent_0"].get("success", False)

    if reached_goal and steps_taken > 0:
        # Calcula a porcentagem de quão perto do ideal ele foi
        # Ex: Ideal 10, Real 12 -> (10/12)*100 = 83.3% de eficiência
        episode_efficiency = (min_steps / steps_taken) * 100.0
    else:
        # Se não chegou ou (caso raro) deu 0 passos, a eficiência é 0
        episode_efficiency = 0.0

    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    if ep % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Salvamento de métricas no arquivo .txt
    with open(LOG_PATH, "a") as f:
        f.write(f"{ep},{total_reward:.4f},{epsilon:.4f},{int(reached_goal)},{episode_efficiency:.2f},{steps_taken}\n")

    if (ep + 1) % 200 == 0:
        torch.save(policy_net.state_dict(), MODEL_SAVE_PATH)

torch.save(policy_net.state_dict(), MODEL_SAVE_PATH)
print("Treinamento finalizado. Modelo salvo em:", MODEL_SAVE_PATH)