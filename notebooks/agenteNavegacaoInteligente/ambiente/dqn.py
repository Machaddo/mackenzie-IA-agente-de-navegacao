# Vitor Leandro Machado - 10409358
# Rodrigo Lucas Rosales - 10365071

# USO Rede Neural DQN

import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim=4, action_dim=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)
