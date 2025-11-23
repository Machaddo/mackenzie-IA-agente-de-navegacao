from collections import deque
import random
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.from_numpy(np.array(states)).float()
        actions     = torch.tensor(actions, dtype=torch.int64)
        rewards     = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.from_numpy(np.array(next_states)).float()
        dones       = torch.tensor(dones, dtype=torch.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
