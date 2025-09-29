# Vitor Leandro Machado - 10409358
# Rodrigo Lucas Rosales - 10365071

# Arquivo contempla o setup do ambiente de navageção em grade para o agente implementado utilizando PettingZoo
# O ambiente pode ser expandido para agentes paralelos, porém neste caso em específico há apenas um agente.
# Trabalhamos com um grid 10x10 com a ideia de entendermos o comportamento do agente
# Incluímos também um calculo do caminho mínimo utilizando BFS. Onde prevemos o menor caminho possível dado um cenário 
# variado de obstaculos. Na função '_compute_min_steps'

# Métricas que colhemos: 
# steps_taken:	Número de passos no episódio
# collisions:	Quantas vezes o agente bateu em obstáculos
# total_reward:	Soma das recompensas recebidas
# min_steps:	Passos mínimos possíveis para atingir o objetivo
# success_rate:	Percentual do caminho ótimo atingido

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from collections import deque

class GridNavigationWithMetricsEnv(ParallelEnv):
    metadata = {"render_modes": ["ansi"], "name": "grid_navigation_metrics_v0"}

    def __init__(self, grid_size, render_mode="ansi"):
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.pos = None
        self.goal = None
        self.agents = []

        # Espaços
        self.action_spaces = {"agent_0": spaces.Discrete(4)}
        obs_low = np.array([0, 0, 0, 0])
        obs_high = np.array([grid_size - 1, grid_size - 1, grid_size - 1, grid_size - 1])
        self.observation_spaces = {"agent_0": spaces.Box(obs_low, obs_high, dtype=np.int32)}

        # Obstáculos fixos
        self.obstacles = np.zeros((grid_size, grid_size), dtype=int)
        self._place_obstacles()

    # Define obstáculos no grid, marcados com 1.
    def _place_obstacles(self):
        self.obstacles[2, 1:8] = 1
        self.obstacles[5, 3:10] = 1
        self.obstacles[7:10, 4] = 1
        return
    
    # Faz uma busca em largura (BFS) para calcular o menor número de passos do agente até o objetivo considerando obstáculos.
    def _compute_min_steps(self, start, goal):
        """Calcula a distância mínima considerando obstáculos usando BFS"""
        visited = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        queue = deque([(tuple(start), 0)])  # posição, passos

        directions = [(0,1), (0,-1), (1,0), (-1,0)]

        while queue:
            (x, y), steps = queue.popleft()
            if (x, y) == tuple(goal):
                return steps
            if visited[x, y]:
                continue
            visited[x, y] = True

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if not visited[nx, ny] and self.obstacles[nx, ny] == 0:
                        queue.append(((nx, ny), steps + 1))
        # Se não há caminho possível
        return np.inf


    # Posiciona o agente no canto superior esquerdo (0,0) e o objetivo no canto inferior direito.
    # Zera métricas do episódio:
        # steps_taken
        # collisions
        # total_reward
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = ["agent_0"]
        self.pos = np.array([0, 0])
        self.goal = np.array([self.grid_size - 1, self.grid_size - 1])

        # Métricas por episódio
        self.steps_taken = 0
        self.collisions = 0
        self.total_reward = 0.0  # começa em zero

        # Calcula distância mínima real considerando obstáculos
        self.min_steps = self._compute_min_steps(self.pos, self.goal)
        self.max_possible_reward = self.min_steps * 0.5 if np.isfinite(self.min_steps) else 0.0

        obs = {"agent_0": np.concatenate([self.pos, self.goal])}
        infos = {"agent_0": {}}
        return obs, infos


    # Recebe ação do agente (0-3) e tenta mover
    # Verifica colisões com obstáculos:
        # Penalidade de -0.2 se colidir
        # Passos válidos dão recompensa de 0.5.
        # Termina o episódio se o agente atingir o objetivo.
        # Atualiza métricas:
        # steps_taken
        # collisions
        # total_reward
        # success_rate
    def step(self, actions):
        move_dict = {
            0: np.array([0, 1]),   # direita
            1: np.array([0, -1]),  # esquerda
            2: np.array([1, 0]),   # baixo
            3: np.array([-1, 0]),  # cima
        }

        self.steps_taken += 1
        intended_pos = self.pos + move_dict[actions["agent_0"]]
        intended_pos = np.clip(intended_pos, 0, self.grid_size - 1)

        # Checa colisão
        if self.obstacles[intended_pos[0], intended_pos[1]] == 1:
            self.collisions += 1
            step_reward -= 0.2
        else:
            self.pos = intended_pos

        terminated = np.array_equal(self.pos, self.goal)

        # A cada passo válido, soma 0.5 de recompensa
        if not np.array_equal(self.pos, intended_pos):
            step_reward = 0.0
        else:
            step_reward = 0.5
            self.total_reward += step_reward

        rewards = {"agent_0": step_reward}
        terminations = {"agent_0": terminated}
        truncations = {"agent_0": False}

        infos = {"agent_0": {
            "steps_taken": self.steps_taken,
            "collisions": self.collisions,
            "total_reward": self.total_reward
        }}

        if terminated:
            # Taxa de sucesso: percentual do caminho ótimo preservado
            if np.isfinite(self.min_steps) and self.min_steps > 0:
                # Se o agente atingiu o mínimo de steps, sucesso é 100%
                success_rate = min(100.0, (self.min_steps / self.steps_taken) * 100.0)
            else:
                success_rate = 0.0
            self.agents = []
            infos["agent_0"]["success_rate"] = success_rate

        return (
            {"agent_0": np.concatenate([self.pos, self.goal])},
            rewards,
            terminations,
            truncations,
            infos,
        )


    # Gera uma representação textual do grid:
    def render(self):
        grid = np.full((self.grid_size, self.grid_size), ".")
        grid[self.obstacles == 1] = "#"
        gx, gy = self.goal
        ax, ay = self.pos
        grid[gx, gy] = "G"
        grid[ax, ay] = "A"
        return "\n".join(" ".join(row) for row in grid)
