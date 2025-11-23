# Ambiente de Navegação em Grid 2D (Compatível com PettingZoo/Gymnasium).

# Características:
# - Grid NxN com obstáculos estáticos.
# - Objetivo Fixo (escolhido no início) e Posição Inicial Aleatória.
# - Observação: Coordenadas normalizadas (0.0 a 1.0).
# - Recompensa: Penalidade por passo/colisão e prêmio por objetivo.

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from collections import deque

class GridNavigationWithMetricsEnv(ParallelEnv):
    # Metadados para configuração de renderização
    metadata = {"render_modes": ["ansi"], "name": "grid_navigation_metrics_v0"}

    def __init__(self, grid_size=20, render_mode="ansi", obstacle_density=0.15, seed=None):
        self.tamanho_grid = grid_size
        self.render_mode = render_mode
        
        # Variáveis de estado
        self.posicao_agente = None   # Onde o agente está agora [x, y]
        self.posicao_objetivo = None # Onde ele precisa chegar [x, y]
        self.agents = []             # Lista de agentes (necessário para PettingZoo)

        # --- DEFINIÇÃO DO ESPAÇO DE AÇÃO ---
        # 0: Direita | 1: Esquerda | 2: Baixo | 3: Cima
        self.action_spaces = {"agent_0": spaces.Discrete(4)}

        # --- DEFINIÇÃO DO ESPAÇO DE OBSERVAÇÃO ---
        # O agente recebe 4 valores normalizados entre 0 e 1:
        obs_min = np.array([0., 0., 0., 0.], dtype=np.float32)
        obs_max = np.array([1., 1., 1., 1.], dtype=np.float32)
        self.observation_spaces = {"agent_0": spaces.Box(obs_min, obs_max, dtype=np.float32)}

        # --- GERAÇÃO DO MAPA ---
        self.densidade_obstaculos = obstacle_density
        self.obstaculos = np.zeros((grid_size, grid_size), dtype=int)

        # Gerador de números aleatórios (RNG)
        self._rng = np.random.RandomState(seed)

        # 1. Preenche o grid com paredes aleatórias
        self._gerar_obstaculos()

        # 2. Define o OBJETIVO FIXO para o ambiente
        while True:
            gx = self._rng.randint(0, self.tamanho_grid)
            gy = self._rng.randint(0, self.tamanho_grid)
            
            if self.obstaculos[gx, gy] == 0:
                self.objetivo_fixo = np.array([gx, gy])
                break

    def _gerar_obstaculos(self):
        # Cria uma matriz de True/False onde True = Parede
        mascara_obstaculos = self._rng.rand(self.tamanho_grid, self.tamanho_grid) < self.densidade_obstaculos
        self.obstaculos = mascara_obstaculos.astype(int)

        # Se tudo for parede, abre um buraco aleatório
        if np.all(self.obstaculos == 1):
            rx = self._rng.randint(self.tamanho_grid)
            ry = self._rng.randint(self.tamanho_grid)
            self.obstaculos[rx, ry] = 0

    def _calcular_passos_minimos_bfs(self, inicio, fim):
        visitados = np.zeros((self.tamanho_grid, self.tamanho_grid), dtype=bool)
        
        # Fila para o algoritmo: guarda ((x, y), distancia_acumulada)
        fila = deque([(tuple(inicio), 0)]) 
        
        # Vetores de movimento (Direita, Esquerda, Baixo, Cima)
        direcoes = [(0,1), (0,-1), (1,0), (-1,0)]

        while fila:
            (x, y), passos = fila.popleft()

            # Se chegou no destino, retorna o número de passos
            if (x, y) == tuple(fim):
                return passos

            if visitados[x, y]:
                continue
            visitados[x, y] = True

            # Verifica os 4 vizinhos
            for dx, dy in direcoes:
                nx, ny = x + dx, y + dy

                # Verifica se está dentro do grid
                if 0 <= nx < self.tamanho_grid and 0 <= ny < self.tamanho_grid:
                    # Verifica se não é parede e não foi visitado
                    if not visitados[nx, ny] and self.obstaculos[nx, ny] == 0:
                        fila.append(((nx, ny), passos + 1))

        # Se a fila acabar e não achar o destino, não existe caminho
        return np.inf

    def _gerar_observacao_normalizada(self):
        max_indice = max(1, self.tamanho_grid - 1)
        return np.array([
            self.posicao_agente[0] / max_indice,
            self.posicao_agente[1] / max_indice,
            self.posicao_objetivo[0] / max_indice,
            self.posicao_objetivo[1] / max_indice
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng.seed(seed)

        self.agents = ["agent_0"]
        
        # Recupera o objetivo fixo definido no __init__
        self.posicao_objetivo = self.objetivo_fixo 

        # Loop de segurança: Tenta achar uma posição inicial válida
        while True:
            # Sorteia candidato a posição inicial
            candidato_pos = np.array([
                self._rng.randint(0, self.tamanho_grid),
                self._rng.randint(0, self.tamanho_grid)
            ])
            
            # Não pode nascer dentro de uma parede
            if self.obstaculos[candidato_pos[0], candidato_pos[1]] == 1:
                continue 
            
            # Não pode nascer em cima do objetivo 
            if np.array_equal(candidato_pos, self.posicao_objetivo):
                continue

            # Verifica se existe caminho (BFS)
            passos_necessarios = self._calcular_passos_minimos_bfs(candidato_pos, self.posicao_objetivo)
            
            if np.isfinite(passos_necessarios):
                self.posicao_agente = candidato_pos
                self.passos_minimos_ideais = passos_necessarios
                break # Posição válida encontrada!

        # --- Inicialização de Métricas ---
        self.passos_realizados = 0
        self.colisoes = 0
        self.recompensa_acumulada = 0.0
        
        # Define limite máximo de passos antes de cortar o episódio (4x o tamanho ou N^2)
        self.maximo_passos = max(4 * self.tamanho_grid, self.tamanho_grid * self.tamanho_grid)

        # Gera a primeira observação
        obs = {"agent_0": self._gerar_observacao_normalizada()}
        
        infos = {"agent_0": {
            "min_steps": float(self.passos_minimos_ideais),
            "steps_taken": 0,
            "success": False
        }}
        
        return obs, infos

    def step(self, actions):
        dicionario_movimentos = {
            0: np.array([0, 1]),   # Direita
            1: np.array([0, -1]),  # Esquerda
            2: np.array([1, 0]),   # Baixo
            3: np.array([-1, 0]),  # Cima
        }

        # --- CONFIGURAÇÃO DE RECOMPENSAS ---
        PENALIDADE_PASSO = -0.01    # Custo de energia por andar
        PENALIDADE_COLISAO = -0.5   # Punição por bater na parede
        RECOMPENSA_VITORIA = 10.0   # Prêmio por chegar ao objetivo

        self.passos_realizados += 1

        # Calcula onde o agente QUER ir
        movimento = dicionario_movimentos[int(actions["agent_0"])]
        posicao_pretendida = self.posicao_agente + movimento
        
        # Garante que ele não saia do mapa (clamping entre 0 e N-1)
        posicao_pretendida = np.clip(posicao_pretendida, 0, self.tamanho_grid - 1)

        # Começa com a penalidade básica do passo
        recompensa_passo = PENALIDADE_PASSO

        # --- VERIFICAÇÃO DE COLISÃO ---
        if self.obstaculos[posicao_pretendida[0], posicao_pretendida[1]] == 1:
            # Bateu na parede!
            self.colisoes += 1
            recompensa_passo += PENALIDADE_COLISAO
            # O agente não se move (bate e volta/fica parado)
        else:
            # Caminho livre, atualiza a posição
            self.posicao_agente = posicao_pretendida

        # --- VERIFICAÇÃO DE VITÓRIA ---
        terminou = np.array_equal(self.posicao_agente, self.posicao_objetivo)

        if terminou:
            recompensa_passo += RECOMPENSA_VITORIA

        self.recompensa_acumulada += recompensa_passo

        # --- VERIFICAÇÃO DE TEMPO ESGOTADO ---
        truncado = self.passos_realizados >= self.maximo_passos

        # --- CÁLCULO DE MÉTRICAS FINAIS ---
        sucesso = False
        taxa_eficiencia = 0.0

        if terminou:
            sucesso = True
            if np.isfinite(self.passos_minimos_ideais) and self.passos_minimos_ideais > 0:
                # Calcula % de eficiência (Ideal / Real)
                taxa_eficiencia = 100.0 * (self.passos_minimos_ideais / self.passos_realizados)

        # Montagem do dicionário de informações extras
        infos = {"agent_0": {
            "steps_taken": self.passos_realizados,
            "collisions": self.colisoes,
            "total_reward": self.recompensa_acumulada,
            "success": sucesso,
            "success_rate": taxa_eficiencia,
            "min_steps": float(self.passos_minimos_ideais)
        }}

        # Retorno padrão exigido pelo Gym/PettingZoo
        return (
            {"agent_0": self._gerar_observacao_normalizada()}, # Observação
            {"agent_0": float(recompensa_passo)},              # Recompensa
            {"agent_0": terminou},                             # Terminated (Ganhou?)
            {"agent_0": truncado},                             # Truncated (Acabou tempo?)
            infos,                                             # Infos extras
        )

    def render(self):
        # Códigos de Cores para deixar o terminal mais evidente das ações
        COR_VERMELHO = "\033[91m"
        COR_VERDE = "\033[92m"
        COR_AMARELO = "\033[93m"
        COR_CINZA = "\033[90m"
        COR_RESET = "\033[0m"

        linhas_texto = []
        
        for r in range(self.tamanho_grid):
            linha_atual = []
            for c in range(self.tamanho_grid):
                coord_atual = [r, c]
                
                # 1. Desenha o Agente
                if np.array_equal(coord_atual, self.posicao_agente):
                    # Se o agente estiver em cima do objetivo
                    if np.array_equal(coord_atual, self.posicao_objetivo):
                        linha_atual.append(f"{COR_VERDE}A{COR_RESET}") 
                    else:
                        linha_atual.append(f"{COR_VERDE}A{COR_RESET}")
                
                # 2. Desenha o Objetivo
                elif np.array_equal(coord_atual, self.posicao_objetivo):
                    linha_atual.append(f"{COR_AMARELO}G{COR_RESET}")
                
                # 3. Desenha Obstáculo
                elif self.obstaculos[r, c] == 1:
                    linha_atual.append(f"{COR_VERMELHO}#{COR_RESET}")
                
                # 4. Desenha Espaço Livre
                else:
                    linha_atual.append(f"{COR_CINZA}.{COR_RESET}")
            
            # Junta os caracteres da linha atual
            linhas_texto.append(" ".join(linha_atual))

        return "\n".join(linhas_texto)