import torch
import numpy as np
import time
import os
import sys
import datetime 

# Obtém o caminho absoluto pasta niveís acima
diretorio_atual = os.path.dirname(os.path.abspath(__file__))
diretorio_pai = os.path.dirname(diretorio_atual)
sys.path.append(diretorio_pai)

from ambiente.gridNavigationEnv import GridNavigationWithMetricsEnv
from ambiente.dqn import DQN

# Dimensões da Rede Neural (Input e Output)
DIMENSAO_ESTADO = 4   # O agente vê 4 valores
DIMENSAO_ACAO = 4     # O agente pode fazer 4 movimentos: Cima, Baixo, Esquerda, Direita

# Seleção automática do dispositivo de processamento (GPU NVIDIA se disponível, senão CPU)
DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuração dos caminhos para carregar o modelo treinado
diretorio_modelos = os.path.join(diretorio_pai, "modelos")
os.makedirs(diretorio_modelos, exist_ok=True)
CAMINHO_MODELO = os.path.join(diretorio_modelos, "modelo_grid_10.pth")

# Controle da velocidade da visualização (tempo de pausa entre passos)
VELOCIDADE_ANIMACAO = 0.02

# FUNÇÕES UTILITÁRIAS
def limpar_tela():
    os.system('cls' if os.name == 'nt' else 'clear')

def salvar_metricas_episodio(n_episodio, status, grid_size, passos, min_passos, recompensa, eficiencia):
    """
    Salva as métricas do episódio atual em um arquivo de texto.
    """
    caminho_log = os.path.join(diretorio_atual, "metricas_validacao_inteligente.txt")
    data_hora = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    with open(caminho_log, "a", encoding="utf-8") as f:
        f.write(f"=== EPISÓDIO {n_episodio} ===\n")
        f.write(f"STATUS            : {status}\n")
        f.write(f"Grid Size         : {grid_size}x{grid_size}\n")
        f.write(f"Passos Realizados : {passos}\n")
        f.write(f"Passos Ideais     : {min_passos:.0f}\n")
        f.write(f"Eficiência        : {eficiencia:.2f}%\n")
        f.write(f"Recompensa Total  : {recompensa:.4f}\n")
        f.write("----------------------------------------\n\n")
    
    print(f"[LOG] Métricas do episódio {n_episodio} salvas.")

# ======= CARREGAMENTO DA REDE NEURAL =======
# Inicializa a arquitetura da rede neural (DQN)
rede_politica = DQN(DIMENSAO_ESTADO, DIMENSAO_ACAO).to(DISPOSITIVO)

try:
    # Tenta carregar os pesos treinados do arquivo
    rede_politica.load_state_dict(torch.load(CAMINHO_MODELO, map_location=DISPOSITIVO))
    print(f"Modelo carregado com sucesso de: {CAMINHO_MODELO}")
except FileNotFoundError:
    print(f"Erro Crítico: O arquivo '{CAMINHO_MODELO}' não foi encontrado.")
    sys.exit()

# Coloca a rede em modo de avaliação
rede_politica.eval()

# ======= INICIALIZAÇÃO DO AMBIENTE =======
# Cria o ambiente de navegação
ambiente = GridNavigationWithMetricsEnv(grid_size=10, seed=42)

input("\nPressione ENTER para começar a visualização...")

# ======= LOOP DE VISUALIZAÇÃO (EPISÓDIOS) =======
for episodio in range(10): # Executa 10 episódios de teste
    
    # Reseta o ambiente para o estado inicial e pega a primeira observação
    observacao, info_extra = ambiente.reset()
    
    # Tratamento para extrair o estado dependendo do formato retornado (dicionário ou array)
    if isinstance(observacao, dict): 
        estado_atual = observacao["agent_0"]
    else: 
        estado_atual = observacao

    concluido = False
    recompensa_total = 0
    
    # --- Loop principal do Episódio (Passo a Passo) ---
    while not concluido:
        
        # --- BLOCO DE RENDERIZAÇÃO ---
        limpar_tela()
        print(f"=== EPISÓDIO {episodio + 1} ===")
        print(f"Meta Ideal (Caminho Mínimo): {ambiente.passos_minimos_ideais:.0f} passos")
        print(f"Passos Realizados: {ambiente.passos_realizados}")
        print("-" * 20)
        
        # Exibe o grid colorido no terminal
        print(ambiente.render()) 
        
        print("-" * 20)
        print("Legenda: A = Agente (Verde) | G = Objetivo (Amarelo) | # = Parede | . = Livre")
        
        # Pausa para criar o efeito de animação
        time.sleep(VELOCIDADE_ANIMACAO) 

        # --- TOMADA DE DECISÃO DO MODELO  ---
        with torch.no_grad(): # Desativa cálculo de gradientes
            # Converte o estado (numpy) para Tensor PyTorch e envia para GPU/CPU
            estado_tensor = torch.tensor(estado_atual, dtype=torch.float32).unsqueeze(0).to(DISPOSITIVO)
            # A rede retorna valores Q para as 4 ações. Pegamos o índice do maior valor (argmax).
            acao_escolhida = rede_politica(estado_tensor).argmax().item()
            
        # --- EXECUÇÃO DA AÇÃO ---
        proxima_obs, recompensas, terminou, truncou, infos = ambiente.step({"agent_0": acao_escolhida})
        
        # Atualiza o estado atual para o próximo loop
        if isinstance(proxima_obs, dict): 
            estado_atual = proxima_obs["agent_0"]
        else: 
            estado_atual = proxima_obs
        
        # Acumula a recompensa (para estatística)
        recompensa_total += recompensas["agent_0"]
        
        # Verifica flags de encerramento (biblioteca PettingZoo/Gymnasium)
        foi_terminado = terminou["agent_0"]
        foi_truncado = truncou.get("agent_0", False)

        # --- FIM DO EPISÓDIO ---
        if foi_terminado or foi_truncado:

            limpar_tela()
            print(f"=== FIM DO EPISÓDIO {episodio + 1} ===")
            print(ambiente.render())
            
            # Determina a mensagem de resultado e status para log
            if foi_terminado:
                msg_resultado = "SUCESSO! CHEGOU AO OBJETIVO"
                status_log = "SUCESSO"
            else:
                msg_resultado = "FALHA (TEMPO ESGOTADO)"
                status_log = "FALHA"

            print(f"\n{msg_resultado}")
            
            # Usa os dados do dicionário 'infos' que vêm do step()
            passos_totais = infos['agent_0']['steps_taken']
            minimos_necessarios = infos['agent_0']['min_steps']
            eficiencia = 0.0

            print(f"Passos totais: {passos_totais}")
            
            # Se venceu, calcula a eficiência (Ideal / Real)
            if foi_terminado:
                eficiencia = (minimos_necessarios / passos_totais) * 100
                print(f"Eficiência do caminho: {eficiencia:.2f}%")
            
            print(f"Recompensa Total Acumulada: {recompensa_total:.2f}")

            # Salva no arquivo
            salvar_metricas_episodio(
                n_episodio=episodio + 1,
                status=status_log,
                grid_size=10,
                passos=passos_totais,
                min_passos=minimos_necessarios,
                recompensa=recompensa_total,
                eficiencia=eficiencia
            )
            
            # Pausa maior no final para leitura dos dados
            time.sleep(2) 
            
            # Sai do loop while e vai para o próximo episódio do for
            break