# Vitor Leandro Machado - 10409358
# Rodrigo Lucas Rosales - 10365071

# Arquivo contempla todo conjunto ideal para simularmos a execução do agente.
# Conseguimos parametrizar uma grade de tamanho 'X' onde podemos visualizar o percurso que o agente toma para atingir o alvo.
# Nesta análise que fizemos o agente se comporta de maneira totalmente aleatória, mas
# nos ajudou a entender como podemos ir para o próximo passo, que seria o treinamento deste agente.

import os
import datetime
from pettingzoo.test import parallel_api_test
from gridNavigationEnv import GridNavigationWithMetricsEnv

class GridNavigationTest:
    def __init__(self, grid_size):
        self.env = GridNavigationWithMetricsEnv(grid_size=grid_size)

    def run_single_episode(self):
        """Executa um único episódio com ações aleatórias"""
        obs, infos = self.env.reset()
        done = False
        print("\n=== Iniciando Episódio Único ===")

        # Variável para registrar o status final no arquivo
        status_final = "FALHA (TEMPO ESGOTADO)"

        while not done:
            action = {"agent_0": self.env.action_spaces["agent_0"].sample()}
            obs, rewards, terms, truncs, infos = self.env.step(action)

            print(self.env.render(), "\n")
            print(f"Steps: {infos['agent_0']['steps_taken']}, "
                  f"Colisões: {infos['agent_0']['collisions']}, "
                  f"Recompensa restante: {infos['agent_0']['total_reward']:.2f}")

            if terms.get("agent_0", False):
                print("🎯 Objetivo alcançado!")
                print(f"Taxa de sucesso: {infos['agent_0']['success_rate']:.2f}%")
                print(f"Recompensa final acumulada: {infos['agent_0']['total_reward']:.2f}")
                status_final = "SUCESSO"
                done = True
            
            # Necessário verificar truncamento (tempo acabou) para encerrar o loop e salvar o arquivo
            if truncs.get("agent_0", False):
                print("⌛ Limite de passos atingido.")
                done = True

        print("\nNúmero de passos para caminho mais curto: ", self.env.min_steps)

        # Chama a função para salvar as métricas
        self.salvar_metricas(infos['agent_0'], status_final)

    def salvar_metricas(self, info, status):
        """Salva as métricas da execução em um arquivo .txt"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "metricas_execucao.txt")
        timestamp = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("--------------------------------------------------\n")
            f.write(f"STATUS FINAL      : {status}\n")
            f.write(f"Grid Size         : {self.env.grid_size}x{self.env.grid_size}\n")
            f.write("--------------------------------------------------\n")
            f.write(f"Passos Realizados : {info['steps_taken']}\n")
            f.write(f"Passos Ideais     : {self.env.min_steps:.0f}\n")
            f.write(f"Colisões          : {info['collisions']}\n")
            f.write(f"Recompensa Total  : {info['total_reward']:.2f}\n")
            f.write(f"Taxa de Sucesso   : {info['success_rate']:.2f}%\n")
            f.write("==================================================\n")
        
        print(f"\n📄 Métricas salvas em: {file_path}")

if __name__ == "__main__":
    tester = GridNavigationTest(grid_size=10)
    tester.run_single_episode()