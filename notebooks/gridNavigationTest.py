# Vitor Leandro Machado - 10409358
# Rodrigo Lucas Rosales - 10365071

# Arquivo contempla todo conjunto ideal para simularmos a execução do agente.
# Conseguimos parametrizar uma grade de tamanho 'X' onde podemos visualizar o percurso que o agente toma para atingir o alvo.
# Nesta análise que fizemos o agente se comporta de maneira totalmente aleatória, mas
# nos ajudou a entender como podemos ir para o próximo passo, que seria o treinamento deste agente.

from pettingzoo.test import parallel_api_test
from gridNavigationEnv import GridNavigationWithMetricsEnv


class GridNavigationTest:
    def __init__(self, grid_size):
        self.env = GridNavigationWithMetricsEnv(grid_size=grid_size)

    def run_api_test(self):
        """Valida compatibilidade com PettingZoo"""
        print("✅ Rodando API Test...")
        parallel_api_test(self.env, num_cycles=500)
        print("✅ API compatível!\n")

    def run_single_episode(self):
        """Executa um único episódio com ações aleatórias"""
        obs, infos = self.env.reset()
        done = False
        print("\n=== Iniciando Episódio Único ===")

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
                done = True


if __name__ == "__main__":
    tester = GridNavigationTest(grid_size=3)
    tester.run_api_test()
    tester.run_single_episode()
