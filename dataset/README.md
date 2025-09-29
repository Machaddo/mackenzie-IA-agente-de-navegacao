## 📊 Dataset

Este projeto **não utiliza um dataset externo tradicional** (como imagens ou tabelas).  

Os dados são gerados **dinamicamente** pelo ambiente de simulação criado com o **PettingZoo**. Nesse ambiente:  

- 🤖 O agente é treinado para **navegar em um grid** até atingir um **ponto-alvo**.  
- 🛑 Ele precisa **desviar de obstáculos** enquanto realiza a navegação.  

Cada episódio de treinamento produz **trajetórias** compostas por:  
- 📌 **Estados** – posição e contexto atual do agente no grid  
- 🎯 **Ações** – movimentos executados pelo agente  
- 🏆 **Recompensas** – feedback recebido com base nas ações  
- 🔄 **Próximos estados** – a nova situação após cada ação  

Esses registros equivalem a um **dataset supervisionado tradicional**, mas são **gerados em tempo real**, à medida que o agente interage com o ambiente.  
