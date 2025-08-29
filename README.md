AI Crypto Trader 🤖

Este repositório contém o código-fonte de um ecossistema completo para trading algorítmico no mercado de criptomoedas, utilizando Machine Learning para prever movimentos de preço e executar operações de forma autônoma.

O Que Temos Neste Projeto? (Features)
O sistema é composto por múltiplos módulos que trabalham em conjunto para cobrir todo o ciclo de vida de uma estratégia de trading quantitativo:

🧠 Cérebro de Inteligência Artificial:

Treinamento de Modelos (model_trainer.py): Utiliza XGBoost para treinar modelos preditivos especializados para cada par de moedas. O script processa um grande volume de dados históricos (recomenda-se 2 anos de candles de 5 minutos) para que o modelo aprenda os padrões do mercado.

Servidor de Inferência (inference_server.py): Uma API Flask dedicada que serve as previsões do modelo em tempo real, desacoplando a IA do robô principal.

📊 Módulos de Estratégia e Validação:

Backtester (backtest.py): Valida a performance da estratégia em dados históricos, simulando custos realistas como taxas e slippage.

Otimizador (otimizador.py): Encontra os melhores hiperparâmetros da estratégia (como limiares de confiança e ADX) para maximizar o retorno ajustado ao risco.

Engenharia de Features (feature_engineering.py): Módulo central que calcula dezenas de indicadores técnicos para alimentar o modelo de IA.

⚙️ Motor de Execução em Tempo Real (bot_bybit.py):

O núcleo operacional que consome os sinais da IA, aplica uma gestão de risco rigorosa, e executa as ordens de compra/venda na exchange Bybit.

Possui um sistema de Trailing Stop Loss que opera em uma thread separada para proteger as operações de forma dinâmica.

Flexível para ser adaptado a outras exchanges com modificações pontuais.

🖥️ Monitoramento e Controle (painel.py):

Uma interface web construída com NiceGUI que permite monitorar o status do bot.

Exibe P&L (lucro e prejuízo), acompanha posições abertas e analisa o histórico de trades de forma intuitiva.

Como Instalar e Executar o Projeto 🚀
Siga este guia para colocar o sistema em funcionamento na sua máquina.

1. Pré-requisitos
Python 3.8 ou superior.

Uma conta na exchange Bybit (real ou testnet) com chaves de API.

2. Instalação
Clone o repositório:

Bash

git clone https://github.com/ronaldoneves95/AI_Crypto_Trader.git
cd AI_Crypto_Trader
Crie e ative um ambiente virtual:

Bash

python -m venv venv
# No Windows
.\venv\Scripts\activate
# No macOS/Linux
source venv/bin/activate
Instale as dependências:

Bash

pip install -r requirements.txt
3. Configuração Essencial
Dados Históricos:

Crie uma pasta chamada data/ na raiz do projeto.

Adicione seus arquivos de dados históricos em formato .csv dentro desta pasta (ex: dados_historicos_BTCUSDT_5m.csv). O modelo foi projetado para ser treinado com dados de candles de 5 minutos dos últimos 2 anos.

Chaves de API:

Abra os arquivos bot_bybit.py e painel.py.

Insira suas chaves de API da Bybit nos locais indicados:


API_KEY = "SUA_API_KEY"
API_SECRET = "SEU_API_SECRET"
Arquivo de Configuração (config.yaml):

Crie um arquivo chamado config.yaml (se não houver) na raiz do projeto para definir os parâmetros da estratégia. Você pode começar com este exemplo:

YAML

risk_perc: 0.01
ml_thresholds:
  BTCUSDT: 0.65
  ETHUSDT: 0.65
adx_threshold: 25
4. Execução
Treine o Modelo de IA:

Antes de iniciar o bot pela primeira vez, você precisa treinar o modelo com seus dados. Execute o seguinte comando no terminal:


python model_trainer.py
Isso irá ler os arquivos .csv da pasta data/ e criar os arquivos de modelo (.pkl) na raiz do projeto.

Inicie o Sistema Completo:

O script bot_bybyit.py foi projetado para iniciar o servidor de inferência, o painel de controle e a lógica de trading.


python bot_bybit.py
Acesse o Painel de Controle:

Abra seu navegador e acesse http://localhost:8080 para monitorar o robô em tempo real.

Autor
Ronaldo Neves Barbosa Neto

LinkedIn: https://www.linkedin.com/in/ronaldo-neto-b5247222b
