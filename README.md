# Tech Challenge 4 — LSTM Stock Price Predictor

Pipeline completa de predição do preço de fechamento de **VALE3.SA** usando LSTM,
do dado bruto ao deploy em API REST com monitoramento.

---

## Como o projeto funciona

```
yfinance → preprocessing → LSTM → MLflow → FastAPI → Prometheus/Grafana
```

1. **Coleta** (`src/data_collection.py`) — baixa dados históricos de VALE3.SA via `yfinance` (jan/2018 → jul/2024) e salva em `data/raw/`.
2. **EDA** (`notebooks/eda.ipynb`) — análise exploratória: série temporal, volume, retornos, volatilidade mensal e outliers.
3. **Pré-processamento** (`src/preprocessing.py`) — calcula indicadores técnicos, normaliza com `MinMaxScaler` e gera janelas deslizantes de look-back para entrada no LSTM.
4. **Modelo** (`src/model.py`) — arquitetura LSTM + Dropout + Dense compilada com Keras.
5. **Treino** (`src/train.py`) — treina com `EarlyStopping`, loga no MLflow e salva o modelo final.
6. **Tuning** (`src/tuning.py`) — grid search automático de hiperparâmetros, cada run registrado no MLflow.
7. **API** (`app/`) — FastAPI servindo o modelo com endpoints `/health`, `/predict` e `/metrics`.

---

## Estrutura de pastas

```
├── data/
│   ├── raw/          # CSV bruto do yfinance
│   └── processed/    # Arrays .npy prontos para treino
├── models/           # Scaler, modelo final e checkpoints
├── notebooks/        # EDA interativo
├── reports/          # Gráficos e métricas
├── src/              # Scripts de coleta, preprocessing, modelo, treino e tuning
├── app/              # API FastAPI
└── requirements.txt
```

---

## Como rodar

```bash
# 1. Ativar o ambiente
source .venv/bin/activate

# 2. Coletar dados
python src/data_collection.py

# 3. Pré-processar
python src/preprocessing.py

# 4. Treinar o modelo final
python src/train.py

# 5. Subir a API
uvicorn app.main:app --reload

# 6. Ver experimentos no MLflow
mlflow ui --backend-store-uri src/mlruns
```

---

## A jornada do modelo — como chegamos aqui

### Ponto de partida: só o preço de fechamento

A primeira versão do pré-processamento usava apenas a coluna `Close` como feature.
O modelo recebia uma janela de N dias de preço e previa o dia seguinte — simples e direto.

Rodamos um **grid search** com 9 combinações de hiperparâmetros:

| look_back | units | MAPE |
|-----------|-------|------|
| 30 | 32 | 1.57% |
| 30 | 64 | 1.38% |
| 30 | 128 | 1.23% |
| 60 | 32 | 1.46% |
| 60 | 64 | 1.36% |
| 60 | 128 | 1.75% |
| 90 | 32 | 1.30% |
| 90 | 64 | 1.59% |
| **90** | **128** | **1.23%** |

Os dois melhores empataram em MAPE=1.23% (`lb30_u128` e `lb90_u128`).
O desempate pelo RMSE favoreceu `look_back=90, units=128` (RMSE=0.83 vs 0.84).

### O problema: o baseline ingênuo era melhor

Antes de celebrar, calculamos o **baseline ingênuo** — o modelo mais simples possível:
*"o preço de amanhã é igual ao de hoje"*.

| Modelo | MAE | RMSE | MAPE |
|--------|-----|------|------|
| Baseline ingênuo | 0.56 | 0.73 | **1.03%** |
| Melhor LSTM v1 | 0.66 | 0.83 | 1.23% |

O LSTM não bateu o baseline. Isso é comum em séries financeiras — preços seguem um
comportamento próximo ao de caminhada aleatória (*random walk*), e o valor de ontem
já carrega a maior parte da informação do amanhã.

### Tentativa de melhoria: features técnicas

Para dar ao modelo mais contexto além do preço puro, adicionamos 5 indicadores técnicos:

| Feature | O que representa |
|---------|-----------------|
| `MA7` | Média móvel de 7 dias — tendência de curto prazo |
| `MA21` | Média móvel de 21 dias — tendência de médio prazo |
| `Volume` | Volume negociado — pressão compradora/vendedora |
| `RSI` | Relative Strength Index (14d) — momentum e sobrecompra/sobrevenda |
| `MACD_hist` | Histograma MACD — divergência de médias exponenciais |

Rodamos o mesmo grid search com as 6 features (experimento `vale3-lstm-tuning-v2`):

| look_back | units | MAPE |
|-----------|-------|------|
| 90 | 32 | 2.37% |
| 60 | 32 | 2.27% |
| 30 | 32 | 2.26% |
| 30 | 64 | 1.44% |
| 60 | 128 | 1.44% |
| 30 | 128 | 1.36% |
| 60 | 64 | 1.39% |
| 90 | 128 | 1.49% |
| **90** | **64** | **1.29%** |

### Resultado final

O melhor modelo com features técnicas (`look_back=90, units=64`) atingiu MAPE=1.29% —
melhor que qualquer configuração com units=32, mas ainda acima do baseline e da v1.

**Conclusão:** as features técnicas não trouxeram ganho consistente neste dataset.
O modelo com só `Close` e `look_back=90, units=128` foi selecionado como configuração final.

### Treino definitivo

Com a configuração vencedora, rodamos o treino completo (até 100 épocas com EarlyStopping).
O modelo parou no epoch 84, restaurando os pesos do epoch 74 — ponto de menor val_loss.

O resultado superou todos os runs do tuning:

| Métrica | Tuning (melhor run) | **Treino final** |
|---------|--------------------|--------------------|
| MAE | 0.66 BRL | **0.61 BRL** |
| RMSE | 0.83 BRL | **0.76 BRL** |
| MAPE | 1.23% | **1.13%** |

Com 1.13% de MAPE, o modelo praticamente empatou com o baseline ingênuo (1.03%) —
um resultado honesto para predição de ações, dentro da faixa *boa* da literatura (1–3%).
O modelo e o scaler foram registrados no MLflow Model Registry (`vale3-lstm` v1).

### Configuração final do modelo

```python
look_back    = 90       # dias de histórico por amostra
n_features   = 1        # apenas Close
units        = 128      # neurônios na 1ª camada LSTM
dropout_rate = 0.2
learning_rate = 0.001
epochs       = 84       # parada antecipada (melhor epoch: 74)
```

**Arquitetura:**
```
Input(90, 1) → LSTM(128, return_sequences) → Dropout(0.2)
             → LSTM(64)                    → Dropout(0.2)
             → Dense(1)
```
