# Tech Challenge 4: LSTM Stock Price Predictor

Pipeline completa de predição do preço de fechamento de **VALE3.SA** usando LSTM,
do dado bruto ao deploy em API REST com monitoramento.

---

## Como o projeto funciona

```
yfinance -> preprocessing -> LSTM -> MLflow -> FastAPI -> Prometheus/Grafana
```

1. **Coleta** (`src/data_collection.py`): baixa dados históricos de VALE3.SA via `yfinance` (jan/2018 ate jul/2024) e salva em `data/raw/`.
2. **EDA** (`notebooks/eda.ipynb`): analise exploratoria com serie temporal, volume, retornos, volatilidade mensal e outliers.
3. **Pre-processamento** (`src/preprocessing.py`): normaliza com `MinMaxScaler` e gera janelas deslizantes de look-back para entrada no LSTM.
4. **Modelo** (`src/model.py`): arquitetura LSTM + Dropout + Dense compilada com Keras.
5. **Treino** (`src/train.py`): treina com `EarlyStopping`, loga no MLflow e salva o modelo final.
6. **Tuning** (`src/tuning.py`): grid search automatico de hiperparametros, cada run registrado no MLflow.
7. **API** (`app/`): FastAPI servindo o modelo com endpoints `/health`, `/predict` e `/metrics`.

---

## Estrutura de pastas

```
data/
    raw/                    # CSV bruto do yfinance
    processed/              # Arrays .npy prontos para treino
models/
    lstm_model.keras        # Modelo final treinado
    scaler.pkl              # MinMaxScaler fitado no treino
    config.json             # Hiperparametros e metricas do modelo final
    checkpoints/            # Melhor checkpoint por epoca
notebooks/
    eda.ipynb               # Analise exploratoria interativa
reports/
    predictions.png         # Grafico predito vs. real no conjunto de teste
    metrics.json            # MAE, RMSE, MAPE do modelo final
src/
    data_collection.py      # Coleta via yfinance
    preprocessing.py        # Normalizacao e geracao de sequencias
    model.py                # Definicao da arquitetura LSTM
    train.py                # Treino com MLflow
    tuning.py               # Grid search de hiperparametros
app/
    main.py                 # Entrypoint FastAPI
    schemas.py              # Schemas Pydantic (request/response)
    routes/predict.py       # Endpoint POST /predict
    services/predictor.py   # Singleton de inferencia
tests/
    test_api.py             # Testes de integracao da API
monitoring/
    prometheus.yml          # Configuracao do Prometheus
requirements.txt
```

---

## Como rodar

```bash
# 1. Ativar o ambiente
source .venv/bin/activate

# 2. Coletar dados
python src/data_collection.py

# 3. Pre-processar
python src/preprocessing.py

# 4. Treinar o modelo final
cd src && python train.py

# 5. Subir a API
uvicorn app.main:app --reload

# 6. Ver experimentos no MLflow
mlflow ui --backend-store-uri src/mlruns

# 7. Rodar os testes
pytest tests/ -v
```

### Endpoints da API

| Metodo | Endpoint | Descricao |
|--------|----------|-----------|
| GET | `/health` | Verifica se a API esta no ar |
| POST | `/predict` | Recebe serie historica e retorna preco previsto |
| GET | `/metrics` | Metricas Prometheus (latencia, requests, erros, CPU, memoria) |
| GET | `/docs` | Documentacao interativa Swagger |

Exemplo de request para `/predict`:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"prices": [61.5, 62.0, 61.8, ...]}'  # minimo 90 valores
```

Exemplo de response:

```json
{
  "ticker": "VALE3.SA",
  "predicted_close": 59.01,
  "look_back_used": 90
}
```

### Servicos de monitoramento

| Servico | Porta | Como subir |
|---------|-------|------------|
| API FastAPI | 8000 | `uvicorn app.main:app --reload` |
| Prometheus | 9090 | `prometheus --config.file=monitoring/prometheus.yml` |
| Grafana | 3000 | `brew services start grafana` |

---

### Prometheus

O Prometheus coleta metricas fazendo scrape do endpoint `GET /metrics` da API a cada 15 segundos, conforme configurado em `monitoring/prometheus.yml`.

**Como verificar se o Prometheus esta coletando:**

1. Acesse `http://localhost:9090/targets`
2. O job `vale3-api` deve aparecer com estado **UP**
3. Para consultar uma metrica diretamente, acesse `http://localhost:9090` e execute uma query PromQL, por exemplo: `http_requests_total`

**Metricas expostas em `GET /metrics`:**

| Metrica | Tipo | Descricao |
|---------|------|-----------|
| `http_requests_total` | Counter | Total de requests por metodo, endpoint e status code |
| `http_request_duration_seconds` | Histogram | Latencia por endpoint (buckets ate 5s) |
| `http_request_errors_total` | Counter | Total de erros 4xx/5xx por endpoint |
| `process_cpu_percent` | Gauge | Uso de CPU do processo em % |
| `process_memory_bytes` | Gauge | Uso de memoria do processo em bytes |
| `model_last_prediction_brl` | Gauge | Ultimo preco previsto em BRL |
| `model_drift_mae` | Gauge | MAE da janela deslizante das ultimas 30 predicoes |
| `model_drift_rmse` | Gauge | RMSE da janela deslizante das ultimas 30 predicoes |

Exemplo de consulta PromQL para taxa de requests por segundo:

```
rate(http_requests_total[1m])
```

Exemplo para latencia no percentil 95:

```
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{endpoint="/predict"}[2m]))
```

---

### Grafana

**Datasource:** o Prometheus ja esta provisionado automaticamente via `conf/provisioning/datasources/prometheus.yml` (uid: `vale3-prometheus`). Nao e necessario adicionar a datasource manualmente.

**Setup inicial (apenas uma vez):**

1. Suba a API, o Prometheus e o Grafana conforme a tabela acima.
2. Reinicie o Grafana para carregar o provisionamento: `brew services restart grafana`
3. Acesse `http://localhost:3000` (usuario: `admin` / senha: `admin`).
4. Va em **Dashboards > Import**, clique em **Upload dashboard JSON file** e selecione `monitoring/grafana_dashboard.json`.

O dashboard ja vem configurado com a datasource correta em todos os paineis.

**Populando o dashboard com dados:**

```bash
python scripts/simulate_traffic.py
```

O script envia uma request ao `/predict` a cada 1 segundo com precos sinteticos. Dentro de 15-30 segundos os graficos comecam a ser populados. Pressione `Ctrl+C` para encerrar.

**Paineis do dashboard:**

| Painel | Tipo | O que mostra |
|--------|------|-------------|
| Requests por segundo | Timeseries | Taxa de requests por endpoint e status code |
| Taxa de erros (4xx/5xx) | Timeseries | Requests com erro por endpoint |
| Latencia p50/p95/p99 | Timeseries | Percentis de latencia do /predict |
| CPU e Memoria | Timeseries | Uso de recursos do processo da API |
| Ultimo preco previsto | Stat | Valor em BRL da ultima predicao |
| Drift MAE | Stat | MAE da janela deslizante (verde abaixo de 1.5, amarelo ate 2.0, vermelho acima) |
| Drift RMSE | Stat | RMSE da janela deslizante |
| Historico de latencia | Timeseries | Latencia media do /predict ao longo do tempo |

**Monitoramento de drift:**

Cada predicao e registrada em `reports/predictions_log.jsonl`. Quando precos reais sao fornecidos junto com a predicao, o sistema calcula MAE e RMSE numa janela deslizante das ultimas 30 observacoes e emite um warning no log se o MAE ultrapassar 2.0 BRL, indicando possivel degradacao do modelo em producao.

---

## A jornada do modelo: como chegamos aqui

### Ponto de partida: so o preco de fechamento

A primeira versao do pre-processamento usava apenas a coluna `Close` como feature.
O modelo recebia uma janela de N dias de preco e previa o dia seguinte, simples e direto.

Rodamos um **grid search** com 9 combinacoes de hiperparametros:

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

### O problema: o baseline ingenuo era melhor

Antes de celebrar, calculamos o **baseline ingenuo**, o modelo mais simples possivel:
"o preco de amanha e igual ao de hoje".

| Modelo | MAE | RMSE | MAPE |
|--------|-----|------|------|
| Baseline ingenuo | 0.56 | 0.73 | **1.03%** |
| Melhor LSTM v1 | 0.66 | 0.83 | 1.23% |

O LSTM nao bateu o baseline. Isso e comum em series financeiras: precos seguem um
comportamento proximo ao de caminhada aleatoria (random walk), e o valor de ontem
ja carrega a maior parte da informacao do amanha.

### Tentativa de melhoria: features tecnicas

Para dar ao modelo mais contexto alem do preco puro, adicionamos 5 indicadores tecnicos:

| Feature | O que representa |
|---------|-----------------|
| `MA7` | Media movel de 7 dias, tendencia de curto prazo |
| `MA21` | Media movel de 21 dias, tendencia de medio prazo |
| `Volume` | Volume negociado, pressao compradora/vendedora |
| `RSI` | Relative Strength Index (14d), momentum e sobrecompra/sobrevenda |
| `MACD_hist` | Histograma MACD, divergencia de medias exponenciais |

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

O melhor modelo com features tecnicas (`look_back=90, units=64`) atingiu MAPE=1.29%,
melhor que qualquer configuracao com units=32, mas ainda acima do baseline e da v1.

As features tecnicas nao trouxeram ganho consistente neste dataset.
O modelo com so `Close` e `look_back=90, units=128` foi selecionado como configuracao final.

### Treino definitivo

Com a configuracao vencedora, rodamos o treino completo (ate 100 epocas com EarlyStopping).
O modelo parou no epoch 84, restaurando os pesos do epoch 74, ponto de menor val_loss.

O resultado superou todos os runs do tuning:

| Metrica | Tuning (melhor run) | Treino final |
|---------|---------------------|--------------|
| MAE | 0.66 BRL | **0.61 BRL** |
| RMSE | 0.83 BRL | **0.76 BRL** |
| MAPE | 1.23% | **1.13%** |

Com MAPE de 1.13%, o modelo praticamente empatou com o baseline ingenuo (1.03%),
um resultado honesto para predicao de acoes, dentro da faixa boa da literatura (1-3%).
O modelo e o scaler foram registrados no MLflow Model Registry (`vale3-lstm` v1).

### Configuracao final do modelo

```python
look_back    = 90       # dias de historico por amostra
n_features   = 1        # apenas Close
units        = 128      # neuronios na 1a camada LSTM
dropout_rate = 0.2
learning_rate = 0.001
epochs       = 84       # parada antecipada (melhor epoch: 74)
```

Arquitetura:
```
Input(90, 1) -> LSTM(128, return_sequences) -> Dropout(0.2)
             -> LSTM(64)                    -> Dropout(0.2)
             -> Dense(1)
```

### Registro e exportacao do modelo

Com o modelo treinado e avaliado, o passo seguinte foi formalizar os artefatos para uso em producao.

Foram gerados tres artefatos principais:

- `models/lstm_model.keras`: modelo Keras serializado, pronto para inferencia.
- `models/scaler.pkl`: MinMaxScaler fitado exclusivamente no conjunto de treino, necessario para normalizar os precos de entrada e inverter a normalizacao na saida.
- `models/config.json`: hiperparametros, features utilizadas e metricas do modelo final, servindo como documentacao imutavel da versao em producao.

O modelo foi tambem registrado no MLflow Model Registry como `vale3-lstm` versao 1 e promovido para o stage `Production`. Para validar o fluxo completo, realizamos uma inferencia de ponta a ponta carregando o modelo diretamente do registry via `mlflow.pyfunc.load_model`, confirmando que o preco previsto bate com o resultado esperado.

### Deploy: API REST com FastAPI

A API foi construida com FastAPI e organizada em tres camadas:

- **Schema** (`app/schemas.py`): validacao de entrada via Pydantic. O payload deve conter pelo menos 90 precos positivos. Payloads invalidos retornam 422 automaticamente, sem chegar ao modelo.
- **Servico** (`app/services/predictor.py`): singleton que carrega o modelo e o scaler uma unica vez no startup da aplicacao. Cada request reutiliza a mesma instancia, evitando overhead de leitura de disco a cada chamada.
- **Rota** (`app/routes/predict.py`): endpoint `POST /predict` que delega ao servico e devolve ticker, preco previsto e janela utilizada.

A API foi testada localmente com `uvicorn` antes de qualquer containerizacao, validando os dois endpoints principais (`/health` e `/predict`) com dados reais do conjunto de teste.

### Testes

Foram escritos 6 testes de integracao com `pytest` e `TestClient` do FastAPI, cobrindo os cenarios principais:

- `/health` retorna status ok.
- `/predict` com 90 precos reais retorna um preco positivo e o ticker correto.
- Payloads com mais de 90 precos sao aceitos (usa os ultimos 90).
- Payloads com menos de 90 precos sao rejeitados com 422.
- Precos negativos ou zero sao rejeitados com 422.

Todos os 6 testes passaram.

### Monitoramento

O monitoramento foi implementado em `app/monitoring.py` e integrado automaticamente via middleware HTTP no FastAPI.

**Metricas Prometheus** expostas em `GET /metrics`:

| Metrica | Tipo | Descricao |
|---------|------|-----------|
| `http_requests_total` | Counter | Total de requests por metodo, endpoint e status code |
| `http_request_duration_seconds` | Histogram | Latencia por endpoint (buckets ate 5s) |
| `http_request_errors_total` | Counter | Total de erros 4xx/5xx por endpoint |
| `process_cpu_percent` | Gauge | Uso de CPU do processo |
| `process_memory_bytes` | Gauge | Uso de memoria do processo |
| `model_last_prediction_brl` | Gauge | Ultimo preco previsto em BRL |
| `model_drift_mae` | Gauge | MAE da janela deslizante de 30 predicoes |
| `model_drift_rmse` | Gauge | RMSE da janela deslizante de 30 predicoes |

**Logging estruturado**: cada predicao gera um log JSON no stdout com timestamp, ticker, look_back, preco previsto e latencia em ms.

**Monitoramento de drift**: cada predicao e registrada em `reports/predictions_log.jsonl`. Quando precos reais sao fornecidos, o sistema calcula MAE/RMSE numa janela deslizante de 30 pares e emite um warning se MAE ultrapassar 2.0 BRL.
