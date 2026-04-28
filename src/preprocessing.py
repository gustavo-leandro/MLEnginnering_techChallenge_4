"""
Pipeline de pré-processamento dos dados brutos de VALE3.SA para treinamento do LSTM.

Fluxo executado por `preprocess()`:
  1. Carrega `data/raw/vale3_raw.csv` e isola a coluna Close.
  2. Divide a série em treino (70 %), validação (15 %) e teste (15 %)
     respeitando a ordem temporal — sem embaralhamento.
  3. Normaliza os preços para [0, 1] com MinMaxScaler fitado APENAS no treino,
     evitando data leakage para validação e teste.
  4. Gera janelas deslizantes de `LOOK_BACK` dias:
       X[i] = preços dos dias [i-60, i)  →  y[i] = preço do dia i
     Cada amostra em X tem shape (look_back, 1), pronto para entrada no LSTM.
  5. Salva os arrays em `data/processed/` e o scaler em `models/scaler.pkl`
     (necessário para inverter a normalização na inferência).

Nota: durante o desenvolvimento foram testadas 6 features técnicas (MA7, MA21,
Volume, RSI, MACD_hist). O modelo com só Close apresentou MAPE=1.23% vs 1.29%
com features técnicas — a versão simples foi mantida como configuração final.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data/raw/vale3_raw.csv"
PROCESSED_DIR = ROOT / "data/processed"
MODELS_DIR = ROOT / "models"

LOOK_BACK = 90
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO implícito = 0.15


def load_raw() -> pd.DataFrame:
    """Lê o CSV bruto, mantém apenas Close e aplica forward fill nos nulos."""
    df = pd.read_csv(RAW_CSV, index_col=0, parse_dates=True)
    df = df[["Close"]].copy()
    df = df.ffill()
    return df


def split_sequential(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Divide o array em treino / validação / teste sem embaralhar."""
    n = len(data)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    return data[:train_end], data[train_end:val_end], data[val_end:]


def make_sequences(data: np.ndarray, look_back: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Gera pares (X, y) de janelas deslizantes.

    Para cada posição i >= look_back:
      X[i] = data[i-look_back : i]   (janela de entrada)
      y[i] = data[i]                  (valor alvo — próximo dia)
    """
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def preprocess() -> dict:
    """
    Executa a pipeline completa e persiste os artefatos.

    Arquivos gerados:
      data/processed/X_train.npy  shape (n_train, 90, 1)
      data/processed/y_train.npy  shape (n_train,)
      data/processed/X_val.npy    shape (n_val,   90, 1)
      data/processed/y_val.npy    shape (n_val,)
      data/processed/X_test.npy   shape (n_test,  90, 1)
      data/processed/y_test.npy   shape (n_test,)
      models/scaler.pkl            MinMaxScaler fitado no treino
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    prices = df["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_raw, val_raw, test_raw = split_sequential(prices)

    # fit apenas no treino para evitar data leakage
    train_scaled = scaler.fit_transform(train_raw)
    val_scaled = scaler.transform(val_raw)
    test_scaled = scaler.transform(test_raw)

    X_train, y_train = make_sequences(train_scaled, LOOK_BACK)
    X_val, y_val = make_sequences(val_scaled, LOOK_BACK)
    X_test, y_test = make_sequences(test_scaled, LOOK_BACK)

    # LSTM espera (samples, timesteps, features)
    X_train = X_train.reshape(*X_train.shape, 1)
    X_val = X_val.reshape(*X_val.shape, 1)
    X_test = X_test.reshape(*X_test.shape, 1)

    np.save(PROCESSED_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "X_val.npy", X_val)
    np.save(PROCESSED_DIR / "y_val.npy", y_val)
    np.save(PROCESSED_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)

    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    summary = {
        "look_back": LOOK_BACK,
        "total_samples": len(prices),
        "train": X_train.shape,
        "val": X_val.shape,
        "test": X_test.shape,
    }
    for k, v in summary.items():
        print(f"{k}: {v}")

    return summary


if __name__ == "__main__":
    preprocess()
