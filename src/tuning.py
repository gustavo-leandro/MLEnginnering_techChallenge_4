"""
Grid search de hiperparâmetros para o modelo LSTM com features técnicas.
Cada combinação é registrada como um run separado no MLflow.

Grade testada (9 runs):
  look_back  : [30, 60, 90]
  units      : [32, 64, 128]
  dropout    : 0.2  (fixo)
  lr         : 1e-3 (fixo)

Ao final, abra a UI para comparar:
  .venv/bin/mlflow ui --backend-store-uri src/mlruns
  Acesse: http://localhost:5000
"""

import pickle
from itertools import product
from pathlib import Path

import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping

from model import build_model
from preprocessing import FEATURES, make_sequences
from train import compute_metrics

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

GRID = {
    "look_back": [30, 60, 90],
    "units": [32, 64, 128],
    "dropout_rate": [0.2],
    "learning_rate": [1e-3],
    "epochs": [100],
    "batch_size": [32],
}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def build_sequences(look_back: int, feature_scaler, close_scaler):
    """Reconstrói sequências com o look_back dado a partir do CSV bruto."""
    from preprocessing import add_technical_indicators

    df_raw = pd.read_csv(ROOT / "data/raw/vale3_raw.csv", index_col=0, parse_dates=True)
    df = add_technical_indicators(df_raw[["Close", "Volume"]].ffill())

    feature_data = df.values
    close_data = df[["Close"]].values

    n = len(feature_data)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_feat = feature_scaler.transform(feature_data[:train_end])
    val_feat = feature_scaler.transform(feature_data[train_end:val_end])
    test_feat = feature_scaler.transform(feature_data[val_end:])

    train_close = close_scaler.transform(close_data[:train_end]).flatten()
    val_close = close_scaler.transform(close_data[train_end:val_end]).flatten()
    test_close = close_scaler.transform(close_data[val_end:]).flatten()

    X_train, y_train = make_sequences(train_feat, train_close, look_back)
    X_val, y_val = make_sequences(val_feat, val_close, look_back)
    X_test, y_test = make_sequences(test_feat, test_close, look_back)

    return X_train, y_train, X_val, y_val, X_test, y_test


def run_combination(params: dict, feature_scaler, close_scaler) -> dict:
    """Executa um único run MLflow para a combinação de hiperparâmetros dada."""
    look_back = params["look_back"]
    run_name = f"lb{look_back}_u{params['units']}_dr{params['dropout_rate']}"

    X_train, y_train, X_val, y_val, X_test, y_test = build_sequences(look_back, feature_scaler, close_scaler)
    n_features = X_train.shape[2]

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({**params, "n_features": n_features, "features": str(FEATURES)})

        model = build_model(
            look_back=look_back,
            n_features=n_features,
            units=params["units"],
            dropout_rate=params["dropout_rate"],
            learning_rate=params["learning_rate"],
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0)],
            verbose=0,
        )

        for epoch, (loss, val_loss) in enumerate(
            zip(history.history["loss"], history.history["val_loss"])
        ):
            mlflow.log_metrics({"train_loss": loss, "val_loss": val_loss}, step=epoch)

        y_pred_scaled = model.predict(X_test, verbose=0)
        y_pred = close_scaler.inverse_transform(y_pred_scaled).flatten()
        y_true = close_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        metrics = compute_metrics(y_true, y_pred)
        mlflow.log_metrics(metrics)

        epochs_run = len(history.history["loss"])
        print(
            f"  {run_name:35s} | epochs={epochs_run:3d} | "
            f"MAE={metrics['mae']:.2f} | RMSE={metrics['rmse']:.2f} | MAPE={metrics['mape']:.2f}%"
        )

    return {"run_name": run_name, **params, **metrics}


def main() -> None:
    mlflow.set_experiment("vale3-lstm-tuning-v2")

    with open(MODELS_DIR / "feature_scaler.pkl", "rb") as f:
        feature_scaler = pickle.load(f)
    with open(MODELS_DIR / "close_scaler.pkl", "rb") as f:
        close_scaler = pickle.load(f)

    keys = list(GRID.keys())
    combinations = [dict(zip(keys, values)) for values in product(*GRID.values())]

    print(f"Iniciando grid search v2 (com features técnicas): {len(combinations)} combinações\n")
    print(f"  {'run':35s} | epochs | MAE    | RMSE   | MAPE")
    print("  " + "-" * 75)

    results = []
    for i, params in enumerate(combinations, 1):
        print(f"[{i}/{len(combinations)}] ", end="", flush=True)
        result = run_combination(params, feature_scaler, close_scaler)
        results.append(result)

    results.sort(key=lambda r: r["mape"])
    print("\n--- Ranking por MAPE ---")
    print(f"  {'run':35s} | MAE    | RMSE   | MAPE")
    print("  " + "-" * 65)
    for r in results:
        print(f"  {r['run_name']:35s} | {r['mae']:6.2f} | {r['rmse']:6.2f} | {r['mape']:6.2f}%")

    best = results[0]
    print(f"\nMelhor configuração: {best['run_name']}")
    print(f"  look_back={best['look_back']}, units={best['units']}, dropout={best['dropout_rate']}")
    print(f"  MAE={best['mae']} | RMSE={best['rmse']} | MAPE={best['mape']}%")
    print("\nAbra o MLflow UI:")
    print("  .venv/bin/mlflow ui --backend-store-uri src/mlruns")
    print("  Acesse: http://localhost:5000")


if __name__ == "__main__":
    main()
