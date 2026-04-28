"""
Treina o modelo LSTM e registra o experimento no MLflow.

Fluxo:
  1. Carrega os arrays pré-processados de `data/processed/` e os dois scalers.
  2. Abre um run MLflow e loga os hiperparâmetros.
  3. Treina com EarlyStopping (paciência=10) e ModelCheckpoint.
  4. Avalia no conjunto de teste: MAE, RMSE, MAPE (escala original de preço BRL).
  5. Loga métricas, gráfico e o modelo no MLflow Model Registry.
  6. Salva o modelo final em `models/lstm_model.keras`.
"""

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error

from model import build_model

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data/processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

HYPERPARAMS = {
    "look_back": 90,
    "n_features": 1,
    "units": 128,
    "dropout_rate": 0.2,
    "learning_rate": 1e-3,
    "epochs": 100,
    "batch_size": 32,
}


def load_data() -> tuple:
    """Carrega arrays .npy e o scaler serializado."""
    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    X_val = np.load(PROCESSED_DIR / "X_val.npy")
    y_val = np.load(PROCESSED_DIR / "y_val.npy")
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")
    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula MAE, RMSE e MAPE em escala original de preço (BRL)."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"mae": round(float(mae), 4), "rmse": round(float(rmse), 4), "mape": round(float(mape), 4)}


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, metrics: dict) -> Path:
    """Gera e salva o gráfico de predito vs. real."""
    REPORTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(y_true, label="Real", color="steelblue", linewidth=1.2)
    ax.plot(y_pred, label="Predito", color="coral", linewidth=1.2, linestyle="--")
    ax.set_title(
        f"VALE3.SA — Predito vs. Real  |  MAE={metrics['mae']:.2f}  "
        f"RMSE={metrics['rmse']:.2f}  MAPE={metrics['mape']:.2f}%"
    )
    ax.set_ylabel("Preço (BRL)")
    ax.legend()
    plt.tight_layout()
    path = REPORTS_DIR / "predictions.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def train() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    (MODELS_DIR / "checkpoints").mkdir(exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_data()
    n_features = X_train.shape[2]

    mlflow.set_experiment("vale3-lstm")

    with mlflow.start_run():
        mlflow.log_params({**HYPERPARAMS, "n_features": n_features})

        model = build_model(
            look_back=HYPERPARAMS["look_back"],
            n_features=n_features,
            units=HYPERPARAMS["units"],
            dropout_rate=HYPERPARAMS["dropout_rate"],
            learning_rate=HYPERPARAMS["learning_rate"],
        )

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
            ModelCheckpoint(
                filepath=str(MODELS_DIR / "checkpoints" / "best.keras"),
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            ),
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=HYPERPARAMS["epochs"],
            batch_size=HYPERPARAMS["batch_size"],
            callbacks=callbacks,
            verbose=1,
        )

        for epoch, (loss, val_loss) in enumerate(
            zip(history.history["loss"], history.history["val_loss"])
        ):
            mlflow.log_metrics({"train_loss": loss, "val_loss": val_loss}, step=epoch)

        y_pred_scaled = model.predict(X_test)
        y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
        y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        metrics = compute_metrics(y_true, y_pred)
        print(f"\nMétricas no teste: {metrics}")
        mlflow.log_metrics(metrics)

        plot_path = plot_predictions(y_true, y_pred, metrics)
        mlflow.log_artifact(str(plot_path))

        with open(REPORTS_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        mlflow.keras.log_model(model, artifact_path="lstm_model", registered_model_name="vale3-lstm")
        model.save(MODELS_DIR / "lstm_model.keras")
        print(f"Modelo salvo em {MODELS_DIR / 'lstm_model.keras'}")


if __name__ == "__main__":
    train()
