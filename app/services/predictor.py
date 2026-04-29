"""
Servico de inferencia: carrega modelo e scaler uma unica vez na inicializacao
e expoe o metodo predict() para uso pelos endpoints.
"""

import json
import pickle
from pathlib import Path

import numpy as np
from keras.models import load_model

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "models" / "config.json"
MODEL_PATH = ROOT / "models" / "lstm_model.keras"
SCALER_PATH = ROOT / "models" / "scaler.pkl"


class Predictor:
    def __init__(self):
        with open(CONFIG_PATH) as f:
            self.config = json.load(f)

        self.look_back: int = self.config["look_back"]
        self.ticker: str = self.config["ticker"]

        self.model = load_model(MODEL_PATH)

        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)

    def predict(self, prices: list[float]) -> float:
        """
        Recebe uma lista de precos historicos e retorna o preco previsto do proximo dia.

        Usa os ultimos `look_back` valores da lista como janela de entrada.
        Normaliza com o scaler fitado no treino e inverte a transformacao na saida.
        """
        window = np.array(prices[-self.look_back:], dtype=np.float32).reshape(-1, 1)
        window_scaled = self.scaler.transform(window)
        X = window_scaled.reshape(1, self.look_back, 1)
        pred_scaled = self.model.predict(X, verbose=0)
        pred_price = self.scaler.inverse_transform(pred_scaled)[0][0]
        return round(float(pred_price), 2)


# instancia singleton carregada uma vez no startup da aplicacao
predictor = Predictor()
