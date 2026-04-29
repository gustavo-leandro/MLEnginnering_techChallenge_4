"""Endpoint POST /predict."""

import time

from fastapi import APIRouter

from app.monitoring import log_prediction, log_prediction_for_drift
from app.schemas import PredictRequest, PredictResponse
from app.services.predictor import predictor

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """
    Recebe uma serie historica de precos de fechamento e retorna
    a previsao do preco de fechamento do proximo dia util.

    O payload deve conter pelo menos 90 precos (look_back do modelo).
    Apenas os ultimos 90 valores serao usados como janela de entrada.
    """
    start = time.perf_counter()
    predicted = predictor.predict(request.prices)
    latency_ms = (time.perf_counter() - start) * 1000

    log_prediction(
        ticker=predictor.ticker,
        look_back=predictor.look_back,
        predicted=predicted,
        latency_ms=latency_ms,
    )

    # registra predicao para calculo de drift (actual=None ate o preco real estar disponivel)
    log_prediction_for_drift(predicted=predicted, actual=None)

    return PredictResponse(
        ticker=predictor.ticker,
        predicted_close=predicted,
        look_back_used=predictor.look_back,
    )
