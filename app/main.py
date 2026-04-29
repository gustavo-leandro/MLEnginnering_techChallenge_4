"""Entrypoint da API FastAPI para predicao de precos de VALE3.SA."""

import time

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.monitoring import (
    ERROR_COUNT,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    update_system_metrics,
)
from app.routes.predict import router as predict_router

app = FastAPI(
    title="VALE3 LSTM Predictor",
    description="API de predicao do preco de fechamento de VALE3.SA usando modelo LSTM.",
    version="1.0.0",
)

app.include_router(predict_router)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Registra latencia, contagem e erros para cada request."""
    start = time.perf_counter()
    response = await call_next(request)
    latency = time.perf_counter() - start

    endpoint = request.url.path
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=endpoint,
        status_code=response.status_code,
    ).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)

    if response.status_code >= 400:
        ERROR_COUNT.labels(endpoint=endpoint).inc()

    update_system_metrics()
    return response


@app.get("/health")
def health():
    """Verifica se a API esta no ar e o modelo carregado."""
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    """Expoe metricas no formato Prometheus para scraping."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
