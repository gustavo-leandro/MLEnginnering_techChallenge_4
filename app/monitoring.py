"""
Monitoramento da API e do modelo em producao.

Responsabilidades:
  - Metricas Prometheus: latencia, contagem de requests, erros, CPU e memoria.
  - Logging estruturado em JSON para cada predicao.
  - Rastreamento de drift do modelo via janela deslizante de MAE/RMSE.
"""

import json
import logging
import os
import time
from collections import deque
from pathlib import Path

import psutil
from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Metricas Prometheus
# ---------------------------------------------------------------------------

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total de requests recebidos",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Latencia dos requests em segundos",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

ERROR_COUNT = Counter(
    "http_request_errors_total",
    "Total de requests com erro (4xx/5xx)",
    ["endpoint"],
)

CPU_USAGE = Gauge("process_cpu_percent", "Uso de CPU do processo (%)")
MEMORY_BYTES = Gauge("process_memory_bytes", "Uso de memoria do processo (bytes)")
LAST_PREDICTION = Gauge("model_last_prediction_brl", "Ultimo preco previsto em BRL")
DRIFT_MAE = Gauge("model_drift_mae", "MAE da janela deslizante de producao")
DRIFT_RMSE = Gauge("model_drift_rmse", "RMSE da janela deslizante de producao")


def update_system_metrics() -> None:
    """Atualiza metricas de CPU e memoria do processo atual."""
    proc = psutil.Process(os.getpid())
    CPU_USAGE.set(proc.cpu_percent(interval=None))
    MEMORY_BYTES.set(proc.memory_info().rss)


# ---------------------------------------------------------------------------
# Logging estruturado JSON
# ---------------------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra"):
            log.update(record.extra)
        return json.dumps(log, ensure_ascii=False)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


prediction_logger = get_logger("prediction")


def log_prediction(ticker: str, look_back: int, predicted: float, latency_ms: float) -> None:
    """Loga uma predicao com seus metadados em formato JSON."""
    prediction_logger.info(
        "prediction",
        extra={
            "extra": {
                "ticker": ticker,
                "look_back": look_back,
                "predicted_close": predicted,
                "latency_ms": round(latency_ms, 2),
            }
        },
    )
    LAST_PREDICTION.set(predicted)


# ---------------------------------------------------------------------------
# Monitoramento de drift
# ---------------------------------------------------------------------------

DRIFT_WINDOW = 30       # tamanho da janela deslizante (pares predicao/real)
DRIFT_MAE_THRESHOLD = 2.0  # BRL — alerta se MAE ultrapassar este valor

PREDICTIONS_LOG = Path(__file__).resolve().parent.parent / "reports" / "predictions_log.jsonl"

_drift_buffer: deque[dict] = deque(maxlen=DRIFT_WINDOW)


def log_prediction_for_drift(predicted: float, actual: float | None = None) -> None:
    """
    Registra um par (predicao, real) para calculo de drift.

    `actual` pode ser None quando o preco real ainda nao esta disponivel.
    O arquivo predictions_log.jsonl persiste os registros em disco.
    """
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "predicted": predicted,
        "actual": actual,
    }
    PREDICTIONS_LOG.parent.mkdir(exist_ok=True)
    with open(PREDICTIONS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

    if actual is not None:
        _drift_buffer.append({"predicted": predicted, "actual": actual})
        _compute_drift()


def _compute_drift() -> None:
    """Calcula MAE e RMSE na janela atual e atualiza metricas + alerta se necessario."""
    pairs = [p for p in _drift_buffer if p["actual"] is not None]
    if not pairs:
        return

    errors = [abs(p["predicted"] - p["actual"]) for p in pairs]
    squared = [(p["predicted"] - p["actual"]) ** 2 for p in pairs]

    mae = sum(errors) / len(errors)
    rmse = (sum(squared) / len(squared)) ** 0.5

    DRIFT_MAE.set(mae)
    DRIFT_RMSE.set(rmse)

    drift_logger = get_logger("drift")
    if mae > DRIFT_MAE_THRESHOLD:
        drift_logger.warning(
            "drift detectado",
            extra={
                "extra": {
                    "window_size": len(pairs),
                    "mae": round(mae, 4),
                    "rmse": round(rmse, 4),
                    "threshold": DRIFT_MAE_THRESHOLD,
                }
            },
        )
    else:
        drift_logger.info(
            "drift ok",
            extra={
                "extra": {
                    "window_size": len(pairs),
                    "mae": round(mae, 4),
                    "rmse": round(rmse, 4),
                }
            },
        )
