"""Schemas Pydantic para request e response da API."""

from pydantic import BaseModel, Field, model_validator


class PredictRequest(BaseModel):
    prices: list[float] = Field(
        ...,
        description="Serie historica de precos de fechamento em BRL, ordem cronologica.",
        min_length=90,
        examples=[[61.5, 62.0, 61.8]],
    )

    @model_validator(mode="after")
    def check_positive(self):
        if any(p <= 0 for p in self.prices):
            raise ValueError("Todos os precos devem ser positivos.")
        return self


class PredictResponse(BaseModel):
    ticker: str
    predicted_close: float = Field(..., description="Preco de fechamento previsto para o proximo dia (BRL).")
    look_back_used: int = Field(..., description="Quantidade de dias usados como janela de entrada.")
