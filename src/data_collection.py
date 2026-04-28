"""Baixa dados históricos de VALE3.SA via yfinance e salva em data/raw/."""

from pathlib import Path

import pandas as pd
import yfinance as yf

TICKER = "VALE3.SA"
START_DATE = "2018-01-01"
END_DATE = "2024-07-20"
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data/raw"


def download_data() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = yf.download(TICKER, start=START_DATE, end=END_DATE, auto_adjust=True)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.to_csv(RAW_DIR / "vale3_raw.csv")
    print(f"Dados baixados: {df.shape[0]} linhas, {df.shape[1]} colunas")
    print(f"Período: {df.index.min().date()} → {df.index.max().date()}")
    print(f"Arquivo salvo em: {RAW_DIR / 'vale3_raw.csv'}")
    print("\n--- Dtypes ---")
    print(df.dtypes)
    print("\n--- Nulos ---")
    print(df.isnull().sum())
    print("\n--- Primeiras linhas ---")
    print(df.head())
    return df


if __name__ == "__main__":
    download_data()
