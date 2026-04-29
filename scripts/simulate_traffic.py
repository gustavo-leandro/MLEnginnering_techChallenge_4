"""
Envia requests ao endpoint /predict a cada 1 segundo para popular o Grafana.
Uso: python scripts/simulate_traffic.py
"""

import time
import random
import requests

API_URL = "http://localhost:8000/predict"
INTERVAL = 1.0
LOOK_BACK = 90

BASE_PRICE = 68.0


def generate_prices(n: int) -> list[float]:
    prices = []
    price = BASE_PRICE
    for _ in range(n):
        price += random.gauss(0, 0.5)
        price = max(price, 10.0)
        prices.append(round(price, 2))
    return prices


def main() -> None:
    print(f"Enviando requests para {API_URL} a cada {INTERVAL}s. Ctrl+C para parar.\n")
    count = 0
    errors = 0

    while True:
        try:
            payload = {"prices": generate_prices(LOOK_BACK)}
            resp = requests.post(API_URL, json=payload, timeout=5)

            count += 1
            if resp.status_code == 200:
                data = resp.json()
                print(
                    f"[{count:04d}] previsto: R$ {data['predicted_close']:.2f} | "
                    f"status: {resp.status_code}"
                )
            else:
                errors += 1
                print(f"[{count:04d}] erro {resp.status_code}: {resp.text[:80]}")

        except requests.exceptions.ConnectionError:
            errors += 1
            print(f"[{count:04d}] API nao disponivel em {API_URL}")

        except KeyboardInterrupt:
            print(f"\nEncerrado. Total: {count} requests, {errors} erros.")
            break

        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
