"""
Definição da arquitetura LSTM para predição de preço de fechamento.

Arquitetura:
  LSTM(units, return_sequences=True)  →  captura dependências de longo prazo
  Dropout(rate)                        →  regularização
  LSTM(units // 2)                     →  comprime a representação temporal
  Dropout(rate)                        →  regularização
  Dense(1)                             →  saída: preço Close normalizado do próximo dia
"""

from keras import Input, Model
from keras.layers import LSTM, Dense, Dropout


def build_model(
    look_back: int = 60,
    n_features: int = 6,
    units: int = 64,
    dropout_rate: float = 0.2,
    learning_rate: float = 1e-3,
) -> Model:
    """
    Constrói e compila o modelo LSTM.

    Args:
        look_back: número de timesteps de entrada (janela histórica).
        n_features: número de features por timestep (Close, MA7, MA21, Volume, RSI, MACD_hist).
        units: neurônios na primeira camada LSTM; a segunda usa units // 2.
        dropout_rate: fração de neurônios desativados após cada LSTM.
        learning_rate: taxa de aprendizado do otimizador Adam.

    Returns:
        Modelo Keras compilado com perda MSE.
    """
    inputs = Input(shape=(look_back, n_features))
    x = LSTM(units, return_sequences=True)(inputs)
    x = Dropout(dropout_rate)(x)
    x = LSTM(units // 2)(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    model.optimizer.learning_rate.assign(learning_rate)
    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()
