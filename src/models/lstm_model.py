import numpy as np
from tensorflow import keras
from src.config import Config
from src.models.base import BaseModel


class LSTMModel(BaseModel):
    def __init__(self, config: Config):
        self.config = config
        self.model: keras.Model = None

    def build(self, input_shape: tuple) -> None:
        cfg = self.config.model.lstm
        inputs = keras.Input(shape=input_shape)
        x = inputs
        for i, units in enumerate(cfg.units):
            return_sequences = i < len(cfg.units) - 1
            x = keras.layers.LSTM(units, return_sequences=return_sequences)(x)
            x = keras.layers.Dropout(cfg.dropout)(x)
        outputs = keras.layers.Dense(1, activation="sigmoid")(x)
        self.model = keras.Model(inputs, outputs)
        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

    def fit(self, X_train, y_train, X_val, y_val) -> dict:
        cfg = self.config.model
        es = keras.callbacks.EarlyStopping(
            monitor=cfg.early_stopping.monitor,
            patience=cfg.early_stopping.patience,
            restore_best_weights=True,
        )
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            callbacks=[es],
            verbose=0,
        )
        return history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.model.predict(X, verbose=0) > 0.5).astype(int).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X, verbose=0).flatten()
