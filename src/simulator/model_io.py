"""Laden der Modelle und Kalibrierer; gemeinsame Vorhersagefunktion fuer Evaluation und Simulator."""

from __future__ import annotations

import pickle

import lightgbm as lgb
import numpy as np

from .features import FEATURES, N_CLASS
from .train import MODELS


def softmax_temp(P: np.ndarray, T: float) -> np.ndarray:
    """Temperaturskalierung auf Wahrscheinlichkeiten: p^(1/T), renormiert."""
    if T == 1.0:
        return P
    L = np.log(np.clip(P, 1e-12, 1.0)) / T
    L -= L.max(axis=1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(axis=1, keepdims=True)


class Models:
    def __init__(self, calibrated: bool = False, num_threads: int = 10):
        self.cancel = lgb.Booster(model_file=str(MODELS / "lgbm_cancel.txt"))
        self.klass = lgb.Booster(model_file=str(MODELS / "lgbm_class.txt"))
        self.nt = num_threads
        self.cal = {"cancel": None, "temp": 1.0}
        p = MODELS / "calib.pkl"
        if calibrated and p.exists():
            self.cal = pickle.load(open(p, "rb"))
        r = np.load(MODELS / "rand_verteilung.npz")
        self.rand_lo, self.rand_hi = r["lo"], r["hi"]

    def p_cancel_raw(self, X: np.ndarray) -> np.ndarray:
        return self.cancel.predict(X, num_threads=self.nt)

    def p_class_raw(self, X: np.ndarray) -> np.ndarray:
        return self.klass.predict(X, num_threads=self.nt).reshape(len(X), N_CLASS)

    def p_cancel(self, X: np.ndarray) -> np.ndarray:
        p = self.p_cancel_raw(X)
        c = self.cal.get("cancel")
        return p if c is None else apply_cancel_cal(c, p)

    def p_class(self, X: np.ndarray) -> np.ndarray:
        return softmax_temp(self.p_class_raw(X), self.cal.get("temp", 1.0))


def apply_cancel_cal(c: dict, p: np.ndarray) -> np.ndarray:
    z = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
    return 1.0 / (1.0 + np.exp(-(c["a"] * z + c["b"])))
