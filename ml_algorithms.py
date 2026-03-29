"""Common machine-learning algorithms implemented in pure Python.

No third-party dependencies are required, making it easy to run in restricted
execution environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, getcontext
import math
from typing import Callable, Dict, List, Sequence, Tuple

Vector = List[float]
Matrix = List[Vector]


# ---------- basic linear algebra helpers ----------
def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def add_bias(X: Matrix) -> Matrix:
    return [[1.0] + row[:] for row in X]


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


# ---------- optimizer definitions ----------
class Optimizer:
    def init_state(self, size: int) -> Dict[str, Vector | int]:
        return {}

    def step(self, grad: Vector, state: Dict[str, Vector | int]) -> Tuple[Vector, Dict[str, Vector | int]]:
        raise NotImplementedError


@dataclass
class SGD(Optimizer):
    lr: float = 0.01

    def step(self, grad: Vector, state: Dict[str, Vector | int]) -> Tuple[Vector, Dict[str, Vector | int]]:
        return [-self.lr * g for g in grad], state


@dataclass
class Momentum(Optimizer):
    lr: float = 0.01
    beta: float = 0.9

    def init_state(self, size: int) -> Dict[str, Vector | int]:
        return {"v": [0.0] * size}

    def step(self, grad: Vector, state: Dict[str, Vector | int]) -> Tuple[Vector, Dict[str, Vector | int]]:
        v = state["v"]  # type: ignore[index]
        new_v = [self.beta * vi + (1.0 - self.beta) * gi for vi, gi in zip(v, grad)]
        state["v"] = new_v
        return [-self.lr * vi for vi in new_v], state


@dataclass
class RMSProp(Optimizer):
    lr: float = 0.01
    beta: float = 0.9
    eps: float = 1e-8

    def init_state(self, size: int) -> Dict[str, Vector | int]:
        return {"s": [0.0] * size}

    def step(self, grad: Vector, state: Dict[str, Vector | int]) -> Tuple[Vector, Dict[str, Vector | int]]:
        s = state["s"]  # type: ignore[index]
        new_s = [self.beta * si + (1.0 - self.beta) * (gi * gi) for si, gi in zip(s, grad)]
        state["s"] = new_s
        delta = [-self.lr * gi / (math.sqrt(si) + self.eps) for gi, si in zip(grad, new_s)]
        return delta, state


@dataclass
class Adam(Optimizer):
    lr: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def init_state(self, size: int) -> Dict[str, Vector | int]:
        return {"m": [0.0] * size, "v": [0.0] * size, "t": 0}

    def step(self, grad: Vector, state: Dict[str, Vector | int]) -> Tuple[Vector, Dict[str, Vector | int]]:
        t = int(state["t"]) + 1
        m = state["m"]  # type: ignore[index]
        v = state["v"]  # type: ignore[index]

        new_m = [self.beta1 * mi + (1.0 - self.beta1) * gi for mi, gi in zip(m, grad)]
        new_v = [self.beta2 * vi + (1.0 - self.beta2) * (gi * gi) for vi, gi in zip(v, grad)]

        m_hat = [mi / (1.0 - self.beta1**t) for mi in new_m]
        v_hat = [vi / (1.0 - self.beta2**t) for vi in new_v]

        delta = [-self.lr * mi / (math.sqrt(vi) + self.eps) for mi, vi in zip(m_hat, v_hat)]

        state["m"] = new_m
        state["v"] = new_v
        state["t"] = t
        return delta, state


@dataclass
class TrainHistory:
    losses: List[float]


class LinearRegressionGD:
    def __init__(self, optimizer: Optimizer, epochs: int = 1000):
        self.optimizer = optimizer
        self.epochs = epochs
        self.w: Vector | None = None

    def fit(self, X: Matrix, y: Sequence[float]) -> TrainHistory:
        Xb = add_bias(X)
        n = len(Xb)
        d = len(Xb[0])
        self.w = [0.0] * d
        state = self.optimizer.init_state(d)
        losses: List[float] = []

        for _ in range(self.epochs):
            pred = [dot(row, self.w) for row in Xb]
            residual = [p - yi for p, yi in zip(pred, y)]
            grad = [0.0] * d
            for j in range(d):
                grad[j] = 2.0 * sum(row[j] * r for row, r in zip(Xb, residual)) / n

            delta, state = self.optimizer.step(grad, state)
            self.w = [wi + di for wi, di in zip(self.w, delta)]
            losses.append(mean([r * r for r in residual]))

        return TrainHistory(losses=losses)

    def predict(self, X: Matrix) -> Vector:
        if self.w is None:
            raise ValueError("Model is not fitted.")
        Xb = add_bias(X)
        return [dot(row, self.w) for row in Xb]


class LogisticRegressionGD:
    def __init__(self, optimizer: Optimizer, epochs: int = 1000):
        self.optimizer = optimizer
        self.epochs = epochs
        self.w: Vector | None = None

    @staticmethod
    def _sigmoid(z: float) -> float:
        # stable implementation
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def fit(self, X: Matrix, y: Sequence[int]) -> TrainHistory:
        Xb = add_bias(X)
        n = len(Xb)
        d = len(Xb[0])
        self.w = [0.0] * d
        state = self.optimizer.init_state(d)
        losses: List[float] = []

        for _ in range(self.epochs):
            logits = [dot(row, self.w) for row in Xb]
            probs = [self._sigmoid(z) for z in logits]

            eps = 1e-12
            loss_terms = [-(yi * math.log(pi + eps) + (1 - yi) * math.log(1 - pi + eps)) for pi, yi in zip(probs, y)]
            losses.append(mean(loss_terms))

            grad = [0.0] * d
            for j in range(d):
                grad[j] = sum(row[j] * (pi - yi) for row, pi, yi in zip(Xb, probs, y)) / n

            delta, state = self.optimizer.step(grad, state)
            self.w = [wi + di for wi, di in zip(self.w, delta)]

        return TrainHistory(losses=losses)

    def predict_proba(self, X: Matrix) -> Vector:
        if self.w is None:
            raise ValueError("Model is not fitted.")
        Xb = add_bias(X)
        return [self._sigmoid(dot(row, self.w)) for row in Xb]

    def predict(self, X: Matrix, threshold: float = 0.5) -> List[int]:
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return mean([1.0 if a == b else 0.0 for a, b in zip(y_true, y_pred)])


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return math.sqrt(mean([(a - b) ** 2 for a, b in zip(y_true, y_pred)]))


def precision_polyfit_error_float64() -> float:
    """Approximate exp(x) by truncated Taylor and measure error in float64."""
    xs = [-1.0 + 2.0 * i / 80 for i in range(81)]

    def exp_taylor_f64(x: float, degree: int = 25) -> float:
        acc = 1.0
        term = 1.0
        for k in range(1, degree + 1):
            term *= x / k
            acc += term
        return acc

    errs = [abs(exp_taylor_f64(x) - math.exp(x)) for x in xs]
    return mean(errs)


def precision_polyfit_error_decimal(prec: int = 34) -> float:
    """Same computation in Decimal(precision=34), fp128-like precision."""
    old_prec = getcontext().prec
    getcontext().prec = prec
    try:
        xs = [Decimal(-1) + Decimal(2) * Decimal(i) / Decimal(80) for i in range(81)]

        def exp_taylor_dec(x: Decimal, degree: int = 25) -> Decimal:
            acc = Decimal(1)
            term = Decimal(1)
            for k in range(1, degree + 1):
                term *= x / Decimal(k)
                acc += term
            return acc

        errs = [abs(exp_taylor_dec(x) - x.exp()) for x in xs]
        return float(sum(errs) / Decimal(len(errs)))
    finally:
        getcontext().prec = old_prec
