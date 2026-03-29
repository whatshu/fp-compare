"""Benchmark optimizers and precision choices for simple ML tasks."""

from __future__ import annotations

import random
import time
from typing import Callable, Dict, List, Sequence, Tuple

from fp128_x86 import FP128Unavailable, compare_taylor_exp_errors
from ml_algorithms import Adam, LinearRegressionGD, LogisticRegressionGD, Momentum, RMSProp, SGD, accuracy, rmse

Matrix = List[List[float]]


def make_regression_data(n: int = 1000, d: int = 12, noise: float = 0.2, seed: int = 42) -> Tuple[Matrix, List[float]]:
    rng = random.Random(seed)
    w = [rng.uniform(-2, 2) for _ in range(d)]
    X: Matrix = []
    y: List[float] = []
    for _ in range(n):
        row = [rng.gauss(0, 1) for _ in range(d)]
        target = sum(a * b for a, b in zip(row, w)) + rng.gauss(0, noise)
        X.append(row)
        y.append(target)
    return X, y


def make_classification_data(n: int = 1000, d: int = 12, seed: int = 7) -> Tuple[Matrix, List[int]]:
    rng = random.Random(seed)
    w = [rng.uniform(-2, 2) for _ in range(d)]
    X: Matrix = []
    y: List[int] = []
    for _ in range(n):
        row = [rng.gauss(0, 1) for _ in range(d)]
        score = sum(a * b for a, b in zip(row, w)) + rng.gauss(0, 0.2)
        X.append(row)
        y.append(1 if score > 0 else 0)
    return X, y


def train_with_optimizer(
    Xr: Matrix,
    yr: Sequence[float],
    Xc: Matrix,
    yc: Sequence[int],
    optimizer_factory: Callable[[], object],
) -> Dict[str, float]:
    reg = LinearRegressionGD(optimizer=optimizer_factory(), epochs=900)
    clf = LogisticRegressionGD(optimizer=optimizer_factory(), epochs=900)

    t0 = time.perf_counter()
    reg_hist = reg.fit(Xr, yr)
    t1 = time.perf_counter()
    clf_hist = clf.fit(Xc, yc)
    t2 = time.perf_counter()

    reg_pred = reg.predict(Xr)
    clf_pred = clf.predict(Xc)

    return {
        "reg_rmse": rmse(yr, reg_pred),
        "reg_loss": reg_hist.losses[-1],
        "clf_acc": accuracy(yc, clf_pred),
        "clf_loss": clf_hist.losses[-1],
        "time_sec": (t2 - t0),
        "reg_train_sec": (t1 - t0),
        "clf_train_sec": (t2 - t1),
    }


def precision_benchmark() -> Dict[str, float]:
    t0 = time.perf_counter()
    e64, e128 = compare_taylor_exp_errors(degree=25, points=80)
    t1 = time.perf_counter()
    return {
        "float64_err": e64,
        "fp128_err": e128,
        "total_sec": t1 - t0,
    }


def main() -> None:
    Xr, yr = make_regression_data()
    Xc, yc = make_classification_data()

    optimizers = {
        "SGD": lambda: SGD(lr=0.03),
        "Momentum": lambda: Momentum(lr=0.03, beta=0.9),
        "RMSProp": lambda: RMSProp(lr=0.01, beta=0.9),
        "Adam": lambda: Adam(lr=0.02),
    }

    print("=== Optimizer benchmark (training) ===")
    for name, factory in optimizers.items():
        out = train_with_optimizer(Xr, yr, Xc, yc, factory)
        print(
            f"{name:8s} | reg_rmse={out['reg_rmse']:.4f} | clf_acc={out['clf_acc']:.4f} | "
            f"reg_loss={out['reg_loss']:.3e} | clf_loss={out['clf_loss']:.3e} | total={out['time_sec']:.3f}s"
        )

    print("\n=== Inference precision benchmark (x86 __float128 via C) ===")
    try:
        p = precision_benchmark()
        print(f"float64  error={p['float64_err']:.3e}")
        print(f"fp128    error={p['fp128_err']:.3e}")
        print(f"speed(total)={p['total_sec']:.4f}s")
        print(
            "precision gain="
            f"{(p['float64_err'] / p['fp128_err'] if p['fp128_err'] > 0 else float('inf')):.2f}x"
        )
    except FP128Unavailable as exc:
        print(f"fp128 benchmark unavailable: {exc}")


if __name__ == "__main__":
    main()
