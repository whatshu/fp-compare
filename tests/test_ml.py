import unittest

from benchmark import make_classification_data, make_regression_data, precision_benchmark, train_with_optimizer
from fp128_x86 import FP128Unavailable
from ml_algorithms import Adam, Momentum, SGD


class TestMLAlgorithms(unittest.TestCase):
    def test_regression_and_classification_learn(self):
        Xr, yr = make_regression_data(n=400, d=8, noise=0.1, seed=1)
        Xc, yc = make_classification_data(n=400, d=8, seed=2)
        out = train_with_optimizer(Xr, yr, Xc, yc, lambda: Adam(lr=0.02))
        self.assertLess(out["reg_rmse"], 0.25)
        self.assertGreater(out["clf_acc"], 0.9)

    def test_optimizer_sanity(self):
        Xr, yr = make_regression_data(n=300, d=6, noise=0.1, seed=3)
        Xc, yc = make_classification_data(n=300, d=6, seed=4)

        out_sgd = train_with_optimizer(Xr, yr, Xc, yc, lambda: SGD(lr=0.02))
        out_m = train_with_optimizer(Xr, yr, Xc, yc, lambda: Momentum(lr=0.02, beta=0.9))

        self.assertLess(out_sgd["reg_rmse"], 0.6)
        self.assertLess(out_m["reg_rmse"], 0.6)

    def test_fp128_precision_not_worse(self):
        try:
            p = precision_benchmark()
        except FP128Unavailable as exc:
            self.skipTest(f"fp128 unavailable: {exc}")
            return
        self.assertLessEqual(p["fp128_err"], p["float64_err"] * 1.05)


if __name__ == "__main__":
    unittest.main()
