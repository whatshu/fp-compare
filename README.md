# FP Compare: 常见机器学习算法 + Optimizer + 精度对比

这个小项目用纯 Python 实现了两个常见算法，并提供可复现实验脚本：

- 线性回归（MSE 损失，梯度下降）
- 二分类逻辑回归（交叉熵损失，梯度下降）

并支持以下优化器进行训练性能对比：

- SGD
- Momentum
- RMSProp
- Adam

此外包含一个推理精度实验：

- `float64`（C 中 `double`）
- `fp128`（x86 平台 C 中 `__float128`，通过 `libquadmath`）

> 为降低 Decimal 的性能损失与语义偏差，fp128 路径已改为 C 实现。

## 运行

```bash
python benchmark.py
```

你会看到：

1. 各优化器在回归 + 分类任务上的损失、精度与耗时对比。
2. `float64` vs `__float128` 在同一推理任务上的误差对比。

## 测试

```bash
python -m unittest discover -s tests -q
```

## fp128 实现细节（x86）

- C 源码：`c_fp128_x86.c`
  - `mean_abs_error_exp_taylor_f64`：`double` 版本误差。
  - `mean_abs_error_exp_taylor_f128`：`__float128` 版本误差。
- Python 封装：`fp128_x86.py`
  - 自动调用 `gcc -shared -lquadmath` 编译成 `_fp128_x86.so`。
  - 用 `ctypes` 调用 C 函数。
- 当前仅支持 x86/x86_64（不在该架构会报告 unavailable）。
