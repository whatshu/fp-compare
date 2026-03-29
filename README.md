# FP Compare: 常见机器学习算法 + Optimizer + 精度对比

这个小项目用纯 Python（无第三方依赖）实现了两个常见算法，并提供了可复现的实验脚本：

- 线性回归（MSE 损失，梯度下降）
- 二分类逻辑回归（交叉熵损失，梯度下降）

并支持以下优化器进行训练性能对比：

- SGD
- Momentum
- RMSProp
- Adam

此外包含一个推理精度实验，对比 `float64` 和更高精度（使用 Decimal 34 位精度，近似 fp128）在病态问题上的误差差异。

## 运行

```bash
python benchmark.py
```

你会看到：

1. 各优化器在回归 + 分类任务上的损失、精度与耗时对比。
2. `float64` vs 高精度 dtype 在高阶多项式拟合任务上的误差对比。

## 测试

```bash
python -m unittest discover -s tests -q
```

## 关于 `fp128`

- 不是所有平台都提供真正 IEEE `float128`。
- 在纯 Python 标准库中，没有原生 IEEE `float128` 类型。
- 本项目使用 `decimal.Decimal` 并将精度设为 34 位，作为“fp128-like”对照组。

## 建议

- 如果你能接受一定速度下降，建议优先在**推理阶段**对数值不稳定模块（例如高阶多项式、病态矩阵求解）使用更高精度。
- 训练阶段优先选择 Adam/RMSProp 做快速收敛，再按需局部切到高精度验证边界样本。
