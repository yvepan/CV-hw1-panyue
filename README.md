# Fashion-MNIST 三层神经网络分类器（NumPy 实现）

纯 NumPy 手工实现三层 MLP，Fashion-MNIST 测试集准确率 **89.28%**。

- **GitHub**：[填写链接]
- **模型权重（Google Drive）**：[填写链接]

---

## 环境依赖

```bash
pip install numpy matplotlib
```

---

## 运行方式

### 推荐：网格搜索 + 完整训练

```bash
python search.py
```

自动完成两阶段：① 8 组超参数组合各训练 15 epochs，结果写入 `search_results.csv`；② 用最优参数重新训练 30 epochs，保存 `best_model.npz` 与 `training_curves.png`。

### 直接训练（固定超参数）

```bash
python train.py
```

在 `train.py` 底部修改 `train_model(...)` 的参数即可。

### 测试评估

```bash
python evaluate.py
```

需先存在 `best_model.npz`。自动读取 `best_params.json` 确定模型结构，输出测试集准确率、混淆矩阵，并生成以下图像：

| 文件 | 内容 |
|------|------|
| `confusion_matrix.png` | 混淆矩阵热力图 |
| `weights_visualization.png` | 第一隐藏层全部神经元权重（28×28） |
| `error_cases.png` | 前 10 个分类错误样本 |

### 激活函数对照实验

```bash
python activation_compare.py
```

固定最优超参数，分别以 ReLU / Sigmoid / Tanh 训练 30 epochs，生成 `activation_comparison.png`（Loss 与 Val Accuracy 对比曲线）及 `activation_compare_results.json`（各 epoch 数值记录）。可在脚本顶部修改 `BASE_PARAMS` 自定义对照条件。

---

## 目录结构

```
hw1/
├── data/fashion/          # Fashion-MNIST 原始数据（gzip 格式）
├── neural_network/
│   ├── layers.py          # Linear, ReLU, Sigmoid, Tanh
│   ├── losses.py          # CrossEntropyLoss
│   ├── model.py           # MLP
│   └── optimizer.py       # SGD + StepLR
├── utils/
│   ├── data_loader.py
│   └── metrics.py
├── train.py
├── search.py
├── evaluate.py
└── activation_compare.py
```

---

## 自定义超参数搜索

修改 `search.py` 中的搜索空间：

```python
search_space = {
    'lr':           [0.1, 0.05],
    'hidden_dims':  [[128, 64], [256, 128]],
    'weight_decay': [1e-4, 1e-3],
}
SEARCH_EPOCHS     = 15
FULL_TRAIN_EPOCHS = 30
```

---

## 实验结果

测试集准确率 **89.28%**，最优超参数：`lr=0.1, hidden_dims=[256,128], weight_decay=1e-4`。

各类别准确率：Bag 97.4%、Trouser 97.5%、Sandal 96.8%、Ankle_boot 95.6%、Sneaker 95.4%、Dress 89.5%、T-shirt 88.2%、Coat 84.4%、Pullover 83.6%、Shirt 64.4%。
