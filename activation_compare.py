"""
激活函数对照实验：在最优超参数下分别用 relu / sigmoid / tanh 训练，
记录每个 epoch 的验证集准确率，最终输出对比曲线图。
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from train import train_model

DATA_DIR = "data/fashion"
ACTIVATIONS = ['relu', 'sigmoid', 'tanh']

# 使用网格搜索得到的最优超参数（激活函数除外）
BASE_PARAMS = dict(
    data_dir       = DATA_DIR,
    hidden_dims    = [256, 128],
    lr             = 0.1,
    weight_decay   = 1e-4,
    batch_size     = 64,
    epochs         = 30,
    lr_decay_step  = 10,
    lr_decay_gamma = 0.5,
    plot           = False,
)

results = {}
for act in ACTIVATIONS:
    print(f"\n{'='*40}")
    print(f"训练激活函数: {act}")
    print('='*40)
    history = train_model(
        **BASE_PARAMS,
        activation = act,
        save_path  = f"_act_{act}.npz",
    )
    results[act] = history
    best_val = max(history['val_acc'])
    print(f"  验证集最高准确率: {best_val:.4f}")

# 保存数值结果
summary = {act: max(v['val_acc']) for act, v in results.items()}
print("\n\n激活函数对照结果:")
for act, acc in summary.items():
    print(f"  {act:8s}: {acc:.4f}")

with open("activation_compare_results.json", "w") as f:
    json.dump({act: {"val_acc": v['val_acc'], "train_loss": v['train_loss']}
               for act, v in results.items()}, f)

# 绘图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
colors = {'relu': '#1f77b4', 'sigmoid': '#ff7f0e', 'tanh': '#2ca02c'}

for act in ACTIVATIONS:
    h = results[act]
    epochs = range(1, len(h['val_acc']) + 1)
    ax1.plot(epochs, h['train_loss'], color=colors[act], label=act)
    ax2.plot(epochs, h['val_acc'],   color=colors[act], label=act)

ax1.set_title('Train Loss — Activation Comparison')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.set_title('Val Accuracy — Activation Comparison')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("activation_comparison.png", dpi=150)
print("\n对比图已保存至 activation_comparison.png")
