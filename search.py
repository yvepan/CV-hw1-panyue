import os
import csv
import json
import shutil
import itertools
from train import train_model

if __name__ == "__main__":
    DATA_DIR         = "data/fashion"
    BEST_MODEL_PATH  = "best_model.npz"
    BEST_PARAMS_PATH = "best_params.json"
    SEARCH_LOG_PATH  = "search_results.csv"
    TEMP_MODEL_PATH  = "_temp_model.npz"

    # ── 固定参数（不参与搜索）────────────────────────────────────────────────
    FIXED_PARAMS = {
        'activation': 'relu',
        'batch_size': 64,
    }
    SEARCH_EPOCHS     = 15   # 搜索阶段每组训练轮数（减少以加快速度）
    FULL_TRAIN_EPOCHS = 30   # 最优参数完整训练轮数

    # ── 搜索空间：学习率 / 隐藏层大小 / 正则化强度，各设两个候选值 ──────────
    search_space = {
        'lr':           [0.1, 0.05],
        'hidden_dims':  [[128, 64], [256, 128]],
        'weight_decay': [1e-4, 1e-3],
    }

    keys         = list(search_space.keys())
    combinations = list(itertools.product(*[search_space[k] for k in keys]))
    total        = len(combinations)
    print(f"共有 {total} 组超参数组合待搜索（每组训练 {SEARCH_EPOCHS} epochs）...\n")

    best_overall_acc = 0.0
    best_params      = None

    # 初始化 CSV 日志（写表头）
    with open(SEARCH_LOG_PATH, 'w', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=keys + ['val_acc']).writeheader()

    # ── 阶段一：网格搜索 ──────────────────────────────────────────────────────
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        print(f"[{i+1}/{total}] 评估参数: {params}")

        history = train_model(
            data_dir     = DATA_DIR,
            hidden_dims  = params['hidden_dims'],
            activation   = FIXED_PARAMS['activation'],
            lr           = params['lr'],
            weight_decay = params['weight_decay'],
            batch_size   = FIXED_PARAMS['batch_size'],
            epochs       = SEARCH_EPOCHS,
            lr_decay_step  = 10,
            lr_decay_gamma = 0.5,
            save_path    = TEMP_MODEL_PATH,
            plot         = False,          # 搜索阶段不绘图
        )

        best_val_acc = max(history['val_acc'])
        print(f"  验证集最高准确率: {best_val_acc:.4f}")

        # 追加到 CSV
        with open(SEARCH_LOG_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys + ['val_acc'])
            writer.writerow({**{k: params[k] for k in keys}, 'val_acc': f"{best_val_acc:.4f}"})

        # 仅当超过全局最优时，才把临时权重提升为正式权重
        if best_val_acc > best_overall_acc:
            best_overall_acc = best_val_acc
            best_params      = params
            shutil.copy(TEMP_MODEL_PATH, BEST_MODEL_PATH)
            print(f"  ★ 新的全局最优！")

    # 清理临时文件
    if os.path.exists(TEMP_MODEL_PATH):
        os.remove(TEMP_MODEL_PATH)

    print("\n" + "=" * 50)
    print("搜索结束！")
    print(f"最佳验证集准确率（{SEARCH_EPOCHS} epochs）: {best_overall_acc:.4f}")
    print("最佳参数组合:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"\n搜索日志已保存至: {SEARCH_LOG_PATH}")

    # ── 阶段二：用最优参数进行完整训练 ───────────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"使用最优参数进行完整训练（{FULL_TRAIN_EPOCHS} epochs）...")
    print("=" * 50)

    train_model(
        data_dir     = DATA_DIR,
        hidden_dims  = best_params['hidden_dims'],
        activation   = FIXED_PARAMS['activation'],
        lr           = best_params['lr'],
        weight_decay = best_params['weight_decay'],
        batch_size   = FIXED_PARAMS['batch_size'],
        epochs       = FULL_TRAIN_EPOCHS,
        lr_decay_step  = 10,
        lr_decay_gamma = 0.5,
        save_path    = BEST_MODEL_PATH,
        plot         = True,               # 完整训练阶段绘制曲线
    )

    # 保存最优参数（含固定参数）供 evaluate.py 使用
    with open(BEST_PARAMS_PATH, 'w', encoding='utf-8') as f:
        json.dump({**best_params, **FIXED_PARAMS}, f, indent=2, ensure_ascii=False)

    print(f"\n模型权重已保存至: {BEST_MODEL_PATH}")
    print(f"最优参数已保存至: {BEST_PARAMS_PATH}")
