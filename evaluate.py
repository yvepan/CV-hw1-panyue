import os
import json
import numpy as np
import matplotlib.pyplot as plt
from utils.data_loader import get_data_loaders, DataLoader
from utils.metrics import calculate_accuracy, compute_confusion_matrix, print_confusion_matrix
from neural_network.model import MLP

def plot_weights(model, save_path="weights_visualization.png"):
    """
    可视化第一层权重的空间模式
    将全部隐藏神经元对应的 28x28 权重展示出来
    """
    # 获取第一层的权重 W (784, hidden_dim)
    w = model.layers[0].params['W']
    num_neurons = w.shape[1]

    # 自动计算行列数，尽量接近正方形
    ncols = int(np.ceil(np.sqrt(num_neurons)))
    nrows = int(np.ceil(num_neurons / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.2, nrows * 1.2))
    fig.suptitle(f"First hidden layer weights — all {num_neurons} neurons (28x28)")

    for i, ax in enumerate(axes.flat):
        if i < num_neurons:
            img = w[:, i].reshape(28, 28)
            ax.imshow(img, cmap='coolwarm')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"权重可视化图像已保存至 {save_path}")

def plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png"):
    """
    将混淆矩阵保存为热力图图像
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    # 在每个格子里填写数值
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center', fontsize=8,
                    color='white' if cm[i, j] > thresh else 'black')

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix on Test Set')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"混淆矩阵图像已保存至 {save_path}")


def plot_error_cases(model, X_test, y_test, y_pred, save_path="error_cases.png"):
    """
    错例分析：挑出被错误分类的图像保存成对比图
    """
    # 找到分错的索引
    error_indices = np.where(y_pred != y_test)[0]
    
    if len(error_indices) == 0:
        print("没有分类错误的样本！")
        return
        
    # 我们挑选前 10 个错例展示
    num_plots = min(10, len(error_indices))
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Error Analysis (True Label VS Predicted Label)")
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
                   
    for i, ax in enumerate(axes.flat):
        if i < num_plots:
            idx = error_indices[i]
            img = X_test[idx].reshape(28, 28)
            
            true_label = class_names[y_test[idx]]
            pred_label = class_names[y_pred[idx]]
            
            ax.imshow(img, cmap='gray')
            ax.set_title(f"True: {true_label}\nPred: {pred_label}", color='red')
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"错例分析图像已保存至 {save_path}")


if __name__ == "__main__":
    DATA_DIR         = "data/fashion"
    MODEL_PATH       = "best_model.npz"
    BEST_PARAMS_PATH = "best_params.json"

    print("加载测试数据...")
    _, _, (X_test, y_test) = get_data_loaders(data_dir=DATA_DIR)

    # 类别名称
    class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle_boot']

    # 从 best_params.json 读取模型结构，若不存在则使用默认值
    if os.path.exists(BEST_PARAMS_PATH):
        with open(BEST_PARAMS_PATH, 'r', encoding='utf-8') as f:
            best_params = json.load(f)
        hidden_dims = best_params['hidden_dims']
        activation  = best_params['activation']
        print(f"从 {BEST_PARAMS_PATH} 读取模型结构: hidden_dims={hidden_dims}, activation={activation}")
    else:
        hidden_dims = [128, 64]
        activation  = 'relu'
        print(f"未找到 {BEST_PARAMS_PATH}，使用默认结构: hidden_dims={hidden_dims}, activation={activation}")

    # 初始化模型并加载权重
    print("初始化模型并加载权重...")
    model = MLP(input_dim=784, hidden_dims=hidden_dims, output_dim=10, activation=activation)

    if os.path.exists(MODEL_PATH):
        model.load_weights(MODEL_PATH)
        print("权重加载成功！")
    else:
        print(f"找不到权重文件 {MODEL_PATH}，请先运行 search.py 或 train.py。")
        exit(1)
        
    print("\n在测试集上进行推理评估...")
    test_loader = DataLoader(X_test, y_test, batch_size=128, shuffle=False)
    
    all_preds_list = []
    
    for inputs, targets in test_loader:
        logits = model.forward(inputs)
        batch_preds = np.argmax(logits, axis=1)
        all_preds_list.append(batch_preds)
        
    # 合并所有预测结果
    all_preds = np.concatenate(all_preds_list)
    
    # 计算准确率
    test_acc = calculate_accuracy(all_preds, y_test)
    print(f"\n======================================")
    print(f"独立测试集分类准确率 (Accuracy): {test_acc * 100:.2f}%")
    print(f"======================================\n")
    
    # 打印并保存混淆矩阵
    print("测试集混淆矩阵 (Confusion Matrix):")
    cm = compute_confusion_matrix(all_preds, y_test, num_classes=10)
    print_confusion_matrix(cm, class_names=class_names)

    print("\n======================================")
    print("正在生成可视化图像和错例分析...")
    # 此时需要确保 matplotlib 已安装：pip install matplotlib
    try:
        plot_weights(model, "weights_visualization.png")
        plot_error_cases(model, X_test, y_test, all_preds, "error_cases.png")
        plot_confusion_matrix(cm, class_names, "confusion_matrix.png")
    except ImportError:
        print("缺少 matplotlib，请安装: pip install matplotlib")
