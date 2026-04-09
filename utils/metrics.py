import numpy as np

def calculate_accuracy(y_pred, y_true):
    """
    计算分类准确率
    y_pred: 模型输出的概率分布或对数（logits），形状为 (N, C) 或者直接是预测的类别索引 (N,)
    y_true: 真实的类别标签，形状为 (N,)
    """
    if y_pred.ndim == 2:
        predictions = np.argmax(y_pred, axis=1)
    else:
        predictions = y_pred
    
    correct = np.sum(predictions == y_true)
    return correct / len(y_true)

def compute_confusion_matrix(y_pred, y_true, num_classes=10):
    """
    计算混淆矩阵
    y_pred: 模型预测结果
    y_true: 真实标签
    """
    if y_pred.ndim == 2:
        predictions = np.argmax(y_pred, axis=1)
    else:
        predictions = y_pred
        
    matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, predictions):
        matrix[t, p] += 1
        
    return matrix

def print_confusion_matrix(matrix, class_names=None):
    """
    格式化打印混淆矩阵
    """
    if class_names is None:
        class_names = [f"C{i}" for i in range(matrix.shape[0])]
        
    # 打印表头
    header = f"{'':>10}" + "".join([f"{name:>8}" for name in class_names])
    print(header)
    
    # 打印每一行
    for i, name in enumerate(class_names):
        row_str = f"{name:>10}"
        for j in range(matrix.shape[1]):
            row_str += f"{matrix[i, j]:>8}"
        print(row_str)
