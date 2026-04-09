import numpy as np

class CrossEntropyLoss:
    """交叉熵损失，内部结合了 Softmax 操作以保证数值稳定性"""
    def __init__(self):
        self.cache = None

    def forward(self, logits, y_true):
        """
        logits: (N, C) 形状的模型原始输出
        y_true: (N,) 形状的真实标签 (类别索引)
        """
        N = logits.shape[0]
        
        # 为了数值稳定减去最大值
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted_logits)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # 缓存用于反向传播
        self.cache = (probs, y_true)
        
        # 提取真实标签概率以求负对数似然
        corect_logprobs = -np.log(probs[range(N), y_true] + 1e-8)
        data_loss = np.sum(corect_logprobs) / N
        return data_loss

    def backward(self):
        """
        返回关于 logits 的梯度 (N, C)
        """
        probs, y_true = self.cache
        N = probs.shape[0]
        
        # softmax - 交叉熵 的梯度为: probs - 1 (在真实标签位置)
        dx = probs.copy()
        dx[range(N), y_true] -= 1
        dx /= N
        return dx
