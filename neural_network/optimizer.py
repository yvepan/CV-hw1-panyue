import numpy as np

class SGD:
    """Stochastic Gradient Descent 优化器，带 L2 正则化 (Weight Decay)"""
    def __init__(self, model, lr=0.01, weight_decay=0.0):
        """
        model: 模型实例
        lr: 初始学习率
        weight_decay: L2正则化系数
        """
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self):
        """执行一步参数更新"""
        for layer in self.model.layers:
            # 只有有参数的层（如 Linear）需要更新
            if hasattr(layer, 'params'):
                for key in layer.params:
                    # 加上 L2 正则化的梯度: dx += weight_decay * W
                    grad = layer.grads[key]
                    
                    if self.weight_decay > 0.0 and key == 'W':
                        grad += self.weight_decay * layer.params[key]
                        
                    layer.params[key] -= self.lr * grad
                    
    def zero_grad(self):
        """清空梯度"""
        for layer in self.model.layers:
            if hasattr(layer, 'grads'):
                for key in layer.grads:
                    layer.grads[key] = np.zeros_like(layer.grads[key])

class StepLR:
    """简单的学习率按比例衰减调度策略"""
    def __init__(self, optimizer, step_size, gamma=0.1):
        """
        optimizer: 优化器实例
        step_size: 多少个 epoch 衰减一次
        gamma: 衰减系数
        """
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = 0

    def step(self):
        self.last_epoch += 1
        if self.last_epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma
