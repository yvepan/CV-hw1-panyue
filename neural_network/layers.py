import numpy as np

class Layer:
    """基础层类，所有层都应继承此层。"""
    def __init__(self):
        self.params = {}      # 可训练参数
        self.grads = {}       # 梯度
        self.cache = None     # 缓存前向传播中的数据用于反向传播

    def forward(self, x):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError

class Linear(Layer):
    """全连接层 (Dense Layer)"""
    def __init__(self, in_features, out_features):
        super().__init__()
        # 使用 Kaiming 初始化 (He initialization) 以利于 ReLU，但也适用于其他
        limit = np.sqrt(2.0 / in_features)
        self.params['W'] = np.random.randn(in_features, out_features) * limit
        self.params['b'] = np.zeros(out_features)

    def forward(self, x):
        """
        x 形状: (N, in_features)
        返回: (N, out_features)
        """
        self.cache = x
        return np.dot(x, self.params['W']) + self.params['b']

    def backward(self, dout):
        """
        dout (Loss 对当前层输出的梯度): 形状 (N, out_features)
        返回: 对输入的梯度 dx 形状 (N, in_features)
        """
        x = self.cache
        
        # 计算梯度
        self.grads['W'] = np.dot(x.T, dout)
        self.grads['b'] = np.sum(dout, axis=0)
        
        # 计算向前传递的输入梯度
        dx = np.dot(dout, self.params['W'].T)
        return dx

class ReLU(Layer):
    """ReLU 激活函数"""
    def forward(self, x):
        self.cache = x
        return np.maximum(0, x)

    def backward(self, dout):
        x = self.cache
        dx = dout.copy()
        dx[x <= 0] = 0
        return dx

class Sigmoid(Layer):
    """Sigmoid 激活函数"""
    def forward(self, x):
        # 截断以防止溢出
        x_clipped = np.clip(x, -500, 500)
        out = 1.0 / (1.0 + np.exp(-x_clipped))
        self.cache = out
        return out

    def backward(self, dout):
        out = self.cache
        dx = dout * out * (1.0 - out)
        return dx

class Tanh(Layer):
    """Tanh 激活函数"""
    def forward(self, x):
        out = np.tanh(x)
        self.cache = out
        return out

    def backward(self, dout):
        out = self.cache
        dx = dout * (1.0 - out ** 2)
        return dx
