import numpy as np
from neural_network.layers import Linear, ReLU, Sigmoid, Tanh

class MLP:
    """三层多层感知机分类器"""
    def __init__(self, input_dim=784, hidden_dims=[128, 64], output_dim=10, activation='relu'):
        """
        input_dim: 输入特征维度 (Fashion-MNIST 为 28x28=784)
        hidden_dims: 包含两个元素的列表，表示两个隐藏层维度 [h1, h2]
        output_dim: 输出类别数 (默认为 10)
        activation: 激活函数, 可选 'relu', 'sigmoid', 'tanh'
        """
        self.layers = []
        
        # 第一层 (Linear + 激活)
        self.layers.append(Linear(input_dim, hidden_dims[0]))
        self.layers.append(self._get_activation(activation))
        
        # 第二层 (Linear + 激活)
        self.layers.append(Linear(hidden_dims[0], hidden_dims[1]))
        self.layers.append(self._get_activation(activation))
        
        # 第三层 (输出层，不加激活，后面交给带 Softmax 的交叉熵损失)
        self.layers.append(Linear(hidden_dims[1], output_dim))
        
    def _get_activation(self, name):
        name = name.lower()
        if name == 'relu':
            return ReLU()
        elif name == 'sigmoid':
            return Sigmoid()
        elif name == 'tanh':
            return Tanh()
        else:
            raise ValueError(f"不支持的激活函数: {name}")

    def forward(self, x):
        """前向传播"""
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dout):
        """反向传播"""
        # 逆序遍历层
        dout_temp = dout
        for layer in reversed(self.layers):
            dout_temp = layer.backward(dout_temp)
            
    def get_l2_loss(self, weight_decay):
        """计算模型的 L2 正则化代价值（仅用于打印/记录 loss）"""
        l2_loss = 0.0
        if weight_decay > 0.0:
            for layer in self.layers:
                if hasattr(layer, 'params') and 'W' in layer.params:
                    l2_loss += 0.5 * weight_decay * np.sum(layer.params['W'] ** 2)
        return l2_loss

    def save_weights(self, filepath):
        """保存模型参数"""
        weights = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'params') and 'W' in layer.params:
                weights[f"layer_{i}_W"] = layer.params['W']
                weights[f"layer_{i}_b"] = layer.params['b']
        np.savez(filepath, **weights)

    def load_weights(self, filepath):
        """加载模型参数"""
        weights = np.load(filepath)
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'params') and 'W' in layer.params:
                layer.params['W'] = weights[f"layer_{i}_W"]
                layer.params['b'] = weights[f"layer_{i}_b"]
