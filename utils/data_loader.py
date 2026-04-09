import os
import gzip
import numpy as np

def load_mnist_images(filename):
    """读取 MNIST 格式的图像数据。"""
    with gzip.open(filename, 'rb') as f:
        # 跳过前 16 个字节的魔数和维度信息，直接读取像素数据
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        # 将一维数据 reshape 为 (N, 28, 28)
        data = data.reshape(-1, 28, 28)
    return data

def load_mnist_labels(filename):
    """读取 MNIST 格式的标签数据。"""
    with gzip.open(filename, 'rb') as f:
        # 跳过前 8 个字节的魔数和维度信息，直接读取标签数据
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

def get_data_loaders(data_dir, batch_size=64, val_split=0.1, random_seed=42):
    """
    加载 Fashion-MNIST 数据集并进行预处理。
    
    返回:
        train_images, train_labels, val_images, val_labels, test_images, test_labels
    """
    # 构建各个文件的完整路径
    train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
    test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
    
    # 1. 加载数据
    train_images = load_mnist_images(train_images_path)
    train_labels = load_mnist_labels(train_labels_path)
    test_images = load_mnist_images(test_images_path)
    test_labels = load_mnist_labels(test_labels_path)
    
    # 2. 数据归一化和展平 (从 Nx28x28 变成 Nx784，值为 0~1)
    train_images = train_images.reshape(train_images.shape[0], -1).astype(np.float32) / 255.0
    test_images = test_images.reshape(test_images.shape[0], -1).astype(np.float32) / 255.0
    
    # 3. 划分训练集和验证集
    num_train = len(train_images)
    indices = np.arange(num_train)
    
    # 随机打乱
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
    val_size = int(num_train * val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    val_images_split = train_images[val_indices]
    val_labels_split = train_labels[val_indices]
    
    train_images_split = train_images[train_indices]
    train_labels_split = train_labels[train_indices]
    
    return (train_images_split, train_labels_split), (val_images_split, val_labels_split), (test_images, test_labels)

class DataLoader:
    """简单的数据批次迭代器"""
    def __init__(self, X, y, batch_size, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X))
        
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_idx = 0
        return self
        
    def __next__(self):
        if self.current_idx >= len(self.X):
            raise StopIteration
            
        batch_indices = self.indices[self.current_idx : self.current_idx + self.batch_size]
        self.current_idx += self.batch_size
        return self.X[batch_indices], self.y[batch_indices]

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
