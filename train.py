import os
import numpy as np
from utils.data_loader import get_data_loaders, DataLoader
from utils.metrics import calculate_accuracy
from neural_network.model import MLP
from neural_network.losses import CrossEntropyLoss
from neural_network.optimizer import SGD, StepLR

def train_model(
    data_dir,
    hidden_dims=[128, 64],
    activation='relu',
    lr=0.1,
    weight_decay=1e-4,
    batch_size=64,
    epochs=20,
    lr_decay_step=10,
    lr_decay_gamma=0.5,
    save_path="best_model.npz",
    plot=True
):
    """
    完整的训练流程，并自动保存验证集上最好的模型
    """
    print("加载数据...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size
    )
    
    train_loader = DataLoader(X_train, y_train, batch_size)
    val_loader = DataLoader(X_val, y_val, batch_size, shuffle=False)
    
    print(f"训练集大小: {len(X_train)} | 验证集大小: {len(X_val)} | 测试集大小: {len(X_test)}")
    
    # 初始化模型、损失函数、优化器
    model = MLP(input_dim=784, hidden_dims=hidden_dims, output_dim=10, activation=activation)
    criterion = CrossEntropyLoss()
    optimizer = SGD(model, lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)
    
    best_val_acc = 0.0
    
    # 记录用于图表的数据
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print("\n开始训练...")
    for epoch in range(epochs):
        # 训练阶段
        train_loss_epoch = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_i, (inputs, targets) in enumerate(train_loader):
            # 前向传播
            logits = model.forward(inputs)
            
            # 损失计算前向
            data_loss = criterion.forward(logits, targets)
            reg_loss = model.get_l2_loss(weight_decay)
            loss = data_loss + reg_loss
            
            # 后向传播
            dout = criterion.backward()
            model.backward(dout)
            
            # 参数更新
            optimizer.step()
            
            # 统计
            train_loss_epoch += loss * len(inputs)
            train_correct += np.sum(np.argmax(logits, axis=1) == targets)
            train_total += len(targets)
            
        train_loss_avg = train_loss_epoch / train_total
        train_acc_avg = train_correct / train_total
        
        # 验证阶段
        val_loss_epoch = 0.0
        val_correct = 0
        val_total = 0
        
        for inputs, targets in val_loader:
            logits = model.forward(inputs)
            data_loss = criterion.forward(logits, targets)
            reg_loss = model.get_l2_loss(weight_decay)
            loss = data_loss + reg_loss
            
            val_loss_epoch += loss * len(inputs)
            val_correct += np.sum(np.argmax(logits, axis=1) == targets)
            val_total += len(targets)
            
        val_loss_avg = val_loss_epoch / val_total
        val_acc_avg = val_correct / val_total
        
        # 记录历史
        history['train_loss'].append(train_loss_avg)
        history['train_acc'].append(train_acc_avg)
        history['val_loss'].append(val_loss_avg)
        history['val_acc'].append(val_acc_avg)
        
        # 保存最佳模型
        if val_acc_avg > best_val_acc:
            best_val_acc = val_acc_avg
            model.save_weights(save_path)
            saved_indicator = "*"
        else:
            saved_indicator = ""
            
        # 学习率调度
        scheduler.step()
        
        print(f"Epoch {epoch+1:02d}/{epochs} | LR: {optimizer.lr:.4f} "
              f"| Train Loss: {train_loss_avg:.4f} Acc: {train_acc_avg:.4f} "
              f"| Val Loss: {val_loss_avg:.4f} Acc: {val_acc_avg:.4f} {saved_indicator}")

    print(f"\n训练结束！最佳验证集准确率: {best_val_acc:.4f}，模型已保存至 {save_path}")
    
    # 绘制训练曲线（搜索阶段可关闭）
    if not plot:
        return history

    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss Curve
        ax1.plot(history['train_loss'], label='Train Loss', color='blue', marker='o')
        ax1.plot(history['val_loss'], label='Val Loss', color='red', marker='x')
        ax1.set_title("Loss Curve over Epochs")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy Curve
        ax2.plot(history['train_acc'], label='Train Acc', color='blue', marker='o')
        ax2.plot(history['val_acc'], label='Val Acc', color='red', marker='x')
        ax2.set_title("Accuracy Curve over Epochs")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True)
        
        plot_path = "training_curves.png"
        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"训练和验证集的曲线图已保存至 {plot_path}")
        
    except ImportError:
        print("未检测到 matplotlib，跳过这部分的绘制...")
        
    return history

if __name__ == "__main__":
    # 配置
    DATA_DIR = "data/fashion"
    
    # 训练运行
    train_model(
        data_dir=DATA_DIR,
        hidden_dims=[128, 64],
        activation='relu',  # 可切为 'sigmoid', 'tanh'
        lr=0.1,             # 学习率
        weight_decay=1e-4,  # L2正则化强度
        batch_size=64,
        epochs=30,
        lr_decay_step=10,
        lr_decay_gamma=0.5,
        save_path="best_model.npz"
    )
