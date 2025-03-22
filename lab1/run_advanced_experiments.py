from mnist_ffn import load_data, train, test, plot_curves
from models.advanced_mlp_models import MLPMixer, ResMLP, VisionPermutator
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
import time

# 设置随机种子以保证可重复性
torch.manual_seed(42)

# 检查是否可用CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 创建结果目录
os.makedirs('results', exist_ok=True)

# 运行实验函数
def run_experiment(model, batch_size, lr, epochs, model_name):
    print(f"正在运行实验: {model_name}")
    
    # 加载数据
    train_loader, test_loader = load_data(batch_size)
    
    # 将模型移动到设备上
    model = model.to(device)
    print(model)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 存储指标的列表
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    
    start_time = time.time()
    
    # 训练循环
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        print(f'轮次: {epoch+1}/{epochs}, 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%, 测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.2f}%')
    
    elapsed_time = time.time() - start_time
    print(f'训练完成，耗时 {elapsed_time:.2f} 秒')
    print(f'最佳测试准确率: {max(test_accs):.2f}%')
    
    # 绘制并保存学习曲线
    plot_curves(train_losses, test_losses, train_accs, test_accs, model_name)
    
    # 保存模型
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), f'models/{model_name}.pth')
    
    return model, max(test_accs), train_losses, test_losses, train_accs, test_accs

# 定义实验配置
experiments = [
    # MLP-Mixer模型实验
    {
        'model': MLPMixer(
            image_size=28,
            patch_size=4,  # 这会产生7x7=49个patches
            num_classes=10,
            num_blocks=6,
            hidden_dim=256,
            token_hidden_dim=384,
            channel_hidden_dim=512
        ),
        'batch_size': 128,
        'lr': 0.001,
        'epochs': 15,
        'model_name': 'mlp_mixer'
    },
    
    # ResMLP模型实验
    {
        'model': ResMLP(
            image_size=28,
            patch_size=4,
            num_classes=10,
            num_blocks=6,
            hidden_dim=256,
            mlp_hidden_dim=512
        ),
        'batch_size': 128,
        'lr': 0.001,
        'epochs': 15,
        'model_name': 'resmlp'
    },
    
    # Vision Permutator模型实验
    {
        'model': VisionPermutator(
            image_size=28,
            patch_size=4,
            num_classes=10,
            num_blocks=6,
            hidden_dim=256,  # 确保能被segment_dim整除
            mlp_hidden_dim=512,
            segment_dim=8  # 256 / 8 = 32，是整数
        ),
        'batch_size': 128,
        'lr': 0.001,
        'epochs': 15,
        'model_name': 'vision_permutator'
    }
]

# 运行所有实验并收集结果
results = []
for config in experiments:
    print(f"\n{'='*50}")
    print(f"运行实验: {config['model_name']}")
    print(f"{'='*50}")
    
    model, best_acc, train_losses, test_losses, train_accs, test_accs = run_experiment(**config)
    
    # 存储结果
    results.append({
        'model_name': config['model_name'],
        'batch_size': config['batch_size'],
        'learning_rate': config['lr'],
        'best_accuracy': best_acc,
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1]
    })

# 创建结果数据框并保存为CSV
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('best_accuracy', ascending=False)
results_df.to_csv('results/advanced_experiment_results.csv', index=False)

print("\n所有实验已完成!")
print("排序后的模型性能:")
print(results_df.to_string(index=False)) 