import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import json
import os


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model


def plot_training_curve(train_losses, val_losses, train_ppls=None, val_ppls=None,
                        save_path='results/training_curves.png'):
    """绘制训练曲线，包括损失和困惑度"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 损失曲线
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # 困惑度曲线（如果有）
    if train_ppls and val_ppls:
        ax2.plot(train_ppls, label='Training PPL', color='blue', linestyle='--')
        ax2.plot(val_ppls, label='Validation PPL', color='red', linestyle='--')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Perplexity')
        ax2.set_title('Training and Validation Perplexity')
        ax2.legend()
        ax2.grid(True)
        ax2.set_yscale('log')  # 使用对数坐标

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def calculate_perplexity(loss):
    """计算困惑度"""
    return torch.exp(torch.tensor(loss)).item()


def analyze_gradients(model, epoch, save_path='results/gradients/'):
    """分析梯度分布"""
    os.makedirs(save_path, exist_ok=True)

    gradients = []
    names = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            gradients.append(grad_norm)
            names.append(name)

    # 绘制梯度分布
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(gradients)), gradients)
    plt.xlabel('Parameters')
    plt.ylabel('Gradient Norm')
    plt.title(f'Gradient Distribution - Epoch {epoch}')
    plt.xticks(range(len(gradients)), names, rotation=90)
    plt.tight_layout()
    plt.savefig(f'{save_path}gradients_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'max_grad': max(gradients) if gradients else 0,
        'mean_grad': np.mean(gradients) if gradients else 0,
        'min_grad': min(gradients) if gradients else 0
    }


def save_experiment_results(results, config_name, save_path='results/experiments/'):
    """保存实验结果的JSON文件"""
    os.makedirs(save_path, exist_ok=True)

    filename = f"{save_path}{config_name}_results.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)


def load_experiment_results(config_name, save_path='results/experiments/'):
    """加载实验结果"""
    filename = f"{save_path}{config_name}_results.json"
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except:
        return None


def plot_ablation_study(results_dict, metric='final_val_loss', save_path='results/ablation_study.png'):
    """绘制消融实验结果"""
    configs = list(results_dict.keys())
    values = [results_dict[config][metric] for config in configs]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(configs, values)
    plt.xlabel('Configuration')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title('Ablation Study Results')
    plt.xticks(rotation=45, ha='right')

    # 在柱子上添加数值
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(values),
                 f'{value:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def compute_accuracy(predictions, targets, ignore_index=0):
    """计算准确率（忽略padding）"""
    mask = targets != ignore_index
    correct = (predictions.argmax(dim=-1) == targets) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    return accuracy.item()