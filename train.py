import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import os
import math
import time
import numpy as np
import json
import sys
import matplotlib.pyplot as plt

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import Config
from model import Transformer
from data_loader import get_data_loaders
from utils import *


class AdvancedTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device

        # 创建结果目录
        os.makedirs('results', exist_ok=True)
        os.makedirs('results/gradients', exist_ok=True)
        os.makedirs('results/experiments', exist_ok=True)
        os.makedirs('results/configs', exist_ok=True)

        # 数据加载
        print("加载数据...")
        self.train_loader, self.val_loader, self.src_vocab, self.tgt_vocab = get_data_loaders(config)

        # 模型初始化
        self.model = Transformer(config, len(self.src_vocab), len(self.tgt_vocab)).to(self.device)

        # 优化器和损失函数
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # 学习率调度器
        if config.use_learning_rate_scheduler:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        else:
            self.scheduler = None

        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.train_ppls = []
        self.val_ppls = []
        self.learning_rates = []
        self.gradient_stats = []
        self.train_accuracies = []
        self.val_accuracies = []

        # 最佳模型跟踪
        self.best_val_loss = float('inf')
        self.best_val_ppl = float('inf')
        self.best_model_state = None

        print(f"开始训练配置: {config.config_name}")
        print(f"设备: {self.device}")
        print(f"训练数据: {len(self.train_loader.dataset)} 条")
        print(f"验证数据: {len(self.val_loader.dataset)} 条")
        print(f"批次数量: {len(self.train_loader)}")
        print(f"模型总参数: {sum(p.numel() for p in self.model.parameters()):,}")

        # 保存配置
        config.save()

    def compute_accuracy(self, output, targets):
        """计算准确率"""
        # 忽略padding
        mask = targets != 0
        predictions = output.argmax(dim=-1)
        correct = (predictions == targets) & mask
        accuracy = correct.sum().float() / mask.sum().float()
        return accuracy.item()

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} Training")

        for batch_idx, (src, tgt) in enumerate(progress_bar):
            src, tgt = src.to(self.device), tgt.to(self.device)

            # 创建目标掩码（因果掩码）
            tgt_seq_len = tgt.size(1)
            tgt_mask = self.model.generate_square_subsequent_mask(tgt_seq_len).to(self.device)

            self.optimizer.zero_grad()

            # 前向传播 - 使用teacher forcing
            output = self.model(src, tgt[:, :-1], tgt_mask=tgt_mask[:-1, :-1])

            # 计算损失
            loss = self.criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))

            # 计算准确率
            accuracy = self.compute_accuracy(output, tgt[:, 1:])

            # 反向传播
            loss.backward()

            # 梯度裁剪
            if self.config.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            # 分析梯度（每5个epoch分析一次）
            if epoch % 5 == 0 and batch_idx == 0:
                grad_stats = analyze_gradients(self.model, epoch)
                self.gradient_stats.append(grad_stats)

            self.optimizer.step()

            # 统计
            batch_loss = loss.item()
            total_loss += batch_loss
            total_accuracy += accuracy
            num_batches += 1

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'acc': f'{accuracy:.4f}',
                'avg_loss': f'{total_loss / num_batches:.4f}'
            })

        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        perplexity = calculate_perplexity(avg_loss)

        return avg_loss, perplexity, avg_accuracy

    def validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0

        with torch.no_grad():
            for src, tgt in tqdm(self.val_loader, desc=f"Epoch {epoch + 1} Validation"):
                src, tgt = src.to(self.device), tgt.to(self.device)

                tgt_seq_len = tgt.size(1)
                tgt_mask = self.model.generate_square_subsequent_mask(tgt_seq_len).to(self.device)

                output = self.model(src, tgt[:, :-1], tgt_mask=tgt_mask[:-1, :-1])
                loss = self.criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))

                # 计算准确率
                accuracy = self.compute_accuracy(output, tgt[:, 1:])

                total_loss += loss.item()
                total_accuracy += accuracy
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0
        perplexity = calculate_perplexity(avg_loss)

        return avg_loss, perplexity, avg_accuracy

    def train(self):
        """执行完整训练过程"""
        start_time = time.time()

        print(f"\n开始训练配置: {self.config.config_name}")
        print("=" * 60)

        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()

            # 训练
            train_loss, train_ppl, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_ppls.append(train_ppl)
            self.train_accuracies.append(train_acc)

            # 验证
            val_loss, val_ppl, val_acc = self.validate_epoch(epoch)
            self.val_losses.append(val_loss)
            self.val_ppls.append(val_ppl)
            self.val_accuracies.append(val_acc)

            # 学习率调度
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            if self.scheduler:
                self.scheduler.step()

            epoch_time = time.time() - epoch_start

            # 打印epoch结果
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs} - {epoch_time:.1f}s")
            print(f"  训练损失: {train_loss:.4f}, 训练困惑度: {train_ppl:.2f}, 训练准确率: {train_acc:.4f}")
            print(f"  验证损失: {val_loss:.4f}, 验证困惑度: {val_ppl:.2f}, 验证准确率: {val_acc:.4f}")
            print(f"  学习率: {current_lr:.2e}")

            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_ppl = val_ppl
                self.best_model_state = self.model.state_dict().copy()
                save_model(self.model, f'results/best_model_{self.config.config_name}.pth')
                print(f"  ✓ 新的最佳模型已保存!")

            # 每2个epoch保存一次检查点
            if (epoch + 1) % 2 == 0 or epoch == self.config.num_epochs - 1:
                save_model(self.model, f'results/checkpoint_epoch_{epoch + 1}_{self.config.config_name}.pth')

        # 训练完成
        total_time = time.time() - start_time
        print(f"\n训练完成! 总时间: {total_time:.1f}s")

        # 保存最终模型
        save_model(self.model, f'results/final_model_{self.config.config_name}.pth')

        # 恢复最佳模型
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        return self._get_training_results(total_time)

    def _get_training_results(self, total_time):
        """整理训练结果"""
        return {
            'config_name': self.config.config_name,
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'final_train_ppl': self.train_ppls[-1] if self.train_ppls else 0,
            'final_val_ppl': self.val_ppls[-1] if self.val_ppls else 0,
            'final_train_acc': self.train_accuracies[-1] if self.train_accuracies else 0,
            'final_val_acc': self.val_accuracies[-1] if self.val_accuracies else 0,
            'best_val_loss': self.best_val_loss,
            'best_val_ppl': self.best_val_ppl,
            'total_training_time': total_time,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_ppls': self.train_ppls,
            'val_ppls': self.val_ppls,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'gradient_stats': self.gradient_stats,
            'model_params': sum(p.numel() for p in self.model.parameters()),
            'config': self.config.to_dict()
        }

    def plot_results(self):
        """绘制所有训练结果"""
        # 训练曲线
        plot_training_curve(
            self.train_losses,
            self.val_losses,
            self.train_ppls,
            self.val_ppls,
            f'results/training_curves_{self.config.config_name}.png'
        )

        # 学习率曲线
        plt.figure(figsize=(10, 5))
        plt.plot(self.learning_rates)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        plt.savefig(f'results/learning_rate_{self.config.config_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 准确率曲线
        if self.train_accuracies and self.val_accuracies:
            plt.figure(figsize=(10, 5))
            plt.plot(self.train_accuracies, label='Training Accuracy')
            plt.plot(self.val_accuracies, label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'results/accuracy_curve_{self.config.config_name}.png', dpi=300, bbox_inches='tight')
            plt.close()


def check_data_availability(config):
    """检查数据可用性"""
    print("检查数据可用性...")

    data_available = True
    for split, path in config.data_paths.items():
        if path == 'virtual':
            print(f"  {split}: 使用虚拟数据")
            continue

        if os.path.exists(path):
            print(f"  {split}: ✓ 找到数据 ({path})")
        else:
            print(f"  {split}: ✗ 数据不存在 ({path})")
            data_available = False

    if not data_available:
        print("\n警告: 部分数据文件不存在，将使用虚拟数据")
        print("要下载真实数据，请运行: python download_data.py")

    return data_available


def run_single_experiment(config_name, dataset_type="virtual"):
    """运行单个实验配置"""
    print(f"\n{'=' * 60}")
    print(f"开始实验: {config_name}")
    print(f"{'=' * 60}")

    # 加载配置
    config = Config(config_name, dataset_type)

    # 检查数据可用性
    check_data_availability(config)

    # 创建训练器并训练
    trainer = AdvancedTrainer(config)
    results = trainer.train()

    # 绘制结果
    trainer.plot_results()

    # 保存实验结果
    save_experiment_results(results, config_name)

    print(f"实验 {config_name} 完成!")
    return results


def run_ablation_study(experiment_configs, dataset_type="virtual"):
    """运行消融实验"""
    all_results = {}

    for config_name in experiment_configs:
        try:
            print(f"\n{'=' * 60}")
            print(f"开始消融实验: {config_name}")
            print(f"{'=' * 60}")

            results = run_single_experiment(config_name, dataset_type)
            all_results[config_name] = results

            # 打印简要结果
            print(f"\n{config_name} 结果:")
            print(f"  最佳验证损失: {results['best_val_loss']:.4f}")
            print(f"  最佳验证困惑度: {results['best_val_ppl']:.2f}")
            print(f"  最终验证准确率: {results['final_val_acc']:.4f}")
            print(f"  训练时间: {results['total_training_time']:.1f}s")
            print(f"  模型参数: {results['model_params']:,}")

        except Exception as e:
            print(f"实验 {config_name} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 绘制消融实验结果
    if all_results:
        # 损失比较
        plot_ablation_study(all_results, 'best_val_loss', 'results/ablation_loss.png')
        # 困惑度比较
        plot_ablation_study(all_results, 'best_val_ppl', 'results/ablation_perplexity.png')
        # 准确率比较
        plot_ablation_study(all_results, 'final_val_acc', 'results/ablation_accuracy.png')

        # 保存所有结果
        with open('results/ablation_study_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)

        print("\n消融实验完成!")
        print("结果已保存到: results/ablation_study_results.json")

        # 生成简要报告
        generate_ablation_report(all_results)

    return all_results


def generate_ablation_report(all_results):
    """生成消融实验报告"""
    if not all_results:
        print("没有结果可生成报告")
        return

    # 找到基础配置
    base_config = None
    for config in all_results.keys():
        if config == 'base' or 'base' in config:
            base_config = config
            break

    if not base_config:
        base_config = list(all_results.keys())[0]

    base_results = all_results[base_config]

    print("\n" + "=" * 80)
    print("消融实验分析报告")
    print("=" * 80)

    print(f"\n基准配置: {base_config}")
    print(f"基准结果 - 损失: {base_results['best_val_loss']:.4f}, "
          f"困惑度: {base_results['best_val_ppl']:.2f}, "
          f"准确率: {base_results['final_val_acc']:.4f}")
    print("\n对比结果:")
    print("-" * 100)
    print(f"{'配置':<20} {'损失':<10} {'变化率':<10} {'困惑度':<10} {'变化率':<10} {'准确率':<10} {'参数':<10}")
    print("-" * 100)

    for config_name, results in all_results.items():
        loss_change = (results['best_val_loss'] - base_results['best_val_loss']) / base_results['best_val_loss'] * 100
        ppl_change = (results['best_val_ppl'] - base_results['best_val_ppl']) / base_results['best_val_ppl'] * 100
        acc_change = (results['final_val_acc'] - base_results['final_val_acc']) / base_results['final_val_acc'] * 100

        print(f"{config_name:<20} {results['best_val_loss']:<10.4f} {loss_change:>+7.1f}% "
              f"{results['best_val_ppl']:<10.1f} {ppl_change:>+7.1f}% "
              f"{results['final_val_acc']:<10.4f} {acc_change:>+7.1f}% "
              f"{results['model_params'] / 1e6:<8.1f}M")

    print("-" * 100)


def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    np.random.seed(42)

    # 用户选择数据集类型
    print("选择数据集类型:")
    print("1. IWSLT2017 (英德翻译)")
    print("2. Tiny Shakespeare (小型文本)")
    print("3. 虚拟数据 (默认)")

    dataset_choice = input("请输入选择 (1, 2 或 3，直接回车使用默认): ").strip()
    dataset_type = "virtual"

    if dataset_choice == "1":
        dataset_type = "iwslt2017"
    elif dataset_choice == "2":
        dataset_type = "tiny_shakespeare"

    # 定义实验配置
    experiment_configs = [
        "base",  # 基础配置
        "no_positional_encoding",  # 无位置编码
        "no_residual",  # 无残差连接
        "single_head",  # 单头注意力
        "small_model",  # 小模型
        "no_scheduler",  # 无学习率调度
        "high_dropout",  # 高dropout
    ]

    print("\n可用的实验配置:")
    for i, config in enumerate(experiment_configs):
        print(f"  {i + 1}. {config}")

    # 用户选择运行模式
    try:
        choice = input(
            "\n选择运行模式:\n1. 单个实验\n2. 完整消融实验\n3. 基础训练\n选择 (1-3，直接回车使用默认): ").strip()

        if choice == "1":
            # 单个实验
            print("\n可用的配置:")
            for i, config in enumerate(experiment_configs):
                print(f"  {i + 1}. {config}")

            config_choice = input("输入配置编号或名称: ").strip()

            # 尝试解析编号
            try:
                config_idx = int(config_choice) - 1
                if 0 <= config_idx < len(experiment_configs):
                    config_name = experiment_configs[config_idx]
                else:
                    config_name = "base"
            except:
                # 如果是名称
                config_name = config_choice if config_choice in experiment_configs else "base"

            run_single_experiment(config_name, dataset_type)

        elif choice == "2":
            # 完整消融实验
            print(f"\n开始完整消融实验，共 {len(experiment_configs)} 个配置")
            run_ablation_study(experiment_configs, dataset_type)

        else:
            # 基础训练
            run_single_experiment("base", dataset_type)

    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"训练出错: {e}")
        import traceback
        traceback.print_exc()

        # 尝试基础训练
        try:
            print("\n尝试基础训练...")
            run_single_experiment("base", dataset_type)
        except Exception as e2:
            print(f"基础训练也失败: {e2}")


if __name__ == "__main__":
    main()