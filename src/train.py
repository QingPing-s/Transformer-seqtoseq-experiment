import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import os
import math
import time
import numpy as np

from config import Config
from src.model import Transformer
from src.data_loader import get_data_loaders
from src.utils import *


class AdvancedTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device

        # 创建结果目录
        os.makedirs('results', exist_ok=True)
        os.makedirs('results/gradients', exist_ok=True)
        os.makedirs('results/experiments', exist_ok=True)

        # 数据加载
        print("加载数据...")
        self.train_loader, self.val_loader, self.src_vocab, self.tgt_vocab = get_data_loaders(config)

        # 模型初始化
        self.model = Transformer(config, len(self.src_vocab), len(self.tgt_vocab)).to(self.device)

        # 优化器和损失函数
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
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

        # 最佳模型跟踪
        self.best_val_loss = float('inf')
        self.best_model_state = None

        print(f"开始训练配置: {config.config_name}")
        print(f"设备: {self.device}")
        print(f"训练数据: {len(self.train_loader.dataset)} 条")
        print(f"验证数据: {len(self.val_loader.dataset)} 条")
        print(f"模型总参数: {sum(p.numel() for p in self.model.parameters()):,}")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_tokens = 0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} Training")

        for src, tgt in progress_bar:
            src, tgt = src.to(self.device), tgt.to(self.device)

            # 创建目标掩码
            tgt_seq_len = tgt.size(1)
            tgt_mask = self.model.generate_square_subsequent_mask(tgt_seq_len).to(self.device)

            self.optimizer.zero_grad()

            # 前向传播
            output = self.model(src, tgt[:, :-1], tgt_mask=tgt_mask[:-1, :-1])

            # 计算损失
            loss = self.criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))

            # 反向传播
            loss.backward()

            # 梯度裁剪
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            # 分析梯度（每5个epoch分析一次）
            if epoch % 5 == 0 and num_batches == 0:
                grad_stats = analyze_gradients(self.model, epoch)
                self.gradient_stats.append(grad_stats)

            self.optimizer.step()

            # 统计
            batch_loss = loss.item()
            total_loss += batch_loss
            num_batches += 1

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'avg_loss': f'{total_loss / num_batches:.4f}'
            })

        avg_loss = total_loss / num_batches
        perplexity = calculate_perplexity(avg_loss)

        return avg_loss, perplexity

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
                accuracy = compute_accuracy(output, tgt[:, 1:])

                total_loss += loss.item()
                total_accuracy += accuracy
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
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
            train_loss, train_ppl = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_ppls.append(train_ppl)

            # 验证
            val_loss, val_ppl, val_accuracy = self.validate_epoch(epoch)
            self.val_losses.append(val_loss)
            self.val_ppls.append(val_ppl)

            # 学习率调度
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            if self.scheduler:
                self.scheduler.step()

            epoch_time = time.time() - epoch_start

            # 打印epoch结果
            print(f"Epoch {epoch + 1}/{self.config.num_epochs} - {epoch_time:.1f}s")
            print(f"  训练损失: {train_loss:.4f}, 训练困惑度: {train_ppl:.2f}")
            print(f"  验证损失: {val_loss:.4f}, 验证困惑度: {val_ppl:.2f}, 验证准确率: {val_accuracy:.4f}")
            print(f"  学习率: {current_lr:.2e}")

            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                save_model(self.model, f'results/best_model_{self.config.config_name}.pth')
                print(f"  ✓ 新的最佳模型已保存!")

            # 每2个epoch保存一次检查点
            if (epoch + 1) % 2 == 0:
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
            'best_val_loss': self.best_val_loss,
            'best_val_ppl': calculate_perplexity(self.best_val_loss),
            'total_training_time': total_time,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_ppls': self.train_ppls,
            'val_ppls': self.val_ppls,
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


def run_single_experiment(config_name):
    """运行单个实验配置"""
    print(f"\n{'=' * 60}")
    print(f"开始实验: {config_name}")
    print(f"{'=' * 60}")

    # 加载配置
    config = Config(config_name)
    config.save()  # 保存配置

    # 创建训练器并训练
    trainer = AdvancedTrainer(config)
    results = trainer.train()

    # 绘制结果
    trainer.plot_results()

    # 保存实验结果
    save_experiment_results(results, config_name)

    print(f"实验 {config_name} 完成!")
    return results


def run_ablation_study(experiment_configs):
    """运行消融实验"""
    all_results = {}

    for config_name in experiment_configs:
        try:
            results = run_single_experiment(config_name)
            all_results[config_name] = results

            # 打印简要结果
            print(f"\n{config_name} 结果:")
            print(f"  最佳验证损失: {results['best_val_loss']:.4f}")
            print(f"  最佳验证困惑度: {results['best_val_ppl']:.2f}")
            print(f"  训练时间: {results['total_training_time']:.1f}s")

        except Exception as e:
            print(f"实验 {config_name} 失败: {e}")
            continue

    # 绘制消融实验结果
    if all_results:
        # 损失比较
        plot_ablation_study(all_results, 'best_val_loss', 'results/ablation_loss.png')
        # 困惑度比较
        plot_ablation_study(all_results, 'best_val_ppl', 'results/ablation_perplexity.png')

        # 保存所有结果
        with open('results/ablation_study_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)

        print("\n消融实验完成!")
        print("结果已保存到: results/ablation_study_results.json")

    return all_results


def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # 检查数据文件
    config = Config()
    for split, path in config.data_paths.items():
        if not os.path.exists(path):
            print(f"警告: 数据文件 {path} 不存在")

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

    print("可用的实验配置:")
    for i, config in enumerate(experiment_configs):
        print(f"  {i + 1}. {config}")

    # 用户选择
    try:
        choice = input("\n选择运行模式:\n1. 单个实验\n2. 完整消融实验\n3. 基础训练\n选择 (1-3): ").strip()

        if choice == "1":
            # 单个实验
            config_name = input(f"输入配置名称 ({', '.join(experiment_configs)}): ").strip()
            if config_name in experiment_configs:
                run_single_experiment(config_name)
            else:
                print("无效的配置名称，使用基础配置")
                run_single_experiment("base")

        elif choice == "2":
            # 完整消融实验
            run_ablation_study(experiment_configs)

        else:
            # 基础训练
            run_single_experiment("base")

    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"训练出错: {e}")
        # 尝试基础训练
        try:
            run_single_experiment("base")
        except:
            print("基础训练也失败")


if __name__ == "__main__":
    main()