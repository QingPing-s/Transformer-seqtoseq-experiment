import json
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict


def load_all_results(results_dir='results/experiments/'):
    """加载所有实验结果"""
    all_results = {}

    if not os.path.exists(results_dir):
        print(f"结果目录不存在: {results_dir}")
        return {}

    for filename in os.listdir(results_dir):
        if filename.endswith('_results.json'):
            config_name = filename.replace('_results.json', '')
            try:
                with open(os.path.join(results_dir, filename), 'r') as f:
                    all_results[config_name] = json.load(f)
            except:
                print(f"加载失败: {filename}")

    return all_results


def plot_comprehensive_analysis(all_results):
    """绘制综合分析图表"""
    if not all_results:
        print("没有可分析的结果")
        return

    configs = list(all_results.keys())

    # 创建多个子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 最终验证损失比较
    final_losses = [all_results[config]['final_val_loss'] for config in configs]
    ax1.bar(configs, final_losses, color='skyblue', alpha=0.7)
    ax1.set_title('Final Validation Loss')
    ax1.set_ylabel('Loss')
    ax1.tick_params(axis='x', rotation=45)

    # 在柱子上添加数值
    for i, v in enumerate(final_losses):
        ax1.text(i, v, f'{v:.4f}', ha='center', va='bottom')

    # 2. 最佳验证困惑度比较
    best_ppls = [all_results[config]['best_val_ppl'] for config in configs]
    ax2.bar(configs, best_ppls, color='lightcoral', alpha=0.7)
    ax2.set_title('Best Validation Perplexity')
    ax2.set_ylabel('Perplexity')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_yscale('log')  # 对数坐标

    for i, v in enumerate(best_ppls):
        ax2.text(i, v, f'{v:.1f}', ha='center', va='bottom')

    # 3. 训练时间比较
    training_times = [all_results[config]['total_training_time'] for config in configs]
    ax3.bar(configs, training_times, color='lightgreen', alpha=0.7)
    ax3.set_title('Training Time')
    ax3.set_ylabel('Time (seconds)')
    ax3.tick_params(axis='x', rotation=45)

    for i, v in enumerate(training_times):
        ax3.text(i, v, f'{v:.0f}s', ha='center', va='bottom')

    # 4. 模型参数数量比较
    param_counts = [all_results[config]['model_params'] for config in configs]
    ax4.bar(configs, param_counts, color='gold', alpha=0.7)
    ax4.set_title('Model Parameters')
    ax4.set_ylabel('Parameter Count')
    ax4.tick_params(axis='x', rotation=45)

    for i, v in enumerate(param_counts):
        ax4.text(i, v, f'{v / 1e6:.1f}M', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('results/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_dynamics_comparison(all_results):
    """比较不同配置的训练动态"""
    if not all_results:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 训练损失比较
    for config_name, results in all_results.items():
        train_losses = results['train_losses']
        ax1.plot(train_losses, label=config_name, linewidth=2)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True)

    # 验证困惑度比较
    for config_name, results in all_results.items():
        if 'val_ppls' in results and results['val_ppls']:
            val_ppls = results['val_ppls']
            ax2.plot(val_ppls, label=config_name, linewidth=2)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Perplexity')
    ax2.set_title('Validation Perplexity Comparison')
    ax2.legend()
    ax2.grid(True)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig('results/training_dynamics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


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

    print("=" * 80)
    print("消融实验分析报告")
    print("=" * 80)

    print(f"\n基准配置: {base_config}")
    print(f"基准结果 - 损失: {base_results['best_val_loss']:.4f}, 困惑度: {base_results['best_val_ppl']:.2f}")
    print("\n对比结果:")
    print("-" * 80)
    print(f"{'配置':<20} {'损失':<10} {'变化率':<10} {'困惑度':<10} {'变化率':<10} {'参数':<10}")
    print("-" * 80)

    for config_name, results in all_results.items():
        if config_name == base_config:
            continue

        loss_change = (results['best_val_loss'] - base_results['best_val_loss']) / base_results['best_val_loss'] * 100
        ppl_change = (results['best_val_ppl'] - base_results['best_val_ppl']) / base_results['best_val_ppl'] * 100

        print(f"{config_name:<20} {results['best_val_loss']:<10.4f} {loss_change:>+7.1f}% "
              f"{results['best_val_ppl']:<10.1f} {ppl_change:>+7.1f}% "
              f"{results['model_params'] / 1e6:<8.1f}M")

    print("-" * 80)


def main():
    """主分析函数"""
    # 加载所有结果
    all_results = load_all_results()

    if not all_results:
        print("没有找到实验结果")
        return

    print(f"加载了 {len(all_results)} 个实验的结果")

    # 生成各种分析图表
    plot_comprehensive_analysis(all_results)
    plot_training_dynamics_comparison(all_results)

    # 生成文本报告
    generate_ablation_report(all_results)

    # 保存汇总结果
    summary = {}
    for config, results in all_results.items():
        summary[config] = {
            'best_val_loss': results['best_val_loss'],
            'best_val_ppl': results['best_val_ppl'],
            'training_time': results['total_training_time'],
            'model_params': results['model_params']
        }

    with open('results/experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n分析完成! 结果保存在 results/ 目录")


if __name__ == "__main__":
    main()