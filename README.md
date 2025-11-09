

# Transformer Encoder-Decoder实现作业 

### ✨ 项目特色
🔥 从零实现：完全手写Transformer所有核心组件

📊 消融实验：6种不同配置的对比分析

🎯 进阶功能：梯度裁剪、困惑度分析、学习率调度等

📈 可视化：完整的训练曲线和实验结果分析

🔧 模块化设计：易于扩展和修改的代码结构


src/
├── model_relpos.py           # 模型定义（Encoder–Decoder + 相对位置偏置）
├── data_iwslt.py             # 数据加载与 SentencePiece 分词器
├── train_mt.py               # 训练与验证
├── eval_bleu.py              # BLEU 评估
├── sample_mt.py              # 翻译示例
├── run_sensitivity.py        # 超参数敏感性分析脚本
├── ablation/                 # 消融实验相关文件
│   ├── model_ablation.py     # 支持消融实验的模型实现
│   ├── model_no_relpos.py    # 无相对位置偏置的模型变体
│   ├── train_ablation.py     # 消融实验训练脚本
│   ├── run_ablation_relpos.py# 相对位置偏置消融实验
│   └── run_comprehensive_ablation_v2.py # 综合消融实验
scripts/
├── run_iwslt.sh              # 一键运行脚本
results/
├── run_experiments/          # 各实验结果目录
│   ├── run_base/             # 基线模型结果
│   └── sensitivity/          # 超参分析实验结果
├── ablation_comprehensive_summary.csv # 消融实验结果汇总
├── sensitivity_d_model.csv   # d_model 敏感性分析结果
├── sensitivity_num_layers.csv# 层数敏感性分析结果
└── sensitivity_batch_size.csv# 批大小敏感性分析结果


### 项目结构
Transformer-seqtoseq-experiment/
├── src/                    # 源代码目录
│   ├── model.py           # Transformer模型实现
│   ├── data_loader.py     # 数据加载与处理
│   └── utils.py           # 工具函数
├── config.py              # 配置文件管理
├── train.py               # 主训练脚本
├── analyze_results.py     # 结果分析脚本
├── requirements.txt        # 依赖包列表
└── results/               # 实验结果（自动生成）
    ├── training_curves_*.png
    ├── ablation_*.png
    └── experiments/

### ⚡快速开始
# 克隆项目
git clone https://github.com/your-username/transformer-assignment.git
cd transformer-assignment

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows

# 安装依赖
pip install -r requirements.txt

### 数据准备
data/
├── iwslt2017-train.arrow
├── iwslt2017-validation.arrow
└── iwslt2017-test.arrow

### 基础训练
# 设置随机种子确保可复现性
python -c "import torch; torch.manual_seed(42)"

# 运行基础训练
python train.py
#训练过程中会显示实时进度和指标：
Epoch 1/10 - 45s
  训练损失: 4.2156, 训练困惑度: 67.72
  验证损失: 3.9872, 验证困惑度: 53.91, 验证准确率: 0.1245
  学习率: 3.00e-04
  ✓ 新的最佳模型已保存

### 🔬实验功能
消融实验
项目支持6种不同的实验配置对比：

#单个实验
python train.py
#结果分析
python analyze_results.py

📊 实验结果示例
<img width="400" height="292" alt="4" src="https://github.com/user-attachments/assets/aedee79a-c239-4abb-869c-9dfdaeae3073" />
<img width="394" height="294" alt="5" src="https://github.com/user-attachments/assets/1acb9451-9a95-4374-85d8-04801312bc2b" />





