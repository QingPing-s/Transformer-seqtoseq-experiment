

# Transformer Encoder-Decoder实现作业 

### ✨ 项目特色
🔥 从零实现：完全手写Transformer所有核心组件

📊 消融实验：6种不同配置的对比分析

🎯 进阶功能：梯度裁剪、困惑度分析、学习率调度等

📈 可视化：完整的训练曲线和实验结果分析

🔧 模块化设计：易于扩展和修改的代码结构



### 项目结构
```text
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
```

### ⚙️ 环境配置
```text
torch>=2.0.0
torchtext>=0.15.0
datasets>=2.10.0
tokenizers>=0.13.0
numpy>=1.21.0
matplotlib>=3.5.0
tqdm>=4.64.0
sacrebleu>=2.2.0
```

### 安装依赖
```text
pip install -r requirements.txt
```


### 数据准备
```text
data/
├── iwslt2017-train.arrow
├── iwslt2017-validation.arrow
└── iwslt2017-test.arrow
```

# 基础训练
### 设置随机种子确保可复现性
```text
python -c "import torch; torch.manual_seed(42)"
```


### 运行基础训练
```text
python train.py
```
###训练过程中会显示实时进度和指标：
```text
  Epoch 1/10 - 45s
  训练损失: 4.2156, 训练困惑度: 67.72
  验证损失: 3.9872, 验证困惑度: 53.91, 验证准确率: 0.1245
  学习率: 3.00e-04
  ✓ 新的最佳模型已保存
```

### 🔬消融实验
项目支持6种不同的实验配置对比：

### 单个实验
```text
python train.py
```
### 结果分析
```text
python analyze_results.py
```

### 📊 实验结果

<img width="400" height="292" alt="4" src="https://github.com/user-attachments/assets/aedee79a-c239-4abb-869c-9dfdaeae3073" />
<img width="394" height="294" alt="5" src="https://github.com/user-attachments/assets/1acb9451-9a95-4374-85d8-04801312bc2b" />





