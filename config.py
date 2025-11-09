import torch
import json


class Config:
    def __init__(self, config_name="base"):
        self.config_name = config_name

        # 基础模型参数
        self.d_model = 128
        self.nhead = 4
        self.num_encoder_layers = 2
        self.num_decoder_layers = 2
        self.dim_feedforward = 512
        self.dropout = 0.1
        self.max_seq_length = 64

        # 训练参数
        self.batch_size = 32
        self.learning_rate = 3e-4
        self.num_epochs = 10
        self.grad_clip = 1.0
        self.weight_decay = 0.01  # AdamW权重衰减

        # 进阶功能开关
        self.use_positional_encoding = True
        self.use_residual_connections = True
        self.use_layernorm = True
        self.use_multihead_attention = True
        self.use_learning_rate_scheduler = True

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 数据路径
        self.data_paths = {
            'train': 'data/iwslt2017-train.arrow',
            'val': 'data/iwslt2017-validation.arrow',
            'test': 'data/iwslt2017-test.arrow'
        }

        # 根据配置名称调整参数
        self._apply_config(config_name)

    def _apply_config(self, config_name):
        """应用不同的实验配置"""
        if config_name == "no_positional_encoding":
            self.use_positional_encoding = False
            self.config_name = "no_pos_encoding"

        elif config_name == "no_residual":
            self.use_residual_connections = False
            self.use_layernorm = False
            self.config_name = "no_residual"

        elif config_name == "single_head":
            self.nhead = 1
            self.config_name = "single_head"

        elif config_name == "small_model":
            self.d_model = 64
            self.dim_feedforward = 256
            self.config_name = "small_model"

        elif config_name == "no_scheduler":
            self.use_learning_rate_scheduler = False
            self.config_name = "no_scheduler"

        elif config_name == "high_dropout":
            self.dropout = 0.3
            self.config_name = "high_dropout"

    def to_dict(self):
        """转换为字典，用于保存配置"""
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_') and k != 'device'}

    def save(self, path='results/configs/'):
        """保存配置到文件"""
        import os
        os.makedirs(path, exist_ok=True)
        with open(f'{path}{self.config_name}_config.json', 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, config_name, path='results/configs/'):
        """从文件加载配置"""
        try:
            with open(f'{path}{config_name}_config.json', 'r') as f:
                config_dict = json.load(f)

            config = cls(config_name)
            for k, v in config_dict.items():
                if hasattr(config, k):
                    setattr(config, k, v)
            return config
        except:
            return cls(config_name)