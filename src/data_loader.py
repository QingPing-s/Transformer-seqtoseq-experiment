import torch
from torch.utils.data import Dataset, DataLoader
import os
from datasets import load_dataset, Dataset as HFDataset
import pandas as pd


class IWSLTDataset(Dataset):
    def __init__(self, data_path, max_length=64, split='train'):
        self.max_length = max_length
        self.split = split

        # 尝试多种数据加载方式
        self.data = self._load_data(data_path, split)

    def _load_data(self, data_path, split):
        """尝试多种方式加载数据"""
        data = None

        # 方式1: 检查本地文件是否存在
        if os.path.exists(data_path):
            try:
                # 尝试加载本地arrow文件
                data = HFDataset.from_file(data_path)
                print(f"成功加载本地数据: {data_path}")
                return data
            except Exception as e:
                print(f"本地文件加载失败 {data_path}: {e}")

        # 方式2: 从Hugging Face下载数据集
        try:
            print(f"尝试从Hugging Face下载IWSLT2017数据集 ({split})...")
            if split == 'train':
                dataset = load_dataset("iwslt2017", "iwslt2017-de-en", split='train')
            elif split == 'val':
                dataset = load_dataset("iwslt2017", "iwslt2017-de-en", split='validation')
            else:
                dataset = load_dataset("极iwslt2017", "iwslt2017-de-en", split='test')

            print(f"成功下载数据集: {len(dataset)} 条样本")
            return dataset
        except Exception as e:
            print(f"Hugging Face下载失败: {e}")

        # 方式3: 使用更大的虚拟数据集
        print("使用增强的虚拟数据集...")
        return self._create_enhanced_dummy_data()

    def _create_enhanced_dummy_data(self):
        """创建更真实的虚拟数据集"""
        # 英语-德语翻译样本
        translation_pairs = [
            # 日常对话
            ("Hello, how are you?", "Hallo, wie geht es dir?"),
            ("What is your name?", "Wie heißt du?"),
            ("Where are you from?", "Woher kommst du?"),
            ("Thank you very much.", "Vielen Dank."),
            ("I don't understand.", "Ich verstehe nicht."),
            ("Can you help me?", "Kannst du mir helfen?"),
            ("What time is it?", "Wie spät ist es?"),
            ("Where is the station?", "Wo ist der Bahnhof?"),
            ("How much does it cost?", "Wie viel kostet das?"),
            ("I would like coffee.", "Ich hätte gern Kaffee."),

            # 更长的句子
            ("The weather is very nice today.", "Das Wetter ist heute sehr schön."),
            ("I need to go to the supermarket.", "Ich muss zum Supermarkt gehen."),
            ("She is reading an interesting book.", "Sie liest ein interessantes Buch."),
            ("We are going to the cinema tomorrow.", "Wir gehen morgen ins Kino."),
            ("He works in a large company.", "Er arbeitet in einer großen Firma."),

            # 不同时态和语气
            ("I have already eaten dinner.", "Ich habe bereits zu Abend gegessen."),
            ("They will arrive at 8 o'clock.", "Sie werden um 8 Uhr ankommen."),
            ("If I had time, I would travel.", "Wenn ich Zeit hätte, würde ich reisen."),
            ("She said that she was tired.", "Sie sagte, dass sie mü极de war."),

            # 更复杂的句子
            ("Despite the rain, we decided to go for a walk.",
             "Trotz des Regents entschieden wir uns, spazieren zu gehen."),
            ("The book that I bought yesterday is very interesting.",
             "Das Buch, das ich gestern gekauft habe, ist sehr interessant."),
        ]

        # 扩展数据集
        expanded_pairs = []
        for i in range(50):  # 重复50次以创建足够大的数据集
            for en, de in translation_pairs:
                # 添加一些变体使数据更丰富
                if i % 5 == 0:
                    de = de.replace(".", "!")
                elif i % 7 == 0:
                    de = de.replace("?", ".")

                expanded_pairs.append({
                    'translation': {
                        'en': f"{en} (Variation {i})",
                        'de': de
                    }
                })

        # 转换为Hugging Face数据集格式
        data_dict = {
            'translation': [
                {'en': pair['translation']['en'], 'de': pair['translation']['de']}
                for pair in expanded_pairs
            ]
        }

        return HFDataset.from_dict(data_dict)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            item = self.data[idx]

            # 处理不同的数据格式
            if 'translation极' in item:
                # HF datasets格式
                src_text = item['translation']['en']
                tgt_text = item['translation']['de']
            elif 'en' in item and 'de' in item:
                # 直接包含en/de字段
                src_text = item['en']
                tgt_text = item['de']
            else:
                # 尝试获取文本字段
                keys = list(item.keys())
                if len(keys) >= 2:
                    src_text = str(item[keys[0]])
                    tgt_text = str(item[keys[1]])
                else:
                    src_text = "Hello world"
                    tgt_text = "Hallo Welt"

            # 限制长度并确保是字符串
            src_text = str(src_text)[:self.max_length]
            tgt_text = str(tgt_text)[:self.max_length]

        except Exception as e:
            print(f"数据加载错误: {e}")
            src_text = "Error loading text"
            tgt_text = "Fehler beim Laden des Textes"

        return {
            'src_text': src_text,
            'tgt_text': tgt_text
        }


def build_vocab(texts, max_vocab_size=1000):
    """构建更丰富的词汇表"""
    # 收集所有字符
    chars = set()
    word_counts = {}

    for text in texts:
        # 字符级
        chars.update(str(text))
        # 简单的单词级（按空格分割）
        words = str(text).split()
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

    # 选择最常见的单词
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    top_words = [word for word, count in sorted_words[:max_vocab_size // 2]]

    # 合并字符和单词
    vocab_items = list(chars) + top_words

    # 创建词汇表
    vocab = {item: i + 2 for i, item in enumerate(vocab_items[:max_vocab_size - 2])}
    vocab['<pad>'] = 0
    vocab['<unk>'] = 1

    return vocab


def text_to_ids(text, vocab, max_length):
    """将文本转换为ID序列，支持字符和单词混合"""
    text_str = str(text)

    # 尝试单词级，如果单词不在词汇表中则回退到字符级
    tokens = []
    words = text_str.split()

    for word in words:
        if word in vocab:
            tokens.append(vocab[word])
        else:
            # 单词不在词汇表中，使用字符级
            for char in word[:5]:  # 限制单词长度
                if char in vocab:
                    tokens.append(vocab[char])

    # 如果单词级处理失败，使用纯字符级
    if not tokens:
        tokens = [vocab.get(char, vocab['<unk>']) for char in text_str[:max_length]]

    # 填充或截断
    if len(tokens) < max_length:
        tokens += [vocab['<pad>']] * (max_length - len(tokens))
    else:
        tokens = tokens[:max_length]

    return torch.tensor(tokens, dtype=torch.long)


def collate_fn(batch, src_vocab, tgt_vocab, max_length=64):
    src_texts = [item['src_text'] for item in batch]
    tgt_texts = [item['tgt_text'] for item in batch]

    # 转换为ID序列
    src_ids = torch.stack([text_to_ids(text, src_vocab, max_length) for text in src_texts])
    tgt_ids = torch.stack([text_to_ids(text, tgt_vocab, max_length) for text in tgt_texts])

    return src_ids, tgt_ids


def get_data_loaders(config):
    print("构建词汇表...")

    # 使用训练数据构建词汇表
    train_dataset = IWSLTDataset(config.data_paths['train'], config.max_seq_length, 'train')

    # 收集所有文本构建词汇表（采样更多数据）
    all_src_texts = []
    all_tgt_texts = []

    sample_size = min(2000, len(train_dataset))  # 增加采样数量
    for i in range(sample_size):
        item = train_dataset[i]
        all_src_texts.append(item['src_text'])
        all_tgt_texts.append(item['tgt_text'])

    src_vocab = build_vocab(all_src_texts, max_vocab_size=500)  # 增大词汇表
    tgt_vocab = build_vocab(all_tgt_texts, max_vocab_size=500)

    print(f"源语言词汇表大小: {len(src_vocab)}")
    print(f"目标语言词汇表大小: {len(tgt_vocab)}")

    # 创建数据加载器
    train_dataset = IWSLTDataset(config.data_paths['train'], config.max_seq_length, 'train')
    val_dataset = IWSLTDataset(config.data_paths['val'], config.max_seq_length, 'val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, src_vocab, tgt_vocab, config.max_seq_length)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, src_vocab, tgt_vocab, config.max_seq_length)
    )

    return train_loader, val_loader, src_vocab, tgt_vocab