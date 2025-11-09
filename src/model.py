import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, use_attention=True):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.use_attention = use_attention

        if use_attention:
            self.w_q = nn.Linear(d_model, d_model)
            self.w_k = nn.Linear(d_model, d_model)
            self.w_v = nn.Linear(d_model, d_model)
            self.w_o = nn.Linear(d_model, d_model)
        else:
            # 如果没有注意力，使用简单的线性变换
            self.linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        if not self.use_attention:
            # 不使用注意力机制，直接返回线性变换
            return self.linear(query)

        batch_size = query.size(0)

        # Linear projections
        Q = self.w_q(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        return self.w_o(attn_output)


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1,
                 use_residual=True, use_layernorm=True, use_attention=True):
        super().__init__()
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm

        self.self_attn = MultiHeadAttention(d_model, nhead, dropout, use_attention)
        self.ffn = PositionWiseFFN(d_model, dim_feedforward, dropout)

        if use_layernorm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention
        attn_output = self.self_attn(x, x, x, mask)

        # 残差连接和层归一化
        if self.use_residual:
            x = x + self.dropout(attn_output)
        else:
            x = self.dropout(attn_output)

        if self.use_layernorm:
            x = self.norm1(x)

        # FFN
        ffn_output = self.ffn(x)

        # 残差连接和层归一化
        if self.use_residual:
            x = x + self.dropout(ffn_output)
        else:
            x = self.dropout(ffn_output)

        if self.use_layernorm:
            x = self.norm2(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1,
                 use_residual=True, use_layernorm=True, use_attention=True):
        super().__init__()
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm

        self.self_attn = MultiHeadAttention(d_model, nhead, dropout, use_attention)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout, use_attention)
        self.ffn = PositionWiseFFN(d_model, dim_feedforward, dropout)

        if use_layernorm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        # Self-attention with masking for decoder
        self_attn_output = self.self_attn(x, x, x, tgt_mask)

        if self.use_residual:
            x = x + self.dropout(self_attn_output)
        else:
            x = self.dropout(self_attn_output)

        if self.use_layernorm:
            x = self.norm1(x)

        # Cross-attention with encoder output
        cross_attn_output = self.cross_attn(x, memory, memory, src_mask)

        if self.use_residual:
            x = x + self.dropout(cross_attn_output)
        else:
            x = self.dropout(cross_attn_output)

        if self.use_layernorm:
            x = self.norm2(x)

        # FFN
        ffn_output = self.ffn(x)

        if self.use_residual:
            x = x + self.dropout(ffn_output)
        else:
            x = self.dropout(ffn_output)

        if self.use_layernorm:
            x = self.norm3(x)

        return x


class Transformer(nn.Module):
    def __init__(self, config, src_vocab_size=1000, tgt_vocab_size=1000):
        super().__init__()
        self.config = config

        # 使用传入的词汇表大小
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        # 源语言和目标语言使用相同的embedding（简化处理）
        vocab_size = max(src_vocab_size, tgt_vocab_size)
        self.embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=0)

        if config.use_positional_encoding:
            self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_length)
        else:
            self.pos_encoding = None

        # 编码器
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                config.d_model, config.nhead, config.dim_feedforward, config.dropout,
                config.use_residual_connections, config.use_layernorm, config.use_multihead_attention
            )
            for _ in range(config.num_encoder_layers)
        ])

        # 解码器
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                config.d_model, config.nhead, config.dim_feedforward, config.dropout,
                config.use_residual_connections, config.use_layernorm, config.use_multihead_attention
            )
            for _ in range(config.num_decoder_layers)
        ])

        # 输出投影层使用目标语言词汇表大小
        self.output_proj = nn.Linear(config.d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        batch_size, src_seq_len = src.size()
        batch_size, tgt_seq_len = tgt.size()

        # Embedding
        src_embedded = self.embedding(src) * math.sqrt(self.config.d_model)
        tgt_embedded = self.embedding(tgt) * math.sqrt(self.config.d_model)

        # 位置编码
        if self.pos_encoding is not None:
            src_embedded = self.dropout(self.pos_encoding(src_embedded.transpose(0, 1))).transpose(0, 1)
            tgt_embedded = self.dropout(self.pos_encoding(tgt_embedded.transpose(0, 1))).transpose(0, 1)
        else:
            src_embedded = self.dropout(src_embedded)
            tgt_embedded = self.dropout(tgt_embedded)

        # 编码器前向传播
        memory = src_embedded
        for layer in self.encoder_layers:
            memory = layer(memory, src_mask)

        # 解码器前向传播
        output = tgt_embedded
        for layer in self.decoder_layers:
            output = layer(output, memory, src_mask, tgt_mask)

        return self.output_proj(output)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask