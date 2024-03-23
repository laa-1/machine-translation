import math
import torch.nn

from parameter import *


class PositionalEncoding(torch.nn.Module):
    """位置编码器"""

    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        # 初始化Shape为(max_len, d_model)的PE(positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor[[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 创建一个包含指数项的张量，用于后续计算位置编码中的奇数和偶数位置的值
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        # 计算PE(pos, 2i)，得到位置编码中偶数位置的值，通过将sin函数应用于位置和指数项的乘积
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)，得到位置编码中奇数位置的值，通过将cos函数应用于位置和指数项的乘积
        pe[:, 1::2] = torch.cos(position * div_term)
        # 在位置编码张量的最外维度上添加一个额外的维度，用于处理batch数据
        pe = pe.unsqueeze(0)
        # 不会参与梯度更新，但在保存和加载模型时会一并保存和加载
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 将位置编码加入到词向量中，x为embedding后的inputs, 形状为(batch_size, vocab_size, word_vector_size)
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def get_key_padding_mask(tokens):
    return tokens == pad_token_id


class TranslationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.src_embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=word_vector_size, padding_idx=pad_token_id)
        self.tgt_embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=word_vector_size, padding_idx=pad_token_id)
        self.positional_encoding = PositionalEncoding(word_vector_size, dropout=0.1)
        self.transformer = torch.nn.Transformer(d_model=word_vector_size, dropout=0.1, batch_first=True)
        self.linear = torch.nn.Linear(word_vector_size, vocab_size)

    def forward(self, src, tgt):
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1]).to(device)
        # 获取mark
        src_key_padding_mask = get_key_padding_mask(src)
        tgt_key_padding_mask = get_key_padding_mask(tgt)
        # 对src和tgt进行编码，将token索引映射到高维当中，也就是用词向量来代替表示每个词，(batch_size, tokens_max_length) -> (batch_size, tokens_max_length, word_vector_size)
        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)
        # 给src和tgt的词向量增加位置信息，(batch_size, tokens_max_length, word_vector_size)不变
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        # 传入到transformer的模块中，输出为从实际目标序列中第二个token开始到最后一个token的预测结果，(batch_size, tokens_max_length, word_vector_size)不变
        out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        if self.training:
            # 每个token的预测将预测结果映射到一个多分类中，也就是目标序列中，该位置的是某个token的概率，(batch_size, tokens_max_length, vocab_size)
            # 由于计torch.nn.CrossEntropyLoss()包含了softmax计算，这里不必再加softmax
            out = self.linear(out)
        else:
            # 推理时，只需要最后一个词的预测结果
            # 推理时，只需要取值最大的作为结果即可，也不需要softmax
            out = self.linear(out[:, -1])
        return out
