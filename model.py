import torch
import torch.nn as nn
import math


'''
    Input Embedding层
    将文字序列映射为向量矩阵（将一个数字固定映射为一个向量，这个映射的过程是可以通过训练学习出来的）
'''
class InputEmbeddings(nn.Module):
    """
        d_model: 向量维度大小
        vocab_size: 字典大小
    """
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    '''
        将一个数字（字典索引）固定映射为一个向量（维度为参数d_model），这个映射的过程是可以通过训练学习出来的,
        math.sqrt(self.d_model)这个系数是论文里说明就是这么相乘用的
    '''
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """
        d_model: 向量维度大小
        seq_len: 最大序列长度
        dropout: 随机失活（避免过拟合）
    """
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1) seq_len行，1列
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) #(seq_len, 1) seq_len行，1列
        # Create a vector of shape (1, d_model)，位置编码公式中的分母
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply sine to even indices,应用到偶数列
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices，应用到奇数列
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add a batch dimension to the positional encoding, 将位置编码添加一个维度,2D变3D
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer，将位置编码注册为缓冲区，pe是不变的，所以注册为缓冲区
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add the positional encoding to the input, 将位置编码添加到输入中
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)