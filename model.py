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

'''
位置编码
将文字序列中的每一个单词（token标志）在整个序列中的位置信息编码进单词的向量结果中
'''
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

'''
层归一化
'''
class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))    # alpha is a learnable parameter(Multiplied)
        self.bias = nn.Parameter(torch.zeros(1))    # bias is a learnable parameter(Added)

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        # Keep the dimension for broadcasting
        mean = x.mean(dim=-1, keepdim=True)
        # Keep the dimension for broadcasting
        std = x.std(dim=-1, keepdim=True)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

'''
前向全连接网络
'''
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)    # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)    # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        x = self.linear_2(self.dropout(torch.relu(self.linear_1(x))))   # 这里其实就是对论文里关于前向网络的公式进行实现

'''
多头注意力机制的具体实现
'''
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # Embedding vector size
        self.h = h  # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head 每一个头的向量维度
        self.w_q = nn.Linear(d_model, d_model)  # 对应多头注意力block中的第一个输入要乘的矩阵 Wq
        self.w_k = nn.Linear(d_model, d_model)  # 对应多头注意力block中的第二个输入要乘的矩阵 Wk
        self.w_v = nn.Linear(d_model, d_model)  # 对应多头注意力block中的第三个输入要乘的矩阵 Wv

        self.w_o = nn.Linear(d_model, d_model)  # 多头concat之后的输出矩阵
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            # 作用就是人为的控制token之间的关联关系，比如字符序列中，我不希望字符a和字符b之间有关联，
            # 那么设置矩阵中对应位置的值为接近0的数值，那么后续通过softmax后，值为接近0的
            attention_scores.masked_fill_(mask == 0, -1e9)
        # (batch, h, seq_len, seq_len) # Apply softmax，这里其实就能看到单词token之间的关联程度（分数了）
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)     # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)       # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v)     # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # 这里就是将q、k、v进行了多头的切分工作，即对d_model进行切分，分成h头，每一个头的维度为d_k
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention 在每一个头上分别运用注意力公式
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Combine all the heads together，将多个头的注意力结果合并
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo，将合并后的结果乘以矩阵W_o，论文里的公式要求最后这里要乘一下
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)

'''
残差连接，其实就是论文里用于在多个层之间进行跳跃连接的那个机制
'''
class ResidualConnection(nn.Module):

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        # 一次跳跃，Add + Norm操作，实操上和论文有点区别，论文里先过一层sublayer，再过Norm，
        # 这里是先过Norm，再过sublayer，看了很多复现工程，都是样的，我们也这样实现
        return x + self.dropout(sublayer(self.norm(x)))

'''
编码器中的一个block（单元），内含两个残差连接，一个多头注意力block，一个前向连接block
'''
class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) ->  None:
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # 第一个残差连接，多头注意力block
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        # 第二个残差连接，前向连接block
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

'''
编码器完整实现，内含N个EncoderBlock
'''
class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        # 过N个EncoderBlock
        for layer in self.layers:
            x = layer(x, mask)
        # 工程实现上最后过一道Norm
        return self.norm(x)

'''
解码器中的一个block，内含一个自多头注意力block，一个交叉多头注意力block，三个残差连接，一个前馈连接
'''
class DecoderBlock(nn.Module):

    def __int__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__int__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    '''
        x:解码器的输入
        encoder_output：编码器的输出，会作为解码器内交叉多头注意力block的k、v输入
        src_mask：编码器的mask
        tgt_mask：解码器的mask
    '''
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # 第一个残差连接，自多头注意力block
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        # 第二个残差连接，交叉多头注意力block
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        # 第三个残差连接，前向连接block
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

'''
解码器完整实现，内含N个DecoderBlock
'''
class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # 过N个DecoderBlock
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        # 工程实现上最后过一道Norm
        return self.norm(x)


