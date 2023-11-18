import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq len, d_model)   simplification of the formula for numerical stability
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # apply sin to even indices in the array
        pe[:, 0::2] = torch.sin(position * div_term)
        # apply cos to odd indices in the array
        pe[:, 1::2] = torch.cos(position * div_term)
        # add a batch dimension
        pe = pe.unsqueeze(0)
        # save the pe with the model
        self.register_buffer("pe", pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * (self.pe[:, : x.size(1), :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedforwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # w1 and b1 from paper
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)  # w2 and b2

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn = None

    @staticmethod
    def attention(self, q, k, v, d_k, mask=None, dropout: nn.Dropout = None):
        d_k = q.size(-1)
        # batch_size, h, seq_len, d_k => batch_size, h, seq_len, seq_len
        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = F.softmax(
            attention_scores, dim=-1
        )  # batch_size, h, seq_len, seq_len
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        output = attention_scores @ v
        return output, attention_scores  # attention_scores for visualization

    def forward(self, q, k, v, mask=None):
        query = self.w_q(
            q
        )  # batch_size, seq_len, d_model=> batch_size, seq_len, d_model
        key = self.w_k(k)
        value = self.w_v(v)

        # split the query, key and value into h different heads
        # batch_size, seq_len, d_model => batch_size, seq_len, h, d_k => batch_size, h, seq_len, d_k
        query = query.view(query.size(0), query.size(1), self.h, self.d_k).transpose(
            1, 2
        )  # transpose(1,2) => batch_size, h, seq_len, d_k

        key = key.view(key.size(0), key.size(1), self.h, self.d_k).transpose(1, 2)
        value = value.view(value.size(0), value.size(1), self.h, self.d_k).transpose(
            1, 2
        )

        x, self.attn = self.attention(query, key, value, self.d_k, mask, self.dropout)
        # batch_size, h, seq_len, d_k => batch_size, seq_len, h, d_k => batch_size, seq_len, d_model
        x = x.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.h * self.d_k)
        # batch_size, seq_len, d_model
        x = self.w_o(x)
        return x


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        d_model: int,
        h: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttentionBlock(d_model, h, dropout)
        self.feedforward = FeedforwardBlock(d_model, d_ff, dropout)
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.attention(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feedforward)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feedforward: FeedforwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention = self_attention_block
        self.cross_attention = cross_attention_block
        self.feedforward = FeedforwardBlock
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](
            x, lambda x: self.attention(x, x, x, tgt_mask)
        )  # self attention
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask),
        )  # cross attention
        x = self.residual_connections[2](x, self.feedforward)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(
    nn.Module
):  # reverse of embedding layer to vocab size # d_model => vocab_size
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        tgt_embed: InputEmbeddings,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        return self.encoder(self.src_pos(self.src_embed(src)), src_mask)

    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        return self.decoder(
            self.tgt_pos(self.tgt_embed(tgt)), encoder_output, src_mask, tgt_mask
        )

    def project(self, x):
        return self.generator(x)


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff=2048,
) -> Transformer:
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    # postional encoding
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        encoder_block = EncoderBlock(
            encoder_self_attention_block, d_model, h, d_ff, dropout
        )
        encoder_blocks.append(encoder_block)
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    # build decoder
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_feedforward_block = FeedforwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            decoder_self_attention_block,
            decoder_cross_attention_block,
            decoder_feedforward_block,
            dropout,
        )
        decoder_blocks.append(decoder_block)
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # build projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    # build transformer
    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer
    )
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer


if __name__ == "__main__":
    transformer = build_transformer(100, 100, 100, 100)
    print(transformer)
