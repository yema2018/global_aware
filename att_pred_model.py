import torch
import torch.nn as nn
import numpy as np


def create_padding_mask(ori_mask):
    ori_mask = torch.eq(ori_mask, 0).type(torch.int)
    return ori_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)


def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    sines = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    cosines = np.cos(angle_rads[:, 1::2])

    pos_encoding = np.concatenate([sines, cosines], axis=-1)

    pos_encoding = pos_encoding[np.newaxis, ...]

    return torch.tensor(pos_encoding, dtype=torch.float32)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = torch.tensor(k.size()[-1], dtype=torch.float32)
    scaled_attention_logits = matmul_qk / torch.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e19)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = scaled_attention_logits.softmax(-1)  # (..., seq_len_q, seq_len_k)

    output = torch.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)

    def forward(self, v, k, q, mask):
        batch_size = q.size()[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = scaled_attention.transpose(1, 2) # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model) # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, torch.mean(attention_weights, dim=1)


def point_wise_feed_forward_network(d_model, dff):
    return nn.Sequential(nn.Linear(d_model, dff),
                         nn.ReLU(),
                         nn.Linear(dff, d_model))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class PreAttModel(nn.Module):
    def __init__(self, layers, d_model, num_heads, dff, rate):
        super(PreAttModel, self).__init__()
        self.layer = layers
        self.m = nn.ModuleList([EncoderLayer(d_model, num_heads, dff, rate) for _ in range(layers)])
        self.out_layer = nn.Linear(d_model, 1)
        # self.pos_encoding = positional_encoding(10000, d_model)
        # self.layernorm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(rate)
        self.pos_enc = nn.Parameter(torch.zeros([1, 1024, d_model]), requires_grad=True)

    def forward(self, inp, mask):
        seq_len = inp.size()[1]
        mask = create_padding_mask(mask)

        h = inp + self.pos_enc[:, :seq_len, :]
        h = self.dropout(h)
        for i in range(self.layer):
            h = self.m[i](h, mask)

        mask = mask.squeeze(1).squeeze(1)
        h = self.out_layer(h)
        logits = h.squeeze(-1)
        logits += mask * -1e19
        logits = torch.exp(logits)

        return logits  # shape == (batch_size, seq_len)
