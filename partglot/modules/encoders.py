from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


class LSTM(nn.Module):
    def __init__(self, text_dim, embedding_dim, vocab_size, padding_idx=0):
        super().__init__()
        self.padding_idx = padding_idx
        self.word_embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        self.rnn = nn.LSTM(embedding_dim, text_dim, batch_first=True)
        self.w_attn = nn.Parameter(torch.Tensor(1, text_dim))
        nn.init.xavier_uniform_(self.w_attn)

    def forward(self, padded_tokens, dropout=0.5):
        w_emb = self.word_embedding(padded_tokens)
        w_emb = F.dropout(w_emb, dropout, self.training)
        len_seq = (padded_tokens != self.padding_idx).sum(dim=1).cpu()
        x_packed = pack_padded_sequence(
            w_emb, len_seq, enforce_sorted=False, batch_first=True
        )

        B = padded_tokens.shape[0]

        rnn_out, _ = self.rnn(x_packed)
        rnn_out, dummy = pad_packed_sequence(rnn_out, batch_first=True)
        h = rnn_out[torch.arange(B), len_seq - 1]
        final_feat, attn = self.word_attention(rnn_out, h, len_seq)
        return final_feat, attn

    def word_attention(self, R, h, len_seq):
        """
        Input:
            R: hidden states of the entire words
            h: the final hidden state after processing the entire words
            len_seq: the length of the sequence
        Output:
            final_feat: the final feature after the bilinear attention
            attn: word attention weights
        """
        B, N, D = R.shape
        device = R.device
        len_seq = len_seq.to(device)

        W_attn = (self.w_attn * torch.eye(D).to(device))[None].repeat(B, 1, 1)
        score = torch.bmm(torch.bmm(R, W_attn), h.unsqueeze(-1))

        mask = torch.arange(N).reshape(1, N, 1).repeat(B, 1, 1).to(device)
        mask = mask < len_seq.reshape(B, 1, 1)

        score = score.masked_fill(mask == 0, -1e9)
        attn = F.softmax(score, 1)
        final_feat = torch.bmm(R.transpose(1, 2), attn).squeeze(-1)

        return final_feat, attn.squeeze(-1)


class SupSegsEncoder(nn.Module):
    def __init__(self, sup_segs_dim):
        super().__init__()
        dim = sup_segs_dim
        self.conv1 = nn.Sequential(nn.Conv1d(3, dim, 1), nn.BatchNorm1d(dim))
        self.conv2 = nn.Sequential(nn.Conv1d(dim, dim, 1), nn.BatchNorm1d(dim))
        self.conv3 = nn.Sequential(nn.Conv1d(dim, dim, 1), nn.BatchNorm1d(dim))
        self.conv4 = nn.Sequential(nn.Conv1d(dim, dim, 1), nn.BatchNorm1d(dim))
        self.fc = nn.Conv1d(dim, dim, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(out1))
        out3 = F.relu(self.conv3(out2))
        out4 = F.relu(self.conv4(out3))

        out = self.fc(out4)
        out = out.transpose(1, 2)

        return out


class CrossAttention(nn.Module):
    def __init__(self, src_dim, tgt_dim):
        super().__init__()
        feedforward_dim = src_dim * 2

        d_q = d_k = src_dim
        d_v = tgt_dim

        self.to_q = nn.Linear(src_dim, d_q, bias=False)
        self.to_k = nn.Linear(tgt_dim, d_k, bias=False)
        self.to_v = nn.Linear(tgt_dim, d_v, bias=False)

        self.dp1 = nn.Dropout(0.1)
        self.dp2 = nn.Dropout(0.1)

        self.linear1 = nn.Linear(src_dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, src_dim)

        self.norm = nn.LayerNorm(src_dim)

    def _forward_attention(self, src, tgt, tgt_mask, double_softmax=False):
        """
        src: [B, len_src, D]
        tgt: [B, len_tgt, D]
        tgt_mask: [B, len_tgt]
        double_softmax (if True): applies softmax along src as well before applying softmax along tgt.
        """
        outputs = dict()

        q, k, v = self.to_q(src), self.to_k(tgt), self.to_v(tgt)

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        v = F.normalize(v, p=2, dim=-1)

        score = torch.bmm(q, k.transpose(1, 2))

        if double_softmax:
            assert score.shape[1] > 1
            score = F.softmax(score, 1)
            outputs["attn_along_pn"] = score
        else:
            outputs["attn_along_pn"] = None

        if tgt_mask is not None:
            tgt_mask = tgt_mask.unsqueeze(1)
            score = score.masked_fill(tgt_mask == 0, -1e9)

        attn = F.softmax(score, 2)

        outputs["q"], outputs["k"], outputs["v"] = q, k, v
        outputs["attn"] = attn

        return outputs

    def forward(
        self,
        src,
        tgt,
        tgt_mask,
        part_indicator: Optional[torch.Tensor] = None,
        double_softmax=False,
        return_only_attn=False,
    ):
        """
        src: [B, len_src, D]
        tgt: [B, len_tgt, D]
        tgt_mask: [B, len_tgt]
        part_indicator: [B, len(part_names)]
        """
        assert src.shape[0] == tgt.shape[0]
        if double_softmax and not return_only_attn:
            assert part_indicator is not None

        o = self._forward_attention(src, tgt, tgt_mask, double_softmax)
        v, attn = o["v"], o["attn"]
        if return_only_attn:
            return None, attn, None

        attn_along_pn = o["attn_along_pn"]

        attn_dp = self.dp1(attn)

        if double_softmax:
            attn_slice = attn_dp * part_indicator.unsqueeze(-1)
            attn_slice = torch.sum(attn_slice, 1, keepdim=True)
            attn_dp = attn_slice

        attn_f = torch.bmm(attn_dp, v)

        attn_f2 = self.linear2(F.relu(self.linear1(attn_f)))
        attn_f = attn_f + self.dp2(attn_f2)
        attn_f = self.norm(attn_f)

        return attn_f, attn, attn_along_pn


class MLP(nn.Module):
    def __init__(self, input_dim, out_channels=[100, 50, 1]):
        super().__init__()

        previous_feat_dim = input_dim
        all_ops = []

        for depth in range(len(out_channels)):
            out_dim = out_channels[depth]
            affine_op = nn.Linear(previous_feat_dim, out_dim, bias=True)
            all_ops.append(affine_op)

            if depth < len(out_channels) - 1:
                all_ops.append(nn.BatchNorm1d(out_dim))
                all_ops.append(nn.ReLU(inplace=True))

            previous_feat_dim = out_dim

        self.net = nn.Sequential(*all_ops)

    def forward(self, x):
        return self.net(x)
