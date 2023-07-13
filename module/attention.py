import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttentionRelative(nn.MultiheadAttention):
    """
    Multihead attention with relative positional encoding
    """

    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttentionRelative, self).__init__(embed_dim, num_heads, dropout=0.0, bias=True,
                                                         add_bias_kv=False, add_zero_attn=False,
                                                         kdim=None, vdim=None)

    def forward(self, query, key, value, attn_mask=None, pos_enc=None, pos_indexes=None):
        """
        Multihead attention

        :param query: [W,HN,C]
        :param key: [W,HN,C]
        :param value: [W,HN,C]
        :param attn_mask: mask to invalidate attention, -inf is used for invalid attention, [W,W]
        :param pos_enc: [2W-1,C]
        :param pos_indexes: index to select relative encodings, flattened in transformer WW
        :return: output value vector, attention with softmax (for debugging) and raw attention (used for last layer)
        """

        w, bsz, embed_dim = query.size()        # w, 2hN, 128
        head_dim = embed_dim // self.num_heads  # 128//8 = 16
        assert head_dim * self.num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # 先做线性变换
        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            # in_proj_weight (128*3,128), in_proj_bias (128*3)
            q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
            # (w,2hN,128)
        elif torch.equal(key, value):
            # cross-attention
            _b = self.in_proj_bias
            _start = 0
            _end = embed_dim
            _w = self.in_proj_weight[_start:_end, :]  # (128,128)
            if _b is not None:
                _b = _b[_start:_end]     # (128)
            q = F.linear(query, _w, _b)  # (w,2hN,128)

            if key is None:
                assert value is None
                k = None
                v = None
            else:
                _b = self.in_proj_bias
                _start = embed_dim
                _end = None
                _w = self.in_proj_weight[_start:, :]  # (128*2,128)
                if _b is not None:
                    _b = _b[_start:]  # (128*2)
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)  # (w,2hN,128)

        # 对 pos_enc 做相同的线性变换
        if pos_enc is not None:
            # reshape pos_enc
            pos_enc = torch.index_select(pos_enc, 0, pos_indexes).view(w, w, -1)  # (w,w,128)
            # compute q_r, k_r
            _start = 0
            _end = 2 * embed_dim
            _w = self.in_proj_weight[_start:_end, :]
            _b = self.in_proj_bias[_start:_end]
            q_r, k_r = F.linear(pos_enc, _w, _b).chunk(2, dim=-1)  # (w,w,128)
        else:
            q_r = None
            k_r = None

        # scale query
        scaling = float(head_dim) ** -0.5
        q = q * scaling
        if q_r is not None:
            q_r = q_r * scaling

        # reshape
        q = q.contiguous().view(w, bsz, self.num_heads, head_dim)  # (w,2hN,8,16)
        if k is not None:
            k = k.contiguous().view(-1, bsz, self.num_heads, head_dim)
        if v is not None:
            v = v.contiguous().view(-1, bsz, self.num_heads, head_dim)

        if q_r is not None:
            q_r = q_r.contiguous().view(w, w, self.num_heads, head_dim)  # (w,w,8,16)
        if k_r is not None:
            k_r = k_r.contiguous().view(w, w, self.num_heads, head_dim)

        # compute attn weight
        attn_feat = torch.einsum('wnec,vnec->newv', q, k)  # (2hN,8,w,w)

        # add positional terms
        if pos_enc is not None:
            # 公式 7
            attn_feat_pos = torch.einsum('wnec,wvec->newv', q, k_r)  # (2hN,8,w,w)
            attn_pos_feat = torch.einsum('vnec,wvec->newv', k, q_r)  # (2hN,8,w,w)
            attn = attn_feat + attn_feat_pos + attn_pos_feat
        else:
            attn = attn_feat

        assert list(attn.size()) == [bsz, self.num_heads, w, w]

        # apply attn mask
        if attn_mask is not None:
            # 图 5(b) 黑色部分 -inf, 白色部分 0
            attn_mask = attn_mask[None, None, ...]  # (1,1,w,w)
            attn += attn_mask

        # raw attn
        raw_attn = attn

        # softmax
        attn = F.softmax(attn, dim=-1)

        # compute v, equivalent to einsum('',attn,v),
        # need to do this because apex does not support einsum when precision is mixed
        # (16hN,w,w), (16hN,w,16) -> (16hN,w,16)
        v_o = torch.bmm(attn.view(bsz * self.num_heads, w, w),
                        v.permute(1, 2, 0, 3).view(bsz * self.num_heads, w, head_dim))
        assert list(v_o.size()) == [bsz * self.num_heads, w, head_dim]
        v_o = v_o.reshape(bsz, self.num_heads, w, head_dim).permute(2, 0, 1, 3).reshape(w, bsz, embed_dim)
        # (w,2hN,128)
        v_o = F.linear(v_o, self.out_proj.weight, self.out_proj.bias)  # (w,2hN,128)

        # average attention weights over heads
        attn = attn.sum(dim=1) / self.num_heads  # 好像没用

        # raw attn
        raw_attn = raw_attn.sum(dim=1)  # (2hN,w,w)

        return v_o, attn, raw_attn
