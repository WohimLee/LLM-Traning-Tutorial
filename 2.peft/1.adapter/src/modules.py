import math
import torch
import torch.nn as nn
# --------------------------
# Model Components
# --------------------------
class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, key_padding_mask=None):
        # q,k,v: (B,T,D)
        B, Tq, D = q.size()
        Tk = k.size(1)

        Q = self.q_proj(q).view(B, Tq, self.num_heads, self.d_k).transpose(1, 2)  # (B,H,Tq,d_k)
        K = self.k_proj(k).view(B, Tk, self.num_heads, self.d_k).transpose(1, 2)  # (B,H,Tk,d_k)
        V = self.v_proj(v).view(B, Tk, self.num_heads, self.d_k).transpose(1, 2)  # (B,H,Tk,d_k)

        logits = (Q @ K.transpose(-1, -2)) / math.sqrt(self.d_k)  # (B,H,Tq,Tk)

        # causal mask: (Tq,Tk)
        causal_mask = torch.tril(torch.ones(Tq, Tk, device=logits.device, dtype=torch.bool))

        # padding mask：key_padding_mask (B,Tk) True=有效
        if key_padding_mask is not None:
            # (B,1,1,Tk)
            k_mask = key_padding_mask.unsqueeze(1).unsqueeze(1).to(torch.bool)
            mask = causal_mask.unsqueeze(0).unsqueeze(0) & k_mask  # (B,1,Tq,Tk)
        else:
            mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1,1,Tq,Tk)

        logits = logits.masked_fill(~mask, -1e9)
        scores = torch.softmax(logits, dim=-1)
        out = scores @ V  # (B,H,Tq,d_k)

        out = out.transpose(1, 2).contiguous().view(B, Tq, D)  # (B,Tq,D)
        return self.dropout(self.o_proj(out))

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.gama = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x, eps=1e-5):
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        out = (x - mean) / torch.sqrt(var + eps)
        return self.gama * out + self.beta

class FeedForward(nn.Module):
    def __init__(self, d_model, dff, dropout):
        super().__init__()
        self.W1 = nn.Linear(d_model, dff)
        self.W2 = nn.Linear(dff, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.W1(x)
        x = self.act(x)
        x = self.W2(x)
        return self.dropout(x)

class Adapter(nn.Module):
    """
    Houlsby Adapter: bottleneck MLP + residual
    x -> down -> act -> up -> dropout -> +x
    """
    def __init__(self, d_model, bottleneck=64, dropout=0.1):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, d_model)
        self.dropout = nn.Dropout(dropout)

        # 让 adapter 初始≈0，不破坏 base
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.dropout(self.up(self.act(self.down(x))))

# --------------------------
# Blocks / Models
# --------------------------
class GPTBlockBase(nn.Module):
    """Base block: no adapters."""
    def __init__(self, d_model, num_heads, dff, dropout):
        super().__init__()
        self.mha = MultiheadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, dff, dropout)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

    def forward(self, x, key_padding_mask):
        x0 = x
        x = self.mha(x, x, x, key_padding_mask=key_padding_mask)
        x = self.ln1(x0 + x)

        x0 = x
        x = self.ffn(x)
        x = self.ln2(x0 + x)
        return x

class GPTBlockAdapter(nn.Module):
    """Adapter block: Houlsby style (post-attn adapter + post-ffn adapter)."""
    def __init__(self, d_model, num_heads, dff, dropout, adapter_bottleneck=64):
        super().__init__()
        self.mha = MultiheadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, dff, dropout)

        self.attn_adapter = Adapter(d_model, bottleneck=adapter_bottleneck, dropout=dropout)
        self.ffn_adapter = Adapter(d_model, bottleneck=adapter_bottleneck, dropout=dropout)

        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

    def forward(self, x, key_padding_mask):
        x0 = x
        x = self.mha(x, x, x, key_padding_mask=key_padding_mask)
        x = self.attn_adapter(x)
        x = self.ln1(x0 + x)

        x0 = x
        x = self.ffn(x)
        x = self.ffn_adapter(x)
        x = self.ln2(x0 + x)
        return x

class GPTBase(nn.Module):
    """Pretrain base model (no adapters)."""
    def __init__(self, vocab_size, d_model, max_length, num_heads, dff, dropout, N):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_length, d_model)
        self.blocks = nn.ModuleList([GPTBlockBase(d_model, num_heads, dff, dropout) for _ in range(N)])
        self.cls = nn.Linear(d_model, vocab_size)
        self.register_buffer("pos_range", torch.arange(max_length).view(1, -1), persistent=False)

    def forward(self, input_ids, key_padding_mask=None):
        B, T = input_ids.size()
        x = self.emb(input_ids) + self.pos(self.pos_range[:, :T])
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        return self.cls(x)

class GPTWithAdapters(nn.Module):
    """Adapter model for finetune. Load base weights with strict=False."""
    def __init__(self, vocab_size, d_model, max_length, num_heads, dff, dropout, N, adapter_bottleneck=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_length, d_model)
        self.blocks = nn.ModuleList(
            [GPTBlockAdapter(d_model, num_heads, dff, dropout, adapter_bottleneck=adapter_bottleneck) for _ in range(N)]
        )
        self.cls = nn.Linear(d_model, vocab_size)
        self.register_buffer("pos_range", torch.arange(max_length).view(1, -1), persistent=False)

    def forward(self, input_ids, key_padding_mask=None):
        B, T = input_ids.size()
        x = self.emb(input_ids) + self.pos(self.pos_range[:, :T])
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        return self.cls(x)