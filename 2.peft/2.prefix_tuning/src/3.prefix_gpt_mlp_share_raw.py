# prefix_shared_mlp_tuning_gpt_poetry_full.py
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

'''
更省参数的 Prefix-MLP：
- 共享一个 MLP：d_model -> hidden -> hidden
- 每层只保留一个很小的线性投影：hidden -> 2 * num_heads * d_k（输出该层 K/V）
- 这样每层不再有一整套 MLP，只剩一个小投影层，参数会明显少。

两阶段：Base 全量训练 → 共享 Prefix-MLP 微调
'''

# -----------------------
# Repro / device
# -----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Data
# -----------------------
def read_data(file, num=None):
    with open(file, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
    lines = [x.strip() for x in lines if x.strip()]
    if num is not None:
        lines = lines[:num]
    return lines

def tokenize(all_poetries):
    vocab = {"[pad]": 0, "[unk]": 1, "[sos]": 2, "[eos]": 3, "，": 4, "。": 5}
    for line in all_poetries:
        for ch in line:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab

class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.idx2word = [None] * len(vocab)
        for w, i in vocab.items():
            self.idx2word[i] = w

        self.pad_token_id = vocab["[pad]"]
        self.unk_token_id = vocab["[unk]"]
        self.sos_token_id = vocab["[sos]"]
        self.eos_token_id = vocab["[eos]"]

    def encode(self, text, add_special_tokens=False, padding="max_length", max_length=None, truncation=False):
        ids = [self.vocab.get(ch, self.unk_token_id) for ch in text]
        if add_special_tokens:
            ids = [self.sos_token_id] + ids
        if truncation and max_length is not None:
            ids = ids[:max_length]
        if padding == "max_length" and max_length is not None:
            if len(ids) < max_length:
                ids = ids + [self.pad_token_id] * (max_length - len(ids))
            else:
                ids = ids[:max_length]
        return ids

    def decode(self, ids):
        return [self.idx2word[i] if 0 <= i < len(self.idx2word) else "[unk]" for i in ids]

class Poetry(Dataset):
    def __init__(self, all_poetries, tokenizer: Tokenizer, max_length):
        self.all_poetries = all_poetries
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        text = self.all_poetries[idx]
        input_ids = self.tokenizer.encode(
            text=text,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        key_padding_mask = torch.tensor(input_ids, dtype=torch.long) != self.tokenizer.pad_token_id  # True=valid
        labels = input_ids[1:] + [self.tokenizer.eos_token_id]
        labels = labels[:self.max_length]
        if len(labels) < self.max_length:
            labels += [self.tokenizer.pad_token_id] * (self.max_length - len(labels))
        return input_ids, labels, key_padding_mask

    def __len__(self):
        return len(self.all_poetries)

def collate_fn(batch):
    input_ids, labels, key_padding_mask = zip(*batch)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "key_padding_mask": torch.stack(key_padding_mask).bool(),
    }

# -----------------------
# Model parts
# -----------------------
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

    def forward(self, q, k, v, key_padding_mask=None, prefix_k=None, prefix_v=None):
        B, Tq, _ = q.size()
        Tk = k.size(1)

        Q = self.q_proj(q).view(B, Tq, self.num_heads, self.d_k).transpose(1, 2)  # (B,H,Tq,d_k)
        K = self.k_proj(k).view(B, Tk, self.num_heads, self.d_k).transpose(1, 2)  # (B,H,Tk,d_k)
        V = self.v_proj(v).view(B, Tk, self.num_heads, self.d_k).transpose(1, 2)  # (B,H,Tk,d_k)

        P = 0
        if prefix_k is not None and prefix_v is not None:
            P = prefix_k.size(2)
            K = torch.cat([prefix_k, K], dim=2)  # (B,H,P+Tk,d_k)
            V = torch.cat([prefix_v, V], dim=2)

        logits = (Q @ K.transpose(-1, -2)) / math.sqrt(self.d_k)  # (B,H,Tq,P+Tk)
        device = logits.device

        if P > 0:
            prefix_allow = torch.ones(B, 1, Tq, P, device=device, dtype=torch.bool)
        else:
            prefix_allow = None

        causal = torch.tril(torch.ones(Tq, Tk, device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)

        if key_padding_mask is None:
            key_padding_mask = torch.ones(B, Tk, device=device, dtype=torch.bool)
        else:
            key_padding_mask = key_padding_mask.to(device=device, dtype=torch.bool)

        k_valid = key_padding_mask.unsqueeze(1).unsqueeze(1)  # (B,1,1,Tk)
        real_allow = causal & k_valid                          # (B,1,Tq,Tk)

        if P > 0:
            allow = torch.cat([prefix_allow, real_allow], dim=-1)
        else:
            allow = real_allow

        q_valid = key_padding_mask[:, :Tq].unsqueeze(1).unsqueeze(-1)
        allow = allow & q_valid

        logits = logits.masked_fill(~allow, -1e9)
        scores = torch.softmax(logits, dim=-1)

        out = scores @ V  # (B,H,Tq,d_k)
        out = out.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        return self.dropout(self.o_proj(out))

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.gama = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x, eps=1e-5):
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        return self.gama * (x - mean) / torch.sqrt(var + eps) + self.beta

class FeedForward(nn.Module):
    def __init__(self, d_model, dff, dropout):
        super().__init__()
        self.W1 = nn.Linear(d_model, dff)
        self.W2 = nn.Linear(dff, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.W2(self.act(self.W1(x))))

# -----------------------
# Shared Prefix-MLP (more parameter-efficient)
# -----------------------
class PrefixSharedMLP(nn.Module):
    """
    更省参数版本：
      prefix_tokens: (P, D)  trainable
      shared_mlp:    D -> hidden -> hidden   (shared across layers)
      per_layer_proj[l]: hidden -> (2*H*d_k) (small per layer)

    Output per layer: prefix_k/prefix_v in head space (B,H,P,d_k)
    """
    def __init__(self, num_layers, d_model, num_heads, prefix_len, hidden=256, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.prefix_len = prefix_len

        self.prefix_tokens = nn.Parameter(torch.randn(prefix_len, d_model) * 0.02)
        self.drop = nn.Dropout(dropout)

        # shared trunk
        self.shared_mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        # tiny head per layer
        self.per_layer_proj = nn.ModuleList([
            nn.Linear(hidden, 2 * num_heads * self.d_k) for _ in range(num_layers)
        ])

    def forward(self, batch_size, device=None, dtype=None):
        """
        Return list length L:
          prefixes[l] = (prefix_k, prefix_v) each of shape (B,H,P,d_k)
        """
        z = self.drop(self.prefix_tokens)  # (P,D)
        if device is not None:
            z = z.to(device)
        if dtype is not None:
            z = z.to(dtype)

        h = self.shared_mlp(z)  # (P,hidden)

        prefixes = []
        for proj in self.per_layer_proj:
            out = proj(h)  # (P, 2*H*d_k)
            out = out.view(self.prefix_len, 2, self.num_heads, self.d_k)  # (P,2,H,d_k)
            out = out.permute(1, 2, 0, 3).contiguous()                    # (2,H,P,d_k)
            pk, pv = out[0], out[1]                                       # (H,P,d_k)
            pk = pk.unsqueeze(0).expand(batch_size, -1, -1, -1)           # (B,H,P,d_k)
            pv = pv.unsqueeze(0).expand(batch_size, -1, -1, -1)
            prefixes.append((pk, pv))
        return prefixes

# -----------------------
# Blocks / Models
# -----------------------
class GPTBlockBase(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout):
        super().__init__()
        self.mha = MultiheadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, dff, dropout)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

    def forward(self, x, key_padding_mask):
        x0 = x
        x = self.mha(x, x, x, key_padding_mask=key_padding_mask, prefix_k=None, prefix_v=None)
        x = self.ln1(x0 + x)

        x0 = x
        x = self.ffn(x)
        x = self.ln2(x0 + x)
        return x

class GPTBlockPrefix(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout):
        super().__init__()
        self.mha = MultiheadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, dff, dropout)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

    def forward(self, x, key_padding_mask, prefix_k, prefix_v):
        x0 = x
        x = self.mha(x, x, x, key_padding_mask=key_padding_mask, prefix_k=prefix_k, prefix_v=prefix_v)
        x = self.ln1(x0 + x)

        x0 = x
        x = self.ffn(x)
        x = self.ln2(x0 + x)
        return x

class GPTBase(nn.Module):
    def __init__(self, vocab_size, d_model, max_length, num_heads, dff, dropout, N):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_length, d_model)
        self.blocks = nn.ModuleList([GPTBlockBase(d_model, num_heads, dff, dropout) for _ in range(N)])
        self.cls = nn.Linear(d_model, vocab_size)
        self.register_buffer("pos_range", torch.arange(max_length).view(1, -1), persistent=False)

    def forward(self, input_ids, key_padding_mask=None):
        B, T = input_ids.size()
        x = self.emb(input_ids) + self.pos(self.pos_range[:, :T].to(input_ids.device))
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        return self.cls(x)

class GPTPrefixSharedMLPModel(nn.Module):
    def __init__(self, vocab_size, d_model, max_length, num_heads, dff, dropout, N, prefix_len, hidden=256):
        super().__init__()
        self.N = N
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_length, d_model)
        self.blocks = nn.ModuleList([GPTBlockPrefix(d_model, num_heads, dff, dropout) for _ in range(N)])
        self.cls = nn.Linear(d_model, vocab_size)
        self.register_buffer("pos_range", torch.arange(max_length).view(1, -1), persistent=False)

        self.prefix = PrefixSharedMLP(
            num_layers=N,
            d_model=d_model,
            num_heads=num_heads,
            prefix_len=prefix_len,
            hidden=hidden,
            dropout=0.0
        )

    def forward(self, input_ids, key_padding_mask=None):
        B, T = input_ids.size()
        x = self.emb(input_ids) + self.pos(self.pos_range[:, :T].to(input_ids.device))

        prefixes = self.prefix(batch_size=B, device=x.device, dtype=x.dtype)
        for i, blk in enumerate(self.blocks):
            pk, pv = prefixes[i]
            x = blk(x, key_padding_mask=key_padding_mask, prefix_k=pk, prefix_v=pv)
        return self.cls(x)

# -----------------------
# Train / Gen
# -----------------------
@torch.no_grad()
def gen_poetry(model, dataset, tokenizer, max_length, device):
    model.eval()
    idx = np.random.randint(0, len(dataset))
    input_ids, _, _ = dataset[idx]
    first_token = input_ids[1] if len(input_ids) > 1 else tokenizer.sos_token_id

    res = [tokenizer.sos_token_id, first_token]
    for _ in range(max_length - 2):
        x = torch.tensor(res, dtype=torch.long, device=device).view(1, -1)
        logits = model(x)
        probs = torch.softmax(logits[0, -1], dim=-1)
        nxt = torch.argmax(probs).item()
        if nxt == tokenizer.eos_token_id:
            break
        res.append(nxt)
    return "".join(tokenizer.decode(res[1:]))

def train_loop(model, dataloader, optimizer, loss_fn, device, epochs, stage_name, dataset=None, tokenizer=None, max_length=128):
    model.to(device)
    for e in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f"{stage_name} epoch {e}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            key_padding_mask = batch["key_padding_mask"].to(device)

            logits = model(input_ids, key_padding_mask=key_padding_mask)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / max(1, len(dataloader))
        print(f"[{stage_name}] epoch {e} avg_loss={avg:.4f}")
        if dataset is not None and tokenizer is not None:
            print(f"[{stage_name}] sample: {gen_poetry(model, dataset, tokenizer, max_length, device)}")

# -----------------------
# Main (Two-stage: Base -> Shared Prefix-MLP tuning)
# -----------------------
if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print("Device:", device)

    file = "dataset/text-generation/poetry_data.txt"
    base_ckpt = "base_gpt_ckpt.pt"
    prefix_ckpt = "prefix_shared_mlp_tuned_ckpt.pt"

    all_poetries = read_data(file, num=1000)
    vocab = tokenize(all_poetries)
    tokenizer = Tokenizer(vocab)

    max_length = 128
    dataset = Poetry(all_poetries, tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    vocab_size = len(vocab)
    d_model = 768
    num_heads = 12
    dff = 4 * d_model
    dropout = 0.1
    N = 1

    prefix_len = 16
    prefix_hidden = 256  # 省参数就用 128/256；更强可用 512

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # -----------------------
    # Stage 1: train base
    # -----------------------
    base_epochs = 20
    base_lr = 1e-3

    if not os.path.exists(base_ckpt):
        print("\n=== Stage 1: TRAIN BASE ===")
        base_model = GPTBase(vocab_size, d_model, max_length, num_heads, dff, dropout, N)
        base_optim = torch.optim.Adam(base_model.parameters(), lr=base_lr)

        train_loop(
            model=base_model,
            dataloader=dataloader,
            optimizer=base_optim,
            loss_fn=loss_fn,
            device=device,
            epochs=base_epochs,
            stage_name="BASE",
            dataset=dataset,
            tokenizer=tokenizer,
            max_length=max_length
        )
        torch.save({"model_state": base_model.state_dict()}, base_ckpt)
        print(f"Saved base checkpoint: {base_ckpt}")
    else:
        print(f"\nFound base checkpoint: {base_ckpt} (skip Stage 1)")

    # -----------------------
    # Stage 2: shared prefix-mlp tuning
    # -----------------------
    print("\n=== Stage 2: SHARED PREFIX-MLP TUNING ===")
    model = GPTPrefixSharedMLPModel(
        vocab_size, d_model, max_length, num_heads, dff, dropout, N,
        prefix_len=prefix_len,
        hidden=prefix_hidden
    )

    ckpt = torch.load(base_ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    print("Loaded base into shared-prefix model (strict=False).")
    print("Missing keys (expected: prefix params):", len(missing))
    print("Unexpected keys:", len(unexpected))

    # Freeze all
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze prefix params (shared + per-layer tiny proj)
    for name, p in model.named_parameters():
        if name.startswith("prefix."):
            p.requires_grad = True

    # Optional: also train LM head
    train_cls_head = True
    if train_cls_head:
        for name, p in model.named_parameters():
            if name.startswith("cls."):
                p.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    print("Trainable params:", sum(p.numel() for p in trainable))
    print("Total params:", sum(p.numel() for p in model.parameters()))

    prefix_epochs = 20
    prefix_lr = 1e-3
    optim = torch.optim.Adam(trainable, lr=prefix_lr)

    train_loop(
        model=model,
        dataloader=dataloader,
        optimizer=optim,
        loss_fn=loss_fn,
        device=device,
        epochs=prefix_epochs,
        stage_name="PREFIX-SHARED",
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=max_length
    )

    torch.save({"model_state": model.state_dict()}, prefix_ckpt)
    print(f"Saved shared prefix-mlp tuned checkpoint: {prefix_ckpt}")
