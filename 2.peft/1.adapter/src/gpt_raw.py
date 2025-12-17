
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from modules import GPTBase, GPTWithAdapters

'''
Stage 1：全量训练 base（无 adapter）并保存 base_ckpt.pt

Stage 2：加载 base 权重到带 adapter 的模型，冻结 base，仅训练 adapter（+可选 cls head）
'''

# --------------------------
# Utils
# --------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# Data
# --------------------------
def read_data(file, num=None):
    with open(file, "r", encoding="utf-8") as f:
        all_poetries = f.read().strip().split("\n")
    if num is not None:
        all_poetries = all_poetries[:num]
    # 简单清洗：去掉空行
    all_poetries = [x.strip() for x in all_poetries if x.strip()]
    return all_poetries

def tokenize(all_poetries):
    vocab = {"[pad]": 0, "[unk]": 1, "[sos]": 2, "[eos]": 3, "，": 4, "。": 5}
    for line in all_poetries:
        for token in line:
            if token not in vocab:
                vocab[token] = len(vocab)
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
        ids = [self.vocab.get(token, self.unk_token_id) for token in text]

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
        poetry = self.all_poetries[idx]

        input_ids = self.tokenizer.encode(
            text=poetry,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        # key_padding_mask: True 表示有效token，False 表示 pad
        # 你原来写的 padding_mask 返回 torch.tensor(x)!=0，这里保留含义（pad=0）
        key_padding_mask = torch.tensor(input_ids) != self.tokenizer.pad_token_id

        # 语言模型训练：label 是 input 向右移一位，末尾补 eos
        # label 长度与 input 相同
        label_ids = input_ids[1:] + [self.tokenizer.eos_token_id]
        label_ids = label_ids[:self.max_length]
        # 对 pad 位置也 pad
        if len(label_ids) < self.max_length:
            label_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(label_ids))

        return input_ids, label_ids, key_padding_mask

    def __len__(self):
        return len(self.all_poetries)

def collate_fn(batch):
    input_ids, label_ids, key_padding_mask = zip(*batch)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "label_ids": torch.tensor(label_ids, dtype=torch.long),
        "key_padding_mask": torch.stack(key_padding_mask).bool(),  # (B, T)
    }


# --------------------------
# Generation (optional)
# --------------------------
@torch.no_grad()
def gen_poetry(model, dataset, tokenizer, max_length, device):
    model.eval()
    idx = np.random.randint(0, len(dataset))
    # 取某首诗的第一个内容 token 当“题头”提示
    sample_input_ids, _, _ = dataset[idx]
    if len(sample_input_ids) < 2:
        return ""
    first_token_id = sample_input_ids[1]

    res_ids = [tokenizer.sos_token_id, first_token_id]
    for _ in range(max_length - 2):
        input_ids = torch.tensor(res_ids, dtype=torch.long, device=device).view(1, -1)
        logits = model(input_ids)  # (1,T,V)
        probs = torch.softmax(logits[0, -1], dim=-1)
        next_id = torch.argmax(probs).item()
        if next_id == tokenizer.eos_token_id:
            break
        res_ids.append(next_id)

    res_text = "".join(tokenizer.decode(res_ids[1:]))
    return res_text

# --------------------------
# Train loops
# --------------------------
def train_one_stage(
    model, dataloader, optimizer, loss_fn, device, epochs,
    tokenizer=None, dataset=None, max_length=128, stage_name="stage"
):
    model.to(device)
    for e in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in tqdm(dataloader, desc=f"{stage_name} epoch {e}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            label_ids = batch["label_ids"].to(device)
            key_padding_mask = batch["key_padding_mask"].to(device)

            logits = model(input_ids, key_padding_mask=key_padding_mask)  # (B,T,V)
            loss = loss_fn(logits.view(-1, logits.size(-1)), label_ids.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(dataloader))
        print(f"[{stage_name}] Epoch {e} avg loss = {avg_loss:.4f}")

        if tokenizer is not None and dataset is not None:
            sample = gen_poetry(model, dataset, tokenizer, max_length, device)
            print(f"[{stage_name}] sample: {sample}")

# --------------------------
# Main: Stage A
# --------------------------
if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print("Device:", device)

    # =========
    # Config
    # =========
    file = "data/poetry_data.txt"
    num_lines = 1000              # 读多少行
    max_length = 128
    batch_size = 32

    vocab_d_model = 768
    num_heads = 12
    dff = 4 * vocab_d_model
    dropout = 0.1
    N = 1                         # 你原来是 1；想更强可以改 6/12

    # Stage 1: base pretrain
    base_epochs = 20              # 示例：你可以加大
    base_lr = 1e-3
    base_ckpt_path = "output/base_ckpt.pt"

    # Stage 2: adapter finetune
    adapter_epochs = 20
    adapter_lr = 1e-3
    adapter_bottleneck = 64       # adapter 宽度（越小越省参数）
    train_cls_head = True         # 常见做法：adapter + lm head 一起训

    # =========
    # Load data + split
    # =========
    all_poetries = read_data(file, num=num_lines)
    vocab = tokenize(all_poetries)
    tokenizer = Tokenizer(vocab)
    vocab_size = len(vocab)

    # 简单 split：前 80% 用于 base pretrain，后 20% 当 finetune（你也可以换成不同文件）
    split = int(0.8 * len(all_poetries))
    pretrain_poems = all_poetries[:split]
    finetune_poems = all_poetries[split:]

    pretrain_ds = Poetry(pretrain_poems, tokenizer, max_length=max_length)
    finetune_ds = Poetry(finetune_poems, tokenizer, max_length=max_length)

    pretrain_dl = DataLoader(pretrain_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    finetune_dl = DataLoader(finetune_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # =========
    # Stage 1: Pretrain base (full params)
    # =========
    if not os.path.exists(base_ckpt_path):
        print("\n=== Stage 1: Pretrain BASE (full finetune) ===")
        base_model = GPTBase(
            vocab_size=vocab_size,
            d_model=vocab_d_model,
            max_length=max_length,
            num_heads=num_heads,
            dff=dff,
            dropout=dropout,
            N=N
        )
        base_optim = torch.optim.Adam(base_model.parameters(), lr=base_lr)

        train_one_stage(
            model=base_model,
            dataloader=pretrain_dl,
            optimizer=base_optim,
            loss_fn=loss_fn,
            device=device,
            epochs=base_epochs,
            tokenizer=tokenizer,
            dataset=pretrain_ds,
            max_length=max_length,
            stage_name="BASE"
        )

        torch.save({"model_state": base_model.state_dict()}, base_ckpt_path)
        print(f"Saved base checkpoint to: {base_ckpt_path}")
    else:
        print(f"\nFound existing base checkpoint: {base_ckpt_path} (skip Stage 1)")

    # =========
    # Stage 2: Load base -> add adapters -> freeze base -> train adapters (+ optional cls)
    # =========
    print("\n=== Stage 2: Finetune ADAPTERS (freeze base) ===")
    adapter_model = GPTWithAdapters(
        vocab_size=vocab_size,
        d_model=vocab_d_model,
        max_length=max_length,
        num_heads=num_heads,
        dff=dff,
        dropout=dropout,
        N=N,
        adapter_bottleneck=adapter_bottleneck
    )

    ckpt = torch.load(base_ckpt_path, map_location="cpu")
    # strict=False：base 没有 adapter 参数，加载时会自动跳过 adapter 的 key
    missing, unexpected = adapter_model.load_state_dict(ckpt["model_state"], strict=False)
    print("Load base into adapter model (strict=False).")
    print("Missing keys (expected, adapters):", len(missing))
    print("Unexpected keys:", len(unexpected))

    # Freeze all
    for p in adapter_model.parameters():
        p.requires_grad = False

    # Unfreeze adapters
    for name, p in adapter_model.named_parameters():
        if "attn_adapter" in name or "ffn_adapter" in name:
            p.requires_grad = True

    # Optional: train lm head
    if train_cls_head:
        for name, p in adapter_model.named_parameters():
            if name.startswith("cls."):
                p.requires_grad = True

    trainable_params = [p for p in adapter_model.parameters() if p.requires_grad]
    print("Trainable params:", sum(p.numel() for p in trainable_params))
    print("Total params:", sum(p.numel() for p in adapter_model.parameters()))

    adapter_optim = torch.optim.Adam(trainable_params, lr=adapter_lr)

    train_one_stage(
        model=adapter_model,
        dataloader=finetune_dl,
        optimizer=adapter_optim,
        loss_fn=loss_fn,
        device=device,
        epochs=adapter_epochs,
        tokenizer=tokenizer,
        dataset=finetune_ds,
        max_length=max_length,
        stage_name="ADAPTER"
    )

    # 保存 adapter finetune 结果
    torch.save({"model_state": adapter_model.state_dict()}, "adapter_finetuned_ckpt.pt")
    print("Saved adapter finetuned checkpoint to: adapter_finetuned_ckpt.pt")
