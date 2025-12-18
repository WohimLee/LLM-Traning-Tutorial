import pandas as pd, json, random, os

csv_path = "data/finance/jwangkun_deepseek-fin.csv"
out_dir = "fin_qwen3_sft"
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(csv_path)

# 基础清洗：把 NaN 变成空串，去掉首尾空格
for col in ["instruction", "input", "output", "system"]:
    if col in df.columns:
        df[col] = df[col].fillna("").astype(str).map(lambda x: x.strip())

# 丢掉 instruction/output 为空的行
df = df[(df["instruction"] != "") & (df["output"] != "")].reset_index(drop=True)

# 切分 train/eval（95/5）
random.seed(42)
idx = list(range(len(df)))
random.shuffle(idx)
split = int(len(idx) * 0.95)
train_df = df.iloc[idx[:split]].reset_index(drop=True)
eval_df  = df.iloc[idx[split:]].reset_index(drop=True)

def write_jsonl(rows, filepath, mode="messages",
               default_system="你是一个严谨、耐心的金融领域中文助手。"):
    with open(filepath, "w", encoding="utf-8") as f:
        for _, r in rows.iterrows():
            instr = r["instruction"]
            inp   = r.get("input", "")
            out   = r["output"]
            sys   = (r.get("system", "") or "").strip() or default_system

            if mode == "alpaca":
                obj = {"instruction": instr, "input": inp, "output": out}

            elif mode == "messages":
                user = instr if not inp else f"{instr}\n\n补充信息：{inp}"
                obj = {"messages": [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": out},
                ]}
            else:
                raise ValueError("mode must be alpaca/messages")

            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

write_jsonl(train_df, os.path.join(out_dir, "train_alpaca.jsonl"),   mode="alpaca")
write_jsonl(eval_df,  os.path.join(out_dir, "eval_alpaca.jsonl"),    mode="alpaca")
write_jsonl(train_df, os.path.join(out_dir, "train_messages.jsonl"), mode="messages")
write_jsonl(eval_df,  os.path.join(out_dir, "eval_messages.jsonl"),  mode="messages")

print("done:", out_dir)
