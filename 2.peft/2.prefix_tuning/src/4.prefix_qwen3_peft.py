# train_prefix_tuning.py


import os
import argparse
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

from peft import PrefixTuningConfig, get_peft_model, TaskType

from utils import (
    set_seed,
    load_jsonl,
    SFTMessagesDataset,
    DataCollatorForSFT,
    ppl_from_eval_loss,
    generate_samples,
    average_rouge_l,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True, help="本地模型路径，例如 /path/to/qwen3-0.6B")
    p.add_argument("--train_file", type=str, default="train_messages.jsonl")
    p.add_argument("--eval_file", type=str, default="eval_messages.jsonl")

    p.add_argument("--output_dir", type=str, default="./outputs_prefix_tuning")
    p.add_argument("--max_length", type=int, default=2048)

    # Prefix-Tuning 超参
    p.add_argument("--num_virtual_tokens", type=int, default=64, help="prefix 长度，常用 16/32/64/128")
    p.add_argument("--prefix_projection", action="store_true", help="是否启用 prefix projection（更灵活但参数更多）")

    # 训练超参
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--per_device_eval_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=200)

    # 生成评估
    p.add_argument("--gen_samples", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_p", type=float, default=0.9)

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # --------
    # Tokenizer
    # --------
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Qwen 系列通常 eos/pad 处理：pad 用 eos
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --------
    # Base model
    # --------
    dtype = None
    if torch.cuda.is_available():
        # 优先 bf16（若硬件支持），否则 fp16
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # 避免梯度检查点/缓存冲突（prefix/peft 通常建议关 cache）
    model.config.use_cache = False

    # --------
    # Prefix-Tuning (PEFT)
    # --------
    peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=args.num_virtual_tokens,
        prefix_projection=args.prefix_projection,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # --------
    # Data
    # --------
    train_rows = load_jsonl(args.train_file)
    eval_rows = load_jsonl(args.eval_file)

    train_ds = SFTMessagesDataset(train_rows, tokenizer=tokenizer, max_length=args.max_length)
    eval_ds = SFTMessagesDataset(eval_rows, tokenizer=tokenizer, max_length=args.max_length)
    collator = DataCollatorForSFT(tokenizer=tokenizer)

    # --------
    # Training
    # --------
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and (not use_bf16)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,

        num_train_epochs=args.num_train_epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,

        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,

        logging_steps=args.logging_steps,
        report_to="none",

        bf16=use_bf16,
        fp16=use_fp16,

        # 对小模型/长序列更稳一点
        max_grad_norm=1.0,

        # 让保存内容更干净
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # 保存 prefix adapter
    adapter_dir = os.path.join(args.output_dir, "prefix_adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # --------
    # Eval: loss / ppl
    # --------
    metrics = trainer.evaluate()
    eval_loss = float(metrics.get("eval_loss", float("nan")))
    ppl = ppl_from_eval_loss(eval_loss)
    print("\n===== Eval Metrics =====")
    print({**metrics, "eval_ppl": ppl})

    # --------
    # Qualitative: generation + (optional) ROUGE-L
    # --------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = generate_samples(
        model=trainer.model,
        tokenizer=tokenizer,
        eval_rows=eval_rows,
        n=args.gen_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed + 999,
        device=device,
    )
    rouge = average_rouge_l(samples)

    print("\n===== Sample Generations (first 3) =====")
    for i, s in enumerate(samples[:3]):
        print(f"\n--- Sample {i} ---")
        print("[PROMPT]\n", s["prompt"][-800:])  # 避免太长，截一下尾部
        print("[PRED]\n", s["pred"])
        print("[REF]\n", s["ref"])

    print("\n===== Simple ROUGE-L (char-level) =====")
    print({"rouge_l_f1": rouge})


if __name__ == "__main__":
    main()
