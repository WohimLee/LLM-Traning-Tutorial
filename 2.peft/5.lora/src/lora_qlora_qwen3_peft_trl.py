import os
import torch
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from trl.chat_template_utils import add_response_schema
from trl import DataCollatorForCompletionOnlyLM


# =========================
# ====== 配置区 ==========
# =========================

MODEL_PATH = "/path/to/Qwen3-0.6B"   # 本地模型路径
TRAIN_FILE = "train_messages.jsonl"
EVAL_FILE  = "eval_messages.jsonl"
OUTPUT_DIR = "qwen3_06b_lora_out"

USE_QLORA = False     # True = QLoRA (4bit), False = LoRA
MAX_SEQ_LEN = 2048

PER_DEVICE_TRAIN_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE  = 2
GRADIENT_ACCUMULATION_STEPS = 8

LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 1
WARMUP_RATIO = 0.03

LOGGING_STEPS = 20
SAVE_STEPS = 200
EVAL_STEPS = 200

# LoRA 超参
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# =========================
# ====== 主流程 ==========
# =========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1) 载入数据
dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE,
        "eval": EVAL_FILE,
    },
)

# 2) tokenizer + Qwen chat template
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

# 给 tokenizer 增加 TRL 所需的 response/instruction schema
tokenizer = add_response_schema(tokenizer)

# 3) messages -> text
def render_chat(batch):
    texts = []
    for msgs in batch["messages"]:
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,  # 训练数据已有 assistant
        )
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(
    render_chat,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# 4) 模型加载（LoRA / QLoRA）
quant_config = None
if USE_QLORA:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        bnb_4bit_use_double_quant=True,
    )

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    quantization_config=quant_config,
    trust_remote_code=True,
)

# 5) LoRA 配置
peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# 6) 只在 assistant 回复上算 loss
data_collator = DataCollatorForCompletionOnlyLM(
    tokenizer=tokenizer,
    response_template=tokenizer.response_template,
    instruction_template=tokenizer.instruction_template,
)

# 7) Trainer 配置
trainer_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_seq_length=MAX_SEQ_LEN,
    dataset_text_field="text",

    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",

    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    eval_steps=EVAL_STEPS,
    evaluation_strategy="steps",
    save_strategy="steps",

    bf16=torch.cuda.is_available(),
    fp16=False,

    packing=False,     # 建议先关，稳定
    report_to="none",
)

# 8) Trainer
trainer = SFTTrainer(
    model=model,
    args=trainer_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
    tokenizer=tokenizer,
    peft_config=peft_config,
    data_collator=data_collator,
)

# 9) 开始训练
trainer.train()

# 10) 保存
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n✅ Training finished. Model saved to {OUTPUT_DIR}")
