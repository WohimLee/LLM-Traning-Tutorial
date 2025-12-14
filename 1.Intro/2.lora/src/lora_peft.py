import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

def preprocess(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=128)

# 1. 加载数据集（以情感分析SST2为例）
dataset = load_dataset("glue","sst2",  cache_dir="./my_glue_data")
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

print("数据集加载完毕")

encoded_dataset = dataset.map(preprocess, batched=True)
encoded_dataset = encoded_dataset.rename_column("label", "labels")
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 2. 加载模型
model = AutoModelForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)

# 3. 配置LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,   # 任务类型
    r=8,                          # LoRA秩
    lora_alpha=32,                # LoRA缩放
    lora_dropout=0.1,             # Dropout
    target_modules=["query", "value"]  # 目标模块（BERT常用query/value）
)
model = get_peft_model(model, lora_config)

# 4. 训练参数
training_args = TrainingArguments(
    output_dir="./lora-demo",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    # evaluation_strategy="epoch",
    # save_strategy="epoch",
    logging_steps=1,
    learning_rate=2e-4,
    fp16=torch.cuda.is_available(),
)

# 5. Trainer训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"].select(range(200)),  # 只用200条做demo
    eval_dataset=encoded_dataset["validation"].select(range(50)),
)

trainer.train()