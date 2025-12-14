import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, lora_alpha=32, lora_dropout=0.1):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # LoRA 部分
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            self.scaling = lora_alpha / r
        else:
            self.lora_A = None
            self.lora_B = None
            self.scaling = 1.0

    def forward(self, x):
        result = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.r > 0:
            lora_out = self.lora_dropout(x) @ self.lora_A.T
            lora_out = lora_out @ self.lora_B.T
            result = result + self.scaling * lora_out
        return result

def patch_bert_with_lora(model, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["query", "value"]):
    for name, module in model.named_modules():
        if any(tm in name for tm in target_modules) and isinstance(module, nn.Linear):
            parent = model
            name_parts = name.split('.')
            for n in name_parts[:-1]:
                parent = getattr(parent, n)
            orig_linear = getattr(parent, name_parts[-1])
            lora_linear = LoRALinear(
                orig_linear.in_features,
                orig_linear.out_features,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            )
            # 复制原始权重
            lora_linear.weight.data = orig_linear.weight.data.clone()
            lora_linear.bias.data = orig_linear.bias.data.clone()
            # 冻结原始权重，只训练 LoRA 分支
            lora_linear.weight.requires_grad = False
            lora_linear.bias.requires_grad = False
            setattr(parent, name_parts[-1], lora_linear)
    return model

def preprocess(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=128)

# 1. 加载数据集
dataset = load_dataset("glue", "sst2", cache_dir="./my_glue_data")
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
encoded_dataset = dataset.map(preprocess, batched=True)
encoded_dataset = encoded_dataset.rename_column("label", "labels")
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 2. 加载模型
model = AutoModelForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)

# 3. 注入 LoRA
model = patch_bert_with_lora(model, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["query", "value"])

# 4. 训练参数
training_args = TrainingArguments(
    output_dir="./lora-demo",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    logging_steps=1,
    learning_rate=2e-4,
    fp16=torch.cuda.is_available(),
)

# 5. Trainer训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"].select(range(200)),
    eval_dataset=encoded_dataset["validation"].select(range(50)),
)

trainer.train() 