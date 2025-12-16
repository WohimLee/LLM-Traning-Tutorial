# 导入PyTorch深度学习框架
import torch
# 导入Hugging Face数据集处理库
from datasets import Dataset
# 导入JSON数据处理库
import json

# 导入PEFT（Parameter-Efficient Fine-Tuning）相关组件，用于高效微调
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
# 导入Transformers库相关组件，用于加载预训练模型和分词器
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification
# 导入TRL（Transformer Reinforcement Learning）库，用于奖励模型训练
from trl import RewardTrainer, RewardConfig

# 定义预训练模型的路径（需要用户替换为实际路径）
model_path = r'your_model_path'
# 从预训练模型路径加载分词器，禁用快速分词器以确保兼容性
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
# 设置填充方向为右侧填充
tokenizer.padding_side = "right"
# 将结束标记设置为填充标记
tokenizer.pad_token = tokenizer.eos_token

# 配置4位量化参数，用于减少模型内存占用
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 启用4位量化
    bnb_4bit_use_double_quant=True,  # 使用双重量化进一步压缩
    bnb_4bit_quant_type="nf4",  # 使用nf4量化类型
    bnb_4bit_compute_dtype=torch.float16  # 计算时使用float16精度
)

# 从预训练模型路径加载序列分类模型，配置为单标签分类
model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                           num_labels=1,  # 输出标签数为1（奖励分数）
                                                           quantization_config=bnb_config)  # 应用量化配置
# 设置模型的填充标记ID与分词器保持一致
model.config.pad_token_id = tokenizer.pad_token_id

# 配置LoRA（Low-Rank Adaptation）参数，用于高效微调
peft_config = LoraConfig(
    r=8,  # LoRA的秩，控制适配器的复杂度
    target_modules=["q_proj",  # 目标模块：查询投影层
                    "v_proj",  # 值投影层
                    "k_proj",  # 键投影层
                    "o_proj",  # 输出投影层
                    "gate_proj",  # 门控投影层
                    "down_proj",  # 下投影层
                    "up_proj"  # 上投影层
                    ],
    task_type=TaskType.SEQ_CLS,  # 任务类型：序列分类
    lora_alpha=16,  # LoRA的缩放参数
    lora_dropout=0.05  # LoRA的dropout率
)

# 为4位量化训练准备模型
model = prepare_model_for_kbit_training(model)
# 将LoRA适配器应用到模型上
model = get_peft_model(model, peft_config)
# 打印可训练参数的数量和比例
model.print_trainable_parameters()

# 初始化数据列表
items = []
# 读取偏好数据文件，每行包含一个问题、一个优选答案和一个被拒绝答案
with open("./preference.json", "r", encoding="utf8") as f:
    for line in f:
        item = json.loads(line)  # 解析JSON格式的数据
        items.append(item)  # 将数据项添加到列表中

# 将数据列表转换为Hugging Face数据集格式
dataset = Dataset.from_list(items)

# 定义数据预处理函数，将原始数据转换为模型训练所需的格式
def process_func(example):
    # 将问题和优选答案拼接
    chosen = example["question"] + example["chosen"]
    # 将问题和被拒绝答案拼接
    rejected = example["question"] + example["rejected"]

    # 对优选答案进行分词处理
    tokenized_chosen = tokenizer(chosen)
    # 对被拒绝答案进行分词处理
    tokenized_rejected = tokenizer(rejected)

    # 创建新的数据格式
    new_example = {}
    # 存储优选答案的输入ID
    new_example["input_ids_chosen"] = tokenized_chosen["input_ids"]
    # 存储优选答案的注意力掩码
    new_example["attention_mask_chosen"] = tokenized_chosen["attention_mask"]
    # 存储被拒绝答案的输入ID
    new_example["input_ids_rejected"] = tokenized_rejected["input_ids"]
    # 存储被拒绝答案的注意力掩码
    new_example["attention_mask_rejected"] = tokenized_rejected["attention_mask"]
    return new_example

# 对数据集应用预处理函数，移除原始列并添加处理后的列
dataset = dataset.map(process_func, remove_columns=['question', 'chosen', 'rejected'])
# 打印处理后的数据集信息
print(dataset)

# 配置奖励模型训练参数
config = RewardConfig(output_dir="./reward_model")  # 设置输出目录
config.num_train_epochs = 1  # 设置训练轮数为1
config.per_device_train_batch_size = 1  # 设置每个设备的批次大小为1

# 创建奖励模型训练器
trainer = RewardTrainer(
    model=model,  # 传入配置好的模型
    tokenizer=tokenizer,  # 传入分词器
    args=config,  # 传入训练配置
    train_dataset=dataset  # 传入训练数据集
)
# 开始训练奖励模型
trainer.train()
# 保存训练好的奖励模型到指定目录
trainer.save_model("./reward_model")
