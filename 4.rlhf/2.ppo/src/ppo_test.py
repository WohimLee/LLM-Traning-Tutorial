# 导入PyTorch深度学习框架
import torch
# 导入PEFT（Parameter-Efficient Fine-Tuning）相关组件，用于高效微调
from peft import LoraConfig, TaskType
# 导入Transformers库相关组件，用于加载预训练模型和分词器
from transformers import AutoTokenizer, BitsAndBytesConfig
# 导入TRL（Transformer Reinforcement Learning）库，用于PPO强化学习训练
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
# 导入Hugging Face数据集处理库
from datasets import Dataset
# 导入JSON数据处理库
import json

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
    bnb_4bit_compute_dtype=torch.bfloat16  # 计算时使用bfloat16精度
)

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
    task_type=TaskType.CAUSAL_LM,  # 任务类型：因果语言模型
    lora_alpha=16,  # LoRA的缩放参数
    lora_dropout=0.05  # LoRA的dropout率
)

# 加载带有价值头的因果语言模型，用于PPO训练
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path,
                                                          reward_adapter="./reward_model",  # 加载预训练的奖励模型适配器
                                                          peft_config=peft_config,  # 应用LoRA配置
                                                          quantization_config=bnb_config  # 应用量化配置
                                                          )
# 将模型移动到GPU设备上进行计算
model.to("cuda")

# 初始化数据列表
items = []
# 读取查询数据文件，每行包含一个查询问题
with open("./data/queries.json", "r", encoding="utf8") as f:
    for line in f:
        items.append(json.loads(line))  # 解析JSON格式的数据并添加到列表
# 将数据列表转换为Hugging Face数据集格式
queries_dataset = Dataset.from_list(items)

# 定义数据整理函数，将查询文本转换为模型输入格式
def collator(data):
    queries = []  # 初始化查询列表
    for item in data:
        # 对每个查询进行分词处理，转换为张量格式并移动到GPU
        queries.append(tokenizer(item["query"], return_tensors="pt")["input_ids"].squeeze().to("cuda"))
    return queries  # 返回处理后的查询列表

# 配置PPO（Proximal Policy Optimization）训练参数
ppo_config = PPOConfig(kl_penalty="full",  # 使用完整的KL散度惩罚
                       ppo_epochs=3,  # PPO训练轮数为3
                       batch_size=2,  # 批次大小为2
                       mini_batch_size=1)  # 小批次大小为1

# 创建PPO训练器
ppo_trainer = PPOTrainer(config=ppo_config,  # 传入PPO配置
                         model=model,  # 传入模型
                         ref_model=None,  # 参考模型设为None（使用当前模型作为参考）
                         tokenizer=tokenizer,  # 传入分词器
                         dataset=queries_dataset,  # 传入训练数据集
                         data_collator=collator)  # 传入数据整理函数

# 配置文本生成参数
generation_kwargs = {
    "min_length": -1,  # 最小生成长度设为-1（无限制）
    "top_k": 0.0,  # top-k采样设为0（禁用）
    "top_p": 1.0,  # top-p采样设为1.0（使用所有token）
    "do_sample": True,  # 启用采样模式
    "pad_token_id": tokenizer.pad_token_id,  # 设置填充标记ID
    "max_new_tokens": 32,  # 最大生成新token数为32
}

# 开始PPO训练循环
for batch in ppo_trainer.dataloader:
    query_tensors = batch  # 获取当前批次的查询张量

    # 使用模型生成回复
    response_tensors = ppo_trainer.generate(
        query_tensors, return_prompt=False,  **generation_kwargs)
    
    scores = []  # 初始化分数列表
    # 计算每个查询-回复对的奖励分数
    for query, response in zip(query_tensors, response_tensors):
        # 将查询和回复拼接在一起
        input_ids = torch.concat([query, response], dim=0)
        # 增加批次维度
        input_ids = torch.unsqueeze(input_ids, dim=0)
        # 使用奖励模型计算奖励分数
        score = ppo_trainer.model.compute_reward_score(input_ids=input_ids)[0, -1, 0]
        scores.append(score)  # 将分数添加到列表中
    
    # 执行PPO训练步骤，更新模型参数
    stats = ppo_trainer.step(query_tensors, response_tensors, scores)

# 保存训练好的PPO模型
ppo_trainer.save_pretrained("./rl_model")
