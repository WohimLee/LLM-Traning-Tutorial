
## 1 LoRA（Low-Rank Adaptation）
>原理
- 冻结原始模型参数，只在每个需要微调的权重矩阵旁边插入两个低秩矩阵 $A ∈ ℝ^{d×r}$ 和 $B ∈ ℝ^{r×d}$，近似表示权重的更新 $ΔW = BA$。

>公式

$$W' = W + BA$$
- 其中 $r ≪ d$

>优点：

- 参数量大幅减少（只训练 A, B）
- 易于合并（merge）回原模型
- 适用于大多数 transformer 模型的 Attention 和 FFN 模块

使用场景：任务定制微调（如情感分析、问答等）

## 2 QLoRA（Quantized LoRA）
>原理
- 是在 LoRA 的基础上，将原始大模型量化为 `4-bit` 或 `8-bit`（例如使用 GPTQ、BitsAndBytes），只对 LoRA 插入的参数进行训练。

>步骤

- 将模型量化（节省显存）
- 插入 LoRA 层，只微调 LoRA 参数

优点：

支持在消费级 GPU（如 24GB）上微调 65B 模型

显著降低显存和内存需求

典型工具：PEFT + bitsandbytes

使用场景：在极端硬件限制下训练大模型

✅ 3. P-Tuning (Prompt Tuning)
P-Tuning v1（连续 embedding prompt）：

原理：在输入 embedding 层前加上一些“可学习”的连续向量作为 prompt，不改变模型结构和参数

限制：适用于较小模型，大模型难以稳定优化

P-Tuning v2（支持深层插入）：

原理：类似 prefix-tuning（下一节），可以把 prompt 注入到 transformer 的多个层中

特点：

参数量更少

兼容多任务

代表项目：Tsinghua 发布的 OpenPrompt、PromptPapers

✅ 4. Prefix-Tuning
原理：对 transformer 的每一层 self-attention 添加可学习的“前缀键值对”向量（prefix key/value），类似 prompt 的形式影响模型推理过程

实现：前缀不会直接输入文本中，而是附加在每层 attention 的 key/value 上

优点：

极低的训练参数量

支持多任务调优

缺点：

对模型结构有较强依赖

训练稳定性略差于 LoRA/QLoRA