## PEFT

### 一、第一代：结构插入派（2019–2020）
#### 1️⃣ Adapter（最早成熟）

- 代表论文: Houlsby et al., 2019
- 核心思想: 在 Transformer 的每一层中 插入一个小的可训练模块
```
Transformer Layer
   ├── 原始参数（冻结）
   └── Adapter（可训练，小瓶颈）
```

>特点

- ✅ 不改输入
- ✅ 任务切换方便
- ❌ 推理时多一层计算
- ❌ 对大模型（LLM）不够友好

>定位
- 高效微调的“祖师爷”

### 二、第二代：不改模型结构（2021）
#### 2️⃣ Prefix Tuning（2021）

- 代表论文: Li & Liang, 2021
- 核心思想: 不改模型参数，而是 给每一层注意力加“前缀 Key/Value”
```py
Attention(K, V) = [Prefix_KV ; 原始KV]
```

>关键点
- Prefix 是 连续向量（soft prompt）
- 加在 每一层的 self-attention

>优点
- 不改 Transformer 结构
- 参数量少

>缺点
- 对 encoder-only / decoder-only 模型效果不稳定
- 训练复杂

### 三、第三代：输入层软提示（2021）
#### 3️⃣ Prompt Tuning（Google, 2021）

- 核心思想: 把 prompt 变成 可训练 embedding
```
[ P1, P2, P3, ..., Pk, x1, x2, ... ]
```

>特点

- 只训练输入 embedding
- 模型参数完全冻结

>问题

- ❌ 对小模型有效
- ❌ 对大模型（>10B）效果急剧下降

#### 4️⃣ P-Tuning v1（清华，2021）

改进点: Prompt Tuning + MLP 生成 Prompt
```
z → MLP → soft prompt embeddings
```

>优点

- 比 Prompt Tuning 稳定
- 可学习性更强

>缺点

- ❌ 在深层 Transformer 中影响太弱
- ❌ LLM 表现仍然一般

### 四、关键跃迁：P-Tuning v2（2022）
#### 5️⃣ P-Tuning v2（重要分水岭）

- 核心思想: Prefix Tuning + Prompt Tuning 的统一
- 不只在输入层
- 在每一层 Attention 注入可训练参数
- 本质上：“把 Prefix Tuning 系统化并工程化”

>优势

- 首次在 百亿级模型 上可用
- 和全量微调性能接近

>意义
- 让“不改模型权重”这条路第一次走通

### 五、第四代：低秩参数更新（2021–至今）
#### 6️⃣ LoRA（2021，微软）

革命性方法

- 核心思想: 权重更新是低秩的
```
W' = W + ΔW
ΔW = A · B   （rank ≪ d）
```

>特点
- 插在 Linear 层
- 训练参数极少
- 推理时可合并权重（无额外开销）

>优点

- 效果强
- 工程简单
- 兼容所有 Transformer

结果: 🔥 成为 LLM 微调事实标准

#### 7️⃣ QLoRA（2023）

LoRA + 量化
```
Base model: 4-bit quantized
Trainable: LoRA (FP16)
```

>关键技术

- NF4 量化
- Double Quant
- Paged Optimizer

>效果

- 单张 24GB 显卡可训 65B 模型
- 几乎不掉性能

>现状
- 工业 & 开源最常用方案

### 六、它们的关系总结（一句话版）


| 方法            | 本质           |
| ------------- | ------------ |
| Adapter       | 插模块          |
| Prefix Tuning | Attention 前缀 |
| Prompt Tuning | 输入层软 prompt  |
| P-Tuning v1   | Prompt + MLP |
| P-Tuning v2   | Prefix 的系统化  |
| LoRA          | 权重低秩更新       |
| QLoRA         | LoRA + 量化    |


### 七、三条技术路线的对比
#### 🟦 路线 1：软提示派
```
Prompt → P-Tuning → P-Tuning v2 / Prefix
```
- 不改权重
- 适合参数冻结场景

#### 🟩 路线 2：结构插入派
```
Adapter
```
- NLP 早期主流
- LLM 中式微

#### 🟥 路线 3：权重更新派（赢家）
```
LoRA → QLoRA
```

- 工业标准
- 最优性价比

### 八、现在该怎么选？（实战建议）

| 场景         | 推荐             |
| ---------- | -------------- |
| 个人 / 学术实验  | LoRA           |
| 显存极低       | QLoRA          |
| 参数必须冻结     | P-Tuning v2    |
| 多任务快速切换    | Adapter / LoRA |
| SFT / 指令微调 | LoRA / QLoRA   |

### 九、一句话终极总结

Adapter 是起点，Prompt/P-Tuning 是过渡，LoRA 是胜利者，QLoRA 是工程极限版本。