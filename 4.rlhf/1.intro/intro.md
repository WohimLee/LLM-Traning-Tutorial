## Intro
### 一、先给结论地图（不看细节也能懂）

一句话版：

RLHF：用“人类偏好”来教模型什么是“好回答”

PPO：最早、最标准的 RLHF 训练算法（稳定但复杂、贵）

DPO：把 RLHF 变成一个纯监督学习问题（简单、便宜）

GRPO：在 PPO 思路下 去掉 value model，更适合大模型规模化训练

### 二、RLHF 是什么？（核心思想）

RLHF = Reinforcement Learning from Human Feedback

解决的问题

预训练语言模型只学到：

“像人一样说话”

但我们真正想要的是：

“说对的、说好的、符合人类偏好的话”

RLHF 的三步走

SFT（监督微调）

人类写“好答案”

模型模仿人类

训练奖励模型（Reward Model, RM）

给模型多个回答

人类排序：A > B > C

训练一个模型来给回答打分

用强化学习优化语言模型

目标：最大化奖励模型给的分数

⚠️ 这一步最难、最贵

👉 PPO / DPO / GRPO 就是第 3 步的不同做法

### 三、PPO：RLHF 的“经典解法”
PPO 是什么？

Proximal Policy Optimization

一种稳定的强化学习算法

OpenAI 最早在 ChatGPT / InstructGPT 中大量使用

PPO 在干什么？

可以理解为：

“在不偏离原模型太多的前提下，让模型回答更讨人喜欢”

PPO 的关键组件

Policy Model（正在训练的大模型）

Reference Model（冻结的旧模型，用来约束）

Reward Model（人类偏好）

Value Model（估计长期收益）

为什么 PPO 好？

✅ 稳定
✅ 理论成熟
❌ 非常复杂
❌ 显存、算力、工程成本高
❌ 要同时训练/维护多个模型

👉 PPO = 工业级，但很重

### 四、DPO：把 RLHF 变成“普通训练”
DPO 是什么？

Direct Preference Optimization

核心思想（非常重要）

不做强化学习了，直接学“人类更喜欢哪个”

人类给的数据是：

(prompt, chosen_answer, rejected_answer)


DPO 直接优化目标：

让模型对 chosen 的概率 > rejected

本质变化
PPO	DPO
强化学习	监督学习
需要 reward model	❌ 不需要
需要 value model	❌ 不需要
复杂不稳定	简单稳定
为什么 DPO 能工作？

数学上可以证明：

在特定假设下，DPO 等价于 PPO 的最优解

现实效果

小模型 / 中模型：DPO ≈ PPO

工程成本：大幅下降

成为 2023–2024 年最流行方法之一（LLaMA / Qwen 等）

### 五、GRPO：PPO 的“轻量化进化版”
GRPO 是什么？

Group Relative Policy Optimization

最早由 DeepSeek 提出并大规模使用。

它解决 PPO 的什么痛点？

👉 Value Model 太贵、太不稳定

GRPO 的关键思想

不预测“绝对奖励”，只比较“相对好坏”

做法：

对同一个 prompt

采样一组回答（group）

用 reward model 给它们打分

只看谁比谁好

📌 用 组内相对优势 替代 value model

对比 PPO
	PPO	GRPO
Value Model	需要	❌ 不需要
奖励	绝对值	相对值
稳定性	中	高
成本	高	更低
可扩展性	一般	很强

👉 GRPO = “更适合大模型时代的 PPO”

### 六、把四者放在一张总表里
方法	本质	是否 RL	是否需要 RM	是否需要 Value	工程复杂度
RLHF	框架	—	✅	取决于算法	—
PPO	强化学习	✅	✅	✅	⭐⭐⭐⭐⭐
DPO	排序监督	❌	❌	❌	⭐⭐
GRPO	相对 RL	✅	✅	❌	⭐⭐⭐


### 七、什么时候用谁？（实践视角）
👉 用 PPO

资源充足

追求极致性能

工业级闭源大模型

👉 用 DPO

开源模型

算力有限

想快速、稳定对齐

👉 用 GRPO

超大模型

长推理 / CoT / reasoning

希望 RL 效果 + 更低成本

### 八、一句话总结

RLHF 是“目标”
PPO 是“经典但昂贵的实现”
DPO 是“把 RLHF 简化成监督学习”
GRPO 是“为大模型重新设计的 PPO”