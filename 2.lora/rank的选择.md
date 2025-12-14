

## 1 rank 是什么？
在LoRA中，原始的权重矩阵 $W \in \mathbb{R}^{\wedge}\left\{d\right.$＿out $\left.\times d \_i n\right\}$ 不直接参与训绕，而是添加一个可训绕的低秩矩阵表示：

$$
\Delta W=A \times B, \text { 其中 } A \in \mathbb{R}^{d_o u t \times r}, B \in \mathbb{R}^{r \times d_i n}
$$


所以 $r$ 控制了引入的参数数量和表示能力。

## 2 如何选择 rank？
>常用经验值
- 小模型（<1B）：r = 4 或 r = 8
- 中模型（1B~7B）：r = 8 到 r = 32
- 大模型（13B 及以上）：r = 16、32，甚至 64

>任务复杂度
- 简单分类、匹配任务：r = 4 ~ 8 足够
- 复杂生成、代码、对话任务：r = 16+ 更合适

>数据规模
- 数据越少，r 应该越小，以防止过拟合
- 数据量大时，可以适当提高 r 提升模型能力

>显存限制
- r 越大，显存开销越大（但比全参数小很多）
- 在 GPU 紧张场景下，优先选小 r

>预实验调优
- 可以固定其他超参，仅调 r 进行对比实验，如：
```
r = 4, 8, 16, 32
```
观察验证集loss或task-specific metric变化

## 3 一个通用建议

任务类型	|数据量	|推荐 rank
:-|:-|:-
文本分类	|小	|4~8
QA / NER	|中	|8~16
生成类任务	|大	|16~32+

## 3 进阶方法（自动选择 rank）
- AutoLoRA / Adaptive LoRA：动态决定各层的 rank，但目前这些方法还未广泛部署。
- LoRA Dropout + 正则：配合适当 Dropout 或 L1/L2 正则，也能一定程度上缓解 rank 选择带来的风险。