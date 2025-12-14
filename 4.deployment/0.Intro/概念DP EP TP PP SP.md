
## 大模型并行训练与推理：DP / EP / TP / PP / SP 
$$
\mathrm{GPU} \text { 总数 }=D P \times T P \times P P \times E P \times S P
$$

| 缩写     | 全称                | 作用方向                    | 含义                                                |
| ------ | ----------------- | ----------------------- | ------------------------------------------------- |
| **DP** | Data Parallel     | 样本维度                    | 每个副本处理不同 batch 样本（模型参数相同，数据不同）                    |
| **TP** | Tensor Parallel   | 权重维度                    | 把单个模型层的权重矩阵拆分到多 GPU 上做矩阵乘                         |
| **PP** | Pipeline Parallel | 层级维度                    | 把模型的不同层切成几段，分配到不同 GPU 顺序执行                        |
| **EP** | Expert Parallel   | Mixture-of-Experts 模型专用 | 不同 GPU 负责不同的专家网络                                  |
| **SP** | Sequence Parallel | 序列维度                    | 把输入序列按 token 维度切分到不同 GPU（如 transformer block 内并行） |

---

### 0 快速结论

- 五维并行：**DP**（数据）/**TP**（张量）/**PP**（流水线）/**SP**（序列）/**EP**（专家）
- 参数选型黄金法则：`GPU总数 = DP × EP × TP × PP × SP`, 优先让通信最频繁的并行维度 靠近高带宽域 (HBD)（如单机 NVLink/PCIe Gen5/NVSwitch）
- 常见编排优先级（经验）：TP > SP > EP > PP > DP（通信密集在内层、通信稀疏在外层）
- 部署时优先关注 **推理吞吐** 与 **延迟**：

  * **vLLM**：动态 KV 缓存管理，支持张量并行/流水线并行。
  * **SGLang**：以 **高速 KV Cache 调度**、高效批处理 (batching) 为核心，支持 DP/TP。
  * **LMDeploy**：面向推理的并行调度器，结合 PP+TP+SP，支持长序列优化。

- 吞吐/显存/长序列/稀疏化的对应解法：
    - 仅追求吞吐：先 DP，必要时配合 ZeRO/FSDP
    - 模型单层太“胖”：TP
    - 模型层数太“高”：PP 配合微批填充气泡
    - 序列太“长”：SP（Megatron TP‑SP / Ulysses / Context Parallel）
    - 稀疏化 MoE：EP（All‑to‑All 为关键瓶颈）

---

### 1 术语与直观理解（推理）

- **DP（Data Parallelism）**：复制同一模型到多卡，分片数据并行训练；反向 AllReduce 梯度。通信次数少、量相对固定，可与反向重叠（ZeRO/FSDP）
- **TP（Tensor Parallelism）**：层内把大矩阵切到多卡（列并行/行并行；多头切分）。前后向频繁 AllReduce/AllGather，通信量大

- **PP（Pipeline Parallelism）**：按层把模型切分为多段（stage），微批（micro‑batch）流过各段；相邻 stage P2P 传激活/梯度。有气泡，需用足够微批填充

- **SP（Sequence Parallelism）**：沿序列维度（tokens）切分。Megatron 的 TP‑SP 把 LN/Dropout 等在 L 维并行；Ulysses/Context Parallel 提供全注意力并行的不同实现。常与 TP 结合

- **EP（Expert Parallelism）**：MoE 专家分布到多卡，token 由路由器分发；前后向各一次 All‑to‑All（通常共 2 次），负载均衡很关键。与 TP 结合时可做 group‑wise All‑to‑All 优化






### 2 DP / TP / PP / SP / EP 到三框架的「参数映射速查」

| 并行维度   | vLLM                                        | SGLang            | LMDeploy                     |
| ------ | ------------------------------------------- | ----------------- | ---------------------------- |
| **DP** | 多实例/Ray workers 聚合                          | 多实例副本 + 网关        | 多实例副本 + 网关                   |
| **TP** | `--tensor-parallel-size N`                  | `--tp N`（或等价）     | `--tp N`（TurboMind/ TRT‑LLM） |
| **PP** | `--pipeline-parallel-size P`（若可用）           | 视版本支持             | 视后端支持（与构建一致）                 |
| **SP** | Chunked prefill/KV 优化；（若支持）Context Parallel | 前缀缓存/分块注意力/版本相关并行 | paged‑KV/分块注意力（后端能力）         |
| **EP** | MoE/专家分片（版本相关；建议与 TP 对齐）                    | 版本相关              | 版本/后端相关（MoE 支持时配置专家组）        |

