
#### 🧩 环境与运行配置

| 项目                                                 | 值 / 含义                                                                              |
| -------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `VLLM_USE_V1=0`                                    | 禁用 `custom_logits_processors`                                                       |
| `MINERU_MODEL_SOURCE=modelscope`                   | 模型来源于 ModelScope 平台                                                                 |
| `CUDA_VISIBLE_DEVICES=0,1`                         | 使用 GPU 0 和 1                                                                        |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 启用 CUDA 可扩展内存段                                                                      |
| 启动命令                                               | `mineru-vllm-server --gpu-memory-utilization 0.3 --host 0.0.0.0 --port 30000 -tp 2` |


#### ⚙️ 模型与架构信息
| 参数 / 信息                                  | 内容                                                                              |
| ---------------------------------------- | ------------------------------------------------------------------------------- |
| 模型路径                                     | `/home/buding/.cache/modelscope/hub/models/OpenDataLab/MinerU2___5-2509-1___2B` |
| 模型架构                                     | `Qwen2VLForConditionalGeneration`                                               |
| `torch_dtype`（已弃用）                       | 被替换为 `dtype`                                                                    |
| 实际使用 `dtype`                             | `torch.float16`（因 RTX 2080 Ti 不支持 `torch.bfloat16`）                             |
| `max_model_len`                          | `16384`                                                                         |
| vLLM 版本                                  | `0.10.2`                                                                        |
| Tensor Parallel (`tensor_parallel_size`) | `2`                                                                             |

#### 🧠 多模态与序列长度警告


| 参数                    | 问题 / 建议                                                                                                             |
| --------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `max_model_len=16384` | 小于多模态最坏情况所需 token 数（18225）                                                                                          |
| 警告                    | `Token indices sequence length is longer than the specified maximum sequence length for this model (18225 > 16384)` |
| 建议                    | 增大 `max_model_len` 或减小 `mm_counts`，否则多模态推理可能失败                                                                      |
| `max_num_seqs`        | 自动被设置为最小值 1（因 16384 // 18225 < 1）                                                                                   |

#### ⚡ GPU 与内存配置（gpu_memory_utilization）

| 参数 / 指标                                            | 值 / 说明                                                                                       |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| GPU 型号                                             | NVIDIA GeForce RTX 2080 Ti                                                                   |
| `gpu_memory_utilization`                           | `0.3`（即 30%）                                                                                 |
| 单卡总显存 (`total_gpu_memory`)                         | ~21.5 GiB                                                                                    |
| 可用预算 (`total_gpu_memory * gpu_memory_utilization`) | ~6.44 GiB                                                                                    |
| `model weights`                                    | 1.09 GiB                                                                                     |
| `non_torch_memory`                                 | 0.13 GiB                                                                                     |
| `PyTorch activation peak memory`                   | 1.84 GiB                                                                                     |
| KV Cache 剩余                                        | ~3.38 GiB                                                                                    |
| 调整建议                                               | 可通过：<br>• `--kv-cache-memory=3062017740`（与配置匹配）<br>• `--kv-cache-memory=18929574400`（充分利用显存） |


🔄 并行与通信配置

| 项目                       | 说明                                            |
| ------------------------ | --------------------------------------------- |
| `tensor_parallel_size=2` | 两卡张量并行                                        |
| 通信库                      | `nccl==2.27.3`                                |
| `Custom AllReduce`       | 被禁用，原因：GPU 间缺乏 P2P 能力                         |
| 可选参数                     | `--disable-custom-all-reduce=True`            |
| `ProcessGroupGloo` 警告    | 无法解析主机名，使用回环地址。可通过 `GLOO_SOCKET_IFNAME` 指定接口。 |


🧮 CUDA 图与执行模式
| 参数                   | 含义 / 建议                                                        |
| -------------------- | -------------------------------------------------------------- |
| `use_cudagraph=True` | 启用了 CUDA 图捕获（提升推理效率）                                           |
| 捕获耗时                 | ~9 秒，占用 0.38 GiB                                               |
| 若遇 OOM               | 降低 `gpu_memory_utilization`，或使用 `--enforce-eager` 切换到 eager 模式 |
| 其它调优项                | 可减少 `max_num_seqs` 以降低显存使用量                                    |


#### 🚨 关键警告与建议汇总
| 类型                   | 内容 / 解决方案                                         |
| -------------------- | ------------------------------------------------- |
| 多模态输入过长              | 增加 `max_model_len` 或减少 `mm_counts`                |
| GPU P2P 不支持          | 加 `--disable-custom-all-reduce=True` 以抑制警告        |
| 显存利用率偏低              | 调整 `gpu_memory_utilization` 或 `--kv-cache-memory` |
| CUDA 图潜在风险           | 可使用 `--enforce-eager` 避免非静态模型问题                   |
| dtype 降级警告           | RTX 2080 Ti 不支持 `bfloat16`，自动使用 `float16`         |
| FlashAttention-2 不可用 | 因 GPU 属于 Turing 架构，改用 `XFormers backend`          |
