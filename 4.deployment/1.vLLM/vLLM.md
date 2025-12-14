
### vLLM 示例

#### 配置说明

* **TP**：`--tensor-parallel-size`，指定张量并行 GPU 数。
* **PP**：`--pipeline-parallel-size`，切分模型层数到多 GPU。
* **SP**：长序列推理时通过 `--max-num-batched-tokens` 结合 **PagedAttention** 达到类似效果。
* **DP**：通过多进程 + 负载均衡器实现，常见于生产部署。

#### 启动命令示例

```bash
python -m vllm.entrypoints.openai.api_server \
  --model facebook/opt-13b \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 2 \
  --max-num-batched-tokens 4096
```

> 含义：13B OPT 模型部署在 4 卡（2TP × 2PP），并支持批量请求的动态 KV Cache 管理。



### 5.1 vLLM（高吞吐推理，PagedAttention/连续批推理）

**并行映射**

* **TP**：`--tensor-parallel-size N`（单层切分，多卡并行）。
* **PP**：部分版本提供 `--pipeline-parallel-size P`（按层分段）。
* **DP**：多副本（多进程/多主机）+ 负载均衡（如 Ray/容器编排），对外一个网关。
* **SP**：vLLM 以 **PagedAttention**、**Chunked Prefill**、**分块KV构建** 等手段缓解长序列显存；若使用 **Context Parallel**/TP‑SP，请与所用分支文档对齐。
* **EP**（MoE）：新版本支持 MoE 推理与专家切分（具体开关以版本为准）；通常建议 **EP 组与 TP 组对齐**，尽量在 HBD 内完成专家路由。

**单机多卡（TP）示例**

```bash
# 基础 API 服务（单机 4 卡，TP=4）
python -m vllm.entrypoints.api_server \
  --model <hf_model_or_path> \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --port 8000
```

**单机多卡（TP+PP）示例**

```bash
# 当模型很“高”且版本支持 PP 时
python -m vllm.entrypoints.api_server \
  --model <hf_model_or_path> \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --max-model-len 8192
```

**多机扩展（DP × TP × PP）示例**

```bash
# 以 Ray 为例，多个 worker 组成 DP，worker 内继续 TP/PP
python -m vllm.entrypoints.api_server \
  --model <hf_model_or_path> \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --ray-address <ray://head:10001> \
  --num-gpu-workers <workers>
```

**MoE/EP 提示**

* 若版本提供专家并行开关（例如 `--moe-…` 选项），优先设置 **EP ≤ 单机可用 GPU** 并与 TP 对齐，启用 **group‑wise 路由** 以减少跨组 all‑to‑all。

**实践建议**

* 把 `TP/PP` 压在 **NVLink/NVSwitch** 内；`DP` 通过多实例 + 网关汇聚。
* 长上下文优先开启 **chunked prefill/kv 压缩/CPU Offload**；监控 KV 命中率与显存峰值。

