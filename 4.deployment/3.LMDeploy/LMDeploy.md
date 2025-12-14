

### 4 LMDeploy 示例

#### 配置说明

* **TP**：张量并行通过 `tp` 参数配置。
* **PP**：流水线并行 `pp` 参数。
* **SP**：LMDeploy 已支持 **Context Parallel**，用于极长序列推理。
* **DP**：多进程副本部署。

#### 启动命令示例

```bash
lmdeploy serve api_server \
  internlm/internlm-chat-20b \
  --tp 2 --pp 2 --sp 1 --dp 2 \
  --max-batch-size 32 --max-seq-len 8192
```

> 含义：InternLM‑20B 部署在 8 卡（2TP × 2PP × 2DP），支持长序列上下文推理。

---



### 5.3 LMDeploy（TurboMind/TensorRT‑LLM 背端加速）

**并行映射**

* **TP**：TurboMind/TensorRT‑LLM 支持多卡张量并行（常用 `--tp N`/配置文件设定）。
* **PP**：部分模型/版本支持按层流水线切分（需与导出/构建阶段一致）。
* **DP**：按实例副本数水平扩展，对外统一网关。
* **SP**：依赖后端的 KV 切分/分块注意力等能力；长序列需结合 **paged‑KV/quant‑KV**。
* **EP**：针对 MoE 模型的专家切分与路由（取决于后端与模型支持）。

**单机多卡（TP）示例**

```bash
# 以 TurboMind 后端的 API Server 为例
lmdeploy serve api_server <hf_model_or_path> \
  --tp 4 \
  --max-seq-len 8192 \
  --port 23333
```

**离线构建 + 部署（TP/PP 对齐）**

```bash
# 先按目标并行度导出/构建引擎，再以同样的 TP/PP 启动服务
lmdeploy convert turbomind \
  --model <hf_model_or_path> \
  --tp 4  # 与运行时保持一致

lmdeploy serve api_server <converted_model_dir> --tp 4
```

**MoE/EP 提示**

* 若使用 MoE 权重，确保构建阶段开启相同的 **专家并行/路由策略**；尽可能让专家路由停留在 **单机/同交换域**。

---