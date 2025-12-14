
### 3 SGLang 示例

#### 配置说明

* 核心是 **高速批处理 (batching)** 和 **KV 缓存调度**。
* **DP**：通过多副本服务实现。
* **TP**：通过 `--tp-size` 启动参数指定。
* **SP**：长上下文处理通过 KV 分片实现。

#### 启动命令示例

```bash
python3 -m sglang.launch_server \
  --model mistralai/Mistral-7B-v0.1 \
  --tp-size 2 \
  --dp-size 2 \
  --max-context-len 8192
```

> 含义：Mistral‑7B 部署在 4 卡（2TP × 2DP），支持 8192 token 上下文

---


### 5.2 SGLang（高并发低延迟推理，程序化编排）

**并行映射**

* **TP**：通过启动参数或配置指定张量并行度（例如 `--tp N`/`--tensor-parallel-size N` 风格）。
* **DP**：多实例水平扩展（K8s/进程多副本），上层网关/服务发现做汇聚。
* **SP**：支持高效前缀/缓存、批处理与（版本相关的）上下文并行策略，用于长序列与多会话复用。
* **PP/EP**：是否支持以及开关名与版本相关；若启用，请遵循“通信密集在内层”的编排策略。

**单机多卡（TP）示例**

```bash
# 启动 SGLang 服务（示例 CLI，具体参数以所用版本为准）
sglang serve \
  --model <hf_model_or_path> \
  --tp 4 \
  --max-model-len 8192 \
  --port 30000
```

**多实例（DP）扩展**

```bash
# 通过多副本水平扩展（示例：K8s Deployment/不同节点各跑一份）
# 外部通过 Ingress/Service/自研网关做负载均衡，实现 DP。
```

**程序化并行（路由/批处理）**

* 利用 SGLang 的 **调度/批处理脚本** 把同形状请求合批，降低跨卡通信放大效应。
* 对长上下文任务，优先启用 **前缀缓存** 与 **分块预填充**。

---