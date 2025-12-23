## Open Compass
- Document: https://opencompass.readthedocs.io/en/latest/
- Installation: https://opencompass.readthedocs.io/en/latest/get_started/installation.html

>环境准备
```sh
conda create --name opencompass_lmdeploy python=3.10 -y

conda activate opencompass


# For support of most datasets and models
pip install -U opencompass

# Complete installation (supports more datasets)
pip install "opencompass[full]"

# API Testing (e.g., OpenAI, Qwen)
pip install "opencompass[api]"

# Model inference backends. Since these backends often have dependency conflicts,
# we recommend using separate virtual environments to manage them.
pip install "opencompass[lmdeploy]"
pip install "opencompass[vllm]"
```