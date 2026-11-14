# GenEvolve 2B：ReActOPSD（4×A100 80GB）

在 **GenEvolve 的 Qwen3-VL-2B-Instruct SFT checkpoint-550** 后接两轮 Route-SFT，再接相同的 GRPO 与本方法的 Actor-routed OPSD。与 2B 基线共享同一份原始 SFT 权重；不重新做冷启动 SFT，不改工具、反馈、渲染或评分。

对照来源：[2B GRPO](https://github.com/leiye302/genevolve-2b-grpo-a100-4x/tree/2f197591100d8b0bf61e65be9748a0752f91ae30) · [2B GenEvolve 自进化及 SFT](https://github.com/leiye302/GenEvolve-2B-Reproduction/tree/57263f0583c633c3d515913b9935c3850cdf798e) · [作者代码](https://github.com/MeiGen-AI/GenEvolve/tree/23c847c559ccc0f95bbf4b3d8925898463822f4c)。做配对比较时，用 baseline 实际读取的同一份 HF 导出，逐文件比较权重 SHA-256，不能只比较模型名、目录名或训练步数。

## 与对照组共用的设置

| 项目 | 固定口径 |
| --- | --- |
| 冷启动 | 同一份 GenEvolve 2B SFT-550；仅方法组再做 Route-SFT |
| 数据与 rollout | 固定版本 GenEvolve-Data-RL，训练 2,575 题；seed 42，每步 8 题 × 每题 6 条轨迹，四卡各负责 2 个完整题组 |
| 采样 | temperature 0.7，top-p 0.95，最多 11 回合／每回合 4,096 新 token，整轨迹 30,000，初始输入 6,144，上下文 65,536 |
| GRPO | 原组内标准化、0.20/0.28 非对称裁剪、轨迹 KL 惩罚系数 0.001、程序奖励权重 0.5、LR 1e-6、cosine；保留基线 FSDP 布局 |
| 保存与续训 | 二阶段每 10 步和最后一步存，保留最近两份可续训断点；另导出最终 HF 模型 |
| 渲染与评分 | 固定 Qwen-Image-Edit-2511，40 步、CFG 4、seed 0、最多两张参考图；基线的 `qwen3.7-flash-2026-07-15` 奖励和 KScore 评审 |
| 终评 | GenEvolve-Bench 同序 594 题、KScore、原版 prompt fallback；temperature 0.6、top-p 0.9、逐题 seed=42+序号；可选同版 RL-val 600 或 WISE 1000 |

二阶段默认与两个对照入口一样：`ceil(2575 / 8) = 322` updates。传 `--steps N` 可指定配对预算；**三组必须使用相同的计划步数**，因为 cosine 也以该步数为周期。在计划跑 322 步但停于 30 步，与从头设 `--steps 30` 不等价。实际运行长度以每组运行清单为准。

## 一阶段：6 Good + 2 Bad，恰好两轮

封版包共 2,376 条：1,782 Good / 594 Bad。Bad 含 394 条自然错误和 200 条同状态构造对照（与源 Good 同 batch）。Good 保留真实采样动作；Bad 使用原生 `<tool_call>…</tool_call>` 或 `<answer>…</answer>` 纠正后缀，前面加一个注册后的 `<RETHINK>`。训练与 rollout 保持同一文本工具历史，不给 Actor 插入额外图像观察或换解析器。

**数据来源与训练起点不同。** 保留之前选定的复审 v3 数据，其离线轨迹由 GenEvolve 8B 采集，不声称是 2B 自采。分开记录 8B 采集来源和 2B 实际训练权重的 SHA-256。启动时用传入的 2B SFT-550 tokenizer 核对全部历史 token、边界和标签解码；若不一致就报错，不重分词、不跳过样本。

- mask 至 `</think>\n`（含分隔符），只监督动作后缀及 EOS。用首个目标 token 的预测位置，在**全词表**上算 marker BCE；逐样本平均后缀 CE + 0.1 × BCE，无 reasoning CE、无 KL、无 LoRA。
- 按原序扫描 297 个固定批次两遍：全局 8，四卡 × micro 1 × 累积 2，**594 updates**。LR 2e-6，前 10% warmup、cosine、AdamW (0.9, 0.95)、weight decay 0.01、clip 1.0、seed 0。语言参数全参训练，视觉塔不动。
- 最长 11,223 tokens；使用 12,288 上限，超长直接报错，不重分词或截断。每轮存原生可续训状态和 HF 导出。该封版包只供训练，没有另造 SFT val split；日志中的 train loss 和给定前缀 marker 概率不是自由生成指标。

## 二阶段：GRPO + Actor-routed OPSD

完整轨迹做 GRPO。Teacher 是**当前这轮 Actor 的无梯度副本**，评分同一批原始生成 token：

| Actor 的该回合 | Teacher 的额外条件 |
| --- | --- |
| 无特殊 token `<RETHINK>` | GenEvolve 通用静态 skill |
| Actor 生成特殊 token `<RETHINK>` | 同一静态 skill + 该回合的局部 hint；缺失、超长或引用当前未见图像时退回静态 skill |

detach Teacher 和门控 `sigmoid(5 × (logp_teacher − logp_student))`；按有效 token 平均，系数 **0.01**。方法版不叠加对照的 GenEvolve 经验库／top-10% SDL。基线的 frozen reference 只用于原奖励 KL，不是蒸馏 Teacher。局部 API 每条带 marker 的轨迹发一组请求（失败有限重试）；只给指定回合提示，不打 Good/Bad、不替 Actor 执行动作、不做 episode 经验总结。Student、工具、终评均不读 hint。marker 格式和位置只统计，不作为路由门禁。

## 运行

Linux x86-64、Python 3.11、CUDA 12.8 和 C/C++ 构建工具。沿用对照的四张 A100 80GB、FSDP 和 CPU offload；使用同版独立渲染服务，或另开渲染机执行 `bash scripts/setup.sh --renderer`、`bash scripts/serve_renderer.sh`。每个训练阶段首次启动至少需 100 GiB 可用空间；两阶段同盘时还要计入已保存的一阶段产物、数据、生成图以及渲染器缓存。实际峰值尚未在四卡验证。

运行方提供两个独立输入，不从作者机器读取：

- `PARENT`：对照组实际使用的 **GenEvolve 2B checkpoint-550 完整 HF 导出**，可指向 [2B 复现仓库](https://github.com/leiye302/GenEvolve-2B-Reproduction/tree/57263f0583c633c3d515913b9935c3850cdf798e)的 `WORK/sft` 或同文件的 `checkpoint-550`。须含权重、tokenizer、processor、`trainer_state.json`；不能传原始基模、8B、二阶段权重或只有 ZeRO 状态的目录。
- `ROUTE_DATA`：复审封版目录 `route_sft_6g2b_final_v3_reviewed`，必须包含 `manifest.json`、`train.jsonl`、`batches.jsonl`，与代码分开交付。仓库只接受该包的 SHA；旧 v2 不兼容。若此前用 v2 启动过训练，请用新的 `WORK` 目录，不要跨数据版本续训。`check-route` 核对文件 SHA、真实 tokenizer 输入和解码，不占卡、不调用 API。

权重和 v3 训练包不在公开 Git 仓库内。从实验交接取得这两个输入，放在运行机器可访问的目录。可用本仓库的 `check-sft --sft-checkpoint "$PARENT"` 分别读取方法和基线的 HF 导出，逐文件核对 SHA-256。方法端也会把指纹记入运行清单，续训和自动衔接必须保持同一份 2B 起点。[v3 的 8B 采集来源指纹](configs/route_data_source_8b_sha256.json)仅用于校验数据，不用于加载模型。

```bash
git clone https://github.com/leiye302/genevolve-2b-rethink-a100-4x.git
cd genevolve-2b-rethink-a100-4x
bash scripts/setup.sh
source .venv/bin/activate
export WORK=/path/to/independent-run
export PARENT=/path/to/genevolve-2b-sft/checkpoint-550
export ROUTE_DATA=/path/to/route_sft_6g2b_final_v3_reviewed
# 使用自己的四张获配 GPU；由调度器指定时保持它的映射。
python run.py check-route --parent "$PARENT" --data "$ROUTE_DATA"
```

二阶段和评测需要 Serper + 基线 Judge；仅方法训练额外需要局部 hint API（`glm-5.3-flash`、低思考、HTTPS OpenAI-compatible chat completions；OpenCode Go 每条轨迹使用独立会话 ID）。先确认服务商允许接收训练轨迹。hint 请求只带本条 rollout 的文本证据和指定步号，不带真值图或本地图片路径。凭据通过环境注入，不写进 Git 或命令行。详见 [环境变量模板](configs/services.env.example)。

```bash
export QWEN_SERVICE_URL=http://render-host:8001
export GENEVOLVE_HINT_API_URL=https://opencode.ai/inference/go/openai/v1/chat/completions
read -rsp 'Serper key: ' SERPER_API_KEY; printf '\n'
read -rsp 'Judge key: ' JUDGE_API_KEY; printf '\n'
read -rsp 'Hint key: ' GENEVOLVE_HINT_API_KEY; printf '\n'
export SERPER_API_KEY JUDGE_API_KEY GENEVOLVE_HINT_API_KEY
# 准备官方数据 → 一阶段 Route-SFT → 核验 594 步导出 → 二阶段
python run.py --work "$WORK" all --parent "$PARENT" --data "$ROUTE_DATA"
# 可选独立终评；或在 all 命令末尾加 --eval，训练完成后自动做全量终评。
python run.py --work "$WORK" eval --split bench
```

只训练一阶段时，可单独执行 `python run.py --work "$WORK" route-sft --parent "$PARENT" --data "$ROUTE_DATA"`，无需 API；手动接二阶段时先执行 `python run.py --work "$WORK" prepare`，再执行 `python run.py --work "$WORK" rethink --sft-checkpoint "$WORK/route_sft/hf-step-00594"`。重新执行 `all` 会复用已验收的官方数据和一阶段最终导出；若一阶段尚未结束，则从已保存的 epoch 断点续训。任何一阶段报错或导出校验失败均不会开二阶段。

同 2B 纯 GRPO 对照可用本仓库的 `grpo --sft-checkpoint "$PARENT"`，写入 `WORK/grpo`，不调用 hint API。GenEvolve 的 GRPO+SDL 仍在独立对照仓库。原冷启动和 Route-SFT 两轮的 HF 模型可复用同一 `eval --policy ... --split bench`；小规模试测可用 `--max-cases N`，不应混入正式终评。重启训练使用原命令、同一 WORK、数据、起点、计划步数与服务配置；方法二阶段只接受两轮完成且通过 SHA 检查的 Route-SFT 导出。不要把 8B 的工作目录或二阶段断点用于 2B。逐步指标、rollout、运行清单与续训状态保留在 WORK。

终评在不改分数的前提下另写 `route_telemetry.json`，按决策回合统计原生特殊 marker，并单列位置正确率，不按总 token 数作分母。RL-val 同样记录。

## 离线检查

```bash
bash scripts/setup.sh --sources-only
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-smoke.txt
python -m unittest discover -s tests -v
```

离线检查以 mock 工具、Judge、hint 覆盖 rollout、解码、Teacher 输入、梯度和断点写入。真实 GPU 训练、API 质量和图像评分需在目标机验证。
