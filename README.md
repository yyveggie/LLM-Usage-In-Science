# llm-usage-in-science：科学写作中 LLM 使用量的复现与换 LLM 实验

复现三篇论文共用的核心方法 **distributional LLM quantification（分布式 LLM 使用量估计）**，并提供一个**可插拔的"替换生成 LLM"实验框架**：把原论文构造 AI 分布 Q 的模型（GPT-3.5）换成任意 OpenAI 兼容模型（GPT-4o / Claude / Qwen / DeepSeek 等），看结论是否变化。

## 三篇论文

三篇是一条方法链：

**① Mapping the Increasing Use of LLMs in Scientific Papers**（Liang 等，Stanford，COLM 2024）
- 数据：2020.1–2024.2，arXiv / bioRxiv / Nature 共 95 万篇论文的摘要与引言。
- 方法：提出并落地分布式 α 估计（人类分布 P、AI 分布 Q 的混合 `D_α=(1-α)P+αQ`，MLE 求 α），并用"两阶段生成"（真人段落→提纲→扩写，GPT-3.5）构造 AI 训练语料。
- 发现：LLM 使用稳步上升，**计算机科学最快最高（摘要 α≈17.5%）**，数学/Nature 最低；高使用关联第一作者高产、领域拥挤、论文更短。

**② Quantifying Large Language Model Usage in Scientific Papers**（同团队，Nature Human Behaviour 2025）
- ①的期刊扩展版：数据扩到 112 万篇、时间延到 2024.9，方法一致。
- 发现：CS 摘要 α 升到 **22.5%**；新增区域分析——**非英语母语地区（中国、欧洲大陆）使用率更高**；摘要/引言/结论的 α 高于方法/实验部分。
- 项目里 `run_02`（会议版终点）和 `run_03`（期刊版终点）复现的就是这条线。

**③ AI-Assisted Writing Is Growing Fastest Among Non-English-Speaking and Less Established Scientists**（Liu 等，UW-Madison + 北大）
- 不同团队沿用同一套估计方法（直接复用②的 bioRxiv `p_t/q_t`），换数据为 **PubMed Central 200 万篇全文** + OpenAlex 作者画像，并用 **DiD/DDD 回归**。
- 发现：ChatGPT 后 AI 写作激增，**非英语国家增长约 400% vs 英语国家约 183%**；英语水平越低增幅越大（ρ=-0.65）；论文少/被引少/资历浅/低排名机构的作者用得最多。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[llm-swap]"     # 只跑复现可用 pip install -e .
python -m spacy download en_core_web_sm   # 换 LLM 实验的分词对齐需要
```

安装后所有脚本以模块方式运行（包：`core` / `reproduction` / `llm_swap` / `data_tools`）。

## 三篇论文的实验操作

三个入口直接放在项目根目录，彼此独立，`python xxx.py` 即可运行（已内置路径引导，无需 `pip install` 也能跑，只要装好依赖）。实验 2/3 需先按下文下载官方数据；实验 1 无需外部数据。

**实验 1 — AI-Assisted Writing**：分布式 α 估计器（synthetic 已知-α 验证）+ DiD/DDD 回归 demo。

```bash
python run_01_ai_assisted_non_english_core.py
```

**实验 2 — Mapping（会议版，终点 2024.02）**：官方 distribution + 月度 inference 上估计 α，默认对比 `2022_11` vs `2024_2`。

```bash
python run_02_mapping_increasing_use_core.py
```

**实验 3 — Quantifying（期刊版，终点 2024.09）**：同法，终点 `2024_9`；`--all-months` 出完整月度趋势。

```bash
python run_03_quantifying_usage_core.py
python run_03_quantifying_usage_core.py --all-months
```

结果写入 `results/0{1,2,3}_*.csv`。预期：CS 增幅最大（会议版摘要 α≈17.5%、期刊版≈22%），Math/Nature 最低，`2022_11` 各 venue α≈2–3%（低假阳性）。

## 数据下载与官方验证

官方数据默认放在 `data/official_data/`（已 `.gitignore`，需自行下载）：

```bash
bash src/data_tools/download_official_cs_data.sh          # 仅 CS distribution + validation
python -m data_tools.download_official_inference_data     # 全 venue（约 520MB，断点续传）
python -m reproduction.official_cs_validation             # 官方 CS 已知-α 验证
```

## 换 LLM 实验

核心：α 估计中只有 **AI 分布 Q（`q_t`）** 与"用哪个 LLM"绑定。换 LLM = 只重算 `q_t`，复用官方 `p_t`，其余链路不变。模型在 `config/models.yaml` 配置，API key 走环境变量。

```bash
# 1) 采集真人种子段落（ChatGPT 之前的摘要，带标点；正式做法）
#    CS 场景（arXiv）：
python -m data_tools.collect_arxiv_human --categories cs.CL cs.LG cs.CV \
  --end-date 202211290000 --max-papers 500 --output data/human_corpus/arxiv_cs.jsonl
#    生物医学场景（bioRxiv，对应第三篇）：
python -m data_tools.collect_biorxiv_human \
  --end-date 2022-11-29 --max-papers 500 --output data/human_corpus/biorxiv.jsonl

# 2) 用指定 LLM 生成 AI 语料（不采集时可用 --source official 兜底）
export OPENAI_API_KEY=...
python -m llm_swap.build_ai_corpus --model gpt-4o \
  --source jsonl --input data/human_corpus/arxiv_cs.jsonl --venue CS --limit 300

# 3) 拟合该模型分布（复用官方 p_t，输出官方 schema）
python -m llm_swap.fit_swapped_distribution --model gpt-4o --venue CS \
  --ai-corpus results/ai_corpus/gpt-4o_CS.parquet

# 4) 跨模型在官方真实语料上对比 α
python -m llm_swap.run_llm_swap --models official gpt-4o --venues CS \
  --months 2022_11 2023_6 2024_2

# 5) 单个模型的已知-α 验证（复用官方脚本）
python -m reproduction.official_cs_validation \
  --distribution results/swapped_distribution/gpt-4o/CS.parquet
```

输出的 `q_t` 与官方 schema 一致（`Word/logP/logQ/log1-P/log1-Q`），可无缝接入 `run_02/03` 与官方验证脚本。

**分词对齐**：`q_t` 的准确性依赖 AI 语料分词与官方词表一致。`build_ai_corpus` 默认用 spaCy（`--tokenizer spacy`，对齐官方），跑前可先核验重合度：

```bash
python -m data_tools.inspect_official_vocab --venue CS --source jsonl --input data/human_corpus/arxiv_cs.jsonl
```

选重合度更高的分词器；并用第 6 步的官方已知-α 验证（误差应 ≤3.5%）确认对齐是否到位。
