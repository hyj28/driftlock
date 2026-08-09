# driftlock — 项目计划

> 长程 agent 的检查点 + 进度感知回滚层。
> 本文档是工作计划，随进展更新。公开面向的介绍见 `README.md`。

**最后更新**：2026-08-09

---

## 1. 一句话主张

同样的模型、同样的算力预算，加一层检查点 + 进度感知回滚，把长程终端任务的成功率从 X% 提到 Y%——并且能说清天花板在哪。

（X / Y 是待测量的占位符，实验出结果前不填。）

---

## 2. 为什么做这个

### 2.1 赛道选择

2026 上半年 17 万篇 AI 论文的统计显示，**长程规划（long-horizon planning）是增长最快的单一话题**：提及数从 264 涨到 1,611（+510%）。体量最大的是 agentic workflows（4,585 → 10,496）。更关键的结论是：agent 的**专项组件**正以整体赛道 2–5 倍的速度增长——领域已经从"能不能造 agent"翻篇到"怎么让 agent 规划、用工具、评判自己的输出"。

结论：**值钱的是组件层，不是又一个通用 agent。**

### 2.2 为什么不做"自进化 / 自写 skill"

那条线在 2026 上半年已经极度拥挤：SkillFoundry、SkillOpt、SkillComposer、SkillGen、SkillDAG、SkillX、SkillRL、EvoSkills、Trace2Skill、SkillWeaver、AutoSkill……半年十几篇，甚至已有专门基准 SkillResolve-Bench。走 happy path 会被一句"这和 SkillX 有什么区别"问死。

### 2.3 为什么攻击"目标漂移"

- **工业界公认是痛点、学术侧解法薄。** 搜索结果里主要是工程博客和 playbook（Wire、NxCode、Redis、digitalapplied 的回滚模式参考），arxiv 论文密度远低于 skill 那条线。
- **已有可直接沿用的度量。** ICLR 2026 的 *Asymmetric Goal Drift in Coding Agents* 已把目标漂移形式化，给出 `GD_actions`（做了不该做的）与 `GD_inaction`（该做的没做），并有配套压测 repo [jhammant/agent-drift](https://github.com/jhammant/agent-drift)。**沿用已发表指标，数字才有人信。**
- **有可攻击的定量规律。** 任务时长翻倍，失败率翻四倍。能把这条曲线压平一点，就是极干净的结论。
- **有清晰的能力鸿沟。** 人类专家 4 分钟内能做完的任务，前沿模型接近 100% 成功；人类要 4 小时以上的，成功率不到 10%。

### 2.4 公认的三个长程失败模式

1. **上下文腐化** —— 历史越长，有用信息越难被检索到；跨过某个利用率阈值后性能断崖下跌。对前沿模型和小模型一视同仁（Claude Sonnet 4、GPT-4.1、Qwen3-32B、Gemini 2.5 Flash 均随输入 token 增长而退化）。
2. **误差累积与目标漂移** ← **本项目攻击的目标**
3. 上述二者的交互

---

## 3. 技术方案

### 3.1 核心机制：检查点 + 进度感知回滚

把文件系统状态（Docker 层 + git）和 agent 状态一起做快照；一个独立判别器周期性地问"当前状态还值不值得继续"，不值就回滚到最近的健康检查点重试。

### 3.2 判别器：两级结构

| 层 | 做什么 | 成本 |
|---|---|---|
| **一级：启发式粗筛** | N 步无文件变更、动作循环、报错率突增、reward 停滞 | 零 token |
| **二级：LLM 精判** | 粗筛触发后，把（原始目标 + 当前计划 + 最近轨迹 + 文件 diff）交给便宜模型判断语义漂移 | DeepSeek V4-Flash，可忽略 |

这个结构的额外好处：**启发式 / 纯 LLM / 两级 天然就是三个消融组**，实验设计不用另想。

### 3.3 必须回答的杀手锏问题

> "你这套检查点回滚，跟失败了重跑一遍有什么区别？是不是就是多花算力换成功率？"

答案必须是：**判别器能在任务失败之前就识别出轨迹已经坏了**——早停 + 精准回滚到最后一个健康点，而不是全盘重来。因此**判别器的质量就是项目的质量**，也因此需要下面的等算力对照组。

---

## 4. 实验设计

### 4.1 基准

**LHTB（Long-Horizon Terminal-Bench）子集**，[repo](https://github.com/zli12321/LHTB) / [数据集](https://huggingface.co/datasets/IntelligenceLab/Long-Horizon-Terminal-Bench)。

关键事实（用于成本与时间估算）：

| 项 | 数值 |
|---|---|
| 全量规模 | 46 题 / 9 类 |
| 单任务 token | 约 990 万（平均 231 步，范围 120–320 步） |
| 单任务耗时 | 平均 69–93 分钟，预算上限 90 分钟 |
| 单模型跑完全套墙钟 | 53–71 小时 |
| 评分 | 连续 reward 0–1（含部分分），≥0.95 记为 solved |
| 验证 | 隐藏验证器 + 确定性重放 + 固定种子 |
| 最佳成绩 | Grok 4.5 平均 reward 0.505，46 题解出 13 题 |
| 无人解出 | 29/46；782 次运行里仅 7% 达到 solved |
| 成本参考 | MiniMax M3 $6.13/任务；Claude 系 $38–73/任务 |

**取 8–12 题子集**，按第 1 周实测的部分分挑选——必须避开 29 道无人解出的题（只会给地板效应）。保留官方 harness 与隐藏验证器，数字可对标榜单。

任务结构标准化，五件套：`task.toml` / `instruction.md` / `environment/` / `tests/` / `solution/`。

### 4.2 四组对照

| 组 | 作用 |
|---|---|
| 1. 无干预基线 | 参照系 |
| 2. **等算力朴素重试** | 给 baseline 同样的 token 预算去盲重试——**一拳打死"你只是多花算力"** |
| 3. 检查点 + 回滚（本方法） | 主张 |
| 4. **预言机上界** | 用事后上帝视角的完美判别器测天花板——说明判别器还差多少 |

### 4.3 指标

- 成功率（LHTB 连续 reward，含部分分）
- `GD_actions` / `GD_inaction`（沿用 ICLR 2026 定义）
- 每任务 token 成本
- **任务时长 vs 失败率曲线的斜率**（能压平这条曲线是最强结论）

---

## 5. 环境与成本

### 5.1 计算宿主：便宜 x86 云主机

**原因**：LHTB 镜像基本都是 amd64-only，官方要求 Apple Silicon 用户设 `DOCKER_DEFAULT_PLATFORM=linux/amd64`，即走模拟。本地 M4 Air（24GB、无风扇）整夜满载会严重降频，且部分镜像可能直接跑不起来。

**分工**：本机写代码 + 分析数据；云主机跑实验。

### 5.2 模型

| 用途 | 模型 | 价格（每百万 token） |
|---|---|---|
| agent 主体 | DeepSeek **V4-Pro** | 输入 $0.435 / 缓存命中 $0.003625 / 输出 $0.87 |
| LLM 判别器 | DeepSeek **V4-Flash** | 输入 $0.14 / 缓存命中 $0.0028 / 输出 $0.28 |

**缓存命中价便宜 50–120 倍，这对本项目是决定性的**——agent 循环每步都在重发几乎相同的历史，缓存命中率天然极高。

### 5.3 预算（$100 硬约束）

按 12 题 × 4 组、缓存命中率约 90% 估算：

- 输入约 4.76 亿 token：10% 未命中 ≈ $20.7 + 90% 命中 ≈ $1.6
- 输出约 1,200 万 token ≈ $10.4
- **一轮完整四组实验 ≈ $33**

加上云主机两三个月 $30–45，**$100 只够 1–2 轮完整实验 + 开发期零散调试**。

**对策：先用 8 道题起步**，把第一轮压到 $20 出头，留出改判别器再跑第二轮的余量。结论稳了再扩到 12 题。

---

## 6. 阶段与卡点

| 阶段 | 内容 | 卡点（不过就调整方案） |
|---|---|---|
| **第 1 周** | 云主机搭起来，跑通 2 道原版 LHTB 题；用 V4-Pro 跑约 20 道题筛选，按**实测部分分**挑子集 | 若 V4-Pro 在所有题上都接近零分 → 地板效应，必须换更强模型或更简单的题 |
| **第 2–3 周** | fork Harbor，插入检查点 / 快照 / 回滚骨架；先做纯启发式判别器 | 若 fork 改不动 → 退回 AgentCE-Bench |
| **第 4–5 周** | 加 LLM 精判层；跑第一轮完整四组 | 若本方法打不过等算力重试 → 立刻改判别器，别硬跑 |
| **第 6–7 周** | 消融实验、失败案例分析、预言机上界 | — |
| **第 8 周** | 开源库打包（pip 可装、一键复现脚本）+ 技术博客 | — |

---

## 7. 风险登记

| # | 风险 | 影响 | 缓解 |
|---|---|---|---|
| 1 | **地板效应**：V4-Pro 太弱，改进空间被压死 | 项目无结论 | 第 1 周按实测部分分筛题；备选换更强模型 |
| 2 | **fork harness 工程量被低估**：Harbor / Terminus-2 无 agent 插件接口，回滚层要硬插进别人的循环 | 进度延后 2+ 周 | 第 2 周结束设卡点；备选 AgentCE-Bench |
| 3 | **判别器不够准**：判断与随机无异，四组曲线重叠 | 项目无结论 | 预言机组提前暴露该问题 |
| 4 | **amd64 模拟**：镜像跑不起来或过慢 | 无法迭代 | 已改用 x86 云主机规避；第 1 周先验证 |
| 5 | **墙钟时间**：12 题 × 4 组 = 48 次运行 × 上限 90 分钟 | 一天一轮实验 | 云主机并行；必要时缩到 8 题 |
| 6 | **预算超支**：一轮 $33，只够 1–2 轮 | 无法迭代 | 8 题起步；开发期用 V4-Flash |

---

## 8. 交付物

1. **开源库** —— 能被别人套到自己 agent 上的回滚中间层：pip 可装、有 README、有一键复现脚本
2. **深度技术博客** —— 四组曲线 + 失败案例分析 + 判别器设计取舍

---

## 9. 关键参考

**基准与 harness**
- [LHTB — Long-Horizon Terminal-Bench](https://zli12321.github.io/LHTB/) · [GitHub](https://github.com/zli12321/LHTB) · [HF 数据集](https://huggingface.co/datasets/IntelligenceLab/Long-Horizon-Terminal-Bench)
- [AgentCE-Bench](https://arxiv.org/pdf/2604.06111) —— 备选：horizon 可调、轻量环境
- [LOCA-bench](https://arxiv.org/pdf/2602.07962) —— 可控的极端上下文增长
- [LongHorizon-Harness](https://arxiv.org/html/2608.01964)

**失败模式与度量**
- [Beyond the Leaderboard: 工具使用、规划与推理失败综述](https://arxiv.org/pdf/2607.05775)
- [The Long-Horizon Task Mirage?](https://arxiv.org/html/2604.11978v1)
- *Asymmetric Goal Drift in Coding Agents*（ICLR 2026）—— `GD_actions` / `GD_inaction` 来源
- [jhammant/agent-drift](https://github.com/jhammant/agent-drift) —— 漂移压测

**工程参考**
- [Agent Rollback and Checkpoint Patterns](https://www.digitalapplied.com/blog/agent-rollback-checkpoint-patterns-2026-engineering-reference)
- [Agent drift: why long-running AI agents lose the plot](https://usewire.io/blog/agent-drift-why-long-running-ai-agents-lose-the-plot/)
- [Long-Horizon Agent Trajectory Governance Playbook](https://www.nxcode.io/resources/news/long-horizon-agent-trajectory-governance-playbook-2026)

**相邻工作（注意区分边界）**
- [Self-Compacting Language Model Agents](https://arxiv.org/pdf/2606.23525) —— 上下文压缩方向
- [A Survey of Self-Evolving Agents](https://arxiv.org/abs/2507.21046)

**定价**
- [DeepSeek 官方定价](https://deepseek.ai/pricing) · [汇总](https://benchlm.ai/deepseek/api-pricing)
