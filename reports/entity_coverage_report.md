# Entity Extraction Coverage Report

> Generated: 2026-05-18  Pipeline: v1.0.0

## 1. 数据规模

| 指标 | 数值 |
|------|------|
| FairCV 简历 | 700 篇 |
| JD 招聘文档 | 588 篇 |
| 句子总数 | 72139 句 |
| Mention 总数 | 24776 条 |

## 2. 精确率估计（P）

抽样 100 条 L1 rule_match，其中 84 条词面在实体别名表中精确命中。

**估计精确率 P ≈ 84.00%**

> 注：L1 精确率以 alias 精确匹配率为代理指标，理论上限为 1.0，实际存在约 5% 的歧义（如"Go"同时匹配语言和动词）。

## 3. 召回率估计（R）

对语料中频次 ≥ 5 次的实体计算文本出现次数 vs mention 命中次数。

**平均召回率 R ≈ 77.59%**（评估了 104 个高频实体）

| entity_id | 文本出现 | mention 命中 | recall |
| --------- | -------- | ----------- | ------ |
| interest_security                        |  2140 |    2037 | 95.19% |
| knowledge_algorithms                     |  1597 |    1598 | 100.00% |
| interest_frontend                        |  1280 |    1452 | 100.00% |
| skill_python                             |  1100 |    1098 | 99.82% |
| skill_java                               |  1057 |     678 | 64.14% |
| skill_sql                                |  1009 |     295 | 29.24% |
| interest_research                        |   916 |     921 | 100.00% |
| interest_ml                              |   804 |     381 | 47.39% |
| knowledge_data_structures                |   784 |     783 | 99.87% |
| soft_communication                       |   756 |     767 | 100.00% |
| knowledge_system_design                  |   699 |     803 | 100.00% |
| soft_documentation                       |   660 |     603 | 91.36% |
| tool_elasticsearch                       |   638 |       9 | 1.41% |
| knowledge_operating_systems              |   598 |      38 | 6.35% |
| tool_mysql                               |   590 |     589 | 99.83% |

**F1（L1 层）≈ 80.67%**

## 4. 消融实验

| 配置 | Mention 数 | 唯一实体数 | 精确率估计 |
| ---- | ---------- | ---------- | ---------- |
| L1 仅规则层 | 22674 | 111 | 95% |
| L1 + L2 已确认 | 22756 | 111 | 88% |

L2 发现的新实体数（不在 L1 中）：0

LLM 候选新词面：1290 个（需人工消歧确认）

## 5. 新实体发现

| 指标 | 数值 |
|------|------|
| DBSCAN 新实体簇 | 45 |
| 自动新建节点 | 14 |
| 待人工审阅 | 393 |

**代表性新实体（簇 size ≥ 3）：**  
`tool_html`、`tool_jquery`、`tool_oracle`、`tool_springmvc`、`tool_springcloud`、`tool_hibernate`、`tool_github`、`tool_easyui`

## 6. Role 覆盖分析

分析 47 个职业节点的直接 evidence 支撑覆盖率。

**平均 evidence 覆盖率：68.44%**

覆盖率最低的职业（待补充 evidence）：

| role | 需要 | 已覆盖 | 覆盖率 | 缺失示例 |
| ---- | ---- | ------ | ------ | -------- |
| 前端工程师 | 1 | 0 | 0% | constraint_dislike_ui_polish |
| 机器学习工程师 | 1 | 0 | 0% | constraint_dislike_math_theory |
| AI 应用工程师 | 1 | 0 | 0% | constraint_dislike_research_uncertainty |
| DevOps 工程师 | 1 | 0 | 0% | constraint_dislike_oncall |
| SRE 工程师 | 1 | 0 | 0% | constraint_dislike_oncall |

覆盖率最高的职业：

| role | 需要 | 已覆盖 | 覆盖率 |
| ---- | ---- | ------ | ------ |
| IoT 工程师 | 1 | 1 | 100% |
| 云安全工程师 | 2 | 2 | 100% |
| 应用安全工程师 | 2 | 2 | 100% |
| 渗透测试工程师 | 1 | 1 | 100% |
| 大模型应用工程师 | 2 | 2 | 100% |
