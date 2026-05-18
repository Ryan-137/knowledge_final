# optimize 实体抽取与消歧流程

`optimize/` 是课程项目中用于扩展职业知识图谱实体、别名和外部标准引用的离线管道。它读取原项目 `data/` 中的 source、dictionary、seed 数据，所有中间产物写入 `optimize/pipeline_data/`，最终审阅产物写入 `optimize/output/`。

## 边界原则

- 原项目 `data/` 只读，不在管道中直接覆盖。
- 管道中间数据统一写入 `optimize/pipeline_data/`。
- 最终输出统一写入 `optimize/output/`，人工审阅后再合并到 `data/sources/`。
- `optimize/output/` 中已有的三份 enriched 文件是合并候选，不等于线上图谱已生效。

## 目录结构

```text
optimize/
  config.py                         路径、阈值、模型和数据采集配置
  requirements.txt                  optimize 管道依赖
  preview_output.py                 输出前的候选实体与外部对齐预览

  pipeline_data/                    管道生成数据，已被 .gitignore 忽略
    raw/                            原始语料
      fairCV/                       FairCV 简历数据
      jd/                           JD 招聘文本
      external/esco/                ESCO 技能索引
      external/onet/                O*NET 技术工具索引
    staging/
      staged_documents.jsonl        清洗、分章节、分句后的文档
      mentions.jsonl                L1/L2 mention 识别与消歧结果
    canonical/
      candidate_surfaces.json       候选新词面
      entity_cooccurrence_candidates.jsonl  文档级实体共现候选
      disambiguation_log.jsonl      消歧审计日志
      new_entity_clusters.json      嵌入聚类得到的新实体簇
      external_alignment.json       ESCO / O*NET 外部标准对齐结果
      entity_quality_report.json    输出质量校验报告
      entity_case_report.json       案例报告
    data_catalog.md                 数据源目录，由采集脚本维护

  output/
    skills_enriched.json            待审阅的 skills.json 替换候选
    aliases_enriched.json           待审阅的 aliases.json 替换候选
    imported_profiles_new.json      待追加到 imported_profiles.json 的 profile

  data_collection/                  数据采集
    fetch_fairCV.py                 获取/导入 FairCV 简历数据
    crawl_jd.py                     Selenium JD 爬虫与 CSV 导入
    import_skillspan_jd.py          将 CN_skillspan train 聚合为 JD raw 文档
    fetch_external_standards.py     获取 ESCO / O*NET 外部标准
    login_helper.py                 招聘站点登录态辅助
    catalog.py                      维护 pipeline_data/data_catalog.md

  staging/
    clean_documents.py              raw 文档清洗、章节切分、句子偏移生成
    segment_sentences.py            分句工具

  ner/
    rule_ner.py                     L1 规则 NER，基于现有 aliases 和正则
    distant_supervision.py          L2a 候选词面挖掘与实体共现候选
    llm_ner.py                      L2b LLM 结构化抽取
    merge_mentions.py               L3 mention 去重与置信度融合
    abbr_expansion.json             缩写扩展表

  disambiguation/
    string_normalize.py             第一级精确匹配消歧
    embedding_disambiguate.py       第二级 embedding 相似度与 DBSCAN 聚类

  external_align/
    align_esco_onet.py              与 ESCO / O*NET 做 embedding 对齐

  output/
    generate_output.py              生成三份 enriched 输出文件

  evaluation/
    validate_entity_quality.py      质量门禁，校验输出实体、别名和 profile
    coverage_report.py              覆盖率、消融和召回估算报告
    case_report.py                  典型案例报告
    evaluate_skillspan_ner.py       CN_skillspan 测试集上的规则 NER 评测
    review_queue.py                 人工审阅 embedding 待确认项并回写 aliases 输出
```

## 环境准备

本项目约定在 conda 环境 `xxx` 中执行 Python 命令。执行前先确认解释器路径，避免把依赖装到系统 Python。

```powershell
conda activate xxx
python -c "import sys; print(sys.executable)"
pip install -r optimize/requirements.txt
python -m spacy download zh_core_web_sm
python -m spacy download en_core_web_sm
```

LLM 抽取默认使用 DeepSeek 兼容 OpenAI SDK 接口，环境变量名由 `optimize/config.py` 中的 `cfg.llm.api_key_env` 决定，当前为 `DEEPSEEK_API_KEY`。

```powershell
$env:DEEPSEEK_API_KEY="你的 key"
```

## 标准执行顺序

按下面顺序执行可以得到完整的 enriched 输出。调试时优先给采集、清洗和 NER 步骤加 `--limit`、`--max-docs` 或 `--max-samples`。

```powershell
conda activate xxx
python -c "import sys; print(sys.executable)"

# 1. 数据采集
python -m optimize.data_collection.fetch_fairCV --max-samples 500
python -m optimize.data_collection.crawl_jd --source csv_import --csv-path path\to\jobs.csv
python -m optimize.data_collection.import_skillspan_jd
python -m optimize.data_collection.fetch_external_standards

# 2. 清洗与分句
python -m optimize.staging.clean_documents

# 3. 实体识别与候选挖掘
python -m optimize.ner.rule_ner
python -m optimize.ner.distant_supervision
python -m optimize.ner.llm_ner --max-docs 200
python -m optimize.ner.merge_mentions

# 4. 消歧与外部标准对齐
python -m optimize.disambiguation.string_normalize
python -m optimize.disambiguation.embedding_disambiguate
python -m optimize.external_align.align_esco_onet

# 5. 输出前预览、生成和校验
python -m optimize.preview_output
python -m optimize.output.generate_output
python -m optimize.evaluation.validate_entity_quality
python -m optimize.evaluation.coverage_report
python -m optimize.evaluation.case_report
```

## 常用调试命令

```powershell
# 只清洗部分来源
python -m optimize.staging.clean_documents --sources fairCV --limit 50

# 只跑规则 NER 的小样本
python -m optimize.ner.rule_ner --sources fairCV --limit 50

# LLM NER 先 dry-run，不写入 mentions.jsonl
python -m optimize.ner.llm_ner --max-docs 5 --dry-run

# 嵌入消歧调小 batch
python -m optimize.disambiguation.embedding_disambiguate --batch-size 32

# 输出时只接受更大的新实体簇
python -m optimize.output.generate_output --min-cluster-size 4

# warning 也阻断合并
python -m optimize.evaluation.validate_entity_quality --fail-on-warning

# 查看质量报告中的典型案例
python -m optimize.evaluation.case_report --limit-per-section 10
```

## 数据采集说明

### FairCV

优先使用本地 Parquet 或 HuggingFace mirror，避免网络不稳定影响流程。

```powershell
python -m optimize.data_collection.fetch_fairCV --parquet-path path\to\train-00000.parquet
python -m optimize.data_collection.fetch_fairCV --local-dir path\to\fairCV_hf_cache
python -m optimize.data_collection.fetch_fairCV --max-samples 500 --positions "后端开发工程师,数据工程师"
```

### JD 招聘文本

线上招聘站点通常有登录态和反爬限制。能拿到 CSV 时优先使用 CSV 导入；必须爬取时先用 `login_helper.py` 准备 Chrome profile。

```powershell
python -m optimize.data_collection.login_helper --site lagou --profile optimize/.chrome_profile
python -m optimize.data_collection.crawl_jd --source lagou --chrome-profile optimize/.chrome_profile --max 30 --no-headless
python -m optimize.data_collection.crawl_jd --source csv_import --csv-path path\to\jobs.csv
```

### CN_skillspan

`import_skillspan_jd.py` 默认读取 `optimize/CN_skillspan_lkst_train.json`，该文件在 `.gitignore` 中忽略，需要本地自行准备。

```powershell
python -m optimize.data_collection.import_skillspan_jd --input optimize/CN_skillspan_lkst_train.json
```

### ESCO / O*NET

默认脚本会尝试下载外部标准。若 ESCO 地址变更或网络失败，先手动下载 ZIP，再传入本地路径。

```powershell
python -m optimize.data_collection.fetch_external_standards
python -m optimize.data_collection.fetch_external_standards --esco-zip path\to\esco.zip
python -m optimize.data_collection.fetch_external_standards --skip-esco
python -m optimize.data_collection.fetch_external_standards --skip-onet
```

## 输出合并方式

`optimize.output.generate_output` 只生成候选文件，不直接修改 `data/sources/`。确认 `validate_entity_quality.py` 没有硬错误后，再手动合并。

```powershell
Copy-Item optimize\output\skills_enriched.json data\sources\skills.json
Copy-Item optimize\output\aliases_enriched.json data\sources\aliases.json
```

`imported_profiles_new.json` 不应直接覆盖 `data/sources/imported_profiles.json`，需要把其中新增 profile 追加进原文件。合并后重新编译和校验图谱。

```powershell
python scripts/build_graph.py
python scripts/validate_graph.py
```

## 下游接口

| 文件 | 用途 |
| --- | --- |
| `optimize/output/skills_enriched.json` | 审阅后替换 `data/sources/skills.json` |
| `optimize/output/aliases_enriched.json` | 审阅后替换 `data/sources/aliases.json` |
| `optimize/output/imported_profiles_new.json` | 审阅后追加到 `data/sources/imported_profiles.json` |
| `optimize/pipeline_data/staging/mentions.jsonl` | 关系抽取、覆盖率统计和案例分析输入 |
| `optimize/pipeline_data/canonical/entity_cooccurrence_candidates.jsonl` | 文档级实体共现候选，供关系抽取参考 |
| `optimize/pipeline_data/canonical/disambiguation_log.jsonl` | 消歧审计与人工复核 |
| `optimize/pipeline_data/canonical/entity_quality_report.json` | 合并前质量门禁报告 |

该管道只负责实体、别名、profile 和外部标准引用的增量生成，不负责生成 `supports`、`requires` 等关系边，也不负责设置边权重。
