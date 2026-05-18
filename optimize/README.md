# optimize — Entity Extraction & Disambiguation Pipeline

Owner: ztt (data collection / rule NER / external alignment) &
       qyw (LLM NER / embedding disambiguation / evaluation)

The pipeline enriches `data/sources/skills.json`, `data/sources/aliases.json`,
and `data/sources/imported_profiles.json`.  These files are then compiled into
the runtime graph by `scripts/build_graph.py`.

## Isolation principle

All pipeline-generated data is written to **`optimize/pipeline_data/`** and
**`optimize/output/`**.  The original project files under `data/` are
**never modified**.  When ready to apply the enriched entities to the project,
manually review `optimize/output/` and merge into `data/sources/` by hand.

```
optimize/
  pipeline_data/           All pipeline-generated data (WRITE zone)
    raw/                   Downloaded raw documents
      fairCV/              FairCV resumes (700 records)
      jd/                  JD records (4421 records)
      external/esco/       ESCO skill index
      external/onet/       O*NET technology tools index
    staging/               Cleaned and segmented documents
      staged_documents.jsonl
      mentions.jsonl       All NER mentions (L1 + L2b, merged & deduplicated)
    canonical/             Pipeline-generated entity candidates
      candidate_surfaces.json
      disambiguation_log.jsonl
      new_entity_clusters.json
      entity_cooccurrence_candidates.jsonl
    data_catalog.md
  output/                  Final enriched files for manual review & merge
    skills_enriched.json   → merge into data/sources/skills.json
    aliases_enriched.json  → merge into data/sources/aliases.json
    imported_profiles_new.json → append to data/sources/imported_profiles.json
  config.py              Central configuration (all paths and thresholds)
  utils/
    logging_utils.py     Logger factory
    file_utils.py        JSON / JSONL I/O, atomic writes
    hash_utils.py        SHA-256 fingerprinting
  data_collection/       Step 1 — raw data acquisition
    catalog.py           Maintains docs/data_catalog.md
    fetch_fairCV.py      FairCV dataset (HuggingFace or local CSV)
    crawl_jd.py          JD crawler (Selenium-based) + CSV import fallback
    fetch_external_standards.py  ESCO skills + O*NET technology tools
  staging/               Step 2 — text cleaning and segmentation
    clean_documents.py
    segment_sentences.py
  ner/                   Step 3 — three-layer NER pipeline
    abbr_expansion.json  Abbreviation expansion table
    rule_ner.py          L1: spaCy EntityRuler + regex rules   (ztt)
    distant_supervision.py  L2a: auto-labelling from existing aliases  (qyw)
    llm_ner.py           L2b: LLM structured extraction   (qyw)
    merge_mentions.py    L3: cross-layer merge and confidence fusion
  disambiguation/        Step 4 — entity disambiguation
    string_normalize.py  Tier 1: normalisation + exact match
    embedding_disambiguate.py  Tier 2: multilingual embeddings + DBSCAN
    llm_disambiguate.py  Tier 3: LLM binary-classification fallback
  external_align/        Step 5 — ESCO / O*NET alignment
    align_esco_onet.py
  output/                Step 6 — write back to data/sources/
    generate_skills.py
    generate_aliases.py
    generate_profiles.py
    validate_output.py
  evaluation/            Step 7 — P/R/F1, ablation, coverage report
    golden_set_eval.py
    ablation_study.py
    coverage_report.py
  notebooks/
    review_tool.ipynb    Interactive human review for needs_review queue
```

## Quick start

```bash
pip install -r optimize/requirements.txt
python -m spacy download zh_core_web_sm en_core_web_sm

# Run each step in order
python -m optimize.data_collection.fetch_fairCV
python -m optimize.data_collection.crawl_jd
python -m optimize.data_collection.fetch_external_standards
python -m optimize.staging.clean_documents
python -m optimize.ner.rule_ner
python -m optimize.ner.distant_supervision
python -m optimize.ner.llm_ner
python -m optimize.ner.merge_mentions
python -m optimize.disambiguation.string_normalize
python -m optimize.disambiguation.embedding_disambiguate
python -m optimize.external_align.align_esco_onet
python -m optimize.output.generate_skills
python -m optimize.output.generate_aliases
python -m optimize.output.generate_profiles
python -m optimize.output.validate_output
python -m optimize.evaluation.coverage_report
```

## Data source acquisition notes

### FairCV

FairCV is a semi-structured resume dataset for fairness research.

- **HuggingFace route (preferred):** configure `cfg.collection.fairCV_dataset_name`
  and run `fetch_fairCV.py`.  If the dataset is gated, run
  `huggingface-cli login` first.
- **Local CSV fallback:** if the dataset is unavailable, download a compatible
  CSV (e.g. from Kaggle) and supply `--csv-path /path/to/file.csv`.

### JD crawler (拉勾 / other platforms)

Modern recruitment platforms use heavy client-side rendering and session
validation.  The recommended approach:

1. Install Chrome and run `pip install selenium webdriver-manager`.
2. Log in to the target site once in a Chrome window.  Supply the profile
   directory with `--chrome-profile`.
3. Run `crawl_jd.py --source lagou`.
4. If live scraping fails, download a CSV dataset (e.g. from Kaggle:
   "Chinese Job Postings") and use `--source csv_import --csv-path ...`.

### ESCO

The download URL may change between releases.  If the default URL fails:
1. Visit https://esco.ec.europa.eu/en/use-esco/download
2. Download the skills CSV ZIP manually.
3. Run: `python -m optimize.data_collection.fetch_external_standards --esco-zip /path/to/file.zip`

### O*NET Technology Skills

Direct download usually works without authentication.  If the URL is outdated,
visit https://www.onetcenter.org/database.html, download
"Technology Skills.txt", and place it at `data/raw/external/onet/technology_skills.txt`.

## Deliverables and downstream interface

| File | Consumer |
| ---- | -------- |
| `data/sources/skills.json` (enriched) | `scripts/build_graph.py` |
| `data/sources/aliases.json` (enriched) | `scripts/build_graph.py` |
| `data/sources/imported_profiles.json` (new entries) | `scripts/build_graph.py` |
| `data/staging/mentions.jsonl` | wxs & sx (relation extraction) |
| `data/canonical/disambiguation_log.jsonl` | audit / defence |
| `reports/entity_coverage_report.md` | defence materials |

This pipeline does **not** produce relation types, edge weights, or any
`supports` / `requires` definitions.  Those are the responsibility of wxs & sx.
