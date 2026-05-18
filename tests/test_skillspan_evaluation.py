from __future__ import annotations

import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimize.data_collection.import_skillspan_jd import build_raw_docs
from optimize.evaluation.evaluate_skillspan_ner import (
    EvalSpan,
    build_report,
    parse_gold_sample,
)


class SkillSpanEvaluationTests(unittest.TestCase):
    def test_parse_single_gold_span(self) -> None:
        sample = parse_gold_sample(
            {
                "id": 1,
                "input": "熟悉Python开发。",
                "output": "熟悉@@Python开发##S。",
                "meta": {"source_domain": "人工智能招聘"},
            }
        )

        self.assertEqual(len(sample.gold_spans), 1)
        span = sample.gold_spans[0]
        self.assertEqual((span.start, span.end, span.text, span.label), (2, 10, "Python开发", "S"))
        self.assertEqual(len(sample.invalid_gold), 0)

    def test_parse_overlap_gold_as_invalid(self) -> None:
        sample = parse_gold_sample(
            {
                "id": 2,
                "input": "具备组织性和主动性。",
                "output": "具备@@组织性##T@@组织性和主动性##T。",
                "meta": {"source_domain": "人工智能招聘"},
            }
        )

        self.assertEqual(len(sample.gold_spans), 0)
        self.assertEqual(len(sample.invalid_gold), 2)
        self.assertTrue(all(item.reason == "overlap_or_nested" for item in sample.invalid_gold))

    def test_parse_unmapped_gold_as_invalid(self) -> None:
        sample = parse_gold_sample(
            {
                "id": 3,
                "input": "熟悉Linux。",
                "output": "熟悉@@Kubernetes##S。",
                "meta": {"source_domain": "人工智能招聘"},
            }
        )

        self.assertEqual(len(sample.gold_spans), 0)
        self.assertEqual(sample.invalid_gold[0].reason, "unable_to_map")

    def test_metrics_count_false_positive_on_no_gold_sentence(self) -> None:
        gold_sample = parse_gold_sample(
            {
                "id": 4,
                "input": "这里没有能力标签。",
                "output": "这里没有能力标签。",
                "meta": {"source_domain": "事业单位招聘"},
            }
        )
        pred = EvalSpan(sample_id="4", start=2, end=4, text="没有", label="S", entity_id="skill_fake")

        report = build_report([gold_sample], {"4": [pred]}, mode="rule", test_path="dummy.json")  # type: ignore[arg-type]

        self.assertEqual(report["metrics"]["span_exact"]["tp"], 0)
        self.assertEqual(report["metrics"]["span_exact"]["fp"], 1)
        self.assertEqual(report["metrics"]["span_exact"]["fn"], 0)

    def test_build_raw_docs_groups_train_rows(self) -> None:
        docs = build_raw_docs(
            [
                {"global_id": "100", "sent_id": 1, "sentence": "岗位要求：熟悉Python。", "source_domain": "人工智能招聘"},
                {"global_id": "100", "sent_id": 0, "sentence": "岗位职责：负责后端开发。", "source_domain": "人工智能招聘"},
                {"global_id": "101", "sent_id": 0, "sentence": "&lt", "source_domain": "人工智能招聘"},
            ],
            snapshot_date="2026-05-18",
        )

        self.assertEqual(len(docs), 1)
        doc = docs[0]
        self.assertTrue(doc["doc_id"].startswith("jd_cn_skillspan_lkst_100_"))
        self.assertIn("岗位职责：负责后端开发。", doc["content"]["full_text"])
        self.assertIn("岗位要求：熟悉Python。", doc["content"]["full_text"])
        self.assertEqual(doc["source_name"], "cn_skillspan_lkst")


if __name__ == "__main__":
    unittest.main()
