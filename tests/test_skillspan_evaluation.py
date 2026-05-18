from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimize.disambiguation import embedding_disambiguate
from optimize.data_collection.import_skillspan_jd import build_raw_docs
from optimize.evaluation.evaluate_skillspan_ner import (
    EvalSpan,
    build_report,
    parse_gold_sample,
)
from optimize.ner import llm_ner, rule_ner
from optimize.ner.rule_ner import AliasPattern, RuleNER
from optimize.utils.file_utils import read_jsonl, write_jsonl


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


class OptimizePipelineRepairTests(unittest.TestCase):
    def test_rule_ner_abbr_expansion_keeps_original_offsets(self) -> None:
        ner = RuleNER.__new__(RuleNER)
        ner._abbr = {
            "ml": "machine learning",
            "nlp": "natural language processing",
            "c++": "cpp",
        }
        ner._prefs = {}
        ner._phrase_rules = []
        ner._alias_patterns = [
            AliasPattern("knowledge_machine_learning", "machine learning", RuleNER._compile_pattern("machine learning")),
            AliasPattern("knowledge_nlp", "natural language processing", RuleNER._compile_pattern("natural language processing")),
            AliasPattern("skill_cpp", "cpp", RuleNER._compile_pattern("cpp")),
        ]

        text = "熟悉ML、NLP和C++"
        mentions = ner.scan_sentence(text, 100, "doc1", "sec1", "tech_skills")
        by_surface = {m.surface: m for m in mentions}

        self.assertEqual(set(by_surface), {"ML", "NLP", "C++"})
        for surface, mention in by_surface.items():
            start = text.index(surface)
            self.assertEqual((mention.char_start, mention.char_end), (100 + start, 100 + start + len(surface)))

    def test_llm_entities_must_anchor_to_original_text_and_pass_confidence(self) -> None:
        stats = {"dropped_unanchored": 0, "dropped_low_confidence": 0}
        mentions = llm_ner._parse_llm_entities(
            raw_entities=[
                {"surface": "LangChain", "type": "tool", "confidence": 0.9, "is_negative": False},
                {"surface": "不存在实体", "type": "tool", "confidence": 0.9, "is_negative": False},
                {"surface": "Ray", "type": "tool", "confidence": 0.1, "is_negative": False},
            ],
            doc_id="doc1",
            section_id="sec1",
            section_type="requirements",
            section_text="要求熟悉LangChain和Ray。",
            section_char_start=20,
            covered_aliases=set(),
            stats=stats,
        )

        self.assertEqual([m["surface"] for m in mentions], ["LangChain"])
        self.assertEqual((mentions[0]["char_start"], mentions[0]["char_end"]), (24, 33))
        self.assertEqual(stats["dropped_unanchored"], 1)
        self.assertEqual(stats["dropped_low_confidence"], 1)

    def test_source_filter_accepts_jd_group_and_specific_source_name(self) -> None:
        doc = {"source_group": "jd", "source_name": "cn_skillspan_lkst"}

        self.assertTrue(rule_ner._matches_sources(doc, ["jd"]))
        self.assertTrue(rule_ner._matches_sources(doc, ["cn_skillspan_lkst"]))
        self.assertFalse(rule_ner._matches_sources(doc, ["fairCV"]))
        self.assertTrue(llm_ner._matches_sources(doc, ["jd"]))

    def test_rule_ner_default_overwrites_and_append_explicitly_accumulates(self) -> None:
        class FakeMention:
            def __init__(self, doc_id: str) -> None:
                self._doc_id = doc_id

            def as_dict(self) -> dict:
                return {
                    "mention_id": f"m_{self._doc_id}",
                    "doc_id": self._doc_id,
                    "section_id": "sec",
                    "section_type": "tech_skills",
                    "surface": "Python",
                    "normalized": "python",
                    "char_start": 0,
                    "char_end": 6,
                    "context_snippet": "Python",
                    "candidates": [],
                    "linked_entity_id": "skill_python",
                    "link_method": "alias_exact",
                    "link_confidence": 0.93,
                    "status": "rule_match",
                    "is_negative": False,
                    "intensity": "neutral",
                }

        class FakeRuleNER:
            def process_document(self, doc: dict) -> list[FakeMention]:
                return [FakeMention(doc["doc_id"])]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            staging = root / "staging"
            staging.mkdir()
            staged_path = staging / "staged_documents.jsonl"
            write_jsonl(
                staged_path,
                [
                    {"doc_id": "doc1", "source_name": "csv_import", "source_group": "jd", "sha256": "sha1"},
                    {"doc_id": "doc2", "source_name": "fairCV", "source_group": "fairCV", "sha256": "sha2"},
                ],
            )

            mentions_path = staging / "mentions.jsonl"
            with (
                patch.object(rule_ner, "_MENTIONS_PATH", mentions_path),
                patch.object(rule_ner, "RuleNER", FakeRuleNER),
            ):
                rule_ner.run()
                rule_ner.run()
                self.assertEqual(len(read_jsonl(mentions_path)), 2)

                rule_ner.run(append=True)
                self.assertEqual(len(read_jsonl(mentions_path)), 4)

    def test_embedding_cache_manifest_invalidates_stale_entity_embeddings(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.calls = 0

            def encode(self, texts: list[str], **_: object):
                import numpy as np

                self.calls += 1
                return np.ones((len(texts), 2), dtype=float)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch.object(embedding_disambiguate, "_EMB_CACHE_PATH", root / "entity_embeddings.npy"),
                patch.object(embedding_disambiguate, "_EMB_IDS_PATH", root / "entity_ids.json"),
                patch.object(embedding_disambiguate, "_EMB_MANIFEST_PATH", root / "manifest.json"),
            ):
                model = FakeModel()
                aliases = {"skill_python": ["python"]}
                nodes = [{"id": "skill_python", "name": "Python", "layer": "evidence"}]

                with patch.object(embedding_disambiguate, "_build_embedding_manifest", return_value={"model": "m1"}):
                    embedding_disambiguate._load_or_compute_entity_embeddings(model, aliases, nodes, 8)
                    embedding_disambiguate._load_or_compute_entity_embeddings(model, aliases, nodes, 8)
                self.assertEqual(model.calls, 1)

                with patch.object(embedding_disambiguate, "_build_embedding_manifest", return_value={"model": "m2"}):
                    embedding_disambiguate._load_or_compute_entity_embeddings(model, aliases, nodes, 8)
                self.assertEqual(model.calls, 2)


if __name__ == "__main__":
    unittest.main()
