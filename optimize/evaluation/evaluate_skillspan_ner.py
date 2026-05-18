"""使用 CN_skillspan LKST test 集评测中文能力片段抽取。

该评测脚本只读取 test split，不把 test 数据写入 raw/staging/canonical 的
候选发现流程，避免数据泄漏。输出为 JSON 指标报告，不生成 Markdown 文件。

运行方式：
    python -m optimize.evaluation.evaluate_skillspan_ner \
        --test-path optimize/CN_skillspan_lkst_test.json
    python -m optimize.evaluation.evaluate_skillspan_ner --limit 20
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re
from typing import Any

from optimize.config import cfg
from optimize.utils.file_utils import read_json, write_json

_GOLD_RE = re.compile(r"@@(.+?)##([LKST])")
_LABELS = ("L", "K", "S", "T")
_NODE_TYPE_TO_LKST = {
    "language": "L",
    "knowledge": "K",
    "skill": "S",
    "tool": "S",
    "soft_skill": "T",
}


@dataclass(frozen=True)
class EvalSpan:
    """评测中的一个片段。"""

    sample_id: str
    start: int
    end: int
    text: str
    label: str | None = None
    entity_id: str | None = None

    def binary_key(self) -> tuple[str, int, int]:
        return (self.sample_id, self.start, self.end)

    def typed_key(self) -> tuple[str, int, int, str] | None:
        if self.label is None:
            return None
        return (self.sample_id, self.start, self.end, self.label)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "label": self.label,
            "entity_id": self.entity_id,
        }


@dataclass(frozen=True)
class InvalidGold:
    """无法进入主指标的 gold 标注异常。"""

    sample_id: str
    text: str
    label: str | None
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "text": self.text,
            "label": self.label,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class GoldSample:
    """一条 test 样本及其可评测 gold span。"""

    sample_id: str
    text: str
    source_domain: str
    gold_spans: tuple[EvalSpan, ...]
    invalid_gold: tuple[InvalidGold, ...]
    marker_count: int


def _find_text_span(text: str, surface: str, next_start: dict[str, int]) -> tuple[int, int] | None:
    """把标注片段回填到原句字符区间，重复词面按出现顺序匹配。"""
    if not surface:
        return None
    pos = text.find(surface, next_start.get(surface, 0))
    if pos < 0:
        pos = text.find(surface)
    if pos < 0:
        return None
    next_start[surface] = pos + len(surface)
    return pos, pos + len(surface)


def parse_gold_sample(row: dict[str, Any], fallback_index: int = 0) -> GoldSample:
    """解析一条 instruction/input/output 格式的 SkillSpan test 样本。"""
    sample_id = str(row.get("id", fallback_index))
    text = str(row.get("input", "") or "")
    output = str(row.get("output", "") or "")
    meta = row.get("meta", {}) if isinstance(row.get("meta", {}), dict) else {}
    source_domain = str(meta.get("source_domain", "") or "unknown")

    marker_matches = list(_GOLD_RE.finditer(output))
    spans: list[EvalSpan] = []
    invalid: list[InvalidGold] = []
    next_start: dict[str, int] = {}

    for match in marker_matches:
        surface = match.group(1)
        label = match.group(2)
        mapped = _find_text_span(text, surface, next_start)
        if mapped is None:
            invalid.append(InvalidGold(sample_id, surface, label, "unable_to_map"))
            continue
        start, end = mapped
        spans.append(EvalSpan(sample_id, start, end, text[start:end], label))

    if "@@" in output and "##" in output and not marker_matches:
        invalid.append(InvalidGold(sample_id, "", None, "malformed_marker"))

    # 重叠或嵌套 gold 不进入主指标，避免同一句的多重标注污染 P/R/F1。
    overlap_indexes: set[int] = set()
    for i, left in enumerate(spans):
        for j in range(i + 1, len(spans)):
            right = spans[j]
            if max(left.start, right.start) < min(left.end, right.end):
                overlap_indexes.add(i)
                overlap_indexes.add(j)

    valid_spans: list[EvalSpan] = []
    for idx, span in enumerate(spans):
        if idx in overlap_indexes:
            invalid.append(InvalidGold(sample_id, span.text, span.label, "overlap_or_nested"))
            continue
        valid_spans.append(span)

    return GoldSample(
        sample_id=sample_id,
        text=text,
        source_domain=source_domain,
        gold_spans=tuple(valid_spans),
        invalid_gold=tuple(invalid),
        marker_count=len(marker_matches),
    )


def parse_gold_samples(rows: list[dict[str, Any]]) -> list[GoldSample]:
    """批量解析 SkillSpan test 样本。"""
    return [parse_gold_sample(row, idx) for idx, row in enumerate(rows)]


def _prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _exact_metrics(gold: list[EvalSpan], pred: list[EvalSpan], typed: bool) -> dict[str, Any]:
    if typed:
        gold_keys = {s.typed_key() for s in gold if s.typed_key() is not None}
        pred_keys = {s.typed_key() for s in pred if s.typed_key() is not None}
    else:
        gold_keys = {s.binary_key() for s in gold}
        pred_keys = {s.binary_key() for s in pred}
    tp = len(gold_keys & pred_keys)
    fp = len(pred_keys - gold_keys)
    fn = len(gold_keys - pred_keys)
    return _prf(tp, fp, fn)


def _overlaps(left: EvalSpan, right: EvalSpan) -> bool:
    return left.sample_id == right.sample_id and max(left.start, right.start) < min(left.end, right.end)


def _relaxed_overlap_metrics(gold: list[EvalSpan], pred: list[EvalSpan]) -> dict[str, Any]:
    """按样本做一对一 overlap 匹配，计算 relaxed binary P/R/F1。"""
    gold_by_sample: dict[str, list[EvalSpan]] = defaultdict(list)
    pred_by_sample: dict[str, list[EvalSpan]] = defaultdict(list)
    for span in gold:
        gold_by_sample[span.sample_id].append(span)
    for span in pred:
        pred_by_sample[span.sample_id].append(span)

    tp = 0
    for sample_id, preds in pred_by_sample.items():
        used_gold: set[int] = set()
        golds = gold_by_sample.get(sample_id, [])
        for pred_span in sorted(preds, key=lambda s: (s.start, s.end)):
            for idx, gold_span in enumerate(golds):
                if idx in used_gold:
                    continue
                if _overlaps(pred_span, gold_span):
                    used_gold.add(idx)
                    tp += 1
                    break

    fp = len(pred) - tp
    fn = len(gold) - tp
    return _prf(tp, fp, fn)


def _node_type_map() -> dict[str, str]:
    """读取图谱节点类型，并映射到 LKST 粗粒度标签。"""
    nodes = read_json(cfg.paths.seeds_nodes)
    result: dict[str, str] = {}
    for node in nodes:
        node_type = str(node.get("node_type", "") or node.get("metadata", {}).get("category", ""))
        label = _NODE_TYPE_TO_LKST.get(node_type)
        if label:
            result[node["id"]] = label
    return result


def predict_rule_spans(samples: list[GoldSample]) -> dict[str, list[EvalSpan]]:
    """调用现有规则 NER，对 test 句子产生预测 span。"""
    from optimize.ner.rule_ner import RuleNER

    ner = RuleNER()
    entity_label = _node_type_map()
    predictions: dict[str, list[EvalSpan]] = {}

    for sample in samples:
        mentions = ner.scan_sentence(
            sentence_text=sample.text,
            sentence_abs_start=0,
            doc_id=f"skillspan_eval_{sample.sample_id}",
            section_id=f"skillspan_eval_{sample.sample_id}_requirements_0",
            section_type="requirements",
        )
        spans: dict[tuple[int, int, str | None, str | None], EvalSpan] = {}
        for mention in mentions:
            start = mention.char_start
            end = mention.char_end
            if start < 0 or end <= start or end > len(sample.text):
                continue
            label = entity_label.get(mention.entity_id)
            key = (start, end, label, mention.entity_id)
            spans[key] = EvalSpan(
                sample_id=sample.sample_id,
                start=start,
                end=end,
                text=sample.text[start:end],
                label=label,
                entity_id=mention.entity_id,
            )
        predictions[sample.sample_id] = list(spans.values())

    return predictions


def build_report(
    samples: list[GoldSample],
    predictions_by_sample: dict[str, list[EvalSpan]],
    *,
    mode: str,
    test_path: Path,
) -> dict[str, Any]:
    """汇总整体、分标签、分来源指标。"""
    gold = [span for sample in samples for span in sample.gold_spans]
    pred = [span for sample in samples for span in predictions_by_sample.get(sample.sample_id, [])]
    invalid = [bad for sample in samples for bad in sample.invalid_gold]
    total_markers = sum(sample.marker_count for sample in samples)

    by_label = {
        label: _exact_metrics(
            [span for span in gold if span.label == label],
            [span for span in pred if span.label == label],
            typed=True,
        )
        for label in _LABELS
    }

    by_source_domain: dict[str, Any] = {}
    for source_domain in sorted({sample.source_domain for sample in samples}):
        sample_ids = {sample.sample_id for sample in samples if sample.source_domain == source_domain}
        by_source_domain[source_domain] = _exact_metrics(
            [span for span in gold if span.sample_id in sample_ids],
            [span for span in pred if span.sample_id in sample_ids],
            typed=False,
        )

    invalid_reason_counts = Counter(bad.reason for bad in invalid)
    gold_label_counts = Counter(span.label for span in gold)
    raw_marker_label_counts = Counter()
    for sample in samples:
        for bad in sample.invalid_gold:
            if bad.label:
                raw_marker_label_counts[bad.label] += 1
        for span in sample.gold_spans:
            if span.label:
                raw_marker_label_counts[span.label] += 1

    return {
        "generated_at": date.today().isoformat(),
        "mode": mode,
        "test_path": str(test_path),
        "data_summary": {
            "samples": len(samples),
            "samples_with_gold": sum(1 for sample in samples if sample.gold_spans),
            "total_gold_markers": total_markers,
            "valid_gold_spans": len(gold),
            "invalid_gold_spans": len(invalid),
            "pred_spans": len(pred),
            "gold_label_counts": dict(sorted(gold_label_counts.items())),
            "raw_marker_label_counts": dict(sorted(raw_marker_label_counts.items())),
            "source_domain_counts": dict(Counter(sample.source_domain for sample in samples)),
            "invalid_gold_rate": round(len(invalid) / total_markers, 4) if total_markers else 0.0,
        },
        "metrics": {
            "span_exact": _exact_metrics(gold, pred, typed=False),
            "typed_exact": _exact_metrics(gold, pred, typed=True),
            "relaxed_overlap": _relaxed_overlap_metrics(gold, pred),
        },
        "by_label": by_label,
        "by_source_domain": by_source_domain,
        "invalid_gold": {
            "reason_counts": dict(sorted(invalid_reason_counts.items())),
            "examples": [bad.as_dict() for bad in invalid[:30]],
        },
    }


def run(
    test_path: Path,
    output_path: Path,
    *,
    mode: str = "rule",
    limit: int | None = None,
) -> dict[str, Any]:
    """执行 SkillSpan test 评测。"""
    if mode != "rule":
        raise ValueError("当前只支持 --mode rule")

    rows = read_json(test_path)
    if not isinstance(rows, list):
        raise ValueError(f"{test_path} 必须是 JSON 数组")
    if limit is not None:
        rows = rows[:limit]

    samples = parse_gold_samples(rows)
    predictions = predict_rule_spans(samples)
    report = build_report(samples, predictions, mode=mode, test_path=test_path)
    write_json(output_path, report)
    return report


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-path", type=Path, default=Path("optimize/CN_skillspan_lkst_test.json"),
                   help="CN_skillspan_lkst_test.json 路径")
    p.add_argument("--output-path", type=Path,
                   default=cfg.paths.canonical_root / "skillspan_eval_report.json",
                   help="评测 JSON 输出路径")
    p.add_argument("--limit", type=int, default=None, help="调试时限制评测样本数")
    p.add_argument("--mode", choices=["rule"], default="rule", help="评测模式，初版仅支持 rule")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    stats = run(
        test_path=args.test_path,
        output_path=args.output_path,
        mode=args.mode,
        limit=args.limit,
    )
    span_f1 = stats["metrics"]["span_exact"]["f1"]
    typed_f1 = stats["metrics"]["typed_exact"]["f1"]
    relaxed_f1 = stats["metrics"]["relaxed_overlap"]["f1"]
    print(
        "完成："
        f"samples={stats['data_summary']['samples']} "
        f"gold={stats['data_summary']['valid_gold_spans']} "
        f"pred={stats['data_summary']['pred_spans']} "
        f"span_f1={span_f1:.4f} typed_f1={typed_f1:.4f} relaxed_f1={relaxed_f1:.4f}"
    )
    print(f"报告已写入：{args.output_path}")
