from __future__ import annotations

import json
import unittest
from pathlib import Path

from econochart.config import ROOT
from econochart.data.cleanup_legacy import _validated_target
from econochart.data.generator import generate_companies
from econochart.data.public import _chartqapro_example
from econochart.data.schema import validate_record
from econochart.data.training import make_grpo_example, make_sft_example
from econochart.data.validate import validate_dataset
from econochart.io import read_jsonl


class DataPipelineTests(unittest.TestCase):
    def test_generation_is_deterministic_and_coherent(self) -> None:
        first = generate_companies(192, seed=20260821)
        second = generate_companies(192, seed=20260821)
        self.assertEqual(first, second)
        self.assertEqual({company["split"] for company in first}, {"train", "val", "test"})
        self.assertEqual(len({company["entity_id"] for company in first}), 192)
        for company in first:
            metrics = company["metrics"]
            for revenue, cost, profit in zip(
                metrics["revenue_million"], metrics["cost_million"], metrics["profit_million"]
            ):
                self.assertAlmostEqual(revenue - cost, profit, delta=0.2)
            self.assertAlmostEqual(sum(company["product_mix_pct"].values()), 100.0, delta=0.2)
            self.assertAlmostEqual(sum(company["region_mix_pct"].values()), 100.0, delta=0.2)

    def test_repository_sample_passes_full_validation(self) -> None:
        report = validate_dataset(ROOT / "data" / "samples" / "econochart_v2", full_image_scan=True)
        self.assertEqual(report["status"], "passed", report["issues"])
        self.assertEqual(report["dataset_root"], "data/samples/econochart_v2")
        self.assertEqual(sum(report["rows"].values()), 64)
        self.assertEqual(sum(report["entities"].values()), 8)
        stored = json.loads(
            (ROOT / "data" / "samples" / "econochart_v2" / "validation_report.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(stored, report)

    def test_sample_schema_and_split_isolation(self) -> None:
        root = ROOT / "data" / "samples" / "econochart_v2" / "annotations"
        rows_by_split = {split: list(read_jsonl(root / f"{split}.jsonl")) for split in ("train", "val", "test")}
        for rows in rows_by_split.values():
            for row in rows:
                self.assertEqual(validate_record(row), [])
        entity_sets = {split: {row["entity_id"] for row in rows} for split, rows in rows_by_split.items()}
        self.assertFalse(entity_sets["train"] & entity_sets["val"])
        self.assertFalse(entity_sets["train"] & entity_sets["test"])
        self.assertFalse(entity_sets["val"] & entity_sets["test"])

    def test_chartqapro_conversation_keeps_history_and_targets_last_turn(self) -> None:
        prompt, answer, answers, years = _chartqapro_example(
            {
                "Question": ["What is A?", "How does it compare with B?"],
                "Answer": ["10", "A is twice B"],
                "Year": ["NO", "NO"],
            }
        )
        self.assertIn("User: What is A?", prompt)
        self.assertIn("Assistant: 10", prompt)
        self.assertTrue(prompt.endswith("Current question: How does it compare with B?"))
        self.assertEqual(answer, "A is twice B")
        self.assertEqual(answers, ["10", "A is twice B"])
        self.assertEqual(years, ["NO", "NO"])

    def test_manifest_checksums_are_present(self) -> None:
        manifest = json.loads(
            (ROOT / "data" / "samples" / "econochart_v2" / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["num_records"], 64)
        self.assertIn("annotations/train.jsonl", manifest["checksums"])

    def test_training_examples_place_image_before_question(self) -> None:
        record = next(
            read_jsonl(ROOT / "data" / "samples" / "econochart_v2" / "annotations" / "train.jsonl")
        )
        sft = make_sft_example(record)
        self.assertEqual(sft["prompt"][1]["content"][0], {"type": "image"})
        self.assertEqual(sft["prompt"][1]["content"][1]["text"], record["question"])
        self.assertEqual(sft["completion"][0]["content"][0]["text"], record["answer"])
        grpo_record = next(
            read_jsonl(ROOT / "data" / "samples" / "econochart_v2" / "annotations" / "grpo_train.jsonl")
        )
        grpo = make_grpo_example(grpo_record)
        self.assertIsNotNone(grpo)
        self.assertEqual(grpo["prompt"][1]["content"][0], {"type": "image"})

    def test_legacy_cleanup_rejects_targets_outside_data(self) -> None:
        with self.assertRaises(ValueError):
            _validated_target(Path("README.md"))


if __name__ == "__main__":
    unittest.main()
