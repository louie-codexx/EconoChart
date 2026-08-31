from __future__ import annotations

import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from econochart.config import ROOT
from econochart.data.cleanup_legacy import _validated_target
from econochart.data.generator import generate_companies
from econochart.data.public import (
    _chartqapro_example,
    _decontaminate_chartqa_rows,
    _remove_unreferenced_chartqa_split_images,
    _safe_extract_zip,
    prepare_chartqapro,
    prepare_mmefinance,
)
from econochart.data.schema import validate_record
from econochart.data.subsets import select_grpo_records, select_sft_records, select_validation_records
from econochart.data.training import load_record_sources, make_grpo_example, make_sft_example
from econochart.data.validate import validate_dataset
from econochart.io import read_jsonl, write_jsonl


class DataPipelineTests(unittest.TestCase):
    @staticmethod
    def _chartqa_row(record_id: str, split: str, digest: str, source_index: int) -> dict:
        return {
            "id": record_id,
            "split": split,
            "metadata": {"image_sha256": digest, "source_index": source_index},
        }

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

    def test_chartqapro_preserves_year_flags_that_are_not_turn_aligned(self) -> None:
        prompt, answer, answers, years = _chartqapro_example(
            {
                "Question": [
                    "What are the four categories?",
                    "In which year was the percentage highest?",
                ],
                "Answer": ["[# filed, % omitted, % withdrawn, % voted]", "2016"],
                "Year": ["NO", "YES", "NO", "NO"],
            }
        )

        self.assertIn("Assistant: [# filed, % omitted, % withdrawn, % voted]", prompt)
        self.assertEqual(answer, "2016")
        self.assertEqual(answers, ["[# filed, % omitted, % withdrawn, % voted]", "2016"])
        self.assertEqual(years, ["NO", "YES", "NO", "NO"])

    def test_chartqapro_rejects_unpaired_qa_and_invalid_year_flags(self) -> None:
        with self.assertRaisesRegex(ValueError, "Question/Answer length mismatch"):
            _chartqapro_example(
                {
                    "Question": ["First?", "Second?"],
                    "Answer": ["only one answer"],
                    "Year": ["NO"],
                }
            )
        with self.assertRaisesRegex(ValueError, "invalid Year flags"):
            _chartqapro_example(
                {
                    "Question": ["Question?"],
                    "Answer": ["Answer"],
                    "Year": ["MAYBE"],
                }
            )

    def test_chartqapro_prepare_excludes_empty_final_answers_without_refill(self) -> None:
        rows = [
            {
                "image": "retained-mismatch",
                "Question": ["First?", "Second?"],
                "Answer": ["first", "second"],
                "Year": ["NO", "YES", "NO", "NO"],
                "Question Type": "Conversational",
                "Paragraph": "",
            },
            {
                "image": "removed-mismatch",
                "Question": ["First?", "Second?"],
                "Answer": ["first", ""],
                "Year": ["NO"],
                "Question Type": "Conversational",
                "Paragraph": "",
            },
            {
                "image": "removed-aligned",
                "Question": ["First?", "Second?"],
                "Answer": ["first", ""],
                "Year": ["NO", "NO"],
                "Question Type": "Conversational",
                "Paragraph": "",
            },
            {
                "image": "retained-factoid",
                "Question": ["Value?"],
                "Answer": ["42"],
                "Year": ["NO"],
                "Question Type": "Factoid",
                "Paragraph": "context",
            },
        ]
        fake_datasets = SimpleNamespace(load_dataset=lambda *_args, **_kwargs: rows)

        with tempfile.TemporaryDirectory() as temp_dir:
            temporary_root = Path(temp_dir)
            output_root = temporary_root / "data" / "generated" / "public" / "chartqapro"
            config = {
                "project": {"seed": 20260821},
                "public": {
                    "chartqapro": {
                        "dataset_id": "test/chartqapro",
                        "output_root": str(output_root),
                        "max_samples": {"test": None},
                    }
                },
            }
            save_results = [
                ("a" * 64, "data/generated/public/chartqapro/images/test/a.png"),
                ("b" * 64, "data/generated/public/chartqapro/images/test/b.png"),
            ]
            with (
                patch.dict("sys.modules", {"datasets": fake_datasets}),
                patch("econochart.data.public.ROOT", temporary_root),
                patch("econochart.data.public._coerce_pil_image", side_effect=lambda image: image),
                patch("econochart.data.public._save_deduplicated_image", side_effect=save_results) as save_image,
            ):
                manifest = prepare_chartqapro(config)

            self.assertEqual(save_image.call_count, 2)
            self.assertEqual(manifest["selected_source_records"], {"test": 4})
            self.assertEqual(manifest["records"], {"test": 2})
            self.assertEqual(manifest["source_question_types"], {"Conversational": 3, "Factoid": 1})
            self.assertEqual(manifest["question_types"], {"Conversational": 1, "Factoid": 1})
            self.assertEqual(
                manifest["source_quality"],
                {
                    "policy": (
                        "preserve raw Year flags; exclude rows with empty final official answers; "
                        "do not refill"
                    ),
                    "before_records": 4,
                    "after_records": 2,
                    "removed_empty_final_answer_count": 2,
                    "removed_empty_final_answer_source_indices": [1, 2],
                    "removed_empty_final_answer_record_ids": [
                        "chartqapro_test_00001",
                        "chartqapro_test_00002",
                    ],
                    "year_flag_length_mismatch_count": 2,
                    "year_flag_length_mismatch_source_indices": [0, 1],
                    "retained_year_flag_length_mismatch_source_indices": [0],
                    "cap_refilled": False,
                },
            )
            self.assertEqual(
                manifest["role"],
                "fixed evaluable external challenge subset; never used for training or model selection",
            )
            written_rows = list(read_jsonl(output_root / "annotations" / "test.jsonl"))
            self.assertEqual(
                [row["id"] for row in written_rows],
                ["chartqapro_test_00000", "chartqapro_test_00003"],
            )
            self.assertEqual(written_rows[0]["metadata"]["year_flags"], ["NO", "YES", "NO", "NO"])
            self.assertTrue(all(validate_record(row) == [] for row in written_rows))

    def test_mmefinance_prepare_preserves_open_answers_and_audit_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temporary_root = Path(temp_dir)
            source_tsv = temporary_root / "MMfin.tsv"
            source_zip = temporary_root / "MMfin.zip"
            output_root = temporary_root / "data" / "generated" / "public" / "mmefinance"
            fields = [
                "index",
                "image_path",
                "image_type",
                "image_style",
                "task_category",
                "question",
                "answer",
                "background",
            ]
            source_rows = [
                {
                    "index": "0",
                    "image_path": "candlestick_chart/sample-a.png",
                    "image_type": "Candlestick Chart",
                    "image_style": "Professional",
                    "task_category": "Risk Warning",
                    "question": "What risk is visible?",
                    "answer": "A sharp drawdown raises short-term volatility risk.",
                    "background": "Use only the displayed price series.",
                },
                {
                    "index": "1",
                    "image_path": "table/sample-b.jpg",
                    "image_type": "Table",
                    "image_style": "Document",
                    "task_category": "Accurate Numerical Calculation",
                    "question": "What is the two-year total?",
                    "answer": "The total is 42.5 million.\nCalculation: 20.0 + 22.5.",
                    "background": "Values are in millions.",
                },
            ]
            with source_tsv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
                writer.writeheader()
                writer.writerows(source_rows)
            with zipfile.ZipFile(source_zip, "w") as archive:
                archive.writestr("MMfin/candlestick_chart/sample-a.png", b"fixture-a")
                archive.writestr("MMfin/table/sample-b.jpg", b"fixture-b")

            config = {
                "public": {
                    "mmefinance": {
                        "dataset_id": "test/MME-Finance",
                        "revision": "fixture-revision",
                        "annotation_file": "MMfin.tsv",
                        "image_archive": "MMfin.zip",
                        "output_root": str(output_root),
                        "expected_rows": 2,
                    }
                }
            }
            with (
                patch("econochart.data.public.ROOT", temporary_root),
                patch(
                    "econochart.data.public._download_hf_dataset_file",
                    side_effect=[source_tsv, source_zip],
                ),
            ):
                manifest = prepare_mmefinance(config)

            self.assertEqual(manifest["records"], {"test": 2})
            self.assertEqual(manifest["unique_images"], 2)
            self.assertEqual(
                manifest["task_categories"],
                {"Accurate Numerical Calculation": 1, "Risk Warning": 1},
            )
            self.assertEqual(manifest["image_types"], {"Candlestick Chart": 1, "Table": 1})
            self.assertIn("evaluation only", manifest["role"])
            self.assertEqual(manifest["source_files"]["extracted_file_count"], 2)

            written_rows = list(read_jsonl(output_root / "annotations" / "test.jsonl"))
            self.assertEqual(
                [row["id"] for row in written_rows],
                ["mmefinance_test_00000", "mmefinance_test_00001"],
            )
            self.assertEqual(written_rows[1]["answer"], source_rows[1]["answer"])
            self.assertEqual(
                written_rows[1]["metadata"]["task_category"],
                "Accurate Numerical Calculation",
            )
            self.assertEqual(written_rows[0]["metadata"]["dataset_revision"], "fixture-revision")
            self.assertTrue(all(validate_record(row) == [] for row in written_rows))
            self.assertTrue(
                (
                    temporary_root
                    / written_rows[0]["image"]
                ).is_file()
            )

    def test_public_zip_extraction_rejects_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temporary_root = Path(temp_dir)
            archive_path = temporary_root / "unsafe.zip"
            destination = temporary_root / "safe"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../escape.txt", "unsafe")
            with self.assertRaisesRegex(ValueError, "Unsafe ZIP member path"):
                _safe_extract_zip(archive_path, destination)
            self.assertFalse((temporary_root / "escape.txt").exists())

    def test_chartqa_decontamination_preserves_test_and_does_not_refill(self) -> None:
        shared = "1" * 64
        train_only = "2" * 64
        val_only = "3" * 64
        test_only = "4" * 64
        train_test_shared = "5" * 64
        val_test_shared = "6" * 64
        rows = {
            "train": [
                self._chartqa_row("train-shared-a", "train", shared, 1),
                self._chartqa_row("train-only", "train", train_only, 2),
                self._chartqa_row("train-shared-b", "train", shared, 3),
                self._chartqa_row("train-test-shared", "train", train_test_shared, 4),
                self._chartqa_row("train-val-test-shared", "train", val_test_shared, 5),
            ],
            "val": [
                self._chartqa_row("val-shared", "val", shared, 6),
                self._chartqa_row("val-only", "val", val_only, 7),
                self._chartqa_row("val-test-shared", "val", val_test_shared, 8),
            ],
            "test": [
                self._chartqa_row("test-only", "test", test_only, 9),
                self._chartqa_row("test-shared", "test", train_test_shared, 10),
                self._chartqa_row("test-val-shared", "test", val_test_shared, 11),
            ],
        }

        cleaned, audit = _decontaminate_chartqa_rows(rows)

        self.assertEqual([row["id"] for row in cleaned["train"]], ["train-only"])
        self.assertEqual(
            [row["id"] for row in cleaned["val"]],
            ["val-shared", "val-only"],
        )
        self.assertEqual(cleaned["test"], rows["test"])
        self.assertFalse(audit["validation_cap_refilled"])
        self.assertFalse(audit["training_cap_refilled"])
        self.assertEqual(audit["before_records"], {"train": 5, "val": 3, "test": 3})
        self.assertEqual(audit["after_records"], {"train": 1, "val": 2, "test": 3})
        self.assertEqual(audit["initial_overlap_counts"]["train_val"], 2)
        self.assertEqual(audit["initial_overlap_counts"]["train_test"], 2)
        self.assertEqual(audit["initial_overlap_counts"]["val_test"], 1)
        self.assertEqual(audit["final_overlap_counts"], {
            "train_val": 0,
            "train_test": 0,
            "val_test": 0,
        })
        self.assertEqual(audit["removed_validation_record_ids"], ["val-test-shared"])
        self.assertEqual(audit["removed_validation_image_sha256"], [val_test_shared])
        self.assertEqual(
            audit["removed_train_record_ids"],
            [
                "train-shared-a",
                "train-shared-b",
                "train-test-shared",
                "train-val-test-shared",
            ],
        )
        self.assertEqual(
            audit["removed_train_image_sha256"],
            [shared, train_test_shared, val_test_shared],
        )
        self.assertEqual(
            [row["overlaps_with"] for row in audit["removed_train_records"]],
            [["val"], ["val"], ["test"], ["test"]],
        )

    def test_chartqa_decontamination_removes_all_validation_rows_shared_with_test(self) -> None:
        shared = "a" * 64
        rows = {
            "train": [],
            "val": [
                self._chartqa_row("val-shared-a", "val", shared, 1),
                self._chartqa_row("val-shared-b", "val", shared, 2),
            ],
            "test": [self._chartqa_row("test-shared", "test", shared, 3)],
        }

        cleaned, audit = _decontaminate_chartqa_rows(rows)

        self.assertEqual(cleaned["val"], [])
        self.assertEqual(cleaned["test"], rows["test"])
        self.assertEqual(
            audit["removed_validation_record_ids"],
            ["val-shared-a", "val-shared-b"],
        )
        self.assertEqual(audit["removed_validation_image_sha256"], [shared])
        self.assertTrue(all(count == 0 for count in audit["final_overlap_counts"].values()))

    def test_chartqa_decontamination_uses_full_hash_not_short_chart_id(self) -> None:
        common_prefix = "b" * 16
        rows = {
            "train": [
                self._chartqa_row("train", "train", common_prefix + "1" * 48, 1)
            ],
            "val": [self._chartqa_row("val", "val", common_prefix + "2" * 48, 2)],
            "test": [],
        }

        cleaned, audit = _decontaminate_chartqa_rows(rows)

        self.assertEqual(cleaned, rows)
        self.assertEqual(audit["removed_train_records"], [])
        self.assertTrue(all(count == 0 for count in audit["final_overlap_counts"].values()))

    def test_chartqa_image_cleanup_is_limited_to_requested_mutable_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temporary_root = Path(temp_dir)
            output_root = temporary_root / "data" / "generated" / "public" / "chartqa"
            train_images = output_root / "images" / "train"
            val_images = output_root / "images" / "val"
            test_images = output_root / "images" / "test"
            train_images.mkdir(parents=True)
            val_images.mkdir(parents=True)
            test_images.mkdir(parents=True)
            kept_image = train_images / "kept.png"
            removed_image = train_images / "removed.png"
            unrelated_file = train_images / "notes.txt"
            kept_val_image = val_images / "kept.png"
            removed_val_image = val_images / "removed.png"
            test_image = test_images / "fixed.png"
            kept_image.write_bytes(b"kept")
            removed_image.write_bytes(b"removed")
            unrelated_file.write_text("keep", encoding="utf-8")
            kept_val_image.write_bytes(b"kept val")
            removed_val_image.write_bytes(b"removed val")
            test_image.write_bytes(b"fixed")
            train_rows = [{"image": kept_image.relative_to(temporary_root).as_posix()}]
            val_rows = [{"image": kept_val_image.relative_to(temporary_root).as_posix()}]

            with patch("econochart.data.public.ROOT", temporary_root):
                removed_train = _remove_unreferenced_chartqa_split_images(
                    output_root,
                    "train",
                    train_rows,
                )
                removed_validation = _remove_unreferenced_chartqa_split_images(
                    output_root,
                    "val",
                    val_rows,
                )

            self.assertEqual(
                removed_train,
                [removed_image.relative_to(temporary_root).as_posix()],
            )
            self.assertEqual(
                removed_validation,
                [removed_val_image.relative_to(temporary_root).as_posix()],
            )
            self.assertTrue(kept_image.is_file())
            self.assertFalse(removed_image.exists())
            self.assertTrue(unrelated_file.is_file())
            self.assertTrue(kept_val_image.is_file())
            self.assertFalse(removed_val_image.exists())
            self.assertTrue(test_image.is_file())

            with self.assertRaisesRegex(ValueError, "only supports mutable splits"):
                _remove_unreferenced_chartqa_split_images(output_root, "test", [])

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

    def test_explicit_sampling_namespace_is_independent_of_checkout_path(self) -> None:
        rows = list(
            read_jsonl(ROOT / "data" / "samples" / "econochart_v2" / "annotations" / "train.jsonl")
        )
        with tempfile.TemporaryDirectory() as temporary:
            first_path = Path(temporary) / "first" / "train.jsonl"
            second_path = Path(temporary) / "second" / "train.jsonl"
            write_jsonl(first_path, rows)
            write_jsonl(second_path, rows)
            common = {
                "max_samples": 8,
                "sampling_namespace": "portable-public-mixture-v1",
            }
            first = load_record_sources(
                [{**common, "path": str(first_path)}],
                expected_split="train",
                seed=20260821,
            )
            second = load_record_sources(
                [{**common, "path": str(second_path)}],
                expected_split="train",
                seed=20260821,
            )
        self.assertEqual([row["id"] for row in first], [row["id"] for row in second])

    def test_sft_subsets_are_nested_deterministic_and_cover_every_train_chart(self) -> None:
        root = ROOT / "data" / "samples" / "econochart_v2" / "annotations"
        rows = list(read_jsonl(root / "train.jsonl"))
        first_screen, first_train = select_sft_records(
            rows,
            screen_records_per_chart=1,
            train_records_per_chart=2,
            seed=20260821,
        )
        second_screen, second_train = select_sft_records(
            rows,
            screen_records_per_chart=1,
            train_records_per_chart=2,
            seed=20260821,
        )
        self.assertEqual(first_screen, second_screen)
        self.assertEqual(first_train, second_train)
        self.assertEqual(len(first_screen), 12)
        self.assertEqual(len(first_train), 24)
        self.assertEqual(len({row["chart_id"] for row in first_screen}), 12)
        self.assertEqual(len({row["chart_id"] for row in first_train}), 12)
        self.assertEqual(len({row["entity_id"] for row in first_train}), 6)
        self.assertTrue({row["id"] for row in first_screen} <= {row["id"] for row in first_train})

    def test_grpo_subset_preserves_entities_and_requested_task_quota(self) -> None:
        root = ROOT / "data" / "samples" / "econochart_v2" / "annotations"
        rows = list(read_jsonl(root / "grpo_train.jsonl"))
        weights = {
            "numerical_reasoning": 0.25,
            "relationship_analysis": 0.25,
            "risk_diagnosis": 0.20,
            "trend_analysis": 0.10,
            "value_retrieval": 0.20,
        }
        selected = select_grpo_records(rows, prompt_count=9, task_weights=weights, seed=20260821)
        repeated = select_grpo_records(rows, prompt_count=9, task_weights=weights, seed=20260821)
        self.assertEqual(selected, repeated)
        self.assertEqual(len(selected), 9)
        self.assertEqual(len({row["chart_id"] for row in selected}), 9)
        self.assertEqual(len({row["entity_id"] for row in selected}), 6)
        self.assertEqual(
            {task: sum(row["task_type"] == task for row in selected) for task in weights},
            {
                "numerical_reasoning": 2,
                "relationship_analysis": 2,
                "risk_diagnosis": 2,
                "trend_analysis": 1,
                "value_retrieval": 2,
            },
        )

    def test_development_panel_is_sampled_only_from_validation(self) -> None:
        root = ROOT / "data" / "samples" / "econochart_v2" / "annotations"
        rows = list(read_jsonl(root / "val.jsonl"))
        selected = select_validation_records(rows, sample_count=4, seed=20260821)
        self.assertEqual(len(selected), 4)
        self.assertTrue(all(row["split"] == "val" for row in selected))
        self.assertEqual(selected, select_validation_records(rows, sample_count=4, seed=20260821))

    def test_legacy_cleanup_rejects_targets_outside_data(self) -> None:
        with self.assertRaises(ValueError):
            _validated_target(Path("README.md"))


if __name__ == "__main__":
    unittest.main()
