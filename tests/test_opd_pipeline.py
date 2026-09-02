from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from econochart.config import ROOT
from econochart.data.opd import (
    assert_no_forbidden_overlap,
    build_opd_subsets,
    select_opd_records,
    split_disjoint_image_panels,
)
from econochart.data.schema import make_record
from econochart.io import sha256_file, write_jsonl


def _record(record_id: str, *, split: str, chart: str, task: str, image: str | None = None) -> dict:
    return make_record(
        record_id=record_id,
        dataset="econochart-v2.0",
        split=split,
        entity_id=f"entity-{chart}",
        chart_id=chart,
        image=image or f"images/{split}/{chart}.png",
        industry="software",
        view_type="financial",
        task_type=task,
        difficulty="hard" if task == "numerical_reasoning" else "medium",
        question=f"Question {record_id}?",
        answer=f"Answer {record_id}",
        ground_truth={
            "required_sections": ["结论", "数据依据"],
            "numeric_targets": [],
            "numeric_support": [],
            "grpo_eligible": True,
        },
        metadata={"scenario": "growth"},
    )


class OpdPipelineTests(unittest.TestCase):
    def test_selection_is_deterministic_unseen_and_chart_capped(self) -> None:
        rows = [
            _record(
                f"train-{index:02d}",
                split="train",
                chart=f"chart-{index // 3:02d}",
                task="numerical_reasoning" if index % 2 == 0 else "risk_diagnosis",
            )
            for index in range(24)
        ]
        kwargs = {
            "excluded_ids": {"train-00", "train-01"},
            "round_rows": [6, 6],
            "task_weights": {"numerical_reasoning": 0.5, "risk_diagnosis": 0.5},
            "seed": 20260821,
            "max_records_per_chart": 2,
        }
        first, second = select_opd_records(rows, **kwargs)
        repeated_first, repeated_second = select_opd_records(rows, **kwargs)

        self.assertEqual(first, repeated_first)
        self.assertEqual(second, repeated_second)
        selected = [*first, *second]
        selected_ids = {row["id"] for row in selected}
        self.assertEqual(len(first), 6)
        self.assertEqual(len(second), 6)
        self.assertEqual(len(selected_ids), 12)
        self.assertFalse(selected_ids & kwargs["excluded_ids"])
        chart_counts = {}
        for row in selected:
            chart_counts[row["chart_id"]] = chart_counts.get(row["chart_id"], 0) + 1
        self.assertLessEqual(max(chart_counts.values()), 2)
        self.assertEqual(
            {task: sum(row["task_type"] == task for row in selected) for task in kwargs["task_weights"]},
            {"numerical_reasoning": 6, "risk_diagnosis": 6},
        )

    def test_forbidden_evaluation_overlap_blocks_ids_and_images(self) -> None:
        selected = [_record("train-a", split="train", chart="chart-a", task="numerical_reasoning")]
        clean = [_record("test-b", split="test", chart="chart-b", task="risk_diagnosis")]
        self.assertEqual(
            assert_no_forbidden_overlap(selected, clean),
            {
                "id_overlap_count": 0,
                "image_path_overlap_count": 0,
                "image_sha256_overlap_count": 0,
                "image_content_checked": False,
            },
        )
        same_image = [
            _record(
                "test-c",
                split="test",
                chart="chart-c",
                task="risk_diagnosis",
                image=selected[0]["image"],
            )
        ]
        with self.assertRaisesRegex(ValueError, "overlaps isolated data"):
            assert_no_forbidden_overlap(selected, same_image)

    def test_content_hash_blocks_renamed_image_and_panel_split_is_group_disjoint(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            root = Path(temp_dir)
            first_image = root / "first.png"
            renamed_image = root / "renamed.png"
            other_image = root / "other.png"
            first_image.write_bytes(b"same-image")
            renamed_image.write_bytes(b"same-image")
            other_image.write_bytes(b"other-image")
            first_relative = first_image.relative_to(ROOT).as_posix()
            renamed_relative = renamed_image.relative_to(ROOT).as_posix()
            other_relative = other_image.relative_to(ROOT).as_posix()
            selected = [
                _record(
                    "train-a",
                    split="train",
                    chart="chart-a",
                    task="numerical_reasoning",
                    image=first_relative,
                )
            ]
            renamed = [
                _record(
                    "test-a",
                    split="test",
                    chart="chart-b",
                    task="risk_diagnosis",
                    image=renamed_relative,
                )
            ]
            with self.assertRaisesRegex(ValueError, "image_sha256"):
                assert_no_forbidden_overlap(selected, renamed, verify_image_content=True)

            panel_records = [
                _record(
                    "val-a1",
                    split="val",
                    chart="chart-a",
                    task="numerical_reasoning",
                    image=first_relative,
                ),
                _record(
                    "val-a2",
                    split="val",
                    chart="chart-a",
                    task="risk_diagnosis",
                    image=first_relative,
                ),
                _record(
                    "val-b1",
                    split="val",
                    chart="chart-b",
                    task="numerical_reasoning",
                    image=other_relative,
                ),
                _record(
                    "val-b2",
                    split="val",
                    chart="chart-b",
                    task="risk_diagnosis",
                    image=other_relative,
                ),
            ]
            qualification, development = split_disjoint_image_panels(
                panel_records,
                first_rows=2,
                seed=20260821,
                verify_image_content=True,
            )
            self.assertFalse(
                {row["image"] for row in qualification} & {row["image"] for row in development}
            )

    def test_builder_writes_disjoint_training_and_development_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "dataset"
            output_dir = root / "opd"
            dataset_root.mkdir()
            (dataset_root / "manifest.json").write_text('{"dataset":"fixture"}\n', encoding="utf-8")
            train = [
                _record(
                    f"train-{index:02d}",
                    split="train",
                    chart=f"chart-{index // 3:02d}",
                    task="numerical_reasoning" if index % 2 == 0 else "risk_diagnosis",
                )
                for index in range(24)
            ]
            seen_sft = train[:2]
            seen_grpo = train[2:4]
            validation = [
                _record(
                    f"val-{index:02d}",
                    split="val",
                    chart=f"val-chart-{index:02d}",
                    task="numerical_reasoning" if index % 2 == 0 else "risk_diagnosis",
                )
                for index in range(4)
            ]
            forbidden = [
                _record(
                    f"test-{index:02d}",
                    split="test",
                    chart=f"test-chart-{index:02d}",
                    task="numerical_reasoning" if index % 2 == 0 else "risk_diagnosis",
                )
                for index in range(2)
            ]
            paths = {
                "train": root / "train.jsonl",
                "sft": root / "sft.jsonl",
                "grpo": root / "grpo.jsonl",
                "validation": root / "validation.jsonl",
                "forbidden": root / "forbidden.jsonl",
            }
            for name, rows in (
                ("train", train),
                ("sft", seen_sft),
                ("grpo", seen_grpo),
                ("validation", validation),
                ("forbidden", forbidden),
            ):
                write_jsonl(paths[name], rows)

            config = {
                "seed": 20260821,
                "experiment": {"id": "fixture-opd"},
                "data": {
                    "dataset_root": str(dataset_root),
                    "source_manifest_sha256": sha256_file(dataset_root / "manifest.json"),
                    "train_source": {"path": str(paths["train"]), "expected_rows": 24, "expected_split": "train"},
                    "seen_sources": [
                        {"path": str(paths["sft"]), "expected_rows": 2, "expected_split": "train"},
                        {"path": str(paths["grpo"]), "expected_rows": 2, "expected_split": "train"},
                    ],
                    "qualification_source": {
                        "path": str(paths["validation"]),
                        "expected_rows": 4,
                        "expected_split": "val",
                    },
                    "forbidden_evaluation_sources": [
                        {"path": str(paths["forbidden"]), "expected_rows": 2, "expected_split": "test"}
                    ],
                    "output_dir": str(output_dir),
                },
                "selection": {
                    "round_rows": [6, 6],
                    "round_2_policy": "conditional_on_round_1_dev_gate",
                    "max_records_per_chart": 2,
                    "dual_rollout_fraction": 0.25,
                    "task_weights": {"numerical_reasoning": 0.5, "risk_diagnosis": 0.5},
                    "qualification_rows": 2,
                    "development_rows": 2,
                },
                "artifacts": {
                    "round_1": "round1.jsonl",
                    "round_2": "round2.jsonl",
                    "teacher_qualification": "qualification.jsonl",
                    "development": "development.jsonl",
                    "manifest": "manifest.json",
                },
                "policy": {"training_only": True},
            }
            manifest = build_opd_subsets(config)

            self.assertEqual(manifest["gates"]["seen_id_overlap_count"], 0)
            self.assertEqual(manifest["gates"]["round_id_overlap_count"], 0)
            self.assertEqual(manifest["gates"]["qualification_to_development_id_overlap_count"], 0)
            self.assertEqual(
                manifest["gates"]["qualification_to_development_image_path_overlap_count"], 0
            )
            self.assertEqual(manifest["artifacts"]["round_1"]["rows"], 6)
            self.assertEqual(manifest["artifacts"]["round_2"]["rows"], 6)
            self.assertEqual(manifest["artifacts"]["teacher_qualification"]["rows"], 2)
            self.assertEqual(manifest["artifacts"]["development"]["rows"], 2)
            self.assertEqual(manifest["artifacts"]["round_1"]["rollout_count_distribution"], {"1": 4, "2": 2})
            stored = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertNotIn("manifest_sha256", stored)


if __name__ == "__main__":
    unittest.main()
