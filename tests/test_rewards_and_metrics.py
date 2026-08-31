from __future__ import annotations

import copy
import unittest

from econochart.config import ROOT
from econochart.data.schema import parse_ground_truth
from econochart.evaluation.compare import compare_rows
from econochart.evaluation.inference import _build_messages
from econochart.evaluation.metrics import (
    aggregate_metrics,
    anls,
    chartqapro_accuracy,
    chartqapro_official_rows,
    mmefinance_audit_rows,
    mmefinance_surrogate_scores,
    relaxed_accuracy,
    score_row,
)
from econochart.evaluation.runner import _align_existing_predictions
from econochart.io import read_jsonl
from econochart.rewards.components import score_completion


class RewardAndMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = list(
            read_jsonl(ROOT / "data" / "samples" / "econochart_v2" / "annotations" / "test.jsonl")
        )
        cls.all_rows = []
        for split in ("train", "val", "test"):
            cls.all_rows.extend(
                read_jsonl(ROOT / "data" / "samples" / "econochart_v2" / "annotations" / f"{split}.jsonl")
            )

    def test_reference_answers_receive_high_reward(self) -> None:
        for row in self.all_rows:
            score = score_completion(row["answer"], row["ground_truth"])
            self.assertAlmostEqual(score["overall"], 1.0, msg=(row["id"], score))
            for component, value in score.items():
                if value is not None:
                    self.assertAlmostEqual(value, 1.0, msg=(row["id"], component, value))

    def test_incorrect_unstructured_answer_scores_lower(self) -> None:
        row = self.rows[0]
        reference = score_completion(row["answer"], row["ground_truth"])["overall"]
        incorrect = score_completion("所有指标完全稳定，没有风险。", row["ground_truth"])["overall"]
        self.assertLess(incorrect, reference)

    def test_missing_required_sections_receive_zero_format_reward(self) -> None:
        row = next(
            item
            for item in self.all_rows
            if parse_ground_truth(item["ground_truth"]).get("required_sections")
        )
        score = score_completion("所有指标完全稳定，没有风险。", row["ground_truth"])
        self.assertEqual(score["format"], 0.0)

    def test_numeric_reward_penalizes_unsupported_and_misassigned_values(self) -> None:
        row = next(
            item
            for item in self.all_rows
            if item["view_type"] == "financial" and item["task_type"] == "value_retrieval"
        )
        reference = score_completion(row["answer"], row["ground_truth"])
        dumped = score_completion(row["answer"] + "\n补充数据：999999。", row["ground_truth"])
        self.assertEqual(reference["numeric_precision"], 1.0)
        self.assertLess(dumped["numeric_precision"], 1.0)
        self.assertLess(dumped["numeric"], reference["numeric"])

        targets = parse_ground_truth(row["ground_truth"])["numeric_targets"]
        revenue, profit = targets[0]["value"], targets[1]["value"]
        swapped = (
            f"【结论】收入为{profit}百万元。以下为较长的独立说明文字。\n"
            f"利润为{revenue}百万元。以上为较长的独立说明文字。\n"
            "【数据依据】上述两个数值被故意交换。"
        )
        swapped_score = score_completion(swapped, row["ground_truth"])
        self.assertLess(swapped_score["numeric_recall"], 1.0)

    def test_chartqa_relaxed_accuracy(self) -> None:
        self.assertEqual(relaxed_accuracy("104", ["100"]), 1.0)
        self.assertEqual(relaxed_accuracy("106", ["100"]), 0.0)
        self.assertEqual(relaxed_accuracy("50%", ["0.5"]), 1.0)
        self.assertEqual(relaxed_accuracy("north", ["North"]), 1.0)

    def test_anls_threshold(self) -> None:
        self.assertEqual(anls("United Airlines", "united airlines"), 1.0)
        self.assertEqual(anls("alpha", "zzzzz"), 0.0)

    def test_chartqapro_year_requires_exact_match(self) -> None:
        row = {
            "answer": "2020",
            "metadata": {"year_flags": ["YES"], "question_type": "Factoid"},
        }
        self.assertEqual(chartqapro_accuracy(row, "2020"), 1.0)
        self.assertEqual(chartqapro_accuracy(row, "2021"), 0.0)

    def test_chartqapro_conversation_scores_with_last_year_flag_and_preserves_raw_export(self) -> None:
        row = {
            "dataset": "chartqapro",
            "answer": "2020",
            "prediction": "2021",
            "metadata": {
                "answer_sequence": ["first", "2020"],
                "year_flags": ["YES", "YES", "YES", "NO"],
                "question_type": "Conversational",
            },
        }

        self.assertEqual(chartqapro_accuracy(row, row["prediction"]), 1.0)
        self.assertEqual(
            chartqapro_official_rows([row]),
            [
                {
                    "Answer": ["first", "2020"],
                    "Question Type": "Conversational",
                    "Year": ["YES", "YES", "YES", "NO"],
                    "prediction": "2021",
                }
            ],
        )

    def test_internal_dataset_version_is_scored(self) -> None:
        row = copy.deepcopy(self.rows[0])
        row["prediction"] = row["answer"]
        score = score_row(row)
        self.assertIn("overall", score)
        self.assertGreaterEqual(score["overall"], 0.9)

    def test_aggregation_builds_capability_slices(self) -> None:
        rows = []
        for row in self.rows:
            prediction = copy.deepcopy(row)
            prediction.update({"prediction": row["answer"], "latency_seconds": 0.25, "completion_tokens": 50})
            rows.append(prediction)
        report = aggregate_metrics(rows)
        self.assertEqual(report["rows"], len(rows))
        self.assertIn("task_type", report["slices"])
        self.assertIn("econochart-v2.0", report["slices"]["dataset"])
        self.assertAlmostEqual(report["efficiency"]["mean_latency_seconds"], 0.25)

    def test_mmefinance_open_answer_surrogates_are_auditable(self) -> None:
        reference = "Revenue rose from 100 million to 120 million, a 20% increase."
        prediction = "Revenue increased 20% from 100 to 120 million."
        scores = mmefinance_surrogate_scores(reference, prediction)
        self.assertEqual(scores["surrogate_exact_match"], 0.0)
        self.assertGreater(scores["surrogate_token_f1"], 0.5)
        self.assertEqual(scores["surrogate_numeric_precision"], 1.0)
        self.assertEqual(scores["surrogate_numeric_recall"], 1.0)
        self.assertEqual(scores["output_nonempty"], 1.0)

        row = {
            "id": "mmefinance_test_00001",
            "dataset": "mmefinance",
            "image": "data/generated/public/mmefinance/images/test/table/sample.jpg",
            "task_type": "public_mmefinance",
            "view_type": "Table",
            "industry": "finance",
            "difficulty": "hard",
            "question": "How did revenue change?",
            "answer": reference,
            "prediction": prediction,
            "ground_truth": {"answer_aliases": [reference]},
            "metadata": {
                "source_index": 1,
                "source_image_path": "table/sample.jpg",
                "task_category": "Reason Explanation",
                "image_type": "Table",
                "image_style": "Document",
                "background": "Values are in millions.",
            },
        }
        report = aggregate_metrics([row])
        self.assertIn("score_contract", report)
        self.assertIn("Reason Explanation", report["slices"]["task_category"])
        self.assertIn("Table", report["slices"]["image_type"])
        self.assertIn("Document", report["slices"]["image_style"])
        self.assertIn("surrogate_token_f1", report["overall"])

        audit_rows = mmefinance_audit_rows([row])
        self.assertEqual(len(audit_rows), 1)
        self.assertEqual(audit_rows[0]["index"], 1)
        self.assertEqual(audit_rows[0]["task_category"], "Reason Explanation")
        self.assertEqual(audit_rows[0]["reference_answer"], reference)
        self.assertEqual(audit_rows[0]["prediction"], prediction)
        self.assertIn("surrogate_numeric_recall", audit_rows[0]["surrogate_scores"])

        image = object()
        messages = _build_messages(row, image)
        user_text = messages[1]["content"][1]["text"]
        self.assertIn("Background:\nValues are in millions.", user_text)
        self.assertTrue(user_text.endswith("Question:\nHow did revenue change?"))

    def test_paired_comparison_detects_reference_improvement(self) -> None:
        baseline = []
        candidate = []
        for row in self.rows:
            baseline.append({**copy.deepcopy(row), "prediction": "所有指标稳定。"})
            candidate.append({**copy.deepcopy(row), "prediction": row["answer"]})
        report = compare_rows(baseline, candidate, bootstrap_samples=200, seed=7)
        self.assertGreater(report["overall"]["overall"]["baseline_to_candidate_delta"], 0)
        self.assertGreater(report["overall"]["overall"]["probability_delta_positive"], 0.95)
        self.assertIn(self.rows[0]["industry"], report["slices"]["industry"])
        scenario = self.rows[0]["metadata"]["scenario"]
        self.assertIn(scenario, report["slices"]["scenario"])

    def test_evaluation_resume_requires_a_matching_prefix(self) -> None:
        prefix = [{**copy.deepcopy(row), "prediction": row["answer"]} for row in self.rows[:2]]
        aligned = _align_existing_predictions(self.rows, prefix)
        self.assertEqual([row["id"] for row in aligned], [row["id"] for row in self.rows[:2]])

        non_prefix = [{**copy.deepcopy(self.rows[1]), "prediction": self.rows[1]["answer"]}]
        with self.assertRaises(ValueError):
            _align_existing_predictions(self.rows, non_prefix)

        mismatched = copy.deepcopy(prefix)
        mismatched[0]["question"] = "changed"
        with self.assertRaises(ValueError):
            _align_existing_predictions(self.rows, mismatched)

    def test_inference_messages_use_multimodal_content_blocks(self) -> None:
        image = object()
        messages = _build_messages(self.rows[0], image)
        self.assertTrue(all(isinstance(message["content"], list) for message in messages))
        self.assertEqual(messages[0]["content"][0]["type"], "text")
        self.assertEqual(messages[1]["content"][0]["type"], "image")
        self.assertIs(messages[1]["content"][0]["image"], image)


if __name__ == "__main__":
    unittest.main()
