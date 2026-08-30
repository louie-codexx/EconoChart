from __future__ import annotations

import copy
import unittest

from econochart.data.schema import encode_ground_truth
from econochart.evaluation.error_migration import analyze_error_migration


def _row(record_id: str, answer: str, prediction: str, *, completion_tokens: int = 1) -> dict:
    numeric = None
    try:
        numeric = float(answer)
    except ValueError:
        pass
    ground_truth = {
        "answer_aliases": [answer],
        "numeric_targets": [] if numeric is None else [{"name": "answer", "value": numeric}],
        "required_sections": [],
        "trend_targets": [],
        "evidence_keywords": [],
        "risk_labels": [],
    }
    return {
        "id": record_id,
        "dataset": "chartqa",
        "split": "test",
        "image": f"images/{record_id}.png",
        "question": f"question {record_id}",
        "answer": answer,
        "ground_truth": encode_ground_truth(ground_truth),
        "metadata": {},
        "prediction": prediction,
        "completion_tokens": completion_tokens,
    }


class ErrorMigrationTests(unittest.TestCase):
    def test_analysis_reconciles_transitions_and_output_proxies(self) -> None:
        baseline = [
            _row("numeric_format", "100", "100"),
            _row("text_regression", "north", "north"),
            _row("improvement", "yes", "no"),
            _row("both_correct", "200", "200"),
        ]
        candidate = [
            _row("numeric_format", "100", "【结论】100", completion_tokens=128),
            _row("text_regression", "north", "south"),
            _row("improvement", "yes", "yes"),
            _row("both_correct", "200", "200"),
        ]
        report = analyze_error_migration(baseline, candidate, max_new_tokens=128)

        self.assertEqual(report["inputs"]["paired_rows"], 4)
        transitions = report["overall"]["exact_match"]["binary_transitions"]
        self.assertEqual(
            transitions,
            {
                "both_correct": 1,
                "baseline_correct_candidate_wrong": 2,
                "baseline_wrong_candidate_correct": 1,
                "both_wrong": 0,
            },
        )
        self.assertAlmostEqual(report["overall"]["exact_match"]["baseline_to_candidate_delta"], -0.25)
        behavior = report["output_behavior"]["candidate"]
        self.assertEqual(behavior["token_cap_proxy_hits"], 1)
        self.assertEqual(behavior["structured_marker_outputs"], 1)
        reasons = report["exact_regression_proxy_attribution"]["primary_reason_counts"]
        self.assertEqual(reasons["new_token_cap_proxy_hit"], 1)
        self.assertEqual(reasons["text_label_or_content_error"], 1)
        signatures = report["exact_regression_proxy_attribution"]["nonexclusive_signature_counts"]
        self.assertEqual(signatures["new_structured_template"], 1)
        self.assertEqual(signatures["candidate_token_cap_proxy"], 1)

    def test_analysis_rejects_reference_mismatch(self) -> None:
        baseline = [_row("same", "100", "100")]
        candidate = [copy.deepcopy(baseline[0])]
        candidate[0]["question"] = "changed"
        with self.assertRaisesRegex(ValueError, "Reference field 'question' differs"):
            analyze_error_migration(baseline, candidate)

    def test_analysis_rejects_duplicate_ids(self) -> None:
        row = _row("duplicate", "100", "100")
        with self.assertRaisesRegex(ValueError, "duplicate ID"):
            analyze_error_migration([row, copy.deepcopy(row)], [row])


if __name__ == "__main__":
    unittest.main()
