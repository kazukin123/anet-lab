"""Contract tests for review-run completion and experience budgets."""
import unittest

from summarize_review_validation import validate_execution_record


class ReviewExecutionRecordTests(unittest.TestCase):
    def test_fixed_acceptance_budgets(self):
        for case in ("atari_default", "atari_efficiency", "lunarlander_cpu", "nz_learning", "nz_reference", "old_checkpoint"):
            with self.subTest(case=case):
                budget = 256 if case == "old_checkpoint" else 500000
                validate_execution_record(case, {"exit_code": 0, "seed": 1, "experience_budget": budget})

    def test_extended_integration_budgets(self):
        for case in ("nz_learning", "nz_reference"):
            with self.subTest(case=case):
                validate_execution_record(case, {"exit_code": 0, "seed": 1, "experience_budget": 600000})

    def test_invalid_budgets_are_rejected(self):
        cases = {
            "atari_default": (499999, 600000),
            "atari_efficiency": (499999, 600000),
            "lunarlander_cpu": (499999, 600000),
            "nz_learning": (0, 499999),
            "nz_reference": (0, 499999),
            "old_checkpoint": (255, 257),
        }
        for case, budgets in cases.items():
            for budget in budgets:
                with self.subTest(case=case, budget=budget), self.assertRaises(AssertionError):
                    validate_execution_record(case, {"exit_code": 0, "seed": 1, "experience_budget": budget})

    def test_failed_runs_and_different_seeds_are_rejected(self):
        for exit_code, seed in ((1, 1), (0, 2)):
            with self.subTest(exit_code=exit_code, seed=seed), self.assertRaises(AssertionError):
                validate_execution_record("nz_reference", {"exit_code": exit_code, "seed": seed, "experience_budget": 600000})


if __name__ == "__main__":
    unittest.main()
