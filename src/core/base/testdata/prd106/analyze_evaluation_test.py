"""評価の数え方を、得点とクリア数が食い違うゲームも含めて確認する。"""
import unittest

from analyze_evaluation import summarize, analyze_read


class EvaluationSummaryTest(unittest.TestCase):
    def test_fixed_failure_is_excluded_but_low_score_clear_eight_is_a_failure(self):
        # 面6の同じ得点20本だけが集中し、面7/8の失敗は異なる得点へ散らす。
        episodes = [{"game_score": 18925, "ram_metric.[1]": 5}] * 20
        episodes += [{"game_score": 20000 + index, "ram_metric.[1]": 6} for index in range(10)]
        episodes += [{"game_score": 28000 + index, "ram_metric.[1]": 8} for index in range(10)]
        episodes += [{"game_score": 30000, "ram_metric.[1]": 8}] * 60
        result = summarize(episodes, excluded=(6,))
        self.assertEqual(result["episodes"], 100)
        self.assertEqual(result["concentrated"], [6])
        self.assertAlmostEqual(result["breakthrough"], 0.6)
        self.assertAlmostEqual(result["passage"], 0.75)
        self.assertEqual(result["levels"][7]["ended"], 10)
        self.assertEqual(result["levels"][7]["reached"], 70)

    def test_harm_ratio_and_training_difference_use_their_own_denominators(self):
        def population(failures):
            return ([{"game_score": 28000 + index, "ram_metric.[1]": 7} for index in range(failures)]
                    + [{"game_score": 30000, "ram_metric.[1]": 8}] * (100 - failures))
        result = analyze_read({"G": population(0), "NQ": population(50),
                               "NS": population(20), "NB": population(80)}, "G", "NQ", "NS", "NB", draws=200)
        self.assertAlmostEqual(result["ratios"]["Q"]["value"], 0.625)
        self.assertAlmostEqual(result["ratios"]["S"]["value"], 0.25)
        self.assertAlmostEqual(result["impact"]["value"], 0.3)
        self.assertLess(result["impact"]["ci"][0], result["impact"]["value"])
        self.assertGreater(result["impact"]["ci"][1], result["impact"]["value"])

    def test_zero_denominator_has_no_ratios_but_keeps_training_difference(self):
        passed = [{"game_score": 30000, "ram_metric.[1]": 8}] * 100
        failed = [{"game_score": 28000 + index, "ram_metric.[1]": 7} for index in range(50)] + passed[:50]
        result = analyze_read({"G": passed, "NQ": failed, "NS": passed, "NB": passed},
                              "G", "NQ", "NS", "NB", draws=200)
        self.assertIsNone(result["ratios"])
        self.assertEqual(result["read"], "害が再現しない")
        self.assertEqual(result["excluded_ratio_draw_fraction"], 1.0)
        self.assertAlmostEqual(result["impact"]["value"], -0.5)
        self.assertEqual(result["impact_read"], "skip側を切ると下がる")

    def test_concentration_removal_changes_h_but_does_not_change_training_difference(self):
        def population(fixed, other):
            return ([{"game_score": 18925, "ram_metric.[1]": 5}] * fixed
                    + [{"game_score": 28000 + index, "ram_metric.[1]": 7} for index in range(other)]
                    + [{"game_score": 30000, "ram_metric.[1]": 8}] * (100 - fixed - other))
        result = analyze_read({"G": population(20, 5), "NQ": population(20, 20),
                               "NS": population(0, 35), "NB": population(0, 50)}, "G", "NQ", "NS", "NB", draws=200)
        self.assertEqual(result["excluded"], [6])
        self.assertAlmostEqual(result["ratios"]["Q"]["value"], 3 / 7)
        self.assertAlmostEqual(result["impact"]["value"], 0.1)
        self.assertAlmostEqual(result["passage_impact"]["value"], 0.25)


if __name__ == "__main__":
    unittest.main()
