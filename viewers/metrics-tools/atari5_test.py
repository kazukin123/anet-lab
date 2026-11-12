#!/usr/bin/env python3

import math
import re
import unittest
from array import array
from pathlib import Path

import atari5 as subject
from inspect_run import parse_range


REPO_ROOT = Path(__file__).resolve().parents[2]
ATARI_ENV_SOURCE = REPO_ROOT / "core" / "envs" / "atari1" / "src" / "AtariEnv.cpp"

# HnsTableDqn57() の本体だけを切り出すための目印。49 ゲーム表を巻き込まないよう終端も見る。
HNS57_TABLE_BEGIN = "HnsTableDqn57()"
HNS57_ENTRY_RE = re.compile(
    r'\{\s*"(?P<game>[a-z_]+)"\s*,\s*\{\s*(?P<random>-?[\d.]+)f\s*,\s*(?P<human>-?[\d.]+)f\s*\}\s*\}'
)


def series(steps, values):
    """TagSeries の読み取り面だけを持つ最小の代役。array 型も本物へ合わせる。"""
    holder = type("Holder", (), {})()
    holder.steps = array("q", steps)
    holder.values = array("d", values)
    holder.present = True
    return holder


class NormalizeTest(unittest.TestCase):
    def test_random_and_human_anchor_the_scale(self):
        for game, (random_score, human_score) in subject.HNS57.items():
            self.assertAlmostEqual(subject.normalize(game, random_score), 0.0, places=6)
            self.assertAlmostEqual(subject.normalize(game, human_score), 100.0, places=6)

    def test_double_dunk_scale_is_small(self):
        # human - random が 2.2 点しかないので、1 点が 45 HNS 点になる。
        self.assertAlmostEqual(subject.normalize("double_dunk", -17.5), 50.0, places=4)


class Atari5ScoreTest(unittest.TestCase):
    def test_matches_the_closed_form(self):
        hns = {
            "battle_zone": 301.9,
            "double_dunk": 1621.9,
            "name_this_game": 358.0,
            "phoenix": 1749.6,
            "qbert": 154.1,
        }
        expected = (
            10.0
            ** sum(subject.A5_WEIGHTS[game] * math.log10(1.0 + hns[game]) for game in hns)
            - 1.0
        )
        score, problems = subject.atari5_score(hns)
        self.assertEqual(problems, [])
        self.assertAlmostEqual(score, expected, places=9)

    def test_random_policy_scores_zero(self):
        # 切片を 0 にしてあるので、全ゲーム random (hns 0%) なら 0 になる。
        score, problems = subject.atari5_score({game: 0.0 for game in subject.GAMES})
        self.assertEqual(problems, [])
        self.assertAlmostEqual(score, 0.0, places=9)

    def test_missing_game_is_reported_not_dropped(self):
        hns = {game: 100.0 for game in subject.GAMES}
        del hns["phoenix"]
        score, problems = subject.atari5_score(hns)
        self.assertIsNone(score)
        self.assertEqual(len(problems), 1)
        self.assertTrue(problems[0].startswith(subject.INCOMPLETE_SET))
        self.assertIn("phoenix", problems[0])

    def test_below_random_game_blocks_the_estimate(self):
        # 項を落とすと重み和が変わって過大評価になるので、算出しないのが正しい。
        hns = {game: 100.0 for game in subject.GAMES}
        hns["double_dunk"] = -181.8
        score, problems = subject.atari5_score(hns)
        self.assertIsNone(score)
        self.assertEqual(len(problems), 1)
        self.assertIn("double_dunk", problems[0])

    def test_weights_are_the_published_ones(self):
        self.assertEqual(sorted(subject.A5_WEIGHTS), sorted(subject.GAMES))
        self.assertAlmostEqual(sum(subject.A5_WEIGHTS.values()), 0.9653, places=6)


class AggregateTest(unittest.TestCase):
    def test_reports_all_three_centers(self):
        node, problems = subject.aggregate(
            {
                "battle_zone": 300.0,
                "double_dunk": 1600.0,
                "name_this_game": 350.0,
                "phoenix": 1700.0,
                "qbert": 150.0,
            }
        )
        self.assertEqual(problems, [])
        self.assertAlmostEqual(node["median"], 350.0)
        self.assertAlmostEqual(node["mean"], 820.0)
        self.assertAlmostEqual(node["trimmed_mean3"], (300.0 + 350.0 + 1600.0) / 3.0)
        self.assertEqual(node["n_games"], 5)

    def test_incomplete_set_leaves_trimmed_mean_empty(self):
        node, _ = subject.aggregate({"qbert": 10.0, "phoenix": 20.0})
        self.assertIsNone(node["atari5"])
        self.assertIsNone(node["trimmed_mean3"])
        self.assertEqual(node["n_games"], 2)


class DropIncompleteTest(unittest.TestCase):
    def test_keeps_window_problems_and_drops_run_shortage(self):
        problems = [f"{subject.INCOMPLETE_SET} missing: phoenix", "qbert: something else"]
        self.assertEqual(subject.drop_incomplete(problems, True), problems)
        self.assertEqual(subject.drop_incomplete(problems, False), ["qbert: something else"])


class RangeTest(unittest.TestCase):
    def test_trailing_window_is_relative_to_the_reached_step(self):
        self.assertEqual(subject.resolve_bounds(parse_range("-10M:"), 50_000_000),
                         (40_000_000, 50_000_000))

    def test_absolute_range_is_taken_as_written(self):
        self.assertEqual(subject.resolve_bounds(parse_range("10M:20M"), 50_000_000),
                         (10_000_000, 20_000_000))

    def test_missing_spec_covers_everything(self):
        self.assertEqual(subject.resolve_bounds(None, 42), (0, 42))


class SliceTest(unittest.TestCase):
    def test_bounds_are_inclusive_on_both_ends(self):
        entry = series([1, 2, 3, 4, 5], [10.0, 20.0, 30.0, 40.0, 50.0])
        self.assertEqual(subject.slice_values(entry, 2, 4), [20.0, 30.0, 40.0])

    def test_stats_separate_the_window_from_the_whole_run(self):
        entry = series([1, 2, 3], [10.0, 50.0, 30.0])
        stats = subject.stats_of(entry, 1, 2)
        self.assertAlmostEqual(stats["mean"], 30.0)
        self.assertEqual(stats["n"], 2)
        self.assertAlmostEqual(stats["last"], 30.0)
        self.assertAlmostEqual(stats["max"], 50.0)
        self.assertEqual(stats["n_all"], 3)

    def test_absent_entry_yields_nulls_not_zeros(self):
        stats = subject.stats_of(None, 0, 10)
        self.assertIsNone(stats["mean"])
        self.assertIsNone(stats["last"])
        self.assertIsNone(stats["max"])


class ReferenceTableTest(unittest.TestCase):
    def test_every_entry_is_complete_and_normalizable(self):
        for key, spec in subject.REFERENCES.items():
            with self.subTest(reference=key):
                self.assertEqual(sorted(spec["scores"]), sorted(subject.GAMES))
                self.assertIn(spec["source"], subject.SOURCE_LABELS)
                self.assertTrue(spec["label"])
                self.assertTrue(spec["note"])
                node, problems = subject.aggregate(
                    {game: subject.normalize(game, value) for game, value in spec["scores"].items()}
                )
                self.assertEqual(problems, [])
                self.assertIsNotNone(node["atari5"])

    def test_ordering_matches_the_published_ranking(self):
        def score(key):
            spec = subject.REFERENCES[key]
            value, _ = subject.atari5_score(
                {game: subject.normalize(game, raw) for game, raw in spec["scores"].items()}
            )
            return value

        self.assertGreater(score("btr"), score("rainbow"))
        self.assertGreater(score("rainbow"), score("c51"))
        self.assertGreater(score("c51"), score("dqn"))
        self.assertGreater(score("muzero"), score("btr"))

    def test_output_strings_stay_ascii(self):
        # 日本語 Windows の cp932 コンソールへ出すため、出力文字列は ASCII に保つ。
        for key, spec in subject.REFERENCES.items():
            with self.subTest(reference=key):
                for field in ("label", "budget", "note"):
                    spec[field].encode("ascii")
        for label in subject.SOURCE_LABELS.values():
            label.encode("ascii")
        for _, description in subject.CHANNELS.values():
            description.encode("ascii")


class HnsConstantsTest(unittest.TestCase):
    """HNS57 が AtariEnv.cpp の 57 ゲーム表から離れていないことを確かめる。

    参照エージェントの正規化だけがこの定数を使うので、C++ 側が動いても Run の
    メトリクスは静かに正しいままになる。ここで落とすのが唯一の検出点。
    """

    def test_matches_the_cpp_table(self):
        source = ATARI_ENV_SOURCE.read_text(encoding="utf-8")
        begin = source.index(HNS57_TABLE_BEGIN)
        end = source.index("};", begin)
        table = {
            match.group("game"): (float(match.group("random")), float(match.group("human")))
            for match in HNS57_ENTRY_RE.finditer(source[begin:end])
        }
        self.assertGreaterEqual(len(table), len(subject.GAMES))
        for game in subject.GAMES:
            with self.subTest(game=game):
                self.assertIn(game, table)
                self.assertEqual(subject.HNS57[game], table[game])


class ChannelTagTest(unittest.TestCase):
    def test_requests_score_hns_and_length_for_every_channel(self):
        tags = subject.channel_tags(["eval2"])
        self.assertIn("52_eval2/10_game_score_mean", tags)
        self.assertIn("52_eval2/12_hns57_mean", tags)
        self.assertIn("52_eval2/20_game_len_mean", tags)
        self.assertIn(subject.PERF_ELAPSE_TAG, tags)
        self.assertIn(subject.PERF_RATE_TAG, tags)

    def test_default_channels_are_known(self):
        for name in subject.DEFAULT_CHANNELS:
            self.assertIn(name, subject.CHANNELS)


class CliTest(unittest.TestCase):
    def test_subcommand_is_required(self):
        self.assertEqual(subject.main([]), subject.EXIT_USAGE)

    def test_unknown_reference_is_a_usage_error(self):
        code = subject.main(["compare", "nonexistent_run", "--ref", "nope"])
        self.assertEqual(code, subject.EXIT_USAGE)


if __name__ == "__main__":
    unittest.main()
