"""合成artifactで受入検証器の誤配線・欠損・値域の見逃しを防ぐ。"""
import json
from pathlib import Path
import tempfile
import unittest

from validate_run import analyze, PREFIX, EXPECTED, PROGRESS, CLOCK

class ValidationTests(unittest.TestCase):
    def fixture(self, directory):
        run=Path(directory)/"run_fixture"; (run/"json").mkdir(parents=True)
        definitions={tag:{"source_key":key,"event":"learn","step_axis":"learn_step","target":"update_result","scope":"train","interval":503} for tag,key in EXPECTED.items()}
        definitions["munch"]={"source_key":"munchausen_bonus_mean"}
        definitions["rfit"]={"source_key":"replay_fit_per_selectivity"}
        (run/"json/metrics.scalar.defs.json").write_text(json.dumps({"data":definitions}),encoding="utf-8")
        (run/(run.name+".log")).write_text("Draining background observers",encoding="utf-8")
        rows=[]
        for step,seconds in [(503,1),(1006,2)]:
            rows += [{"type":"scalar","tag":PROGRESS,"step":step,"value":step},
                {"type":"scalar","tag":CLOCK,"step":step,"value":seconds/3600}]
            rows += [{"type":"scalar","tag":tag,"step":step,"value":0.75} for tag in EXPECTED]
        self.write_rows(run,rows)
        return run,definitions,rows

    def write_rows(self,run,rows):
        (run/"metrics.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows),encoding="utf-8")

    def test_complete_artifact_and_fixed_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            run,_,_=self.fixture(directory)
            result=analyze(run,atari=True)
            self.assertEqual(result["max_learn_step"],1006)
            self.assertAlmostEqual(result["wall_seconds_503_1006"],1)
            self.assertEqual(set(result["points"]),set(EXPECTED))

    def test_missing_and_invalid_measurements_fail(self):
        for replacement in (None,-0.1,1.1,float("nan")):
            with self.subTest(replacement=replacement),tempfile.TemporaryDirectory() as directory:
                run,_,rows=self.fixture(directory)
                if replacement is None: rows=rows[:-1]
                else: rows[-1]["value"]=replacement
                self.write_rows(run,rows)
                with self.assertRaises(ValueError): analyze(run)

    def test_wrong_binding_and_lost_atari_profiles_fail(self):
        for mutation in ("axis","source","atari"):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                run,definitions,_=self.fixture(directory)
                if mutation=="axis": definitions[next(iter(EXPECTED))]["step_axis"]="exp_step"
                elif mutation=="source": definitions[next(iter(EXPECTED))]["source_key"]="attention_entropy.[99]"
                else: definitions.pop("rfit")
                (run/"json/metrics.scalar.defs.json").write_text(json.dumps({"data":definitions}),encoding="utf-8")
                with self.assertRaises(ValueError): analyze(run,atari=True)

if __name__=="__main__": unittest.main()
