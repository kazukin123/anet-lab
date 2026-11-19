"""正常完了・実更新点・有限値を照合して未成立率を求める。"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

class RunValidationTest(unittest.TestCase):
    def test_scheduled_points_follow_first_call_and_bucket_crossing(self):
        with tempfile.TemporaryDirectory() as temporary:
            run=Path(temporary)
            (run/"json").mkdir()
            definitions={"49_agent_gradnoise/33_signal_raw":{"source_key":"grad_noise_signal_raw","interval":3,
                "step_axis":"learn_step","event":"learn","target":"update_result"}}
            (run/"json/metrics.scalar.defs.json").write_text(json.dumps({"data":definitions}),encoding="utf-8")
            rows=[{"type":"scalar","tag":"90_perf/98_prd092_learn_step","step":step,"value":step}
                  for step in [0,1,5,6,10]]
            rows += [{"type":"scalar","tag":"49_agent_gradnoise/33_signal_raw","step":step,"value":2}
                     for step in [0,6,10]]
            (run/"metrics.jsonl").write_text("".join(json.dumps(x)+"\n" for x in rows),encoding="utf-8")
            tool=Path(__file__).with_name("validate_run.py")
            result=subprocess.run([sys.executable,str(tool),str(run),"--exit-code","0"],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            metric=json.loads(result.stdout)["metrics"]["49_agent_gradnoise/33_signal_raw"]
            self.assertEqual(metric["planned"],4)
            self.assertEqual(metric["finite"],3)
            self.assertEqual(metric["unavailable_rate"],0.25)
            failed=subprocess.run([sys.executable,str(tool),str(run),"--exit-code","1"],capture_output=True,text=True)
            self.assertNotEqual(failed.returncode,0)

if __name__=="__main__":
    unittest.main()
