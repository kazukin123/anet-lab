"""PRD104の新しい既定葉以外に設定差が無いことを確認する。"""
import json
import re

from common import OUT, REPO, resolve, save


def main():
    reports = {}
    # defaultとskipの双方を、保存入力と現行入力で同じ公開resolverに通す。
    for env in ("GridMaze", "LunarLander", "Atari", "DropMerge"):
        for mode in ("default", "skip"):
            pair = {}
            for stage, base in (("before", OUT / "baseline-config"), ("after", REPO / "apps/runner/config")):
                destination = OUT / "config" / stage / f"{env}_{mode}"
                destination.mkdir(parents=True, exist_ok=False)
                source = (base / (env + ".txt")).read_text(encoding="utf-8-sig")
                chains = re.findall(r"(?m)^run\.\$\s*=\s*([^#\n]+)", source)
                chain = chains[-1].strip() if chains else ""
                if mode == "skip" and "run.@skip" not in chain:
                    chain += (" > " if chain else "") + "run.@skip"
                config = destination / "run.txt"
                lines = [f"$include <{base.as_posix()}/_main.txt>", f"$include <{base.as_posix()}/{env}.txt>"]
                if mode == "skip":
                    lines.append("run.$ = " + chain)
                config.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
                pair[stage] = resolve(config, destination)
            before, after = pair["before"], pair["after"]
            # 追加した既定葉だけを許可し、既存値の変更や削除は失敗にする。
            changes = {key: [before.get(key), after.get(key)] for key in before.keys() | after.keys()
                       if before.get(key) != after.get(key)}
            def allowed(key, values):
                old, new = values
                return old is None and ((key.endswith("learner.skip.target_window") and new == "decision")
                    or (key.endswith(".skip.max_length_mode") and new == "full"))
            unexpected = {key: values for key, values in changes.items() if not allowed(key, values)}
            reports[f"{env}_{mode}"] = {"changes": changes, "unexpected": unexpected}
            print(env, mode, "changes=", len(changes), "unexpected=", len(unexpected), flush=True)
    save(OUT / "config/comparison.json", reports)
    if any(value["unexpected"] for value in reports.values()):
        raise SystemExit("Unexpected configuration changes")


if __name__ == "__main__":
    main()
