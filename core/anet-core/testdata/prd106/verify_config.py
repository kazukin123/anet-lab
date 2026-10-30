"""新しい shared の既定葉以外の全キー・文字列値が維持されることを確認する。"""
import re

from common import OUT, REPO, resolve, save


def main():
    reports = {}
    # before と after を同じ保存済み resolver に通し、resolver 自体の差を混ぜない。
    resolver = OUT / "baseline-bin/anet-core-test.exe"
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
                lines = [f"$include <{base.as_posix()}/_main.txt>", f"$include <{base.as_posix()}/{env}.txt>"]
                if mode == "skip":
                    lines.append("run.$ = " + chain)
                config = destination / "run.txt"
                config.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
                pair[stage] = resolve(config, destination, binary=resolver)
            before, after = pair["before"], pair["after"]
            changes = {key: [before.get(key), after.get(key)] for key in before.keys() | after.keys()
                       if before.get(key) != after.get(key)}
            unexpected = {key: values for key, values in changes.items()
                          if not (values == [None, "shared"] and key.endswith(".skip.nn_forward_mode"))}
            actors = sorted({match[1] for key in after
                             if (match := re.fullmatch(r"DefaultDQNAgent\.actor\.\[([^]]+)\]\.network", key))})
            assert actors
            for actor in actors:
                prefix = f"DefaultDQNAgent.actor.[{actor}].skip."
                assert after[prefix + "nn_forward_mode"] == "shared", (env, mode, actor)
                assert prefix + "nn_forward_key" not in after, (env, mode, actor)
            reports[f"{env}_{mode}"] = {"actors": actors, "changes": changes, "unexpected": unexpected}
            print(env, mode, "changes=", len(changes), "unexpected=", len(unexpected), flush=True)
    save(OUT / "config/comparison.json", reports)
    if any(value["unexpected"] for value in reports.values()):
        raise SystemExit("Unexpected configuration changes")


if __name__ == "__main__":
    main()
