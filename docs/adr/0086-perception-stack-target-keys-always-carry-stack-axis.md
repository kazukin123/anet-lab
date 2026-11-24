# Perception の stack の対象のキーは、1 枚でも必ず stack 軸を持つ

> 2026-11-24 注記（[ADR 0096](0096-stack-axis-and-chain-validity-are-ensured-by-the-caller-not-the-forwarder-mechanism.md)）。重ねる部品を設定に書かなくても 1 枚の重ねが入る仕掛けは無くなり、長さ 1 の軸は、使う側（DefaultDQN）が足す。重ねる枚数は、並びの部品ではなく、Agent 全体の設定が持つ（[ADR 0095](0095-frame-stack-setting-is-agent-wide-and-replay-buffer-answers-sample-spec.md)）。「stack の対象のキーは、1 枚でも必ず stack 軸を持つ」と、Network の定義の先頭の module で軸の扱いを書く決まりは、そのまま残る。

重ねる枚数が 1 のとき、行動選択と可視化には長さ 1 の軸があり、Network の入力 spec と Sample には無い、という食い違いがあった（[PRD161](../memo/0161_observation_pipeline_10rfd.md)）。**stack の対象のキーは必ず `[B, S, *shape]`（S は 1 以上。枚数の既定は 1 で、重ねる部品を設定に書かなくても 1 枚の重ねが入る）とし、Network の定義は先頭の module で軸の扱いを書き、書いていないときは WARN で知らせて止めない**ことを決定する。stack の有無で形の意味が変わらず、軸の扱いが定義から読める。

## Considered Options

- **1 枚のときは軸を足さない**: 枚数で形の意味が変わり、次元の数から見分ける道具が残る。却下。
- **必ず軸を持ち、Network の入口で既定で畳む**: 定義から軸の扱いが読めなくなる。却下。
- **先頭の module が無ければエラーで止める**: 軸を自分で扱う新しい module を塞ぐ。却下。
