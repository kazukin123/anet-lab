# eval schedule は既定で学習直後に評価せず、発火の時点を offset でずらす

`run.eval_schedule.[tag]` の評価は、発火の列 {offset + k·interval} の先頭から `warmup_count` 回（既定 1）を走らせず、学習直後の評価が要る Run（評価専用 Run と、新旧の実行ファイルを比べる検証資材）だけが 0 を書く。初期の重みの評価は解析で使わない捨て点で、Atari の ε=0 の評価では打ち切り上限に張り付いて学習を止めることもあり、BTR と Stable-Baselines3 も学習開始時には評価しない。2 本の評価を同じ時点に発火させないことは、interval を互いに素に曲げずに `offset` で書く（PRD918）。

## Considered Options

- **既定 0（今の挙動）にして、要る所で 1 を書く**: 初期の重みの点が、書き忘れたすべての Run の系列の先頭に残る。却下
- **offset は格子だけをずらし、学習直後の 1 回は step 0 のまま**: 2 本の評価が学習直後に必ず重なる。却下
- **offset を interval 未満に限る**: 書いたとおりに動くので上限は要らず、開始の遅れも step で書けなくなる。却下

## Consequences

- `IntervalGate` は変えず、offset と先頭で止める回数は eval schedule の observer が受け持つ。メトリクスの interval は ADR 0028 のまま初回に発火する。
- offset は step で書くので、本体で offset を持つタグの interval をプロファイルや bat で変えるときは、offset も合わせて書く。
