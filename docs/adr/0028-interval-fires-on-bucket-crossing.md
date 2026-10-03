# 飛び飛びのカウンタに対する interval を bucket-crossing で定義する

Observer が参照する step 軸は、軸によって刻み幅が異なる。train_step は round ごとに +1 だが、learn_step は `num_envs × replay_ratio / batch` 刻み、exp_step は `num_envs` 刻みで飛ぶ。これらに対して従来は `step % interval == 0` で発火を判定していたため、実効周期が `LCM(刻み, interval)` に伸び、刻みが interval を割り切らない構成では発火が丸ごと欠落していた。実測では `interval=100` の指定に対し num_envs=512 で 400 更新周期、num_envs=64 で 100 更新周期となり、同じ設定の Run 同士で eval 回数が 4 倍違っていた。刻みは num_envs / replay_ratio / batch から決まるため、**設定に書いた interval の意味が他の設定値に汚染される**構造になっていた。

**interval の発火判定を「`step / interval` の商が前回発火時より増えたら発火する」（bucket-crossing）と定義する**。各バケット `[k·interval, (k+1)·interval)` を跨いだ最初のイベントで必ず 1 回だけ発火し、刻みが interval を割り切るかどうかに依存しない。実効周期は `max(interval, 刻み)` となり、位相ジッタは 1 イベント以内に収まる。

決め手は、欠落の有無が刻みの整除性という**設定間の偶然の関係**に支配されていた点にある。Run 間比較の前提が黙って壊れるため、周期が伸びること自体より発見しにくい。

## Considered Options

- **`step % interval == 0` を維持し、interval を刻みの倍数にする規約を課す** — 却下。num_envs や replay_ratio を変えるたびに全 config の interval を再計算する隠れ契約になる。しかも刻みは非整数になりうる（`update_credit` が float のため）ので、倍数規約自体が成立しない構成が存在する。
- **catch-up 方式（`while (step >= next) { fire; next += interval }`）** — 却下。1 イベントで複数バケットを跨いだとき連続発火するが、跨いだ区間の中間状態は既に失われているため、同じ状態を複数回評価・記録するだけで情報が増えない。
- **判定軸を常に train_step（刻み 1）に固定する** — 却下。step 軸は出力する step 値（グラフの横軸）も兼ねているため、判定軸を固定すると横軸の選択の自由が失われる。
- **bucket-crossing** — 採用。

## Consequences

- 発火判定に「前回発火バケット」の状態が必要になり、判定は純関数ではなくなる。共通部品 `IntervalGate` に閉じ込め、状態を持つ主体（Observer インスタンス、metrics のタグ単位インスタンス）が保持する。
- 実効周期が変わるため、config の interval 値は現行の実効値を踏まえて再設計する。既存 Run とはサンプル密度が変わる。
- `interval` が刻みより小さい場合は毎イベント発火に丸まる。イベントより細かい発火は構造的に不可能であり、これを仕様とする。
- 初回（step=0）は発火する。従来の剰余判定と同じ挙動で、学習開始直後のベースライン点が残る。
- 詳細設計は `docs/memo/053_interval_gate_perf_ema_10prd.md`。
