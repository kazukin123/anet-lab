---
status: proposed
---

# Policy churn は fixed probe で optimizer 前後を測り、target は sync 後の不一致を測る

表現の健全性だけでは Breakout の成績や壁突破を説明できないため、1 learner update が online DQN の greedy 方策をどれだけ変えたかと、更新後の online / target 方策がどれだけ食い違うかを観測する。BTR の実装を直写しせず、**churn 専用の一様非復元 probe、caller-owned の独立 RNG、IQN fixed midpoint taus、autocast 無効の FP32 forward、optimizer step 前後の online 比較、target update 後の online / target 比較**を採用する。これにより、測定差を乱数、異なる状態集合、BF16 の丸めではなく parameter update と target の遅れに帰属させ、測定の ON / OFF が学習系列へ干渉しない境界を作る。

## Considered Options

- **BTR と同じ PER sample・毎 update・可変 tau**: 参照実装との見かけの近さは高いが、学習対象への偏りと tau の再抽選が churn に混入し、絶対値の意味を固定できないため棄却した。
- **plasticity の実バッチを共有**: 状態 Tensor の寿命と実行順を plasticity 側へ持ち込み、両 metrics の cadence も結合するため棄却した。sampler の仕組みだけを共有し、実バッチと RNG は分離する。
- **target update の前後を比較**: hard update では意味を持つが、soft update では離散的な sync が無く、online churn と同じ意味にもならないため棄却した。両 mode で同じ定義になる sync 後の policy disagreement を採用する。
- **custom interval を hard interval と必ず互いに素にする**: 位相被覆は良くなるが、利用目的に属する設定判断であり、実行可能性の契約ではないため fail-fast にしない。`C / gcd(C, interval) = 1` の構造的縮退だけを起動時の 1 回 WARN で可視化し、2 位相以上の被覆は実験設定側の判断として許容する。

## Consequences

- 1 回の測定は同じ probe 状態と fixed taus を online-before / online-after / target-after で共有し、Q 由来指標を比較可能にする。
- ReplayBuffer の `SampleUniqueUniform` は caller が RNG を渡す契約へクリーンブレークし、plasticity と churn は別の named seed を所有する。この移行は既存の plasticity probe 抽選系列を意図的に断絶するため、新旧ビルド間の metrics checksum では `34_agent_plasticity/4x_probe_*` を除外するか、本変更を基準断絶点として扱う。
- target 指標のゼロ点は hard sync 後に置かれ、hard / soft のどちらでも「更新後の online と target の不一致」という意味を保つ。hard update の位相は独立した `target_sync_age` で記録する。
- 初回の公開範囲は DefaultDQN に限定する。ImageCls、Rainbow、NoisyNet は各ゲートの契約が確定するまで拡張しない。
- 詳細なメトリクス、cadence、NaN、購読ゲート、受入条件は [PRD 066](../memo/done/066_policy_churn_metrics_10prd.md) を正本とする。
