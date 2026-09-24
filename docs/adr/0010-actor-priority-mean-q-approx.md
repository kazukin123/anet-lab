# Actor初期優先度は平均Q空間の近似とし、Learner target-network評価/UQE targetの再現を見送る

> **後発決定**: Actorヒントのtransport表現と有効性マスクに関する決定はADR 0012で置き換えた。本ADRの平均Q近似、追加forward禁止、`WithAction`、UQE再現見送りの判断は引き続き有効である。
> **後発決定（2026-09-04）**: Actorヒントの列数と`WithAction`の再計算対象は [ADR 0036](0036-actor-q-hint-three-columns-munchausen.md) で3列（Munchausen項を追加）へ改訂した。追加forward禁止とUQE再現見送りは不変である。

PRD 035（`docs/memo/035_approx_actor_priority_per_10prd.md`）の近似Actor PER初期優先度は、行動選択で生成済みのonline出力から`actor_q_sa`（実行行動の平均Q）と`actor_state_value`（`max_a Q_online(s,a)`）だけを持ち出し、Learnerと同じ割引率・TBO変換・`per_eps`で近似TD誤差を作る。Learnerのtarget構成のうち次の2点は意図的に再現しない。理由が異なるため分けて記録する。

**Learner側のtarget-network評価**は再現しない。Double DQN有効時は`Q_target(s', argmax_a Q_online)`、無効時もtarget networkによる行動選択・評価を行うため、どちらもActor側での再現にはtarget networkの保持と追加forwardが必要になる。これは「優先度のための追加forwardなし」という本機能の前提に反する。また行動時点とLearnerサンプル時点ではtarget networkのスナップショットが別物のため、コストを払っても厳密再現は原理的に不可能である。

**UQE/楽観的target選択**は事情が異なる。quantile出力（`q_dist`）は行動時に手元にあるため、再現してもforwardは増えない。見送る理由は次の3点である。(1) UQE版状態価値という追加のヒントtensorが必要になり、ヒントが3値で収まらなくなる。(2) UQEのtauは減衰スケジュールで時間変化し、ThompsonSamplingのtauは呼び出しごとに再抽選されるため、どちらも行動時とLearnerサンプル時で一致しない。(3) 初期優先度の目的は絶対値の一致ではなく初回サンプリング前の順位付けであり、平均Q近似で足りると判断した。なお`greedy_only`指定でもUQEの楽観的選択基準は維持される実装（epsilonだけをゼロ化し、選択基準は維持）なので、UQE構成では近似Actor初期優先度とLearner優先度に系統差が出ることを本ADRで明示的に許容する。

既存の`DQNActionInfo`は補助診断用auxに全行動Qを一時的に保持するが、ReplayBufferへ永続化しない。本機能は全行動Qを追加保持せず、`WithAction`で実行行動が差し替えられた場合に限り、既存auxから差し替え後の`actor_q_sa`をgatherし直す。これにより追加forwardや不正なQヒントの流用を避ける。

## Consequences

- Actorヒントは`actor_q_sa`／`actor_state_value`／有効性マスクの3値に固定される。UQE版状態価値は追加しない。
- `WithAction`は既存の全行動Qから`actor_q_sa`を再計算し、`actor_state_value`を維持する。ヒントがあるのに全行動Qが欠ける状態は契約違反とする。
- UQE構成（UQE/ThompsonSampling policy）では、Actor近似優先度とLearner優先度の順位相関が非UQE構成より低く出得る。PRD 035の診断メトリクス（source別比率、Actor/Learner順位相関）で監視する。
- 再訪条件: UQE構成の実運用で近似Actor初期優先度とLearner優先度の順位相関が明確に弱く、かつ`actor_approx`の効果が非UQE構成より劣化する実測が得られた場合、UQE版状態価値（行動時tau固定のtail-mean等）の追加を検討する。その場合もforward追加なしの前提は維持する。
- Action Masking導入時は`max_a Q_online`が無効行動のQを拾い得るため、state value定義の補正を別途検討する（PRD 035の対象外リスト参照）。
