# Actor Qヒントを 3 列へ拡張し、近似Actor初期優先度を Munchausen ターゲットと同型にする

ADR 0010 / ADR 0012 の Actor Qヒントは `[Q(s,a), max_a Q(s,·)]` の 2 列で、`DqnInitialPriorityEstimator` は `target_return + discount·h⁻¹(max Q)` → `h` の Bellman ターゲットで近似 TD 誤差を作る。ADR 0035 で Munchausen ターゲットが `target_return + bonus + discount·V_soft` へ変わると、推定器はそのままでは Munchausen項（∈ [α·l0, 0]、clip 報酬と同オーダー）と soft価値を知らず、`per_initial_priority_mode = actor_approx` の初期優先度が Learner 優先度と系統的にずれる。Ape-X 型の actor 側初期優先度と Munchausen を併用した公開実装は見当たらない（Dopamine 公式 / BTR は max または uniform、Acme は定数）ため、整合は自前で決める必要がある。

**Actor Qヒントを `[q_sa, state_value, munchausen_term]` の 3 列へ拡張する**ことを決定する。`state_value` は Munchausen OFF なら従来どおり `max_a Q`、ON なら soft価値 `τ·logsumexp(Q_real/τ)` を Q空間（TBO なら `h` を掛けた値）で格納し、推定器の `h⁻¹` 経路を無改修で通す。`munchausen_term` は OFF なら 0、ON なら `α·clip(τ·ln π(a|s), l0, 0)`（実空間）とする。どちらも Actor が行動推論で既に持つ全行動 Q から logsumexp 1 回で計算し、追加 forward は行わない（ADR 0010 の前提を維持）。

推定器は `target = target_return + start.munchausen_term; if (!terminal) target += discount·h⁻¹(boot.state_value); if (use_tbo) target = h(target)` とし、Learner の Munchausen ターゲットと同じ形になる。`ValidateHint` は 3 列すべての finite を要求する。`DQNActionInfo::WithAction` は行動差し替え時に `q_sa` と `munchausen_term` を再 gather し、`state_value` を維持する。そのため Actor は per-action の Munchausen項 `[B,A]` を aux（`munchausen_term_all`）に一時保持し、hint がある `WithAction` ではその欠落を契約違反とする。

Actor が使う network は Train Actor snapshot（online 系）であり、Learner の `log_policy_source = target` でも hint 側は online 近似になる。IQN+UQE では `q_values` が risk-biased action score（ADR 0019）なので π もその近似になる。いずれも初期優先度の目的が初回サンプリング前の順位付けであることから、ADR 0010 が許容した系統差の延長として受け入れる。

## Considered Options

- **文書化のみ**: コード変更ゼロだが、actor_approx 併用時の初期優先度が Munchausen項の分だけ系統的にずれ、PRD 035 の計器（Actor/Learner 順位相関）が読めなくなるため棄却。
- **併用時に WARN のみ**: ずれを知らせるだけで解消しないため棄却。
- **別 PRD へ defer**: 影響は Actor / hint codec / 推定器に閉じ、RB 共通層は hint 幅を動的に扱うため（`payload.size(1)`、`SmallVector<float, 4>`）、PRD 067 内で一括して契約を揃える方が二度手間にならないと判断した。
- **hint を target network で計算**: ADR 0010 の「追加 forward なし」に反するため棄却。
- **ON 時だけ 3 列にする可変幅**: 推定器の schema 検証が mode 依存になり、`WithAction` と pack の契約が二重になるため棄却。常に 3 列とし OFF は 0 を入れる。

## Consequences

- ADR 0010 の「Actorヒントは 3 値固定（`actor_q_sa` / `actor_state_value` / 有効性マスク）」と ADR 0012 の「DQN payload は `K = 2`」は本 ADR で `K = 3` へ改訂する。ADR 0012 の carrier（単一 `float32[B,K]`）、completer、推定器の責務分離、ADR 0010 の平均Q近似・追加 forward 禁止・UQE 再現見送りは不変である。
- `kActorQHintColumnCount` は 3 になり、旧 2 列 payload は schema 違反として fail-fast する。RB 共通層と `InitialPriorityCompleter` は変更しない。
- Munchausen OFF では `state_value = max Q`、`munchausen_term = 0` なので推定値は現行と同値である。
- Actor の per-step コストは `[B,A]` の logsumexp と `[B,A]` の aux tensor 1 本分増える（`emit_actor_q_hint` が立つ actor_approx 構成でのみ）。
- `CONTEXT.md`「Actor Qヒント」を 3 列の定義へ改訂する。詳細契約は `docs/memo/067_MunchausenRL_10prd.md` §8、当時の 2 列契約は `docs/memo/done/035_approx_actor_priority_per_10prd.md` に記録として残る。
- 再訪条件: actor_approx と Munchausen の併用で Actor/Learner 順位相関（`39_agent_per/01`）が非 Munchausen 構成より明確に低い実測が得られた場合、hint 側の π を Learner と揃える手段（target snapshot の保持等）を ADR 0010 の再訪条件と合わせて検討する。
