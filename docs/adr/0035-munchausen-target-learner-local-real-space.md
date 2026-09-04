# Munchausen ターゲットは DQN Learner の target 計算に局所化し、実空間で計算し、log-policy 源を mode で持つ

Munchausen RL（Vieillard et al. 2020）は Bellman ターゲットの報酬側へ実行行動の scaled log-policy `α·[τ·ln π(a_t|s_t)]_{l0}^0` を加え、次状態の bootstrap を argmax から softmax 混合 `Σ_a π(a|s')(Q'(s',a) − τ·ln π(a|s'))` へ置き換える拡張である。anet-lab では BTR 差分キャンペーンで γ 0.997 だけを採り、その補償器である Munchausen を持たない構図が記録されていた（PRD 067）。DQN 系の Bellman ターゲットは `TDLearner` / `QRLearner` / `IQNLearner` がそれぞれ組み立て、TBO 有効時は h 空間で扱うため、どの Learner が・どの Q空間で・どの network から π を作るかを決める必要がある。

**Munchausen ターゲットは DQN Learner の target 計算に局所化する**ことを決定する。Munchausen項と soft価値ブートストラップは 3 Learner が共有する名前付き namespace のヘルパで計算し、ReplayBuffer の target return（N-step 割引和）、Actor の行動選択、loss 関数、PER 優先度の計算経路は変更しない。bonus は先頭遷移にだけ加え、終端でも加える。soft価値側だけを終端でマスクし、clip `[l0, 0]` は bonus 側にだけ適用する。分位点表現では π を分位点平均から作り、分位点ごとに同じ π で混合する。

**すべて実空間Q値で計算する**。TBO 有効時は全行動の Q（分位点表現では分位点ごとの Z）へ `h⁻¹` を掛けてから π・ln π・soft価値・bonus を作り、最後に `h` を掛ける。分位点表現の実空間期待値は分位点ごとに `h⁻¹` を掛けた後の平均であり、分位点平均へ `h⁻¹` を掛けた値ではない。方策温度 τ は報酬スケール前提の値（論文 / BTR: 0.03）であり、h 空間の softmax では意味が変わるためである。

**bonus の ln π を出す network は `learner.munchausen.log_policy_source = target | online` の mode で持ち、既定を `target` とする**。`target` は論文と Dopamine 公式 M-IQN に一致し、obs を target network へ追加で通す。追加 forward のコストは obs と next_obs を batch 連結した 1 回の target forward（2B）にまとめて抑える。`online` は BTR の IQN 経路（`self.net.qvals(states)`）に一致し、既に計算済みの online 出力を detach 再利用するため追加 forward も追加 RNG 消費もない。

**soft価値ブートストラップは行動選択を伴わない**ため、enabled 時は `target_policy` による argmax 選択（Double DQN の online / target 切替、`use_optimistic_target` の UQE 選択）を呼ばない。`use_double_dqn=true` または `use_optimistic_target=true` との併用は契約違反ではなく「効果のない設定」なので、構築時に 1 回だけ WARN する。`@munchausen` プロファイル自身が `learner.use_double_dqn = false` を書き、設定 dump に実態を残す。

設定は `learner.munchausen.{enabled, log_policy_source, alpha, entropy_tau, clip_value_min}` とし、enabled に関わらず常時検証する（`alpha` ∈ [0,1]、`entropy_tau` > 0、`clip_value_min` ≤ 0、いずれも finite、source は 2 値）。キー名は既存実装の多数派（`alpha` / `entropy_tau`）と Dopamine 公式（`clip_value_min`）に合わせ、`tau` 単独は `uqe_tau_*` / `tau_rule` / `soft_update_tau` / `grad_clip_tau` との多義を避けるため使わない。

## Considered Options

- **online 単一方式（BTR 準拠）**: 追加 forward ゼロで最も安いが、論文の定義（前反復の方策 π_θ̄ への暗黙 KL 正則化）から外れる。mode の 1 値として残し、単一方式にはしない。
- **target 単一方式（論文準拠）**: 定義に忠実だが Learner 律速の Atari RR1 で追加コストを避ける手段がなくなる。mode の既定値として採用し、単一方式にはしない。
- **h 空間で softmax**: 実装は最小だが τ の意味が TBO の有無で変わり、論文値を流用できないため棄却。
- **`use_tbo` との併用 fail-fast**: Atari は TBO OFF だが LunarLander / DropMerge は TBO ON で運用中であり、DQNBased 共通の要件に反するため棄却。
- **`use_double_dqn` 併用の fail-fast**: チェーン合成（A1 が後段で true を供給）で起動不能になりやすく、効果なしの設定は AGENTS.md の WARN 区分に当たるため棄却。
- **Actor 側の hint を target network で計算**: ADR 0010 の「優先度のための追加 forward なし」に反するため棄却（hint の扱いは ADR 0036）。
- **soft価値を `τ·logsumexp` で実装**: scalar では同値だが分位点混合と形が揃わないため実装は明示混合とし、`τ·logsumexp` はテストの oracle に使う。

## Consequences

- `enabled=false`（既定）では計算経路・RNG 消費とも現行と同一であり、既存 Run の再現性を変えない。ON 経路は別分岐として追加し、OFF 経路のコードは触らない。
- `enabled=true` では argmax 選択が走らないため、IQN の `target_policy.tau_rule` を含めて target 側の RNG 消費が変わる。`target` source の IQN は taus を 2B×M で 1 回生成する。同 seed 再現性は決定論 backend の下で維持する。
- PER の Learner 優先度は Munchausen 込みの target に自動で追従する。低確率行動には bonus（∈ [α·l0, 0]）が乗るため優先度の分布が変わり得るが、論文 / BTR と同じ振る舞いである。
- Learner の GPU compute は `target` source で target forward 分だけ増える。実測は PRD 067 受入 4 で記録し、性能評価の腕として `online` を並べる。
- softmax / logsumexp は AMP 領域でも fp32 で計算する（IQN 診断と同じ扱い）。
- `action_mask` を持つ env では softmax が非合法行動を含む。Learner の argmax も現行 mask を見ないため相対的な劣化はないが、mask 対応時には soft価値も同時に直す。
- Rainbow は対象外。`LearnerConfig` の既定 OFF に乗り、`RainbowAgentConfig` は `munchausen.*` を読まない（ADR 0001 の TBO と同じ整理）。
- 診断は `36_agent_munchausen` 群（bonus 平均、clip 率、次状態方策のエントロピー、soft価値と max Q の差）を IQN 診断と同じ固定 index pack で運ぶ。エントロピー → 0 は soft価値が max bootstrap へ退化していることを表す。
- 詳細契約と受入条件は `docs/memo/067_MunchausenRL_10prd.md` に置く。用語は `CONTEXT.md`「Munchausen項」「soft価値ブートストラップ」「方策温度」。
