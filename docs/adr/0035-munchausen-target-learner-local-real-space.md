# MunchausenターゲットはDQN Learnerに局所化し、実空間で計算し、3つのlog-policy modeを持つ

Munchausen RL（Vieillard et al. 2020）は、Bellmanターゲットの報酬側へ実行行動のscaled log-policyを加え、次状態のbootstrapをsoft価値へ置き換える拡張である。[NeurIPS論文](https://proceedings.neurips.cc/paper_files/paper/2020/file/2c6a0bae0f071cbbf0bb3d5b11d90a82-Paper.pdf)はM-DQNを1-step、M-IQNを3-stepで評価しており、[補遺](https://papers.nips.cc/paper_files/paper/2020/file/2c6a0bae0f071cbbf0bb3d5b11d90a82-Supplemental.pdf)は分位点ごとの方策混合を定義している。anet-labでは既存のN-step target returnへ整合させる必要がある。

**Munchausenターゲットは `TDLearner` / `QRLearner` / `IQNLearner` のtarget計算へ局所化する**ことを決定する。bonusはReplayBufferが集約済みのN-step returnの先頭遷移へ一度だけ加え、終端でも残す。bootstrapだけを終端でmaskし、実際のstep数に対応する `gamma^n` を掛ける。これは論文の1-step表記をそのまま同一視したものではなく、BTR互換のN-step拡張として扱う。

**すべてのMunchausen計算はFP32実空間で行う**。TBO有効時はscalar Qまたは各分位点へ個別に `h^-1` を適用してから平均、方策、bonus、soft価値を計算し、target完成後に `h` を適用する。分位点表現では実空間分位点の平均から共通方策を作り、各分位点を同じ方策で行動方向に混合する。

**bonusのlog-policyは `learner.munchausen.log_policy_mode = target | online | online_reuse` で選び、既定を `target` とする**。

- `target`: 正規化済み `obs` と `next_obs` をbatch方向へ連結し、target networkを2Bで1回forwardする。IQNはtarget規則のM tausを2B分生成する。
- `online`: 既存のcurrent/target forward後、online networkをNoGradかつeval modeで追加forwardする。IQNはcurrent規則のN tausを新規生成する。
- `online_reuse`: 既存のtrain-mode current出力をdetachして再利用し、追加forwardも追加RNG消費も行わない。

IQNでは既存current/targetのtau生成順を維持した後に `online` 用tausを生成する。したがって `online` と `online_reuse` の差は、fresh eval forwardとそれに伴うRNG消費として観測できる。

**soft価値ブートストラップはargmax選択を行わない**。このため `munchausen.enabled=true` と `learner.use_double_dqn=true`、または `munchausen.enabled=true` と `use_optimistic_target=true` の併用は、明示設定が効果を持たない互換性のない組み合わせとして構築時に `ANET_SYSTEM_ERROR` でfail-fastする。エラーには競合する両キー、指定値、期待値 `false` を含める。Munchausenがdisabledの場合は従来機能として許可する。

設定は `learner.munchausen.{enabled, log_policy_mode, alpha, entropy_tau, clip_value_min}` とし、`enabled` に関わらず常時検証する。modeは閉じた3値、`alpha` はfiniteな `[0,1]`、`entropy_tau` はfiniteかつ `> 0`、`clip_value_min` はfiniteかつ `<= 0` とする。旧 `log_policy_source` はクリーンブレークで廃止し、aliasや互換分岐を持たない。

## Considered Options

- **`target` 単一方式**: 論文の定義に近いが、Learner律速時に追加forwardの費用を切り分けられないため、既定modeに留める。
- **`online_reuse` 単一方式**: 追加forwardなしで最も安いが、train/eval差とfresh/stale出力の差を分離できないため、選択肢の1つに留める。
- **`online` と `online_reuse` の統合**: 同じonline networkでも、eval-mode fresh forwardとtrain-mode再利用は数値経路・RNG・性能が異なるため統合しない。
- **SACを見越した共通moduleの先行抽出**: SACのActor/Critic契約が未確定で、現時点では利用箇所が1つしかないため行わない。まず既存の `dqn_based_agent.*` 機能グループ内の純粋helperとして実装し、実際の共通処理が2利用箇所になった時点で再検討する。
- **h空間でsoftmax**: 方策温度の意味がTBOの有無で変わるため棄却する。
- **`use_tbo` との併用fail-fast**: DQNBased共通機能として必要な構成を狭めるため棄却する。
- **効果のないDouble DQN / optimistic target設定をWARNして継続**: 明示要求を満たさない構成を実行してしまうため棄却する。
- **soft価値を `tau*logsumexp` だけで実装**: scalarでは同値だが分位点混合と形が揃わないため、実装は明示的な方策混合とし、同値式はtest oracleに使う。

## Consequences

- Learnerの数値経路とRNGについて、`enabled=false` の標準Atari構成は実装前と完全一致させる。Rainbowはアルゴリズム上OFFのままだが、共通transportのshape不変までは保証しない。
- `target` は2Bのtarget forward、`online` は独立したfresh online forward、`online_reuse` は追加forwardなしとなる。ProfileRangeは `forward_target`、`forward_munchausen_online`、`munchausen_target` を区別する。
- PERのLearner優先度はMunchausen込みtargetへ追従する。診断5値とEMA 2行、および固定index readbackの詳細契約はPRD 067に置く。
- throughput、`exp_step_per_sec`、ProfileRangeはmodeごとに記録するが、本ADRでは性能の合否閾値を設けない。
- `action_mask` は既知の未対応事項である。非合法行動をsoft価値へ含め得るため将来対応が必要だが、現行実装との相対的な安全性は主張しない。
- Actor初期優先度用hintはADR 0036の狭いActor側契約に従い、LearnerのmodeをActorへ伝播しない。
- 詳細な数式、config、テスト、受入条件は `docs/memo/067_MunchausenRL_10prd.md` を正本とする。
