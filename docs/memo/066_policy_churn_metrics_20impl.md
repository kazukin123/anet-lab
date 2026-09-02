# PRD066 Policy churn メトリクス実装メモ

## 概要

- [PRD066](./066_policy_churn_metrics_10prd.md) と [ADR0033](../adr/0033-policy-churn-fixed-probe-and-target-lag.md) を正本とし、DefaultDQN の 1 learner update 前後の online expected Q / greedy action 変化と、target update 後の online / target 差を `35_agent_churn` の 7 scalar として公開する。
- churn 専用 probe は ReplayBuffer から一様・非復元で 1024 件取得し、online-before / online-after / target-after で同じ Observation と、IQN では同じ fixed midpoint taus 32 本を共有する。
- probe RNG は `DefaultDQNAgent` 所有 Resource とする。`ReplayBuffer::SampleUniqueUniform` は caller-owned `RandomGenerator&` を受ける契約へクリーンブレークし、通常 sample と plasticity probe の乱数系列から churn を分離する。
- 購読された source key と cadence に必要な sample、forward、集計だけを行い、購読ゼロでは churn payload を含む全処理を不活性にする。初回公開は DefaultDQN のみとし、Rainbow / ImageCls / NoisyNet は対象外とする。
- `CONTEXT.md`、PRD、ADR0033 は既に用語と決定を固定しているため、実装と矛盾が見つからない限り変更しない。現行契約が変わる DQN / ReplayBuffer 設計ページは実装と同じ変更で同期する。

## 主な変更

### ReplayBuffer と RNG 所有権

- 公開 seam を `SampleUniqueUniform(ExperienceSamples& out_samples, int64_t batch_size, RandomGenerator& random) const` へ置換する。旧 overload、ReplayBuffer 所有の probe RNG、互換分岐は残さない。
- `DefaultReplayBuffer` は呼び出し時の `RandomGenerator` から CPU generator を取得し、sampleable index を一様・非復元で抽選する。件数不足時は `false` を返して出力と RNG を不変にし、全件要求時も RNG を消費しない。priority、`MarkSampledOnce`、eviction 統計、通常 sampler には触れない。
- `PrefetchingReplayBuffer` は既存どおり受理済み Push と in-flight prefetch を FIFO 順で settle してから、同じ caller RNG を inner へ透過する。通常 prefetched batch を消費・並べ替えない。
- `DefaultDQNAgent` は自身の seed から `"plasticity_probe"` と `"policy_churn_probe"` の named seed を作り、2 個の `RandomGenerator` Resource を所有する。inner Learner は lifetime を所有せず参照だけを保持し、plasticity / churn の各 sample 呼出しへ対応する RNG を渡す。
- `rl.hpp` / `replay_buffer.hpp` / `replay_buffer_impl.*` と、`trainer_test.cpp`、`replay_buffer_test.cpp`、`dqn_based_agent_test.cpp` の全実装・test double を同一変更で新契約へ移行する。

### Config と購読要求

- `LearnerConfig` に `policy_churn.probe.batch_size = 1024` と `policy_churn.iqn.num_taus = 32` を追加する。`DefaultDQNAgentConfig` で次を読み、各値 `< 1` をキー・指定値・期待範囲を含む `ANET_SYSTEM_ERROR` で fail-fast する。

```text
DefaultDQNAgent.learner.policy_churn.probe.batch_size
DefaultDQNAgent.learner.policy_churn.iqn.num_taus
```

- DefaultDQN の解決済み `quantile_mode` を Learner の内部 mode へ渡し、production code の型判定や `dynamic_cast` を使わず、TD / QR は通常の `q`、IQN は fixed taus を注入した `q` を expected Q として得る。Rainbow は共通 Learner を使うが churn 購読を公開しない。
- `Learner::ConfigureScalarMetricSubscriptions()` は train-scope `LEARN`、target=`UPDATE_RESULT`、7 個の `policy_churn_*` source key だけを解釈する。各 key の `IntervalGate` を保持し、連続する `learn_step` ごとに request を評価して、異なる custom interval でも metrics 定義と同じ bucket crossing で発火させる。
- request を online 群（`01`〜`04`）、target 群（`11` / `12`）、age（`13`）へ畳み、同一 update 内で probe、online-after、fixed taus を共有する。age-only request では probe も network forward も行わない。
- hard target かつ `hard_update_interval / gcd(hard_update_interval, metrics_interval) == 1` となる churn 購読を `DefaultDQNAgent::ConfigureScalarMetricSubscriptions()` で一意 interval ごとに集約し、固定 `target_sync_age=0`、metrics interval、hard update interval を含む英語 WARN を起動時 1 回だけ出す。位相数 2 以上と soft update は WARN しない。

### 測定シーケンスと payload

- update ごとの `PolicyChurnState` を inner `dqn::Learner` に置く。State は request、probe Observation、IQN fixed taus、online-before / online-after Q を保持し、次 update の開始時に必ず初期化する。
- Q 由来 request が発火したときだけ、通常 learner minibatch とは独立に churn probe を 1 回取得し、device 転送、既存 Observation 正規化を行う。完全な `probe.batch_size` を取得できない場合は sample size を縮小せず、Q 由来 6 key を requested-but-unavailable として NaN にする。
- `Learner::Optimize()` の全 optimizer 経路で、backward と grad clip の完了後・実際の `optimizer_->step()` / `GradScaler::Step()` 直前に online-before、step 完了直後に online-after を捕捉する。処理は `NoGradGuard`、eval mode、`Autocast(device, false, kFloat32)` を明示し、既存 AMP/BF16 scope から独立させる。optimizer skip または学習率 0 なら同一 Q を得てゼロ点になる。
- target update は現行どおり `UpdateFromSamples()` 後に行い、その直後に target-after を捕捉する。internal `UpdateFromSamples()` の戻り値を mutable な具象 `dqn::BatchUpdateResult` に狭め、production `dynamic_cast` なしで target update 後に churn payload を確定してから public `BatchUpdateResultList` へ格納する。
- IQN は `(i + 0.5) / num_taus` の Tensor を 1 回だけ構築し、online-before / online-after / target-after の入力 shallow copyへ同じ Tensor を注入する。TD / QR は同じ normalized Observation をそのまま使う。TBO の逆変換は行わず network 出力空間で差分を取る。
- Q tensor から次を device 上で計算し、購読された値と NaN を固定長 float32 scalar pack 1 本へまとめて `BatchUpdateResult` に保持する。`GetScalar()` は 7 key を常に既知として認識し、pack/該当値未成立時は NaN、未知 key のみ `std::nullopt` を返す。
  - action churn: `mean(argmax(before) != argmax(after))`
  - online abs delta: `mean(abs(after - before))`
  - online signed max/min: `max/min(mean_state(after - before))`
  - target disagreement: `mean(argmax(online_after) != argmax(target_after))`
  - target abs delta: `mean(abs(online_after - target_after))`
  - target sync age: hard update は target update 後の `learn_step % hard_update_interval`、soft update は NaN
- sample、各 forward、集計には安定した `Learner::PolicyChurn.*` 名の `ANET_PROFILE_SCOPE` 系計測を置く。単純な key lookup には計測を追加しない。

### Baseline metrics と設計文書

- `apps/runner/config/agent.txt` の DefaultDQN baseline に `policy_churn.probe.batch_size=1024` と `policy_churn.iqn.num_taus=32` を追加する。
- `apps/runner/config/metrics_scalar.txt` の baseline に PRD の 7 行をすべて既定 ON、`@learn $learn_step $update_result interval:503` で追加する。tag / source key は PRD の表と完全一致させ、EMA 専用行や alias は追加しない。
- `docs/design/150_replay_buffer.jp.md` を caller-owned RNG 契約と plasticity/churn の独立系列へ更新する。`docs/design/200_dqn_agents.jp.md` を optimizer 前後、target update 後、FP32 fixed probe、payload、Resource/State、公開範囲へ同期する。必要なら `docs/design/110_agents_and_learning.jp.md` の ownership 記述だけを最小更新する。
- 過去の Run artifact、done PRD / impl、既存実験記録は変更しない。plasticity probe の新旧抽選系列は意図的な基準断絶とし、互換コードや旧 checksum tripwire は追加しない。

## テスト

- Public interface / surface:
  - `ReplayBuffer::SampleUniqueUniform(..., RandomGenerator&)`
  - `DefaultDQNAgentConfig` の 2 設定と fail-fast
  - `Agent::ConfigureScalarMetricSubscriptions()` → `Learner::UpdateFromBatch()` → `BatchUpdateResult::GetScalar()`
  - `ObserverFactory` が解決する baseline tag / source key / event / target / interval
- 優先 behavior と TDD 順序:
  1. tracer bullet: 合成 2-action network、固定 probe、1 回の決定論的 update、`01_action_churn_ratio` 購読を通し、`UpdateFromBatch()` の結果から既知 action flip 比率を取得する 1 テストを RED にする。新 Replay API、caller RNG、request、optimizer 前後 capture、payload の最小経路だけを実装して GREEN にする。
  2. 同じ合成更新で `02`〜`04` の既知 literal、学習率 0 / optimizer skip の厳密な 0 を 1 behavior ずつ追加する。GREEN 後だけ重複する集計を整理する。
  3. 外側 BF16 autocast 中でも churn forward が autocast disabled / FP32 であり、近接 Q の argmax と差分を保持するテストを追加する。TD、QR、IQN fixed midpoint taus 共有を public result から検証する。
  4. hard target の非 sync 差、sync 後の厳密な 0、modulo age、soft target の有限 disagreement / Q 差と age NaN を順に追加する。
  5. 購読ゼロ、online-only、target-only、age-only、複合購読、probe 不足を RecordingReplayBuffer と forward counter で検証し、必要最小限の sample / forward だけが走ること、既知 NaN と未知 `nullopt` を確認する。
  6. ReplayBuffer seam で caller seed 再現、plasticity/churn RNG の独立、件数不足・全件時の RNG 不変、通常 sample/PER metadata 不変、Prefetching settle と通常 prefetch 列不変を検証する。
  7. config の `<1` fail-fast、baseline 7 定義、interval 503、hard interval と同位相の WARN 1 回、位相数 2 以上の無警告を追加する。
  8. 同一ビルド・同 seed の短い DefaultDQN Run を churn ON/OFF で比較し、学習 parameter、通常 Replay sample、plasticity probe、churn 以外の既存 metrics checksum が一致することを確認する。Release の代表 Run は ON/OFF 各 3 回を交互順で測り、warmup 後 throughput 中央値差が ±3% 以内であることを確認する。
- test は private helper や内部 call count 自体を仕様化せず、上記 seam の observable result と、非干渉契約を確認するための boundary counter のみに限定する。horizontal slicing は行わない。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test --parallel 1'
core\anet-core\bin\Debug\anet-core-test.exe "[dqn][policy_churn]"
core\anet-core\bin\Debug\anet-core-test.exe "[replay_buffer][probe]"
core\anet-core\bin\Debug\anet-core-test.exe "[observer_factory][metrics_defs]"
core\anet-core\bin\Debug\anet-core-test.exe
```

- focused RED/GREEN では対象 test case/tag だけを実行し、GREEN 後に関連 tag、最後に全 core test へ広げる。
- ON/OFF 非干渉と Release throughput は同一実装・同一条件で別途 Run を採取する。性能差が ±3% を超えた場合は baseline 既定 ON を自動変更せず、PRD の再審議へ戻す。

## レビュー追補（2026-09-01）

- 受入2は単なるautocast状態の観測ではなく、FP32で`[1.002, 1.001] -> [1.002, 1.003]`とargmaxが反転する一方、BF16では前後とも`[1.0, 1.0]`へ丸まる近接Qの公開結果で判定する。`action_ratio=1`、`abs_mean=0.001`、`signed_max=0.002`、`signed_min=0`と、churn forwardだけautocast無効であることを同じテストで確認する。
- 受入7は検証層を分ける。caller-owned RNGによる通常Replay sampleとplasticity/churn系列の独立、および購読なしでsample 0となる構造契約は単体テストで判定する。学習parameter、plasticity probe、churn以外の既存metrics checksum一致は、移行後の同一実行体・同一seedによるchurn ON/OFF短Runの統合アッセイで判定する。
- 同一の新ビルド内で比較するchurn ON/OFF Runでは`34_agent_plasticity/4x_probe_*`も一致対象に含める。一方、ReplayBuffer所有RNGからAgent所有`plasticity_probe` RNGへ移行した前後の新旧ビルド比較では抽選系列が意図的に断絶するため、`4x_probe_*`を一致対象に含めず、受入7の代替にも使わない。
- 受入3のIQN経路は、実`DefaultDQNAgent`の`CosineEmbedding -> TauProj -> fusion -> IQNHead`へstride-0のfixed midpoint tausを渡し、online/targetのQ由来6指標が有限値になる単体テストで判定する。加えて、実Runner設定の配線を次のRelease Runで確認する。

```powershell
Push-Location apps\runner
cmd /s /c 'bin\Release\AnetRLRunner.exe --workspace atari-2nd "run.$=run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@pl_check" "backend.$=backend.@non-deterministic" E1.game=breakout A1.learner.update_warmup_steps=8192 app.run_name=run_{t}_prd066_iqn_wiring'
Pop-Location
```

- 通常の`run.@pl_check`は100k expで終了する一方、選択中のAtari IQN構成はwarmup 200kである。churn forwardを実際に発火させるため、この配線Runだけ`A1.learner.update_warmup_steps=8192`を明示する。
- 合格条件は、shape、stride、`.view()`、taus bind errorなしで100k expを完走し、解決済みconfigが`quantile_mode=iqn`、probe batch 1024、churn taus 32であること、`inspect_run.py tags <run>`でQ由来6tagが`status=ok`かつ`count>0`となることである。現行はsoft targetなので`target_sync_age`は既知NaNとして`status=ok`、`count=0`でよい。
- 配線Runを実行できない場合、単体テストだけでRunner側の受入を完了扱いにしない。実行したRun path、コマンド、tag結果を本メモの検証結果へ追記する。

## 前提

- `vars_.learn_step` は increment 前の現在 update 番号であり、現行 `NetworkModel::UpdateTarget(step)` と同じ値を measurement cadence と `target_sync_age` に使う。hard sync step の age は 0 になる。
- Network は通常 eval 状態で、学習 forward だけ `TrainingModeGuard(..., true)` を使う。churn forward は明示的に eval + NoGrad + autocast disabled とし、学習 graph や BN/dropout stateを変更しない。
- DefaultDQN だけが 2 RNG Resource、config、購読、baseline metrics を公開する。共通 `dqn_based_agent` 内部に実装しても Rainbow から source key を公開・有効化しない。
- PRD/ADRで確定済みのため新 ADR と `CONTEXT.md` 用語追加は不要とする。実装中に public interface 追加や挙動変更が必要になった場合は、このメモまたはユーザー判断へ戻る。
- 既存の未コミット変更を保持し、Git staging、commit、pushは行わない。

## 簡素化監査

- Keep: 7 scalar、probe 1024、IQN fixed midpoint 32、caller-owned 2 RNG、cadence 503、縮退 WARN、DefaultDQN 限定公開。いずれも PRD の観測可能性・比較可能性・非干渉要件へ直接対応する。
- Shrink: Q 統計は固定長 scalar pack 1 本、measurement state は inner Learner 1 箇所、RNG は既存 `RandomGenerator` を再利用する。新しい汎用 metrics framework、sampler hierarchy、noise snapshot APIは作らない。
- Defer behind gate: Rainbow / ImageCls / NoisyNet、action transition 表、histogram、介入、科学的 Run 比較。
- Cut: plasticity batch/RNG共有、旧 `SampleUniqueUniform` overload、暗黙 batch 縮小、coprime fail-fast、warmup NaN点、EMA専用 key。
- Phase independence: Replay API移行、churn tracer bullet、target/diagn断、baseline/docs の順に各 GREEN を保つ。ただし public API のクリーンブレークは同一 commit 内で全現用呼出し元を移行し、途中状態を成果物にしない。
- Success measurability: 11 個の PRD受入条件を deterministic test、ON/OFF checksum、Release throughput 中央値で判定できる。通常学習で churn が正値になることや disagreement の単調性は成功条件にしない。

## 実装・検証結果（2026-09-01 レビュー追補）

- 近接QのBF16受入テストを追加した。FP32のargmax反転と`action_ratio=1`、`abs_mean=0.001`、`signed_max=0.002`、`signed_min=0`を公開`UpdateFromBatch()`結果で確認し、3 forward中、学習forward 1回だけautocast有効、churn 2回は無効であることを確認した。
- 実`DefaultDQNAgent` IQN learnerテストをQ由来6指標の購読へ拡張し、churn taus 5本のstride-0 expandをonline-before / online-after / target-afterの実IQN headへ通して、6値すべてfiniteとなることを確認した。`[dqn][iqn][learner]`は`383 assertions`で成功した。
- `[dqn][policy_churn]`は`8 test cases / 48 assertions`で成功した。最終Debug全core testは`530 cases / 5400 assertions`（`528 passed / 2 failed as expected`、assertionは`5398 passed / 2 failed as expected`）で完走した。
- Release `AnetRLRunner`のビルドに成功し、次のRunをexit code 0で100k expまで完走した。shape、stride、`.view()`、taus bind、System error、Exceptionのログは無かった。
  - Run: `apps/runner/workspaces/atari-2nd/runs/run_20260901-051901_prd066_iqn_wiring`
  - 解決済みconfig: `DefaultDQNAgent.quantile_mode=iqn`、`policy_churn.probe.batch_size=1024`、`policy_churn.iqn.num_taus=32`、`update_warmup_steps=8192`、`app.batchrun.exp_exit_step=100000`
  - `inspect_run.py tags`: Q由来6tagはすべて`status=ok / count=1 / learn_step=0`。soft targetの`13_target_sync_age`は`status=ok / count=0`相当（stepなし）だった。
- 受入7の同一新ビルドchurn ON/OFF checksum統合アッセイと、受入11のRelease throughput各3回比較は、本レビュー追補では実行していない。単体テストや上記IQN配線Runを代替合格にせず、別途Run採取が必要である。
