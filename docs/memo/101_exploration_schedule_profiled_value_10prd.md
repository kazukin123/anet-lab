# 探索と学習のパラメータを ProfiledValue で時間変化させる PRD

> 2026-10-24 起票（暫定。質疑を省き、決めることを §8 に残した）。ユーザーの指示「ProfiledValue ベースで、探索系パラメータ全般のアニーリング対応を行う」。
> 2026-10-24 再グリル（改訂）。ゴールを 2 段階に広げた（段階 2 で学習側のパラメータも対象にする）。定数を bare で書く記法（ADR 0064）、概念ごとの入れ子のキー名、log ラダーの 0 端の規則、時間で変わる γ の扱い（ADR 0065）を決めた。コードとの照合で、Thompson sampling に τ の区間は無いこと、log ラダーの 0 端は今も 1e-4 で動いていることが分かり、§4-1・§8-4 の `thompson.tau_low/high` を削った。別モデル（Opus）の検討を受け、γ は「schedule のときだけ抽選時に収益を作り直す」に変え、root の `.value` 記法を無くし、ラダーの下限の規則を一般化し、step 診断の利用者と移行の写し方の規則を足した。細部は推奨で埋めた。
> 2026-10-24 改訂（同日 2 回目）。実装計画の未決監査で、PER β の適用時機（今は前回の更新の終わりに次回分を評価）が D18 の「先頭で評価」と両立しないと分かり、β だけ今の時機を保つことにした（D18、§2.2、§6.2-3a）。
> 2026-10-25 改訂。実装計画前のレビュー（差し戻し 5 件、再レビューで 1 件）に対応。G5・D14 の等価性の例外を、D8 の伸縮で中間 lane も動く範囲へ合わせた。root の `.value` は `sync_interval.value` として現用にあると分かり移行対象に加えた（§2.1、D3、D12、§6.1-2・7）。廃止 root `.value` 専用の fail-fast を削った（D3、§6.1-4、§10.1）。log ラダーの下限の規則を連続な伸縮の形に改めた（D8、§5.4・5.5、§6.1-5）。`Learner` 行の β、CHANGELOG と残存検査の範囲、D7 と §10.1 の実装の語を直した。成績比較は完了条件に含めず、比較の設計を §3 に書いた。
> 2026-10-25 実装時の改訂。基準Runと現用設定の照合で、LunarLanderも spatial・log のUQE ε末端0を使うと判明した。ユーザー判断により本体の0を維持し、DropMergeと同じ等価性の例外へ加えた。両環境の等価性検査は末端1e-4の検査用設定で行う（G5、D14、§6.1-1）。
> 起点: qbert の第 8 面の壁で BTR との差を詰める比較（2026-10-23〜24）。BTR の公開実装は、学習中の ε を全 env 共通に 1.0 から時定数 2M 遷移の指数で減らし、予算の半分（25M 遷移）で 0 にする（以後の探索は NoisyNet だけ）。anet の今の設定では「途中で ε を切る」も「指数で減らす」も書けず、BTR に寄せた anet の Run は ε 0.01 を最後まで残した。自分たちの環境の BTR は 40M あたりから第 8 面を越え始めたが、BTR に寄せた anet は 46M まで越えなかった。差は 25M 以降のデータの質（qbert では 1 回のランダムな行動で台から落ちうる）にある、という仮説を確かめるにも、ε の形を自由に書ける必要がある。
> 関連: [anet/schedule.hpp](../../core/anet-core/include/anet/schedule.hpp)（ProfiledValue）、[099_atari_resize_mode_10prd.md](099_atari_resize_mode_10prd.md)（同じ比較で見つけた前処理の差）、[098_temporl_skip_policy_10prd.md](098_temporl_skip_policy_10prd.md)（skip の探索も同じ ActionPolicy の設定を使う。Replay の step 報酬の保存は ADR 0063）、[084_noisynet_10prd.md](084_noisynet_10prd.md)（NoisyNet）、[091_checkpoint_load_optimizer_options_10prd.md](091_checkpoint_load_optimizer_options_10prd.md)（optimizer options の引き継ぎ）、[094_evalpanel_step_history_10prd.md](094_evalpanel_step_history_10prd.md)（step 診断）、[reports/btr_code_reading_2026-08-31.md](../../reports/btr_code_reading_2026-08-31.md)、[ADR 0064](../adr/0064-profiledvalue-bare-scalar-reads-as-value-field.md)、[ADR 0065](../adr/0065-scheduled-gamma-recomputes-return-at-sample.md)。

## 1. 目的と問題

行動を選ぶときの探索のパラメータ（ε、UQE の ε と τ、lane ごとのラダー）と、学習側のパラメータ（PER β、学習率、weight_decay、target の soft update の τ、Munchausen の α と entropy_tau、割引率 γ）を、学習の進みに合わせて変化させる形を、既存の `ProfiledValue`（一定・直線・cosine・cosine の繰り返し・段の組み合わせ）と、新しく足す指数で自由に書けるようにする。

今の問題は 4 つある。

1. **探索で書ける形が直線 1 本だけ。** ε は「start から end へ decay_steps で直線に下げ、あとは end のまま」しか書けない。途中で切る、しばらく保ってから下げる、指数で下げる、が書けない。
2. **lane ごとのラダーは時間で変わらない。** spatial exploration を有効にすると、start と end は「lane の両端の値」の意味に変わり、学習の最後まで同じラダーのままになる。ラダーを途中で狭める・消す、が書けない。
3. **同じキーが 2 つの意味を持つ。** `eps_start`・`eps_end` は、spatial の有無で「時間の両端」にも「lane の両端」にもなる。UQE の `uqe_eps_*`・`uqe_tau_*` も同じ。設定を読んだだけでは、どちらの意味か分からない。
4. **学習側に時間変化の器が無い。** 時間で変わるのは PER β（直線）だけで、学習率・weight_decay・soft update の τ・Munchausen の α と entropy_tau・γ は定数しか書けない。学習率を途中で下げる、β を予算の半分で収束させる、を Run の中で書けない。

## 2. 現状（2026-10-24 にコードで確認したこと）

### 2.1 探索側

- 探索の設定は `ActionPolicyConfig`（[agent.hpp](../../core/anet-core/include/anet/agent.hpp) 100〜139 行）の素の値で持つ。
  - ε-greedy: `eps_start`・`eps_end`・`eps_decay_steps`・`use_spatial_exploration`・`spatial_scale_type`（log / linear）
  - UQE・Thompson sampling: `uqe_tau_start`・`uqe_tau_end`・`uqe_tau_decay_steps`・`uqe_use_tail_mean`（既定 false）・`uqe_eps_start`・`uqe_eps_end`・`uqe_eps_decay_steps`
- 読み口は 2 か所。DefaultDQN の `ReadActionPolicyConfig`（[default_dqn_agent.hpp](../../core/anet-core/include/anet/default_dqn_agent.hpp) 523〜546 行）と、Rainbow の Actor カタログ（[rainbow_agent.hpp](../../core/anet-core/include/anet/rainbow_agent.hpp) 41〜43 行。ε の 3 キーだけを読む）。`target_policy` は `use_optimistic_target = true` のとき train の方策を写し、ε を 0 に強制してから `target_policy.*` の部分上書きを読み、`use_spatial_exploration` を強制 false にする（同 190〜227 行）。
- 時間の変化は `ActionPolicy::UpdateEpsilon` と `UQEActionPolicy::UpdateTau`（[dqn_based_agent.cpp](../../core/anet-core/src/dqn_based_agent.cpp) 1652〜1673 行・1779〜1790 行）の直線だけで、float で計算する。`*_decay_steps <= 0` なら早期 return で start のまま。step は `UpdateSchedule` に渡る `counts.exp_step`（1698〜1702 行、1792〜1797 行）。spatial のときは何もしない（早期 return）。Thompson sampling は ε だけ更新し τ は更新しない（2113〜2118 行）。
- **Thompson sampling に τ の区間は無い。** 非 spatial の Thompson は τ を U[0,1] から引き、IQN では全範囲を使う（2120〜2144 行）。`uqe_tau_start/end` が効くのは spatial のラダーの両端だけである（1772 行）。設計書 [200 §2.3](../design/200_dqn_agents.jp.md) も同じ（「非 spatial Thompson は [0,1]、spatial Thompson は lane ごとの下限」）。暫定 PRD が §4-1・§8-4 に置いた「Thompson の τ の区間を `thompson.tau_low/high` に分ける」案は前提が違うので、本改訂で削った。
- spatial のラダーは、構築時に `CreateSpatialLaneTensor`（1546 行。`CreateSpatialTensor` 1509〜1544 行を lane 順に反転）で 1 回だけ作る（ε-greedy 1692 行、UQE 1771〜1772 行）。**log のとき 0 以下の端は 1e-4 に clamp し、WARN を 1 回出す**（1529〜1540 行）。clamp 後の値がそのまま端の lane の値になるので、0 と書いても 0 にはならない。DropMerge の UQE ε ラダーは `uqe_eps_end = 0.0`（[DropMerge.txt](../../apps/runner/config/DropMerge.txt) 636 行）で、この clamp に依存して 0.6〜1e-4 で動いている。
- Greedy 方策は ε-greedy の設定を複製して ε を 0 にして作る（[default_dqn_agent.cpp](../../core/anet-core/src/default_dqn_agent.cpp) 278〜285 行、[rainbow_agent.cpp](../../core/anet-core/src/rainbow_agent.cpp) 104 行）。
- 指標は `GetScalar` の `epsilon`・`uqe_tau`（1675〜1680 行）。UQE と Thompson の `epsilon` は UQE の ε を返す。spatial のときは両方 NaN（1485〜1486 行）。Thompson の `uqe_tau` は構築時の `uqe_tau_start` を返し続ける（1776 行）。設定側は `$actor epsilon`（[metrics_scalar.txt](../../apps/runner/config/metrics_scalar.txt) 31 行）と `$agent target_policy.uqe_tau`（[GridMaze.txt](../../apps/runner/config/GridMaze.txt) 171 行）。
- ProfiledValue（[schedule.hpp](../../core/anet-core/include/anet/schedule.hpp)）は、`constant`・`linear`・`cosine`・`cosine_restart`・`phased` を持ち（49〜64 行）、設定の読み取り（`ConfigReader<ProfiledValueConfig<T>>`、206〜253 行）、下限と上限（`min_value`・`max_value`、76〜100 行）、lane のような添字で値を並べる `EvaluateByIndex`（282 行）まである。補間は double で行って型へ戻す（311〜316 行、327〜330 行）。`phased` は名前つきの段を `phases = a b c` の順に並べ、最後の段の後は最後の段の終わりの値を保つ。最終 type が選ぶ欄だけを検証し、選ばれていない欄（dormant）は保持する（180 行）。root の `value` 欄は設定からも読める（216 行）。現用では `sync_interval.value` が root の `.value` 記法で書かれている（[agent.txt](../../apps/runner/config/agent.txt) 356 行、[Atari.txt](../../apps/runner/config/Atari.txt) 654 行、[DropMerge.txt](../../apps/runner/config/DropMerge.txt) 209 行、[13_batch_run.bat](../../apps/13_batch_run.bat) 60 行）。テストも root の `.value` を入力と出力に使う（[config_test.cpp](../../core/anet-core/src/config_test.cpp) 483〜529 行の ImageCls `learning_rate.value`、555〜578 行の `interval.value`、[image_cls_agent_test.cpp](../../core/anet-core/src/image_cls_agent_test.cpp) 361 行）。PRD072 の golden にも `sync_interval.value` が入っている。ImageCls の `learning_rate` は設定ファイルでは段の中だけである。
- 設定の器 `ConfigData` は平らな文字列の map で（[config.hpp](../../core/anet-core/include/anet/config.hpp) 24〜133 行）、`X` と `X.type` は別のキーとして共存できる。dump の JSON も平らなキーのまま（107〜114 行）。読み取りは default prefix と override prefix の両方を見る（185〜232 行）。既定葉（`?=`）か確定葉（`=`）かは `Properties::DefaultKeys` が読み込み途中だけ持ち、`ConfigData` へは渡さない（154〜160 行）。
- ProfiledValue の既存の利用は、Actor の同期の間隔（`actor.[key].sync_interval`、[default_dqn_agent.hpp](../../core/anet-core/include/anet/default_dqn_agent.hpp) 177 行付近）と ImageCls の学習率（[ImageCls.txt](../../apps/runner/config/ImageCls.txt) 76〜120 行の phased。[image_cls_agent.cpp](../../core/anet-core/src/image_cls_agent.cpp) 405 行で `exp_step` により更新し、307〜313 行で optimizer の param group の options へ書き戻す）。
- 今の探索のキーは、設定（agent.txt・Atari.txt・DropMerge.txt・GridMaze.txt・LunarLander.txt）、bat（11・12・13）、テスト（`dqn_based_agent_test.cpp`・`dqn_munchausen_test.cpp`・`config_baseline_test.cpp` 218〜220 行・`trainer_test.cpp` 1229 行）、testdata（PRD072 の golden など）に計 106 か所超ある（2026-10-24 の grep）。
- BTR の公開実装（`Agent.py` の `EpsilonGreedy.update_eps` と `choose_action`）: ε は遷移ごとに `eps -= (eps - eps_final) / eps_steps`（＝時定数 eps_steps の指数減衰、既定 2M、下限 0.01）。学習の開始前と、予算の半分（`total_frames / 2`）までの間だけ使い、その後は NoisyNet だけで探索する。

### 2.2 学習側

- 時間で変わるのは PER β だけ。`per_beta_start/end/step` の直線（float）で、軸は `exp_step`（[dqn_based_agent.cpp](../../core/anet-core/src/dqn_based_agent.cpp) 3728〜3740 行）。適用の時機は 1 更新ずれている: 更新の抽選は前回の更新の終わりに計算した β を使い（3819 行の抽選、3895 行の `UpdatePerBeta(counts.exp_step)`）、最初の更新は構築時の `per_beta_start`（3219 行）を使う。Rainbow も同じ 3 キー（[rainbow_agent.hpp](../../core/anet-core/include/anet/rainbow_agent.hpp) 64〜66 行）。
- 学習率は `learner.alpha` の定数で、`weight_decay`・`adam_eps` とともに optimizer の options になる（`BuildOptimizerOptions`、3114〜3131 行。NoisyNet の σ 専用 group は `noisy_weight_decay`）。FusedAdamW は毎 step options から lr と weight_decay を読む（[nn_util.cpp](../../core/anet-core/src/nn_util.cpp) 314〜329 行）。`alpha` の名は Munchausen の `alpha`・PER の `per_alpha` と重なる。
- target の更新は `NetworkModel::UpdateTarget`（817〜833 行）。`hard_update_interval > 0` なら learn_step の倍数で hard copy、そうでなければ毎 learn step に `soft_update_tau` で soft copy。「τ は [0, 0.1] または 1」の検査は、spectral norm を持つ network があり、かつ soft update のときだけ行う（768〜783 行）。
- Munchausen の `alpha`・`entropy_tau` は Learner の target（364〜367 行）だけでなく、Actor の Q ヒント（2369 行の `MakeActorQHint`）と、Evaluation Trace の step 診断 `DqnAgentStepDiagnosticsProvider`（499〜555 行。構築時に写した `LearnerConfig` から Munchausen・γ・n-step を使い、`Observe` は step を受け取らない。PRD094）でも使う。設定は構築時に検証する（[default_dqn_agent.hpp](../../core/anet-core/include/anet/default_dqn_agent.hpp) 268〜289 行）。
- γ を使う場所。(a) Replay の n-step 収益は Push 時に `DefaultExperienceBuilder` が γ で逆順に足して保存する（[replay_buffer_impl.cpp](../../core/anet-core/src/replay_buffer_impl.cpp) 252〜283 行）。初期優先度の割引も Push 時（500 行、535 行）。(b) Replay は step ごとの報酬と終わり方を Push のたびに無条件に保存している（`RecordStep`、621〜627 行・1534 行。ADR 0063）。TempoRL の窓収益は抽選時にその保存値から前向きに γ で足している（1370〜1384 行）。(c) Learner の bootstrap は学習時に γ^n を掛ける（4079 行ほか）。(d) 報酬スケーラの自動 post_scale は構築時の γ で `reference_q_std × (1 − γ)` を作る（[scaler_impl.cpp](../../core/anet-core/src/scaler_impl.cpp) 748〜767 行）。構築時に実効ホライズンをログに出す（[default_dqn_agent.cpp](../../core/anet-core/src/default_dqn_agent.cpp) 57〜70 行）。(e) step 診断も自分の builder と割引に γ を使う（503 行、555 行）。(f) MuZero の Replay も同じ builder に γ を渡す（2502 行）。
- checkpoint の引き継ぎ（PRD091）は、optimizer の options（lr・weight_decay・eps・betas）を読み、設定の値と違えば WARN を出して設定の値で上書きする（3987〜4022 行）。Run の `StepCounts` と schedule は引き継がず、新しい Run は exp_step 0 から始まる（[091 PRD](091_checkpoint_load_optimizer_options_10prd.md) の非ゴール）。
- Rainbow は共通の `dqn::Learner` を使う（[rainbow_agent.hpp](../../core/anet-core/include/anet/rainbow_agent.hpp) 113 行）。学習側のキーは `RainbowAgentConfig` が自分で読む（52〜75 行）。
- 学習側の定数のキー（`learner.alpha`・`weight_decay`・`soft_update_tau`・`munchausen.*`・`gamma`）は、現用の設定 5 ファイルと bat 3 本に計 92 か所ある（`learner.alpha` 22、ほか 70。2026-10-24 の grep）。

## 3. ゴール・非ゴール・成功の測り方

### ゴール

段階 1（探索側）:

- **G1 時間の形**: ε-greedy の ε、UQE の ε と τ を、それぞれ 1 つの ProfiledValue で書ける（一定・直線・cosine・cosine の繰り返し・段の組み合わせ・指数）。
- **G2 ラダーの時間変化**: spatial のラダーの両端（最初の lane と最後の lane の値）を、それぞれ ProfiledValue で書ける。各 step の lane の値は、その時点の両端から今と同じ並べ方（log / linear）で作る。
- **G3 指数の形**: ProfiledValue に指数減衰（`exponential`）を足し、`phased` の段としても使える。BTR の ε（指数で減らして 25M で 0）を正確に書ける。
- **G4 意味の分離**: 「時間の値」と「lane の両端」を別のキーにする。両方を設定に書いておき、`spatial.enabled` の切り替えだけで、値を書き換えずにラダーあり・なしを行き来できるようにする。クリーンブレークで移る。設定・bat・golden・文書・CHANGELOG の読み替え注記を同じ変更で直し、実装は新しいキーだけを読む（旧キーの検出・案内・別名・互換の読みを置かない）。
- **G5 既定の等価性**: 既存の設定を新しいキーに置き換えたとき、学習系列が変わらない（metrics checksum の一致。比較から除く tag は §6.1-1 に名前で定める。意味が変わる探索の指標と、今回足す指標だけで、loss・TD 誤差・Q・収益などは含める）。例外は D14 の DropMerge と LunarLander の UQE ε ラダー（`last = 0` を書く本体の設定では、D8 の伸縮で最後の lane が 1e-4 から 0 になり、中間 lane も各 lane がほぼ 1e-4 だけ下がる。相対では小さい ε の lane ほど大きく、1e-4 に近い lane は半分以下になる）だけで、`last = 1e-4` と書いた検査用の設定で等価性を確かめる。

段階 2（学習側）:

- **G6 学習側の時間の形**: PER β、学習率、weight_decay、soft update の τ、Munchausen の α と entropy_tau、γ を、それぞれ 1 つの ProfiledValue で書ける。定数で書けば今と同じ学習系列になる。
- **G7 γ の整合**: 時間で変わる γ でも、学習に使う n-step 収益と bootstrap が同じ γ になる（schedule のときだけ抽選時に保存済みの step 報酬から収益を作り直す。定数のときは今の経路のまま）。

共通:

- **G8 定数の記法**: ProfiledValue の定数を `X = v` の 1 行で書ける。既存の定数の行は、改名する学習率（`learner.alpha` → `learner.learning_rate`）と、既に ProfiledValue である `sync_interval.value` の記法の置き換え（D3）を除き、書き換えずに ProfiledValue 化できる。

### 機構ごとのゴールへの寄与

| 機構 | 寄与 |
|---|---|
| ε・UQE ε・τ の ProfiledValue 化（G1） | BTR の ε の形を書く本体。成功の測り方の実験に必須 |
| `exponential` 型（G3） | BTR の時定数 2M の指数。linear や phased の段では近似にしかならない |
| 時間の値とラダー両端のキー分離（G4） | ラダーあり・なしの比較で値を書き換えずに済む。2026-10-23 の BTR 寄せ Run で 3 キーを書き直した手戻りの再発防止 |
| ラダー両端の ProfiledValue 化（G2） | 今回の実験には要らないが、「ラダーを 25M から消す」比較は 25M 以降のデータの質という同じ仮説の次の腕なので、後で要るものとして同時に作る |
| 学習側 7 値の ProfiledValue 化（G6・G7） | 学習率を途中で下げる、β を予算の半分で収束させる、を Run を分けずに書く。再グリルでゴールに加えた |
| bare の定数記法（G8） | 学習側の定数 92 か所を書き換えずに済み、スケジュールが要らない設定が 1 行で済む |

### 非ゴール

退けた案を名指しし、それが本当の制約か（クリーンブレークで移せないか）を添える。

- NoisyNet の雑音の大きさを時間で変えること。σ は学習するパラメータで、外から掛ける倍率は forward の雑音の作り方に新しい入力が要る。設定を書き換えて移せる種類ではなく、別の設計が要る（§7）。
- env 側の確率（sticky action など）。Env の設定で、Agent の探索とは別の関心。
- 探索の方式そのもの（ε-greedy・UQE・Thompson sampling）の追加や変更。方式は方策の種類の問題で、パラメータの時間変化という本 PRD の関心とは別。制約ではなく対象を絞る判断で、後から同じ器の上に足せる。
- MuZero の温度 `temp_start/end/decay_steps`。試作 Agent の探索パラメータで、本線の DQN 系と設定の器が違い、今は本線に戻す予定が無い。ProfiledValue の器自体は使えるので、本線に戻すときに同じ形で足す。制約ではなく対象を絞る判断である。
- step の軸を `learn_step` やエピソード数にすること（§7）。
- bare の値が眠る事故の検査（静的検査・実行時の fail-fast。§7）。文書に書くだけにするのは今回の裁定で、仕組みは後から足せる。

### 成功の測り方

- 本 PRD の完了は §6 の受入で判定する。成績の比較は完了条件に含めず、後続の実験で行う（D24）。
- 段階 1 の効果の確かめ方（後続の実験の設計）: BTR の ε の形（指数で減らして 25M で 0）を設定だけで書き、anet の BTR 寄せの qbert で、ε 0.01 を残した腕（`run_{t}_a5_qb_btrlike_s1` と同じ設定）と 25M で 0 にする腕の 2 腕を各 3 seed、50M で回す。見るのは 45〜50M の学習中の第 8 面突破率（29,000 点以上かつ round_clear ≥ 8）で、同設定 seed のブレ幅を超えて差があるかで判定する。改善が無くても結果を実験記録に残して比較を終える。
- 段階 2: 学習率を途中で下げる Run（DropMerge の停滞区間で 7.1e-5 へ下げた比較の続き）を、Run を分けずに 1 本の phased で書けることと、`$agent learning_rate` の系列が設定どおりに切り替わること。

## 4. 確定事項

決定ごとに理由と、依存する決定や事実を書く。矛盾する既存の ADR は無い（[ADR 0040](../adr/0040-config-selection-final-value-and-run-profile-tier.md) は例示に旧キーを含むだけで、判断は変わらない）。

### 4.1 器と記法

- **D1 時間で変わる値は ProfiledValue にする。** 探索側は `ActionPolicyConfig` の素の値を ProfiledValue の設定に置き換え、学習側は `LearnerConfig` と `NetworkModelConfig` の 7 値を置き換える。新しく足す ProfiledValue の設定の既定 type は constant にする（既定の直線 1.0→0.05 を写すと、eval などカタログ項目の bare がすべて眠るため）。理由: 既に設定読み・検証・phased・下限上限を持つ器があり、ImageCls の学習率で実績がある（§2.1）。依存: D3（記法）。
- **D2 `exponential` 型を足す。** `value(t) = end + (start − end) × exp(−t / time_constant)`。欄は `start`・`end`・`time_constant`（0 より大）。root で使うときは `steps` を読まず漸近し続ける。`phased` の段では `steps` が段の長さで、段の終わりの値は「段の終わりの step での式の値」とする（段の後ろの値を保つ `phased` の規則と整合させる）。BTR の式（遷移ごとに差の 1/eps_steps を引く）とは離散と連続の差だけがあり、減衰項に対する相対差は t = N（時定数）で 3e-7、t = 25M で 3e-6 程度、ε の値に対する絶対差は 1e-6 未満である。理由: BTR の `eps_steps` と同じ意味の欄 1 つで書け、半減期などへの換算を利用者に求めない。
- **D3 ProfiledValue の定数は bare で書く（[ADR 0064](../adr/0064-profiledvalue-bare-scalar-reads-as-value-field.md)）。** root の定数は `X = v` の 1 行で書き、これが `value` 欄になる。root の `.value` 記法は読まない（実装は root の `.value` を読まないだけで、廃止した記法の検出や案内は置かない。クリーンブレーク方針）。段の中の定数は今どおり `.phase.[name].value`。`type` の既定は constant なので、定数は 1 行で済む。`.type` が constant でない層の下に bare を書くと、その値は `value` 欄として眠る（ProfiledValue の既存の dormant の契約と同じ。§2.1）。眠る規則は本 PRD・設計書 200・ガイド 020 に書き、検査は足さない（§7）。既存の ProfiledValue の利用も同じ記法に乗る: 現用の `sync_interval.value = N` は `sync_interval = N` へ写し（設定 3 ファイルと bat 1 本、PRD072 の golden）、設定と ImageCls のテストの root `.value` の入力と出力も bare に写す（D12）。ImageCls の設定ファイルは段の中だけなので変わらない。理由: 学習側の定数の行（改名する学習率を除く約 70 か所）を書き換えずに ProfiledValue 化でき、スケジュールが要らない設定が 1 行で済む。2 つの記法を無くせば衝突の規則も要らない。設定の器は平らなキーの集まりなので `X` と `X.type` が共存でき、dump も壊れない（§2.1）。依存: D1。
- **D4 config dump の記録。** 書いた記法に関係なく、実効の type が constant なら bare の 1 行、それ以外なら `type` とその type が使う欄（既定込み）を記録する。`sync_interval` も定数なら `sync_interval = 400` の 1 行になる（今の dump は `sync_interval.value` ほか全欄）。理由: dump は Run の真実として読まれるので、baseline の bare と Run プロファイルの phased が混在する層でも記録が一意で、定数と誤読されない。PRD072 の全キー比較も記法の差で揺れない。依存: D3。

### 4.2 探索側のキーと挙動

- **D5 キー名は概念ごとの入れ子。** 時間の値は `epsilon`・`uqe.epsilon`・`uqe.tau`（`uqe_use_tail_mean` は `uqe.use_tail_mean` へ）。ラダーは `spatial.enabled`（旧 `use_spatial_exploration`）・`spatial.scale`（旧 `spatial_scale_type`。log / linear）・`spatial.epsilon.first`・`spatial.epsilon.last`・`spatial.uqe.epsilon.first/last`・`spatial.uqe.tau.first/last`（各 ProfiledValue）。理由: 既存の入れ子（`tau_rule.*`・`munchausen.*`・`full_distribution_query.*`）と同じ流儀で、`spatial.*` の下に切り替え・並べ方・両端が全部入り、設定を読むだけで関係が分かる。設定階層をフレームワークの概念と同じにする方針に合う。依存: D1。
- **D6 時間の値とラダーの両端は別のキーで、両方を既定に置く。** どちらを使うかは `spatial.enabled` だけで決まる。使わない側の値は評価せず保持だけする（値そのものの検査は両側に行う）。理由: ラダーあり・なしの比較で値を書き換えずに済む。2026-10-23 の BTR 寄せ Run では `eps_start`・`eps_end`・`eps_decay_steps` の 3 キーを書き直す必要があった（§5）。依存: D5。
- **D7 ラダーは両端の今の値から作る。** Actor は `UpdateSchedule` で時間の値と両端を `exp_step` により 1 回評価し、lane ごとの値をその時点の両端から並べる。両端が一定なら構築時の値を使い続け、今と同じ回数になる。両端が変わるときの作り直しの頻度や置き場は実装に任せる（schedule があれば実質毎 step で、同期を伴わない小さい演算が増えるだけ）。並べ方は今の log / linear。既存の設定で各 lane に割り当たる値（どの lane が小さい ε を持つか）は今と同じにする。理由: G2 を満たしつつ、既定の挙動を変えない。依存: D5、D6。
- **D8 log のラダーの下限。** log の間隔は両端を `max(端, 1e-4)` に置いて計算し、できた列を、両端が書いた値に一致するように線形に伸縮する（伸縮後の端は書いた値そのもの。clamp 後の両端が等しいときは、書いた両端の線形補間にする）。両端が 1e-4 以上なら伸縮は恒等で今と同じ値になる。これにより、0.4〜0 は 4 lane で 0.4 / 0.0251 / 0.0015 / 0 と最後の lane だけが 0、0〜0 は全 lane が 0、0.4〜5e-5 は最後の lane が書いたとおり 5e-5、0.05〜1e-4（qbert）は今と同じ値になる。端が時間で 0 へ下がる途中も値は連続で、first が 1e-4 を跨ぐところでも中間 lane（3 lane なら 5e-5）は変わらない。負の端は `min_value = 0` で設定時に fail-fast。今の WARN と clamp のコードは消す。既存の log ラダー（Atari 0.4〜0.01、qbert の 0.05〜1e-4）は両端が 1e-4 以上なので値が変わらない。理由: 0 は書けて 0 で動く（「設定に書いたことがそのまま起きる」）、WARN を正当な設定に出さない、既存の値を変えない、の 3 つを規則 1 つで満たす。1e-4 は内側の間隔を決めるコード定数で、設計書 200 に書く。依存: D7。退けた案: 「log は端 > 0 を要求して fail-fast し、0 に届くラダーは新しい並べ方 `exponential` で書く」（モードが 1 つ増える）、「下限をキーにする `spatial.log_floor`」（置換の意味が残る）、「clamp のまま WARN だけ消す」（0 と書いて 1e-4 で動く形が残る）、「下限以下になった lane だけを線形補間で置き換える」（first が 1e-4 を跨ぐ瞬間に中間 lane が約 1e-4 から 5e-5 へ飛ぶ）。
- **D9 Thompson sampling は τ の時間の値を使わない。** 非 spatial は今のとおり U[0,1]（IQN は全範囲）、spatial は `spatial.uqe.tau.*` のラダーを使う。`uqe.epsilon` は今と同じく Thompson にも効く。Thompson の `uqe_tau` の指標は spatial でも非 spatial でも NaN にする（今は構築時の値を返し続けている。lane ごとの τ は D11 の `uqe_tau_lane_min/max` で読める）。理由: §2.1 の事実（区間は存在しない）。暫定 PRD の `thompson.tau_low/high` は削った。
- **D10 Greedy と target_policy の保持。** `policy_type = Greedy` は `epsilon` を constant 0 に置き換えた ε-greedy として作る（DefaultDQN・Rainbow とも今の作りと同じ）。`use_optimistic_target = true` の `target_policy` は train の方策を写したあと `epsilon` と `uqe.epsilon` を constant 0 にし、`target_policy.*` の部分上書きを読み、`spatial.enabled` は今と同じく強制 false。理由: 挙動を変えない。
- **D11 指標。** `$actor` の `epsilon`（ε-greedy では ε、UQE と Thompson では今どおり UQE の ε）と、UQE 方策の `uqe_tau`（Thompson の `uqe_tau` は D9 のとおり spatial でも NaN）は、spatial のとき lane の平均を返し（今は NaN）、`epsilon_lane_min`・`epsilon_lane_max`・`uqe_epsilon`・`uqe_epsilon_lane_min`・`uqe_epsilon_lane_max`・`uqe_tau_lane_min`・`uqe_tau_lane_max` を足す。tag は `32_agent_base` 群に置き、群名と `/` を含めて 40 文字以内にする。理由: ラダーが時間で変わると平均と両端が無いと読めない。既存の `05_epsilon_learn` の群に揃える。依存: D7。

### 4.3 移行（段階 1）

- **D12 移行の範囲（クリーンブレーク）。** 読み口 2 か所（`ReadActionPolicyConfig`、`RainbowAgentConfig`）、`ActionPolicy` 系、Greedy の作り方（DefaultDQN・Rainbow）、現用の設定 5 ファイル（agent・Atari・DropMerge・GridMaze・LunarLander。agent・Atari・DropMerge の `sync_interval.value` も bare へ）、bat 3 本（11・12・13。13 の `sync_interval.value` も）、PRD072 の golden と比較器（`testdata/prd072/`、`config_resolver_prd072.json`。`sync_interval.value` を含む）、テスト（`config_baseline_test.cpp` の Rainbow 既定の検査、`dqn_based_agent_test.cpp`・`dqn_munchausen_test.cpp`・`trainer_test.cpp` の設定、`config_test.cpp` の ImageCls `learning_rate.value` と `interval.value`、`image_cls_agent_test.cpp` の `learning_rate.value`）、設計書 200 の jp（§2.3・§3 の表と状態の所有。`uqe_tau_decay` の語を含む）、設計書 100 の jp（旧キーを含む箇所）、ガイド 020 の jp の設定例、`CONTEXT.md` の既存項目「tau配置方式」（`uqe_tau_decay` を名指ししていた。本改訂で直した）、CHANGELOG の「非互換変更」に旧キーと新キーの対応表（過去の Run の config dump を読む人向け）。設計書とガイドの en（020・100・200）は本変更で触らず、翻訳スキル（`anet-translate-docs`）が jp の blob の変化から作り直す。実装は新しいキーだけを読み、旧キーの検出・案内・別名・互換の読みを置かない。理由: AGENTS.md のクリーンブレーク方針、[impl.md](../quality/impl.md) 18〜20、PRD001 D14。en を翻訳に任せるのは、en は jp から機械的に作る写しで、同じ変更で手で直すと翻訳スキルの追跡（jp の blob）と二重になるため。
- **D13 書き換えない過去の記録。** `docs/experiments/` の実験記録と config、[ADR 0040](../adr/0040-config-selection-final-value-and-run-profile-tier.md) の例示、`core/anet-core/testdata/prd092/config/` の逐語保存（過去 Run の実効 config の fixture。[README](../../core/anet-core/testdata/prd092/README.md) 121 行）、wiki の過去 Run の記述（subjects の dropmerge・breakout など）は書き換えず、廃止キーの残存検査（§6.1-7）から除く。理由: 観点 17。
- **D14 DropMerge と LunarLander の移行値。** `uqe_eps_end = 0.0` は `spatial.uqe.epsilon.last = 0` のまま移す。DropMerge 本体の設定（0.6〜0）では D8 の伸縮により、最後の lane が 1e-4 から 0 になり、中間 lane の値も `0.6 × (x − 1e-4) / (0.6 − 1e-4)`（x は今の値）へ動く。各 lane がほぼ 1e-4 だけ下がり、相対では小さい ε の lane ほど大きく、1e-4 に近い lane は半分以下になる（x = 0.01 で約 −1%、x = 2e-4 で約 −50%）ので、今の学習系列とは一致しない。受入 1 の等価性は `last = 1e-4` と書いた検査用の設定（両端が 1e-4 以上なので伸縮は恒等で今と同じ値）で確かめ、本体の DropMerge.txt は 0 のまま置く。LunarLander の spatial・log の UQE ε（0.3〜0）も同じ規則で本体の末端0を維持し、中間値は 0.3 × (x − 1e-4) / (0.3 − 1e-4) へ変わる。LunarLander も学習系列の等価性の例外とし、等価性検査だけ last = 1e-4 にする。理由: 書いた意図（0）を保ち、等価性の確認は旧挙動と同じ値になる設定で行う。
- **D15 ラダー両端の既定値（agent.txt の @baseline）。** ε は first 0.4・last 0.01（今の Atari の値）、UQE ε は 0.05・0.05、UQE τ は 0.9・0.5（今の `uqe_*_start/end` の既定を両端に置く）。今ラダーで動いている設定（Atari・DropMerge・LunarLander の A2）は移行で両端を明示的に書く。理由: 既定は切り替えだけで試せる値でよく、実験の値は env 側が持つ（設定はベース＋部分上書き）。
- **D16 旧値の写し方の規則。** (a) `*_decay_steps = 0`（今は早期 return で start のまま）は constant の start の値へ写す（linear の `steps = 0` は fail-fast になるため）。(b) 部分上書き（DropMerge.txt 643 行・GridMaze.txt 112 行の `target_policy.uqe_tau_start = 0.85` など）は bare ではなく sub-key（`target_policy.uqe.tau.start = 0.85`）へ写す（train から写した linear の下で bare が眠るため）。(c) spatial ON の train を `use_optimistic_target = true` で target に写している組み合わせは、旧値がラダーの両端と target の時間の両端を兼ねていたので、時間の値と両端の両方へ写す（現用の設定には無い）。(d) 既存の log ラダーは `spatial.scale = log` のまま値を写す。理由: 旧挙動と一致させる唯一の写し方であり、D3 の眠る規則に当たらない。

### 4.4 学習側（段階 2）

- **D17 対象の 7 値とキー。** `learner.per_beta`（旧 `per_beta_start/end/step`）、`learner.learning_rate`（旧 `learner.alpha`）、`learner.weight_decay`、`learner.gamma`、`learner.munchausen.alpha`、`learner.munchausen.entropy_tau`、`model.soft_update_tau`。各 ProfiledValue。`learner.alpha` を改名するのは、`alpha` が Munchausen の α と PER の `per_alpha` と重なる多義を解消し、ImageCls・MuZero・checkpoint の WARN の表記（`learning_rate`）と揃えるため。`noisy_weight_decay`・`hard_update_interval`・`grad_clip_tau`・`tbo_epsilon`・`per_alpha`・`per_eps`・`adam_eps` は定数のまま（σ 専用の weight_decay は既定 0 で変える需要が無く、ほかは対象外。必要なら同じ器で足す）。理由: 再グリルの裁定「全部そろえる」。Rainbow は共通の `dqn::Learner` を使うので同じキーになり、`RainbowAgentConfig` の読み口も移行する。依存: D1、D3。
- **D18 評価の時機と書き戻し。** Learner は update の先頭で、β を除く 6 値を `counts.exp_step` により 1 回評価し、学習率と weight_decay を optimizer の全 param group に反映する（σ 専用 group の weight_decay は `noisy_weight_decay` のまま）。soft update の τ は毎 learn step の soft copy にその時点の値を使う。**PER β だけは今の適用時機を保つ**: 更新の終わりに `counts.exp_step` で評価した値を次の更新の抽選の重みに使い、最初の更新は step 0 で評価した値を使う（§2.2 のとおり 1 更新ずれる今の形と同じ）。理由: 6 値はこれまで定数だったので先頭で評価しても値が変わらないが、β は今も直線で変わっているので、先頭で評価すると全更新の重み付けがずれて G6 の等価性（§6.2-1）が崩れる。1 更新の遅れは `exp_step` の軸ではほぼ同じ値で、学習上の意味は変わらない。ImageCls と同じ形で、FusedAdamW も options から毎 step 読む（§2.2）。軸はすべて `exp_step`（§7 の step 軸の先送りと整合）。依存: D17。
- **D19 Munchausen の α・entropy_tau と γ の Actor 側・診断側。** Actor は同じ ProfiledValue 設定を持ち、`UpdateSchedule` で `exp_step` により評価して Q ヒント（近似 PER の初期優先度）に使う。Evaluation Trace の step 診断（`DqnAgentStepDiagnosticsProvider`）も Actor と同じ `exp_step` で評価した α・entropy_tau・γ を受け取って使う（`Observe` は step を持たないので、評価した値を Actor 側から渡す）。Replay の初期優先度の推定は Push 時のヒントを使うので、Push 時の値になる。理由: ヒントと診断は Learner の target と同型でなければならない（ADR 0036、ADR 0058）。依存: D17。
- **D20 γ の扱い（[ADR 0065](../adr/0065-scheduled-gamma-recomputes-return-at-sample.md)）。** γ が定数（bare か `type = constant`）なら、今どおり Push 時に作った n-step 収益を学習に使い、経路も値も今と同じにする。γ が schedule なら、学習に使う n-step 収益を抽選時に、その遷移の保存済みの step 報酬から今の γ で作り直す（`DefaultExperienceBuilder` と同じ逆順の和で足し、定数のときに保存値と一致する形にする）。bootstrap の γ^n も同じ今の γ なので、学習の target の中に γ のずれは無い。Push 時の収益は初期優先度のヒントの完成に使い、保存形（slot の形）は変えない。報酬スケーラの自動 post_scale と実効ホライズンのログは構築時（step 0）の γ。MuZero の Replay は定数の γ を渡し続ける。Replay は γ を自分で持たず、Push と Sample の入力として呼び出し側が渡す今の γ を使う。理由: Replay は step 報酬を既に保存している（§2.2 (b)）ので保存形を変えずに作り直せ、定数のときは経路を分けることで等価性と計算量を今のまま保てる。経路の切り替えは設定の `type` による構築時の分岐で、値による黙った再解釈ではない。依存: D17。退けた案: 「保存済み収益は Push 時の γ、bootstrap は学習時の γ の 2 時点を契約にする」（変化中に target の中へ既知のずれを作る）、「γ は今回 constant だけ許す」（器としては γ だけ空く）、「常に抽選時に作り直す」（定数でも計算が増え、Horner の順を揃えないと checksum が変わる）。
- **D21 値域の検査。** ProfiledValue の `min_value`・`max_value` で設定時に端点（constant の value、各段の start/end）を検査する。ε・UQE ε・β・γ・Munchausen α は [0,1]、τ は [0,1]、学習率・weight_decay は 0 以上、entropy_tau は 0 より大、`time_constant` は 0 より大。soft update の τ の「[0, 0.1] または 1」は今と同じく spectral norm を持つ network があり soft update のときだけ、端点に適用する（schedule を持つなら全端点が [0, 0.1]、1 は constant のときだけ）。理由: 今の検証をそのまま端点へ移し、範囲を広げず、黙った clamp や既定置換をしない（AGENTS.md 設定値の扱い）。依存: D1。
- **D22 checkpoint の引き継ぎ。** optimizer options の比較と上書きは、新しい Run の step 0 で評価した schedule の値と行う。以後は毎 update で schedule の値を書き戻すので、読み込み時の比較は「checkpoint の optimizer が持っていた lr と、この Run の初期値が違う」ことを知らせる意味になる。ProfiledValue は状態を持たないので checkpoint の形は変えない。理由: 引き継ぎは Run の再開ではなく StepCounts を引き継がない（PRD091）。依存: D18。
- **D23 学習側の指標。** `$agent` に `learning_rate`・`weight_decay`・`soft_update_tau`・`gamma`・`munchausen_alpha`・`munchausen_entropy_tau` を足す（`per_beta` は既存）。tag は `32_agent_base` 群で 40 文字以内。理由: schedule どおりに動いたことを Run の系列で確かめるため（成功の測り方）。

### 4.5 順序と運用

- **D24 段階と受入。** 段階 1（探索側）、段階 2（学習側）の順に実装し、受入は段階ごとに持つ。本 PRD の完了は §6 の受入で判定し、qbert の成績比較（§3）は完了条件に含めない。理由: 成功の測り方の実験は段階 1 の受入が済めば始められ、実験の実行と結果は実験記録が正本になる。
- **D25 走っている実験との関係。** 移行は、走行中の Run（サブ機のキュー、メイン機の隔離した実行体の Run）が終わってから入れ替える。理由: bat は前段の終了を待って次段を起動する段つなぎで動いていて、途中で bat か実行体の一方だけが新キーになると、次段の Run が旧キーの引数を新しい実行体に渡して（またはその逆で）起動時に止まる。同じ理由で bat と実行体は同時に替える（2026-10-24 の TempoRL のキー名変更で、bat だけが先に変わって次段が止まったずれと同じことを避ける）。
- **D26 用語。** `CONTEXT.md` の「NoisyNet・探索」に「spatial exploration（ε ラダー）」と「時間の値」を足す。既存の語では言い換えられない。`vectorized exploration`（PRD002 の導入時の名）は lane ごとの固定の並べ方だけを指し、両端が時間で変わることと、時間の値との区別を含まない。`eps_start`・`eps_end` は §1 の問題 3 のとおり時間の両端と lane の両端を兼ねていて、まさにその二義を解くのが本 PRD である。理由: 用語集に無い概念に名前を付けるときは、既存の語で言い換えられないことを書く。

## 5. 設定キー

### 5.1 既定（agent.txt の @baseline、今の値と同じ意味）

定数は bare、スケジュールは sub-key で書く（D3）。数字は今の agent.txt の値を写す（下は形を示す例で、実装時に今の値を転記する）。

```
DefaultDQNAgent.@baseline : actor.[train].policy.epsilon.type  ?= linear
DefaultDQNAgent.@baseline : actor.[train].policy.epsilon.start ?= 1.0
DefaultDQNAgent.@baseline : actor.[train].policy.epsilon.end   ?= 0.05
DefaultDQNAgent.@baseline : actor.[train].policy.epsilon.steps ?= 10,000,000
DefaultDQNAgent.@baseline : actor.[train].policy.uqe.tau.type  ?= linear
DefaultDQNAgent.@baseline : actor.[train].policy.uqe.tau.start ?= 0.9
DefaultDQNAgent.@baseline : actor.[train].policy.uqe.tau.end   ?= 0.5
DefaultDQNAgent.@baseline : actor.[train].policy.uqe.tau.steps ?= 10,000,000
DefaultDQNAgent.@baseline : actor.[train].policy.uqe.epsilon   ?= 0.05
DefaultDQNAgent.@baseline : actor.[train].policy.uqe.use_tail_mean ?= false
DefaultDQNAgent.@baseline : actor.[train].policy.spatial.enabled ?= false
DefaultDQNAgent.@baseline : actor.[train].policy.spatial.scale   ?= log
DefaultDQNAgent.@baseline : actor.[train].policy.spatial.epsilon.first     ?= 0.4
DefaultDQNAgent.@baseline : actor.[train].policy.spatial.epsilon.last      ?= 0.01
DefaultDQNAgent.@baseline : actor.[train].policy.spatial.uqe.epsilon.first ?= 0.05
DefaultDQNAgent.@baseline : actor.[train].policy.spatial.uqe.epsilon.last  ?= 0.05
DefaultDQNAgent.@baseline : actor.[train].policy.spatial.uqe.tau.first     ?= 0.9
DefaultDQNAgent.@baseline : actor.[train].policy.spatial.uqe.tau.last      ?= 0.5
DefaultDQNAgent.actor.@eval_base : policy.epsilon ?= 0.05
DefaultDQNAgent.actor.@eval_base : policy.uqe.tau ?= 0.5
DefaultDQNAgent.actor.@eval_base : policy.uqe.epsilon ?= 0
DefaultDQNAgent.@baseline : learner.learning_rate ?= 2.5e-4
DefaultDQNAgent.@baseline : learner.weight_decay  ?= 1e-2
DefaultDQNAgent.@baseline : learner.gamma         ?= 0.99
DefaultDQNAgent.@baseline : learner.per_beta.type  ?= linear
DefaultDQNAgent.@baseline : learner.per_beta.start ?= 0.4
DefaultDQNAgent.@baseline : learner.per_beta.end   ?= 1.0
DefaultDQNAgent.@baseline : learner.per_beta.steps ?= 10,000,000
DefaultDQNAgent.@baseline : learner.munchausen.alpha       ?= 0.9
DefaultDQNAgent.@baseline : learner.munchausen.entropy_tau ?= 0.03
DefaultDQNAgent.@baseline : model.soft_update_tau ?= 0.001
```

### 5.2 今の Atari のラダー（0.4〜0.01、時間で変わらない）

```
run.@a5_apex : A2.actor.[train].policy.spatial.enabled = true
run.@a5_apex : A2.actor.[train].policy.spatial.epsilon.first = 0.4
run.@a5_apex : A2.actor.[train].policy.spatial.epsilon.last  = 0.01
```

### 5.3 BTR の ε（指数で減らし、25M で 0）

```
run.@btr_eps : A2.actor.[train].policy.spatial.enabled = false
run.@btr_eps : A2.actor.[train].policy.epsilon.type = phased
run.@btr_eps : A2.actor.[train].policy.epsilon.phases = decay off
run.@btr_eps : A2.actor.[train].policy.epsilon.phase.[decay].type = exponential
run.@btr_eps : A2.actor.[train].policy.epsilon.phase.[decay].start = 1.0
run.@btr_eps : A2.actor.[train].policy.epsilon.phase.[decay].end = 0.01
run.@btr_eps : A2.actor.[train].policy.epsilon.phase.[decay].time_constant = 2,000,000
run.@btr_eps : A2.actor.[train].policy.epsilon.phase.[decay].steps = 25,000,000
run.@btr_eps : A2.actor.[train].policy.epsilon.phase.[off].type = constant
run.@btr_eps : A2.actor.[train].policy.epsilon.phase.[off].value = 0
run.@btr_eps : A2.actor.[train].policy.epsilon.phase.[off].steps = 1
```

`run.@a5_apex` の後に `run.@btr_eps` を重ねると、ラダーの両端は書かれたまま使われず、外せばラダーに戻る。今は A1 で `eps_start = 1.0`・`eps_end = 0.01` を時間の両端として書き、A2 で `eps_start = 0.4` をラダーの上端に書き換え、`eps_end = 0.01` をそのままラダーの下端に流用している。ラダーを切るだけの比較でも `eps_start`・`eps_end`・`eps_decay_steps` を書き直す必要があり、2026-10-23 の BTR 寄せの Run がそうだった。

### 5.4 ラダーを 25M から 35M で消す

```
run.@ladder_fade : A2.actor.[train].policy.spatial.epsilon.first.type = phased
run.@ladder_fade : A2.actor.[train].policy.spatial.epsilon.first.phases = keep fade
run.@ladder_fade : A2.actor.[train].policy.spatial.epsilon.first.phase.[keep].type = constant
run.@ladder_fade : A2.actor.[train].policy.spatial.epsilon.first.phase.[keep].value = 0.4
run.@ladder_fade : A2.actor.[train].policy.spatial.epsilon.first.phase.[keep].steps = 25,000,000
run.@ladder_fade : A2.actor.[train].policy.spatial.epsilon.first.phase.[fade].type = linear
run.@ladder_fade : A2.actor.[train].policy.spatial.epsilon.first.phase.[fade].start = 0.4
run.@ladder_fade : A2.actor.[train].policy.spatial.epsilon.first.phase.[fade].end = 0
run.@ladder_fade : A2.actor.[train].policy.spatial.epsilon.first.phase.[fade].steps = 10,000,000
```

`last` も同じ形で 0.01 から 0 へ書く。log の並べ方では、両端が下がるにつれて列が線形に伸縮し、端が 0 に着いた lane はちょうど 0 になる（D8）。

### 5.5 ラダーの下端を 0 にする（4 lane の例）

```
run.@ladder0 : A2.actor.[train].policy.spatial.epsilon.first = 0.4
run.@ladder0 : A2.actor.[train].policy.spatial.epsilon.last  = 0
```

lane の値は 0.4 / 0.0251 / 0.0015 / 0 になる（0.4〜1e-4 の log 間隔の列を 0.4〜0 へ伸縮。最後の lane だけ 0。D8）。

### 5.6 学習率を 30M で下げる（段階 2）

```
run.@lr_drop : A1.learner.learning_rate.type = phased
run.@lr_drop : A1.learner.learning_rate.phases = hold drop
run.@lr_drop : A1.learner.learning_rate.phase.[hold].value = 2.5e-4
run.@lr_drop : A1.learner.learning_rate.phase.[hold].steps = 30,000,000
run.@lr_drop : A1.learner.learning_rate.phase.[drop].value = 7.1e-5
run.@lr_drop : A1.learner.learning_rate.phase.[drop].steps = 1
```

baseline の bare `learning_rate ?= 2.5e-4` は `value` 欄として眠り、使われるのは phased の段である（D3）。逆に、baseline が `epsilon.type ?= linear` の下で `A2.actor.[train].policy.epsilon = 0` と書くと、その 0 は眠って linear のまま動く。定数に戻すには `epsilon.type = constant` を添える。

### 5.7 β を予算の半分で収束させる（段階 2。今の Atari.txt の `${@vars.half_exp_step}` と同じ意味）

```
A1.learner.per_beta.type  = linear
A1.learner.per_beta.start = 0.4
A1.learner.per_beta.end   = 1.0
A1.learner.per_beta.steps = ${@vars.half_exp_step}
```

### 5.8 部分上書きの写し方（D16 (b)）

```
# 旧: A1.target_policy.uqe_tau_start = 0.85（train の linear 0.95→0.85 を写したあと start だけ上書き）
A1.target_policy.uqe.tau.start = 0.85
```

## 6. 検証と受入

### 6.1 段階 1（探索側）

1. **既定の等価性**: 旧キーから新キーへ置き換えた設定で、Atari（ラダー）・DropMerge と LunarLander（UQE、両方とも `spatial.uqe.epsilon.last = 1e-4` の検査用設定）・GridMaze の短い決定的 Run の metrics checksum が、置き換え前と一致する。比較から除く tag は次の名前のものだけ: `$actor` の `epsilon`・`uqe_tau` と、D11 で足す `epsilon_lane_min`・`epsilon_lane_max`・`uqe_epsilon`・`uqe_epsilon_lane_min`・`uqe_epsilon_lane_max`・`uqe_tau_lane_min`・`uqe_tau_lane_max`、`$agent` の `per_beta` と、D23 で足す `learning_rate`・`weight_decay`・`soft_update_tau`・`gamma`・`munchausen_alpha`・`munchausen_entropy_tau`。loss・TD 誤差・Q・収益・優先度などの学習の系列はすべて比較に含める。比較の相手は実装前に保存した基準の実行体・依存 DLL・出力とする。ProfiledValue の補間が double で行われることによる float との差で一致が崩れたときは、ε・τ・β の評価を今の float の経路に揃えて再比較する。
2. **設定の等価性**: 7 環境の全キー・文字列値を比べ、探索のキーの置き換えと `sync_interval.value` → `sync_interval` の記法の置き換え以外に差が無い（PRD072 の検査を流用。golden と比較器は作り直す）。`check_default_leaves.py` が errors 0。
3. **単体テスト（指数）**: `exponential` の値が式どおり。`phased` の段として使えて、段の境目で値が飛ぶ形（decay → off）を書ける。BTR の離散の式との差が全 step で ε の値に対する絶対差 1e-6 以内。root で `steps` を書かなくても通り、段で `steps = 0` と `time_constant <= 0` は fail-fast。
4. **単体テスト（記法）**: bare の `X = v` が constant として読まれ、`type` を書かない定数が 1 行で済む。bare の下に `.type = linear` と `.start/.end/.steps` を重ねると linear で動き、dump には linear の欄だけが記録される（bare の値は記録されない）。`sync_interval = 400` が今の `sync_interval.value = 400` と同じ同期周期になる。
5. **単体テスト（ラダー）**: 両端が一定なら構築時の値と毎 step の値が一致する。両端が変わるときは、各 step の lane の値が両端の今の値から作られる。log で 0.4〜0 は 0.4 / 0.0251 / 0.0015 / 0（4 lane）、0〜0 は全 lane 0、0.4〜5e-5 は最後の lane が 5e-5、3 lane で first を 1e-4 の前後で動かしても中間 lane は 5e-5 で連続、負の端は fail-fast。既存の 0.4〜0.01・0.05〜1e-4 の log ラダーの値が今と一致する。
6. **単体テスト（切り替えだけで済む）**: 時間の値（`epsilon` など）とラダーの両端（`spatial.epsilon.first/last` など）を両方書いた設定で、`spatial.enabled` だけを変えると、それぞれの側の値で動く。使わない側の値は評価しない。値そのものの検査（0〜1 の範囲など）は両方の側に行う。
7. **移行の完了**: 廃止したキー名（`eps_start`・`eps_end`・`eps_decay_steps`・`use_spatial_exploration`・`spatial_scale_type`・`uqe_tau_*`・`uqe_eps_*`・`uqe_use_tail_mean`）と ProfiledValue の root の `.value`（`sync_interval.value`、`learning_rate.value` など。段の中の `.phase.[x].value` は現行契約なので除く）を検索して、現用の範囲（コード・設定・bat・テスト・PRD072 の testdata・現行の文書の jp と `CONTEXT.md`）に参照が残っていない。D13 の過去の記録、翻訳スキルが作り直す en（D12）、CHANGELOG（旧キーと新キーの対応表を書く場所なので検査の対象外）は除く。実装に旧キーのための検出や分岐が無い（[impl.md](../quality/impl.md) の 18〜20）。CHANGELOG に読み替えの対応表がある。
8. **指標**: spatial の Run で `$actor epsilon` が lane の平均を、`epsilon_lane_min`・`epsilon_lane_max` が両端を返す。UQE 方策の `uqe_tau` も spatial で lane の平均。非 spatial では今と同じ値。Thompson の `uqe_tau` は spatial でも非 spatial でも NaN。
9. **実 Run**: BTR の ε の形で Atari の短い Run を回し、`epsilon` の scalar が 25M の境目で 0 になることを確かめる（予算を短くした確認用の設定で境目を手前に置いてよい）。qbert の成績比較は受入に含めない（§3、D24）。

### 6.2 段階 2（学習側）

1. **既定の等価性**: 7 値を定数（bare）と `per_beta` の linear で書いた設定で、4 環境の短い決定的 Run の metrics checksum が、段階 1 完了時と一致する。比較から除く tag は §6.1-1 と同じ名前の一覧。
2. **設定の等価性**: 7 環境の全キー・文字列値を比べ、学習側のキーの置き換え（`alpha` → `learning_rate`、`per_beta_*` → `per_beta.*`）以外に差が無い。
3. **単体テスト（書き戻し）**: 学習率と weight_decay の phased で、段の境目の update から optimizer の全 group の lr と weight_decay が新しい値になり、σ 専用 group の weight_decay は `noisy_weight_decay` のまま。
3a. **単体テスト（β の時機）**: `per_beta` の linear で、更新 k の抽選に渡る β が更新 k−1 の終わりの `exp_step` で評価した値、最初の更新では step 0 の値になる。今の `per_beta_start/end/step` の直線と同じ step 列で同じ値になる。
4. **単体テスト（γ）**: 定数の γ では、抽選時に作り直した収益と Push 時の保存値が全遷移で一致し、学習は保存値の経路を使う。schedule の γ では、境目の前に Push した遷移を境目の後に抽選すると、収益と bootstrap が後の γ で計算される。初期優先度のヒントは Push 時の γ。報酬スケーラの post_scale は構築時の γ。
5. **単体テスト（Munchausen と診断）**: α の schedule で、Actor の Q ヒント、step 診断、Learner の target が同じ step で同じ α を使う。
6. **単体テスト（引き継ぎ）**: lr の phased の後段で保存した checkpoint を同じ設定の新しい Run に引き継ぐと、step 0 の値との差が WARN に出て、以後の update は schedule の値で動く。
7. **値域**: 端点が範囲外の設定（β 1.5、spectral norm ありで soft update の τ の linear 1.0→0.001、entropy_tau 0）が設定時に fail-fast し、メッセージにキー・値・期待範囲が出る。spectral norm が無ければ τ の linear 1.0→0.001 は今と同じく通る。
8. **移行の完了**: `learner.alpha`・`per_beta_start/end/step` が現用の範囲に残っていない（D13 の除外は同じ）。
9. **指標**: `$agent learning_rate`・`weight_decay`・`soft_update_tau`・`gamma`・`munchausen_alpha`・`munchausen_entropy_tau` が設定の schedule どおりの系列になる。
10. **実 Run**: §5.6 の phased で短い Run を回し、`learning_rate` の系列が境目で切り替わる。

## 7. 作らないもの・先送り

| 項目 | 裁定 | 理由と再開条件 |
|---|---|---|
| NoisyNet の雑音の倍率を時間で変える | 先送り | σ は学習するパラメータで、外から掛ける倍率は forward の雑音の作り方（`nn_forward` の noisy の指定）に新しい入力が要る。BTR も雑音は変えない。ε の形で差が説明できないときに起こす |
| step の軸を learn_step やエピソード数にする | 先送り | 今の探索と β はすべて exp_step で動いている。別の軸が要る例が出たら起こす |
| 眠る bare の事故を防ぐ仕組み（`check_default_leaves.py` に「確定葉の bare の下に別層の type ≠ constant」を足す静的検査、既定葉の情報を読み取りまで運んで確定葉の眠る bare を fail-fast にする実行時検査） | 先送り | 今回は眠る規則を文書に書くだけにした。再開条件: 眠った値に気づかずに Run を回した事故が実際に起きたとき。静的検査は設定と bat を覆い、実行時検査は手入力の CLI も覆うが、env の確定葉の定数を Run プロファイルで schedule に差し替える用法も止めてしまう |
| MuZero の温度 `temp_start/end/decay_steps` | 範囲外 | 試作 Agent。本線に戻すときに同じ器で足す |
| `noisy_weight_decay`・`hard_update_interval`・`grad_clip_tau`・`tbo_epsilon`・`per_alpha`・`per_eps`・`adam_eps` の時間変化 | 先送り | Run の中で変えた実績が無い。必要になったら同じ器で足す |
| Thompson sampling の τ の区間（`thompson.tau_low/high`） | 不採用 | 区間は存在しない（§2.1）。暫定 PRD の案を削った |
| log ラダーの 0 端を fail-fast にして新しい並べ方 `exponential` で 0 を書く、下限をキーにする、clamp を残す | 不採用 | D8 の退けた案。モードやキーが増える、または書いた値と違う値で動く |
| 最上位の lane を時間の値 ε(t)、ラダーを固定の指数 α で表す形（Ape-X の ε^(1+α·u)。キーが 1 個と α に減り、ラダーを消すのも schedule 1 本で書ける） | 不採用 | 時間の値とラダーを別々に置いて切り替えだけで行き来する G4 が成り立たない。UQE の τ（0.9〜0.5）は比で表す形に合わず、「最後の lane だけ 0」も書けない |
| γ の 2 時点の契約（保存済み収益は Push 時の γ、bootstrap は学習時の γ） | 不採用 | 変化中に target の中へ既知のずれを作る。Replay が step 報酬を保存しているので、作り直す方が安い（D20） |
| γ を今回 constant だけに限る | 不採用 | 器としては γ だけ空く。schedule のときだけ作り直す形で両立できた（D20） |
| 旧キーの検出・案内・読み替え（廃止する root の `.value` の検出を含む） | 不採用 | クリーンブレーク。移行は同じ変更の中で設定・golden・文書を直して担保し、実装に過去の契約を残さない |
| qbert の成績比較を本 PRD の完了条件にする | 範囲外 | 比較の設計は §3 に書き、実行と結果は実験記録の campaign が正本。完了条件に入れると実装の完了が Run の予算に縛られる。再開条件: なし |
| ProfiledValue の定数を常に `.value` で書く、bare を書いたら type を constant に強制する | 不採用 | 前者は定数の行が全部 1 段増える。後者は baseline の bare を Run プロファイルで schedule に差し替える用法が fail-fast になる（ADR 0064） |

## 8. 未決

なし。暫定の 11 件は §4 で確定した（§8 の旧 1〜11 は D5・D2・D8・D9・D21・§7 の step 軸・D7・D11・D12・D25・D15 に対応する）。

## 9. 依存と順序

実装上の依存は無い。TempoRL（PRD098）の設定の見直しが作業ツリーで未コミットなので、同じファイル（agent.txt・Atari.txt・default_dqn_agent.hpp）を触る時期は、098 のコミットの後にする。移行の時機は D25 のとおり。

## 10. 実装契約

### 10.1 クラス一覧

従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| `ProfiledValueConfig` / `ProfiledValuePhaseConfig` | 変更 | `time_constant` 欄を足す。root の定数を bare で読み、root の `.value` は読まない。既定 type は constant。dump は実効 type の欄だけを記録する | 312 ユーティリティ（`schedule.hpp`） |
| `ProfiledValue` | 変更 | `exponential` 型の評価（root と段）。端点の検査に `time_constant > 0` を足す | 312 ユーティリティ（`schedule.hpp`） |
| `SpatialLadderSchedule` | 新規 | 両端の ProfiledValue と並べ方（log / linear）を持ち、`exp_step` で両端を評価して lane ごとの値を作る。log の下限の規則（D8）と負の端の fail-fast を持つ | 321 Agent 共通基盤（`dqn_based_agent.hpp` / `.cpp`） |
| `ActionPolicyConfig` | 変更 | 素の値を `epsilon`・`uqe.epsilon`・`uqe.tau` の ProfiledValue 設定と `spatial.*`（enabled・scale・3 組の両端）に置き換える | 321 Agent 共通基盤（`agent.hpp`） |
| `ActionPolicy` / `EpsilonGreedyActionPolicy` / `UQEActionPolicy` / `ThompsonSamplingActionPolicy` | 変更 | `UpdateSchedule` で時間の値と `SpatialLadderSchedule` を評価する。`CreateSpatialTensor` の clamp と WARN を消す。`GetScalar` に lane の平均・最小・最大を足し、Thompson の `uqe_tau` を NaN にする | 321 Agent 共通基盤（`dqn_based_agent.hpp` / `.cpp`） |
| `DefaultDQNAgentConfig::ReadActionPolicyConfig` / `RainbowAgentConfig` | 変更 | 新キーの読み。target_policy の強制 0 を `epsilon`・`uqe.epsilon` の constant 0 で行う。学習側 7 値の ProfiledValue の読み | 324 Agent 個別実装 DQN 系列（`default_dqn_agent.hpp`、`rainbow_agent.hpp`） |
| `DefaultDQNAgent` / `RainbowAgent` | 変更 | Greedy 方策を `epsilon` constant 0 で作る。実効ホライズンのログと報酬スケーラは step 0 の γ を使う。Actor と step 診断へ Munchausen と γ の ProfiledValue 設定を渡す | 324 Agent 個別実装 DQN 系列（`default_dqn_agent.cpp`、`rainbow_agent.cpp`） |
| `LearnerConfig` / `NetworkModelConfig` | 変更 | `per_beta`・`learning_rate`・`weight_decay`・`gamma`・`munchausen.alpha`・`munchausen.entropy_tau`・`soft_update_tau` を ProfiledValue 設定にする | 321 Agent 共通基盤（`dqn_based_agent.hpp`） |
| `Learner`（`TDLearner` / `QRLearner` / `IQNLearner` 共通部） | 変更 | update の先頭で β を除く 6 値を評価し、学習率と weight_decay を optimizer へ反映する。β は更新の終わりに評価して次の抽選に使う（D18）。Replay の Push と Sample に今の γ を渡す。引き継ぎ時の比較に step 0 の値を使う。`GetScalar` に学習側の指標を足す | 321 Agent 共通基盤（`dqn_based_agent.cpp`） |
| `NetworkModel` | 変更 | soft update の τ を毎回値で受ける。構築時の検証を端点の検証へ移す（spectral norm があるときだけ） | 321 Agent 共通基盤（`dqn_based_agent.cpp`） |
| `Actor`（DQN） / `ActorQHintConfig` | 変更 | Munchausen の α・entropy_tau を `exp_step` で評価して Q ヒントに使い、step 診断へ同じ値を渡す | 321 Agent 共通基盤（`dqn_based_agent.cpp`） |
| `DqnAgentStepDiagnosticsProvider` | 変更 | 構築時に写した γ・Munchausen ではなく、Actor から渡された今の値で収益と割引とヒントを作る | 321 Agent 共通基盤（`dqn_based_agent.cpp`） |
| `ReplayBuffer`（`DefaultExperienceBuilder`・`DefaultSampleExtractor`・初期優先度の完成） | 変更 | γ を Push と Sample の入力で受け取り、自分で持たない。γ が schedule の構成では、抽選時に step 報酬から逆順の和で収益を作り直す経路を持つ | 318 強化学習 共通基盤（`replay_buffer_impl.cpp`） |

削除するクラスは無い。

### 10.2 契約の要点（実装計画へ渡すもの）

- 新キーの一覧は §4.2・§4.4、既定は §5.1、旧値の写し方は D16。旧キーは読まない。
- 評価の時機: Actor は `UpdateSchedule(counts)` で `counts.exp_step`、Learner は `UpdateFromBatch` の先頭で `counts.exp_step`。PER β だけは更新の終わりに評価して次の更新の抽選に使う（D18）。
- 等価性の基準は実装前に保存した実行体と出力（§6）。比較から除く tag は §6.1-1 の名前の一覧だけで、ほかの系列はすべて比べる。
- 文書: 設計書 200 の §2.3（UQE の τ、Thompson、ラダーの下限の規則）・§3（状態の所有に ProfiledValue の評価を足す）・設定表・眠る bare の規則、ガイド 020 の設定例と眠る bare の規則、CONTEXT.md、CHANGELOG の対応表。

## 11. 参照

- BTR 公開実装 `Agent.py`（`EpsilonGreedy`、`choose_action` の `eps_disable` と `total_frames / 2`）（`VIPTankz/BTR`、手元の写し `C:\dev\BTR`）
- [anet/schedule.hpp](../../core/anet-core/include/anet/schedule.hpp)（ProfiledValue）
- [ADR 0064](../adr/0064-profiledvalue-bare-scalar-reads-as-value-field.md)、[ADR 0065](../adr/0065-scheduled-gamma-recomputes-return-at-sample.md)、[ADR 0063](../adr/0063-replay-slot-keeps-per-step-reward-and-end-and-samples-variable-windows.md)（step 報酬の保存）
- [PRD091](091_checkpoint_load_optimizer_options_10prd.md)（optimizer options の引き継ぎ）、[PRD094](094_evalpanel_step_history_10prd.md)（step 診断）、[PRD072](done/072_config_selection_final_value_10prd.md)（設定の比較）、[PRD002](done/002_vec_exp_10prd.md)（spatial exploration の導入）、[PRD031](done/031_imagecls_cosine_lr_10prd.md)（ProfiledValue の導入）
