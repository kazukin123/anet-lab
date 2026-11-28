# RFD0207: DQN 系の小さい直しとリファクタリングをまとめて行う（学習の設定の値域、診断値の NaN、fatal の重複、Learner の State と値の振り分け、写しの集約、特性テスト）

> 2026-11-27 起票（暫定）。質疑を省いた。確定した決定は持たず、AI が置いた既定と決めきれない点を §4 の未決表に置いた。§5〜§7 は §4 の暫定の既定を前提に書き、前提にした行の番号を名指す。
> 2026-11-28 改訂（暫定）。入口レビューを受け、設定値域、診断の成立条件、テスト移行、短縮比較と性能測定の条件を具体化した。
> 起点: 2026-11-27 の会話。開発者は、仕様の調整がほぼ要らない小さい課題を集めて、夜のうちに別の枠で実装まで進めたいと述べた。触るファイルが重ならないように、置き場ごとに 3 本（RFD0205 文書、RFD0206 基盤とアプリ、本書 DQN 系）に分けることを選んだ（§9）。
> 課題: 12 件（§8 の対応表）。どれも `src/algo/dqn` の中の直しである。

## 1. 目的と問題

### 1.1 問題

DQN 系（`src/algo/dqn`）に、方針まで課題に書かれた小さい直しが 12 件たまっている。4 種類に分かれる。

- **学習の設定の値域を検証しない（1 件）。** `learner.grad_clip_tau`・`learner.td_clip_value`・`learner.update_interval` などは、0 以下でも構築が通る。勾配が 0 か逆向きになる、TD の clip が黙って外れる、更新の間隔が黙って 1 になる、などが起きる（[issue-20261123-eb0a0c](../management/issues/issue-20261123-eb0a0c.md)）。
- **診断値とログの誤り（3 件）。** UQE を使わない方策の `uqe_tau` が NaN でなく 0 を返す（[issue-20261125-c702d2](../management/issues/issue-20261125-c702d2.md)）。点の UQE で `iqn_policy_margin_mc_ratio` が同じ τ の複製から作られ、約 20 万の意味のない値になる（[issue-20261125-770358](../management/issues/issue-20261125-770358.md)）。行動の番号が不正なメトリクスの鍵で、同じ fatal を 2 回出す（[issue-20261125-ea9fef](../management/issues/issue-20261125-ea9fef.md)）。
- **持ち主のずれと写し（7 件）。** Learner だけが使う State を Agent が持つ（[issue-20261125-84c03f](../management/issues/issue-20261125-84c03f.md)）。Agent の値の振り分けが Learner の群の一覧を写す（[issue-20261125-6a8e46](../management/issues/issue-20261125-6a8e46.md)）。選んだ行動の値を取り出す式が 13 か所にある（[issue-20261125-9e8834](../management/issues/issue-20261125-9e8834.md)）。probe の抽選の手順が 4 か所にある（[issue-20261125-9f1801](../management/issues/issue-20261125-9f1801.md)）。価値の分解の診断の並びを手で合わせている（[issue-20261125-5ebf2f](../management/issues/issue-20261125-5ebf2f.md)）。Actor Q ヒントの入力の組み立てが 2 か所にある（[issue-20261125-de8cfd](../management/issues/issue-20261125-de8cfd.md)）。内的の割引率の選び方が 6 か所にある（[issue-20261123-d74c3f](../management/issues/issue-20261123-d74c3f.md)）。
- **テストの穴（1 件）。** stack を有効にした DefaultDQNAgent の可視化用の forward（`GetTensorDictFunction`）を、2 枚以上の stack で通すテストが無い（[issue-20261123-e43866](../management/issues/issue-20261123-e43866.md)）。

### 1.2 ゴール

- 題材の課題 12 件がすべて片付く（各課題の事象が今のコードから無くなる）。
- 標準の Forwarder を使う正しい設定では、学習と行動選択の値、設定の記録、ログ・WARN・ERROR の文言と順と回数、乱数の消費の順を維持する。わざと変えるものは次の 4 つ。
  - 範囲外の学習の設定が、構築で止まる（U1・U2）。
  - UQE を使わない方策の `uqe_tau` の点が、記録されなくなる（U3）。
  - 点の UQE で全範囲の問い合わせが無い構成の `iqn_policy_margin_mc_ratio` の点が、記録されなくなる（U4）。
  - 行動の番号が不正な指標キーを拒否するとき、fatal の同じ文を 1 回出す（U5）。
- 写しを 1 か所に寄せ、次に群・入力・指標を足すときの直し漏れを無くす。

### 1.3 非ゴール（退けた案と、本当の制約か）

- **課題が名前を挙げる別の課題を、範囲に入れない。** [issue-20261125-fa1be2](../management/issues/issue-20261125-fa1be2.md)（`ActorQHintConfig` を 2 つに分ける）、[issue-20261125-8eff2e](../management/issues/issue-20261125-8eff2e.md)（合わせた Q を作る経路が 2 つ）、[issue-20261103-2d66d2](../management/issues/issue-20261103-2d66d2.md)（RFD0134。ロックの範囲）、[issue-20261111-17fea4](../management/issues/issue-20261111-17fea4.md)（方策の種類を文字列で持ち回る）、[issue-20261112-4efa57](../management/issues/issue-20261112-4efa57.md)（replay fit の診断の写し）、[issue-20261111-e80c7b](../management/issues/issue-20261111-e80c7b.md)（`DQNActionInfo::GetScalar` の分割）、[issue-20261123-c3b5ef](../management/issues/issue-20261123-c3b5ef.md)（n 先のヒントを待つ手順の写し）、[issue-20261111-54c7e1](../management/issues/issue-20261111-54c7e1.md)（呼ばれないコード）、[issue-20261103-aa8701](../management/issues/issue-20261103-aa8701.md)（学習の進みの保存）が当たる。本当の制約ではない。どれも、形を選ぶ判断が残る別の関心なので、課題ごとに扱う。54c7e1 が挙げる `RuntimeVars::uqe_tau` は、U6 で `RuntimeVars` ごと無くなる（§6.5）。
- **`learner.replay_ratio` の負の値で間隔のモードを選ぶ今の形を、モードの設定キーに改める案。** 本当の制約ではない（クリーンブレークで移せる）。設定キーを変える別の関心なので、範囲に入れない。本書は、0 を範囲外として止めるだけにする（U2。課題の候補は U14）。
- **点の UQE の問い合わせを常に 1 本にして、行動の forward を軽くする案**（770358 の対応方針の最後の行）。本当の制約ではない。平均の丸めが変わり、同じ seed の結果が一致しなくなるので、挙動を変えない本書の線に入らない（別の関心）。
- **Evaluation Trace の列、Actor Q ヒントの列、step 診断の値の意味を変えること。** 本当の制約ではない。本書は組み立ての写しをまとめるだけで、列の数と値は今のまま（P10、U9）。

### 1.4 機構とゴールの対応

| 足すもの・変えるもの | ゴールへの寄与 | 課題 |
|---|---|---|
| 学習の設定の値域の検証を 1 か所に置く | 範囲外の値を構築で止め、DefaultDQN と Rainbow で同じ検査を通す | eb0a0c |
| 方策の指標と IQN の診断の、未成立の条件 | 方策に使わない指標と、成り立たない比を NaN にする | c702d2・770358 |
| 行動の番号の読み取りと検査の分離 | fatal を 1 回にする | ea9fef |
| Learner が学習の進み（`learn_step`）と PER の β を持つ | State を使う側に置き、Agent の構築の引数と振り分けの一覧を減らす | 84c03f・6a8e46 |
| 選んだ行動の 1 本を取り出す共通の関数 | 13 か所の式と、3 通りの判定を 1 つにする | 9e8834 |
| probe の抽選の共通の関数 | 4 か所の手順を 1 つにし、使わない次の観測を通さない | 9f1801 |
| 価値の分解の診断 2 群の名前付きの並び | 並びと名前の食い違いでビルドを止める | 5ebf2f |
| Actor Q ヒントの入力を作る共通の関数 | Actor と step 診断で、Replay の初期優先度と同じ入力を作る | de8cfd |
| 内的の割引率の設定を返す関数 | 6 か所の選び方を 1 つにする | d74c3f |
| 可視化用の forward の特性テスト | 2 枚以上の stack の入力の作り方を固定する | e43866 |

### 1.5 成功の測り方

- §7 の受入がすべて通る。
- 写しが消えたことを、§7 の検索の受入（件数）で確かめる。
- 前後比較（§7 の A13）で、わざと変える 2 種類の tag（U3・U4）を除く全 scalar の点が一致する。

## 2. 前提

| 番号 | 種類 | 内容 | 出典 | 確かめ |
|---|---|---|---|---|
| P1 | 守る線 | 範囲外の値は `ANET_SYSTEM_ERROR` で止め、0 を別の意味に読み替えず、clamp もしない。使われない側の設定キーも、そのキーだけで決まる値域を検証する。エラー文には、キー・指定値・期待する範囲を書く | [AGENTS.md](../../AGENTS.md) の「設定値の扱い」（454 行） | 対象外 |
| P2 | 守る線 | DQN 系の Agent まわりの挙動を変えない直しでは、設定キー、メトリクスの名前、checkpoint の形式、ログと WARN・ERROR の文言・順・回数、乱数の消費の順、Agent が記録する設定を変えない。通知の文面では設定キーを完全キーで書き、RainbowAgent が読まない設定を完全キーの形で名指さない | [324 DQN のトピック別まとめ](../wiki/concepts/framework/324_dqn.md) の「変えるときの条件」（270・274 行） | 対象外 |
| P3 | 既存部品の働き | DQN の前後比較の組は、RFD0128 §6.4 の 34 構成（RFD0125 の 31 組、LunarLander と Atari qbert の RND、診断 6 群をすべて購読する DropMerge）。RFD0182 の `run_comparison.py` が、変更前の Debug Runner から採取し、scalar・設定の記録・WARN/ERROR を照合する。この組の検証資材は RFD0191 より前の設定キーで書かれている | [RFD0128](0128_dqn_agent_refactor_bundle_10rfd.md) §6.4、[RFD0182 の検証資材](../../src/core/base/testdata/rfd0182/README.md) の「学習挙動の比較」、[RFD0128 の検証資材](../../src/core/base/testdata/rfd0128/README.md) の 3 行 | 対象外 |
| P4 | 既存部品の働き | 学習器の更新の速さは、Release で、着手前に保存した実行体と直した後の実行体を交互に 3 巡以上流し、更新 1 回あたりの実時間の中央値が、直す前の最大値を超えないことで確かめる | [321 Agent 共通のトピック別まとめ](../wiki/concepts/framework/321_agent_core.md) の「変えるときの条件」（215 行。RFD0125 §4 D16・§6 A9） | 対象外 |
| P5 | 事実 | 設計 200 §2.4 は、方策の指標について「方策に使わない既知の指標は NaN となる」と書く | [設計 200](../design/200_dqn_agents.jp.md) の 253 行 | 対象外 |
| P6 | 事実 | 設計 200 §9.1 は、`iqn_policy_margin_mc_ratio` を「IQN+UQE 以外または `K < 2` では NaN」と書き、点の問い合わせを本数だけ複製する場合に触れない | [設計 200](../design/200_dqn_agents.jp.md) の 689〜696 行 | 対象外 |
| P7 | 既存部品の働き | scalar のメトリクスの Observer は、有限でない値を記録しない | [observers.cpp](../../src/core/base/src/observers.cpp) の `MetricsLogObserverBase::OnGenericUpdate`（1069 行）の 1121・1142 行 | 対象外 |
| P8 | 事実 | 出荷設定に、U1・U2 の 7 つの設定キーを範囲外にする行は無い | `git grep` で `apps/runner/config` を、`grad_clip_tau`・`td_clip_value`・`update_interval`・`quantile_huber_kappa`・`per_alpha`・`replay_ratio`・`replay_batch_size` で引いた（2026-11-27） | 対象外 |
| P9 | 過去の判断 | RFD0128 は、診断の指標を名前で指し、並びの食い違いをビルドで止める形（`MakeDiagnosticMetricKeys` ほか）を入れ、「名前を付けるだけで、食い違いをビルドで止めない形」を退けた | [RFD0128](0128_dqn_agent_refactor_bundle_10rfd.md) §5 D12・§5.1、[dqn_math.hpp](../../src/algo/dqn/src/dqn_math.hpp) の 35〜111 行 | 対象外 |
| P10 | 過去の判断 | step trace の近似優先度は、Actor と共有するヒント計算で作り、Replay の初期優先度と一致させる。Actor Q ヒントは常に 5 列で、価値の分解の Head ごとの値を運び、列の数をモードで変えない | ADR [0058](../adr/0058-approx-priority-column-computed-by-agent-step-diagnostics.md)・[0075](../adr/0075-actor-q-hint-five-columns-carry-per-head-values.md)、[321 Agent 共通のトピック別まとめ](../wiki/concepts/framework/321_agent_core.md) の 202 行（RFD0094 §4.2 D6・D7） | 対象外 |
| P11 | 過去の判断 | probe 用の乱数は、Agent の基の seed と乱数の名前から Learner が構築時に作り、購読を設定し直しても作り直さない。これを守れば、診断の値も学習の値も変わらない | [321 Agent 共通のトピック別まとめ](../wiki/concepts/framework/321_agent_core.md) の 219 行（RFD0128 §5 D13） | 対象外 |
| P12 | 事実 | RFD0200 は実装済み（`b3e29497`）。可視化用の依頼（`PerceptionFromFrame`）の Forwarder は、同じ 1 枚を枚数ぶん並べる。重ねる設定は `frame_stack.*`（`keys.mode` が `all` か `listed`）である | [RFD0200](0200_forwarder_public_interface_rework_10rfd.md) の P5・D28、[default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) の 579〜583 行、[dqn_concrete_test.cpp](../../src/algo/dqn/src/dqn_concrete_test.cpp) の 593〜601 行 | 対象外 |
| P13 | 事実 | `ANET_ASSERT_SHAPE` は既定で有効、`ANET_ASSERT_NAN` は `DEBUG` のときだけ有効になる | [tensor_check.hpp](../../src/core/base/include/anet/tensor_check.hpp) の 21〜31 行 | 対象外 |
| P14 | 事実 | RFD0001 は、`quantile_huber_kappa` の検査を IQN に限り、「QR は既存挙動のまま検証追加しない」と書いた。IQN の RFD の範囲の線で、QR で κ=0 を使う意図の記述ではない | [RFD0001](done/0001_iqn_10rfd.md) の 104 行、[default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp) の 805 行 | 対象外 |

## 3. 現状（2026-11-27 に確かめた）

行番号は 2026-11-27 の作業ツリー（`48c98707`）のものである。課題の本文の行番号は、その後のコミットでずれている。

### 3.1 学習の設定の値域（eb0a0c）

- DefaultDQN は `ReadLearner`（[default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp) 462 行）で 3 つのキーを読む（480・482・486 行）が、値域を検証しない。同じ関数は `noisy_weight_decay`・probe の batch size・`tbo_epsilon` を `ValidateRange` で検証する（474・520〜548 行）。`ValidateRange` のエラー文は `Invalid <完全キー>=<値>; expected <期待>.` の形である（161 行）。Rainbow は [rainbow_agent.hpp](../../src/algo/dqn/src/rainbow_agent.hpp) の 55〜62 行で同じキーを検証せずに読む。`LearnerConfig`（[dqn_config.hpp](../../src/algo/dqn/src/dqn_config.hpp) 247 行）は、全体の検証の関数を持たない。
- 使う側: 勾配の clip は [dqn_learner.cpp](../../src/algo/dqn/src/dqn_learner.cpp) の 1160〜1353 行の 9 か所で `ForeachClipGradNorm_` に `grad_clip_tau` を渡す。この関数は勾配に `(tau / (norm + 1e-6)).clamp_max(1.0)` を掛ける（[nn_util.cpp](../../src/core/base/src/nn_util.cpp) 188 行）。TD の clip は `use_td_clip && td_clip_value > 0.0f` のときだけ当てる（dqn_learner.cpp 2528 行）。間隔のモードの credit は `1.0f / std::max(1, update_interval)`（262 行）。
- 課題の 2026-11-25 の追記の 4 つ: `quantile_huber_kappa` の検査は IQN のときだけ（default_dqn_agent.hpp 805 行）。`per_alpha` は `ValidateReplayPriorityConfig`（[dqn_math.cpp](../../src/algo/dqn/src/dqn_math.cpp) 240 行）が 0 の WARN（262 行）だけを出し、負と NaN を見ない。`replay_ratio` は `> 0` でなければ間隔のモードに入り（dqn_learner.cpp 257〜263 行）、0 も負と同じに扱う。`replay_batch_size` が 0 なら RR のモードの credit が +inf になる（259 行）。
- 範囲外を通すテストと出荷設定は無い（P8）。

### 3.2 方策の `uqe_tau`（c702d2）

- `ActionPolicy::GetScalar`（[dqn_actor.cpp](../../src/algo/dqn/src/dqn_actor.cpp) 876 行）の `uqe_tau` は `thompson_ ? kNaN : current_uqe_tau_`（882 行）で、`uses_uqe_` を見ない。同じ関数の `uqe_epsilon`（883 行）と `uqe_tau_lane_min`・`_max`（888・889 行）は `uses_uqe_` を見る。`current_uqe_tau_` の初期値は 0（[dqn_actor.hpp](../../src/algo/dqn/src/dqn_actor.hpp) 157 行）で、spatial のときだけ NaN にする（dqn_actor.cpp 669 行）。
- 読む tag: `metrics_scalar.txt` の `@baseline`・`@full` の `[32_agent_base/04_uqe_tau] ?= $actor uqe_tau`（30・206 行）、DropMerge の `[32_agent_base/04_uqe_tau_target]`（DropMerge.txt 1371 行）、GridMaze の同じ tag（GridMaze.txt 228 行）。
- テストは UQE の spatial の平均だけを見る（[dqn_actor_test.cpp](../../src/algo/dqn/src/dqn_actor_test.cpp) 1804 行。検索語 `"uqe_tau"` でテストに 1 件）。

### 3.3 点の UQE の比（770358）

- `MakeUQEActionInfo` は、点の UQE で全範囲の問い合わせが無いとき、同じ τ を `tau_rule.num_taus` 本に複製する（dqn_actor.cpp 1128〜1129 行の spatial、1138〜1142 行の scalar）。比の成立の条件は `risk_num_taus >= 2`（1231 行）だけで、分母は `std + 1.0e-6f`（1240 行）になる。
- 読む tag: LunarLander の `[33_agent_action/21_iqn_policy_margin_mc_ratio_ema]`（LunarLander.txt 448 行）、DropMerge の 3 本（DropMerge.txt 1372・1452・1484 行）。

### 3.4 fatal の重複（ea9fef）

- `ParseActionScalarIndex`（dqn_actor.cpp 238 行）は、`try` の中で読み残しを見つけると `ANET_SYSTEM_ERROR` を呼び（256〜258 行）、すぐ後ろの `catch (const std::exception&)` が同じ文でもう一度呼ぶ（259〜261 行）。

### 3.5 `RuntimeVars` と値の振り分け（84c03f・6a8e46）

- `RuntimeVars`（[dqn_math.hpp](../../src/algo/dqn/src/dqn_math.hpp) 225〜229 行）の欄は `uqe_tau`（読み書き 0）、`learn_step`、`per_beta` の 3 つ。`learn_step` の読み書きは dqn_learner.cpp だけ（518・632・682・750・846・1811・1815〜1817・1832・1864・1870・2413・2442 行）。`per_beta` は Learner が書き（1016・1672 行）、Learner（1791 行）と振り分け（[dqn_assembler.cpp](../../src/algo/dqn/src/dqn_assembler.cpp) 411 行）が読む。
- 検索語 `RuntimeVars`: テストを除く `src/` で 28 行・11 ファイル、テストで 83 行・5 ファイル（`dqn_actor_test.cpp`・`dqn_concrete_test.cpp`・`dqn_learner_test.cpp`・`dqn_math_test.cpp`・`dqn_munchausen_test.cpp`）。`dqn_learner_test.cpp` の `learn_step` の読み出しは 8 行（186・250・2745・3325・3488・3489・3896・4277 行）、`per_beta` の読み出しは 1 行（4378 行）。`dqn_actor_test.cpp` の 1913 行は `learn_step` への代入。
- 文書: 設計 200 の jp・en に 7 行ずつ（jp は 292・333・358・359・631・633・658 行）、[RFD0143](0143_serialize_10rfd.md) に 2 行（18・68 行）。ADR 0048 と過去の実装記録にもあるが、過去の記録なので直さない。
- 振り分け: `DqnComponentAssembler::GetScalar`（dqn_assembler.cpp 391〜419 行）は、`target_policy.` の接頭辞（ロックなし）、schedule の 6 つ、分解の 9 つ、`per_beta`、接頭辞 `plasticity_probe_`・`noisy_`・`replay` を自分で判定してから、共有ロックを取って `learner.GetScalar` を呼ぶ。`Learner::GetScalar`（dqn_learner.cpp 276〜319 行）は同じ群をもう一度判定し、知らないキーには `nullopt` を返す。`GetTensor`・`GetTensorVector`（dqn_assembler.cpp 421〜435 行）も `replay` の接頭辞を判定するが、`Learner::GetTensor`・`GetTensorVector`（dqn_learner.cpp 851〜865 行）も同じ判定をして、知らないキーには `nullopt` を返す。

### 3.6 選んだ行動の値の取り出し（9e8834）

- 13 か所: dqn_learner.cpp の 594〜595・603〜604 行（`EvaluateReplayFit`）、2223〜2227 行（`GatherActionQuantiles`）、2500〜2501 行（`ComputeSkipLoss`）、2612〜2613 行（`MakeTarget`）、2656〜2657 行（`MakeDecomposedTargets`）、2788〜2789 行（`UpdateFromSamples`）、2885〜2886 行（`ForwardCurrent`）。[dqn_network_model.cpp](../../src/algo/dqn/src/dqn_network_model.cpp) の 336〜338・342〜343 行（`ReadSkipValues`）。dqn_actor.cpp の 111・132・145 行（`MakeActorQHint`）。
- 検索語 `1, 1 ?\}\)\.expand\(`（正規表現）で、dqn_learner.cpp と dqn_network_model.cpp に 9 行。
- 分布を持つかの判定は、設定の文字列（dqn_learner.cpp 591・2647 行ほか）、tensor の軸数（2787 行）、出力の鍵の有無の 3 通りある。設定の文字列は、どの出力の鍵を読むか（`q` か `q_dist` か）の選択にも使っている。
- `GatherActionQuantiles` だけが `ANET_ASSERT_SHAPE` と `ANET_ASSERT_NAN` を持つ（2224〜2229 行）。

### 3.7 probe の抽選（9f1801）

- 4 か所が同じ順の行を持つ: `CaptureNoisyDiagnostics`（dqn_learner.cpp 650〜653 行）、`CaptureAttentionEntropy`（689〜693 行）、`CapturePlasticityProbe`（726〜730 行）、`CapturePolicyChurnProbe`（760〜764 行）。どれも `.obs` だけを使う。`MakeSamplePerception`（1046〜1056 行）は `next_obs` も Forwarder に通す。
- IQN の等間隔の τ は、Noisy（655〜658 行）と policy churn（766〜772 行）が手で書き、`MakeTauMidpointPositions`（dqn_math.cpp 134 行。`(arange(N) + 0.5) / N`）と同じ式である。
- 検索語 `SampleUniqueUniform(` で dqn_learner.cpp に 4 行。

### 3.8 価値の分解の診断の並び（5ebf2f）

- 学習の群: キーは手書きの `std::array<const char*, 9> kValueDecompositionMetricKeys`（dqn_math.hpp 273〜275 行）。値は `torch::stack` の 9 要素（dqn_learner.cpp 2860〜2863 行）で、並びを手で合わせている。読み出しは `Learner::GetScalar` の専用のループ（278〜283 行）。
- 行動選択の群: キーの配列 3 本（dqn_actor.cpp 37〜44 行。lane・平均・session）、値は `torch::stack({disagreement, regret, external_gap, internal_gap}, 1)`（355 行）、読み出しは 2 か所の専用のループ（360〜376 行と 1729〜1734 行）。
- step 診断は schedule の指標を番号で指す: `scalar(3)`（dqn_assembler.cpp 117 行）、`scalar(4)`・`scalar(5)`（134 行）。名前付きの番号 `LearnerScheduleMetric` は dqn_math.hpp 113〜115 行にある。
- 検索語と件数: `kValueDecompositionMetricKeys` は `src/` に 4 行、`kValueSelection(Lane|Mean|Session)Keys`（正規表現）は 6 行。

### 3.9 Actor Q ヒントの入力（de8cfd）

- Actor は dqn_actor.cpp 1584〜1602 行で、`q_values` の検査、Munchausen の値、TBO、分解なら aux の 4 つの取り出し、外的の bootstrap を組んで `MakeActorQHint` を呼ぶ。step 診断は dqn_assembler.cpp 127〜148 行で、同じ検査と取り出しを別の文で書き、分解なら `MakeActorQHint` を 2 回呼ぶ（136・146 行）。1 回目の結果は要素ごとの `item<float>()`（163 行）、2 回目は accessor（167 行）で読む。
- 外的の bootstrap への変換は、[default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 938 行と dqn_assembler.cpp 145 行の 2 か所にある。
- 検索語 `MakeActorQHint(` で `src/` に 7 行（定義 1、宣言 1、本番の呼び出し 3、テスト 2）。

### 3.10 内的の割引率の選び方（d74c3f）

- 6 か所: default_dqn_agent.cpp 558 行（報酬スケーラへ渡す割引率。decomposed を見る）、939 行（`ActorQHintConfig::intrinsic_gamma`。decomposed を見ない）、dqn_assembler.cpp 126 行と 181〜182 行（step 診断の予定の有無と保存済みの収益の利用）、dqn_learner.cpp 993 行（Replay の収益の作り直し。decomposed の分岐の中）、2631 行（`ComponentGammas`。2630 行で decomposed を見る）。
- 今の読み手は decomposed でないとき内的の割引率を読まない（Actor は decomposed のときだけ評価する。dqn_actor.cpp 1580 行）ので、条件の食い違いは今の結果を変えていない。
- 検索語 `kDiscount(Separate|Shared)`（正規表現）は、テストを除く `src/` に 12 行（定義、default_dqn_agent.hpp の検証 251・296・322 行、上の 6 か所）。

### 3.11 可視化用の forward（e43866）

- `DefaultDQNAgent::GetTensorDictFunction`（default_dqn_agent.cpp 846 行）は、可視化用の Forwarder（`visualization_perception_`。583 行で作る）に観測を通してから forward する（P12）。
- 重ねる枚数が 1 のときの一致は、[dqn_concrete_test.cpp](../../src/algo/dqn/src/dqn_concrete_test.cpp) の 327〜341 行が確かめる。`keys.mode` の `all`・`listed` は、593 行のテストが行動選択の経路だけで確かめる。2 枚以上の stack で `GetTensorDictFunction` を通すテストは無い（検索語 `GetTensorDictFunction` と `frame_stack` でテストを引いた）。基盤の層では、[forwarder_test.cpp](../../src/core/base/src/forwarder_test.cpp) の 411〜422 行が、可視化用の依頼で 1 枚が枚数ぶん並ぶことを確かめる。

## 4. 未決表

| 番号 | 論点 | 暫定の既定 | 理由と依存 |
|---|---|---|---|
| U1 | eb0a0c の 3 つの値域と置き場 | `learner.grad_clip_tau` と `learner.td_clip_value` は「有限で 0 より大きい」、`learner.update_interval` は「1 以上」。検証は `LearnerConfig` の 1 関数に置く。DefaultDQN と Rainbow は完全キーを設定したあと、設定の読み取りから呼ぶ。設定型を直接受け取る構築も、共通 Learner の構築で同じ検証を通す。`use_grad_clip`・`use_td_clip` が偽のとき、`replay_ratio` が正のときも検証する。エラー文は `ValidateRange` と同じ形にし、キーは完全キーで書く。dqn_learner.cpp 2528 行の `td_clip_value > 0.0f` の条件は消す（検証のあとは常に真で、残すと 0 で clip を切れるように読めるため） | P1（使われない側も検証する、sentinel を作らない）、P2（完全キー、Rainbow が読まないキーを名指さない。Rainbow が読まないキーは型の既定の値なので、範囲に入り、文に出ない） |
| U2 | eb0a0c の 2026-11-25 の追記の 4 つ | 課題の事象に含まれるので範囲に入れる。`learner.quantile_huber_kappa` は「有限で 0 より大きい」を `quantile_mode` によらず検証し、今の IQN だけの検査（default_dqn_agent.hpp 805 行）を置き換える。`learner.per_alpha` は「有限で 0 以上」（0 は受理し、`use_per` が真かつ `per_initial_priority_mode=actor_approx` の既存 WARN は維持）。`learner.replay_ratio` は「有限で 0 でない」（正は RR のモード、負は間隔のモードのまま）。`learner.replay_batch_size` は「1 以上」。置き場とエラー文は U1 と同じ | P1。課題が片付くには事象の全部を直す（ゴール）。P14 の RFD0001 の線は IQN の RFD の範囲で、QR で κ≤0 を使う意図ではない。`replay_ratio` の負で間隔を選ぶ形は今のまま残し、改めるのは範囲外（§1.3、U14） |
| U3 | c702d2 の値 | `uqe_tau` は、`uses_uqe_` が真で Thompson でないときだけ `current_uqe_tau_` を返し、それ以外は NaN を返す（`uqe_tau_lane_min`・`_max` と同じ条件）。Greedy・spatial なしの EpsilonGreedy・spatial ありの EpsilonGreedy の 3 つで NaN になるテストを足す | P5（方策に使わない指標は NaN）、P7（NaN は記録されない）。学習と乱数に触れない |
| U4 | 770358 の未成立の条件 | 比を計算する条件を「IQN で、risk の問い合わせが点でない（`use_tail_mean` が真か、非 spatial Thompson の行動選択用の全範囲の枝）、かつ本数が 2 以上」にする。行動を決める問い合わせが点の場合、可視化用の別の全範囲の問い合わせの有無にかかわらず NaN。設計 200 §9.1（jp・en）の NaN の条件に「点の問い合わせ」を足す。ユーザーガイド 030 の `iqn_policy_margin_mc_ratio` の説明（jp 187 行・en 176 行）に「点の UQE では出ない」を 1 句足す。点で K=3・全範囲の問い合わせ無しの節（dqn_actor_test.cpp 943〜966 行）に、比が NaN であることの確認を足す | P6、P7。forward・乱数・行動選択に触れない。030 は利用者が値を読む手引きなので、値が出ない構成を書かないと読み違える |
| U5 | ea9fef の直し | `try` の中では数の読み取りだけを行い、読み残しの検査と `ANET_SYSTEM_ERROR` を `try` の外へ出す。文言は変えない。不正な番号（`[1x]` と `[x]`）で、例外が 1 回だけ出て、fatal のログが 1 行だけ出ることのテストを足す | P2（文言を変えない） |
| U6 | 84c03f の State の置き場 | `learn_step` と `per_beta` を `dqn::Learner` のメンバにし、`RuntimeVars` の型と、Learner・`DqnComponentAssembler`・DefaultDQN・Rainbow の構築の引数を消す。`per_beta` は `Learner::GetScalar` で答え、値は今 Agent が返す値と同じにする（`use_per` でないとき 0、評価の時機は前回の更新の終わりに次回分、初期値は step 0）。テストが学習の進みを読むために、`dqn::Learner` に `learn_step` を読むだけの関数を 1 つ足す（DQN 系の内部のヘッダ。基盤の公開ヘッダは変えない）。テストは構築の引数を消し、`vars.learn_step` を読む 8 行をこの関数に、`vars.per_beta` を読む 1 行を `GetScalar("per_beta")` に替える。`dqn_actor_test.cpp` 1913 行の `vars.learn_step = 1` は、公開の更新経路から学習を 1 回進める。既存テストの Replay・warm-up・購読の準備を調整し、既存の同期の期待値を通す。設計 200（jp・en）の 7 行ずつと、RFD0143 の 2 行を、Learner が持つ形に直し、RFD0143 の冒頭に改訂の行を足す | 課題の対応方針のとおり。P2（メトリクスの名前と値を変えない）。RFD0143 は未実装の RFD で、学習の進みの保存の置き場を書くので、今の形と合わせる。過去の記録（ADR 0048、実装記録）は直さない |
| U7 | 6a8e46 の振り分け | `DqnComponentAssembler::GetScalar` は、`target_policy.` の接頭辞を今のまま（ロックなし）先に扱い、それ以外は共有ロックを取って `learner.GetScalar(key, index)` を呼び、値があれば返す。群の一覧と `per_beta` の分岐を消す。`GetTensor`・`GetTensorVector` の `replay` の接頭辞の判定も消し、Learner に任せる。`Learner::GetScalar` の中で群を引く順は変えない | U6 のあと（`per_beta` を Learner が答える）。P2。課題が書くとおり、DefaultDQN は今もどの群にも当たらないキーで最後に同じロックを取るので、ロックの回数の差は読み出しごとに 1 回で、性能には関わらない。ロックの範囲の見直しは 2d66d2 に残す（§1.3） |
| U8 | 9e8834 の共通の関数 | 「行動（または継続長）ごとの値と、選んだ番号から、行動の軸で選んだ 1 本を取り出し、行動の軸を除いた値を返す」関数を `dqn_math` に 1 つ置き、§3.6 の 13 か所が使う。行動の軸より後ろの軸（分位点など）の有無は値から導く（分布の有無の判定を値の軸の数にそろえる）。どの出力の鍵を読むか（`q` か `q_dist` か）の選び方は今のまま設定から決める。形の検査（`ANET_ASSERT_SHAPE`）は共通の関数に置く。NaN の検査（`ANET_ASSERT_NAN`）は、今ある経路（`GatherActionQuantiles`）でだけ当て、ほかの 12 か所には足さない | 課題の対応方針のとおり。NaN の検査を 13 か所へ広げると、Debug の実行で、今は止まらない経路（診断の値に NaN を含みうるもの）が止まりうる（P13）。課題の方針の「検査を共通の関数に移す」から、NaN の検査だけを外した |
| U9 | de8cfd の共通の関数と範囲 | 「行動の補助出力と、Munchausen の値、TBO、価値分解の有無、外的の bootstrap 設定から、ヒントの入力値を組み立てる」関数を 1 つ置き、Actor と step 診断が使う。外的の bootstrap への変換もこの関数に入れ、2 か所の写しを消す。step 診断が分解のとき呼ぶ 2 回のうち、1 回目（合わせた Q の `q_sa`・`state_value` を列に出すためだけ）は、入力を作る関数が合わせた Q の値を別の出力として 1 回の計算で返す形にして、`MakeActorQHint` の呼び出しを 1 回にする。合わせた値は、今の 1 回目の結果の該当の列と同じ値にし、テストで確かめる。ヒント設定の構成を変える fa1be2 は範囲に入れない。c3b5ef（n 先のヒントを待つ手順の写し）は範囲に入れない | 課題の対応方針のとおり。P10（Replay の初期優先度と同じ入力、列の数は変えない）。fa1be2 は設定の構造体を分ける別の判断（§1.3）。分解のときの 2 回目の結果の Q_sa の列は外的の Head の値で、合わせた Q の値ではない（dqn_assembler.cpp 163〜170 行）。合わせた値を同じに作るのに Munchausen・TBO の扱いを `MakeActorQHint` の外へ写すことになるなら、写しを増やすので、2 回の呼び出しを残し、入力の組み立てだけをまとめる（その理由を実装計画に書く） |
| U10 | d74c3f の関数と、decomposed でないときの扱い | `LearnerConfig` に「内的の成分が使う割引率の定数または予定を選ぶ」関数を 1 つ置く。decomposed かつ `discount_mode` が `separate` なら `learner.intrinsic.gamma`、それ以外（decomposed でない、または `shared`）なら `learner.gamma` を返す。6 か所はこの関数を呼び、予定の有無は、返した設定が定数かどうかで判定する。decomposed の分岐の中にある呼び出しは、分岐をそのまま残す | default_dqn_agent.cpp 558 行の今の式と同じ選び方で、ほかの 5 か所の今の結果も変えない（§3.10 のとおり、decomposed でないとき内的の割引率を読む読み手が無い）。decomposed でないとき外的の割引率を返す形にすると、Ignored と知らせた `learner.intrinsic.gamma`（default_dqn_agent.hpp 322 行）が効く道が無くなる |
| U11 | 5ebf2f の並びの形 | 学習の群と行動選択の群に、`LearnerScheduleMetric` と同じ形の名前付きの番号（`ValueDecompositionMetric`・`ValueSelectionMetric`）を置き、キーを `MakeDiagnosticMetricKeys` で、値を `StackDiagnosticValues` で作り、読み出しを `ReadDiagnosticScalar` に寄せる。行動選択の群は lane × 指標なので、列を並べる版を `StackDiagnosticValues` の隣に 1 つ足す（lane・平均・session の 3 本のキーは、同じ番号から作る）。step 診断の `scalar(3)`・`scalar(4)`・`scalar(5)` は `LearnerScheduleMetric` の名前で指す。キーの名前、NaN になる条件、`$agent`・`$action_info`・`$actor` のどこから読めるかは変えない | P9（並びの食い違いをビルドで止める）。課題の対応方針の 2 案（列ごとの版を足すか、`MakeDiagnosticValues` で列を並べるか）のうち、行動選択の群の 3 本のキーを同じ番号で作れる前者を置いた。どちらでも値は変わらない |
| U12 | 9f1801 の共通の関数 | 「一様に B 件を引いて device へ移し、件数を確かめ、観測（`obs`）だけを Perception に通して返す。揃わなければ空を返す」関数を、`dqn::Learner` の中だけで使う関数として 1 つ置き、4 群が呼ぶ。次の観測（`next_obs`）は通さない。等間隔の τ は `MakeTauMidpointPositions` に寄せる。probe ごとの乱数と、引く順は変えない | P11。標準の Forwarder の部品 4 種（FrameStack・Normalize・DropKeys・SetSpec）は、Sample 用の forward で乱数を引かず統計を更新しないので、次の観測を通さなくても値は変わらない（課題の対応方針）。アプリが足した部品が forward で乱数を引く構成だけは列が変わるが、出荷設定にその部品は無い |
| U13 | e43866 の特性テスト | RFD0200 の実装のあとの口で書く。`frame_stack.count` を 2 以上にした DefaultDQNAgent で `GetTensorDictFunction("policy-net.forward")` に 1 枚の観測を渡し、(1) 止まらずに forward が通ること、(2) Network へ渡る入力の形（stack の対象のキーは長さ `count` の stack の軸を持ち、対象外のキーは元の形）が `keys.mode` の `all` と `listed` の 2 通りで期待どおりであること、(3) stack の各位置が同じ値であること、を確かめる。入力の見方（capture、Network の入力の spec、同じ値の入力で直接 forward した出力との一致など）は実装で選ぶ | P12。課題の最後の追記（RFD0200 のあとは新しい口で書く）のとおり。既存の 593 行のテストの設定（`MakeNativeStackConfigData`、`keys.listed.names`）を使える |
| U14 | 範囲外の候補 | `learner.replay_ratio` の負の値で間隔のモードを選ぶ形を、モードの設定キーに改める仕事を、新しい課題の候補にする（工程末の起票の候補で開発者が選ぶ） | §1.3。P1 の「意図の分岐はモードの値として明示する」に今の形が当たらない |
| U15 | 課題どうしの実装の順 | e43866（テストだけ）→ eb0a0c → ea9fef → c702d2 → 770358 → 84c03f → 6a8e46 → 5ebf2f → d74c3f → de8cfd → 9e8834 → 9f1801 の順にする | 6a8e46 は 84c03f のあと（U7）。de8cfd は、step 診断の `scalar(4)`・`scalar(5)` を名前にする 5ebf2f と、`ActorQHintConfig::intrinsic_gamma` を直す d74c3f のあとにすると、同じ行を 2 度直さない。9e8834 は `MakeActorQHint` の中を直すので de8cfd のあと。挙動を変える 2 件（c702d2・770358）を前半に置き、前後比較の例外を先に確かめる |

## 5. 仕様

### 5.1 学習の設定の値域（U1・U2）

当て方: キーごとに独立に 1 行で扱う。設定の読み取り後に検証する。設定型で直接渡された値も、共通 Learner の構築時に検証する。設定ファイルの値が空、数として読めない、整数のキーに小数、float のキーに NaN・inf のときは、今ある設定の読み取りが表の検証より先に止める（[config.cpp](../../src/core/base/src/config.cpp) の `ParseNumericConfigValue`、61〜86 行。本書は変えない）。表の「有限」の検証は、設定ファイルを通らずに値を組む構成（テストなど）にも同じ線を引くために置く。

| キー | 型 | 受け付ける値 | 外れたとき | 使われないときも検証するか |
|---|---|---|---|---|
| `learner.grad_clip_tau` | float | 有限で 0 より大きい | `Invalid <完全キー>=<値>; expected finite value > 0.` で止める | する（`use_grad_clip = false` でも） |
| `learner.td_clip_value` | float | 有限で 0 より大きい | 同上 | する（`use_td_clip = false` でも） |
| `learner.update_interval` | int | 1 以上 | `...; expected integer >= 1.` で止める | する（`replay_ratio` が正でも） |
| `learner.quantile_huber_kappa` | float | 有限で 0 より大きい | `...; expected finite value > 0.` で止める | する（`quantile_mode` によらない） |
| `learner.per_alpha` | float | 有限で 0 以上（0 の既存 WARN は PER 有効かつ初期優先度 `actor_approx` の場合に出す） | `...; expected finite value >= 0.` で止める | する（`use_per = false` でも） |
| `learner.replay_ratio` | float | 有限で 0 でない。正は RR のモード、負は間隔のモード | `...; expected finite value > 0, or < 0 to use update_interval.` で止める | — |
| `learner.replay_batch_size` | int | 1 以上 | `...; expected integer >= 1.` で止める | する |

境界の例は表のとおり。正の float を要求する 3 キーでは、非有限値・0・負を拒否する。`per_alpha` では非有限値・負を拒否し、0 は受理する。PER が有効で初期優先度が `actor_approx` の場合は既存 WARN を出す。`replay_ratio` では非有限値・0 を拒否し、正と負を受理する。整数の 2 キーは 0・負を拒否し、1 を受理する。期待の文の言い回しは、今ある `ValidateRange` の呼び出し（`"finite value >= 0"`、`"integer >= 1"` など）にそろえる。RainbowAgent が読まないキーは型の既定の値（範囲の中）のままなので、エラー文に出ない（P2）。

### 5.2 診断値の未成立（U3・U4）

| 指標 | NaN になる条件（今） | NaN になる条件（本書のあと） |
|---|---|---|
| 方策の `uqe_tau` | Thompson、または spatial の EpsilonGreedy | UQE を使わない方策、または Thompson |
| `iqn_policy_margin_mc_ratio` | IQN の UQE・Thompson の経路以外、または risk の本数が 2 未満 | IQN の UQE・Thompson の経路以外、行動を決める問い合わせが点、または risk の本数が 2 未満。非 spatial Thompson の行動選択用の全範囲は、2 本以上なら成立 |

NaN の点は記録されない（P7）。

## 6. 実装契約

### 6.1 変えるもの

§4 の U1〜U13 の既定のとおり。触るファイルは `src/algo/dqn/src` の本体とテスト、設計 200・030（jp・en）、RFD0143 である。

### 6.2 変えないもの（本当の制約か）

- 設定キー、メトリクスの名前、checkpoint の形式、ログと WARN・ERROR の文言・順・回数（新たに拒否する設定値の ERROR と、不正な指標キーで重複していた fatal の削除を除く）、乱数の消費の順、Agent が記録する設定（P2）。本当の制約ではないが、挙動を変えない直しの線として守る。わざと変えるものは §1.2 の 4 つ。
- probe で使う標準 Forwarder の Sample 経路では乱数と統計を変えない。独自部品が次の観測の forward で状態を更新する構成は、この等価性の対象外とする（U12）。
- 基盤（`src/core/base`）の公開ヘッダ。本書の型と関数はどれも `src/algo/dqn/src` の内部のヘッダに置く。本当の制約ではない。基盤の口を変える理由が無い。
- Evaluation Trace の列と Actor Q ヒントの列の数と値（P10）。本当の制約ではない（ADR 0075 を改める判断で変えられる）。本書は写しをまとめるだけで、列を変える理由が無い。
- 出荷設定（`apps/runner/config`）。本当の制約ではない。範囲外の値を書く行が無い（P8）ので、直す行が無い。

### 6.3 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。ただし、一覧に無い機能グループを新しく立てるときは、作る前に開発者に確かめる。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| RuntimeVars | 削除 | （`learn_step`・`per_beta` は dqn::Learner へ移す。`uqe_tau` は読み書きが無い） | DQN 系の共通部品（dqn_math） |
| dqn::Learner | 変更 | 学習の進み（`learn_step`）と PER の β を持ち、β を `GetScalar` で答える。学習の進みを読む関数を持つ。probe の抽選の共通の手順を、中だけで使う関数として持つ | DQN 系の共通部品（dqn_learner） |
| DqnComponentAssembler | 変更 | 値の読み出しを、`target_policy.` の接頭辞のほかは Learner に任せる。step 診断のヒントの入力を共通の関数で作る | DQN 系の共通部品（dqn_assembler） |
| LearnerConfig | 変更 | 学習の設定の値域の検証を 1 か所に持つ。内的の成分が使う割引率の設定を返す | DQN 系の共通部品（dqn_config） |
| ActionPolicy | 変更 | UQE を使わない方策の `uqe_tau` を NaN にする | DQN 系の共通部品（dqn_actor） |
| UQEActionPolicy | 変更 | 点の問い合わせで `iqn_policy_margin_mc_ratio` を NaN にする | DQN 系の共通部品（dqn_actor） |
| ValueDecompositionMetric | 新規 | 価値の分解の学習の診断 9 つの名前付きの番号 | DQN 系の共通部品（dqn_math） |
| ValueSelectionMetric | 新規 | 価値の分解の行動選択の診断 4 つの名前付きの番号 | DQN 系の共通部品（dqn_actor） |
| DefaultDQNAgentConfig | 変更 | 学習の設定の値域を `LearnerConfig` の検証で確かめる | DefaultDQNAgent（default_dqn_agent） |
| RainbowAgentConfig | 変更 | 同上 | RainbowAgent（rainbow_agent） |
| DefaultDQNAgent | 変更 | `RuntimeVars` を持たない | DefaultDQNAgent（default_dqn_agent） |
| RainbowAgent | 変更 | `RuntimeVars` を持たない | RainbowAgent（rainbow_agent） |

新しい機能グループは立てない。選んだ行動の 1 本を取り出す関数、ヒントの入力を作る関数、行動選択の群の列を並べる関数は、それぞれ dqn_math・dqn_actor の既存の機能グループに置く（名前は実装で決める）。

### 6.4 移行対象

経路の検索（2026-11-27、`git grep`、追跡しているファイルの全体）の検索語と件数:

- `RuntimeVars`: テストを除く `src/` 28 行・11 ファイル、テスト 83 行・5 ファイル、設計 200 jp・en 7 行ずつ、RFD0143 2 行。過去の記録（ADR 0048、`docs/memo` の実装記録、課題）は直さない。
- `kValueDecompositionMetricKeys` 4 行、`kValueSelection(Lane|Mean|Session)Keys` 6 行、`scalar\([0-9]\)` は dqn_assembler.cpp に 2 行（3 つの呼び出し）。
- `MakeActorQHint(` 7 行（本番の呼び出し 3、テスト 2 は `dqn_actor_test.cpp` 1624 行と `dqn_concrete_test.cpp` 3634 行で、関数の口を変えないので直さない）。
- `kDiscount(Separate|Shared)` 12 行（定義と default_dqn_agent.hpp の検証 3 行は残す）。
- `SampleUniqueUniform(` は dqn_learner.cpp に 4 行。
- 9e8834 の 13 か所（§3.6）。
- 文書: 設計 200 jp・en の §2.4 は今の本文のままで合う（P5）。§9.1 の NaN の条件（jp 696 行、en 696 行）、ユーザーガイド 030 の `iqn_policy_margin_mc_ratio` の行（jp 187 行、en 176 行）を直す（U4）。wiki のトピック別まとめは道具が作り直す生成物の節を持つが、本書は手で直さない（実装のあとの取り込みで扱う）。
- 振る舞いが変わる今のテストと設定: 範囲外の値を書くテストと出荷設定は無い（P8、課題の検索）。`uqe_tau` を見るテストは UQE の spatial の 1 件だけで、値は変わらない。`iqn_policy_margin_mc_ratio` を見る点の K=1 のテスト（dqn_actor_test.cpp 1015〜1069 行）は、今も NaN を期待するので変わらない。

### 6.5 ほかの RFD・課題との順序

- 本書は、RFD0205（文書）と RFD0206（基盤とアプリ）のあとに実装する前提で書いた。前後比較（§7 の A13）と速さの比べ（A14）の基準の実行体は、本書の変更を入れる直前の作業ツリー（RFD0206 の変更を含む）から作る。RFD0206 の変更が比較の差に混ざらないようにするためである。RFD0205 は設計 200（jp・en）の別の行（checkpoint の説明）を直すので、本書の U4・U6 の行と同じファイルを触る。順に入れれば重ならない。
- 同じファイルを触る、まだ実装していない RFD（`docs/memo` を、本書が触るファイル名と `RuntimeVars`・`ActorQHint`・`ComponentGammas` で引き、指す課題の状態が解決でないもの）: [RFD0107](0107_noisy_gain_lane_ladder_10rfd.md)、[RFD0134](0134_network_lock_audit_10rfd.md)、[RFD0136](0136_best_checkpoint_10rfd.md)、[RFD0142](0142_agent_prev_action_obs_10rfd.md)、[RFD0143](0143_serialize_10rfd.md)、[RFD0169](0169_outside_core_extension_layout_10rfd.md)、[RFD0171](0171_replay_eviction_sample_count_10rfd.md)、[RFD0173](0173_intrinsic_reward_novelty_10rfd.md)、[RFD0201](0201_top_episode_memory_sil_lower_bound_10rfd.md)、[RFD0191](0191_rnd_raw_error_and_recent_learning_10rfd.md)（課題 1 件が対応中）。どれも 2026-11-27 の作業ツリーに未コミットの変更を持たない。本書を先に入れる。あとの RFD は本書のあとのコードに対して書く。RFD0143 の `RuntimeVars` の記述は、本書が直す（U6）。RFD0201 が学習の指標を足すときは、U11 の名前付きの番号に足す。RFD0134 のロックの範囲の見直しは、U7 の形（Assembler は `target_policy.` の接頭辞をロックなしで扱い、それ以外を 1 回の共有ロックで Learner に委ねる。Assembler が値を持たず DefaultDQN へ戻る場合、reward scaler・Perception を調べる既存のロックは別に取る）から始める。これらの RFD が本書の途中で入ることは想定しない。入るときは、本書の課題 1 件ずつの切れ目で入れ、A13 の基準の実行体を作り直す。
- ワークツリーのブランチ `feature/20261125-ppo` は `src/algo/dqn` を触らない（`git diff --stat main...feature/20261125-ppo -- src/algo/dqn`、2026-11-27）。

## 7. 受入

| 番号 | 確かめること | 確かめ方 |
|---|---|---|
| A1 | ビルドとテスト | Debug の全ビルドが通る。`anet-algo-dqn-test` と `anet-core-base-test` の既定スイートが通る。失敗を例外とできるのは、変更前の同じケース・順・seed でも再現し、ケース名、原因、結果を記録したときだけ。§3 が名指す既存のテスト（課題の対応方針が挙げるもの）は、期待値を変えずに通る。既存テストの期待値は U4 の点の UQE に追加する NaN の確認を除いて維持する。U6 では構築の引数と学習の進みの読み方を移行し、新規の確認は A2〜A6、A11、A15 の挙動を対象に置く |
| A2 | 学習の設定の値域（U1・U2） | §5.1 の有効・無効の境界をキーごとに確認する。DefaultDQN の 7 キーと RainbowAgent の 6 キー（DefaultDQN から `replay_ratio` を除く）で、構築時に不正な有限値が完全キー・指定値・期待範囲付きで拒否される。設定層は NaN・±inf を拒否し、共通検証にも非有限値を直接渡して確認する。`use_grad_clip = false` や `use_td_clip = false` のときの 0、`replay_ratio` が正で `update_interval = 0` のときも拒否する。`per_alpha = 0` は受理し、PER 有効かつ `actor_approx` では既存 WARN を 1 回出す。`replay_ratio` は負・正を受理し、両方の更新方式を選べる |
| A3 | `uqe_tau`（U3） | Greedy、spatial なしの EpsilonGreedy、spatial ありの EpsilonGreedy の `GetScalar("uqe_tau")` が NaN であるテストが通る。UQE の方策の値は今のテスト（dqn_actor_test.cpp 1804 行）のまま通る |
| A4 | 点の UQE の比（U4） | 点で K=3・全範囲の問い合わせ無しの節で、`iqn_policy_margin_mc_ratio` が NaN である確認が通る。tail mean で K が 2 以上の今のテストは、期待値を変えずに通る。非 spatial Thompson の行動選択用の全範囲は K が 2 以上で有限のまま、点に可視化用の全範囲を併用する場合は NaN になることも確認する。設計 200 §9.1（jp・en）の NaN の条件の行に「点」の語が入り、030（jp・en）の該当の行に、点の UQE では出ないことが書かれている |
| A5 | fatal の重複（U5） | `action_full_q_margin.[1x]` と `action_full_q_margin.[x]` の鍵で、例外が 1 回だけ出て、捕まえたログの fatal の行が 1 行だけであるテストが通る。文言は今の `DQNActionInfo: invalid action index in scalar key: <鍵>` のまま |
| A6 | `RuntimeVars` の削除（U6） | `git grep -n RuntimeVars -- src docs/design` が 0 行。`git grep -n RuntimeVars -- docs/memo/0143_serialize_10rfd.md` が 0 行で、RFD0143 の冒頭に実際の改訂日を付した行がある。`per_beta` の値は A13 の比較で確かめる。加えて、DefaultDQN の `GetScalar("per_beta")` が、`use_per` のとき最初の更新の前に β の予定の step 0 の値を、`use_per` でないとき 0 を返すテストが通る |
| A7 | 振り分け（U7） | `rg -n -e kLearnerScheduleMetricKeys -e kValueDecompositionMetricKeys -e '"per_beta"' -e kPlasticityProbePrefix -e kNoisyPrefix -e ReplayBuffer::kKeyPrefix src/algo/dqn/src/dqn_assembler.cpp` で、`DqnComponentAssembler::GetScalar`・`GetTensor`・`GetTensorVector` の定義の開始から閉じ括弧までに一致する行が 0。ほかの関数の一致は別に確認する。値は A13 で確かめる |
| A8 | 選んだ行動の取り出し（U8） | 正規表現 `1, 1 ?\}\)\.expand\(` で dqn_learner.cpp と dqn_network_model.cpp を引いて 0 行（今は 9 行）。§3.6 の 13 か所が共通の関数を呼ぶ（実装計画に、か所ごとの呼び出しの行を記録する） |
| A9 | probe の抽選（U12） | `SampleUniqueUniform(` で dqn_learner.cpp を引いて 1 行（今は 4 行）。Noisy と policy churn の等間隔の τ が `MakeTauMidpointPositions` を呼ぶ（`CaptureNoisyDiagnostics` と `CapturePolicyChurnProbe` の本体を `+ 0.5f` で引いて 0 行） |
| A10 | 診断の並び（U11） | `scalar\([0-9]\)` で dqn_assembler.cpp を引いて 0 行。`kValueDecompositionMetricKeys` の定義が `MakeDiagnosticMetricKeys` で作られている。破壊の確かめ: 学習の群と行動選択の群のそれぞれで、列挙型の定義を保ち、キー表または値の組み立ての隣り合う 2 項目を、それぞれ名前と値を結びつけたまま入れ替えるとビルドが止まる。並びを戻すとビルドが通ることを実装計画に記録する（RFD0128 の検証資材の README の「速さと破壊検査」と同じやり方） |
| A11 | ヒントの入力（U9） | `kBootstrapExtrinsic` で default_dqn_agent.cpp と dqn_assembler.cpp を引いて 0 行。外的の bootstrap への変換は、dqn_actor の共通関数に置く。dqn_concrete_test.cpp の 3512・3598・3634 行、dqn_actor_test.cpp の 1395・1624 行、dqn_munchausen_test.cpp の 289・342 行のテストが、期待値を変えずに通る。`dqn_intrinsic_test.cpp` の分解 step 診断を通す既存テストに、合成 Q と Head ごとの値、Replay の初期優先度との一致の確認を足す。分解のとき step 診断の `MakeActorQHint` の呼び出しが 1 回であること（U9 で 2 回を残したときは、その理由が実装計画にあること） |
| A12 | 内的の割引率（U10） | 表の下の検索「割引率の選び方」の結果が、定義、default_dqn_agent.hpp の検証 3 行、新しい関数の中だけである。値は A13 の分解の 2 組で確かめる |
| A13 | 前後比較 | 基準は §6.5 の実行体（Debug）。組は、P3 の 34 構成に、次の 3 組を足した 37 構成。(a)(b) LunarLander の `run.@rnd` に `DefaultDQNAgent.intrinsic.value_mode = decomposed` と `skip.enabled = false` を指定し、`learner.intrinsic.discount_mode` を `shared` にした組と、`separate`（`learner.intrinsic.gamma` の linear、0.9 から 0.99、100 step）にした組。Actor の β は現行プロファイルの learner を使う。(c) DropMerge の `run.@iqn32_actor_approx` の後ろに短縮プロファイルを置いて、backend を決定的にし、`DefaultDQNAgent.learner.use_amp = false` で上書きした組。PER の初期優先度が `actor_approx` になることを確認する。条件は P3 の短縮条件（seed 1、8 lane、8,704 experience、決定的 backend、FP32）。過去の採取物と分けて `.scratch/rfd0207/validation.py` から既存の検証資材を呼び、両側で同じ 37 入力を使う。Run 名には `run_{t}_tmp_207_` を使う。比べるのは全 scalar の点（`90_perf/`、`91_resource/`、`92_resource/` を除く）、Run の `config/` 下の設定記録を全ファイル・全行（Run 名・Run のパスは共通の表記へ正規化）、Agent の json の設定の記録、WARN/ERROR の文言と順（Run 名、Run のパス、cpp/hpp のソース位置は共通の表記に正規化）。停止・ログ欠落・scalar が空の実行は比較の失敗とする。合格は、次の 2 つを除いて全部一致すること。(i) キー `uqe_tau` を UQE でない方策から読む tag（`32_agent_base/04_uqe_tau`（`$actor uqe_tau`）・`32_agent_base/04_uqe_tau_target`（`$agent target_policy.uqe_tau`））の点が後側で消え、消えた点の前側の値がすべて 0 である。(ii) IQN の点の UQE で全範囲の問い合わせが無い構成の、`iqn_policy_margin_mc_ratio` を `$action_info` から読む `33_agent_action/21_iqn_policy_margin_mc_ratio_ema`（学習時）、`51_eval1/53_iqn_policy_margin_mc_ratio_ema` と `52_eval2/53_iqn_policy_margin_mc_ratio_ema`（評価時）の点が後側で消える。差を認めるのは、当該方策の実効設定を確認でき、前側に有限点が存在し、後側でそのタグの点がすべて消えた場合。消えた点の数を構成ごとに実装計画に記録する。検証資材の入力が RFD0191 より前の設定キーで動かないときは、入力の設定キーの名前だけを今の名前に直し、直した対応を実装計画に記録する（値は変えない）。補助学習の batch 8 は `learn.intake.batch_size`。既定の旧 `learn_predictor` に対応し 256 は `rnd.error_norm.standardize.warmup_steps`、predictor の待ちは 0 のままにする |
| A14 | 速さ | P4 の手順（Release、Atari qbert、IQN、128 lane、batch 256、捨て Run のあと新旧を交互に 3 巡以上）で、変更後の更新 1 回あたりの実時間の中央値が、基準の最大値以下。夜間依頼の長い Run を起こさない条件を優先し、A14 の測定は朝の残件にする。着手前に Release 実行体と比較用設定を保存する。基準が作れなければ、その失敗と条件を記録し、性能差は判定不能とする |
| A15 | 可視化の特性（U13） | DefaultDQN の `GetTensorDictFunction("policy-net.forward")` に 1 枚を渡し、実際に Network へ渡した 2 キーの shape と全値を比較する。`all` は両方を積み、`listed` は指定した 1 キーを積む。stack の位置ごとに渡した 1 枚と同じ値になる |

検索「割引率の選び方」（A12。正規表現、テストを除く `src/`）:

```text
git grep -nE "kDiscount(Separate|Shared)" -- src ":!*_test.cpp"
```

見積もり（項番 33 の頻度×単価）: hot path の順番と回数は変えない。Learner の 8 か所（U8）は更新ごとに、今と同じ `gather` を 1 回ずつ呼ぶ。関数を 1 段挟むだけで、tensor の演算の数は変わらない。`ComponentGammas`（U10）は更新ごとに数回呼ばれるが、条件の評価が関数の中に移るだけである。Actor の Q ヒント（U9）と行動選択の群（U11）は step ごとに、今と同じ演算を 1 回ずつ行う。step 診断（U9）は、Evaluation Trace か pane の要求があるときだけ動き、分解のとき `MakeActorQHint` が 2 回から 1 回に減る。probe（U12）は購読した間隔ごとで、次の観測を Perception に通す処理が減る。どれも 1 step・1 更新あたりの増加は 0 なので、A14 は Learner の更新で測れば足りる。Actor の step の速さは受入で測らない。

## 8. issue の対応表

| issue | 題 | area | 根拠の位置 | 直す | 受入 |
|---|---|---|---|---|---|
| [issue-20261123-eb0a0c](../management/issues/issue-20261123-eb0a0c.md) | DQN 系の設定が learner.grad_clip_tau・learner.td_clip_value・learner.update_interval の値域を検証しない | 324_dqn、321_agent_core | default_dqn_agent.hpp 462・480〜486・805 行、rainbow_agent.hpp 55〜62 行、dqn_learner.cpp 257〜263・2528 行、dqn_math.cpp 240 行 | U1・U2 | A1・A2・A13 |
| [issue-20261125-c702d2](../management/issues/issue-20261125-c702d2.md) | UQE を使わない方策の uqe_tau が NaN でなく 0 を返す | 321_agent_core | dqn_actor.cpp 876・882 行、dqn_actor.hpp 157 行 | U3 | A1・A3・A13 |
| [issue-20261125-770358](../management/issues/issue-20261125-770358.md) | 点の UQE で iqn_policy_margin_mc_ratio が同じ τ の複製から作られる | 321_agent_core | dqn_actor.cpp 1128〜1142・1231・1240 行 | U4 | A1・A4・A13 |
| [issue-20261125-ea9fef](../management/issues/issue-20261125-ea9fef.md) | ParseActionScalarIndex が同じ fatal を 2 回出す | 321_agent_core | dqn_actor.cpp 238・252〜261 行 | U5 | A1・A5 |
| [issue-20261125-84c03f](../management/issues/issue-20261125-84c03f.md) | RuntimeVars を Agent が持って参照で渡す | 321_agent_core | dqn_math.hpp 225〜229 行、dqn_learner.cpp 192・1016・1672・1791・1870 行 | U6 | A1・A6・A13 |
| [issue-20261125-6a8e46](../management/issues/issue-20261125-6a8e46.md) | Agent の scalar の振り分けが Learner の群の一覧を写す | 321_agent_core | dqn_assembler.cpp 391〜435 行、dqn_learner.cpp 276〜319・851〜865 行 | U7 | A1・A7・A13 |
| [issue-20261125-9e8834](../management/issues/issue-20261125-9e8834.md) | 選んだ行動の値を取り出す式が 13 か所に写されている | 321_agent_core | §3.6 の 13 か所 | U8 | A1・A8・A13・A14 |
| [issue-20261125-9f1801](../management/issues/issue-20261125-9f1801.md) | probe の 4 群が抽選の手順を写す | 321_agent_core | dqn_learner.cpp 650〜653・689〜693・726〜730・760〜764・1046〜1056 行 | U12 | A1・A9・A13 |
| [issue-20261125-5ebf2f](../management/issues/issue-20261125-5ebf2f.md) | 価値の分解の診断 2 群と step 診断の scalar(3) が、並びの食い違いでビルドが止まる形に載っていない | 321_agent_core | dqn_math.hpp 273〜275 行、dqn_learner.cpp 2860〜2863 行、dqn_actor.cpp 37〜44・355・360〜376・1729〜1734 行、dqn_assembler.cpp 117・134 行 | U11 | A1・A10・A13・A14 |
| [issue-20261125-de8cfd](../management/issues/issue-20261125-de8cfd.md) | Actor Q ヒントの入力の組み立てが Actor と step 診断の 2 か所にある | 321_agent_core | dqn_actor.cpp 1584〜1602 行、dqn_assembler.cpp 127〜148・163・167 行、default_dqn_agent.cpp 938 行 | U9 | A1・A11・A13 |
| [issue-20261123-d74c3f](../management/issues/issue-20261123-d74c3f.md) | 内的の成分の割引率の選び方が DQN 系の 6 か所に写されている | 324_dqn、321_agent_core | default_dqn_agent.cpp 558・939 行、dqn_assembler.cpp 126・181〜182 行、dqn_learner.cpp 993・2631 行 | U10 | A1・A12・A13・A14 |
| [issue-20261123-e43866](../management/issues/issue-20261123-e43866.md) | stack を有効にした GetTensorDictFunction の複製の経路を通すテストが無い | 324_dqn | default_dqn_agent.cpp 579〜583・846 行、dqn_concrete_test.cpp 327〜341・593 行 | U13 | A1・A15 |

## 9. グリルの材料

開発者の発言（2026-11-27。決定として扱わない）:

1. 「仕様調整がほぼ要らない小さいのをいっぱい集めたPRDを作りたい」
2. 「寝てる間に回したくて」
3. 「3本で」（置き場ごとに RFD0205 文書・RFD0206 基盤とアプリ・本書 DQN 系の 3 本に分ける案への返事）

AI の案（決定として扱わない）: 起票の時点で、題材の課題の対応方針に複数の案や「対応のときに決める」が残っていた行に、既定を置いた。U2（追記の 4 キーを範囲に入れる）、U8（NaN の検査を広げない）、U9（2 回の呼び出しを 1 回にする条件）、U10（decomposed でないときは外的の割引率を返す）、U11（列を並べる版を足す）、U13（RFD0200 のあとの口で書く）である。

## 10. 関連

- 設計 [200](../design/200_dqn_agents.jp.md)・[214](../design/214_value_decomposition.jp.md)・[030](../design/030_user_guide_analysis.jp.md)
- ADR [0033](../adr/0033-policy-churn-fixed-probe-and-target-lag.md)・[0058](../adr/0058-approx-priority-column-computed-by-agent-step-diagnostics.md)・[0075](../adr/0075-actor-q-hint-five-columns-carry-per-head-values.md)
- [RFD0094](done/0094_evalpanel_step_history_10rfd.md)、[RFD0121](0121_extrinsic_intrinsic_value_decomposition_10rfd.md)、[RFD0122](0122_small_refactor_and_test_gap_fixes_10rfd.md)、[RFD0125](0125_dqn_learner_update_skeleton_10rfd.md)、[RFD0128](0128_dqn_agent_refactor_bundle_10rfd.md)、[RFD0182](0182_source_layout_10rfd.md)、[RFD0194](0194_config_key_full_names_in_messages_10rfd.md)、[RFD0200](0200_forwarder_public_interface_rework_10rfd.md)
- トピック別まとめ [321 Agent 共通](../wiki/concepts/framework/321_agent_core.md)・[324 DQN](../wiki/concepts/framework/324_dqn.md)
