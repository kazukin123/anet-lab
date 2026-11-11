# PRD125: DQN の学習器の 1 回の更新の進行と Munchausen の target を 1 か所に寄せる

> 2026-11-12 起票（グリル済み）。行番号は起票時点の作業ツリーの位置で、実装時は近傍の記述で探し直す（dqn_based_agent.cpp は並行する作業で行がずれる）。起草のあと、別モデル（Opus）の検討を受けて、反対の無い指摘を反映した。値が変わらないことの根拠（Head が FP32 で動くこと、更新の直前の検査）、要素誤差へ渡す指定と値の無いものの扱い（§5.2）、基底を呼ぶだけの上書きを書かないこと（D4）、比べる組と診断の値（D13・§5.4）、速さの測り方の言い方（D16・A9）、replay fit の診断を範囲外とすること（§3.2）である。BF16 の確かめ方（D14）は、別モデルの案をユーザーが選び、LunarLander の 3 つの分布ぶんの組を足した。
> 起点: 2026-11-11〜12 の会話。ユーザーが、課題 ab98b7 を題材に PRD を起こすよう指示した。課題には 2026-11-11 に、Munchausen の target の二重化が同じ系統の写しとして追記されていた。AI は、それを含めるかを比べて示し、ユーザーは含めると決めた。
> 題材: [issue-20261103-ab98b7](../management/issues/issue-20261103-ab98b7.md)。
> 関連: [PRD122](122_small_refactor_and_test_gap_fixes_10prd.md)（短い Run の前後比較の前例）、[PRD121](121_extrinsic_intrinsic_value_decomposition_10prd.md)・[PRD924](924_intrinsic_reward_novelty_10prd.md)（同じ場所へ変更を足す予定の PRD）、[PRD067](done/067_MunchausenRL_10prd.md)（Munchausen の target）、[PRD098](098_temporl_skip_policy_10prd.md)（skip の loss）、[PRD073](done/073_replay_fit_metrics_10prd.md)（target と要素誤差の切り出し）、[PRD103](done/103_temporl_reference_alignment_10prd.md)（性能の測り方の前例）、[PRD957](957_source_layout_10prd.md)（ソースの配置）、[ADR 0035](../adr/0035-munchausen-target-learner-local-real-space.md)。

## 1. 問題

### 1.1 1 回の更新の進行が、3 つの関数に写されている

DQN 系の学習器は、分布の表し方ごとに 3 つある。通常 Q の `TDLearner`、固定の分位点の `QRLearner`、τ を入力に取る `IQNLearner` である。1 batch ぶんの更新を行う `UpdateFromSamples` を、3 つがそれぞれ実装している。3 つの関数は、次の進行を同じ順で書いている。

観測の正規化 → current の forward → target → skip の loss → 要素誤差 → PER の重み付けと平均 → 優先度の準備 → 最適化 → 優先度の反映 → 結果の組み立て

共通の手順を変えるたびに、3 か所を同じに直すことになる。1 つ漏らすと、分布の表し方によって学習の中身が違ったままになる。写しはすでに 4 点で食い違っている（§2.2）。

### 1.2 Munchausen の target も、複数か所に写されている

本 PRD で target と書くものは、用語集の Bellmanターゲットである。Munchausen の target は、Munchausen項と soft価値ブートストラップを使う Bellmanターゲットを指す。target の式は、target return と bootstrap 価値から Bellmanターゲットを組む式を指す。

Munchausen の target は、通常 Q 用（`Learner::MakeTarget` の中の枝）と分位点用（`QuantileLearnerBase::MakeMunchausenTarget`）の 2 か所にある。2 か所は前半を同じ並びで書いている。違いは、読む出力の名前と、IQN の τ を入力へ足すかだけである（§2.3）。

target の式の後半（n-step の割引 → 終端のマスク → h 空間へ戻す）は、通常の target と Munchausen の target、通常 Q と分位点の組み合わせで、4 か所に同じ形がある。

`log_policy_mode` を足す、capture の扱いを変える、といった変更で、2〜4 か所を同じに直すことになる。

### 1.3 同じ場所へ変更を足す予定がある

価値の分解（[PRD121](121_extrinsic_intrinsic_value_decomposition_10prd.md)）と内的報酬（[PRD924](924_intrinsic_reward_novelty_10prd.md)）は、3 つの学習器の target と要素誤差のあいだへ処理を足す。写しのままでは、同じ変更を 3 回書くことになる。PRD924 の課題は、題材の課題を待っている（§9）。

## 2. 現状（2026-11-12 に確認したこと）

対象は [dqn_based_agent.cpp](../../core/anet-core/src/dqn_based_agent.cpp) と [dqn_based_agent.hpp](../../core/anet-core/src/dqn_based_agent.hpp) である。

### 2.1 3 つの関数の進行と、分布ごとの違い

関数の位置は、`TDLearner::UpdateFromSamples` が 4855 行、`QRLearner::UpdateFromSamples` が 4995 行、`IQNLearner::UpdateFromSamples` が 5124 行からである。

| 手順 | TD | QR | IQN |
| --- | --- | --- | --- |
| 入力の検査 | 無い | 行動・target return・終端の行数の検査 | QR と同じ |
| 観測の正規化 | 行う | 行う。あとに観測の NaN の検査 | QR と同じ |
| current 側の計算 | `q` を forward し、選んだ行動の Q と、最大の Q・1 位と 2 位の差の指標を作る | `q_dist` を forward し、選んだ行動の分位点、上側の広がり、指標を作る | 先に current の τ を乱数で作り、入力へ足す。あとは QR と同じ。上側の広がりは τ の順に並べてから計算する |
| target | `MakeLearningTargets` | 同じ | 同じ |
| skip の loss | `ComputeSkipLoss` に、正規化した観測を渡す | TD と同じ | τ を足した入力と、current の τ を渡す |
| 要素誤差 | `ComputeElementError` | current の平均を添える。分位点の位置は `QRLearner` が補う | current の τ と current の平均を添え、補助の統計を求める |
| 追加の診断 | 無い | 無い | IQN の診断 7 値を作る |
| PER の重み付けと平均 | §2.2 | §2.2 | §2.2 |
| 優先度の準備 → 最適化 → 優先度の反映 → 結果の組み立て | 同じ 4 つの関数を同じ順で呼ぶ | 同じ。上側の広がりを添える | 同じ。IQN の診断と上側の広がりを添える |

- current の forward から PER の重み付けまでは、AMP の範囲（Autocast）の中にある。優先度の準備から後ろは、範囲の外にある。3 つとも同じである。
- 学習器の乱数を引くのは、IQN が current の τ を作る箇所（current の forward の前）と、target を作る関数の中（target の行動の選択と、τ の生成）である。ほかに、forward の中で引く乱数（dropout、NoisyNet）がある。どれも、関数を呼ぶ順と forward の順で決まる。
- target の作り方（`MakeTarget`）と要素誤差（`ComputeElementError`）は、すでに仮想関数である。基底の `Learner` が通常 Q の実装を持ち、`QuantileLearnerBase` と `QRLearner` が上書きする（dqn_based_agent.hpp の 848・852 行）。replay fit の診断も同じ関数を呼ぶ（[PRD073](done/073_replay_fit_metrics_10prd.md)）。
- `UpdateFromSamples` は、基底では純粋仮想である（dqn_based_agent.hpp の 833 行）。外側のループ `Learner::UpdateFromBatch`（3994 行から）が、1 回の更新につき 1 回呼ぶ。

### 2.2 すでにある食い違い

| 食い違い | TD | QR | IQN |
| --- | --- | --- | --- |
| PER が無効のときの重み | 掛けない（4940 行から） | 1 を並べた Tensor を作って掛ける | QR と同じ |
| 入力の検査と NaN の検査 | 無い | ある | ある |
| プロファイルの区間 | `prepare_per`・`optimize`・`update_per`・`make_result` | 左に加えて `normalize`・`forward_current`・`loss` | QR に加えて `generate_current_taus`・`iqn_diagnostics`・`weighted_loss` |
| DEBUG ログ | 無い | `element_loss=` と `weights=` の 2 行。ほかに、コメントアウトされた `obs=` の行が 1 つ | 無い |

- skip の loss は、PER が無効のとき重みを掛けない（4687 行）。TD と同じ形である。TD は、重みを loss の device へ移す呼び出しを挟むが、同じ device なので何もしない。
- 分位点の要素 loss は float32 である。Head が FP32 で動くので（§2.3）、current の分位点も target も FP32 で届く。1 を並べた float32 の Tensor を掛けても、値も型も変わらない。重み付けした要素 loss は、最適化と勾配ノイズの測定へ渡るが、1 を掛ける演算が無くなっても、勾配の値は同じである。
- 学習の経路では、`UpdateFromSamples` の直前に、行動・target return・終端の行数と型の検査がある（`ValidateDeviceSamples`。3972 行から。`UpdateFromBatch` が 4068 行で呼ぶ）。QR と IQN の入力の検査は、これと重なる。重ならないのは、テストが `UpdateFromSamples` を直接呼ぶときだけである。
- 検査は、Debug と RelWithDebInfo のビルドでだけ動く。NaN の検査は Debug だけである（[tensor_check.hpp](../../core/anet-core/include/anet/tensor_check.hpp) の 11〜51 行）。Release では何もしない。
- DEBUG ログは、デバッグログを有効にしたビルドを、デバッガを付けて動かしているときだけ出る（[log.hpp](../../core/anet-core/include/anet/log.hpp) の 379〜392 行）。
- プロファイルの区間の名前は、区間を書いた関数の名前と、区間の名前から作る（[profile.hpp](../../core/anet-core/include/anet/profile.hpp) の 269〜287 行）。区間の名前を読む設定・スクリプト・現行の文書は無い（§5.5）。

### 2.3 Munchausen の target と、target の式

前半の写しは、`Learner::MakeTarget` の Munchausen の枝（4769 行から）と、`QuantileLearnerBase::MakeMunchausenTarget`（4364 行から）にある。どちらも次を同じ並びで行う。

1. target 方式では、観測と次の観測を連結する。
2. IQN では、target の τ を作って入力へ足す（分位点用だけ）。
3. 連結した行を、もとの遷移へ対応付ける。
4. target の Network を forward し、current と next に切り分ける。
5. target 方式では、capture の行数を検査して記録する。合わなければ `Munchausen target capture expected rows=` の文言で止まる。
6. online 方式では、online の Network を eval で forward し直す。IQN では、その前に current の τ を作って入力へ足す（分位点用だけ）。
7. current の行数が batch の件数と違うとき、current を複製する。

違いは、読む出力の名前（`q` と `q_dist`）と、2 と 6 の τ だけである。このあとの、実空間へ戻して方策を作り、soft価値ブートストラップを組む部分は、通常 Q と分位点で中身が違う。

target の式は 4 か所にある。

| 箇所 | 行 | Munchausen項 | h の逆変換 |
| --- | --- | --- | --- |
| 分位点の通常の target（`CalcTargetQuantiles`） | 4344 行から | 無い | 関数の中で行う |
| 分位点の Munchausen の target（`CalcMunchausenTargetQuantiles`） | 4435 行から | 足す | 受け取る値が実空間なので、行わない |
| 通常 Q の Munchausen の target | 4814 行の前後 | 足す | 受け取る値が実空間なので、行わない |
| 通常 Q の通常の target | 4846 行の前後 | 無い | 式の手前で行う |

- 4 か所とも、γ の n 乗と「終端でない」旗を先に掛け、その積に bootstrap 価値を掛ける。掛ける 2 つの順は箇所で違うが、2 つの数の積は順によらない。
- Munchausen項のある 2 か所は、target return に Munchausen項を足してから、bootstrap の項を足す。
- target return は float32 で保存されている（[replay_buffer_impl.cpp](../../core/anet-core/src/replay_buffer_impl.cpp) の 634 行）。学習の経路では、更新の直前の検査 `ValidateDeviceSamples`（3972 行から）も、float32 であることを確かめる。Munchausen の 2 か所は target return を float32 へ変換するが、もとから float32 なので何もしない。
- Network の出力 `q`・`q_dist` は、AMP が有効でも FP32 である。Head が AMP を切って FP32 で動くためである（[nn_impl.cpp](../../core/anet-core/src/nn_impl.cpp) の 1797〜1803 行）。分位点の通常の target は、h の逆変換を FP32 の値にそのまま掛けている。
- 4 か所とも、勾配を作らない範囲の中か、勾配を切った値で計算している。
- 分位点の通常の target（`CalcTargetQuantiles`）には、target の形と NaN の検査がある。
- Munchausen の計算は実空間で行い、完成した target にだけ h を掛ける。h の逆変換を中で行う `CalcTargetQuantiles` は通常の target 専用で、逆変換を 2 回掛けないための決まりである（[ADR 0035](../adr/0035-munchausen-target-learner-local-real-space.md) の 45 行）。

### 2.4 呼び出し元とテスト

- テスト用の派生クラスが 3 つあり、`UpdateFromSamples` を丸ごと上書きしている。`TestLearner`（[dqn_based_agent_test.cpp](../../core/anet-core/src/dqn_based_agent_test.cpp) の 370 行から）、`CapturingLearner`（同 1024 行から）、`SoftHardSelector`（[dqn_munchausen_test.cpp](../../core/anet-core/src/dqn_munchausen_test.cpp) の 154 行から）である。前の 2 つは `dqn::Learner` の、最後の 1 つは `QuantileLearnerBase` の派生である。
- テストは、`UpdateFromSamples` を 14 行で直接呼んでいる。`TDLearner`・`QRLearner`・`IQNLearner` の型に変換して呼ぶ行を含む（dqn_based_agent_test.cpp の 4618〜4620 行）。
- replay fit の診断（`Learner::EvaluateReplayFit`。2978 行から）は、進行の前半を別に書いている。正規化 → IQN の τ → forward → 選んだ行動の値 → `MakeTarget` → `ComputeElementError` の順である。学習の進行とは、forward を eval で行うこと、τ を等間隔に置くこと、AMP の範囲と指標と PER の重みを持たないことが違う。`MakeTarget` を診断用の指定で呼ぶので、Munchausen の評価と target の式（§2.3）を通る。
- target の式の 4 か所と、Munchausen の前半を、名指しで呼ぶテストは無い。`MakeMunchausenTargetTerms`（方策と Munchausen項を作る関数。今回は変えない）を呼ぶテストが 1 つある。3 つの学習器の更新の全体は、`[dqn]`・`[munchausen]`・`[prd098]`・`[prd103]` のテストが通る（[PRD098 の検証資材](../../core/anet-core/testdata/prd098/README.md)、[PRD103 の検証資材](../../core/anet-core/testdata/prd103/README.md)）。

### 2.5 規約と過去の判断

- 挙動を変えない直しでは、hot path の速さが変わらないことを Run のブレ幅で確かめる。乱数の呼び出し順に触るなら、同 seed の再現性を確かめる。テストの無い箇所は、特性テストを先に足す（[リファクタリング観点](../quality/refactor.md) の「ガードレール」）。「挙動を変えない」には、ログの文言を含める（同「運用ルール」）。
- 学習器を変えるときは、乱数の派生と消費の順を変えず、同 seed・決定的モードの短い Run のメトリクスを、実装の前に保存した実行体と比べる。checkpoint のバイト列は比べない（[地図 321](../wiki/categories/321_agent_core.md) の「変えるときの条件」）。
- 短い Run の前後比較は、PRD122 が同じファイルで行った。変更前の実行体と設定を保存し、4 環境を seed 1・決定的モードで流して、時間と資源の tag を除く全 scalar の点を比べる（[PRD122 の検証資材](../../core/anet-core/testdata/prd122/README.md)）。
- もとからある死にコードは、消さずに報告する（[AGENTS.md](../../AGENTS.md) の「LLM コーディング規律」）。

### 2.6 既定の設定が通る経路

- 既定の分布の表し方は QR で、PER は有効である（[agent.txt](../../apps/runner/config/agent.txt) の 253・287 行）。Atari と LunarLander の既定は IQN である。通常 Q と PER 無効は、既定の設定のどの環境も通らない。
- LunarLander は、IQN と QR の配線を両方持つ（[LunarLander.txt](../../apps/runner/config/LunarLander.txt) の 306〜316 行）。通常 Q の配線は QR と同じである。

## 3. ゴール・非ゴール・成功の測り方

### 3.1 ゴール

- G1: 1 回の更新の進行を 1 か所に書く。TD・QR・IQN は、分布ごとに違う部分だけを持つ。§2.2 の食い違いは、1 つの形にそろえる。
- G2: Munchausen の target の前半と、target の式を、それぞれ 1 か所に書く。
- G3: 学習の中身を変えない。同じ seed・決定的モードで、TD・QR・IQN のメトリクスが、直す前と一致する。PER・skip・Munchausen・TBO の有無を含めて確かめる。速さも変えない。
- G4: 題材の課題 ab98b7 が片付く。

G3 の例外は、検査・ログ・プロファイルの 3 つである。どれも学習の値に関わらない。

- Debug 系のビルドで、TD にも入力の検査と NaN の検査が掛かる（D7）。
- DEBUG ログの 2 行が、TD と IQN でも出る。`weights=` の行は、PER が無効のとき出なくなる（D8）。
- プロファイルの区間の名前が変わる（D9）。

### 3.2 非ゴール

| 退ける案 | 理由 | 本当の制約か |
| --- | --- | --- |
| PER の重みの扱いそのもの（IS 重みの正規化、NaN の扱い）を変える | 学習の値が変わり、G3 に反する。本 PRD は、変える箇所を 1 つにするところまでを行う | 制約ではない（別の関心） |
| `use_td_clip` の不具合を一緒に直す | 学習の値が変わり、G3 に反する | 制約ではない（別の関心。課題は別にある） |
| 価値の分解のための、Head ごとの目標の口を先に足す | PRD121 は暫定で、口の形が決まっていない | 制約ではない（別の関心。PRD121 のグリルで形が決まったら、PRD121 が足す） |
| TD・QR・IQN のクラスを 1 つにまとめる | 構築の口と、設定からクラスを選ぶ箇所が変わる。写しを無くすのに要らない | 制約ではない。本 PRD のあとでも行える |
| 外側のループ `UpdateFromBatch` と、結果の入れ物を作り替える | 写しではない | 制約ではない（別の関心） |
| dqn_based_agent.cpp をファイルに分ける | [PRD957](957_source_layout_10prd.md) の範囲である | 制約ではない（別の関心） |
| MuZero・ImageCls の学習器を同じ形にする | `dqn::Learner` の派生ではなく、進行が違う | 制約ではない（別の関心） |
| replay fit の診断が別に書いている進行の前半（§2.4）を、学習の進行と 1 つにする | forward の仕方・τ の置き方・指標の有無が学習と違い、current 側の計算の口をそのまま使えない。題材の課題が挙げる写しでもない | 制約ではない（別の関心。課題 [4efa57](../management/issues/issue-20261112-4efa57.md) に記録した） |

### 3.3 成功の測り方

- 更新の進行を書く関数が、3 つから 1 つになる（A3）。target の式が、4 か所から 1 か所になる（A5）。
- 直す前と後で、31 組の短い Run の全 scalar の点が一致する（A2・A4）。
- 直す前と後で、learn step 1 回あたりの実時間が、直す前の Run のばらつきの内に収まる（A9）。
- 題材の課題を、実装のあとに解決にできる（§6 の受入がすべて通る）。

### 3.4 足すものと、ゴールへの寄与

| 足すもの | 寄与 |
| --- | --- |
| 基底の学習器が持つ、更新の進行（D2） | G1。進行が 1 か所になる |
| current 側の計算と、追加の診断の 2 つの差し替えの口（D3） | G1。分布ごとに違う部分だけを、分布ごとに書ける |
| Munchausen の評価の共通の関数（D10） | G2 |
| target の式の共通の関数と、その単体テスト（D11・D15） | G2 |
| 前後比較の検証資材と、31 組の Run（D13・D17） | G3 の確かめ。既定の設定が通らない通常 Q と PER 無効も通す |
| BF16 の 4 組（D14） | G3 の確かめ。AMP が有効のときだけ値に出る誤りを見つける |

先送りするものは 1 つである。価値の分解のための Head ごとの目標の口は、別の関心なので PRD121 で作る。PRD121 のグリルで口の形が決まったら作れる。

## 4. 確定事項

2026-11-12 のグリルで決めた。D1・D2 は、ユーザーが案を比べて選んだ。D8 は、AI が示した既定（消す）をユーザーが退けて決めた。D6・D7・D9・D13〜D18 は、AI が示した推奨の既定を、ユーザーがまとめて採った。D3〜D5・D10〜D12・D19 は、AI が規約と今のコードから導いて示した。起草のあとに別モデル（Opus）の検討を受け、D4・D7・D9・D13・D16 の範囲と前提を補った。D14 は、別モデルの案と起草時の案をユーザーが比べて選び直した。

| # | 決定 | 理由と前提 |
| --- | --- | --- |
| D1 | **束ね方とステージ。** 更新の進行（§1.1）と Munchausen の target（§1.2）を 1 本の PRD で扱い、実装を 3 つのステージに分ける。ステージ 0 は、実行体の保存と、比べる組の確かめ。ステージ 1 は、更新の進行を寄せる。ステージ 2 は、Munchausen の評価と target の式を寄せる。ステージ 1 と 2 は、それぞれの終わりに前後比較を全部通す。ステージ 1 だけでコミットできる | ユーザーが、Munchausen の target を別の PRD にする形と比べて選んだ。2 つは同じファイルで、確かめ方が同じである。保存した実行体と Run の組を使い回せ、グリル・計画の承認・レビューが 1 回ずつで済む。PRD121 は両方に触る。ステージを分けるのは、変更を小さく区切り、前後比較で原因を絞れるようにするため。依存: なし |
| D2 | **寄せる範囲。** 進行の全体を、基底の学習器 `dqn::Learner` の `UpdateFromSamples` に書く。TD・QR・IQN の `UpdateFromSamples` は消す。進行の中身は §5.2 のとおり | ユーザーが、末尾（PER の重み付けから結果の組み立てまで）だけを共通の関数にする形と比べて選んだ。末尾だけの形は、正規化から要素誤差までの順が 3 か所に残る。PRD121 が target と要素誤差のあいだへ処理を足すとき、同じ問題がまた出る（§1.3）。変更は大きいが、前後比較で学習の中身が同じことを確かめられる（D13）。依存: D13 |
| D3 | **分布ごとの差し替えの口。** 仮想関数を 2 つ足す。1 つは current 側の計算で、forward と、損失に使う値と、指標を返す。もう 1 つは追加の診断で、今は IQN だけが実装する。通常 Q の実装は基底に置き、分位点の側が上書きする。分位点の current 側の計算（forward、選んだ行動の分位点、指標）は、QR と IQN で 1 か所に書き、IQN は τ に関わる部分だけを足す | target と要素誤差が、すでにこの型である（§2.1）。型を 2 つにしない。基底に通常 Q の実装を置くと、`dqn::Learner` から直に派生するテスト用のクラス 2 つを変えずに済む（§2.4）。QR と IQN で 1 か所に書くのは、寄せた結果として新しい写しを作らないため。依存: D2 |
| D4 | **`UpdateFromSamples` の扱い。** 仮想関数のまま残す。`TDLearner`・`QRLearner`・`IQNLearner` からは、今と同じく外から呼べるようにする。3 つの学習器に、基底を呼ぶだけの上書きは書かない | テスト用の派生 3 つが丸ごと上書きしていて、テストが 3 つの型から直接呼んでいる（§2.4）。今あるテストを変えずに通すため（D15）。基底を呼ぶだけの上書きは、仕事をしない層になるので書かない（[リファクタリング観点](../quality/refactor.md) の「仕事をしていない層」）。依存: D2 |
| D5 | **変えない順。** 乱数を引く位置と順、forward の順、AMP の範囲の端（current の forward の前から、PER の重み付けと平均のあとまで）、優先度の準備 → 最適化 → 優先度の反映の順を変えない。IQN の current の τ は、current の forward の前に引く。target の τ と online 方式の τ は、今の位置（§2.3 の 2 と 6）で引く | 乱数の順と forward の順が変わると、同 seed のメトリクスが変わる（§2.5）。優先度の準備を最適化の前に置くのは、今の 3 つの関数に共通の順である。依存: なし |
| D6 | **PER が無効のときの重み。** 掛けない形にそろえる。PER が有効のときは、抽選した batch の重みをそのまま掛ける。重みを device へ移す呼び出しは置かない | TD と skip の loss が、今この形である。QR・IQN が掛けているのは 1 で、値も型も変わらない（§2.2）。Tensor の確保と掛け算が 1 回ずつ減る。依存: なし |
| D7 | **入力と損失の検査。** 3 つの関数のどれかにある検査のうち、分布によらないもの（行動・target return・終端の行数、観測・要素 loss・重み・loss の NaN、要素 loss と loss の形）は、進行に 1 回ずつ置く。TD にも掛かる。分布に固有の検査は、分布ごとの関数に残す。進行に置いた検査で、今あるテストか §5.4 の Run が止まるなら、その検査は元の分布の関数に戻す。値は変えない | Debug 系のビルドでだけ動き、Release の動きは変わらない（§2.2）。TD だけ外す理由が無い。行数の検査は、学習の経路では直前の `ValidateDeviceSamples` と重なる（§2.2）。それでも進行に置くのは、テストが `UpdateFromSamples` を直接呼ぶときにも掛かるようにするためで、QR と IQN の今の動きを保つ。戻す決まりを置くのは、ある分布だけで成り立つ検査が混ざっていても、学習の値を変えずに済ませるため。前後比較は Debug で流すので、止まる検査はそこで分かる（D17）。依存: D17 |
| D8 | **DEBUG ログ。** `element_loss=` と `weights=` の 2 行を残し、進行の中の今と同じ位置（要素誤差のあとと、重み付けの前）に置く。TD と IQN でも出るようになる。`weights=` の行は、PER が有効のときだけ出す。コメントアウトされた `obs=` の行も、進行の入力の検査のあとへ移して残す | ユーザーが決めた。2 行は選んで残してあるログで、通常の Run には影響しない（デバッガを付けたときだけ出る。§2.2）。QR だけに出す形は、進行に分布の名前での分岐が要るので採らない。PER が無効のとき `weights=` を出さないのは、重みの Tensor を作らないため（D6）。コメントアウトの行を残すのは、もとからある死にコードを消さない決まりによる（§2.5）。依存: D6 |
| D9 | **プロファイルの区間。** 進行の区間を、3 つの分布で 1 組にそろえる。組は `normalize`・`forward_current`・`loss`・`prepare_per`・`optimize`・`update_per`・`make_result` である。`loss` は、要素誤差から PER の重み付けと平均までを含む。IQN だけの `generate_current_taus` と `iqn_diagnostics` は、τ を作る箇所と、IQN の診断を作る箇所に残す。`weighted_loss` は無くす。`generate_current_taus` は `forward_current` の内側に、`iqn_diagnostics` は `loss` の内側に入り、IQN の `loss` は診断の時間を含むようになる。区間の名前の前に付く関数の名前は、分布ごとの名前から基底の名前に変わる | 区間の名前を読むものが無い（§2.2・§5.5）。区間を書いた関数の名前が付くので、進行を基底へ移すと名前は必ず変わる。`weighted_loss` を無くすのは、QR の `loss` が今も重み付けを含むため。依存: D2 |
| D10 | **Munchausen の評価の共通の関数。** §2.3 の 1〜7 を 1 つの関数にして、基底の学習器に置く。読む出力の名前と、τ を入力へ足すかを、呼ぶ側が渡す。current と next の値を返す。並びと、止まるときの文言、区間 `forward_target`・`forward_munchausen_online` は変えない。実空間へ戻して方策と soft価値ブートストラップを組む部分は、通常 Q と分位点のそれぞれに残し、区間 `munchausen_target` もそれぞれに残す | 2 か所の違いは、出力の名前と τ だけである（§2.3）。渡す形にするのは、基底の関数に分布の表し方での分岐を置かないため。区間の名前を保つのは、3 つの方式の費用を切り分けるための決まりによる（ADR 0035 の 44 行）。方策と soft価値ブートストラップは中身が違うので、寄せない。依存: D5 |
| D11 | **target の式の共通の関数。** 4 か所の式を 1 つの関数にして、基底の学習器に置く。実空間の bootstrap 価値と、あれば Munchausen項を受け取り、target return・γ の n 乗・終端の旗から target を作って、TBO が有効なら h を 1 回掛けて返す。通常の target は、呼ぶ側が h の逆変換を済ませてから渡す。計算の形は §5.3 のとおり | 4 か所の違いは、Munchausen項の有無と、h の逆変換をどこで行うかだけである（§2.3）。実空間の値だけを受け取る形にすると、逆変換を 2 回掛ける余地が関数の口から無くなり、ADR 0035 の決まり（実空間で計算し、h は完成した target に 1 回）を保てる。旗で逆変換を切り替える形は、旗の渡し違いで 2 回掛かるので採らない。依存: なし |
| D12 | **ADR 0035 への注記。** ステージ 2 で、ADR 0035 の冒頭に注記を足す。内容は、target の組み立てを 1 つの関数にしたことと、`CalcTargetQuantiles` を通常の target 専用とする記述がその関数の口の決まり（実空間の値を受け取る）に置き換わったことである。ADR の決定の中身は変えず、本文も書き換えない。新しい ADR は起こさない | ADR 0035 の 45 行が、消える関数の名前で決まりを書いている（§2.3）。決定（実空間で計算し、h は 1 回）は変わらないので、注記だけで足りる。コミット済みの ADR の本文は書き換えない（[PRD 品質確認観点](../quality/prd.md) の項番 23）。依存: D11 |
| D13 | **学習の中身が同じことの確かめ方。** §5.4 の 31 組の短い Run を、保存した実行体と、直した後の実行体で流し、時間と資源の tag を除く全 scalar の点が一致することを受入にする（A2・A4）。変種の組は、IQN・Munchausen・replay fit の診断の値も記録する設定にして、比べる点に含める。一致しない組が出たら、原因を突き止めて、一致する形へ直す。寄せ方では一致させられないと分かったら、その組と原因を実装記録に書いて作業を止め、扱いを開発者に聞く | 骨格は PER と skip で分かれ、Munchausen の target は方式と TBO で分かれる。分かれ目を 1 回ずつ通す。既定の設定だけでは、通常 Q と PER 無効を通らない（§2.6）。online 方式と skip を重ねた組を置くのは、current の行数が batch の件数と違うときの online の forward を通すため。診断の値は、metrics で購読しないと点にならない。購読しないと、追加の診断の口（D3）や、診断の経路（replay fit の診断は、target を作る関数を診断用に呼ぶ）を書き違えても、一致して見える。時間と資源の tag を除くのは、実行ごとに変わるため。止まる決まりを置くのは、値が変わる寄せ方を G3 の例外として黙って入れないため。依存: D17 |
| D14 | **BF16 の確かめ方。** 31 組は FP32 で流す。加えて、BF16 の組を 4 つ置く。3 つは、LunarLander の「target と TBO と skip」の組（§5.4 の表の最後の行）を BF16 にしたもので、分布の表し方ごとに 1 つ。1 つは、既定の 4 組の Atari の組を BF16 にしたもの。ステージ 0 で、保存した実行体で 2 回ずつ流し、2 回が一致した組だけを前後比較に足す。Atari の組は CUDA で流す。LunarLander の 3 組は、先に CPU で流す。CPU で流せないか、2 回が一致しなければ、CUDA で同じことを試す。流した device のどれでも一致しない組は、前後比較に入れず、今ある単体テスト（`[prd103]` の TD・QR・IQN × Munchausen × BF16）で確かめ、一致しなかったことを実装記録に書く | ユーザーが、Atari の 1 組だけにする形と比べて選んだ（別モデルの検討で出た案）。31 組は AMP が無効なので、AMP の範囲の端を誤って動かしても（skip の forward や PER の重み付けを範囲の外へ出すなど。D5）、値が変わらない。見つけられるのは BF16 の組だけである。Atari の 1 組は IQN だけで、skip の forward を通らない。LunarLander の「target と TBO と skip」の組は、3 つの分布と、skip・Munchausen・TBO を通る。CPU を先に試すのは、小さい Network で、同 seed で再現する見込みが高いため。再現するかは流すまで分からないので、再現しないものは前後比較に入れない。入れると、差の原因を分けられない。依存: D5・D17 |
| D15 | **単体テスト。** 先に固めるテストは、ステージ 0 の 31 組とする。加えて、ステージ 2 で、target の式の共通の関数に、手で計算した値と比べるテストを足す。通す場合は、通常 Q の形と分位点の形、Munchausen項の有無、TBO の有無、終端かどうかの組み合わせの全部である。今あるテストケースの断定は変えない | 丸ごとの値は Run の比較が固める。式の入出力は、名指しで呼ぶテストが今は無いので（§2.4）、足す。Munchausen の評価の関数（D10）は、Network の forward を含むので、Run の比較と今あるテストで確かめる。依存: D11・D13 |
| D16 | **速さの確かめ方。** Release のビルドで、PRD103 の検証資材の性能の測定と同じ構成（Atari の qbert、IQN、Munchausen、BF16、prefetch、評価は止める）を、skip は無効にして、保存した実行体と直した後の実行体で交互に 3 巡以上流す。learn step 1 回あたりの実時間（Actor と Env の時間を含む）を Run ごとに取り、直した後の中央値が、直す前の Run の最大値を超えないことを受入にする（A9）。超えたら、原因を調べて直し、測り直す | 更新は hot path なので、速さが変わらないことをブレ幅で確かめる（§2.5）。交互に流すのは、マシンの速さが時間で動くため。TD と QR も同じ進行を通るので、1 構成で足りる。Munchausen を含む構成にするのは、ステージ 2 が直す経路を通すため。増える処理の見積もりは次のとおり。頻度は、更新 1 回につき 1 回。単価は、仮想関数の呼び出し 2 回と、Tensor の参照 10 個ほどの受け渡しで、合わせて 100 ns に届かない。TD では、プロファイルの区間が 3 つ増える（D9）。QR と IQN が今も持つ区間で、範囲の出入りを記録するだけである。同じ構成の learn step 1 回は 0.04〜0.05 秒で、Run ごとに 15% ほどばらつく（[PRD103 の検証資材](../../core/anet-core/testdata/prd103/README.md) の「性能」の、skip 無効の実測）。増える分は、その 10 万分の 1 に届かない。この判定で拾うのは、Run のばらつきを超える遅れ、つまり作りの誤り（余計な forward、同期の追加など）による遅れである。PER が無効のときは、Tensor の確保と掛け算が 1 回ずつ減る（D6）。依存: D2・D17 |
| D17 | **着手前の保存と検証資材。** ステージ 0 で、着手した時点の作業ツリーから作った Debug と Release の実行体・DLL・設定を `.scratch/prd125/baseline/` に保存する。Git では管理しない。前後比較は Debug で、速さは Release で行う。`core/anet-core/testdata/prd125/` に README と比較のスクリプトを置き、今ある部品（短い条件、隔離した実行、メトリクスの正規化）を呼ぶ | PRD122 と同じ置き方である（§2.5）。比べる相手は、実装の前に保存した実行体にする決まりによる。Debug で比べるのは、検査が動き、形や NaN の誤りをその場で止められるため。部品を呼ぶ形にするのは、写しを増やさないため。依存: なし |
| D18 | **ステージ 0 の確かめ。** 31 組を、保存した実行体で 2 回ずつ流し、どの組も 2 回が一致することを確かめる。一致しない組は、比べる組から外さない。原因を調べ、変種の設定で決定的にできるなら、そうする。できなければ、その組と原因を実装記録に書いて作業を止め、扱いを開発者に聞く | 組が決定的に流れないと、前後の差が直しによるものか分からない。比較の道具が動くことも、直す前に確かめられる。組から外すと、その分かれ目を確かめないまま直すことになる。依存: D13・D17 |
| D19 | **変えないものと文書。** 設定キー、メトリクスの名前と値、公開ヘッダ、checkpoint の形は変えない。設計文書 200 の日英は、`dqn::Learner` の責務の行に、更新の進行と通常 Q の計算を足す。クラス図と、ほかの行は直さない | 直すのは `src/` の内部のヘッダと本体だけである（§2）。クラスと継承は変わらないので、クラス図は今のままで正しい。`TDLearner` などの行は、どの学習器がどの計算を受け持つかを書いていて、直したあとも成り立つ（§5.5）。依存: D2 |

## 5. 実装契約

### 5.1 ステージと直す箇所

| ステージ | ファイルと位置 | 直し方 | 確定事項 |
| --- | --- | --- | --- |
| 0 | `.scratch/prd125/baseline/`、`core/anet-core/testdata/prd125/` | 実行体・DLL・設定を保存し、検証資材を置く。31 組を 2 回ずつ流して一致を確かめる。BF16 の 4 組を 2 回ずつ流す。既定のテストスイートの結果を控える | D14・D17・D18 |
| 1 | dqn_based_agent.cpp の 3 つの `UpdateFromSamples`（4855・4995・5124 行から）、dqn_based_agent.hpp の `Learner` と 4 つの派生クラスの宣言 | 進行を基底へ寄せ、差し替えの口を 2 つ足す。食い違いをそろえる | D2〜D9 |
| 1 | docs/design/200_dqn_agents.jp.md と .en.md | `dqn::Learner` の責務の行を直す | D19 |
| 2 | dqn_based_agent.cpp の `Learner::MakeTarget`（4755 行から）、`QuantileLearnerBase::MakeMunchausenTarget`（4364 行から）、`CalcTargetQuantiles`（4344 行から）、`CalcMunchausenTargetQuantiles`（4435 行から）、dqn_based_agent.hpp の宣言 | Munchausen の評価と target の式を、共通の関数にする | D10・D11 |
| 2 | target の式の単体テスト（置き場は実装計画で決める） | `[prd125]` のテストを足す | D15 |
| 2 | docs/adr/0035-munchausen-target-learner-local-real-space.md の冒頭 | 注記を足す | D12 |

ステージ 0 を終えてから、ステージ 1 に入る。ステージ 1 と 2 は、この順に入れる。

### 5.2 更新の進行

基底の `UpdateFromSamples` は、次を上から順に行う。順は今の 3 つの関数と同じである（D5）。

1. 入力の検査（D7）。コメントアウトされた `obs=` の行（D8）。
2. 観測の正規化。区間 `normalize`。あとに観測の NaN の検査。
3. ここから AMP の範囲に入る。
4. current 側の計算。分布ごとの口（D3）を呼ぶ。区間 `forward_current`。IQN は、この中の最初に current の τ を引く。
5. target を作る。今ある `MakeLearningTargets` を呼ぶ。
6. skip の loss。今ある `ComputeSkipLoss` を呼ぶ。渡す入力と τ は、current 側の計算が返したものを使う。
7. 要素誤差。今ある `ComputeElementError` を呼ぶ。渡す τ、current の平均、補助の統計を求めるかの指定は、current 側の計算が返したものを使う。補助の統計を求めるのは、今と同じく IQN だけである。区間 `loss` はここから 9 の終わりまで。あとに DEBUG ログ `element_loss=`。
8. 追加の診断。分布ごとの口（D3）を呼ぶ。IQN だけが値を返す。
9. PER の重み付けと平均（D6）。PER が有効のとき、掛ける前に DEBUG ログ `weights=`。
10. ここで AMP の範囲を出る。
11. 優先度の準備 → 最適化 → 優先度の反映 → 結果の組み立て。今ある 4 つの関数を、今と同じ引数の中身で呼ぶ。区間 `prepare_per`・`optimize`・`update_per`・`make_result`。

5 と 6 は、進行の区間に入れない。中の関数が自分の区間を持つ（今の QR・IQN と同じ）。

分布ごとの口が返すものは、今の 3 つの関数が進行へ渡している値と同じにする。

値を持たないものは、今と同じく、値の無いものとして渡す。空の値を作って渡さない。当たるのは、通常 Q と QR の追加の診断、通常 Q の上側の広がり、通常 Q の Q の広がりの指標である。優先度の準備と結果の組み立てが、値の有無で、読み戻しを行うかと件数を決めているためである（`PreparePerPriorityUpdate`。3663 行から）。

### 5.3 Munchausen の評価と、target の式

Munchausen の評価の関数は、§2.3 の 1〜7 を、今の並びのまま行う（D10）。

- τ を足すと渡されたときだけ、2 と 6 の τ を作る。作る件数は今と同じである。2 は、target 方式なら current と next の行数の合計、ほかの方式なら next の行数。6 は current の行数。
- 5 の検査と記録は、診断のための呼び出しでは行わない。今と同じである。
- 勾配は作らない。

target の式の関数は、次の形で計算する（D11）。4 か所の今の値と、ビットまで同じになる形である（§2.3）。

- γ の n 乗と「終端でない」旗を先に掛け、その積に bootstrap 価値を掛ける。
- Munchausen項があるときは、target return に Munchausen項を足してから、bootstrap の項を足す。Munchausen項が無いときは、0 を足す形にせず、target return に bootstrap の項を足す。
- 掛け算と足し算の組み方は、上の 2 つのとおりにする。組み替えると、値がビットまで同じにならない。
- bootstrap 価値は、呼ぶ側が今と同じ変換を済ませたものを、そのまま使う。関数の中では変換しない。
- bootstrap 価値は、サンプルごとに 1 つ（通常 Q）でも、サンプルごとに分位点の数だけ（分位点）でも受ける。
- TBO が有効なら、完成した target に h を 1 回掛ける。h の逆変換は、関数の中では行わない。
- 勾配は作らない。
- 今 `CalcTargetQuantiles` にある、target の形と NaN の検査は、共通の関数に置く。ある分布の今の値で止まるなら、D7 と同じく、元の箇所に戻す。

### 5.4 前後比較の Run の組

条件は、PRD122 の短い Run と同じである。seed 1、8 lane、batch 16、warmup 256、replay ratio 1、8,704 experience、評価は止める、FP32、決定的な backend。

既定の 4 組は、GridMaze・LunarLander・Atari（qbert）・DropMerge の、PRD122 と同じ構成である。GridMaze と LunarLander は CPU、Atari と DropMerge は CUDA で流す。

変種の 27 組は、LunarLander（CPU）で、分布の表し方 3 つのそれぞれに、次の 9 組を流す。

| 組 | `learner.use_per` | `learner.use_tbo` | `skip.enabled` | `learner.munchausen.enabled` | `learner.munchausen.log_policy_mode` |
| --- | --- | --- | --- | --- | --- |
| 基準 | true | false | false | false | — |
| PER 無効 | false | false | false | false | — |
| TBO | true | true | false | false | — |
| skip | true | false | true | false | — |
| Munchausen target | true | false | false | true | target |
| Munchausen online | true | false | false | true | online |
| Munchausen online_reuse | true | false | false | true | online_reuse |
| online と skip | true | false | true | true | online |
| target と TBO と skip | true | true | true | true | target |

- 分布の表し方は `quantile_mode` で、`none`（通常 Q）・`qr`・`iqn` の 3 つである。キーはどれも `DefaultDQNAgent.` の下にある。
- 変種の組は、LunarLander の既定の metrics の連鎖を保つ。この連鎖は、IQN の診断 7 値の tag（`36_agent_iqn/01`〜`36_agent_iqn/07`）を定義する `M1` を含む（[LunarLander.txt](../../apps/runner/config/LunarLander.txt) の 437〜450 行）。そのうえで、次の profile を重ねる。変種の 27 組のすべてに `metrics.scalar.@replay_fit`。Munchausen が有効の組に `metrics.scalar.@munchausen`。skip が有効の組に `metrics.scalar.@skip`。短い Run の中で値のある点が出るように、間隔を変種の設定で上書きしてよい。
- 表のキーのほかは、構築できる値へ、変種の設定で明示して上書きしてよい（Munchausen と併用できない `learner.use_double_dqn`、skip が読む特徴の配線など）。直す前と後に同じ凍結した設定を渡すので、比較は成り立つ。
- 各 Run の設定の dump で、表のキーと `quantile_mode` が表の値になっていることを確かめる（A1）。
- BF16 の 4 組（D14）は、上の組の設定から FP32 への上書きを外し、`learner.use_amp` と `learner.use_amp_bf16` を true にしたものである。もとにする組は、変種の「target と TBO と skip」の組（分布の表し方ごとに 1 つ）と、既定の 4 組の Atari の組である。

### 5.5 移行対象と、触らないもの

変える名前と振る舞いを、追跡しているファイルの全体から検索した（2026-11-12）。検索語は、関数とクラスの名前（`UpdateFromSamples`・`CalcTargetQuantiles`・`CalcMunchausenTargetQuantiles`・`MakeMunchausenTarget`・`TDLearner`・`QRLearner`・`IQNLearner`・`QuantileLearnerBase`）と、区間の名前（`forward_current`・`prepare_per`・`update_per`・`make_result`・`weighted_loss`・`generate_current_taus`・`iqn_diagnostics`）である。

直すもの。

| 対象 | 直し方 | ステージ |
| --- | --- | --- |
| core/anet-core/src/dqn_based_agent.cpp・dqn_based_agent.hpp | §5.1〜§5.3 | 1・2 |
| docs/design/200_dqn_agents.jp.md の 287 行、.en.md の対応する行 | `dqn::Learner` の責務に、1 回の更新の進行と、通常 Q の計算を足す | 1 |
| docs/adr/0035-munchausen-target-learner-local-real-space.md の冒頭 | 注記を足す（D12） | 2 |
| core/anet-core/testdata/prd125/ | 新しく置く（D17） | 0 |
| target の式の単体テスト | 新しく足す（D15） | 2 |

触らないもの。

| 見つかった箇所 | 件数 | 触らない理由 |
| --- | --- | --- |
| 関数とクラスの名前を呼ぶ・上書きするテスト（dqn_based_agent_test.cpp、dqn_munchausen_test.cpp） | 2 ファイル | 今あるテストは変えずに通す（D4・D15） |
| 学習器を構築する箇所と前方宣言（default_dqn_agent.cpp の 6 行、rainbow_agent.cpp の 4 行、include/anet/agent.hpp の 2 行） | 3 ファイル | クラスと、構築の口は変わらない |
| 過去の記録と課題の記録（CHANGELOG.md、core/anet-core/testdata/prd084 の 2 ファイル、docs/experiments の 1 ファイル、docs/management/issues の 2 ファイル） | 6 ファイル | 当時の記録と、課題の道具が作る一覧である。題材の課題への追記は A11 で行う |
| 設計文書 200 の日英の、`TDLearner`・`QRLearner`・`IQNLearner`・`QuantileLearnerBase` を書く行（責務の表の 288・289 行、手法の表、TBO の実装の案内、クラス図、更新の流れの図） | 日英それぞれ 20 行 | どの学習器がどの計算を受け持つかを書いていて、直したあとも成り立つ。クラスと継承は変わらない（D19） |
| 設計文書 200 の日英の、`UpdateFromSamples` を書く行 | 日英それぞれ 1 行 | 「`UpdateFromSamples` の前に測る」という記述で、直したあとも成り立つ |
| ADR 0035 の 5 行（3 つの学習器の target の計算へ局所化する） | 1 行 | 直したあとも成り立つ |
| 地図（docs/wiki）の、題材の課題を載せる行 | 5 ページ | 課題の道具が作り直す |
| 地図（docs/wiki）の「成立の経緯」の、`IQNLearner` を書く行 | 3 ページ | 過去の PRD の題を写した行で、直したあとも成り立つ |
| 区間の名前を読む設定・スクリプト・現行の文書 | 0 件 | — |
| apps/runner/config、tools、reports の中の、関数とクラスの名前 | 0 件 | — |

過去の PRD と実装記録（docs/memo の下）は、当時の記録なので直さない。

### 5.6 クラス一覧

従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場 |
| --- | --- | --- | --- |
| `dqn::Learner` | 変更 | 1 回の更新の進行（`UpdateFromSamples` の実装）を持つ。通常 Q の current 側の計算（`ForwardCurrent`）、追加の診断の口（`MakeUpdateDiagnostics`。基底は値を返さない。§5.2）、Munchausen の評価（`ForwardMunchausenValues`）、target の式（`MakeBellmanTarget`）を足す | dqn_based_agent |
| `dqn::CurrentForwardResult` | 新規 | current 側の計算が進行へ返す値をまとめる。forward の入力と出力、損失に使う値、τ、current の平均、補助の統計を求めるかの指定、指標、上側の広がり | dqn_based_agent |
| `dqn::MunchausenForwardResult` | 新規 | Munchausen の評価が返す、current と next の値をまとめる | dqn_based_agent |
| `dqn::QuantileLearnerBase` | 変更 | 分位点の current 側の計算を上書きする。`MakeMunchausenTarget` と通常の target は、共通の評価と式を呼ぶ。`CalcTargetQuantiles` と `CalcMunchausenTargetQuantiles` を消す | dqn_based_agent |
| `dqn::TDLearner` | 変更 | `UpdateFromSamples` の実装を消す。構築だけを持つ | dqn_based_agent |
| `dqn::QRLearner` | 変更 | `UpdateFromSamples` の実装を消す。固定の分位点の位置と、それを補う `ComputeElementError` は残す | dqn_based_agent |
| `dqn::IQNLearner` | 変更 | `UpdateFromSamples` の実装を消す。current 側の計算のうち τ に関わる部分と、IQN の診断を持つ | dqn_based_agent |

### 5.7 issue の対応表

| issue | 題 | area | 現在のコードで確かめた根拠の位置 | 直す確定事項 | 確かめる受入 |
| --- | --- | --- | --- | --- | --- |
| [issue-20261103-ab98b7](../management/issues/issue-20261103-ab98b7.md) | TD・QR・IQN の UpdateFromSamples が同じ骨格を丸写ししている | 321_agent_core | [dqn_based_agent.cpp](../../core/anet-core/src/dqn_based_agent.cpp) の 4855・4995・5124 行から（3 つの関数）、4364・4769 行から（Munchausen の前半）、4355・4440・4814・4846 行（target の式） | D2〜D11 | A2〜A5 |

## 6. 検証と受入

「保存した実行体」は、ステージ 0 で保存したものを指す。「全 scalar の点が一致する」は、時間と資源の tag（`90_perf/`・`91_resource/`・`92_resource/` で始まるもの）を除く全 scalar の点を、tag・step・値で比べて、すべて同じであることを指す。除くのは、実行ごとに変わるためである。

検索の受入（A3・A5）は、固定文字列で引き、当たった行の数を数える。範囲は項ごとに書く。表の「直す前」の数は、2026-11-12 の作業ツリーで数えた。着手した時点で数が違っていたら、着手した時点の数から、表の「直す前」と「直した後」の差と同じだけ減ることを確かめ、着手した時点の数を実装記録に書く。

### A1. ステージ 0（D14・D17・D18）

- `.scratch/prd125/baseline/` に、Debug と Release の実行体・DLL・設定がある。保存した時点の commit、作業ツリーの差分の有無、各ファイルの SHA-256 を書いた一覧がある。
- §5.4 の 31 組を、保存した Debug の実行体で 2 回ずつ流し、どの組も、2 回の全 scalar の点が一致する。
- 31 組のうち変種の 27 組は、Run の設定の dump（`config/config_data.txt`）で、§5.4 の表のキーと `quantile_mode` が表の値になっている。
- 変種の 27 組は、比べる点の中に、診断の tag の点がある。組ごとに次を確かめる。
  - 27 組のすべて: `46_agent_replay_fit/` で始まる tag（replay fit の診断）に、NaN でない値の点が 1 つ以上ある。
  - Munchausen が有効の組: `36_agent_munchausen/` で始まる tag（Munchausen の診断）に、NaN でない値の点が 1 つ以上ある。
  - `quantile_mode` が `iqn` の組: `36_agent_iqn/01`〜`36_agent_iqn/07` で始まる 7 つの tag（IQN の診断 7 値）のすべてに点があり、7 つのうち 1 つ以上が NaN でない値の点を持つ。
- BF16 の 4 組を、保存した Debug の実行体で 2 回ずつ流した結果が、検証資材の出力にある。組ごとに、流した device と、2 回が一致したかが書かれている。LunarLander の組が CPU で一致しなかったときは、CUDA で流した結果もある。
- BF16 の組は、Run の設定の dump で、`learner.use_amp` と `learner.use_amp_bf16` が true になっている。
- 保存した Debug のビルドで既定のテストスイート（`anet-core-test`）を流し、落ちるテストケースの名前の一覧を控えてある。
- `core/anet-core/src/dqn_based_agent_test.cpp` と `dqn_munchausen_test.cpp` の、着手した時点の写しを保存してある。

### A2. ステージ 1 の前後比較（D13）

- 31 組を、保存した Debug の実行体と、ステージ 1 を終えた Debug の実行体で流し、どの組も、全 scalar の点が一致する。両方へ、同じ凍結した設定を渡す。
- BF16 の 4 組のうち、A1 で 2 回が一致した組は、同じ device で流して、同じく一致する。

### A3. ステージ 1 の形（D2・D6〜D9）

| 範囲 | 検索する文字列 | 直す前 | 直した後 |
| --- | --- | --- | --- |
| dqn_based_agent.cpp | `::UpdateFromSamples(` | 3 | 1 |
| dqn_based_agent.hpp | `UpdateFromSamples(` | 4 | 1 |
| dqn_based_agent.cpp | `PreparePerPriorityUpdate(` | 5 | 3 |
| dqn_based_agent.cpp | `ApplyPerPriorityUpdate(` | 5 | 3 |
| dqn_based_agent.cpp | `MakeBatchUpdateResult(` | 4 | 2 |
| dqn_based_agent.cpp | `prepare_per` | 3 | 1 |
| dqn_based_agent.cpp | `ANET_ASSERT_SHAPE(samples.actions, { B })` | 2 | 1 |
| dqn_based_agent.cpp | `torch::ones({ B }, device_)` | 2 | 0 |
| dqn_based_agent.cpp | `weighted_loss` | 1 | 0 |
| dqn_based_agent.cpp | `ANET_LOG_DEBUG("element_loss="` | 1 | 1 |
| dqn_based_agent.cpp | `ANET_LOG_DEBUG("weights="` | 1 | 1 |

### A4. ステージ 2 の前後比較（D13）

- A2 と同じ比較を、ステージ 2 を終えた Debug の実行体で行い、どの組も一致する。

### A5. ステージ 2 の形（D10・D11）

| 範囲 | 検索する文字列 | 直す前 | 直した後 |
| --- | --- | --- | --- |
| dqn_based_agent.cpp | `Munchausen target capture expected rows=` | 2 | 1 |
| dqn_based_agent.cpp | `forward_munchausen_online` | 2 | 1 |
| dqn_based_agent.cpp | `torch::pow(gamma_` | 4 | 1 |

### A6. target の式の単体テスト（D15）

- `[prd125]` のテストが、Debug のビルドで通る。通常 Q の形と分位点の形、Munchausen項の有無、TBO の有無、終端かどうかの 16 通りで、target が手で計算した値と一致する。
- 共通の関数を一時的に 3 通りに変えると、どれでもテストが落ちる。h を掛けない。終端の旗を外す。終端のとき Munchausen項を消す。落ちたテストと、戻したあとに通ることを、実装記録に残す。

### A7. 今あるテスト（D4・D15）

- ステージ 1 と 2 のそれぞれのあとに、Debug のビルドで既定のテストスイートを流し、落ちるテストケースの名前の一覧が、A1 で控えたものと同じである。
- A1 で保存した 2 つのテストファイルの写しと、実装のあとの同じファイルを比べ、消えた行と変わった行が無い。足した行は `[prd125]` のテストだけである。並行する別の作業の変更が混ざるときは、その行を名指しして除き、実装記録に書く。

### A8. ビルド

- Debug と Release のビルドが通る。

### A9. 速さ（D16）

- ステージ 1 のあとと、ステージ 2 のあとに、それぞれ測る。Release のビルドで、D16 の構成を、保存した実行体 → 直した後の実行体の順に、3 巡以上流す。測る前に、どちらかの実行体で 1 本流して捨てる。測っているあいだは、ほかの学習とビルドを重ねない。
- Run ごとに、warmup のあとの learn step 64 から 256 までにかかった実時間を 192 で割り、learn step 1 回あたりの秒数にする。Actor と Env の時間を含む。測り方は PRD103 の検証資材と同じで、経過時間を learn step ごとに記録する tag を、測る Run の設定で足す。
- 直した後の Run の中央値が、保存した実行体の Run の最大値を超えない。
- Run ごとの秒数、実行体ごとの中央値、判定が、検証資材の出力にある。

### A10. 文書（D12・D19）

- docs/design/200_dqn_agents.jp.md と .en.md の `dqn::Learner` の責務の行に、1 回の更新の進行と、通常 Q の計算が書かれている。
- `git diff HEAD -- docs/adr/0035-munchausen-target-learner-local-real-space.md` の差分が、冒頭への注記の追加だけである。消えた行が無い。
- `core/anet-core/testdata/prd125/README.md` に、保存の手順、比較の実行例、31 組の定義の場所が書かれている。

### A11. 課題の記録

- 題材の課題の「対応内容」に、実装の日付、直した内容、前後比較と速さの結果が追記されている。状態は、実装する側からは解決にしない。

## 7. 作らないもの・先送り

| 選ばなかった形 | 裁定 | 理由 |
| --- | --- | --- |
| 末尾（PER の重み付けから結果の組み立てまで）だけを共通の関数にする | 不採用 | 正規化から要素誤差までの順が 3 か所に残り、題材の課題が片付かない（D2） |
| 進行を、関数を受け取る 1 つの補助にして、分布ごとの違いを渡す | 不採用 | target と要素誤差が、仮想関数の上書きの型である。型が 2 つになる（D3） |
| `UpdateFromSamples` を仮想でなくする | 不採用 | テスト用の派生 3 つの作り直しが要る（D4） |
| QR だけの DEBUG ログ 2 行を消す | 不採用 | 選んで残してあるログで、通常の Run に影響しない（D8） |
| DEBUG ログを QR のときだけ出す | 不採用 | 進行に、分布の名前での分岐が要る（D8） |
| PER が無効のとき、1 を掛ける形にそろえる | 不採用 | 値は同じで、Tensor の確保と掛け算が増える（D6） |
| target の式の関数が h 空間の値も受け取り、旗で逆変換を切り替える | 不採用 | 旗の渡し違いで、逆変換が 2 回掛かる（D11） |
| Munchausen の、方策と soft価値ブートストラップを組む部分も 1 つにする | 不採用 | 通常 Q と分位点で、中身が違う（D10） |
| 前後比較を Release だけで行う | 不採用 | 検査が動かず、形や NaN の誤りを値の差でしか見つけられない（D17） |
| 一致しない組を、比べる組から外す | 不採用 | その分かれ目を確かめないまま直すことになる（D18） |
| 価値の分解のための、Head ごとの目標の口 | 範囲外 | PRD121 で作る。再開条件: PRD121 のグリルで口の形が決まったとき（§3.4） |
| TD・QR・IQN のクラスを 1 つにまとめる | 範囲外 | 写しを無くすのに要らない。本 PRD のあとでも行える（§3.2） |
| replay fit の診断の進行の前半を、学習の進行と 1 つにする | 範囲外 | forward の仕方・τ の置き方・指標の有無が学習と違う（§3.2） |
| 3 つの学習器に、基底の `UpdateFromSamples` を呼ぶだけの上書きを書く | 不採用 | 仕事をしない層になる（D4） |
| 入力の行数の検査を進行に置かず、直前の `ValidateDeviceSamples` に任せる | 不採用 | テストが `UpdateFromSamples` を直接呼ぶときに掛からなくなり、QR と IQN の今の動きが変わる（D7） |

## 8. 未決

未決は無い。実装の中で分かることは、BF16 の Run が同 seed で再現するかの 1 つで、結果ごとの扱いを D14 に決めてある。

## 9. 依存と順序

- 題材の課題を待つ課題がある。PRD924（内的報酬）の課題は、題材の課題に依存している。PRD121（価値の分解）は、本 PRD を先に入れるかを、PRD121 の側の未決にしている。本 PRD が先に入れば、PRD121 は target と要素誤差のあいだの 1 か所へ足せる。
- PRD121・PRD924・PRD957 は、同じファイルの同じ関数を動かす。本 PRD の実装と同時には進めない。
- dqn_based_agent.cpp には、並行する作業の未コミットの変更がある。比べる相手は、着手した時点の作業ツリーから作った実行体である（D17）。ステージの途中で、学習の値を変える別の変更を同じ作業ツリーへ入れない。入れると、前後の差の原因を分けられない。

## 10. 参照

- [リファクタリング観点](../quality/refactor.md)（ガードレール、運用ルール）
- [地図 321 Agent 共通基盤](../wiki/categories/321_agent_core.md)（変えるときの条件）
- [ADR 0035](../adr/0035-munchausen-target-learner-local-real-space.md)（Munchausen の target の決まり）
- [設計文書 200 DQN 系 Agent](../design/200_dqn_agents.jp.md)（責務の表）
- [PRD122 の検証資材](../../core/anet-core/testdata/prd122/README.md)、[PRD103 の検証資材](../../core/anet-core/testdata/prd103/README.md)、[PRD098 の検証資材](../../core/anet-core/testdata/prd098/README.md)
