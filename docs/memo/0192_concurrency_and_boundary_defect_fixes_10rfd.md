# RFD0192: 監査で見つかった不具合 5 件をまとめて直す（Notifier の同時実行、動画の画像の大きさ、強制行動の device、重複登録、QValuePanel の平滑化）

> 2026-11-19 起票（anet-create-prd、Claude Code）。題材は課題 5 件（§4 の対応表）。グリルで、ゴールと非ゴール（§2）、Detach の保証（D2）、動画の大きさ違いの扱い（D8）、初期化関数を 2 回呼んだときの扱い（D11）、強制行動の device を合わせる側（D9）をユーザーが選び、細部 18 件（§10）は推奨の既定で埋めた。起草のあと、別モデル（Opus）の検討を 2026-11-20 に受け、分かった事実と、決定から導ける事項を足した（§10 の末尾）。別モデルが反対した 1 点（強制行動を差し替えた結果の行動の device）は、ユーザーが選び直した（D9）。工程末の起票の候補では、ユーザーの選択により、Runner が Observer を付ける箇所のコメントを D7 に足した（§10）。

> 関連: 設計 [140](../design/140_observability.jp.md)・[110](../design/110_agents_and_learning.jp.md)・[120](../design/120_environments.jp.md)・[130](../design/130_neural_networks.jp.md)・[200](../design/200_dqn_agents.jp.md)、ADR [0009](../adr/0009-imagecls-batch-env-seam.md)・[0082](../adr/0082-notifier-detach-does-not-wait-for-in-flight-notifications.md)、[146_notifier_scoped_detach_10prd.md](0146_notifier_scoped_detach_10rfd.md)、[161_observation_pipeline_10prd.md](0161_observation_pipeline_10rfd.md)

## 1. 問題

### 1.1 経緯

2026-07 の監査、2026-11 の監査、RFD0122 の再グリルで見つかった不具合・不整合のうち、範囲外の読み・データ競合・黙った上書き・誤った停止に当たる 5 件が残っている。どれも小さく、互いに独立に直せるので、1 本の RFD にまとめる。

課題の事象に書かれたパス（`core/anet-core/...`）と行番号は、[RFD0182](0182_source_layout_10rfd.md)（ソース配置の見直し）より前のものである。本書の根拠は、2026-11-19 に今のコードで確かめ直した。

### 1.2 今のコードで確かめたこと

行番号は 2026-11-19 のもの。

| 課題 | 今のコード |
|---|---|
| 80c355 | `Notifier` は 4 種類の Observer の列を持ち、排他を持たない（[rl.hpp](../../src/core/base/include/anet/rl.hpp) 1177・1262〜1265 行）。`Attach`（[rl.cpp](../../src/core/base/src/rl.cpp) 966 行から）・`Detach`（990 行から）・`Clear`（1094 行）・`Shutdown`（1102 行）・`WillBlockOnShutdown`（1111 行）・`Notify`（1125 行から）・`LogObservers`（1161 行）が、同じ列を直接読み書きする |
| 80c355 | 学習中に利用者がメニューから Panel を開くと、GUI スレッドが `Attach` する。入口は [RunnerFrame.cpp](../../src/apps/runner/src/RunnerFrame.cpp) の `OnHeatMap`（1485 行。1499 行で Panel を作り、[HeatMapPanel.cpp](../../src/apps/runner/src/HeatMapPanel.cpp) 484 行で `Attach`）と `OnConv2d`（1525 行。1531 行で Panel を作り、[Conv2dPanel.cpp](../../src/apps/runner/src/Conv2dPanel.cpp) 79 行で `AttachScoped`） |
| 80c355 | `Notify` を呼ぶのは [trainer.cpp](../../src/core/base/src/trainer.cpp) の 8 か所（106・333・379・418・621・630・710・719 行）。学習の step を進めるスレッド（621・630・710・719 行）、評価の背景スレッドと GUI スレッド（EvalPanel の 1 step。どちらも 333 行の `EvalRunner::DoStepInternal` を通る）の 3 つのスレッドから呼ばれる |
| 80c355 | const メソッドの裏で書くキャッシュが 3 つある。`BatchStepResult::GetEpisodeEndMask` は 1 回だけ実行する守りを持つ（[rl.cpp](../../src/core/base/src/rl.cpp) 1193 行、[rl.hpp](../../src/core/base/include/anet/rl.hpp) 609 行）。`ReplayInitialPriorityHint::GetPayloadCpu`（rl.cpp 399 行、rl.hpp 512 行）と `BatchActionInfo::GetAction(device)`（rl.cpp 439 行、rl.hpp 571・572 行）は守りを持たない |
| 80c355 | 守りの無い 2 つを、同じ実体について 2 つのスレッドが同時に呼ぶ経路は無い。逐次の Runner は 1 つのスレッドで、行動の選択・Env の step・学習・通知を順に行う（[trainer.cpp](../../src/core/base/src/trainer.cpp) 574〜630 行）。非同期の Runner は、前の step の行動の情報を学習のスレッドへ渡し（739 行）、学習の終わりを待ってから（704 行）通知に使う（717 行）。その間、step を進めるスレッドが触るのは今の step の行動の情報だけである（729・747 行）。通知で受けた行動の情報を、あとで読むために抱える部品は、非同期の Runner（[trainer.hpp](../../src/core/base/include/anet/trainer.hpp) 231 行）と、EvalTracePanel の行（[EvalTracePanel.cpp](../../src/apps/runner/src/EvalTracePanel.cpp) 398 行）である。EvalTracePanel は、EvalPanel の Runner の通知も、あとの読み出し（[gui.cpp](../../src/core/base/src/gui.cpp) 746・773 行）も GUI スレッドの上で行う。確かめたのは、行動の情報を持つメンバの宣言と、通知の行動の情報を別の変数へ入れる代入を `src` で引いた範囲である。キャッシュを書きうる口は、`GetAction(device)` のほかに、`GetScalar` の `action`（[rl.cpp](../../src/core/base/src/rl.cpp) 416 行で `GetAction` を呼ぶ）がある |
| 80c355 | 今、学習中に Observer の列から実際に外れる Observer は無い。`Detach` が効く 2 つ（EvalPanel が学習側の step 数を受ける Observer、RunnerFrame の状態表示用の Observer）は、学習の停止と背景評価の排水のあとに外す（[RunnerFrame.cpp](../../src/apps/runner/src/RunnerFrame.cpp) 1735・1762・1764・1768 行）。`AttachScoped` で付けた 5 つ（Conv2dPanel、EvalPanel の View 用、EvalTracePanel、QValuePanel、TrainPanel）は、`Detach` しても外れない（P3）。HeatMap の Panel の Observer は `Attach` で付けるが、`Detach` を呼ばない。`Clear` を呼ぶのは、最終の終了で学習のスレッドを止めたあとの 1 か所である（[RunnerApp.cpp](../../src/apps/runner/src/RunnerApp.cpp) 840〜842 行、[trainer.cpp](../../src/core/base/src/trainer.cpp) 493 行） |
| 80c355 | Runner の Panel と RunnerFrame が Observer を付けるのは 8 か所である（2026-11-20 に、`src/apps/runner/src` のテスト以外を `Attach(`・`Attach<`・`AttachScoped<` で引いて 10 行。残る 2 行は RunnerApp の、何も抱えない Observer である）。EvalPanel の Runner の通知を受ける 4 つ（[Conv2dPanel.cpp](../../src/apps/runner/src/Conv2dPanel.cpp) 79 行、[EvalPanel.cpp](../../src/apps/runner/src/EvalPanel.cpp) 140 行、[EvalTracePanel.cpp](../../src/apps/runner/src/EvalTracePanel.cpp) 221 行、[QValuePanel.cpp](../../src/apps/runner/src/QValuePanel.cpp) 294 行）は、Panel を触る。この Runner を進めるのは GUI の処理だけなので（EvalPanel.cpp 200・217・315 行、QValuePanel.cpp 966 行）、4 つとも GUI スレッドの上で動く。学習の Runner の通知を受ける 3 つ（[TrainPanel.cpp](../../src/apps/runner/src/TrainPanel.cpp) 57 行、EvalPanel.cpp 139 行、[RunnerFrame.cpp](../../src/apps/runner/src/RunnerFrame.cpp) 607 行）は、学習スレッドから Panel か Frame を触る。TrainPanel は窓と同じ寿命で、窓を閉じる処理は先に学習を止める（RunnerFrame.cpp 1735 行）。残る 2 つは、学習の停止と排水のあとに外す（6 行目）。HeatMap の Panel の Observer（[HeatMapPanel.cpp](../../src/apps/runner/src/HeatMapPanel.cpp) 484 行）は、すべての Runner の通知を受けるが、Panel を触らない。画像を Observer が自分で持ち、Panel が読みに行く。学習中に閉じうる Panel は、メニューから作る Conv2d と HeatMap の 2 つで（RunnerFrame.cpp 1485・1525 行）、どちらも、別のスレッドから Panel を触られない |
| ba635a | `VideoLogger::WriteFrame` は、渡された画像から、構築時の幅×高さ×3 バイトを読む（[metrics_logger.cpp](../../src/core/base/src/metrics_logger.cpp) 508・509 行）。画像の大きさは比べていない。VideoLogger は tag の最初の画像の大きさで作られる（787 行）。以後は同じ tag の画像をそのまま渡す（816 行） |
| ba635a | 動画に渡る画像の大きさは、設定の値か、データの形から決まる（[image.cpp](../../src/core/base/src/image.cpp) 18 行の `ImageSource::Render`、[observers.cpp](../../src/core/base/src/observers.cpp) 107・171・746 行の記録の呼び出し）。窓の大きさからは決まらない。Run の途中で大きさが変わる経路は、読んだ範囲（この 3 か所の呼び出し）では見つけていない。ほかに呼び出しが無いことまでは確かめていない |
| ba635a | 画像を記録する口は、画像が有効かを確かめない（[metrics_logger.cpp](../../src/core/base/src/metrics_logger.cpp) 749〜816 行）。無効な画像では、wx は幅と高さに 0、画素のデータに空を返す（wxWidgets 3.3.3 の `src/common/image.cpp` 1950・2083 行。2026-11-20 に確認）。VideoLogger を作ったあとに無効な画像が来ると、空のデータから構築時の大きさぶんを読む。最初の画像が無効だと、0×0 の大きさで VideoLogger が作られる |
| 93f8a6 | EvalRunner は強制行動を CPU で作り、`WithAction` へ渡す（[trainer.cpp](../../src/core/base/src/trainer.cpp) 293〜297 行）。DQN 系の `WithAction` は、初期優先度のヒントがあるとき、渡された行動と Q 値の device が違えば止める（[dqn_actor.cpp](../../src/algo/dqn/src/dqn_actor.cpp) 204〜207 行）。そのあと、行動の Q・Munchausen の項・内的の Q を、渡された行動の値に取り直す（208〜216 行） |
| 93f8a6 | ヒントは、`use_per` かつ初期優先度の方式が `actor_approx` なら、Actor の役割に依らず有効になる（[dqn_assembler.cpp](../../src/algo/dqn/src/dqn_assembler.cpp) 383・384 行）。評価の Actor の device が GPU のとき、Q 値は GPU にあり、強制行動で止まる。実行では確かめていない（CUDA が要る）。基底と MuZero の `WithAction` は行動を差し替えるだけで、device を問わない（[rl.hpp](../../src/core/base/include/anet/rl.hpp) 551 行） |
| 93f8a6 | 行動の情報は、行動の CPU の写しと GPU の写しを別々に持ち、どちらが元かを記録しない。引数なしで行動を取ると、CPU の写しがあればそれが返る（[rl.hpp](../../src/core/base/include/anet/rl.hpp) 535〜542・571・572 行）。CPU の写しは、誰かが CPU で取り出した時点で作られる（[rl.cpp](../../src/core/base/src/rl.cpp) 444〜450 行） |
| 1499d0 | 登録済みの名前で登録すると、Agent（[agent.cpp](../../src/core/base/src/agent.cpp) 49 行）・View（[gui.cpp](../../src/core/base/src/gui.cpp) 392 行）・NN module（[nn_impl.cpp](../../src/core/base/src/nn_impl.cpp) 2482 行）は上書きする。Env だけが `ANET_SYSTEM_ERROR` で止める（[env.cpp](../../src/core/base/src/env.cpp) 1137〜1155 行。[ADR 0009](../adr/0009-imagecls-batch-env-seam.md) が決めた） |
| 1499d0 | 登録は、RFD0182 で、ライブラリが公開する初期化関数をアプリが呼ぶ形になった。初期化関数は 10 個ある（§6.5）。製品は起動時に 1 回ずつ呼ぶ（[RunnerApp.cpp](../../src/apps/runner/src/RunnerApp.cpp) 393〜405 行。`InitNN` は 393 行の `InitRL` の中、[init.cpp](../../src/core/base/src/init.cpp) 168 行）。課題の「静的初期化の順で後のものが勝つ」は、今は起きない |
| 1499d0 | テストは同じプロセスの中で初期化関数を何度も呼ぶ。テストから `InitNN` に行き着く呼び出しは 109 か所ある（`InitNN()` の直接の呼び出し 38、テスト用の `RegisterStandardNetworkModules()` 63、`InitRL(` 8。2026-11-19 に `src` 全体を引き、`_test.cpp`・`_test.hpp`・`/test/` のファイルに絞って数えた）。Env の初期化関数は、Env の登録と View の登録を続けて行い（[GridMaze.cpp](../../src/env/gridmaze/src/GridMaze.cpp) 9〜13 行）、今は 2 回呼ぶと Env の重複で止まる |
| 1499d0 | 製品の登録に、同じ名前の重複は無い。`InitNN` が登録する type 名は 29 個で、どれも別である（[nn_modules.cpp](../../src/core/base/src/nn_modules.cpp) 3309 行から）。Agent の class ID は 4 つ（`DefaultDQNAgent`・`RainbowAgent`・`ImageClsAgent`・`MuZeroAgent`）、View の class ID は 6 つ（Env の class ID と同じ）で、どれも別である（2026-11-20 に、`src/algo`・`src/env` の `GetTargetAgentClassId`・`GetTargetClassId` の返す値を引いた） |
| 3f4e63 | `QValuePanel.smooth_radius` は小数として読まれ、値域の検証が無い（[QValuePanel.cpp](../../src/apps/runner/src/QValuePanel.cpp) 30・40 行）。平滑化の重みは、`use_hist_smooth` が真で半径が正のときだけ作る（138 行）。加工は `use_hist_smooth` だけを見て重みを読む（545〜556 行）。加工への入力（`QValueHeatMapRequest`）は、有効かどうか・半径・重みの 3 つを別々に持つ（[QValuePanel.hpp](../../src/apps/runner/src/QValuePanel.hpp) 69〜71 行） |
| 3f4e63 | 出荷設定は `use_hist_smooth` が真、`smooth_radius` が 5 である（[common.txt](../../apps/runner/config/common.txt) 65・66 行）。追跡しているファイルにある設定の `smooth_radius` の値は、どれも 5 である（2026-11-19 に `git grep` で、実験記録の設定の写し 560 行、検証資材の設定の写し 11 行と基準 17 行）。設定の整数の読み口は、小数や余りの文字があると止める（[config.cpp](../../src/core/base/src/config.cpp) 82 行） |

### 1.3 Notify の負荷の見積もり

測ってはいない。見積もりである。

- 今の `Notify` は、Observer 1 個ごとに、参照カウントを 1 回上げ下げしている（[rl.cpp](../../src/core/base/src/rl.cpp) 1129 行）。
- 直近の Atari の Run（`run_20261119-202718_mz_tg_g99ne_rr4_s2`）では、Observer は TRAIN 19・LEARN 134・EPISODE_END 27・SESSION_END 54 個付いていた（ログの `Notifier:` の行を数えた）。
- D5 の形では、`Notify` 1 回に足すのは、ロック 1 回と Observer の一覧の取得 1 回である。見込みは 0.1 μs 前後で、Observer の数に依らない。一覧を 1 回取って回せば、Observer ごとの参照カウントの上げ下げは要らなくなるので、その分は今より減る。
- `Notify` は、行動の 1 step ごと（TrainEvent）と、学習のあった step ごと（LearnEvent）に呼ばれる。1 step は NN の forward と Env の step を含むので、足す量は 1 step の 0.01% 程度の見込みである。

## 2. ゴールと非ゴール

解く問題: §1 の 5 件が残っている。

ゴール:

1. `Notifier` の `Attach`・`Detach`・`Notify` を別々のスレッドから同時に呼んでも、Observer の列が壊れない。呼んでよいスレッドと時点が、`Notifier` と関係するクラスに書かれている。
2. VideoLogger が、構築時と違う大きさの画像で範囲外を読まない。
3. `actor_approx` かつ GPU 推論の評価 Actor に強制行動を渡しても止まらず、ヒントは渡した行動の値になる。
4. Agent・View・NN module の同名の重複登録が、黙って上書きされない。
5. QValuePanel で、平滑化が有効で半径が 0 以下の設定でも、範囲外を読まない。
6. 題材の 5 件がすべて片付く。既定の設定の Run の結果（メトリクス）は変えない。

足す仕組みと、当たるゴール:

| 足す仕組み | ゴール |
|---|---|
| Observer の列を守る排他（D1〜D5） | 1 |
| 呼んでよいスレッドと時点のコメント（D7） | 1 |
| 動画の画像の大きさの確かめ（D8） | 2 |
| 強制行動の device 合わせ（D9） | 3 |
| 重複登録の検査と、初期化関数の 1 回だけの登録（D10・D11） | 4 |
| `smooth_radius` の値域の検証と、加工への入力の形（D12・D13） | 5 |

先送りするものは無い。

非ゴール（退けた案と、それが本当の制約か）:

- `AttachScoped` の戻り値で `Detach` できない件を本書で直す案 — 範囲外。[0146_notifier_scoped_detach_10rfd.md](0146_notifier_scoped_detach_10rfd.md) が扱う。制約ではなく別の関心で、同じ RFD に入れなくても題材は片付く。
- 強制行動を `MakeAction` の入力にして `WithAction` を消す案 — 範囲外。[0161_observation_pipeline_10rfd.md](0161_observation_pipeline_10rfd.md) の単位 A が扱う。制約ではなく別の関心で、D9 はそれを待たずに直せる。
- 4 つの Repository を 1 つの形にまとめ、登録の関数名をそろえる案 — 範囲外。課題 [issue-20261109-adc5c4](../management/issues/issue-20261109-adc5c4.md) が扱う。制約ではなく別の関心で、重複登録の扱いは 1 つずつ直せる。
- `ComputeQValueHeatMap` を分割する案（課題 [issue-20261111-90fa5c](../management/issues/issue-20261111-90fa5c.md)）と、HeatMap の描画の重複を直す案（課題 [issue-20261103-e6dbb4](../management/issues/issue-20261103-e6dbb4.md)） — 範囲外。制約ではなく別の関心である。
- 学習スレッドから Panel か Frame を触る 3 つの Observer（TrainPanel、EvalPanel が学習側の step 数を受ける Observer、RunnerFrame の状態表示用の Observer）を、終了の順序に頼らない作りに変える案 — 退ける。P20 のとおり、今は終了の順序で守られていて、D6 がその順序を保つ。制約ではないが、今ある危険が無く、得が小さい。ユーザーの選択である（§10）。
- Observer 自身を、複数のスレッドから同時に呼ばれても安全にする案 — 範囲外。今も Observer は複数のスレッドから呼ばれており（§1.2）、本書はその点を変えない。制約ではなく別の関心である。

## 3. 前提

| # | 種類 | 内容 | 出典 | 確かめ |
|---|---|---|---|---|
| P1 | 守る線 | 失敗の扱いは AGENTS.md の表で決める。構築が終わるまでに判定できる違反は止める。構築後にメトリクス・描画・GUI の一部だけが欠ける異常は、WARN を 1 回出して続ける | [AGENTS.md](../../AGENTS.md) §Fail-Fast 原則の表 | 済み |
| P2 | 守る線 | 契約を変えたら、現用のコード・設定・テスト・文書を同じ変更で移し、旧い形の互換は残さない。過去の記録は書き換えない | [AGENTS.md](../../AGENTS.md) §クリーンブレーク方針 | 済み |
| P3 | 過去の判断 | `AttachScoped` で付けた Observer の `Detach` は、今は何も外さない。RFD0146 がこれを直す計画で、その課題は 80c355 を待っている。RFD0146 は排他を入れず、「Detach は停止後だけ」をコメントの決まりにする計画である | [0146_notifier_scoped_detach_10rfd.md](0146_notifier_scoped_detach_10rfd.md) の「Context / Problem Statement」・NG1・R3 | 済み |
| P4 | 過去の判断 | RFD0161 の単位 A は、`WithAction` を消し、device は Actor の側で合わせる計画である。暫定の RFD で、確定はしていない | [0161_observation_pipeline_10rfd.md](0161_observation_pipeline_10rfd.md) §4.6 | 済み |
| P5 | 事実 | 登録は初期化関数が行う。製品は起動時に 1 回ずつ呼び、テストは同じプロセスの中で何度も呼ぶ | §1.2 の 1499d0 の行 | 済み |
| P6 | 既存部品の働き | `Notify` を呼ぶスレッドは 3 つある。Observer は `Notify` を呼んだスレッドの上で動く。Observer が触る相手（Panel や Frame）の寿命は、Notifier ではなく、Observer を付ける側が守っている。EvalPanel の Runner の通知を受ける Panel の Observer 4 つは、Panel への弱い参照を呼び出しの頭で確かめてから Panel を触り、GUI スレッドの上でだけ動く。学習の Runner の通知を受ける 3 つは、Panel か Frame を直接抱え、終了の順序で守られている（§1.2 の 80c355 の 7 行目） | §1.2 の 80c355 の行、[140_observability.jp.md](../design/140_observability.jp.md) §5.1、[Conv2dPanel.cpp](../../src/apps/runner/src/Conv2dPanel.cpp) 81・82 行 | 済み |
| P7 | 事実 | Boost.Signals2（C++ の通知ライブラリ）は、切断と呼び出しが同時なら、切断のあとも相手が走りうると明記している。相手を呼ぶあいだロックを持たず、一覧は変更するときに写す | [Boost.Signals2 の Thread-Safety の節](https://www.boost.org/doc/libs/release/doc/html/signals2/thread-safety.html)（2026-11-19 に確認） | 対象外 |
| P8 | 事実 | Qt は、別スレッドの相手を直接呼ぶ接続を安全でないとし、別スレッドの相手の破棄は相手のスレッドへ回すよう勧めている | [Qt の Threads and QObjects](https://doc.qt.io/qt-6/threads-qobject.html)（2026-11-19 に確認） | 対象外 |
| P9 | 事実 | 直近の Atari の Run の Observer の数と、今の `Notify` の参照カウントの上げ下げ | §1.3 | 対象外 |
| P10 | 事実 | 動画に渡る画像の大きさは、設定の値かデータの形から決まり、窓の大きさからは決まらない。途中で変わる経路は、読んだ範囲では見つけていない | §1.2 の ba635a の 2 行目 | 対象外 |
| P11 | 既存部品の働き | VideoLogger は、ffmpeg が止まったときなど、今ある実行時の検査では Run を止めている。P1 の表は、今止めている検査はそのままにすると決めている | [metrics_logger.cpp](../../src/core/base/src/metrics_logger.cpp) 500〜506 行、[AGENTS.md](../../AGENTS.md) §Fail-Fast 原則の表 | 対象外 |
| P12 | 事実 | テストから `InitNN` に行き着く呼び出しは 109 か所ある | §1.2 の 1499d0 の 3 行目 | 対象外 |
| P13 | 事実 | Env の初期化関数は、今は 2 回呼ぶと Env の重複で止まる。これを確かめるテストは無い | §1.2 の 1499d0 の 3 行目。テストは 2026-11-19 に、`src` を `(REQUIRE` か `CHECK)_THROWS` と `Regist`・初期化関数の名前の組で引き、Repository の重複を確かめるものが 0 件 | 対象外 |
| P14 | 事実 | LLVM は、初期化関数を「何度呼んでも 1 回だけ登録する」作りにし、登録簿の側は重複を assert で止めている | [PassSupport.h](https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/include/llvm/PassSupport.h)、[PassRegistry.cpp](https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/lib/IR/PassRegistry.cpp)（2026-11-19 に確認） | 対象外 |
| P15 | 既存部品の働き | device をそろえる必要があるのは、DQN 系の `WithAction` が Q 値から値を取り直すからである。基底と MuZero の `WithAction` は device を問わない | §1.2 の 93f8a6 の行 | 対象外 |
| P16 | 事実 | 製品で `WithAction` を呼ぶのは EvalRunner の 1 か所だけである。CUDA が使えるときだけ確かめるテストの書き方は、DQN のテストにすでにある | [trainer.cpp](../../src/core/base/src/trainer.cpp) 294 行（2026-11-19 に `git grep -n "WithAction(" -- src` で、テスト以外の呼び出しはこの 1 行）、[dqn_concrete_test.cpp](../../src/algo/dqn/src/dqn_concrete_test.cpp) 346 行 | 対象外 |
| P17 | 事実 | 今、学習中に Observer の列から実際に外れる Observer は無い。`Clear` は、最終の終了で学習のスレッドを止めたあとに呼ばれる | §1.2 の 80c355 の 6 行目 | 対象外 |
| P18 | 事実 | 無効な画像では、wx は幅と高さに 0、画素のデータに空を返す。画像を記録する口は、画像が有効かを確かめない | §1.2 の ba635a の 3 行目 | 対象外 |
| P19 | 既存部品の働き | 行動の情報は、行動の CPU の写しと GPU の写しを別々に持ち、どちらが元かを記録しない。引数なしで取れる行動は、先に誰かが CPU で取り出したかで変わる | §1.2 の 93f8a6 の 3 行目 | 対象外 |
| P20 | 事実 | Runner の Panel と RunnerFrame が付ける Observer 8 つのうち、学習スレッドから Panel か Frame を触るのは 3 つで、どれも終了の順序で守られている。学習中に閉じうる Panel 2 つは、別のスレッドから触られない | §1.2 の 80c355 の 7 行目 | 対象外 |

## 4. issue の対応表

| ID | 題 | area | 今のコードで確かめた根拠 | 直す確定事項 | 確かめる受入 |
|---|---|---|---|---|---|
| [issue-20261103-80c355](../management/issues/issue-20261103-80c355.md) | Notifier が排他の無い列を、GUI スレッドの Attach/Detach と学習スレッドの Notify で同時に触る | 318_rl_core | [rl.cpp](../../src/core/base/src/rl.cpp) 966・990・1125 行、[rl.hpp](../../src/core/base/include/anet/rl.hpp) 1262〜1265 行 | D1〜D7 | A1〜A3・A13〜A15 |
| [issue-20261103-ba635a](../management/issues/issue-20261103-ba635a.md) | VideoLogger が最初の画像の大きさで固定され、以後の画像の大きさを確かめずに読む | 317_metrics | [metrics_logger.cpp](../../src/core/base/src/metrics_logger.cpp) 508・509・787・816 行 | D8 | A4 |
| [issue-20261103-93f8a6](../management/issues/issue-20261103-93f8a6.md) | actor_approx の GPU 推論の評価 Actor に強制行動を渡すと、device の不一致で WithAction が止まる | 318_rl_core、321_agent_core | [trainer.cpp](../../src/core/base/src/trainer.cpp) 293〜297 行、[dqn_actor.cpp](../../src/algo/dqn/src/dqn_actor.cpp) 204〜207 行 | D9 | A5・A6・A12・A14 |
| [issue-20261109-1499d0](../management/issues/issue-20261109-1499d0.md) | Agent・View・NN module の Repository は同じ名前の重複登録を黙って上書きし、Env だけが止まる | 321_agent_core、319_gui、315_nn | [agent.cpp](../../src/core/base/src/agent.cpp) 49 行、[gui.cpp](../../src/core/base/src/gui.cpp) 392 行、[nn_impl.cpp](../../src/core/base/src/nn_impl.cpp) 2482 行 | D10・D11 | A7〜A9 |
| [issue-20261110-3f4e63](../management/issues/issue-20261110-3f4e63.md) | QValuePanel で平滑化が有効なまま smooth_radius が 0 以下だと、空の重みの範囲外を読む | 110_runner_app | [QValuePanel.cpp](../../src/apps/runner/src/QValuePanel.cpp) 30・138・545 行、[QValuePanel.hpp](../../src/apps/runner/src/QValuePanel.hpp) 69〜71 行 | D12〜D14 | A10・A11 |

## 5. 確定事項

| # | 決定 | 理由と、よりどころ |
|---|---|---|
| D1 | `Notifier` の `Attach`・`Detach`・`Notify`・`Clear`・`Shutdown`・`WillBlockOnShutdown`・`LogObservers` は、どのスレッドから同時に呼んでも、Observer の列が壊れない | 題材 80c355 の事象。`Clear` 以下の 4 つも同じ列を読む・消すので、同じ排他の対象にする（§1.2 の 80c355 の 1 行目） |
| D2 | `Detach` と `Clear` は、進行中の通知を待たずに戻る。戻った時点で始まっていた通知は、外した Observer を呼びうる。その通知が終わるまで、Observer は `Notifier` が生かしておく。`Detach` が戻ったあとに始まった通知は、外した Observer を呼ばない | ユーザーの選択（グリルの質問 1 の A。§10）。P3 から、待つ形の利点（Panel を安全に壊せる）は RFD0146 のあとでしか出ない。待つ形は、待たれている Observer が GUI スレッドの持つロックを待つと固まる組み合わせを新しく作る（§5.1）。P7 の事例がこの形である。P6 のとおり、3 つのスレッドが `Notify` を呼び、Observer が触る相手の寿命は、今も Observer を付ける側が守っている（弱い参照の確かめか、終了の順序）。待たない形は、この分担を変えない。P17 から、今は学習中に外れる Observer が無いので、外した Observer に通知が届く場面と、Observer が通知のスレッドで壊れる場面（§6.2）は、RFD0146 のあとに出る |
| D3 | `Notifier` は、Observer を呼ぶあいだ、自分のロックを持たない。Observer の中から `Attach`・`Detach`・`Notify` を呼んでよい。Observer の中で呼んだ `Attach`・`Detach` の効き目は、次に始まる通知からである。`Shutdown`・`WillBlockOnShutdown`・`LogObservers` も、Observer を呼ぶあいだロックを持たない | D2 の帰結。`Shutdown` は背景評価の排水で長く待つので、ロックを持ったままだと、その間の `Attach`・`Detach` が止まる |
| D4 | Observer を呼ぶ順は登録順のまま、Observer が動くスレッドは `Notify` を呼んだスレッドのまま変えない | [140_observability.jp.md](../design/140_observability.jp.md) §5.1 の今の契約。退けた案は、通知を別のスレッドへ回す形である。制約ではないが、メトリクスの記録の順と step の対応が変わり、題材に要らない |
| D5 | `Notify` 1 回に足す処理は、Observer の数に比例して増やさない | P9 から、LEARN の Observer が 134 個ある Run で、通知のたびに一覧を写すと、参照カウントの上げ下げが今より増える。P7 の事例も、変更のときだけ写す。見積もりは §1.3 |
| D6 | RunnerFrame の終了時の順序（学習の停止、背景評価の排水、そのあとに状態表示用の Observer を `Detach`）は変えない | 今回のゴールに要らない。退けた案は、D1 で列が壊れなくなるので順序の決まりを外す形である。制約ではないが、D2 から、排水より前に外すと、外したあとの Observer に通知が届きうるので、今の順序は有効なままである |
| D7 | 呼んでよいスレッドと時点を、宣言のコメントに書く。`Notifier` には D1〜D4 を書く。`BatchStepResult::GetEpisodeEndMask` には、同時に呼んでよいと書く。`ReplayInitialPriorityHint` と `BatchActionInfo` には、「同じ実体について、キャッシュを書きうる口（`GetPayloadCpu`、`GetAction(device)`、`GetScalar` の `action`）と、その実体を読む口・写す口を、別々のスレッドから同時に呼ばない。スレッドをまたいで渡すときは、渡し終えてから、受け取った側が呼ぶ」と書く。この 2 つに守りは足さない。Runner の Panel と RunnerFrame が Observer を付ける 8 か所（§1.2 の 80c355 の 7 行目）には、その Observer がどのスレッドで動くかと、Panel や Frame が先に壊れない理由を、コメントに書く | 題材 80c355 の対応方針の追記（2026-11-11）が、決まりを書くことを求めている。8 か所のコメントは、ユーザーの選択である（§10）。P20 のとおり、今の安全は、通知を受ける Runner と終了の順序に頼っていて、コードからは読めない。§1.2 の 80c355 の 5 行目のとおり、読んだ範囲では同時に呼ぶ経路が無いので、起きない経路の守りは足さない（[AGENTS.md](../../AGENTS.md) §LLM コーディング規律の Simplicity First）。退けた案は、3 つとも同じ守りにそろえる形である。制約ではないが、行動の 1 step ごとに通る箇所へ、使われない守りを足すことになる |
| D8 | VideoLogger は、構築時と違う大きさの画像を、動画に書かない。その tag について、Run ごとに 1 回だけ WARN を出す。文言には、tag、最初の画像の大きさ、来た画像の大きさを、`<幅>x<高さ>` の形で入れる。同じ大きさの画像は書き続け、Run は止めない。落とした枚数は数えない。VideoLogger を作ったあとに来た無効な画像は、大きさ `0x0` の画像として、同じに扱う。PNG の個別保存（`app.metrics_logger.use_png_dump`）は、大きさが違っても今のまま書く | ユーザーの選択（質問 2 の B。§10）。P1 の、構築後にメトリクスの一部だけが欠ける異常の行に当たる。P10 から、起きるのは画像を作る側に不具合があるときで、診断用の動画のために長い Run を止める損のほうが大きい。無効な画像は、P18 のとおり大きさが 0 で、構築時と違う大きさの画像に当たるので、同じ扱いになる（ゴール 2 の、範囲外を読まないことから導ける）。PNG は 1 枚ずつ独立で、範囲外の読みが起きない。PNG について退けた案は、大きさの違う画像を PNG でも落とす形である。制約ではないが、落とす理由が無い |
| D9 | DQN 系の `WithAction` は、ヒントの値を取り直すときだけ、渡された行動を Q 値の device へ写して使う。device の違いで止める検査は消す。結果の行動は、渡された行動のままにする。ヒントの値を、渡された行動の値に取り直す計算は変えない。基底と MuZero の `WithAction`、EvalRunner は変えない | ユーザーの選択（質問 4 の A と、質問 5 の B。§10）。P15 から、device をそろえる必要は DQN 系の中の計算から来ているので、その都合は DQN 系が持つ（[AGENTS.md](../../AGENTS.md) §汎用機構と利用側の責任境界）。P4 の、RFD0161 が計画する向きと同じである。結果の行動を移さないのは、P19 から「差し替える前の行動の device」が一通りに決まらず、ゴール 3 に要るのは取り直しが止まらないことだけだからである。結果の形は、基底・MuZero と同じになる。「結果の行動は渡されたまま」で退けた案は §5.1 の「結果の行動も移す形」である。「計算は変えない」で退けた案は、取り直しをやめてヒントを捨てる形である。制約ではないが、ヒントが打った行動の値でなくなり、ゴール 3 に反する。「EvalRunner は変えない」で退けた案は §5.1 の「呼ぶ側で合わせる形」である。「基底と MuZero は変えない」で退けた案は、基底で一律に移す形である。制約ではないが、P15 のとおり device を問わないので、移す理由が無い |
| D10 | `AgentRepository`・`ViewRepository`・`NetworkModuleRepository` の登録は、登録済みの名前なら `ANET_SYSTEM_ERROR` で止める。文言は Env と同じ形（`... is already registered. <名前の種類>='<名前>'`）にし、名前を入れる。先に登録した factory は残る | 題材 1499d0 の対応方針。Env の扱い（§1.2）と [ADR 0009](../adr/0009-imagecls-batch-env-seam.md) の「二重登録は fail-fast」にそろえる。P1 の、構築が終わるまでに判定できる違反の行に当たる |
| D11 | 初期化関数（§6.5 の 10 個）は、何度呼んでも、登録を 1 回だけ行う。2 回目からは何もしない。初期化関数が登録の途中で止まった場合、同じ関数をもう一度呼んでも止まる（どの名前で止まるかは問わない）。Repository を直接 2 回呼ぶ重複は、D10 のとおり止まる | ユーザーの選択（質問 3 の A。§10）。D10 だけだと、P12 のテストが 2 回目の呼び出しで止まる。P14 の事例がこの形で、決まりが 1 つで済む。許容した不利益は、P13 の、Env の初期化関数の 2 回目が「止まる」から「何もしない」へ変わる点である |
| D12 | `QValuePanel.smooth_radius` は整数で、1 以上とする。上限は置かない。設定の読み込みで下の表のとおり検証する。`QValuePanel.use_hist_smooth` の値に依らず検証する。半径 0 以下を「平滑化なし」とは読まない。コードの既定は 4 のままにする | [AGENTS.md](../../AGENTS.md) §設定値の扱い（範囲外の値は止める。特定の値に別の意味を持たせない）と §汎用機構と利用側の責任境界（値域は設定の読み込み時に検証する）。上限を置かないのは、大きい半径は全体へ広げるだけで、不正ではないためである。メモリを確保できないほど大きい値は、確保の失敗として止まり、専用の検査は置かない。「既定は 4 のまま」で退けた案は、出荷設定の 5 にそろえる形である。制約ではないが、題材に要らない |
| D13 | ヒートマップの加工への入力（`QValueHeatMapRequest`）は、平滑化の指定を重み 1 組で表す。有効かどうかと半径を、別の値として持たない。重みが空なら平滑化しない | 題材 3f4e63 の対応方針の追記（2026-11-11）の案。3 つの値の食い違い（有効なのに重みが空、など）を作れなくする |
| D14 | 実験記録の設定の写しと、検証資材の基準にある `smooth_radius` は直さない | P2 の、過去の記録は書き換えない。§1.2 の 3f4e63 の 2 行目のとおり、値はどれも 5 で、D12 の検証を通る。本当の制約である。退けた案は、整数の表記に直す形で、直す必要も無い |
| D15 | 性能は、実測の受入を置かない。見積もり（§1.3）を根拠にする。振る舞いが変わらないことは、決定的な設定の短い Run のメトリクスの一致（A13）で確かめる | §1.3 の見込みの差が、Run の実時間のぶれより小さく、測っても合否を決められない |
| D16 | 実装は、§6.8 の 5 つの段階に分け、1 つずつ実装・検証できるようにする | 5 件は互いに独立で、途中で止めても、済んだ段階は完結する |
| D17 | ADR を 1 本起こす（[ADR 0082](../adr/0082-notifier-detach-does-not-wait-for-in-flight-notifications.md)。D2・D3） | 背景を知らないと意外に見え（外したあとも届く）、待つ形との比較があり、Observer の書き方に効くので戻しにくい。ほかの決定は、規約か事例どおりで、戻すのも易しい |

D12 の、`smooth_radius` の検証。上から順に当て、最初に当たった行で扱う。設定に書かれていなければ、コードの既定（4）を使い、検証は 3 行目に当たる。1 行目の条件は、設定の整数の読み口（[config.cpp](../../src/core/base/src/config.cpp) 61〜84 行）の判定そのものである。「例」の列は条件ではない。

| 行 | 条件 | 例（条件ではない） | 扱い |
|---|---|---|---|
| 1 | 設定の整数の読み口が、書かれた値を整数として読めないと判定する | `1.5`、`abc`、空の値、整数の範囲を超える値 | 読み口が `ANET_SYSTEM_ERROR` で止める（キー名と書かれた値を含む今の文言） |
| 2 | 読み口が読んだ整数が 0 以下 | `0`、`-1` | `ANET_SYSTEM_ERROR` で止める。文言に、キー名 `QValuePanel.smooth_radius`、指定値、期待値（1 以上）を入れる |
| 3 | 読み口が読んだ整数が 1 以上 | `1`、`5`、`1,000`（読み口はカンマを除いて 1000 と読む） | 受け付ける |

### 5.1 退けた案

| 案 | 退けた理由 |
|---|---|
| `Detach` が、その Observer への進行中の呼び出しが終わるまで待つ形（質問 1 の B） | 待たれている Observer が、GUI スレッドの持つロックや GUI の処理を待つと、互いに待って固まる。Observer の中から自分を外すときのための特例が要る。利点は RFD0146 のあとでしか出ない（P3） |
| 通知のたびに、ロックの中で Observer の一覧を写す形（題材 80c355 の対応方針の文面） | D5 の理由。Observer の数に比例する処理が、通知のたびに増える |
| 動画の大きさ違いで Run を止める形（質問 2 の A。題材 ba635a の対応方針） | P1 の表では、構築後にメトリクスの一部だけが欠ける異常は、WARN を出して続ける行に当たる。止める形は、その行と食い違う。P11 のとおり、同じ VideoLogger の ffmpeg の停止とは扱いがそろうが、診断用の動画のために学習を止める |
| 大きさが変わったら新しい動画ファイルに切り替える形（質問 2 の C） | ファイル名の決まり、動画の記録の行、読む側の扱いが増える。P10 から、使われる見込みが薄い |
| `InitNN` だけ 1 回だけの登録にする形（質問 3 の B） | 初期化関数ごとに、何度呼んでもよいものと 1 回だけのものが分かれる |
| 初期化関数は 1 回しか呼べない形にして、テストを直す形（質問 3 の C） | `InitRL` が中で `InitNN` を呼ぶので、`InitRL` を 2 回呼ぶだけで止まる。`InitNN` を `InitRL` から出して公開する変更が要る。P12 の 109 か所のうち、テスト用の関数を通る 63 か所は、その関数に守りを入れれば済むが、残る 46 か所（`InitNN()` の直接の呼び出し 38 と `InitRL(` 8）は書き換えが要る |
| 同じ型の factory なら、重複登録を黙って通す形 | 型の照会が要る。同じ型で中身の違う登録を見分けられない |
| 強制行動の device を、呼ぶ側（EvalRunner）で合わせる形（質問 4 の B。題材 93f8a6 の対応方針） | 汎用の EvalRunner が、DQN 系の都合のために device をそろえる形になる（P15）。直しは最小だが、`WithAction` を呼ぶ側が毎回そろえる決まりが残る |
| 渡された行動を、ヒントがあれば Q 値の device、無ければ差し替える前の行動の device へ移し、結果の行動もその device に載せる形（質問 5 の A。細部 14 の案） | P19 から、差し替える前の行動の device が一通りに決まらない。ヒントが無いときの結果の行動まで変わる。DQN 系だけ、結果の行動の device が基底と違う形になる |
| 半径 0 以下を「平滑化なし」として扱う形（題材 3f4e63 の対応方針の候補） | 特定の値に別の意味を持たせることになる（[AGENTS.md](../../AGENTS.md) §設定値の扱い）。する・しないは `use_hist_smooth` が表す |

## 6. 実装契約

### 6.1 クラス一覧

従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務（今回の変更） | 置き場 |
|---|---|---|---|
| `Notifier` | 変更 | Observer の列を排他で守り、Observer をロックの外で呼ぶ（D1〜D5） | 強化学習 共通基盤・rl |
| `BatchStepResult`、`ReplayInitialPriorityHint`、`BatchActionInfo` | 変更 | 宣言に、呼んでよいスレッドと時点のコメントを足すだけである（D7） | 強化学習 共通基盤・rl |
| `TrainPanel`、`EvalPanel`、`QValuePanel`、`Conv2dPanel`、`EvalTracePanel`、`SweepHeatMapPanel`、`RunnerFrame` | 変更 | Observer を付ける箇所に、動くスレッドと、Panel や Frame が先に壊れない理由のコメントを足すだけである（D7） | RunnerApp |
| `VideoLogger`、`MetricsLogger` | 変更 | 構築時と違う大きさの画像を動画に書かず、WARN を 1 回出す（D8）。確かめる場所は、tag と構築時の大きさの両方を知れるところに置く | 可視化・メトリクス・metrics_logger |
| `DQNActionInfo` | 変更 | ヒントの値の取り直しに使う行動だけを、Q 値の device へ写す（D9） | DQN 系・dqn_actor |
| `AgentRepository` | 変更 | 登録済みの class ID の登録を止める（D10） | Agent 共通基盤・agent |
| `ViewRepository` | 変更 | 同上 | GUI 共通基盤・gui |
| `NetworkModuleRepository` | 変更 | 登録済みの type 名の登録を止める（D10） | ニューラルネットワーク・nn_impl |
| `QValuePanel` | 変更 | `smooth_radius` を整数で読み、値域を検証する（D12）。重みを作る条件と、加工が重みを読む条件をそろえる | RunnerApp・QValuePanel |
| `QValueHeatMapRequest` | 変更 | 平滑化の指定を重み 1 組で持つ（D13） | RunnerApp・QValuePanel |

新しい名前（設定キー、メトリクス名、TensorDict キー、公開クラス名）は付けない。

### 6.2 Notifier

- 守る口は D1 の 7 つである。`Attach` は、Observer の種類ごとの口と、`AttachScoped` を含む。
- 4 種類の Observer の列は、それぞれ D1〜D5 を満たす。
- `Notify` が Observer を呼んでいる途中で、ほかのスレッドが `Detach` して、呼び出し側が Observer の参照を手放しても、その `Notify` が戻るまで Observer は壊れない（D2）。
- Observer が壊れるのは、最後の参照が手放されたスレッドの上である。通知の途中で外され、呼び出し側が参照を手放した Observer は、`Notify` を呼んだスレッドで壊れる（D2 の帰結）。Observer は、どのスレッドで壊れてもよいように書く。
- `Clear` が戻ったあとも、その時点で始まっていた通知は、外した Observer を呼びうる（D2）。`Shutdown` と `WillBlockOnShutdown` と `LogObservers` が見るのは、呼んだ時点で付いている Observer だけである。そのあとに付いた Observer は見ない。
- 例外の扱いは変えない。Observer が投げた例外は、今と同じく `Notify` の呼び出し元へ伝わり、その回の残りの Observer は呼ばれない。退けた案は、例外を捕まえて残りの Observer を呼び続ける形である。制約ではないが、失敗を握り潰すことになり、題材に要らない。

### 6.3 VideoLogger

- 大きさを比べるのは、動画へ書く直前である。幅と高さの両方が構築時と同じときだけ書く。
- WARN は、既存の「Run ごとに対象につき 1 回」の口で出し、対象は tag にする。Run を作り直すと、また 1 回出る。
- 無効な画像は、大きさを取り出す前に有効かを見て、大きさ `0x0` として D8 の比較に当てる。wx の検査（P18）を踏まないためである。
- 最初の画像が無効なときに 0×0 で VideoLogger を作る今の挙動（§1.2）と、ffmpeg の異常（P11）の扱いは変えない。最初の画像について退けた案は、最初の有効な画像が来るまで VideoLogger を作らない形である。制約ではないが、題材の事象（作ったあとの読み）の外で、§9 の U3 に置く。ffmpeg について退けた案は、ffmpeg の異常も WARN にそろえる形である。P1 の表が、今止めている実行時の検査はそのままにすると決めているので、本当の制約である。

### 6.4 強制行動

- ヒントの取り直しに使う行動は、渡された行動を Q 値の device へ写したものである。結果の行動の情報が持つ行動は、渡された行動そのものである（D9）。
- device の違いで止めていた検査の文言（`cannot replace a hinted action across devices`）は、コードから消える。
- ヒントの取り直しに要る値（Q 値、Munchausen の項）が無いとき、形が合わないときに止める今の検査は変えない。退けた案は、値が無いときにヒントを捨てて続ける形である。制約ではないが、題材の事象は device の違いだけで、要らない。

### 6.5 Repository と初期化関数

初期化関数は次の 10 個である。

| 置き場 | 関数 |
|---|---|
| ニューラルネットワーク（[nn_modules.cpp](../../src/core/base/src/nn_modules.cpp) 3309 行） | `InitNN` |
| Agent の系統（[dqn.cpp](../../src/algo/dqn/src/dqn.cpp)、[muzero.cpp](../../src/algo/muzero/src/muzero.cpp)、[imagecls.cpp](../../src/algo/imagecls/src/imagecls.cpp)） | `algo::InitDqn`、`algo::InitMuZero`、`algo::InitImageCls` |
| Env（[Atari.cpp](../../src/env/atari/src/Atari.cpp)、[CartPole.cpp](../../src/env/cartpole/src/CartPole.cpp)、[DropMerge.cpp](../../src/env/dropmerge/src/DropMerge.cpp)、[GridMaze.cpp](../../src/env/gridmaze/src/GridMaze.cpp)、[ImageCls.cpp](../../src/env/imagecls/src/ImageCls.cpp)、[LunarLander.cpp](../../src/env/lunarlander/src/LunarLander.cpp)） | `env::InitAtari`、`env::InitCartPole`、`env::InitDropMerge`、`env::InitGridMaze`、`env::InitImageCls`、`env::InitLunarLander` |

- 1 回だけの登録は、初期化関数ごとに持つ。2 つのスレッドが同時に同じ初期化関数を呼んでも、登録は 1 回である。
- `InitRL` は 1 回だけにしない。backend の設定を呼ぶたびに当て直す関数で、中で呼ぶ `InitNN` が 1 回だけになれば足りる。退けた案は、`InitRL` も 1 回だけにする形である。制約ではないが、設定を変えて呼び直すテストが、2 回目から設定を当てられなくなる。
- テスト用の `RegisterStandardNetworkModules` は直さない。今も `InitNN` を呼ぶだけの関数で（[src_testing/nn.cpp](../../src/core/base/src_testing/nn.cpp) 28〜31 行）、D11 で `InitNN` の登録が 1 回だけになれば足りる。
- `EnvRepository` の登録は、すでに重複を止めるので、直す箇所が無い。

### 6.6 QValuePanel

- `smooth_radius` の検証は、QValuePanel が設定を読むところで行う（D12 の表）。Runner の起動時に当たる。
- 重みは、`use_hist_smooth` が真のときに作り、偽のときは空にする。加工は、重みが空でなければ平滑化する（D13）。
- 加工が受け付ける重みは、空か、奇数個（中心 1 つと、左右に同じ数）である。上から順に当てる。空なら平滑化しない。奇数個なら平滑化する。それ以外（2 個以上の偶数個）は、`ANET_SYSTEM_ERROR` で止める。半径を重みの数から出すので、偶数個では範囲の外を読むためである。製品では、重みは起動時に検証した半径（D12）から作るので、偶数個にはならない。この検査は、加工を直接呼ぶ側（テストなど）の呼び出しの誤りを止めるもので、メトリクスの一部が欠ける異常（P1 の WARN の行）ではなく、部品の呼び出しの契約違反として止める。
- 平滑化あり・なしの加工の結果は、半径が 1 以上の設定では、今と同じ値である。

### 6.7 移行対象

| 種類 | 対象 | 直すこと |
|---|---|---|
| コード | [rl.hpp](../../src/core/base/include/anet/rl.hpp)、[rl.cpp](../../src/core/base/src/rl.cpp) | D1〜D5、D7 |
| コード | Runner の Panel と RunnerFrame が Observer を付ける 8 か所のファイル（§1.2 の 80c355 の 7 行目） | D7 のコメント |
| コード | [metrics_logger.hpp](../../src/core/base/include/anet/metrics_logger.hpp)、[metrics_logger.cpp](../../src/core/base/src/metrics_logger.cpp) | D8 |
| コード | [dqn_actor.cpp](../../src/algo/dqn/src/dqn_actor.cpp) | D9 |
| コード | [agent.cpp](../../src/core/base/src/agent.cpp)、[gui.cpp](../../src/core/base/src/gui.cpp)、[nn_impl.cpp](../../src/core/base/src/nn_impl.cpp) | D10 |
| コード | §6.5 の 10 個の初期化関数のファイル | D11 |
| コード | [QValuePanel.cpp](../../src/apps/runner/src/QValuePanel.cpp)、[QValuePanel.hpp](../../src/apps/runner/src/QValuePanel.hpp) | D12・D13 |
| テスト | [QValuePanel_test.cpp](../../src/apps/runner/src/QValuePanel_test.cpp) 24 行の、加工への入力を作る補助 | D13 の形に合わせる。期待値は今と同じである |
| テスト | [GridMazeEnv_test.cpp](../../src/env/gridmaze/src/GridMazeEnv_test.cpp) 623 行の、Env の factory の直接の登録 | `env::InitGridMaze` の呼び出しに替える。A8 の、初期化関数を 2 回呼ぶテストと同じプロセスで走っても、Env の重複で止まらないようにするためである |
| テスト | 重複登録に頼るテスト | 見つけていない（2026-11-19 に、テストのファイルの中の `RegisterAgentFactory<`・`RegisterViewCreator<`・`RegisterNetworkModuleFactory<` と、3 つの Repository の `Instance().Register` を引いて 5 行。どれも、テスト専用の名前を 1 回だけ登録する）。残っていれば、A9 で止まるので、その場で同じ変更の中で直す |
| 設定 | [common.txt](../../apps/runner/config/common.txt) 65・66 行 | 直さない。今の値（真、5）は D12 を通る |
| 設計文書 | [140_observability.jp.md](../design/140_observability.jp.md) | §2 の `Notifier` の行と §5.1 に、D1〜D4 を書く。§5.2 の、排水の後に `Detach` する理由を、D6 の理由（外したあとの Observer に通知が届きうる）に直す。§7 の `videos/<tag>.mkv` の行に、D8 を書く |
| 設計文書 | [110_agents_and_learning.jp.md](../design/110_agents_and_learning.jp.md)、[120_environments.jp.md](../design/120_environments.jp.md)、[130_neural_networks.jp.md](../design/130_neural_networks.jp.md) | Repository と初期化関数の説明に、D10・D11 を書く |
| 設計文書 | [200_dqn_agents.jp.md](../design/200_dqn_agents.jp.md) §7 の、`To`・`WithAction` が値を引き継ぐ段落 | D9 を書く |
| 英訳 | `docs/design/*.en.md` | 本書の実装では直さない。和文を直すと、翻訳の skill（`anet-translate-docs`）が「更新あり」として拾う |
| 過去の記録 | 実験記録の設定の写し、検証資材の設定の写しと基準、課題の事象の行、過去の RFD | 直さない（D14、P2） |
| 道具が作るもの | 課題の一覧と、wiki のトピック別まとめの「課題」の区画 | 手で直さない。課題の道具が作り直す |

旧い契約のための仕組み（別名、互換の分岐、旧い値の読み替え）は残さない。`smooth_radius` に小数を書いた設定は、D12 の表の 1 行目で止まる。

### 6.8 段階

| 段階 | 中身 | 受入 |
|---|---|---|
| 1 | Notifier（D1〜D7） | A1〜A3 |
| 2 | VideoLogger（D8） | A4 |
| 3 | 強制行動（D9） | A5・A6 |
| 4 | Repository と初期化関数（D10・D11） | A7〜A9 |
| 5 | QValuePanel（D12〜D14） | A10・A11 |
| 仕上げ | 設計文書、全体の確かめ | A12〜A15 |

段階 1〜5 の順は問わない。

### 6.9 ほかの RFD との順序

- **RFD0146**（`AttachScoped` の `Detach`）: 本書が先である。RFD0146 が先に入った場合も、本書の内容は変わらない。RFD0146 は `AttachScoped` の戻り値を変え、本書は列の守り方を変えるので、同じクラスの別の箇所である。段階 1 の途中で RFD0146 が入るときは、段階 1 を終えてから入れる。RFD0146 の「Detach は停止後だけ」のコメントの計画は、D1〜D3 と食い違う（§9 の U2）。
- **RFD0161 の単位 A**（強制行動を `MakeAction` の入力にする）: 単位 A が先に入れば、`WithAction` が消えるので、段階 3 は行わない。その場合、題材 93f8a6 は、単位 A の受入（強制した lane の値）で片付いたかを確かめて判定する。本書が先なら、単位 A が `WithAction` ごと消す。段階 3 の途中では入れない。RFD0161 の §1.4 にある「device の検査で止まる読み」の記述は、段階 3 のあと古くなる。RFD0161 は暫定の RFD で、再グリルで今のコードを読み直す。
- **RFD0159・RFD0160・RFD0176**: `WithAction` の経路を使うか、説明に出すだけである。D9 のあとも `WithAction` の口の形は同じなので、順序の決まりは無い。
- **RFD0056**: `Notifier` と `VideoLogger` は説明に出るだけである。順序の決まりは無い。

## 7. 受入

テストの実行体は、`anet-core-base-test`（基盤）、`anet-algo-dqn-test`・`anet-algo-muzero-test`・`anet-algo-imagecls-test`（Agent の系統）、`anet-env-atari-test`・`anet-env-dropmerge-test`・`anet-env-gridmaze-test`・`anet-env-imagecls-test`・`anet-env-lunarlander-test`（Env）、`AnetRLRunner-gui-test`（Runner）である。`AnetRLRunner-gui-test` は、ビルドの既定の対象に入らず、`ctest` にも登録が無いので、手でビルドして起動する。

本節の「既定のスイート」は、実行体を引数なしで起動したときに走るテスト、つまり隠しタグ（`[.]`）の付かないテストを指す。10 個の実行体のどれにも、この意味で使う。

| # | 受入 | 確かめ方 |
|---|---|---|
| A1 | `Notifier` の決まり（D2〜D4）を、1 つのスレッドのテストで確かめる。(a) Observer は登録順に呼ばれる。(b) `Detach` が戻ったあとに始めた `Notify` は、外した Observer を呼ばない。(c) Observer の中から自分を `Detach` すると、その回の残りの Observer は呼ばれ、次の `Notify` から自分は呼ばれない。(d) Observer の中から別の Observer を `Attach` すると、その回は呼ばれず、次の `Notify` から呼ばれる | `anet-core-base-test` のタグ `[notifier][contract]` が通る |
| A2 | 進行中の通知と `Detach` の重なり（D2）。Observer の呼び出しの中で待ち合わせ、その間に別のスレッドが `Detach` して参照を手放す。待ち合わせを解いたあと、呼び出しは最後まで走り、Observer は `Notify` が戻るまで壊れない（テストが持つ、その Observer への弱い参照が、`Notify` が戻る前は生きていて、戻ったあとに切れる） | `anet-core-base-test` のタグ `[notifier][contract]` が通る |
| A3 | 同時実行（D1）。1 つのスレッドが `Notify` を繰り返し、別の 2 つのスレッドが `Attach` と `Detach` を繰り返す。落ちない。付けたままの Observer の呼び出しの回数が、`Notify` の回数と一致する。TrainEvent と LearnEvent の両方で行う。`Notify` の回数は、既定のスイートに入る版が 2,000 回、入らない版が 2,000,000 回 | `anet-core-base-test` のタグ `[notifier][concurrency]` と、既定のスイートに入らないタグ `[.][notifier_stress]` が通る |
| A4 | 動画の大きさ違い（D8）。動画を記録する公開の口（`MetricsLogger` の画像の記録）へ、同じ tag で、512×512、256×256、256×256、512×512 の順に画像を渡す。例外が出ない。WARN は 1 件で、文言に、その tag、`512x512`、`256x256` を含む。別の tag で、512×512 の画像のあとに無効な画像を 2 回渡すと、例外が出ず、その tag の WARN は 1 件で、文言に `0x0` を含む | `anet-core-base-test` のタグ `[metrics][video][frame_size]` が通る。ffmpeg が要る（今ある `[metrics][video]` のテストと同じ） |
| A5 | 強制行動の、今の値を変えない確かめ（D9）。`WithAction` を呼ぶ今のテストが、期待値を変えずに通る | `anet-algo-dqn-test` と `anet-core-base-test` が通る |
| A6 | 強制行動の device（D9）。CUDA が使えるときだけ走るテストで、次を確かめる。(a) Q 値と行動が CUDA にあるヒントつきの行動の情報に、CPU の行動を渡すと、例外が出ず、ヒントの値が、同じ行動を CUDA で渡したときと一致し、結果の行動は CPU にある。(b) ヒントが無く、行動が CUDA にある行動の情報に、CPU の行動を渡すと、結果の行動は CPU にある（今と同じ）。実装と検証は CUDA のある環境で行う。CUDA の無い環境で実装を終えるときは、この項を未達として報告し、題材 93f8a6 の判定を残件有りのままにする | `anet-algo-dqn-test` のタグ `[dqn][forced_action][device]` が通る |
| A7 | 重複登録（D10）。Agent・View・NN module のそれぞれで、同じ名前を 2 回登録すると、2 回目が例外になり、文言に `already registered` とその名前を含む。そのあと、名前から引けるのは、先に登録した factory である | `anet-core-base-test` のタグ `[repository][duplicate]` が通る |
| A8 | 初期化関数（D11）。同じ初期化関数を続けて 2 回呼んでも例外が出ず、登録した名前から factory を引ける。`InitNN` は `anet-core-base-test`、Agent の系統の 3 つはそれぞれの `anet-algo-*-test`、Env の 5 つ（Atari・DropMerge・GridMaze・ImageCls・LunarLander）はそれぞれの `anet-env-*-test` で確かめる。`env::InitCartPole` は、Env のテストの実行体が無いので、`AnetRLRunner-gui-test` で確かめる（この実行体は、今も起動時に `env::InitCartPole` を呼んでいる。[EvalTracePanel_test.cpp](../../src/apps/runner/src/EvalTracePanel_test.cpp) 1635 行） | それぞれの実行体のタグ `[init][idempotent]` が通る |
| A9 | 今あるテストが、重複登録で止まらない。比べるのは、テストケースごとの合否である。実装の前からある既定のスイートのテストケースのうち、実装の前に通っていたものが、実装のあとも、すべて通る。本書が足すテストケース（A1〜A4・A6〜A8・A10・A11 のもの）は数えない。実装の前から落ちているテストケースも数えない。実装の前の結果は、実装に入る前の作業ツリーで 10 個の実行体をビルドし、引数なしで起動して取り、テストケースごとの合否を保存する。`AnetRLRunner-gui-test` も、同じく引数なしで起動して比べる | 10 個の実行体のそれぞれで、保存した実装の前の合否と、実装のあとの合否を、テストケースの名前で突き合わせ、通っていたものが落ちた件数が 0 件 |
| A10 | `smooth_radius` の検証（D12 の表）。`QValuePanel.smooth_radius` が `0`・`-1`・`1.5` のとき、QValuePanel の構築が例外になり、文言に `smooth_radius` を含む。`0` と `-1` では、文言に `QValuePanel.smooth_radius` と指定値を含む。`1` と `5` では構築できる。`QValuePanel.use_hist_smooth` が偽で `smooth_radius` が `0` のときも例外になる | `AnetRLRunner-gui-test` のタグ `[q_value_smooth]` が通る |
| A11 | 加工への入力の形（D13）と、値を変えない確かめ。加工の今の単体テストが、期待値を変えずに通る。重みが空のとき、平滑化なしの結果になる。偶数個の重みを渡すと例外になる | `AnetRLRunner-gui-test` のタグ `[rfd0122][qvalue_heatmap]`（今ある加工のテスト）と、足すテストが通る。検索は表の下の「検索 1」が 0 行 |
| A12 | device の検査の文言が消えた（D9） | 表の下の「検索 2」が 0 行 |
| A13 | 既定の設定の Run の結果を変えない（ゴール 6、D15）。RFD0103 の検証資材（[run_validation.py](../../src/core/base/testdata/rfd0103/run_validation.py)）と同じ 6 構成・同じ短縮条件（8 lane、batch 16、warmup 256、容量 4096、RR1、seed 1、8704 experience、FP32、決定論的 backend、評価停止）で、実装の前に保存した実行体と、実装のあとの実行体を走らせる。`90_perf/`・`91_resource/`・`92_resource/` を除く全 scalar の tag・step・値が、6 構成とも一致する。除くのは、実時間と資源の値で、走らせるたびに変わるためである。値は、検証資材が正規化した記録の表記で比べる（NaN は同じ表記どうしで一致とする） | 検証資材が正規化した記録が、実装の前後で、6 構成ともバイト単位で一致する |
| A14 | 人が確かめる項。(a) 学習中に、メニューから Conv2d と HeatMap の Panel を、それぞれ 10 回ずつ開いて閉じ、Runner が落ちない。(b) `actor_approx` かつ `run.eval_device` が GPU の構成で Runner を起動し、EvalPanel で矢印キーを 10 回押して、Runner が止まらない | 人が Runner を操作して確かめ、結果を実装の報告に書く |
| A15 | 設計文書と ADR。§6.7 の設計文書の行が直っている。[ADR 0082](../adr/0082-notifier-detach-does-not-wait-for-in-flight-notifications.md) がある。[rl.hpp](../../src/core/base/include/anet/rl.hpp) の `Notifier`・`BatchStepResult::GetEpisodeEndMask`・`ReplayInitialPriorityHint`・`BatchActionInfo` の宣言と、Runner の Panel と RunnerFrame が Observer を付ける 8 か所に、D7 のコメントがある。[rl.cpp](../../src/core/base/src/rl.cpp) の D1 の 7 つの口が、どれも同じ排他の下で列を読み書きしている | 文書とコードを読んで確かめる。A3 は、競合が無いことの証明ではなく、回帰の検知である。データ競合は、落ちずに通ることがあるためである |

検索 1（範囲は [QValuePanel.hpp](../../src/apps/runner/src/QValuePanel.hpp) の 1 ファイル。加工への入力から、有効かどうかと半径が消えたことを見る）:

```text
git grep -n -E "use_hist_smooth|smooth_radius" -- src/apps/runner/src/QValuePanel.hpp
```

検索 2（範囲は `src` の全体。固定の文字列で引く）:

```text
git grep -n -F "across devices" -- src
```

## 8. 成功の測り方

- 受入 A1〜A15 が通る。
- 題材の 5 件の判定が「残件無し」になる。
- 実装のあと、学習中の Panel の開閉と、終了時に、`Notifier` の Observer の列を走査する箇所で落ちた dump が出ない。出たら、その経路を題材 80c355 の対応内容に足して、判定を「残件有り」に戻す。

## 9. 未決

| # | 事項 | 扱い |
|---|---|---|
| U1 | Panel を閉じる瞬間に、その Panel の Observer が別のスレッドで動いていて、壊れかけの Panel を触る危険。起草の時点では「今もある危険」と書いていたが、たどり直すと、今のコードでは起きる経路が無かった（§1.2 の 80c355 の 7 行目） | 取り下げた（2026-11-20）。起票しない。今の安全が頼っているものは、D7 のコメントで残す |
| U2 | RFD0146 の R3・G3・AC5（「Detach は Runner の停止後だけ」をコメントの決まりにする）と NG1（排他を入れない）は、本書の D1〜D3 と食い違う。RFD0146 で `Detach` が効くようになると、Panel への弱い参照を持つ Observer が、学習中に外れ、通知のスレッドで壊れうる（§6.2）。弱い参照は、GUI スレッドの外で壊すと、Panel の破棄と重なって守られない。RFD0146 を使うときに、あわせて決める必要がある | 範囲外。RFD0146 の課題 [issue-20261103-9df5c5](../management/issues/issue-20261103-9df5c5.md) の対応方針へ追記した（2026-11-20、ユーザーの選択） |
| U3 | 動画の最初の画像が無効だと、0×0 の大きさで VideoLogger が作られる（§1.2、§6.3）。以後の有効な画像は、D8 で大きさ違いとして落ちる | 範囲外。2026-11-20 の起票の候補では、採らなかった |

## 10. グリルの材料

決定として扱わない。決定は §5 が正本である。

開発者の発言（2026-11-19）:

- ゴールと非ゴール（§2）、共通の前提 P1〜P6 に「これでよい」。説明が要る前提の番号は挙がらなかった。
- 質問 1（`Detach` が戻ったあとの保証）に「A（待たない形）」。
- 質問 2（動画の大きさ違いの扱い）に「B（WARN で落とす）」。
- 質問 3（初期化関数を 2 回呼んだときの扱い）に「A（どれも 1 回だけ登録）」。
- 質問 4（強制行動の device を合わせる側）に「A（受ける側）」。
- 細部 18 件の扱いに「推奨で埋める」。
- 質問 5（2026-11-20。別モデルの検討のあと。強制行動を差し替えた結果の行動の device）に「B（渡されたまま）」。
- 起票の候補 4 件（2026-11-20）に「1,2」と、「1 は起票したら本 RFD に含める」。候補 1 は U1 の当初の内容、候補 2 は U2、候補 3 は U3、候補 4 は RFD0161 の §1.4 の記述だった。
- 候補 1 の根拠を AI がたどり直して誤りと分かり、扱いを聞き直した問い（2026-11-20）に「A で進める」。A は、起票せず、記述を事実に合わせて直し、Observer を付ける箇所のコメントの決まりを D7 に足す案である。

AI が示した細部 18 件（推奨の既定）と、書いた先:

| 細部 | 書いた先 |
|---|---|
| 1. `Notify` に足す負荷を、Observer の数に比例させない | D5 |
| 2. 排他の対象に `Clear`・`Shutdown`・`WillBlockOnShutdown`・`LogObservers` を含め、Observer を呼ぶあいだロックを持たない | D1・D3 |
| 3. 終了時の順序は変えない | D6 |
| 4. 守りの無いキャッシュ 2 つは、守りを足さず、決まりをコメントに書く | D7 |
| 5. 性能の実測の受入を置かず、見積もりと、メトリクスの一致で確かめる | D15、A13 |
| 6. 同時実行のテストを足し、時間のかかる回し方は既定のスイートから外す | A3 |
| 7. WARN の文言に tag と 2 つの大きさを入れ、落とした枚数は数えない | D8 |
| 8. PNG の個別保存は今のまま | D8 |
| 9. 止めるときの文言は Env と同じ形 | D10 |
| 10. 重複登録のテスト 3 つと、初期化関数を 2 回呼ぶテスト 9 個。CartPole は足さない | A7・A8。CartPole は、別モデルの検討で、今ある実行体で確かめられると分かったので、A8 に足した |
| 11. `smooth_radius` の上限は置かない | D12 |
| 12. 加工への入力は、平滑化の指定を重み 1 組で持つ | D13 |
| 13. 過去の記録は直さない | D14 |
| 14. 差し替えた結果の行動は、ヒントの有無に依らず、差し替える前の行動と同じ device に載せる | 採らなかった。別モデルの反対を受けて質問 5 で聞き直し、結果の行動は渡されたままにした（D9、§5.1） |
| 15. 強制行動の device のテストは、CUDA が使えるときだけ走る | A6 |
| 16. ADR は 1 本 | D17 |
| 17. 設計文書を同じ変更で直す | §6.7 |
| 18. 5 件を、1 件ずつ実装・検証できる段階に分ける | D16、§6.8 |

AI が聞かずに決めた事項（規約と今のコードから 1 つに決まるとして示し、番号は挙がらなかった）: Observer を呼ぶ順とスレッドを変えない（D4）、重複登録を止める（D10）、`smooth_radius` の値域と検証の時機（D12）、既定の設定が検証を通ること（§6.7）、ヒントの取り直しの計算を変えないこと（D9）、本書を RFD0146 より先に実装すること（§6.9）。

別モデル（Opus）の検討（2026-11-20）で足した・直した点。どれも、コードで確かめた事実か、決定から導ける事項である。

- Observer が通知のスレッドで壊れうること（§6.2、U2、ADR 0082）と、今は学習中に外れる Observer が無いこと（P17）。
- `Clear` と `Shutdown` が見る Observer の範囲（§6.2）。
- キャッシュの決まりを、関数ではなく実体の単位で書くこと（D7）と、確かめた範囲（§1.2）。
- 弱い参照の確かめは、スレッドをまたぐと守られないという指摘。起票の候補のあとにたどり直し、弱い参照を持つ Observer は GUI スレッドの上でだけ動くと分かったので、P6 にはその事実を書いた（末尾の「起票の候補のあとの訂正」）。RFD0146 のあとに出る場面は U2 に書いた。あわせて、D2 の理由を並べ直した。
- 無効な画像の扱い（D8、§6.3、P18、U3）。
- 加工が受け付ける重みの数（§6.6、A11）。
- 行動の情報が元の device を記録しないこと（P19）。これを受けて、D9 を質問 5 で決め直した。
- 46 か所の内訳（§5.1）、`RegisterStandardNetworkModules` を直さないこと（§6.5）、CartPole の確かめ（A8）、A3 の位置づけ（A15）、参照カウントの上げ下げが減ること（§1.3）。

別モデルの論点のうち採らなかったもの: 「学習中に `EvalPanel::DoClose` が `Detach` している」という根拠。`DoClose` を呼ぶのは、RunnerFrame の閉じる処理（排水のあと）と EvalPanel の破棄だけである（§1.2 の 80c355 の 6 行目）。

起票の候補のあとの訂正（2026-11-20）。AI は、質問 1 と起票の候補 1 で、Panel を閉じる瞬間の危険を「今もある危険」と示したが、Observer が動くスレッドをたどっていなかった。たどった結果は §1.2 の 80c355 の 7 行目である。質問 1 の選択の理由には影響しない。聞き直しの問いでは、コメントを書く箇所を 7 か所と示したが、数え直すと 8 か所だった。P6 の 3 文目から後ろは、この訂正で書き換えた。書き換えた内容（どの Observer がどのスレッドで動き、何で守られているか）は、聞き直しの問いの応答で開発者に示している。
