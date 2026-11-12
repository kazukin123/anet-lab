# PRD938: forward の名前付き出力を capture に一本化し、nn_trace を要求した step だけ作る PRD

> 2026-11-12 改訂: グリルを行い、暫定の印を外した。§4 を確定事項として書き直し（D1〜D18）、起票時の未決表（旧 §8 の 8 行）と提案の骨子（旧 §4 の 5 項）を確定事項・現状・実装計画へ振り分けた（対応は §4.2）。主な決定: nn_trace は要求した step だけ作る（旧非ゴールの「性能の向上」は「要求したときの費用を下げること」に絞った）。要求は `MakeAction` と Runner の step の引数 `MakeActionOptions` で都度渡し、core に要求の登録を持たない（D2〜D4・D6、[ADR 0072](../adr/0072-make-action-inputs-are-passed-per-call-through-runner-step-not-registered.md)）。起票時の既定「購読ヒントと同じ経路で渡す」は採らない（§4.1）。capture の中の鍵は `nn_trace/` でそろえる（D5）。課題 6a32b4 の追記（capture を入口で Context に入れる）を範囲に入れた（D10）。旧 §7「作る条件」は、着手の時機を課題で表すので消した（前提だった 093 の capture はコミット済み）。用語集に `nn_trace` を足し、ADR 0031・0055 の冒頭に注記した（D16）。流れが追いにくいので、実装のときに設計文書 140 へシーケンス図を足すことを、移行対象（§6.4）と受入（A13）に入れた（ユーザーの指示）。範囲外とした 3 件（§8 の 1〜3）は、2 件を既存の課題へ追記し、1 件を見送りにした。
> 2026-11-03 番号を 999 から 938 へ振り直した（PRD114）。
> 2026-10-16 093 の 3 回目のグリルで、接頭辞要求（末尾 `/`）・`GetAll`・NetworkStruct による実行位置の付与（`<branch>/<block インスタンス名>`）を 093 側へ先取りした。本 PRD の残りは trace の書き手と読み手の付け替えと `TraceCallback` の削除。§2・§4 の 1 と 2・§5・§8 の 1・§9 を合わせた。
> 2026-10-16 起票。暫定の既定を置き、決めることを §8 の表に残した。詳細はグリルで詰める前提。
> 起点: [093_attention_entropy_diagnostics_10prd.md](done/093_attention_entropy_diagnostics_10prd.md) のグリルで、`NetworkBranchCapture` を鍵要求型の `NetworkForwardCapture` に一般化した（[ADR 0055](../adr/0055-forward-capture-request-keyed-carries-module-diagnostics.md)）。これで forward の中の名前付き tensor を外へ出す機構が、capture と `TraceCallback` の 2 つになる。関心は 1 つなので、残る `TraceCallback` を capture に吸収する。
> 関連: [ADR 0031](../adr/0031-plasticity-metrics-out-of-band-partial-forward.md)（capture の契約と、TraceCallback の常時 ON 問題）、[ADR 0051](../adr/0051-nn-forward-prepares-concrete-calls-and-purpose-owned-execution.md)（Context）、[done/062_plasticity_metrics_10prd.md](done/062_plasticity_metrics_10prd.md) の 14（TraceCallback が capture に使えない理由）、[done/069_metrics_trace_channel_10prd.md](done/069_metrics_trace_channel_10prd.md) の D1・F9（用語「trace」との区別）、[done/006_conv2d_visual_trace_10prd.md](done/006_conv2d_visual_trace_10prd.md) §7 の 8（常に作る費用への対処を先送りした）、[944_observation_pipeline_10prd.md](944_observation_pipeline_10prd.md)（強制行動を `MakeAction` の入力にする。§9）。

## 1. 目的と問題

forward の途中で作った Tensor を外へ持ち出す口が 2 つある。

| 口 | 渡し方 | 書く側 | 読む側 |
|---|---|---|---|
| `TraceCallback` | forward の途中で関数を呼んで押し出す。何が欲しいかを指定する仕組みが無い | `NetworkStruct` | Actor が受け取り、AuxData の `nn_trace/…` に積む |
| `NetworkForwardCapture`（093 で作った） | 呼ぶ側が鍵で要求し、forward のあとで取り出す。要求の無いものは作らない | body と module | Learner の診断 |

違いは運び方だけで、関心は同じである（[ADR 0055](../adr/0055-forward-capture-request-keyed-carries-module-diagnostics.md) は「forward の持ち帰りの概念を 1 つに保つ」として、nn_trace の取り込みを本 PRD へ先送りした）。そのうえ `TraceCallback` は要求を受けないので、次の 2 つが起きている。

- 読む人がいなくても、Actor が `MakeAction` のたびに nn_trace を作る（[ADR 0031](../adr/0031-plasticity-metrics-out-of-band-partial-forward.md) の A''、[PRD006](done/006_conv2d_visual_trace_10prd.md) §7 の 8）。出荷設定では、学習側の Runner の nn_trace を読む人はいない（§2.4）。
- 追加の計算が要る出力を、この口には乗せられない。要求が無いので、乗せると常に計算することになる。

## 2. 現状（2026-11-12 に作業ツリーで確認）

`nn_impl.cpp` は並行する作業で行が動いているので、位置はシンボル名で示す。実装時はシンボル名で探し直す。

### 2.1 nn_trace の今の経路（`TraceCallback`）

- 型は `TraceCallback`（[common.hpp](../../core/anet-core/include/anet/common.hpp)）。
- 書き手は `NetworkStruct::Forward`（[nn_impl.cpp](../../core/anet-core/src/nn_impl.cpp)）。次の 2 種類を渡す。
  - branch の入力を `00_Input` で渡す。形を問わず、全 branch で出る（IQN の `tau_embedding` の 2 次元の入力も出る）。
  - block の出力のうち、可視化できる block（`IsConv2dVisualizable`）の、4 次元・浮動小数・縦横が 2 以上のものを `<2 桁の番号>_<block 名>` で渡す。番号は、渡した出力だけを 1 から数えた通し番号で、block の位置ではない。
- `NetworkBranch::Execute` が、鍵の頭に `<branch>/` を付ける。
- Actor の側は、`dqn::Actor::MakeAction`（Rainbow も同じ Actor。[dqn_based_agent.cpp](../../core/anet-core/src/dqn_based_agent.cpp)）と `ImageClsActor::MakeAction`（[image_cls_agent.cpp](../../core/anet-core/src/image_cls_agent.cpp)）が、呼び出しのたびに `MakeActionTraceCallback` で callback を作る（[rl.cpp](../../core/anet-core/src/rl.cpp)）。callback は env 0 の行を切り出し、勾配を切り、FP32 にして複製する。行動選択の forward は autocast の下で走るので FP32 にする必要があり、切り出しただけでは batch 全体の記憶域を持ち続けるので複製が要る（[PRD006](done/006_conv2d_visual_trace_10prd.md) §7 の 1・2）。
- `AppendTraceAux` が鍵の頭に `nn_trace/` を付けて AuxData に入れる。AuxData の鍵は `nn_trace/<branch>/00_Input` と `nn_trace/<branch>/<2 桁の番号>_<block 名>` である。接頭辞の定数 `kNnTracePrefix` と、読む側の `ExtractNnTrace` は rl 層にある（[rl.hpp](../../core/anet-core/include/anet/rl.hpp)）。
- callback を forward まで運ぶだけの引数が、次にある。
  - forward の公開の口: `Network::Forward`、`NetworkForwardExecutor::Forward`（[nn.hpp](../../core/anet-core/include/anet/nn.hpp)）。
  - forward の内部: `RunForward`、`NetworkBody::Forward`、`NetworkBranch::Execute`、`NetworkStruct::Forward`（[nn_impl.hpp](../../core/anet-core/src/nn_impl.hpp)、[nn_impl.cpp](../../core/anet-core/src/nn_impl.cpp)）。
  - DQN 系の行動方策: `ActionPolicy::SelectAction`（基底と、`EpsilonGreedyActionPolicy`・`UQEActionPolicy`・`ThompsonSamplingActionPolicy` の実装）と、補助の `ForwardForAction`・`ForwardForActionWithTaus`・`MakeUQEActionInfo`（[dqn_based_agent.hpp](../../core/anet-core/src/dqn_based_agent.hpp)）。宣言は 7 つ。
- `SelectAction` は 1 回の呼び出しで forward を 1 回だけ行う。TempoRL の skip 側の forward と、Learner が target の方策を呼ぶ `SelectAction` は callback を渡していない。

### 2.2 capture の経路（`NetworkForwardCapture`、093）

- 要求は、完全一致の鍵か、末尾が `/` の接頭辞である。`GetAll` は、接頭辞に一致した鍵と Tensor を書かれた順に返す。要求していない鍵の書き込みと、同じ鍵への 2 回目の書き込みは即エラーになる。forward の出口で、満たされなかった要求（完全一致が書かれない、接頭辞に 1 件も一致しない）は即エラーになる（[nn_impl.cpp](../../core/anet-core/src/nn_impl.cpp) の `MatchesCaptureRequest`、`NetworkForwardCapture::Add`、`ValidateForwardCapture`）。起票時の未決表 1（接頭辞要求の記法）はここで済んでいる。
- 鍵は 2 系統ある。body が書く branch 名そのままの鍵（例 `main_feature`）と、module が書く `<型名>/<診断名>` の後ろに `/<branch>/<block>` が付いた鍵（例 `TransformerEncoder/attention_entropy/main_feature/encoder1`）である（`NetworkForwardContext::MakeCaptureKey`、[nn.hpp](../../core/anet-core/include/anet/nn.hpp) の `kTransformerEncoderAttentionEntropyCaptureKey`）。
- capture を渡さない forward は、null の判定 1 回だけで、ほかは変わらない（ADR 0055）。
- 受け渡しは 2 通りが混ざっている。module へは `NetworkForwardContext` が運ぶが、`Network::Forward`・`NetworkForwardExecutor::Forward` から `NetworkBody::Forward`・`ForwardUpTo` までは別の引数で運び、Body の入口で Context に入れている。`NetworkBody::Forward` の引数は 5 つ、`RunForward` は 7 つである（課題 [issue-20261103-6a32b4](../management/issues/issue-20261103-6a32b4.md) の 2026-11-11 の追記）。

### 2.3 forward の中で途中の出力は上書きされない

`nn_impl.cpp`・`nn_modules.cpp`・`nn_heads.cpp` に、block の出力を上書きする処理（末尾が `_` の Tensor の操作、`inplace` の指定）は無い。上書きがあるのは、スペクトル正規化の状態、診断の集計、パラメータの複写だけである。そのため、forward の途中で渡された Tensor を forward のあとで読んでも、値は同じである。

### 2.4 nn_trace を読む側

| 読む側 | 付く Runner | 読み方 |
|---|---|---|
| Conv2d の pane（`Conv2dPanel`） | EvalPanel の Runner だけ。メニューの Conv2d で開く（[RunnerFrame.cpp](../../apps/runner/src/RunnerFrame.cpp) の `OnConv2d`）。何枚でも開ける | Runner 限定の Observer で受けて描く。Trace の pane が追従する pane として、行ごとの控えも作る（[Conv2dPanel.cpp](../../apps/runner/src/Conv2dPanel.cpp)） |
| メトリクスの画像記録（`Conv2dVisualizationObserver`） | 学習側の Runner だけ。`metrics.image.*` に Conv2d の記録を書いた Run で、開始前に 1 回登録する（[RunnerApp.cpp](../../apps/runner/src/RunnerApp.cpp) の `ImageProviderManager::CreateAndRegister`）。出荷設定では [DropMerge.txt](../../apps/runner/config/DropMerge.txt) と [ImageCls.txt](../../apps/runner/config/ImageCls.txt) が 1 件ずつ登録しているが、どちらも `interval = 0` である。`interval` が 0 以下の記録は、Observer が作られて付くだけで録画を始めないので、nn_trace を読まない | `episode_interval` エピソードごとに 1 エピソードを録画する。録画するかは、エピソードの最初の step の通知を受けてから決め、その step から描く（[observers.cpp](../../core/anet-core/src/observers.cpp) の `Conv2dVisualizationObserver::OnTrain`） |
| Trace のダンプ | EvalPanel の Runner | Trace の各行は通知の `action_info` を丸ごと持ち（最大 `app.eval_trace_panel.max_rows` 行、既定 1000）、ダンプは AuxData の鍵を全部書く。1 行のダンプは各 Tensor の形・型・置き場所だけ、詳しいダンプは全体を文字にする（[EvalTracePanel.cpp](../../apps/runner/src/EvalTracePanel.cpp)、[gui.cpp](../../core/anet-core/src/gui.cpp) の `TraceDumpFormatter`）。nn_trace に固有の読み方はしていない |
| テスト | — | `[native_stack]` の 5 件（[dqn_based_agent_test.cpp](../../core/anet-core/src/dqn_based_agent_test.cpp)）と ImageCls の 1 件（[image_cls_agent_test.cpp](../../core/anet-core/src/image_cls_agent_test.cpp) の `[image_cls][trace]`）が、`ExtractNnTrace` で鍵と形を見る。このほか [nn_test.cpp](../../core/anet-core/src/nn_test.cpp) の 2 件が、`Network::Forward` に callback の関数を直接渡して、押し出された Tensor を見る（3 か所） |

背景評価の Runner には、Conv2d の pane も画像記録も付かない。

### 2.5 step を進める側と、今ある入力・出力の仕組み

step を進めているのは次の 3 者である。

| 進める側 | 呼び方 |
|---|---|
| EvalPanel（GUI のスレッド） | タイマーが `DoUpdateFrame`、手動の step が `DoStep()`、手動の行動が `EvalRunner::DoStep(action)`（[EvalPanel.cpp](../../apps/runner/src/EvalPanel.cpp)） |
| 学習側（`RunnerThread` のスレッド） | `DoUpdateFrame(1, …)` を繰り返す（[trainer.cpp](../../core/anet-core/src/trainer.cpp) の `RunnerThread::ProcessStep`）。`RunnerThread` は RunnerApp が作る |
| 背景評価 | `EvalRunner::RunSession` |

このほか、Q-Values の pane（[QValuePanel.cpp](../../apps/runner/src/QValuePanel.cpp)）に、行動の受け口が設定されていないときに `EvalRunner::DoStep(action)` を直接呼ぶ予備の経路がある。本番では RunnerFrame が受け口を設定するので通らない。

呼び出し側には、向きの違う 2 つの仕組みがある。

- 出力（結果を配る）: 読む側が `Notifier` に Observer を付けておくと、step のたびに通知が届く。
- 入力（step に渡す）: `EvalRunner::DoStep(action)` が、手動の行動を呼び出しごとの引数で受ける。基底の `Runner::DoStep`・`DoUpdateFrame` には無く、`EvalRunner` の `DoStep` はオーバーロードが 4 本ある（[trainer.hpp](../../core/anet-core/include/anet/trainer.hpp)）。

Learner の診断は、学習の前に 1 回だけ渡す購読ヒント（`Agent::ConfigureScalarMetricSubscriptions`、ADR 0031）で「何が要るか」を受ける。Conv2d の pane は Run の途中で開閉するので、この経路には乗らない。

### 2.6 `MakeAction` の実装と呼び出し

- `rl::Actor::MakeAction(step, state)` の実装は、本番が 3 つ（`dqn::Actor`、`ImageClsActor`、`MuZeroActor`）、テスト用の差し替え部品が 4 つ（`dqn_based_agent_test.cpp`、`episode_end_test.cpp`、`observers_test.cpp`、`trainer_test.cpp`）。`MuZeroActor` は nn_trace を作らない。
- 呼び出しは、本番が `trainer.cpp` の 3 か所（`EvalRunner::DoStepInternal`、`SerialTrainRunner::DoStep`、`PipelineTrainRunner::DoStep`）、テストが約 65 か所。
- Runner のテスト用の差し替え部品は 3 つある（`episode_end_test.cpp` と `observers_test.cpp` の `TestRunner`、`trainer_test.cpp` の `ThrowingRunner`）。
- Actor は Runner ごとに 1 つで、Runner を作るときに 1 回だけ作る（`RunnerBase` のコンストラクタ）。

## 3. ゴール・非ゴール・成功の測り方

### 3.1 ゴール

- **G1**: 持ち出す口を capture 1 つにする。`TraceCallback` 型と、forward 系・`SelectAction` 系の callback の引数を消す。
- **G2**: nn_trace を要求したときの、AuxData の `nn_trace/*` の鍵の集合と値を、統合の前後で変えない（DefaultDQN 系と ImageCls の両方）。
- **G3**: nn_trace は要求した step だけ作る。読む側（Conv2d の pane、メトリクスの画像記録、テスト）のために step を進める側が要求し、要求の無い step では、Actor は切り出し・FP32 化・複製をしない。

### 3.2 非ゴール

どれも本当の制約ではなく、まとめて直せば移せる。今回は変える理由が無いので外す。

- 表示の内容、鍵名、nn_trace に出す block を変えること。消える案は「`00_Input` を描画できる形の branch だけにする」「番号を全 block の通し番号にする」。鍵が変わり、Conv2d の pane とテストも一緒に直すことになるが、表示を変える理由が今は無い。
- 093 の attention 診断を変えること。
- 要求したときの nn_trace の費用を下げること。消える案は「複製や FP32 化を省く」「録画するエピソードの間だけ要求する」。費用を測っておらず、関心も別である。
- 追加の計算が要る出力（attention の重みなど）を nn_trace に足すこと。読む人がいない。本 PRD の形にしておけば、あとで鍵を足すだけで出せる。
- `EvalRunner::DoStep` の手動の行動の引数を `MakeActionOptions` にまとめること。[PRD944](944_observation_pipeline_10prd.md) の範囲である（§9）。
- Learner の購読ヒントを見直すこと（§8 の 1）。

### 3.3 成功の測り方

- G1: 検索で、`TraceCallback` と `MakeActionTraceCallback` が現用のコードと文書から消えていること（§7 の A9）。
- G2: 変更前のビルドと変更後のビルドで、同じ seed の `nn_trace/*` の鍵・形・型・値が完全に一致すること（D17、§7 の A1）。
- G3: 要求しない呼び出しの AuxData に、`nn_trace/` で始まる鍵が無いこと（§7 の A2・A4・A6・A7）。
- 速さは記録にとどめ、合否の閾値を置かない（D18）。学習側の 1 秒あたりの step 数を、変更の前後で測って記録する。

### 3.4 用語

- **nn_trace**: Actor が行動選択の forward から取り出す、branch の入力と描画できる block の出力。要求した step だけ作り、AuxData の `nn_trace/` に載る。用語集に足す（D5・D16）。既存の語「trace」はメトリクスのチャネルの名前なので、言い換えに使わない。
- **`MakeActionOptions`**: Runner の step を呼ぶ側が、その回の行動選択に添える入力。今の欄は `nn_trace` の 1 つ（D6）。
- Observer・Notifier・Runner・Actor はクラス由来の語で、訳さずに書く。

### 3.5 機構とゴールの対応

| | 機構 | すること | 効くゴール |
|---|---|---|---|
| A | nn_trace の鍵を capture に書く | `NetworkStruct` が、要求があるときだけ、branch の入力と描画できる block の出力を capture に書く（D5・D8） | G1・G2 |
| B | Actor が nn_trace を組み立てる | forward のあとで capture から取り出し、env 0 の行を切り出し、FP32 にして複製し、AuxData に積む（D7） | G1・G2 |
| C | 要求を都度渡す | step を進める側が、`MakeActionOptions` を Runner の step へ渡し、Runner がそのまま `MakeAction` へ渡す（D2〜D4・D6） | G3 |
| D | 消すもの | `TraceCallback` 型、callback の引数、`MakeActionTraceCallback`（D9・D11） | G1 |
| E | capture を入口で Context に入れる | forward の公開の口で capture を Context に入れ、内部の capture の引数を消す（D10） | G1 を補う。持ち出しの道が Context の 1 本にそろう |

先送りした機構は 1 つある。capture の要求に行の指定を足す形である（§4.1）。理由は、実験の結果が要るためで、nn_trace を要求した step の GPU メモリの増分を測ってから決める（D7、§8 の 4）。

## 4. 確定事項

2026-11-12 のグリルで決めた。D1〜D6 は問いへの答え、D10 は範囲の確認で取り込んだもの、D7〜D9・D15・D16 は規約・ADR・ほかの決定から決まるもの、D11・D13・D14・D17・D18 は推奨の既定を採った細部である。D12 は、推奨の既定が規約と食い違っていたので、レビューのあとで直した。

### D1. nn_trace を読む側は 3 者。Trace のダンプは数えない

nn_trace を読む側として数えるのは、Conv2d の pane、メトリクスの画像記録、テストである。step を進める側は、この 3 者のためにだけ nn_trace を要求する（渡し方は D4）。Trace のダンプは数えない。ダンプは AuxData にあるものを書くだけなので、Conv2d の pane を閉じている間に記録した行には `nn_trace/…` の項目が出ない。詳しいダンプで途中の出力の値を見たいときは、Conv2d の pane を開いてから記録する。

理由: 1 行のダンプに出る nn_trace の項目は形・型・置き場所だけで、行動の判断を確かめるというダンプの用途に使っていない。途中の出力は Conv2d の pane で見られる。Trace の記録中も要求する形にすると、EvalPanel の費用が今と変わらず、G3 の効果が学習側だけになる。依存: §2.4 の事実。

### D2. 要求は `MakeAction` の入力で渡す

`rl::Actor::MakeAction` に、3 つ目の入力として `MakeActionOptions` を足す。Actor は、その欄 `nn_trace` が真の呼び出しだけ nn_trace を作る。Actor は、読む側の数や状態を持たない。

理由: 行動を決める点への入力として渡すと、何を求めた呼び出しかが呼び出しの場で読め、`MakeAction` の結果が引数だけで決まる。Actor の中に要求の数を持つ形は、別のスレッドから変わる値で出力が決まる。入力を構造体にしておけば、あとの入力（PRD944 の強制行動を含む）は欄の追加で済み、基底の引数を変えるのは今回の 1 回になる。依存: §2.5（購読ヒントは Run の途中の開閉に追従できない）。

### D3. 3 つ目の引数には既定値を付ける

`MakeAction` の 3 つ目の引数に、何も求めない既定値（空の `MakeActionOptions`）を付ける。基底と override の全部に同じ既定値を書く。追加の入力を求めない呼び出しは、今のまま 2 つの引数で書ける。

理由: テストの呼び出し約 65 か所の大半は nn_trace を求めないので、書き換えずに済む。既定値つきの virtual には `GetScalar(key, index = -1)` の前例がある。依存: D2。Runner が渡し忘れてもコンパイルは通るので、渡していることは受入で確かめる（§7 の A4〜A7）。

### D4. Runner の step にも都度渡す。core に要求の登録を持たない

基底の `Runner::DoStep` と `Runner::DoUpdateFrame` に、同じ `MakeActionOptions` の引数を既定値つきで足す。Runner は、受け取った値をそのまま `MakeAction` へ渡す。`DoUpdateFrame` は、そのフレームの全 step に同じ値を渡す。「nn_trace が要るか」は、step を進める側が、その時点の事情を見て毎回決める。core（Runner、Actor、`Notifier`、Observer）には、読む側が要求を登録する仕組みを足さない。`RunnerThread` は step を進める側であり、作るときに受けた値を step のたびに渡すだけで、読む側の登録先ではない。

理由: 入力は step の引数で渡し、出力は Observer で受ける、という今の分担に沿う（§2.5）。D2 で選んだ「結果は引数だけで決まる」が Runner でも成り立つ。渡す側と step が同じスレッドなので、複数のスレッドで共有する値が要らない。依存: D2・D3。弱点: 読む側が自立しない。pane を開いただけでは nn_trace は出ず、step を進める側につなぐ必要がある。値を渡す箇所は §5.4 の EvalPanel と学習側の 2 つで、渡し忘れは受入で押さえる。

### D5. capture の中の鍵は `nn_trace/<branch>/<番号>_<block>`。呼び名は nn_trace の 1 語

capture の中の nn_trace の鍵は、AuxData と同じ `nn_trace/<branch>/00_Input`、`nn_trace/<branch>/<2 桁の番号>_<block 名>` にする。Actor は `nn_trace/` の接頭辞で要求し、AuxData に積むときに鍵を書き換えない。接頭辞の定数は NN 層へ移し、rl 層の読む側（`ExtractNnTrace`）もそれを使う。定数は 1 つにする。NN 層、`MakeActionOptions` の欄、AuxData のすべてで、呼び名を nn_trace の 1 語にそろえる。

理由: 同じものを層ごとに別の名前で呼ばずに済み、Actor が鍵を付け替える処理も要らない。branch 名が `nn_trace` でも、body の鍵は `nn_trace`（完全一致）、nn_trace の鍵は `nn_trace/…` なので紛れない。依存: G2（AuxData の鍵を変えない）、§2.2（要求は完全一致か接頭辞）。capture の鍵は、body の鍵・module の鍵に続く 3 つ目の系統になり、書き手は `NetworkStruct` である。

### D6. 構造体の名前は `MakeActionOptions`

名前は `MakeActionOptions`、欄は `nn_trace`（真か偽）とする。

理由: 何のための入力かを名前が直接表し、LibTorch の `…Options` の流儀とも近い（ユーザーの選択）。Runner の step の引数に出ても、「`MakeAction` に添える入力」と読める。既存の `ActorRequest`（Actor を作るときの入力）とも紛れない。Learner 側に同じ役割のものを作るときの対は `UpdateFromBatchOptions` になる。欄の `nn_trace` は rl 層が持つ共通の概念で（接頭辞の定数と読み口が rl 層にあり、DQN 系と ImageCls の両方が書く。§2.1）、特定の Agent の名前を汎用の層へ持ち込むものではない。

### D7. 切り出し・FP32 化・複製は、Actor が forward のあとで行う

capture は、書かれた Tensor をそのまま持つ。env 0 の行の切り出し、FP32 化、複製は、Actor が forward のあとで capture から取り出して行う。処理の中身は今の `MakeActionTraceCallback` と同じにする。取り出しは、行動を選ぶ forward が返った直後に行い、capture が持つ Tensor をそこで手放す。nn_trace を要求した step では、描画できる block の出力を batch 全体のまま forward の終わりまで持つので、その間の GPU メモリの山は今より増える。これは受け入れる（§8 の 4 で量を測る）。

理由: env 0 は Actor の lane の概念で、NN 層は知らない（[AGENTS.md](../../AGENTS.md) §汎用機構と利用側の責任境界）。capture に「env 0 を切り出して FP32 にする」という変換の指定を持たせると、汎用の器に Actor の都合が入る（ADR 0055）。行の範囲のような汎用の指定なら足せるが、今回は足さない（§4.1 の先送り）。forward のあとで読んでも値が同じであることは §2.3 による。メモリの山が増えるのは読む人がいる step だけで、読む人がいない step の分（今は毎 step 作っている）が無くなるので、全体としては軽くなる。量が分からないうちに capture の器へ行の指定を足すより、測ってから決める（ユーザーの選択）。依存: §2.3、G2。

### D8. 番号は `NetworkStruct` が付ける

`<2 桁の番号>` は、今と同じく `NetworkStruct` が、描画できる出力だけを 1 から数えて付ける。

理由: どの block の出力が描画できるかは、forward の中でしか分からない。依存: G2。

### D9. DefaultDQN 系と ImageCls を同時に移し、消すものは一度に消す

`dqn::Actor`（Rainbow を含む）と `ImageClsActor` を同時に移す。`TraceCallback` 型、callback の引数、`MakeActionTraceCallback` は、別名や互換の口を残さずに一度に消す。

理由: 型を消すので、片方だけ残すとビルドが通らない。契約の変更は互換の層を積まずに移す（[AGENTS.md](../../AGENTS.md) §クリーンブレーク方針）。依存: G1。

### D10. capture は forward の公開の口で Context に入れる

`Network::Forward`・`NetworkForwardExecutor::Forward`・`Network::ForwardUpTo` が受けた capture を、入口で `NetworkForwardContext` に入れる。内部の `RunForward`・`NetworkBody::Forward`・`NetworkBody::ForwardUpTo` から capture の引数を消し、Body は Context から capture を受け取って、前回値の破棄と出口の検査を行う。公開の口が capture を引数で受ける形（ADR 0055）は変えない（変えない理由は D11）。Head へ渡す Context には capture を入れない（今と同じ）。前回値の破棄と出口の検査も、今と同じく Body の入口と出口で行う。

理由: A で `NetworkStruct` も Context 経由で capture に書くので、持ち出しの道が Context の 1 本にそろう。D9 で同じ関数の引数を直すので、変更が 1 回で済む。課題 6a32b4 の 2026-11-11 の追記を取り込んだ。capture の書き手は body と module（と本 PRD の `NetworkStruct`）で、Head は書き手ではない。出口の検査は Head を動かす前にあるので、Head に capture を見せると、検査を通らない書き込みができてしまう。依存: D9。追記が触れている、既定の forward と準備済みの forward を切り替える呼び先の表の受け渡しは、nn_trace とは別の関心なので取り込まない（§8 の 3）。

### D11. `SelectAction` 系は、callback の引数を capture の引数に置き換える

`ActionPolicy::SelectAction` とその補助（宣言 7 つ、§2.1）の callback の引数を、capture を受ける引数に置き換える。渡さないときは capture なしの forward になる。

理由: forward の口が capture を引数で受ける形にそのままつながり、新しい型を作らずに済む。forward の呼び出し情報 `NetworkForwardCallInfo` に capture を持たせる形は採らない。ADR 0055 の「公開の forward が capture を引数で受ける」形は本当の制約ではなく、093 の診断の呼び出しをまとめて直せば移せる。それでも変えないのは、変える利点が無いためである。呼び出し情報を受けない `Network::Forward` と `Network::ForwardUpTo` にも capture の口が要るので、呼び出し情報へ移しても capture の引数は残り、渡し方が 2 通りになる。依存: §2.1（`SelectAction` は forward を 1 回だけ行う）。

### D12. nn_trace を作れない Actor は、要求を受けたら WARN を 1 回出し、何も入れない

nn_trace を作る実装を持たない本番の Actor（`dqn::Actor`・`ImageClsActor` 以外。今は `MuZeroActor`）は、`nn_trace` が真の呼び出しを受けても、AuxData に何も入れない。最初に受けたときに、その Actor が nn_trace を作らないことを WARN で 1 回だけ知らせる。Run は止めない。`nn_trace` が偽の呼び出しでは何も出さない。テスト用の差し替え部品は、この決まりの対象にしない。

理由: [AGENTS.md](../../AGENTS.md) §Fail-Fast 原則の表は、「実装の欠けによる機能の一部の省略」を、Run ごとに対象につき 1 回の WARN と定めている。Conv2d の pane を開いても画像が出ないのは、これに当たる。今は何も出ないので、pane が空の理由が利用者に分からない。起草の時点では「何もしない。WARN も出さない」としていたが、この表の行を見落としていたので、レビューのあとで直した（ユーザーの選択）。依存: D2。

### D13. Conv2d の pane が「開いている」の判定は、Trace の追従の判定と同じにする

EvalPanel は、自分の Runner に付いた Conv2d の pane が 1 枚以上開いている間、step に `nn_trace` を渡す。「開いている」は、Trace の pane が追従する pane に使っている判定（`IsPaneOpen`）と同じにする。

理由: Trace は「開いている pane」の分だけ行の控えを取る。判定を別にすると、控えを取る行に nn_trace が無い、という食い違いが起きる。依存: D4。

### D14. 録画する Conv2d の画像記録を登録した Run では、学習側の step にずっと渡す

`metrics.image.*` に、`interval` が 1 以上の Conv2d の画像記録を 1 件でも登録した Run では、学習側の step に Run の間ずっと `nn_trace` を渡す。`interval` が 0 以下の記録だけを登録した Run（出荷の DropMerge・ImageCls）では渡さない。録画するエピソードの間だけに絞ることはしない。値は、RunnerApp が登録の内容から決め、`RunnerThread` へ渡す。

理由: `interval` が 0 以下の記録は録画を始めず、nn_trace を読まないので、読む側（D1）に当たらない。登録の有無だけで決めると、出荷の DropMerge と ImageCls の学習で、読む人がいないまま毎 step 作ることになり、G3 に反する。録画するエピソードの間だけに絞るのは、非ゴールの「要求したときの費用を下げること」に当たる。登録は開始前の 1 回だけなので、Run の間は値が変わらない。依存: D1・D4、§2.4。

### D15. 背景評価は変えない

`EvalRunner::RunSession` は `MakeActionOptions` を受けず、nn_trace を求めない。

理由: 背景評価の Runner の nn_trace を読む人がいない（§2.4）。依存: D1・D4。

### D16. 用語集と ADR

- 用語集: `capture` の項の定義の書き手に、`NetworkStruct`（nn_trace）を足す（D5・D10）。`capture` の項の「避ける語」と `trace` の項の説明・「避ける語」から、`TraceCallback` への言及を外す。`nn_trace` の項を足す（§3.4）。
- ADR 0055: 「選ばなかった形」が nn_trace の capture への統合を本 PRD へ先送りしているので、統合を決めた旨を冒頭に注記する。本文は書き換えない。
- ADR 0031: Consequences が、購読ヒントを「TraceCallback が常に作られる問題」へ流用することを想定している。本 PRD はこの流用を採らない（§4.1 の 1 行目）ので、その旨を冒頭に注記する。購読ヒントが Learner の診断を駆動する決定は変えない。
- 新しい ADR を 1 本立てる（[ADR 0072](../adr/0072-make-action-inputs-are-passed-per-call-through-runner-step-not-registered.md)）。「行動選択に添える入力は、Runner の step と `MakeAction` の引数で都度渡し、core に要求の登録を持たない」。

理由: 用語は、消える型への言及を残さない（クリーンブレーク方針）。ADR は、基底のインタフェースの形で後から変えにくいこと、理由を知らないと「なぜ読む側が登録しないのか」と思う形であること、§4.1 の案と比べて選んだことの 3 つを満たす。依存: D2〜D6。

### D17. 「前後で鍵と値が同じ」は、2 つのビルドの書き出しを突き合わせて確かめる

変更の前に、決まった seed で `nn_trace/*` の鍵の並びと、各 Tensor の形・型・値を書き出すテストを足す。変更前のビルドと変更後のビルドで書き出しを突き合わせ、完全に一致することを受入とする（§7 の A1）。

理由: 同じ forward の同じ Tensor を切り出すだけなので、値は完全に一致するはずである。§2.3（途中の出力が上書きされない）も、この突き合わせで確かめられる。依存: G2、D7。

### D18. 速さは記録にとどめる

学習側の 1 秒あたりの step 数を、変更の前後でラウンドロビン 3 巡以上で測り、群ごとの中央値を記録する。nn_trace を要求した step の GPU メモリの増分も記録する（§8 の 4）。どちらにも合否の閾値は置かない。

理由: G3 の合否は「求めないときは AuxData に `nn_trace/` が無い」で決める。今の費用は、nn_trace 1 件につき小さな Tensor の操作 4 回（切り出し、勾配を切る、FP32 化、複製）で、Atari の構成では 1 step に 5 件前後、約 20 回の見積もりである。差が測定のぶれに埋もれる可能性がある。機械の速さは時間で動くので、前後を交互に測る。

### 4.1 選ばなかった形

| 形 | 裁定 | 理由 |
|---|---|---|
| 購読ヒントと同じ経路で「表示中か」を渡す（起票時の既定。[ADR 0031](../adr/0031-plasticity-metrics-out-of-band-partial-forward.md) の Consequences が想定していた流用） | 不採用 | 購読ヒントは学習の前に 1 回だけ渡す仕組みで、Run の途中で開閉する pane に追従できない |
| Actor の設定キーで渡す | 不採用 | 同じく追従できない。設定キーで eval にだけ作らせる形は [ADR 0058](../adr/0058-approx-priority-column-computed-by-agent-step-diagnostics.md) でも退けている |
| Actor が要求の数を持ち、読む側が Actor に登録する | 不採用 | `MakeAction` の出力が、引数に現れない値で決まる。Actor の基底に登録の口が増える |
| Runner が要求の数を持ち、読む側が Runner に登録する | 不採用 | step の結果が、引数に現れない Runner の中の数で決まる。入力の側に「事前の登録」という 2 つ目の渡し方ができる。変更は小さく、読む側が自立する利点はある |
| Observer が「nn_trace が要る」と宣言し、Runner が `Notifier` に聞く | 不採用 | 全イベント・全 Runner が共有する Observer の仕組みに nn_trace の話が入り、Learner の購読に近い形が Actor 側にも増える |
| 真偽の値をそのまま 3 つ目の引数にする | 不採用 | 入力が増えるたびに、基底と全実装の引数が変わる |
| 構造体の欄を、capture と同じ鍵の要求にする | 不採用 | AuxData のどの鍵を「要求したときだけ」にするか、知らない鍵をどう扱うかなど、決めることが増える。確かな必要は nn_trace の 1 つだけ。要求したときだけ作る出力が増えたら、欄を鍵の要求に替える。そのとき直すのは構造体とそれを使う箇所だけで済む |
| 3 つ目の引数に既定値を付けず、全部の呼び出しに書く | 不採用 | テストの呼び出し約 65 か所に空の値を足すことになる（ユーザーの選択） |
| 引数 2 つの版を virtual でない関数として残す | 不採用 | 派生クラスで 3 引数の版を override すると、基底の 2 引数の版が名前ごと隠れる |
| 公開の `MakeAction` を virtual でない 1 本にして既定値をそこだけに書き、実装は別名の virtual を override する | 不採用 | 既定値の食い違いは原理的に起きなくなるが、既定値は空の構造体なので食い違う余地がほとんど無い。Runner の step も同じ作りにそろえると、`EvalRunner` が自分の `DoStep` のオーバーロードを持つので、基底の virtual でない `DoStep` が名前ごと隠れる。今のコードに前例が無く、`GetScalar(key, index = -1)` の作りにそろえる（ユーザーの選択） |
| capture の鍵を module の流儀（`NetworkStruct/activation/…`）にする | 不採用 | capture では別の名前、AuxData では nn_trace の 2 語になり、Actor が頭を付け替える |
| capture の鍵の頭を `trace/` にする | 不採用 | 用語集の `trace`（メトリクスのチャネル）と紛れる |
| Trace の記録中は、Trace の pane も nn_trace を要求する | 不採用 | D1 の理由 |
| 切り出し・FP32 化・複製を capture の指定にする | 不採用 | D7 の理由 |
| capture の要求に「先頭の 1 行だけ」のような行の指定を足し、capture が書き込みの時点でその行だけを複製して持つ | 先送り | 要求した step のメモリの山は今と同じになるが、汎用の器に指定が増える。再開条件: §8 の 4 で測った増分が、開発者が大きいと判断する量だったとき |
| `NetworkForwardCallInfo` に capture を持たせる | 不採用 | D11 の理由（本当の制約ではないが、capture の引数が残って渡し方が 2 通りになる） |
| Actor が forward を包み、`SelectAction` から引数を無くす | 不採用 | 包むための新しい型が要る。`SelectAction` は forward の実行器を直接受けている |
| 読む側が、通知に載った観測から forward をやり直して途中の出力を得る（読む側が自立し、`MakeActionOptions` の受け渡しが要らなくなる） | 不採用 | [PRD006](done/006_conv2d_visual_trace_10prd.md) §2 が、行動選択に使った forward そのものの値を出す方針に決めている。やり直すと、NoisyNet の雑音や IQN の τ の抽選で値が食い違い、G2 が崩れる |
| 段階 1（統合だけ。常に要求）と段階 2（要求したときだけ）に分ける | 不採用 | G3 を範囲に入れたので、分けるのは実装の順番だけの話になる。実装の順番は実装計画で決める |
| 構造体の名前 `ActorCallInfo`・`MakeActionRequest`・`ActionInput`・`ActionOptions` | 不採用 | D6 の理由 |

### 4.2 起票時の未決表・骨子との対応

| 起票時 | 中身 | 行き先 |
|---|---|---|
| 未決表 1、骨子 1 | 接頭辞での要求 | 現状（§2.2）。093 で実装済み |
| 未決表 2 | Actor の要求の決め方 | D1・D2・D3・D4・D6。既定（購読ヒントの経路）は不採用（§4.1） |
| 未決表 3、骨子 3 | 切り出し・FP32 化・複製の場所 | D7 |
| 未決表 4 | `SelectAction` 系の引数 | D11 |
| 未決表 5 | ImageCls | D9 |
| 未決表 6、骨子 5 | 段階分け | 未決表から外した。実装の順番は実装計画で決める（§4.1） |
| 未決表 7 | 用語 | D5・D16 |
| 未決表 8 | ADR | D16 |
| 骨子 2 | `NetworkStruct` が書く | D5・D8 |
| 骨子 4 | 消すもの | D9・D11。起票時に「残す」とした `AppendTraceAux` の扱いは §6.3 |

## 5. 仕様

### 5.1 全体の流れ

```
読む側（Conv2d の pane、画像記録の Observer）
   │ step を進める側が、その時点の事情を見て決める（§5.4）
step を進める側（EvalPanel、学習のスレッド）
   │ DoStep / DoUpdateFrame の引数 MakeActionOptions
Runner
   │ MakeAction の引数 MakeActionOptions（そのまま渡す）
Actor
   │ nn_trace が真のときだけ、nn_trace/ を要求する capture を forward に渡す
NN の forward（NetworkStruct が capture に書く）
   │ forward のあとで取り出し、env 0 の切り出し・FP32 化・複製
AuxData の nn_trace/…  →  通知（Observer）  →  読む側
```

### 5.2 Actor ごとの扱い

次の表を上から順に当て、最初に条件を満たした行で扱う。

| # | 条件 | AuxData |
|---|---|---|
| 1 | `nn_trace` が偽（既定値を含む） | `nn_trace/` で始まる鍵を 1 つも入れない。capture なしの forward になり、切り出し・FP32 化・複製をしない |
| 2 | `nn_trace` が真で、Actor が `dqn::Actor`（Rainbow を含む）か `ImageClsActor` | 今と同じ鍵と値を入れる（G2） |
| 3 | `nn_trace` が真で、Actor が `dqn::Actor` でも `ImageClsActor` でもない | 何も入れない。その Actor が最初に受けたときに、WARN を 1 回出す（D12） |

この表は本番の Actor の決まりである。3 行目に当たる本番の Actor は、今は `MuZeroActor` だけである。テスト用の差し替え部品は対象にしない（受けた値の扱いは、それぞれのテストが決める）。

- `dqn::Actor` が nn_trace を取り出すのは、行動を選ぶ forward（今 callback を渡している forward）だけである。TempoRL の skip 側の forward からは取り出さない（今と同じ）。
- Learner が target の方策を呼ぶ `SelectAction` は capture を渡さない（今と同じ）。

### 5.3 Runner の step

- `Runner::DoStep` と `Runner::DoUpdateFrame` が、`MakeActionOptions` を既定値つきで受ける。
- `EvalRunner` の手動の行動を受ける `DoStep` にも、`MakeActionOptions` を渡せる形を足す。`EvalRunner::DoStep` はオーバーロードが 4 本あり、2 つ目の引数に空の波括弧を渡す既存の呼び出しがある（`trainer_test.cpp`）。全部の版に既定値つきで足すと、歩数を受ける版と区別できずコンパイルが通らなくなるので、どの版に足すかは、既存の呼び出しが曖昧にならない形で実装計画が決める。
- Runner は、受けた値をそのまま、その step の `MakeAction` へ渡す。`DoUpdateFrame` は、そのフレームの全 step に同じ値を渡す。
- `EvalRunner::RunSession` は受けない（D15）。
- `RunnerThread` は、作るときに `MakeActionOptions` を受け取り、step のたびに渡す（D14）。

### 5.4 step を進める側が渡す値

| 進める側 | `nn_trace` に渡す値 |
|---|---|
| EvalPanel（タイマー、手動の step、手動の行動の 3 経路） | 自分の Runner に付いた Conv2d の pane が 1 枚以上開いていれば真（D13）。step のたびに見る |
| 学習側（`RunnerThread`） | RunnerApp が、開始前に、`interval` が 1 以上の Conv2d の画像記録を登録していれば真。Run の間は変わらない（D14） |
| Q-Values の pane の予備の経路（行動の受け口が設定されていないときに、`EvalRunner::DoStep(action)` を直接呼ぶ） | 渡さない。本番では RunnerFrame が受け口を設定し、EvalPanel の手動の行動の経路を通るので、この経路は通らない |
| 背景評価 | 渡さない（D15） |
| テスト | 必要なテストが、呼び出しの引数で真を渡す |

Conv2d の pane を開いた直後は、次の `MakeAction` から nn_trace が付く。閉じた直後は、次の `MakeAction` から付かない。

### 5.5 capture の中の nn_trace

- 書き手は `NetworkStruct`。capture が `nn_trace/` の鍵を要求しているときだけ、branch の入力と、描画できる block の出力（条件は今と同じ。§2.1）を書く。要求が無いときは、鍵の文字列も作らない。
- 鍵は `nn_trace/<branch>/00_Input`、`nn_trace/<branch>/<2 桁の番号>_<block 名>`（D5・D8）。書く順は forward の順である。
- 要求の照合、要求していない鍵と二重の書き込みの即エラー、出口での満たされない要求の即エラーは、capture の今の決まりをそのまま使う（§2.2）。branch の入力は必ず書かれるので、branch が 1 つでも実行されれば `nn_trace/` の要求は満たされる。
- capture を渡さない forward、nn_trace を要求しない capture を渡した forward の結果と費用は変えない。

## 6. 実装契約

### 6.1 公開の形

- `rl::Actor::MakeAction(step, state, options)`: 3 つ目が `MakeActionOptions`。既定値は空（D2・D3）。
- `rl::Runner::DoStep(options)`、`rl::Runner::DoUpdateFrame(max_steps, pre_step_func, post_step_func, options)`: 既定値は空（D4）。
- `MakeActionOptions`: 欄は `nn_trace`（真か偽、既定は偽）。真偽の値なので、不正な値は無い。
- `Network::Forward`、`NetworkForwardExecutor::Forward`: callback の引数を消す。capture の引数は残す（D9・D10）。
- nn_trace の接頭辞の定数は NN 層に 1 つだけ置く（D5）。
- 新しい設定キー、メトリクス名は無い。

### 6.2 失敗の扱い

止める失敗は足さない。capture の今の即エラー（要求していない鍵、二重の書き込み、満たされない要求）がそのまま効く。nn_trace を作れない Actor への要求は、止めずに WARN を 1 回出す（D12）。WARN には、どの Actor かが分かる識別子を含める。

### 6.3 クラス一覧

従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| `MakeActionOptions` | 新規 | step を呼ぶ側が、その回の行動選択に添える入力。欄は `nn_trace` | rl |
| `rl::Actor` | 変更 | `MakeAction` が `MakeActionOptions` を受ける | rl |
| `rl::Runner` | 変更 | `DoStep`・`DoUpdateFrame` が `MakeActionOptions` を受ける | rl |
| `RunnerBase`、`EvalRunner`、`TrainRunner`、`SerialTrainRunner`、`PipelineTrainRunner` | 変更 | 受けた `MakeActionOptions` を `MakeAction` へ渡す | trainer |
| `RunnerThread` | 変更 | 作るときに受けた `MakeActionOptions` を、step のたびに渡す | trainer |
| `dqn::Actor` | 変更 | `nn_trace` が真のときだけ、capture で nn_trace を取り出して AuxData に積む | dqn_based_agent |
| `ActionPolicy`、`EpsilonGreedyActionPolicy`、`UQEActionPolicy`、`ThompsonSamplingActionPolicy` | 変更 | `SelectAction` とその補助が、callback の代わりに capture を受ける | dqn_based_agent |
| `ImageClsActor` | 変更 | `dqn::Actor` と同じ | image_cls_agent |
| `MuZeroActor` | 変更 | 引数を受ける。`nn_trace` が真なら WARN を 1 回出し、何も入れない | muzero_based_agent |
| `TraceCallback` | 削除 | — | common |
| `MakeActionTraceCallback` | 削除 | 切り出し・FP32 化・複製は、Actor が共通に使う関数へ移す | rl |
| `AppendTraceAux` | 変更または削除 | capture から取り出した nn_trace を、切り出し・FP32 化・複製して AuxData に積む関数にまとめる | rl |
| `ExtractNnTrace` | 変更 | NN 層の接頭辞の定数を使う。読み口は変えない | rl |
| `Network`、`NetworkForwardExecutor` | 変更 | callback の引数を消す。受けた capture を入口で Context に入れる | nn |
| `NetworkBody`、`NetworkBranch`、`NetworkStruct` | 変更 | callback の引数を消す。Body は capture を Context から受け取る。`NetworkStruct` は nn_trace の鍵を capture に書く | nn_impl |
| `NetworkForwardContext` | 変更 | Body が前回値の破棄と出口の検査に使えるよう、capture を渡す | nn_impl |
| `EvalPanel` | 変更 | step のたびに、Conv2d の pane が開いているかを見て `nn_trace` を渡す | apps/runner |
| `RunnerFrame` | 変更 | EvalPanel が、Conv2d の pane が開いているかを知る道を作る | apps/runner |
| `RunnerApp` | 変更 | 録画する（`interval` が 1 以上の）Conv2d の画像記録を登録したかどうかを、`RunnerThread` へ渡す | apps/runner |

### 6.4 移行対象

| 対象 | すること |
|---|---|
| [common.hpp](../../core/anet-core/include/anet/common.hpp) | `TraceCallback` を消す |
| [nn.hpp](../../core/anet-core/include/anet/nn.hpp)、[nn_impl.hpp](../../core/anet-core/src/nn_impl.hpp)、[nn_impl.cpp](../../core/anet-core/src/nn_impl.cpp) | callback の引数を消す（D9）。capture を入口で Context に入れる（D10）。`NetworkStruct` が nn_trace を capture に書く（D5・D8）。接頭辞の定数を置く |
| [rl.hpp](../../core/anet-core/include/anet/rl.hpp)、[rl.cpp](../../core/anet-core/src/rl.cpp) | `MakeActionOptions` を足す。`Actor::MakeAction`、`Runner::DoStep`・`DoUpdateFrame` の引数。`MakeActionTraceCallback` を消す。`kNnTracePrefix` を NN 層の定数に替える |
| [trainer.hpp](../../core/anet-core/include/anet/trainer.hpp)、[trainer.cpp](../../core/anet-core/src/trainer.cpp) | Runner の各クラスと `RunnerThread`（§5.3） |
| [dqn_based_agent.hpp](../../core/anet-core/src/dqn_based_agent.hpp)、[dqn_based_agent.cpp](../../core/anet-core/src/dqn_based_agent.cpp) | `dqn::Actor::MakeAction`、`SelectAction` 系の宣言 7 つとその呼び出し |
| [image_cls_agent.hpp](../../core/anet-core/include/anet/image_cls_agent.hpp)、[image_cls_agent.cpp](../../core/anet-core/src/image_cls_agent.cpp) | `ImageClsActor::MakeAction` |
| [muzero_based_agent.hpp](../../core/anet-core/src/muzero_based_agent.hpp)、[muzero_based_agent.cpp](../../core/anet-core/src/muzero_based_agent.cpp) | `MuZeroActor::MakeAction` の引数と、`nn_trace` を求められたときの WARN（D12） |
| [EvalPanel.cpp](../../apps/runner/src/EvalPanel.cpp)、[RunnerFrame.cpp](../../apps/runner/src/RunnerFrame.cpp)、[RunnerApp.cpp](../../apps/runner/src/RunnerApp.cpp) | §5.4 の値を渡す |
| テスト: [dqn_based_agent_test.cpp](../../core/anet-core/src/dqn_based_agent_test.cpp) | `[native_stack]` の 5 件は、`nn_trace` を真にして呼ぶ形に直す。`SelectAction` の差し替え部品 2 つと `MakeAction` の差し替え部品 1 つの引数を合わせる |
| テスト: [image_cls_agent_test.cpp](../../core/anet-core/src/image_cls_agent_test.cpp) | `[image_cls][trace]` の 1 件は、`nn_trace` を真にして呼ぶ形に直す |
| テスト: [episode_end_test.cpp](../../core/anet-core/src/episode_end_test.cpp)、[observers_test.cpp](../../core/anet-core/src/observers_test.cpp)、[trainer_test.cpp](../../core/anet-core/src/trainer_test.cpp) | `MakeAction` の差し替え部品と、Runner の差し替え部品（`TestRunner`・`ThrowingRunner`）の引数を合わせる |
| テスト: [nn_test.cpp](../../core/anet-core/src/nn_test.cpp) | `Network::Forward` に callback の関数を直接渡している呼び出しが 3 か所ある（`[nn][trace]` の「Network forward trace captures branch-prefixed visual layers」に 1 か所、`[nn][branch_capture][plasticity]` の「Network branch capture returns the actual forward feature without changing outputs」に 2 か所）。`nn_trace/` を要求する capture で書き直し、確かめている中身（鍵の頭に branch 名が付くこと、capture の有無で値が変わらないこと）を保つ |
| 設計文書 [130_neural_networks.jp.md](../design/130_neural_networks.jp.md) | `TraceCallback` を書いている 3 か所を直す。(1) forward の説明の「`TraceCallback` を渡すと中間 Tensor を収集できる」は、capture で `nn_trace/` を要求する記述にする。(2)「`GetTensorDictFunction` と TraceCallback が Network 内部へ到達する口」は、capture にする。(3) capture の節の「capture を渡さない forward は、TraceCallback と Actor の数値経路も維持する」は、`TraceCallback` への言及を外す。あわせて capture の節に、nn_trace の鍵と書き手（`NetworkStruct`）を足す |
| 設計文書 [110_agents_and_learning.jp.md](../design/110_agents_and_learning.jp.md) | Actor の契約とクラス図に、`MakeAction` の 3 つ目の入力と、nn_trace を要求した呼び出しだけ作ることを足す |
| 設計文書 [100_runtime_and_configuration.jp.md](../design/100_runtime_and_configuration.jp.md) | Runner の `DoStep`・`DoUpdateFrame` が `MakeActionOptions` を受け、そのまま Actor へ渡すことを足す |
| 設計文書 [140_observability.jp.md](../design/140_observability.jp.md) | §5「処理フロー」に、nn_trace の要求から可視化までのシーケンス図を 1 枚足す。本 PRD の §5.1 の流れ（step を進める側が `MakeActionOptions` を決める → Runner の step → Actor の `MakeAction` → capture つきの forward → Actor が取り出して AuxData へ → 通知 → 読む側）を、EvalPanel と Conv2d の pane の経路で描き、学習側（RunnerApp が決めて `RunnerThread` が渡す）の違いを図か注記で示す。§2 の Image/Graph Observer の行に、Conv2d は要求した step だけ作られる nn_trace を読むことを足す |
| 上の設計文書の `.en.md` | 同じ章を追随させる |
| [CONTEXT.md](../../CONTEXT.md) | D16。本 PRD の起草で直した |
| [ADR 0031](../adr/0031-plasticity-metrics-out-of-band-partial-forward.md)・[ADR 0055](../adr/0055-forward-capture-request-keyed-carries-module-diagnostics.md)（冒頭の注記）、[ADR 0072](../adr/0072-make-action-inputs-are-passed-per-call-through-runner-step-not-registered.md)（新規） | D16。本 PRD の起草で書いた |

過去の記録（`docs/memo/done/` の PRD、ADR 0031・0037・0055 の本文、実験記録）は書き換えない。設計文書の図にある `MakeAction(state)`・`DoStep()` のような略記は、既定値つきの引数を省いた呼び方として正しいままなので直さない。

検索に使った語は `TraceCallback`、`MakeActionTraceCallback`、`AppendTraceAux`、`ExtractNnTrace`、`kNnTracePrefix`、`nn_trace`、`MakeAction`、`DoStep`、`DoUpdateFrame`、`SelectAction` で、追跡しているファイルの全体を引いた。型名を書かずに関数を渡す呼び出しは `TraceCallback` では出ないので、`std::string_view` と `torch::Tensor` を引数に取る関数の形でも `core/`・`apps/` を引いた。当たるのは、`TraceCallback` の定義、`NetworkBranch::Execute`、`MakeActionTraceCallback`、`nn_test.cpp` の 3 か所だけである。

### 6.5 issue の対応表

| issue | 題 | area | 根拠の位置 | 確定事項 | 受入 |
|---|---|---|---|---|---|
| [issue-20261103-6a32b4](../management/issues/issue-20261103-6a32b4.md) | forward の名前付き出力を NetworkForwardCapture に一本化する | 315_nn、321_agent_core、318_rl_core、312_util | [nn_impl.cpp](../../core/anet-core/src/nn_impl.cpp) の `NetworkStruct::Forward`・`NetworkBody::Forward`・`RunForward`、[dqn_based_agent.cpp](../../core/anet-core/src/dqn_based_agent.cpp) の `Actor::MakeAction` | D1〜D18（2026-11-11 の追記は D10） | A1〜A13 |

## 7. 受入

- **A1（G2、D17）**: 変更前のビルドと変更後のビルドで、同じ seed・CPU で `MakeAction` を呼び、AuxData のうち `nn_trace/` で始まる鍵の全部について、鍵の並び（辞書順）・形・型・全要素の値を書き出す。2 つの書き出しが完全に一致する。除くものは無い。対象の構成は、`[native_stack]` の 5 件の構成、IQN の Head を持つ DefaultDQN の構成 1 つ（書き出した鍵に、形が 2 次元の `00_Input` を持つ branch が含まれることを確かめる）、ImageCls の `[image_cls][trace]` の構成。変更前の書き出しは、変更に入る前に同じテストで取る。
- **A2（G3）**: `dqn::Actor` と `ImageClsActor` のそれぞれで、`MakeAction(step, state)` と、`nn_trace` を偽にした呼び出しの AuxData に、`nn_trace/` で始まる鍵が 0 件であることを、テストが確かめる。
- **A3（G2）**: `[native_stack]` の 5 件と `[image_cls][trace]` の 1 件が、`nn_trace` を真にして呼ぶ形で通る。`nn_test.cpp` の 2 件（`[nn][trace]` と `[nn][branch_capture][plasticity]` の、§6.4 に挙げたテスト）が、capture で `nn_trace/` を要求する形で通り、鍵（`nn_trace/<branch>/…`）と値の確かめを保っている。
- **A4（D4）**: `EvalRunner`・`SerialTrainRunner`・`PipelineTrainRunner` のそれぞれで、`nn_trace` を真にした `DoStep` の通知の `action_info` に `nn_trace/` で始まる鍵があり、引数なしの `DoStep()` の通知には無いことを、テストが確かめる。`DoUpdateFrame` に真を渡して 3 step 進めると、3 回の通知の全部にある。`EvalRunner` の手動の行動を受ける `DoStep` でも同じ。
- **A5（D15）**: `EvalRunner::RunSession` の間の通知の `action_info` に、`nn_trace/` で始まる鍵が無いことを、テストが確かめる。
- **A6（D13、G3）**: RunnerFrame のテストで、次を確かめる。どの操作も、利用者の入力が通るイベントを送って行う。
  - pane を開く前に、Eval の Step のコマンドイベント（`ID_EvalStep`）を送って 1 回進める。その通知の `action_info` に `nn_trace/` で始まる鍵が無い。
  - メニューの Conv2d のコマンドイベント（`ID_Conv2d`）を送って pane を開き、`ID_EvalStep` のコマンドイベントで 1 回進める。その通知にはある。
  - pane を開いたまま、Q-Values の表のセルを左クリックするイベントを送って、手動の行動で 1 回進める。その通知にもある。
  - pane の閉じるイベント（wxAUI の pane を閉じるイベント）を送ってから、`ID_EvalStep` と、Q-Values の表のクリックで 1 回ずつ進める。どちらの通知にも無い。
  - タイマーの経路は、pane を開いた状態と閉じた状態のそれぞれで、EvalPanel のタイマーのイベントを 1 回送って進め、開いているときだけあることを確かめる。タイマーは利用者の操作ではないので、イベントを直接送るのは準備に当たる。再生中の動きそのものは、人が確かめる項で見る。
- **A7（D14）**: `interval` が 1 以上の Conv2d の画像記録を `metrics.image.*` に登録した構成で起動するテストで、学習側の通知の `action_info` に `nn_trace/` で始まる鍵がある。`interval = 0` の Conv2d の画像記録だけを登録した構成（出荷の DropMerge・ImageCls と同じ形）と、登録しない構成では無い。
- **A8（D12）**: テストが、作ったばかりの `MuZeroActor` 1 つに、次の順で 3 回 `MakeAction` を呼んで確かめる。(1) まず `nn_trace` を偽で呼ぶ。WARN のログは出ない。(2) 続けて真で呼ぶ。AuxData に `nn_trace/` で始まる鍵が無く、WARN のログが 1 件出る。(3) もう一度真で呼ぶ。AuxData に `nn_trace/` で始まる鍵が無く、WARN のログは増えない。
- **A9（G1）**: `git grep -n "TraceCallback"` と `git grep -n "MakeActionTraceCallback"` を `core/`・`apps/`・`CONTEXT.md`・`docs/design/` の範囲で引き、どちらも 0 件である。`docs/memo/`・`docs/adr/`・`docs/wiki/`・`docs/management/` は過去の記録と経緯を含むので、範囲に入れない。
- **A10（D10）**: [nn_impl.hpp](../../core/anet-core/src/nn_impl.hpp) の `NetworkBody` の `Forward`・`ForwardUpTo` の宣言と、[nn_impl.cpp](../../core/anet-core/src/nn_impl.cpp) の `RunForward` の宣言に、`NetworkForwardCapture` が現れない。
- **A11（093 を変えない）**: capture を渡さない forward の数値が変わらない。着手前に保存した実行体と実装後の実行体に同じ設定を渡し、[PRD103 の検証資材](../../core/anet-core/testdata/prd103/README.md)の 6 構成（seed 1 つ、決定的なモード、同じ短縮条件と比べ方）で、metrics checksum が前後で一致する。比べる対象から除くのは、検証資材が除いている性能と資源の群だけである（Run ごとに変わるため）。checkpoint のファイルの SHA は比べない（直列化が決定的でないため）。093 の attention 診断と可塑性メトリクスの既存のテストが通る。
- **A12**: 着手前に既定のテスト一式を流して結果を控え、実装後に流した結果と比べて、失敗するテストが増えていない。
- **A13（流れの図）**: [140_observability.jp.md](../design/140_observability.jp.md) の §5 に、nn_trace の要求の流れのシーケンス図がある。図の登場者に、step を進める側（EvalPanel）、Runner、Actor、NN の forward、`Notifier`、読む側（Conv2d の pane）があり、`MakeActionOptions` が step を進める側から Runner、Actor へ渡ること、capture で取り出した nn_trace が AuxData に載って通知で読む側へ届くこと、要求しない step では作らないことが、図から読める。

人が確かめる項:

- 実機の Runner で、Eval を再生している間（タイマーで進んでいる間）に Conv2d の pane を開くと画像が出て、閉じて開き直しても出る。pane を 2 枚開いて 1 枚閉じても、残りの pane に出続ける。一時停止して、Step のボタン、行動のキー、Q-Values の表のクリックで進めたときも、開いている pane に画像が出る。
- Trace の pane を開いて記録し、Conv2d の pane を開いていた間の行を選ぶと、Conv2d の控えが表示される。
- Conv2d の pane を閉じている間に記録した行の 1 行のダンプに、`action_info.aux.nn_trace/` の項目が無い（D1）。

速さの記録（D18）は合否に数えない。測った値は実装の報告に書く。

## 8. 決めること（未決表）

決めきれずに残した論点は無い。1〜3 は範囲外としたもので、2026-11-12 に扱いを決めた。4 は実装のあとで測って確かめるものである。

| # | 事項 | 扱い |
|---|---|---|
| 1 | Learner の購読ヒントの見直し。「この Run で使う診断の宣言（開始前。準備と開始前の検査に使う）」と「今回の呼び出しで足す要求（`UpdateFromBatchOptions`）」に役割を分ける案。購読の鍵が文字列で Agent と Learner の 2 層が解釈していること、間隔の規則を読む側と計算する側が別々に持つことが面倒さのもとで、渡す時機を変えるだけでは解けない。`UpdateFromBatch` は 1 回の呼び出しで学習を複数回走らせるので、呼ぶ側は学習回数ごとの要否を決められない | 範囲外（別の関心）。[issue-20261111-24d885](../management/issues/issue-20261111-24d885.md)（診断指標の群ごとの購読の重複）の対応内容へ追記した |
| 2 | PRD944 の Q7（`MakeAction` の引数の形）は、本 PRD の D2・D3・D4・D6 に合わせる。強制行動は `MakeActionOptions` の欄にし、`EvalRunner::DoStep` の手動の行動の引数をそれにまとめられる | 範囲外（944 の範囲）。944 の課題 [issue-20261103-aa5758](../management/issues/issue-20261103-aa5758.md) の対応内容へ追記した |
| 3 | 既定の forward と準備済みの forward を切り替える呼び先の表が、3 層を「null なら既定」で受け渡されている（課題 6a32b4 の 2026-11-11 の追記の一部） | 見送り（直さない）。本 PRD を実装すると内部の引数は 3 つと 5 つに減り、残るのは引数 1 つを 3 つの関数が順に渡すことだけで、必要性が無い（ユーザーの判断）。課題 6a32b4 の対応内容に書いた |
| 4 | nn_trace を要求した step の GPU メモリの増分（D7）。確かめ方: 実装のあと、ImageCls（`run.train.num_envs = 128` の出荷設定）の Runner で、Conv2d の pane を開く前と、開いて 100 step 以上進めたあとの GPU メモリの確保量を比べる。あわせて Atari の出荷設定でも比べる | 実装の報告に増分を書く。開発者が、行の指定つきの要求（§4.1）を課題にするかを決める。合否には数えない |

## 9. ほかの PRD との関係

- **093**（[done/093](done/093_attention_entropy_diagnostics_10prd.md)）: capture の今の決まり（鍵と接頭辞の要求、`GetAll`、実行位置の付与、出口の検査）が前提である。本 PRD は、公開の口が capture を引数で受ける形と、093 の診断を変えない。
- **944**（[944](944_observation_pipeline_10prd.md)）: 強制行動を `MakeAction` の入力にする。本 PRD が、入力の構造体（`MakeActionOptions`）、既定値、Runner の step での受け渡しを先に決めた。944 は同じ構造体に欄を足す形になる（§8 の 2）。どちらを先に実装してもよい。944 を先に実装する場合は、本 PRD の D2・D3・D4・D6 の形で `MakeActionOptions` を作り、本 PRD はそこへ `nn_trace` の欄を足す。
- **006**（[done/006](done/006_conv2d_visual_trace_10prd.md)）: §7 の 8 が先送りした「読む人がいるときだけ作る」を、本 PRD が step の引数の形で実現する。006 が挙げた、Actor 側に立てる印の形は採らない（§4.1）。
- **096**（[096](096_metrics_step_axis_interval_10prd.md)）: §8 の 1 の Learner の見直しをするとき、間隔の単位で関係する。

## 10. 参照

- [ADR 0031](../adr/0031-plasticity-metrics-out-of-band-partial-forward.md) の A''（nn_trace が常に作られる問題）。
- [ADR 0055](../adr/0055-forward-capture-request-keyed-carries-module-diagnostics.md)（forward の持ち帰りは capture 1 つ）。
- [ADR 0058](../adr/0058-approx-priority-column-computed-by-agent-step-diagnostics.md)（設定キーで eval にだけ作らせる形を退けた）。
- [ADR 0071](../adr/0071-warn-means-user-must-fix-and-unread-config-leaves-are-detected.md) と [AGENTS.md](../../AGENTS.md) §Fail-Fast 原則の表（WARN の線引き。D12）。
- [ADR 0072](../adr/0072-make-action-inputs-are-passed-per-call-through-runner-step-not-registered.md)（本 PRD で立てた）。
- [docs/design/130_neural_networks.jp.md](../design/130_neural_networks.jp.md) の capture の節。
