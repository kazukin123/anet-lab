# forward の名前付き出力を NetworkForwardCapture に一本化する 暫定 PRD

> 2026-10-16 093 の 3 回目のグリルで、接頭辞要求（末尾 `/`）・`GetAll`・NetworkStruct による実行位置の付与（`<branch>/<block インスタンス名>`）を 093 側へ先取りした。本 PRD の残りは trace の書き手と読み手の付け替えと `TraceCallback` の削除。§2・§4 の 1 と 2・§5・§8 の 1・§9 を合わせた。
> 2026-10-16 起票。暫定の既定を置き、決めることを §8 の表に残した。詳細はグリルで詰める前提。
> 起点: [093_attention_entropy_diagnostics_10prd.md](093_attention_entropy_diagnostics_10prd.md) のグリルで、`NetworkBranchCapture` を鍵要求型の `NetworkForwardCapture` に一般化した（[ADR 0055](../adr/0055-forward-capture-request-keyed-carries-module-diagnostics.md)）。これで forward の中の名前付き tensor を外へ出す機構が、capture と `TraceCallback` の 2 つになる。関心は 1 つなので、残る `TraceCallback` を capture に吸収する。
> 関連: [ADR 0031](../adr/0031-plasticity-metrics-out-of-band-partial-forward.md)（capture の契約と、TraceCallback の常時 ON 問題）、[ADR 0051](../adr/0051-nn-forward-prepares-concrete-calls-and-purpose-owned-execution.md)（Context）、[done/062_plasticity_metrics_10prd.md](done/062_plasticity_metrics_10prd.md) の 14（TraceCallback が capture に使えない理由）、[done/069_metrics_trace_channel_10prd.md](done/069_metrics_trace_channel_10prd.md) の D1・F9（用語「trace」との区別）。

## 1. 目的と問題

forward から名前付き tensor を外へ出す口を `NetworkForwardCapture` の 1 つにし、`TraceCallback` を無くす。

今は 2 つある。

- `TraceCallback`: push 型。forward 中に tensor ごとに関数を呼ぶ。要求の仕組みが無く、callback があれば可視化できる block 出力を全部出す。書き手は `NetworkStruct`、読み手は GUI の nn_trace。
- `NetworkForwardCapture`（093 で一般化）: pull 型。鍵の集合で要求し、終わってから読む。要求の無い鍵は計算しない。書き手は body と module、読み手は Learner の診断。

違いは運び方だけで、関心は同じである。TraceCallback に要求が無いのは、出すものが既に計算済みの block 出力で追加計算が無いからだが、Actor が `MakeAction` で毎回 callback を作るため、GUI が nn_trace を表示していなくても、可視化できる block ごとに env 0 の slice → FP32 → clone が毎 forward 走る（ADR 0031 の A''、062 PRD の 14）。要求フィルタが無いので、追加計算を伴う出力をここへ乗せることもできない。

## 2. 現状

- 定義: `using TraceCallback = std::function<void(std::string_view, const torch::Tensor&)>`（[common.hpp](../../core/anet-core/include/anet/common.hpp)）。
- 書き手: `NetworkStruct::Forward` が `00_Input` と、`IsConv2dVisualizable` で 4 階の浮動小数 tensor の block 出力を `{:02d}_{block 名}` で呼ぶ（番号は呼んだ出力だけを 1 から数えた通し番号で、block の位置ではない）。`NetworkBranch::Execute` が `branch 名/` を前置する（[nn_impl.cpp](../../core/anet-core/src/nn_impl.cpp) の `NetworkStruct::Forward`、`NetworkBranch::Execute`）。
- 読み手: `MakeActionTraceCallback` が env 0 の slice・detach・FP32・clone を `trace` に積み、`AppendTraceAux` が `nn_trace/` を前置して AuxData へ入れ、`ExtractNnTrace` が GUI で読む（[rl.cpp](../../core/anet-core/src/rl.cpp)、`Conv2dPanel.cpp`、`observers.cpp`）。
- 呼び出し: DefaultDQN の Actor `MakeAction` と ImageCls の行動選択が無条件に作る。policy の `SelectAction` 系 6 署名が callback を引数で運ぶ（[dqn_based_agent.hpp](../../core/anet-core/src/dqn_based_agent.hpp)）。
- capture: 093 で `NetworkForwardCapture` になり、`Forward`・実行器の `Forward`・`ForwardUpTo` が受ける。鍵 1 つに Tensor 1 つで、完全一致の鍵か末尾 `/` の接頭辞で要求し、`Get`・`GetAll`（接頭辞に一致した鍵と Tensor を書かれた順に）で読む。body（branch 名）と module（`型名/診断名`）が書き、module の鍵の後ろには NetworkStruct が実行位置 `<branch>/<block インスタンス名>` を付ける。未達の要求（完全一致が書かれない・接頭辞に 1 件も一致しない）は error。
- 購読ヒント（ADR 0031）は「TraceCallback の常時 ON 問題への将来流用」を想定して汎用に作られている。

## 3. ゴールと非ゴール

ゴール:

1. 名前付き出力の機構を capture 1 つにし、`TraceCallback` 型と forward 群の callback 引数を消す。
2. nn_trace の鍵と値は統合の前後で不変。
3. 要求駆動にし、GUI が nn_trace を表示していないときは trace の計算（slice・FP32・clone）が起きない。

非ゴール:

- GUI の表示内容、鍵名、trace の対象 block の変更。読み口 `ExtractNnTrace` は維持する。
- 093 の attention 診断の変更。
- 性能の向上（不変であればよい。段階 2 の OFF 化は副産物）。

## 4. 提案の骨子（暫定の既定）

1. 接頭辞要求: 093 で導入済み（末尾 `/` の要求鍵を接頭辞として一致させ、`GetAll` で書かれた順に読む）。trace の鍵は block 名で動的で列挙できないので、trace もこれで要求する。`nn_trace/` で旧 trace の全鍵を要求できる。
2. 書き手: `NetworkStruct::Forward` は callback の代わりに Context の capture を見て、trace の鍵が `Wants` なら `Add`。鍵名は現行と同一に保つ。`<branch>/` と block インスタンス名は 093 の実行位置と同じものを使い、`<NN>`（描画対象の出力だけを 1 から数えた番号で、block の位置ではない）は trace の鍵を作るときに NetworkStruct が自分で付ける。
3. 読み手: Actor は `MakeAction` で capture を作って forward へ渡し、終了後に結果を env 0 slice → FP32 → clone して `nn_trace/` で AuxData に積む。GUI 側は不変。
4. 削除: `TraceCallback` 型、`MakeActionTraceCallback`、`Network::Forward`・実行器 `Forward`・`NetworkBody`・`NetworkBranch`・`NetworkStruct` の callback 引数、policy `SelectAction` 系の callback 引数。`AppendTraceAux` と `ExtractNnTrace` は残す。
5. 段階分け: 段階 1 は機構の統合だけで、Actor は今と同じく常時要求する（挙動不変を先に確保）。段階 2 で要求をゲートする。ゲートの既定は、Runner が購読ヒントと同じ経路で「nn_trace を表示中か」を Agent へ渡す形。

## 5. 機構ごとの根拠

| 機構 | 根拠 | 裁定（暫定） |
|---|---|---|
| 接頭辞要求 | trace の鍵は block 名で動的。列挙できない | 093 で導入済み |
| `TraceCallback` の削除 | 1 関心 2 機構の解消が目的そのもの | 作る |
| 要求のゲート（段階 2） | 表示の有無によらず毎 forward に slice・FP32・clone が乗る。実測値は未取得なので、段階 2 の前に Actor 経路の実時間を測る | 段階 2 |

## 6. 受入基準

1. 同じ seed・決定論の設定で、統合の前後で AuxData の `nn_trace/*` の鍵集合と値が一致する。DefaultDQN と ImageCls の両方。
2. capture を渡さない forward（Learner の学習 forward、probe）の数値系列が不変（metrics の checksum）。
3. `TraceCallback` と `MakeActionTraceCallback` がコードから消える。
4. Actor 経路の実時間が、同じビルドの交互 A/B で不変。
5. 段階 2: GUI 非表示のとき、trace の tensor 演算がプロファイラに出ない。

## 7. 作る条件

次のどれかが起きたら着手する。

- trace に要求フィルタが要る（表示していないときの trace の費用を削りたい、または追加計算を伴う出力を trace に乗せたい）。
- 名前付き出力の 3 つ目の経路が要る。

前提: 093 の `NetworkForwardCapture` がコミット済みであること。

## 8. 決めること（暫定の既定）

| # | 決めること | 暫定の既定 | ほかの案 |
|---|---|---|---|
| 1 | 接頭辞要求の記法 | 決着（093 で導入）: 末尾 `/` の鍵を接頭辞とし、`GetAll` で書かれた順に読む | 棄却: 別メソッド `WantsPrefix`、glob |
| 2 | Actor の要求の決め方（段階 2） | Runner から「表示中」を購読ヒントと同じ経路で渡す | 常時要求のまま |
| 3 | env 0 slice・FP32・clone の場所 | Actor 側、forward の後 | capture に変換オプション |
| 4 | policy `SelectAction` 系の署名 | callback 引数を capture ポインタに置換 | 引数を無くし Actor が forward を包む |
| 5 | ImageCls | 同時に移行 | 後回し |
| 6 | 段階分け | 段階 1 統合（挙動不変）→ 段階 2 要求駆動 | 一括 |
| 7 | 用語 | CONTEXT.md の「capture」の _Avoid_ から trace（TraceCallback）の言及を外し、「trace」（metrics チャネル）の説明文の TraceCallback への言及を更新 | — |
| 8 | ADR | ADR 0055 に注記、新 ADR 1 本（trace も capture で運ぶ） | ADR 無し |

## 9. 依存と順序

- 093 の後。093 の capture（鍵集合と接頭辞の要求、`GetAll`、実行位置の付与）が前提で、本 PRD の差分は trace の書き手（NN 付きの鍵）と読み手の付け替え、`TraceCallback` の削除。
- 069 の用語（trace はメトリクスチャネル）との区別は維持する。`TraceCallback` が消えるので、_Avoid_ の記述だけ更新する。

## 10. 参照

- [ADR 0031](../adr/0031-plasticity-metrics-out-of-band-partial-forward.md) の A'' と Consequences（購読ヒントの将来流用）。
- [ADR 0055](../adr/0055-forward-capture-request-keyed-carries-module-diagnostics.md)。
- [docs/design/130_neural_networks.jp.md](../design/130_neural_networks.jp.md) の TraceCallback の記述。
- [093_attention_entropy_diagnostics_10prd.md](093_attention_entropy_diagnostics_10prd.md) の取り出し口の決定。
