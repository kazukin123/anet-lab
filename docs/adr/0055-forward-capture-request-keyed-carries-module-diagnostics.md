# forward の持ち帰りは鍵要求型の capture 1 つに統一し、module の診断も同じ器で運ぶ

attention エントロピー（PRD 093）は Body の state に存在せず TransformerEncoder の中でしか計算できない値で、SDPA は重みを返さない（[ADR 0004](0004-sdpa-attention-via-aten.md)）ため、測定のときだけ明示計算した値を forward の外へ出す口が要った。branch 出力を body が末尾で detach する専用構造体 `NetworkBranchCapture`（[ADR 0031](0031-plasticity-metrics-out-of-band-partial-forward.md)）を、呼び出し側が鍵の集合で要求し、body（branch 名）と module（`型名/診断名` の公開定数）が forward 中に書き込む `NetworkForwardCapture` へ一般化し、`Network::Forward`・実行器の `Forward`・`ForwardUpTo` が同じ型を受ける。内部は `NetworkForwardContext` のポインタ 1 本で module まで届け、渡さない forward は null 判定 1 回で不変、要求した鍵が書かれなければ出口で fail-fast する。汎用の forward に固有目的の引数を付けないことと、forward の持ち帰りの概念を 1 つに保つことを、編集量の少なさより優先した。

## Considered Options

- **`ForwardUpTo` に attention 専用の受け皿引数（当初案）**: 汎用 API に固有目的の引数が付く。棄却。
- **`TraceCallback` に乗せる**: 要求フィルタが無く Actor が `MakeAction` で常時渡すため、購読ゼロでも毎 forward に計算が乗る（ADR 0031 の A''）。棄却。trace を capture へ吸収する統合は `docs/memo/999_nn_trace_capture_unification_10prd.md` でゲート付きに先送り。
- **NN 実行状態（[ADR 0051](0051-nn-forward-prepares-concrete-calls-and-purpose-owned-execution.md) の State）のスロットへ書く**: State は呼び出し側から読めない契約で、`ForwardUpTo` は State を持たない既定経路。棄却。
- **Network に専用メソッド**: 公開面に module 固有の診断名が漏れ、結局同じ配線が要る。棄却。
- **別型の汎用診断 capture を足し `NetworkBranchCapture` は残す**: 編集は最小だが 1 関心 2 機構。棄却。

## Consequences

- branch 名に `/` を含められない（module の鍵との名前空間を構造で分ける。構築時に reject）。
- 可塑性の actual チャネルは `Get(final_feature_branch_key)` で読む。動作は不変。
- 未知 branch 名の fail-fast は「未達の要求は error」の特例になり、Transformer の無い閉包で attention を要求した場合も同じ規則で止まる。
- 2 つ目の module 診断は、鍵定数と module 内の `Wants`／`Add` だけで足せる。
- 仕様の正本は `docs/memo/093_attention_entropy_diagnostics_10prd.md` §4。
