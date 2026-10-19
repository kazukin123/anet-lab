# Evaluation Trace の列はメトリクスの定義と分けて番号付きで宣言し、遷移の時点に合う target だけを読む

EvalPanel の遷移を 1 件とする step trace の列は、`metrics.trace.*` の定義を tag で参照せず、`app.eval_trace_panel.columns.[<番号>] = $<target> <鍵>` の形でパネル側に 1 行 1 列で宣言する。記録の宣言が無くても見られ、GUI がメトリクスの配線に依存せず、表示の鍵である列番号をそのまま設定の鍵にできるためである。1 行の値がすべて同じ遷移に属するよう、読める target を、値がその遷移のどの時点に属するかが決まっているものに限り、学習スレッドで step と無関係に変わる `$agent` のように時点を持たない target は構築時に止める。

## Considered Options

- **metrics の trace 定義の tag を列挙して参照する**: 記録と表示の列が 1 か所になるが、見るために記録の宣言が要り、step trace の記録は PRD 081 で入るので本 PRD だけでは成り立たない。scalar の定義は EMA・間隔・集約を持つ統計で、step の生の値に合わない。却下。
- **すべての target を許し、時点の違いは文書に書く**: 書ける幅は広いが、終端の行の `$env lives` のような見た目で気づけない食い違いが残る。却下。
- **`$action_info_prev`・`_next` のような前後の target**: 行の単位を遷移にそろえれば要らない。却下。
