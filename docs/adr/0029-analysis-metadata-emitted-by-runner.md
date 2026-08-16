# 解析用メタデータは Runner が出力し、解析側は設定から再導出しない

Run 解析ツールは、metric の step 軸や source key を知るために `config/config_data.txt` の解決済み `metrics.scalar.[tag]` を再解析していた。この再導出には二つの欠陥がある。

一つは **step 軸だけでは metric の座標系が決まらない**こと。`EvalRunner::DoStep()` は `@train` 系 event へ自分の `step_counts_` を、`@episode_end` へ呼び出し元（train runner）から渡された `event_counts` を載せる。したがって `51_eval1/13_double_suika_created_mean` と `51_eval1/41_noop_uqe_win_rate` は、config 上どちらも `$eval.[eval1] ... $exp_step` と書かれていながら別の座標系に落ちる。実測で最大 step は 19,993,856 と 151,185、比は Run 中に 0.000039 から 0.0075 へ単調にドリフトするため定数倍換算も成立しない。config には「どの Runner の counts か」を表現する token が存在せず、`@event` と `$runner_scope` の組から間接的に導くしかない。

もう一つは **設定からの再導出が原理的に不健全**なこと。`AutoMerge()` は `.$` で終わるキーを新しい map へコピーしないため、どの profile が選ばれたかという情報は dump から消える。一方 CLI 由来の `.$` は AutoMerge 後の再適用で dump に残る。実 Run では config file が `app.$ = app.online > P`、dump の最終行が `app.$ = app.batchrun`、実効値は起動 bat の CLI 引数による `app.batchrun` 由来、という状態が実在した。config file を正本にしても dump を正本にしても、単独では逆の結論になる。

**Runner が、実際に構築された observer の解決済み定義を tag 単位で Metrics マスタへ出力する**。解析側はそれを正本として読み、設定からの導出は当該レコードを持たない過去 Run への互換 fallback に限定する。レコードは既存の `type: "json"` を使い、tag を `metrics.defs` として 1 レコードにまとめる。

決め手は、config が表現できない情報（どの Runner の counter か）を解析側が推論で埋めていた点にある。推論が当たっている間も、C++ の解決規則と Python の再実装が二重管理になり、仕様変更のたびに黙って乖離する。

## Considered Options

- **config に「どの Runner の counts か」を書けるよう token を追加する** — 却下。eval の step 数は train と 1:N であり、揃える手段が現状ない。設定で切り替える目処が立たないものに設定項目だけ作ると、選べない選択肢を契約へ固定することになる。
- **解析側で `@event` と `$runner_scope` から導出し続ける** — 却下。今日は正しく導出できるが、C++ の解決規則の写しを Python に持つことになる。`observers.cpp` の既定値が変わったときに検出する手段がない。
- **`config/<module>.txt` と同じく専用ファイル（`config/metrics_defs.txt`）へ出す** — 却下。Metrics マスタだけを取り出して解析する経路でメタが失われる。Metrics キャッシュ経由で読むにも別ファイルを開くことになる。
- **新しい record type（`type: "metric_def"`）を tag ごとに出す** — 却下。Metrics Viewer 側の語彙に新 type を足すことになり、`json/` へのミラーも自動では出ない。既存の `type: "json"` なら ingestor・SQLite schema・cache 契約が無変更で済む。
- **既存の `type: "json"` へ `metrics.defs` として集約する** — 採用。

## Consequences

- Metrics マスタが「何を記録したか」に加えて「どう記録すると決めたか」を持つ。マスタ単体で解析が閉じる。
- Metrics Viewer 側は無変更。レコードは `json_lines` テーブルへ入るため、解析ツールは Metrics キャッシュ経由でも読める。`json/metrics.defs.json` へのミラーも既存機構で自動的に出る。
- 過去 Run には `metrics.defs` が無い。互換対象は本決定より前に生成された Run artifact、移行方法は新しい Run を実行すること、削除条件は現用の Run 作業セットが全て `metrics.defs` を持つようになった時点とする。互換期間中は各 metric に `def_source` を立て、Run ごとに 1 回 warning を出す。
- 解析ツールは既存 Run artifact へ後から `metrics.defs` を書き込まない。artifact を変更しない契約を優先する。
- 「step 座標系」を CONTEXT.md の用語として追加する。step 軸名だけを同一性の判定に使わない。
- 実効設定の判定も同じ方針に従い、`config_data.txt` からの推測ではなく `config/<module>.txt` との突合で行う。突合できない領域は「実効でない」ではなく「不明」とする。
- 詳細設計は `docs/memo/054_inspect_run_10prd.md`。
