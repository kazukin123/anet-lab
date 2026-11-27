---
title: Run分析ユーザーガイド
page_kind: サマリ
material: "[[docs/design/030_user_guide_analysis.jp.md|030_user_guide_analysis.jp]]"
material_kind: 設計文書
concepts:
  - "[[docs/wiki/concepts/apps/120_metrics_viewer.md|Metrics Viewer]]"
  - "[[docs/wiki/concepts/measurement/metrics_resolution.md|Metrics Viewerの表示粒度]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/measurement/diagnostic_cost.md|計測の負荷と比較条件]]"
updated: 2026-11-26
---
# Run分析ユーザーガイド

## 元の材料

[Run分析ユーザーガイド](../../../design/030_user_guide_analysis.jp.md) blob:83909d78a5e5a8450059cc33a4912a6ad843c48d

## 要点

- Metrics Viewer は Run の記録を読み、同じタグを複数の Run で重ねて表示する。
- 比較前に環境、実効設定、seed、評価条件、step 軸を確認し、共通の経験数の範囲を選ぶ。
- 表示の解像度と平滑化、欠測、タグの記録間隔は曲線の見え方に影響する。
- Optuna の trial は seed 別の成績と集計に戻り、どの条件で score を得たかを確認する。
- inspect_run.py では取得できるタグや設定を調べ、必要な範囲の metric や個体の trace を抽出できる。

## 内容

### 2. Metrics Viewerを起動する

- 画面上部の`Workspace` selectorで、`runs/`または`config/`を持つworkspaceを切り替える。
- Viewerは、画面で開いていないworkspaceも起動直後と約10秒ごとに検査し、古い版・未作成のキャッシュを背景で準備する。選択Runを優先しながら背景処理も進め、workspaceを切り替えてもgzip変換の確定済み進捗を保持する。
- `apps/70_compress_workspace_metrics.bat`は、選択workspaceの`runs/`直下にある全Runの`metrics.jsonl`を`metrics.jsonl.gz`へ移行する。引数なしで起動すると`runs/`を持つworkspaceを名前順に列挙するので、番号を入力して選択する。`[0] EXIT`を選ぶとlauncherを終了する。
- workspaceを引数指定せずに起動した場合は、`YES`、`NO`、`DRY-RUN`の結果表示後にpauseし、キー入力後にworkspace選択へ戻る。`--no-pause`指定時はpauseを省略する。

### 3. 画面の基本操作

- Plotlyのmodebarではzoom、pan、画像保存、`Reset axes`を利用できる。`Autoscale`ボタンは重複を避けるため非表示である。グラフ本体のdouble-clickはPlotlyのaxis resetを維持しつつ、ViewerのReloadも実行する。
- `Mean`はbucket平均、`Band`はmin/max帯へ平均線を重ねる。点数が少なくL0を表示できる場合は、どのmodeでもraw折れ線になる。
- 同じgraphのRunは、表示しているStep範囲に対して同じ幅の区間で集約される。同じ表示範囲・画面幅なら、少なくとも4Runまで選択数によって粒度を下げない。zoomや画面幅変更では全Runが同じ粒度へ更新され、panでは区間の起点を維持する。記録が無い区間の値は補わない。
- 取り込み中でもRun一覧とcommit済みgraphは操作できる。Run-level errorまたはstep逆行で隔離されたtagは`⚠`とtooltipで示し、commit済み部分は引き続き表示する。

### 4. Run比較の手順

- `metrics.jsonl`のscalar recordは`type`、`tag`、`step`、`value`を持つが、`step`が`train_step`、`learn_step`、`episode_step`、`exp_step`などのどれかはrecord自体に保存しない。軸はRun内のmetrics設定で決まる。
- グラフの横軸を共通範囲にズームし、tag名から情報パネルを開く。`Viewing (開始-終了)` に出るstepの範囲を確かめ、各Runの `Count`、`Max`、`Mean`、`Min`、`Std`、`Last` を読む。
- IQN 探索診断 `metrics.scalar.@iqn_search_p0` の有無による負荷比較は同一binary・seed・実行条件で直列に行い、matched windowの`90_perf/90_elapse_hour`差分から算出したthroughputを比較する。他processやparallel Optuna jobがある測定は採用しない。

### 5. Optuna結果を分析する

- 1 trialのmulti-seed結果 — 使用する画面はOptuna Dashboard / artifact。正本は代表folderの`multiseed_summary.json`、`seed_runs.json`。
- DropMergeのtrial valueはseed別scoreのaggregateである。`score_aggregate`を確認し、`mean`、`median`、`mean-minus-std`、`min`を混同しない。
- late windowの`score_60_80`、`score_80_100`、`late_slope`は伸びや頭打ちを見る補助指標であり、trial valueそのものではない。最終候補はaggregate scoreだけでなく、seed別score、range/std、matched `exp_step`時系列を併せて確認する。
- `run-study --n-jobs > 1`は同一GPU上でrunnerを並列起動できるが、durationとstep/secは相互干渉を受ける。その条件のthroughputを単独Runと直接比較しない。

### 6. `inspect_run.py`でRunを検査・抽出する

- Metrics Viewerは人間向けの可視化画面である。shellから構造化結果を取り出したい場合、特にAIエージェントへRun分析を依頼する場合は`inspect_run.py`を使う。Run artifactを一切変更しないread-onlyのCLIで、実行中のRunへ当てても安全である。
- `source_key`は`metrics.scalar.[tag]`の右辺で採用されたmetric keyである。値の意味を確認したいときは、この文字列でコードを検索する。
- `inspect_run.py` は百分率や末尾からの範囲を `runner` と `step_axis` の組ごとに解決する。同じ評価スロットでも `@session_end` は学習側の座標、`@train` は評価 Runner 自身の座標を使う。
- metricは1回の呼び出しへ束ねる。 Metricsマスタの走査はRunごとに1 passで、tag数には比例しない。tagごとに呼び分けるとpassの回数だけ時間が増える。
- `metrics`はrangeごとに「行=tag、列=Run」の比較表を返す。cellの統計は既定で`mean`、`--stat last`などで切り替える。

### 7. 最小分析チェックリスト

- 比較対象の`config/config_data.txt`を保存した。
- 同じtag、同じstep軸、同じeval定義を確認した。
- 共通の`exp_step`windowを決めた。
- scoreとthroughputを分けて読んだ。
- EMA、interval、signed-logなど表示条件を揃えた。
- 1 seedの結果だけで最終判断していない。
- Optunaではaggregate方式、seed別分布、trial stateを確認した。

## 主要な概念

- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)
- [Metrics Viewerの表示粒度](../../concepts/measurement/metrics_resolution.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [計測の負荷と比較条件](../../concepts/measurement/diagnostic_cost.md)
