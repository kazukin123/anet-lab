# MetricsViewerのキャッシュ再構築を高速化し全ワークスペースで事前実行する

## 元の材料

[MetricsViewerのキャッシュ再構築を高速化し全ワークスペースで事前実行する](../../../memo/done/0090_metricsviewer_cache_rebuild_10rfd.md) blob:c3b7064a24bbe93cb124997b1b99271bf8cbff1e

## 要点

- キャッシュの版が変わるたび、利用者が workspace を開いてから大きな再構築が始まっていた。
- 未選択の workspace も事前に調べ、再構築の必要な Run をバックグラウンドで準備する。
- 処理開始時に Run、元の Metrics マスタ、キャッシュの現在状態を確認し、古い観測だけで書き換えない。
- 使用中の Run を優先しつつ、背景の workspace にも順番を回す。
- 検査・再構築理由・開始・完了を記録し、変化のない定常走査で同じログを繰り返さない。

## 内容

### 問題

MetricsViewerのキャッシュ版が上がると、利用者はキャッシュの作り直しに長時間待たされる。未選択のワークスペースはキャッシュ版の確認と取り込みの対象にならず、そのワークスペースを開くまで再構築の必要性を検出できない。必要になってから待つ運用を、未選択でも検出し、背景で準備する運用へ変える。
schema 2と保存LOD factor4への変更により、旧キャッシュはMetricsマスタから全再構築される。

### 目標と解決策

バッファ単位の入力と件数上限付きSQL batchで、Metricsマスタからの再構築時間を短縮する。
再構築キューは、未処理のRunと再構築理由をprocess内で管理し、次に処理するRunを選ぶためのものである。汎用ジョブ基盤や新しい永続ストアを作らない。process起動時にはキャッシュ自身を確認して必要な作業を再発見する。

### 確定事項と実装上の契約

Reader、JSON parse、L0書き込み、LOD生成・書き込み、commitの時間を分けて測定する。JSONのstreaming parseへの変更は、parseが主要費用と確認できた場合の追加候補とし、初期の必須変更に含めない。
対象は各workspaceのRun directory直下でMetricsマスタを持つRunとする。設定範囲外の退避Runや別のdirectoryを再帰探索しない。

### テストと受入条件

raw/gzipの同一マスタを現行方式と新方式で取り込み、L0、全levelのLOD、TagStats、非scalar行、tag隔離状態が一致する。浮動小数点の比較条件を明記する。
バッファ境界を跨ぐUTF-8・CRLF・長いJSON行・raw未終端末尾・gzip corruptで既存の完全行と失敗契約を保つ。

## 主要な概念

- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)
- [Metrics Viewerの表示粒度](../../concepts/measurement/metrics_resolution.md)
- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
