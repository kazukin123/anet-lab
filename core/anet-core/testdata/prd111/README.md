# PRD111 タグセットの検証

対象は Metrics Viewer のタグセット。Run 選択・表示設定は保存しない。

## 自動検証

リポジトリルートで Maven の `-f apps/metrics-viewer/pom.xml` を使い、`TagSetStoreTest`、`TagSetApiIntegrationTest`、`TagSetPlaywrightTest` を実行する。その後に `mvn -f apps/metrics-viewer/pom.xml test` と `mvn -f apps/metrics-viewer/pom.xml package` を実行する。

- Store: 初期値と作業用の厳密な検証、初期作成、既存保持、保存バイト列、失敗時不変、外部編集の非再読込。
- API: CRUD、閉じたschema、Unicodeと名前の上限、判定順、404/409/500、同時更新。
- Browser: 実APIによる保存・復元、編集、不可視tag、通信失敗、複数コンテキスト、キーボード、タッチ、長い名前、初期復元。
- 既存の操作行テストの期待にTags操作行を1つ追加する。ほかの既存テストは変えずに全体を通す。

## 梱包

実装済みのrefを対象に Release workflow を `workflow_dispatch`（branch ref）で実行する。タグを作らず、GitHub Release は作成しない。artifactの本体zipに `apps/runner/workspaces/metrics_viewer_tag_sets.json` が無いことと、jarの `BOOT-INF/classes/initial_tag_sets.json` があることを確認する。

手元の確認は、テスト用ステージで作業用ファイルのあるWorkspace群をコピーし、workflowと同じ除外処理・zip作成・entry確認を実施する。元の作業用ファイルは保持する。jarはMaven packageの出力を使う。

workflowの未実行、Edge不在によるskip、その他の未実施は実装計画に理由とともに記録し、成功扱いにしない。
