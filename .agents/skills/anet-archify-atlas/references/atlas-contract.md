# ANET Archify Atlas Contract

## 公開マニフェスト

`docs/archify/` には、Atlas の成果物として次の固定名だけを公開する。更新時は同じ名前を置換し、世代別コピーを作らない。

| 順序 | Archify type | basename | 役割 |
|---:|---|---|---|
| 1 | `architecture` | `anet_lab_00_system_architecture` | システム全体の構成、主要パス、依存、境界 |
| 2 | `workflow` | `anet_lab_10_run_workflow` | 実験を準備し、実行し、分析するワークフロー |
| 3 | `sequence` | `anet_lab_20_training_step_sequence` | 代表的な学習ステップの呼び出しと通知順序 |
| 4 | `dataflow` | `anet_lab_30_experiment_dataflow` | 設定、Tensor、経験、metrics、artifact の流れ |
| 5 | `lifecycle` | `anet_lab_40_runtime_lifecycle` | runtime の状態、休止、終了、失敗遷移 |

各 basename について `.archify.json` と `.html` を保存し、索引として `README.md` を保存する。visual-check の sidecar、スクリーンショット、contact sheet は公開しない。

## 共通表現契約

- 5図は同じ canonical term を使う。現在のコード識別子とドメイン用語が異なる場合は、カードで対応関係を説明する。
- authored content は日本語とし、型名、関数名、設定キー、プロトコル、製品名は原表記を保つ。日本語は Viewer UI の対応 locale ではないため `meta.locale` を設定せず、固定 Viewer UI が英語になることを index に明記する。
- 各図には、根拠が確認できる1本の主経路と3つ以上12個以下の主要要素を置く。システム構成図だけは8〜12コンポーネントを必須とする。
- 読み順を曖昧にする枝を避ける。因果、呼び出し、データ移動、状態遷移として意味のある関係だけをエッジにする。
- 責任、所有権、同期方法、例外、代替経路、設定差、実装上の注意はカードへ記述する。
- 1要素の evidence は最大3件とし、リポジトリ相対パスに必要最小限の symbol、target、section を添える。件数を増やす代わりに、宣言・実装・設計意図を代表する根拠を選ぶ。
- 現行 checkout で確認できない要素や関係を、以前の配置や既知の名称だけから補完しない。

## マップ別の調査課題

以下は意味上の調査課題であり、特定のファイル名やクラスが存在することを要求しない。現在の同等概念を証拠から発見し、存在しないものは省略または evidence gap とする。

### 1. System architecture

- 実行UI・CLI、run orchestration、Agent、Actor/Learner、Environment、NN、Replay、Metrics、Viewer・分析ツールなどから、現在の主要8〜12コンポーネントを選ぶ。
- 起動から runtime 構築、environment interaction、learning、metrics/artifact 消費までの主要パスを示す。
- libtorch、GUI framework、build/runtime toolchain、Viewer側runtimeなど、build declaration で確認できる外部依存をカードへまとめる。
- UI/process boundary、worker/thread boundary、CPU/GPU boundary、設定・入力の信頼境界、Run artifact と外部閲覧側のファイル境界を、現在の実装で確認できる範囲だけ示す。

### 2. Run workflow

- workspace または同等の実験単位を選択し、設定を解決して runtime を構築するまでを始点とする。
- start、training、定期処理、evaluation、pause/resume、shutdown、artifact analysis のうち、現行経路で実証できる主経路を結ぶ。
- 条件分岐や任意機能は主経路を分断しない。無効化時の意味、fallback、fail-fast はカードで説明する。

### 3. Training-step sequence

- 実行制御、Actor、Network、Environment、ReplayBuffer、Learner、observer/notifier、Metricsの現在の担当主体を特定する。
- state/spec取得、action生成、step、experience格納、sample、update、通知・記録の代表順序を描く。
- serial/pipeline、train/eval、同期/非同期など複数経路がある場合は、代表的な1本をシーケンスにし、差分をカードへ移す。
- 戻り値、worker例外の再送出、停止要求など、順序の理解に必要な応答だけを残す。

### 4. Experiment dataflow

- config/workspace から spec、state、action、experience、replay、minibatch、device、network/loss へ至る流れを確認する。
- event、scalar/image/video metrics、Run artifact、Viewer、外部分析bridgeなど、実在する生産者と消費者を示す。
- authoritative data と、再生成可能なcache/indexを区別する。永続化、圧縮、device転送、process境界はカードに記載する。
- データ形式やkey名は宣言またはreader/writer双方で確認し、片側しか確認できない契約は evidence gap とする。

### 5. Runtime lifecycle

- 設定解決、constructing、ready、running、paused、evaluating、shutting down、completed、failed に相当する実在状態を発見する。
- pauseが同期停止か新規処理の抑止か、evaluationが独立状態かrunning内イベントかなどは、現在の実装に合わせて表現する。
- worker/background exception の捕捉・保存・再送出、およびshutdown orderingを確認する。
- 実在しない状態を体裁のために追加しない。回復可能な失敗は実際の復帰遷移を持つ場合だけ描く。

## Evidenceと不一致

根拠の優先順位は次のとおりとする。

1. 現行コード、設定schema、build declaration、reader/writerなど、実行される契約
2. 現行の設計索引、用語集、ownership資料、ADR
3. テスト、サンプル、launcher、運用資料
4. 過去の実験記録や生成済み成果物

優先順位の異なる根拠が矛盾する場合、動作は上位の根拠に合わせる。ただし設計意図との差を消さず、`README.md` の doc/code drift に双方の根拠と相違を記録する。過去資料だけにある構造を現行構造として描かない。

## 品質と公開判定

- 全候補で `meta.quality_profile: "showcase"` を使用する。
- 各図は Archify showcase の9 artifact checks、composition errors 0、warnings 0を満たす。
- `deliver` 成功後の specification SHA-256 と artifact SHA-256 は、公開可否を判断する一時的な作業記録として確認する。公開コンテンツには追加しない。
- visual-check exit 0では画像を実際に確認する。目視手段がない場合または visual-check exit 2では作業結果を `skipped` として扱う。exit 1は公開失敗とする。個別の visual review status は公開コンテンツには追加しない。
- 5図のいずれかが最小要素、validation、delivery、visual containmentを満たさなければ、公開版全体を更新しない。
- 公開操作ではこのマニフェストの既存ファイルだけをバックアップ対象にする。失敗時は対象だけを復元し、`README.md`を更新しない。

## README.md 契約

Atlas index は次を簡潔に記録する。

1. Atlasの目的と5図の推奨閲覧順
2. 生成根拠となったGit revision
3. 各JSON/HTMLへの相対リンク
4. evidence gap
5. doc/code drift
6. authored contentは日本語、固定Viewer UIは英語であること

個別の検証結果、receipt、SHA-256、visual review status、時刻だけが変化する情報、一時ファイルのパスは記録しない。既知の不足がない場合も、未確認領域まで網羅したとは主張しない。
