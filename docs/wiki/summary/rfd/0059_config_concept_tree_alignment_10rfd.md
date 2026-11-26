# 設定体系再設計: 概念ツリー整合・素材宣言・ConfigResolver PRD

## 元の材料

[設定体系再設計: 概念ツリー整合・素材宣言・ConfigResolver PRD](../../../memo/done/0059_config_concept_tree_alignment_10rfd.md) blob:07c191239b258b905c527f393ccb622e5793e78a

## 要点

- 設定をカタログ、選択・上書きする素材、Run 署名を表す幹の3層で構成する。
- IQN など複数箇所の配線が必要な機能を一つの素材から選び、片側だけの設定を減らす。
- 直接上書きで契約に違反した場合は、該当キーの近くで失敗させる。
- config_data.txt に実効値を、別の解決記録にどの経路で選ばれたかを残す。
- 同じ Run 署名で seed を変えた群を、再現と比較の単位として識別できるようにする。

## 内容

### 目標と非目標

標準の素材選択経路では、概念を構成する一部だけを選べないようにする。 例: IQN を選んだら agent 側スイッチと NN 配線は 1 つの選択で同時に決まり、片方だけを選ぶ書き方が標準経路に存在しない。
直接 override による契約違反は、発生点に近い早い段階で fail-fast する。 原因の config キーを名指しするエラーにする(現状は NetworkModel 構築まで潜伏し、shape エラーで間接検出される。付録 A)。
Run artifact から「実効値」と「選択経路」を両方読めるようにする。 実効値は純化した `config_data.txt`、選択経路は新設の `json/config_resolution.json` と Metrics master の `config_resolution` record が持つ。
Run を「幹」(= Run 署名)の単位で定義・切替・記録できるようにする。 幹はスロット選択の束であり、同一幹×複数 seed = 比較母集団という運用を構造で表現する。

### 概念モデル: 3 層と幹

頂点は `run.`。workspace は設定の外(Run フォルダの移動・削除で管理。ADR 0021 の分類責務はフォルダ側に残る)。
幹は選択のみを書く(スロット選択行と、宣言済み値スロットのみ)。値スロットは `E.game` 相当・`train.num_envs` のような「選択肢に構造がなく、値自体が署名の一部」のもの。
env 枠・agent 枠は「class_id(実効 leaf)+具象 namespace のチェーン」の 2 行構造とし、畳む機構は作らない。「class_id の値=具象 config prefix」を正式契約に格上げする(現状は暗黙の規約)。
現行 config に実在する「3 箇所を手で同時に編集する契約」(NatureDQN 再現のコメント運用)が、named 幹 1 行に置き換わるのが代表ユースケース。

### 軸カタログと所有権

プリセット化の価値は「値が複数ある」ことではなく、1 つの選択が複数箇所を同時に変えるかで判定する。
rename 対象は「最終ツリーへの直書き行」のみ。実測: `net.$` 7 行+`net.branch.[slot]` 63 行+`net.body.output/$` 37 行+bat の CLI override 5 行=112 行(コメントラダー込み・10 ファイル・機械置換可)。optuna 生成 config は net を触っておらず影響ゼロ。
コード変更: `NetworkConfig` の構築 prefix(既定 "net" → Agent config prefix 連結)と関連テスト。

### 構文契約

既存 config のチェーンは全項「それ以外」に該当するため、後方互換は完全。
同一の実効キーへの複数書き込みは、すべて種別の全順序で決定的に解決する。fail-fast も WARN も発しない。

### ConfigResolver アーキテクチャ

`ConfigManager` の既存 interface(構築 → `GetConfigData()`)は不変のまま、内部を deep module `ConfigResolver` として再構成する。現行 `AutoMerge()`(約 60 行、付録 A)の拡張・肥大化はしない。
第 1 相(フェーズ 2 の前): 全 CLI override を source map へ先出しする。selection(`.$`)・素材キー(`@` を含むキー)に加え、選択の源プレフィクス形(`app.batchrun.exp_exit_step` 等)もここで選択の複製に乗って効く。bat の `app.$=app.batchrun` はここ。
第 2 相(フェーズ 3): selection・素材キーを除くキーへの実効 leaf 上書き。選択・幹の産物すべてに最後に勝つ — 「CLI 最強」の保証はこの相が担う(第 1 相で source map に入れた実効 leaf キーが選択の複製に上書きされても、第 2 相で CLI 値が最終適用される)。
CLI で与えた `.$` が解決後の dump に残る旧挙動の廃止(選択の記録は resolution.json が担う)は維持する — 第 2 相の篩は不変。

### Run artifact 契約

resolution payloadは`data`配下に置く。`json/config_resolution.json`にはtimestampを含めず、Metrics master recordには既存JSON経路がtimestampを追加する。
`selections` はselection適用1回につき1 entry。同一の実効`.$` keyがチェーン内の複数素材から生成された場合は、右勝ち履歴を表すため同じ`key`のentryが複数並ぶ。`key` はドット正規形の完全修飾。`chain` は左から右への適用順で、`term` = 記述どおりの項、`resolved` = 解決先のドット正規形絶対 prefix(相対 `@name` の解決結果がここで読める)。
配列順は解決順(決定的)。詳細モードのフィールド追加時は `schema_version` を上げる。
`--config` での再読込対象は従来どおり `config_data.txt` のみ。resolution.json は分析・診断専用で、読み戻しには使わない。

### optuna / batch 運用契約

trial が `.$` 選択を振る用途は限定する: HEAD / BODY 候補=同一 objective で可。ALGO 切替=metric・loss・score の意味が変わるため原則別 study。RUN_BUDGET 切替=score window と計算予算が変わるため同一 study 内は禁止。
「選択された素材名」を通常の trial parameter として保存し、実効 leaf 値は `config_data.txt` から読む。
resume 可能な構成・方針は素材にしてよい。checkpoint path は Run instance 固有値であり、run-local の leaf override(無印)で与える。素材(@)に具体パスを入れて恒久カタログ化しない。

## 主要な概念

- [316 設定管理](../../concepts/framework/316_config.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
