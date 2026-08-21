# Envインスタンスname PRD

## 問題

Env実装は実行中に状態遷移、episode終端、異常状態などをtext logへ出力することがある。特に`DropMergeEnv`には複数のruntime logがあるが、現在のログだけでは、次のどのEnvインスタンスが出力元なのか判断できない。

- main Trainで使用しているEnvか
- configured Evalで使用しているEnvか
- EvalPanelで使用しているEnvか
- 複数laneのうち何番目のEnvか

`RunMode`は`Reset`と`Step`のTrain/Eval動作を選ぶ値であり、Envインスタンスの識別子ではない。同じ`RunMode`を複数のconfigured Evalが共有でき、EvalPanelもconfigured Evalとは別のEnvを所有するため、`RunMode`だけでは出力元を特定できない。また、thread-poolのworker番号は複数laneを担当し得るため、lane番号の代替にはならない。

一方、EnvへTrain/Eval、Runner、configured Eval tagなどの構造化コンテキストを渡すと、Envが上位の実行構造を認識し、その値で挙動を分岐する余地が生まれる。Envが本来扱う環境固有ロジックへ上位層の複雑さを持ち込みたくない。

Envが意味を理解する実行コンテキストは引き続き`RunMode`だけに保ちながら、人間が各Envインスタンスを識別できる仕組みが必要である。

## 目的

1. すべてのEnvインスタンスへ、生成時に人間向けの`name`を付与する。
2. main Train、configured Eval、EvalPanel、およびlane番号をtext log上で識別できるようにする。
3. `name`をログ専用情報に限定せず、Viewなどの人間向け表示からread-onlyで利用できるようにする。
4. Envが`name`の内容を解析せず、環境の挙動へ影響させない契約を確立する。
5. `RunMode`、configured Eval tag、Env class ID、seedなど、既存概念の意味を変更しない。
6. 設定項目を追加せず、既存の生成経路から固定規則でnameを決定する。
7. 同一Run内のBatchEnv nameを一意にし、ログ・View上の出力元識別がname衝突で失われないようにする。

## 前提

- 1回のRunでは複数種類のEnvを混在させない。
- Env種別の選択には既存のEnv class IDを引き続き使用する。
- Env class IDは同一Run内のEnvインスタンス識別には不要なため、nameへ含めない。
- main Train、各configured Eval、EvalPanelは、それぞれ別のBatchEnvを所有する。
- 現行single Env wrapperと、将来のbatch-native Envのどちらでも、nameの人間向け・挙動非依存という契約を維持する。

## 実装順序

本PRDを[`034_imagecls_batch_input_10prd.md`](034_imagecls_batch_input_10prd.md)より先に実装する。PRD 037は現行の旧top-level `BatchEnvFactory`とconfigured Evalのdirect `VectorizedDiscreteBatchEnv`生成経路へnameを通し、単独で全Envを識別可能にする。その完了状態をPRD 034のPhase 0開始時baselineとする。

実装順序は次で固定する。

1. PRD 037で現行生成seamへname契約を追加する。
2. PRD 034 Phase 0でname契約を維持したままBuilderへ改名する。
3. PRD 034 Phase 1以降でconfigured Evalとbatch-native Envを新seamへ移行する。

## 用語

### Env name

Env nameは、1つのRun内で人間がEnvを見分けるための表示名である。

- 人間向けの不透明な文字列とする。
- Envの生存期間中は不変とする。
- 同一Run内のBatchEnv nameは、大文字小文字を区別した完全一致で一意とする。
- process全体またはRunをまたぐ永続的な一意性は保証しない。別Runでは同じnameを再利用できる。
- ID、key、tag、構造化コンテキストとして扱わない。
- ログ、View、デバッグ表示など、人間向けの表示で利用できる。
- Envおよび利用側は、nameからTrain/Eval、Runner、laneなどの意味を解析しない。

### BatchEnvのnameとlane name

BatchEnvのnameは、BatchEnvを生成する上位層が`name`引数として与える人間向けの名前である。single Envをbatch化する場合、各single EnvのnameはBatchEnvのnameとゼロ始まりのlane indexから一度だけ生成する。BatchEnv生成時の`name`は結果としてlane nameのprefixになるが、`name_prefix`、`batch_name`、または別の構造体を公開APIへ導入しない。

例:

| 生成元 | BatchEnvのname | lane index | Env name |
|---|---|---:|---|
| main Train | `train` | 0 | `train[0]` |
| main Train | `train` | 37 | `train[37]` |
| configured Eval tag `eval1` | `eval1` | 0 | `eval1[0]` |
| configured Eval tag `test1`、`RunMode::Eval1` | `test1` | 0 | `test1[0]` |
| EvalPanel | `EvalPanel` | 0 | `EvalPanel[0]` |

configured Evalのnameは`RunMode`ではなく既存のconfigured Eval tagを材料にする。`RunMode`とtagが異なる場合も、両者を混同しない。

## 解決方針

### 1. 生成側がnameを決定する

Runを構築する上位層がBatchEnvのnameを決定し、BatchEnv生成呼び出しごとの`name`引数として渡す。

- main Trainは固定値`train`を使用する。
- configured Evalは既存のconfigured Eval tagを使用する。
- EvalPanelは固定値`EvalPanel`を使用し、Eval Runnerのnameまたはselected configured Eval tagを流用しない。
- testや単独ツールから直接生成する場合は、その呼び出し側が人間向けのnameを明示する。

PRD 037実装時点のconfigured Evalは旧top-level factoryを経由せず、`VectorizedDiscreteBatchEnv`を直接構築している。この現行経路にも一時的に`name=tag`を渡し、PRD 037だけを実装した状態でもconfigured Evalを識別可能にする。このdirect経路はPRD 034 Phase 1でBuilder経由へ置き換えるが、移行前後でnameを変更しない。

nameを設定ファイルへ追加しない。共有されるfactoryへsetterで保持させず、生成呼び出しごとのimmutableな値として渡す。これにより、Train、background Eval、EvalPanelの生成順や並行実行によるnameの取り違えを防ぐ。

### 2. RunManagerがRun内一意性を検証する

BatchEnv nameを決定する`RunManager`が、factoryまたはEnvを呼ぶ前に同一Run内のname衝突を検出する。比較はcase-sensitiveな文字列完全一致とし、nameの構造や意味は解析しない。

- `RunManager`初期化時に、固定名`train`、全configured Eval tag、固定名`EvalPanel`をrun-localなplanned name集合へ集め、最初のBatchEnvを構築する前に一括検証する。
- `train`と`EvalPanel`はconfigured Eval tagでは使用できない予約名とする。いずれかと一致するtagは`ANET_SYSTEM_ERROR`でfail-fastする。
- planned name集合は事前検証だけに使用し、Env生成成功済みnameのregistryとは分離する。
- `RunManager`はprivateなrun-local registryとして`name -> owner説明`を保持する。main Train、configured Eval、`CreateEvalRunner(name, ...)`の各経路は、Env構築前に既存nameを検査し、Envとrunnerの生成成功後に登録する。
- 生成に失敗したnameはregistryへ残さない。生成成功後のnameは`RunManager`破棄まで解放・再利用しない。
- `CreateEvalRunner(name, ...)`が既存nameと衝突した場合は第二のEnvを構築せず、既存`eval_runners` entryを上書きしない。
- 重複時はWARN、自動suffix、暗黙のrenameへfallbackせず、`ANET_SYSTEM_ERROR`で例外とする。

エラーには重複したname、既存owner、要求owner、同一Run内で一意でなければならないことを含める。メッセージ例は`Duplicate Env name 'eval1' within Run: existing_owner='configured Eval tag eval1', requested_owner='CreateEvalRunner eval1'. Env names must be unique within a Run.`とする。

このregistryはname決定責務を持つ`RunManager`だけが所有する。factory、Builder、Envへregistry、owner、予約状態を渡さず、それらは引き続きnameを無加工で伝播する。`RunManager`の既存thread-safety contractは変更せず、並行`CreateEvalRunner`対応は本PRDの対象外とする。

### 3. batch wrapperがlane nameを完成させる

single Envを複数生成するbatch wrapperは、既に保持しているゼロ始まりのlane indexを使い、`<name>[<lane index>]`形式の最終nameを作る。

- Vectorized方式とThreadPool方式で同じnameを生成する。
- thread IDやworker IDをnameへ使用しない。
- worker数やlaneの割当先を変更してもEnv nameを変えない。
- 最終nameはsingle Envの構築前に確定し、constructor内のログでも利用可能にする。

### 4. Envがnameを所有しread-onlyで公開する

single Envは生成時に完成済みnameを受け取り、生存期間中保持する。`SingleDiscreteEnv`は状態を持たないinterfaceとしてpure virtualな`GetName()`を公開し、`SingleDiscreteEnvBase`がnameを保持して`GetName()`を`final override`する。

`BatchEnv`は状態を持たないinterfaceとしてpure virtualな`GetName()`と`GetEnvName(lane_index)`を公開する。`BatchEnvBase`がBatchEnvのnameと全lane nameを構築時に保持し、両accessorを`final override`する。具象BatchEnvは`BatchEnvBase`を継承し、name accessorを独自実装しない。これにより、現在single Envを直接参照しないViewやevent consumerも、選択中laneのnameを人間向け表示へ利用できる。

APIはnameの取得だけを提供し、nameの構成要素、Runner種別、configured Eval tag、RunModeなどを公開しない。name変更用setterは設けない。

### 5. Envのtext logへnameを表示する

Env実装が出力する既存のactiveなtext logは、ログ本文の先頭へ`[<Env name>] `を付ける。

例:

- `[train[37]] Merged fruits into Rank ...`
- `[eval1[0]] Game Over: overflow timeout. ...`
- `[EvalPanel[0]] Fruit out of bounds. ...`

対象にはconstructor、`Reset`、`Step`、およびそれらから呼ばれるhelper内のログを含む。まず`DropMergeEnv`の全runtime logを確実に対応し、他の具象Envに存在するactiveなEnv本体ログも同じ契約へ揃える。View自身、Runner、Agent、汎用libraryが出力するログへEnv nameを自動付与する変更は行わない。

ログlevel、既存メッセージ本文、warning/errorのflush契約は変更しない。

### 6. nameをEnvの挙動から隔離する

nameは次の用途へ使用してはならない。

- `Reset`、`Step`、Reward、episode終端などEnv挙動の分岐
- Train/Eval判定または`RunMode`の推測
- config prefixや設定値の選択
- Env class IDの検索またはfactoryの選択
- Dataset、cache、resourceの選択
- seed生成、RNG domain、乱数列の決定
- metrics tag、Observer scope、JSONL field、artifact pathの生成
- 保存・読込、serialization key、checkpoint互換性判定

同じconfig、seed、Action列に対してnameだけを変えても、Observation、Reward、終端状態、乱数列は変化しないことを契約とする。

## ユーザーストーリー

1. Runを調査する利用者として、Envログからmain TrainかEvalかを判別したい。そうすれば、異常やepisode終端がどの実行経路で発生したかを確認できる。
2. 複数laneで学習する利用者として、ログを出したlane番号を知りたい。そうすれば、特定laneだけで起きる現象を追跡できる。
3. 複数のconfigured Evalを使う利用者として、同じ`RunMode`を共有するEval同士をtag由来のnameで区別したい。そうすれば、評価条件を取り違えずにログを読める。
4. EvalPanelを使う利用者として、手動Evalのログをbackground EvalやTrainから区別したい。そうすれば、GUI操作による現象を独立して確認できる。
5. Env Viewを実装する開発者として、表示対象laneの人間向けnameを共通APIから取得したい。そうすれば、RunnerやEnv実装の内部構造を再構成せず画面へ表示できる。
6. Envを実装する開発者として、上位のRunner構造を理解せず、自身のnameをログへ表示したい。そうすれば、Env固有ロジックをTrain/Eval構築の複雑さから分離できる。
7. frameworkを保守する開発者として、nameを不透明な文字列として伝播したい。そうすれば、新しい構造化コンテキストや暗黙のglobal stateを導入せずに目的を達成できる。
8. testや単独ツールを作る開発者として、Env生成時に明示したnameをそのまま取得したい。そうすれば、RunManager外でも診断表示の出力元を識別できる。
9. 再現性を検証する利用者として、表示名を変更しても乱数列やEnv結果が変化しないことを期待したい。そうすれば、診断上の命名と実験条件を分離できる。
10. 将来batch-native Envを追加する開発者として、nameを人間向け表示という小さな契約のまま引き継ぎたい。そうすれば、single Env wrapper固有のidentity設計をnative Envへ持ち込まずに済む。

## 実装上の決定

### 共通Env API

- `SingleDiscreteEnv`はconstructorと状態を持たないinterfaceとし、pure virtualな`GetName()`を提供する。`SingleDiscreteEnvBase`は構築引数としてnameを受け取り、immutableに所有し、`GetName()`を`final override`する。
- `BatchEnv`はconstructorと状態を持たないinterfaceとし、pure virtualな`GetName()`と`GetEnvName(lane_index)`を提供する。`BatchEnvBase`は構築時にnameと`num_envs`を受け取り、自身のimmutableなnameと`<name>[lane_index]`形式の全lane nameを構築・所有し、両accessorを`final override`する。
- 具象single Envは`SingleDiscreteEnvBase`、具象BatchEnvは`BatchEnvBase`を継承し、name accessorを独自実装しない。batch-native Envも`BatchEnvBase`へnameを渡すだけとする。
- nameは裸の文字列とし、context struct、role enum、key/value field集合にはしない。
- nameは必須の生成引数とし、構築後setterや暗黙の既定値で補わない。
- 空nameと`num_envs <= 0`は`ANET_CHECK_MSG`で常時fail-fastする。
- `GetEnvName(lane_index)`は`ANET_ASSERT`ではなく`ANET_CHECK_MSG`を使用し、負数または`num_envs`以上のlane indexを常時fail-fastする。不正なindexを別laneや空文字列へfallbackさせず、エラーにはname、index、num_envsを含める。

### 生成・伝播モジュール

- PRD 037実装時の旧top-level APIを`BatchEnvFactory::CreateBatchEnv(name, seed, num_envs)`とする。
- single Env生成APIを`SingleDiscreteEnvFactory::CreateSingleEnv(config_data, device, const std::string& name, optional<seed_t> seed = nullopt, const std::string& config_prefix = "")`とする。既存のdefault引数より前へ必須nameを置き、wrapperが完成済みlane nameを渡す。
- Run管理層はmain Train、configured Eval、EvalPanelそれぞれのBatchEnvのnameを決定する。
- `RunManager`は固定名とconfigured Eval tagをEnv生成前に一括検証し、生成成功済みnameをrun-local registryで管理する。
- BatchEnvを組み立てる上位factory/builder seamは、`name`を生成呼び出し単位で受け取る。
- factory、builder、Envは一意性registryを持たず、nameの検証・予約・owner管理を行わない。
- single Env wrapperはlane indexを追加し、完成済みnameをsingle Env factoryへ渡す。
- CartPole、GridMaze、DropMerge、LunarLander、ImageClsの5具象single Env factoryとEnv constructorはnameを伝播する。
- `BatchEnvBase`のconstructor追加は、`trainer_test.cpp`、`observers_test.cpp`、`episode_end_test.cpp`の`TestBatchEnv`と、`dqn_based_agent_test.cpp`の`JitterBatchEnv`へ明示nameを渡す変更を伴う。test doubleも`BatchEnvBase`を継承し、accessorをoverrideしない。

### PRD 034への移行義務

- PRD 034 Phase 0の`DefaultBatchEnvFactory`から`BatchEnvBuilder`への改名では、`CreateBatchEnv(name, seed, num_envs)`のname引数を落とさない。
- PRD 034 Phase 1でconfigured EvalをBuilder経由へ移行した後も、direct経路で使用した同じconfigured Eval tagを`name`として維持する。
- PRD 034で追加するper-class batch factoryにも必須`name`を渡し、batch-native Envへ共通`BatchEnv` name契約を適用する。
- PRD 034のBuilder改名、configured Eval移行、batch-native化の前後で、一意性検証とregistryの所有者を`RunManager`に維持する。
- `RunMode`、config prefix、configured Eval tagの設定上の役割と`name`を別契約として運び、Envはnameの意味を解析しない。

### Env class IDとの分離

- 1 RunにEnv種が1つという前提に基づき、Env class IDをnameへ含めない。
- Env class IDはrepository lookupとView種別選択に引き続き使用し、本PRDでは削除・改名しない。
- nameからEnv class IDを推測しない。

### Viewとの境界

- Viewがnameを取得できる共通read-only経路までは本PRDの対象とする。
- 既存Viewのタイトル、pane、レイアウトへnameを実際に表示するUI変更は本PRDの対象外とする。
- View用dataへnameを載せる場合も、人間向け文字列のまま扱い、Tensor metadataやEnvSpecの機械可読情報へ混在させない。

### lifetime・thread safety・性能

- nameはEnv構築時に一度だけ生成・copyまたはmoveし、実行中に変更しない。
- immutableなnameの参照はVectorized方式とThreadPool方式の双方で共有mutable stateを増やさない。
- EnvのStepごとにnameを再構築しない。
- ログが無効な経路では、name付与を理由に追加の文字列整形や同期を行わない。
- ログ出力時のprefix追加以外に、Env Stepの計算量、device転送、thread割当を変更しない。

## テスト方針

テストはprivate memberや文字列生成helperの実装ではなく、公開APIと観測可能なログ・Env結果を検証する。

### 共通Env生成テスト

- B=1でBatchEnvのnameから`name[0]`が生成されることを検証する。
- 複数laneで`name[0]`から`name[N-1]`まで重複なく生成されることを検証する。
- Vectorized方式とThreadPool方式で同じlane nameが得られることを検証する。
- ThreadPoolのworker数を変えてもlane nameが変化しないことを検証する。
- BatchEnvのread-only APIから各laneのnameを取得できることを検証する。
- 空name、`num_envs <= 0`、範囲外lane indexが、Release buildでも有効な`ANET_CHECK_MSG`で常時fail-fastし、暗黙に別nameへfallbackしないことを検証する。

### Run生成経路テスト

- main Train Envが`train[0..N-1]`になることを検証する。
- configured Eval Envがconfigured Eval tag由来の`<tag>[0]`になることを検証する。
- configured Eval tagと`RunMode`を意図的に異ならせ、nameが`RunMode`から生成されないことを検証する。
- 異なるconfigured Eval tagが同じ`RunMode`を使用しても、nameで区別できることを検証する。
- EvalPanel用Envが固定`EvalPanel[0]`を持ち、selected configured Eval tagまたは`RunMode`を変更してもnameが変わらないことを検証する。

### Run内一意性テスト

- configured Eval tagが`train`または`EvalPanel`の場合、最初のBatchEnv構築前に`ANET_SYSTEM_ERROR`となることを検証する。
- 相異なるconfigured Eval tagは正常に生成され、各nameとlane nameを識別できることを検証する。
- main Train、configured Eval、EvalPanelと同名の`CreateEvalRunner(name, ...)`はEnv構築前に失敗し、既存runnerを上書きしないことを検証する。
- 同じ動的nameで`CreateEvalRunner(name, ...)`を2回呼ぶと2回目が失敗し、1回目のrunnerが維持されることを検証する。
- Envまたはrunner生成失敗後にnameがregistryへ残らず、成功済みnameはRun終了まで再利用できないことを検証する。
- 別の`RunManager`では同じnameを使用できることを検証する。
- `train`と`Train`を別nameとして扱うcase-sensitive比較を検証する。

### Env挙動とログのテスト

- 同じconfig、seed、Action列でnameだけを変えた2つのEnvが同じObservation、Reward、terminated、truncatedを返すことを検証する。
- constructor内ログで、完成済みnameが利用できることを検証する。
- `DropMergeEnv`の代表的なVerbose、Info、Errorログが完全なEnv nameを1回だけ含み、既存本文とlevelを維持することを検証する。
- Env nameをmetrics tag、seed、artifact pathへ使用しないことを既存の再現性・metrics回帰テストで確認する。
- 既存の具象Env testとcore testを実行し、必須name引数追加による生成経路の漏れがないことを確認する。

## 受入条件

1. main Trainの各single Envが`train[0..N-1]`というnameを持つ。
2. configured Evalのsingle Envが`RunMode`ではなくconfigured Eval tag由来のnameを持つ。
3. EvalPanel用single Envが固定値`EvalPanel`由来のnameを持ち、selected configured Eval tagまたは`RunMode`に依存しない。
4. lane indexはゼロ始まりで、Vectorized方式とThreadPool方式のどちらでも同じである。
5. Env class IDをnameへ含めない。
6. nameはEnv構築前に確定し、constructor内ログで利用でき、生存期間中に変更されない。
7. single EnvとBatchEnvの公開APIからnameをread-onlyで取得できる。
8. Env本体の既存active text logが`[<Env name>] `prefixを持ち、少なくとも`DropMergeEnv`の全runtime logで出力元を識別できる。
9. nameを解析するproduction code、nameでEnv挙動を分岐するproduction code、factoryのmutable name stateが存在しない。
10. nameの変更がEnv結果、seed、RNG列、metrics tag、artifact path、保存形式へ影響しない。
11. name用の設定キー、template、構造化context、thread-local logging contextを追加しない。
12. Viewは共通Env APIから表示対象laneのnameを取得できるが、本PRDだけを理由に既存UIレイアウトを変更しない。
13. main Train、configured Eval、同一`RunMode`の複数Eval、EvalPanel、B=1、複数lane、Vectorized方式、ThreadPool方式の自動テストが通る。
14. Env、実行基盤、可観測性に関する設計文書が新しいname契約と一致する。
15. PRD 037単独の現行direct configured Eval経路で`name=tag`が渡り、PRD 034のBuilder移行後も同じnameを維持する。
16. `SingleDiscreteEnv`と`BatchEnv`は状態を持たないinterfaceであり、name accessorをpure virtualで公開する。`SingleDiscreteEnvBase`と`BatchEnvBase`がaccessorを`final override`し、`GetName()`は構築時name、全`0 <= lane_index < num_envs`の`GetEnvName(lane_index)`は`<name>[lane_index]`を返す。batch-native Envは`BatchEnvBase`を使用し、範囲外はfail-fastする。
17. 同一Run内のBatchEnv nameはcase-sensitiveな完全一致で一意であり、uniqueなBatchEnv nameから生成される全lane nameで出力元を識別できる。
18. 固定名`train`、全configured Eval tag、固定名`EvalPanel`の衝突は、最初のBatchEnv構築前に`ANET_SYSTEM_ERROR`となる。
19. 動的な`CreateEvalRunner(name, ...)`のname衝突は第二のEnv構築前に`ANET_SYSTEM_ERROR`となり、既存runnerを上書きしない。
20. name registryは`RunManager`だけが所有し、生成失敗したnameを残さず、生成成功済みnameをRun終了まで保持する。別Runでは同じnameを再利用できる。

## 対象外

- EnvへTrain/Eval role、Runner情報、configured Eval tagなどの構造化コンテキストを渡すこと
- structured logging、JSON logging、thread-local logging context、logger全体の再設計
- nameの書式を設定、template、コマンドラインで変更できる機能
- 重複nameのWARN継続、自動suffix、暗黙のrename
- process-global name registryまたはRunをまたぐname一意性
- nameの解析、検索用field化、filter・集計機能
- process-global ID、distributed worker ID、永続IDの導入
- 1 Run内で複数Env classを混在させる機能
- Env class ID、configured Eval tag、Runner name、`RunMode`の既存意味の変更
- metrics JSONL schema、metrics tag、Observer scope、artifact pathの変更
- seed domain、再現性contract、保存・読込形式の変更
- 既存Viewへnameを表示する具体的なUI変更
- batch-native EnvまたはBatchEnv Builder再編そのものの実装。ただし、それらが維持すべきname契約と移行義務は本PRDで規定する。

## 実装時のドキュメント更新

- 用語集へEnv nameの人間向け・不透明・挙動非依存という定義を追加する。
- Env設計文書へnameの生成、single/batch伝播、lane、lifetime、View accessorの契約を追加する。
- 実行基盤設計文書へmain Train、configured Eval、EvalPanelのBatchEnv name決定境界、reserved name preflight、run-local registry、重複時のsystem error contractを追加する。
- 可観測性設計文書へEnv text logのname prefixと、metrics tag・Observer scopeとは別概念であることを追加する。
- 将来のbatch-native Env seamを記録した既存ADRの元の決定は履歴として維持し、PRD 037先行で追加されるname伝播とRun内一意性をfollow-upとして追記する。

## 補足

本PRDのnameは「将来機械処理するためのidentity」ではない。将来、ログのfield検索、distributed実行、永続的なsource identityなどが必要になった場合は、Env nameを解析して流用せず、Envの外側に別の明示的な仕組みを設計する。

この制約により、現在必要な人間向け識別を単一文字列で実現しつつ、Envが上位の実行コンテキストを意識しない設計を維持する。
