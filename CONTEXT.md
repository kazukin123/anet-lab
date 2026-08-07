# anet-lab

anet-lab は libtorch を基盤とした強化学習実験プロジェクトです。
この文書は、実装仕様ではなく、プロジェクト内で使う強化学習ドメイン用語の意味を揃えるための用語集です。

## Language

### DQN・価値表現

**DQN系エージェント**:
行動価値を推定する Q ネットワークを中心に学習する Agent 群。DefaultDQN、QR-DQN、Rainbow のような DQN 派生手法を指す。
_Avoid_: Q Agent, value agent

**Train Actor network snapshot**:
`DefaultDQNAgent`のTrain Actorがaction forwardに使用する、Learner online networkから複製されたparameterとbufferの時点コピー。snapshot間で固定されるのはnetworkだけで、ActionPolicy、ObservationNormalizer、RNGは含まない。
_Avoid_: policy snapshot, frozen policy, 方策固定

**価値ストリーム**:
Dueling DQN 系の Q ネットワークで、状態価値 V を推定するための特徴表現と最終射影の流れ。
_Avoid_: V branch, value branch

**アドバンテージストリーム**:
Dueling DQN 系の Q ネットワークで、行動ごとの優位性 A を推定するための特徴表現と最終射影の流れ。
_Avoid_: A branch, advantage branch

**Bellmanターゲット**:
報酬と次状態のブートストラップ価値から作る、現在の Q 値が近づくべき教師値。
_Avoid_: target Q, TD target

**target return**:
実報酬の N-step 割引和。ReplayBuffer に保持される値で、bootstrap 価値は含まない（学習時に次状態のブートストラップ価値が加算されて Bellmanターゲットになる）。
_Avoid_: n-step reward, stored return

**Q空間**:
学習器が Q 値として扱う値の表現空間。TBO 有効時は圧縮済みの h 空間を、TBO 無効時は実空間を指す。
_Avoid_: value scale, output scale

**実空間Q値**:
環境報酬のスケールで解釈できる未変換の Q 値。TBO 有効時は h 空間 Q 値を逆変換して得る。
_Avoid_: raw Q, untransformed Q

**h空間Q値**:
Transformed Bellman Operator の変換関数 h によって圧縮された Q 値。TBO 有効時にネットワークが直接出力し、損失計算で扱う値。
_Avoid_: transformed Q, compressed Q

**報酬スケーラ**:
ReplayBuffer に保存される報酬のスケールを調整する構成要素。TBO とは別に、報酬そのものの大きさを変える。
_Avoid_: reward normalizer, reward transform

**Transformed Bellman Operator**:
Bellman ターゲットを可逆な変換関数 h で圧縮し、大きな Q 値に対する学習を安定させる手法。anet-lab では TBO と略す。
_Avoid_: value transform, Bellman transform

### Replay・PER

**Replay初期優先度ヒント**:
学習Actorが既存の行動推論結果から生成し、ReplayBufferが初期優先度を完成させるまで不透明な`float32[B,K]`として運ぶ小さなメタデータ。Actorが計算した最終優先度ではなく、payloadの意味は生成元Agentと初期優先度推定器が所有する。
_Avoid_: Actor優先度, Actor priority, Actor-computed priority, final priority

**Actor Qヒント**:
Replay初期優先度ヒントへ格納するDQN固有payload。学習Actorが行動推論で既に計算した、実行行動の`Q(s,a)`と`max_a Q_online(s,a)`の2列を指す。ReplayBuffer共通層は列の意味を解釈しない。
_Avoid_: Actor優先度, Actor priority, Actor TD error, final priority

**近似Actor初期優先度**:
ReplayBufferが遷移のサンプリング可能化境界で、Replay初期優先度ヒント（DQNではActor Qヒント）、target return、終端情報、実n-step数から計算する初期優先度。最初の有効なLearner更新後はLearner優先度が最終権威になる。
_Avoid_: Actor優先度, Actor priority, Actor-computed priority, final priority

**優先度source**:
現在のSumTree leafへ適用した値の由来を表す`none`、`fixed_initial`、`max_initial`、`actor_initial`、`learner_updated`の区分。初期状態かどうか、フォールバック理由、item generationとは別の概念。
_Avoid_: initial flag, fallback reason, priority type

**raw priority**:
`per_alpha`適用前の非負優先度。Learnerと近似Actorでは`abs(TD error) + per_eps`へ必要なclipを適用した値を指す。
_Avoid_: leaf priority, sampling probability, TD error

**SumTree leaf priority**:
raw priorityへ`per_alpha`を適用した、SumTreeに保存されサンプリング質量として使われる値。最大優先度初期化とActor/Learner比較はこの空間を扱う。
_Avoid_: raw priority, TD error, sampling probability

**replay item key**:
Sampleされたreplay itemを後続の優先度更新まで識別する、generationと物理slotをpackしたopaqueな`int64`値。Sampleが返し、Learnerが解釈せず`UpdatePriorities`へ返す。
_Avoid_: index, physical index, logical index, slot index

**slot index**:
ReplayBuffer内部のリングストレージ上の物理位置。全envを1次元化した位置は`flat_slot_index`と呼び、外向けのreplay item keyとは区別する。
_Avoid_: replay item key, logical index, sample index

### Module・設定参照

**Module Config**:
`Module`インスタンスへ構築時に実際に注入された不変の設定情報。設定ファイルのinclude・継承・overrideを解決した後の値を、注入先を区別できるscope付きkeyで保持する。`Module::GetConfigData()`は`std::optional<ConfigData>`を返し、`nullopt`は取得未対応、値ありの空`ConfigData`は対応済みだが設定項目なしを表す。複合Moduleは子を含む実効設定を返し、同一scope/keyの同値は統合、異値は契約違反とする。ログdumpと将来のGUI設定ブラウザは同じ情報を利用する。元の記述箇所やoverride経路は追跡せず、実動情報は含めない。`Module`は純粋interfaceを原則とするが、段階導入中は既存実装への波及を避けるため`GetConfigData()`だけdefaultで`nullopt`を返し、全Module対応時にpure virtual化を再検討する。
_Avoid_: raw config, config provenance, runtime property, EnvSpec metadata

**Property**:
構築時に設定から導出された値、`auto`戦略の選択結果、または実行中の状態など、Moduleの実動情報。Module Configとは別の自己記述情報として扱い、`ConfigData`へ混在させない。
_Avoid_: config, resolved config

### Env・実行

**Env name**:
同一Run内でBatchEnvと各laneの出力元を人間が識別するための、不透明でimmutableな表示名。BatchEnv nameはRun内で一意とし、lane nameは`<BatchEnv name>[lane index]`で表す。Envはnameの意味を解析せず、挙動、RunMode、設定、seed、RNG、metrics identityの決定には使用しない。
_Avoid_: Env ID, Env key, role, context

**Actor Env contract**:
AgentがActor生成時に、対象EnvのEnvSpecを入力・出力として受理できるか判断する契約。Train EnvとEval EnvのEnvSpecが同一であること自体は全Agent共通の要件ではなく、汎用的な同一I/O判定はAgentが選べる補助手段として扱う。
_Avoid_: Env間互換性, RunManager compatibility, canonical EnvSpec

### DropMerge終局診断

**NoLegal candidate**:
DropMergeで、現在のfruitを配置できるDROPが1つもない瞬間状態。盤面の安定や状態の継続時間は含まない。
_Avoid_: candidate, 詰み

**blocked persistence**:
NoLegal candidateが連続して成立する物理frame区間と、その継続長。
_Avoid_: persistence, 安定待ち

**投了**:
NoLegal candidateが継続する盤面で方策がNOOPを選び続ける行動。EnvはこれをNoLegalDrop終端として受理する（罰なし）。
_Avoid_: 諦め, give-up, resignation action（明示的な専用行動と混同）

**NEET**:
合法DROPが存在するのにNOOPを選び続ける方策の病理。投了（詰みでのNOOP）とは区別する。
_Avoid_: 停滞（あいまい）, no-op連発

**NoDropTimeout**:
合法DROPがあるのにDROPしないままショットクロックを使い切った敗着終端（done）。詰みでの投了は含まない（それはNoLegalDrop）。
_Avoid_: 時間切れtruncation, タイムアウト打ち切り

### DropMerge観測拡張

**直前行動観測**:
直前stepでAgentが選択した行動をObservationに含める拡張の総称。obsには「そのobsへ至らせた行動」を入れ、episode先頭は未行動（全ゼロ・マーカー無し）とする。記録するのは選択した命令であり、執行の成否は問わない。
_Avoid_: action feedback, action echo

**prev-action trio**:
DropMergeのvector観測末尾へ連結する [valid, noop, drop_x] の3 scalar。drop_xは直前DROP命令列の中心を [-1,1] 正規化した値で、非DROPは0。独立した観測キーではない。
_Avoid_: prev_actionキー（PRD900の別キー設計と混同）, action one-hot

**DROP列マーカー**:
direct系action modeでは未使用になるgridのdropper classを再利用し、直前DROPの命令列をtop rowに描画するマーカー。move系の「現在のdropper位置」表示とは別意味。
_Avoid_: dropper marker（move系表示と混同）

### 実行系統

**RunMode**:
Train / Eval 系（Eval, Eval1, Eval2）という実行系統の区分。Env は生成時に自分の RunMode を固定して保持し（Sampler 選択・終端契約・挙動分岐に使う。`GetRunMode()` で参照）、Reset / Step の実行時引数では受け取らない。Actor の network 選択にも同じ区分を使う。configured eval tag のタグ名（eval1 等）とは別概念。
_Avoid_: per-call mode, eval flag, 実行時モード引数

**configured eval tag**（評価タグ）:
`train.eval.[tag]` で宣言する常設評価系の宣言と識別子。1 タグ = 1 configured eval インスタンス（タグ文字列が Env name になる）で、`interval=0` なら定義のみの寝タグ。EvalPanel はタグの内容（run_mode / env overlay）を鏡写し参照する別インスタンスであり、第二のタグインスタンスにはならない。
_Avoid_: eval profile, eval preset, RunMode（別概念）

**dormant**（寝タグの状態）:
`interval=0` の評価タグが持つ「意図された休止」状態。宣言検証と name 予約だけが行われ、runner / Env / actor / observer は生成されない。意図された状態なので fail-fast の対象外——dormant タグを参照する metrics はエラーではなく、タグごと 1 回の WARN で skip される（未宣言タグの参照＝typo は従来どおりエラー）。
_Avoid_: disabled（エラー状態と紛らわしい）, 無効タグ

### Runner GUI

**主領域**:
Runner 画面中央で Train pane・Eval pane が占める領域。両 pane が定位置（Train=Centre、Eval=Right 内側）に表示されている間だけ 50:50 分割ポリシーの対象になり、どちらかが非表示・浮動・移動中は wxAUI の素の配分に任せる。
_Avoid_: main area, センター領域

**補助 pane**:
HeatMap・Conv2d のように View メニューから動的追加される右端の可視化 pane 群。名前は生成時刻で一意化され、追加時に既存の補助 pane 列（最外周の右列）へ同じ幅で縦積みされる。Train/Eval/QValue/Log の常設 pane とは寿命も配置規約も異なる。
_Avoid_: aux panel, ツールパネル, サブパネル

### Metrics基盤

**Run作業セット**:
`runs`ディレクトリ直下のRunフォルダ群。Metrics Viewerが可視化とキャッシュ構築の対象とする「見たいRun」の集合で、フォルダを入れる・出す・リネームするというファイル操作だけが登録・解除・改名の手段。Viewerは作業セット外のRunを追跡しない。
_Avoid_: runs list（UI表示と混同）, アーカイブ（作業セット外の保管側を指す）

**Metricsマスタ**:
Runフォルダ内でメトリクス記録の正とみなす唯一の系列。現行は`metrics.jsonl`（圧縮後は`.jsonl.gz`も同一マスタのライフサイクル段階）。キャッシュ・表示・外部ブリッジはすべてマスタから従属導出する。
_Avoid_: 生データ（L0スキーマと混同）, ソース（あいまい）

**Metricsキャッシュ**:
マスタから従属構築される破棄可能な導出物。いつ削除してもマスタから同一内容を再構築でき、サイズ・更新時刻・スキーマ版の不整合を検出したら警告なしに全破棄・再構築してよい。マスタと並ぶ第2の正にしない。
_Avoid_: スナップショット（旧kryo実装と混同）, DB（役割が伝わらない）

**Metricsキャッシュ世代**:
1回の全再構築で作られるMetricsキャッシュの同一性。通常追記では同じ世代を保ち、マスタ同一性の喪失やschema不一致による全再構築で新しい世代へ切り替わる。
_Avoid_: schema version（様式の版であって内容の同一性ではない）, update ID（通常追記では変わらない）

**TagStats**:
1つのscalar tagでcommit済みの有効な全点を対象とする正確な統計。表示rangeやLODに依存せず、無効値とtag隔離後の点は含まない。
_Avoid_: viewport stats, LOD stats, range summary

**序数**（メトリクス点の）:
1つのtag内での記録の出現順（0始まり）。同一tagのstepは非減少だが一意ではない（同一stepへ複数episodeの値が正当に載る）ため、点のidentityと順序は序数が持ち、stepは座標値として扱う。
_Avoid_: index（多義）, step（座標値であってidentityではない）

**LODバケット**:
tag内の連続する序数区間（幅は固定倍率の冪）をmin/max/last/件数/総和へ畳み込んだ集約単位。バケット幅は序数で数え、step幅では数えない（tag間・区間内のstep密度差に依存しないため）。全スケールでの描画とバケット境界に整合する区間統計はここから導出する。
_Avoid_: ダウンサンプル点（単一代表値と混同）, ビン（step軸の等幅分割と混同）

### 観測と可視化

**Observation**:
環境がエージェントへ渡す観測。anet-lab では複数の観測キー（`vector` / `grid` / `action_mask`）を持つ `TensorDict`。
_Avoid_: obs tensor, state input

**観測キー**:
Observation `TensorDict` の各エントリを識別する文字列キー。`vector`（低次元ベクトル観測）、`grid`（画像・格子観測）、`action_mask`（合法手マスク。学習入力には含めない）。
_Avoid_: obs field, channel, feature key

**probe**:
可視化のために experience / agent / network いずれかの source から、キーと index を指定して数値列を取り出すデータ抽出子。
_Avoid_: accessor, getter, sampler

**状態スイープ**:
選択された vector-type の観測キーから2成分を選び、格子状に走査して各点を Q ネットワークに通し、値をヒートマップ化する可視化手法。
_Avoid_: sweep heatmap, state grid scan

### 分類・データセット

**targets**:
分類データセットの各画像に付いた正解クラス ID（画像 1 枚に 1 個、長さ＝画像数）。observation の `vector` に載り、学習器の教師信号になる。
_Avoid_: labels, ground truth, y

**class_names**:
クラス ID から人間可読なクラス名への対応（長さ＝クラス数）。行動空間の `value_labels`（離散行動＝予測クラスの表示名）はこれを指す（各画像の正解 ID である targets とは別物）。
_Avoid_: labels, classes, categories

**DatasetKey**:
利用者が設定カタログで明示する、データセット実体の process 内 identity。同一 key は同一の manifest / cache を共有し、異なる key は設定が同じでも別実体。path 比較から identity を導出しない。
_Avoid_: dataset name, dataset id, path identity

**ImageDataset**:
process 内で共有される分類データセット実体（manifest / decode / pre-augment cache）。DatasetKey を identity として複数 Env から共有され、走査位置や乱数のような mutable 状態を持たない。
_Avoid_: data source, per-env dataset

**ImageDataSource**:
1 つの Env が専有するデータ供給機構（sampler / RNG / augment / collate）。mutable 状態を持つため定義上共有されず、カタログ identity を持たない（設定は Env 配下に置く）。共有されるのは参照先の ImageDataset だけ。
_Avoid_: shared source, source catalog, dataset

**Train ImageDataSource**:
ImageCls の学習系統に使う標準の ImageDataSource 定義。ImageCls は、使用する RunMode にかかわらず Train ImageDataSource と Eval ImageDataSource の両方が特定されていることを前提とし、両者が同じ ImageDataset を明示的に参照することも許容する。
_Avoid_: main source, default source, train dataset

**Eval ImageDataSource**:
ImageCls の評価系統に使う標準の ImageDataSource 定義。ImageCls は、使用する RunMode にかかわらず Train ImageDataSource と Eval ImageDataSource の両方が特定されていることを前提とし、両者が同じ ImageDataset を明示的に参照することも許容する。
_Avoid_: eval tag source, configured eval source, eval dataset

**ImageCls Dataset pair**:
標準の Train ImageDataSource と Eval ImageDataSource が参照する、1つの画像分類問題を構成するImageDatasetの組。同じImageDatasetを両側から参照することもできるが、入力shapeとclass_namesを含む観測・行動契約は一致しなければならない。
_Avoid_: train/eval catalog, dataset profile, paired source

**epoch**:
データセットを一巡（全サンプルを 1 回ずつ）走査する単位。train は毎 epoch シャッフルし直す（一巡ごとに scalar `epoch_count` が進む）。「データ被覆」の関心であり、metrics を区切る窓（episode）とは別軸で、両者を等値にしない。eval の episode は「eval 1 回の採点区間」であり、`eval_window.mode=full`なら1 epochと一致し、`rotating`なら複数windowをかけて一巡する。
_Avoid_: pass, round, sweep

**eval window**:
eval の accuracy 1 点を作る採点区間。`eval_window.mode=full` なら全件 1 周、`rotating` なら `eval_window.rotating.size` 件ずつカーソル継続で消化する（複数 window で全件を一巡）。`rotating.size`は非選択中もrotating方式の完全な設定として保持され、未設定状態を持たない。データ被覆の単位である epoch（dataset cycle）とは別軸。Env はこの区間の終端を episode（lane 0 の done）へ翻訳して報告する。
_Avoid_: eval episode 長, eval バッチ, サンプル数（size と区別）

**accuracy**（env scalar キー）:
直近に確定した採点サイクルの正解率。サイクル＝train は epoch（wrap で確定、初回 wrap 前は NaN）、eval は eval 1 回分（終端で確定）。境界で snapshot し `GetScalar("accuracy")` は常に snapshot を読む。per-lane の窓値（旧 episode 単位 stream キー）は持たない。
_Avoid_: episode_accuracy, pass_accuracy, batch_accuracy
