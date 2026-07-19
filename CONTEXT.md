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

### Env・実行

**Env name**:
同一Run内でBatchEnvと各laneの出力元を人間が識別するための、不透明でimmutableな表示名。BatchEnv nameはRun内で一意とし、lane nameは`<BatchEnv name>[lane index]`で表す。Envはnameの意味を解析せず、挙動、RunMode、設定、seed、RNG、metrics identityの決定には使用しない。
_Avoid_: Env ID, Env key, role, context

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

**epoch**:
データセットを一巡（全サンプルを 1 回ずつ）走査する単位。train は毎 epoch シャッフルし直す（一巡ごとに scalar `epoch_count` が進む）。「データ被覆」の関心であり、metrics を区切る窓（episode）とは別軸で、両者を等値にしない。eval の episode は「eval 1 回の採点区間」（`eval_samples=all` なら 1 epoch と一致、指定時はローテーションで複数 eval かけて一巡）。
_Avoid_: pass, round, sweep

**eval window**:
eval の accuracy 1 点を作る採点区間。`eval_window.mode=full` なら全件 1 周、`rotating` なら `eval_window.size` 件ずつカーソル継続で消化する（複数 window で全件を一巡）。データ被覆の単位である epoch（dataset cycle）とは別軸。Env はこの区間の終端を episode（lane 0 の done）へ翻訳して報告する。
_Avoid_: eval episode 長, eval バッチ, サンプル数（size と区別）

**accuracy**（env scalar キー）:
直近に確定した採点サイクルの正解率。サイクル＝train は epoch（wrap で確定、初回 wrap 前は NaN）、eval は eval 1 回分（終端で確定）。境界で snapshot し `GetScalar("accuracy")` は常に snapshot を読む。per-lane の窓値（旧 episode 単位 stream キー）は持たない。
_Avoid_: episode_accuracy, pass_accuracy, batch_accuracy
