# anet-lab

anet-lab は libtorch を基盤とした強化学習実験プロジェクトです。
この文書は、実装仕様ではなく、プロジェクト内で使う強化学習ドメイン用語の意味を揃えるための用語集です。

## Language

### DQN・価値表現

**DQN系エージェント**:
行動価値を推定する Q ネットワークを中心に学習する Agent 群。DefaultDQN、QR-DQN、Rainbow のような DQN 派生手法を指す。
_Avoid_: Q Agent, value agent

**価値ストリーム**:
Dueling DQN 系の Q ネットワークで、状態価値 V を推定するための特徴表現と最終射影の流れ。
_Avoid_: V branch, value branch

**アドバンテージストリーム**:
Dueling DQN 系の Q ネットワークで、行動ごとの優位性 A を推定するための特徴表現と最終射影の流れ。
_Avoid_: A branch, advantage branch

**Bellmanターゲット**:
報酬と次状態のブートストラップ価値から作る、現在の Q 値が近づくべき教師値。
_Avoid_: target Q, TD target

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
観測の2成分（既定では `vector` の2次元）を格子状に走査し、各点を Q ネットワークに通して値をヒートマップ化する可視化手法。
_Avoid_: sweep heatmap, state grid scan
