# ニューラルネットワーク

## 元の材料

[ニューラルネットワーク](../../../design/130_neural_networks.jp.md) blob:57c61b8f0a630d76118920795939e3621897313f

## 要点

- Network は TensorDict の各キーを TensorSpec に照らし、設定で組み立てた Body と Head に渡す。
- Body は分岐を含む計算グラフを実行して特徴を作り、Head が用途別の出力を返す。
- フレームを重ねる軸と入力の形は構築時の契約に含まれ、Forward への入力がその仕様に合う必要がある。
- clone とモデル同期では複製先の所有・更新のタイミングを扱う。dtype と device は呼び出す側の実行条件にも依存する。
- 中間出力の取得や部分 Forward、Noisy Linear の実行状態は、通常の学習・評価から区別して指定する。

## 内容

### 2. 基本概念と外部contract

- Networkの入力と出力は、文字列キーからTensorへのmapである`TensorDict`で受け渡す。
- stack 軸を持つ入力が branch に届く場合、先頭 module が軸を扱う宣言を持たなければ、構築時に branch・module・キー・形を WARN に出す。`Flatten`・`StackMerge`・`Permute`・`Reshape` と、DropMerge の2つの Embedder は stack 軸を扱う。
- structureに現れないblockは実体化せず、Runではそのスキーマを検証しない。既存blockのtypeだけを別の種類へ書き換えると、元の種類の設定キーが残り、使ったときに未読キーとして止まる。別名のblockを定義し、structureの側で選び直す。

### 3. コンポーネント定義

- `NetworkConfig` — block catalog、branch、出力key、config profileを保持するimmutableな構築情報。
- `NetworkBuilder` — Config、入力spec、HeadFactory、deviceから完成したNetworkを構築する入口。
- `NetworkModuleRepository` — type名からNetworkModuleFactoryを解決するprocess registry。
- `NetworkHeadFactory` — dummy featureからAgent固有Headを構築する。
- `NetworkForwardExecutor` — 対象Networkの寿命と用途別の準備済み処理を保持する公開実行器。

### 5. 静的構造

- Repositoryが管理するのはfactoryであり、学習中のmodule instanceやparameterではない。実体のNetworkとparameterはAgent側のResourceとして所有される。

### 6. 処理フロー

- DefaultDQNのHead設定はAgent配下の `head.init.*` / `head.projection.*` / `head.use_dueling` が読み口である。skipは `skip.head.init.*` / `skip.head.projection.*` と `skip.head.use_dueling`（bool）・`skip.network_mode` を使う。Bodyの配線は `net.*` / `skip.net.*` が所有し、Head設定をNetworkConfigへ混在させない。
- dummy forwardにより、設定で省略可能な入出力次元を実Tensor shapeから確定し、Headの入力shapeを構築時に検証する。
- capture で `nn_trace/` を要求すると、branch の入力と描画できる block の出力を収集できる。要求しない forward では作らない。

### 7. 設定・lifetime・エラー・性能特性

- input specと実入力TensorDictのshape/dtypeが一致しない場合はNetwork境界で検出する。
- `spectral`はzero-init weightを構築時に拒否する。zero-initを維持する残差blockには`init2.mode=he`等を指定するか、`spectral_cap`を使う。
- `Clone(device)`は同じ構築情報とparameterを持つ別Network instanceを作る。
- Observationの`uint8`から浮動小数への変換などはBoundaryPreprocessorのcontractとして扱う。
- `MakeGraphViz`はNetworkのbranch、factorごとの依存edge、block、shape、parameter情報を構築情報から出力する。branch設定詳細を有効にした場合は`bind_concat_dim`も表示する。

### 8. テストと拡張時の確認事項

- Configからの正常構築と不正設定のfail-fastをtestする。
- dummy forwardと実forwardでshape/dtype/deviceが一致する。
- train/eval modeでDropout、Normalization、DropPathの意味が正しい。
- AMP/BF16とFP32の両経路を、対応するAgent testから確認する。
- bindを拡張する場合はparse、依存順、循環、実行shape、ToJson、GraphViz、未使用入力WARNを同じterm/factor契約で確認する。
- 主なunit testはnn_test.cppにあり、Agentとの結合はDQN/ImageClsのtestでも検証する。

## 主要な概念

- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [314 テンソル処理](../../concepts/framework/314_tensor.md)
- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
- [NoisyNetの観測](../../concepts/methods/noisynet.md)
