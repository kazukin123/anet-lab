# ニューラルネットワーク

> 主たる観点: 機能単位（Networkとmodule。内部の処理工程を時系列で併記）

## 1. はじめに

### 1.1 目的

本書は、設定からNetworkを構築し、TensorDictをNetwork BodyとHeadへ流すANETのニューラルネットワーク基盤を説明する。
moduleの物理配置、構築時の検証、forward、clone/sync、可視化の境界を一続きで理解できることを目的とする。

### 1.2 対象読者

- Network構成やmoduleを追加・変更する開発者
- Agent固有Headと共通Backboneの境界を確認する開発者
- shape、dtype、device、初期化、性能特性をレビューする担当者

### 1.3 記載範囲

現行の`NetworkConfig`、`NetworkBuilder`、Body/Branch/Block/Module、Head、forwardと可視化を扱う。
Agent共通の所有権は[Agentと学習](110_agents_and_learning.jp.md)、DQN固有のlossとNetwork構成は[DQN系Agent](200_dqn_agents.jp.md)を参照する。

## 2. 基本概念と外部contract

### 2.1 TensorDictとTensorSpec

Networkの入力と出力は、文字列キーからTensorへのmapである`TensorDict`で受け渡す。
`TensorSpecMap`は各入力キーのshape、dtype、値域を表す。EnvSpecから引き継いだObservation specを基礎とし、Agentが所有する追加入力specもNetwork構築前に同じmapへ加えられる。DefaultDQNのIQNでは、EnvのObservationではない`taus`をAgentが追加し、Actor/Learnerがforward直前に入力TensorDictのcopyへ注入する。

`NetworkBoundaryPreprocessor`はNetwork Bodyの入口で、次を担当する。

- 必要な入力キーの存在とspecを確認する。
- batch dimensionを含むshapeを検証する。
- raw指定されていない入力をNetwork向けの形式へ変換する。
- 元のObservationを変更せず、forward用TensorDictを作る。

### 2.2 frame stack入力の軸contract

`DefaultDQNAgent`で`use_stacker=true`かつ`stack_count=S>1`の場合、stack対象キーのNetwork入力specは、EnvSpecの`original_shape`に先頭のstack軸を加えた`[S, *original_shape]`になる。Network構築時のdummy inputとActor／Learnerの実入力は、いずれもbatch軸を加えた`[B, S, *original_shape]`を使う。`stack_keys`対象外のキーにはstack軸を追加しない。GraphVizのinput spec表示もこのcontractを反映する。

Network構成では、stack軸を暗黙に復元せず、目的に応じて次のmoduleを明示する。

| 入力と用途 | module構成 | Network上のshape |
|---|---|---|
| Vector stackをMLPへ入力 | `Flatten` | `[B,S,F]`から`[B,S*F]` |
| 連続Grid stackをConv2dへ入力 | `StackMerge` | `[B,S,C,H,W]`から`[B,S*C,H,W]` |
| Vector stackを時間方向Conv1dへ入力 | `Permute(0,2,1)`の後に`Conv1d` | `[B,S,F]`から`[B,F,S]` |

`Reshape`は任意shape変換の汎用moduleとして利用できるが、stack軸contractを復元するための必須処理ではない。

離散GridはNetworkへのraw specでは`[S,1,H,W]`を保持する一方、`NetworkBoundaryPreprocessor`がone-hot化するときにstackとclassをchannelへ統合する。したがってbranch入口は従来どおり`[B,S*C,H,W]`となり、`StackMerge`を追加せずConv2dへ接続する。連続Gridはこのone-hot境界を通らないため、branch入口でも`[B,S,C,H,W]`を保持する。

### 2.3 BodyとHead

- Network Bodyは設定で定義されたbranch DAGを実行し、特徴量TensorDictを生成する。
- Network HeadはAgent固有の出力へ変換する。DQNのQ値、分位数、ImageClsのclass logitsなどが該当する。
- `Network`はBodyとHeadをまとめ、単一の`Forward(TensorDict)`を公開する。

この分離により、共通Backboneを複数のAgent固有Headから利用できる。

NetworkForwardConfig はキーと選択済み ConfigData をコピーして保持する不変の搬送型で、具象設定型・解釈・検証を持たない。Network::CreateForwardExecutor(config, clock_kind, seed) は構築後の Module／Head に準備を委譲する。通常の部品は本体を直接呼び、設定を使う具象部品だけが私有の解釈済み enum・数値を持つ処理を返す。子 Module へも準備を伝達し、bind・依存閉包・trace・capture・FP32 保護は既存の計算本体を共用する。新しい設定の追加は利用する具象部品だけに閉じる。

CreateForwardExecutor は shared_ptr で所有された Network に対して呼ぶ。

公開のnn.hppはNNを使う側の契約を置く。NetworkHead／NetworkHeadExecutor／NetworkHeadFactory、HeadGraphVizInfo、WeightNormConfigの定義は内部のnn_impl.hppへ置く。Module／HeadのCreateExecutorは、専用の実行器が不要ならnullptrを返し、通常の本体を直接呼ぶ。公開ヘッダには署名に必要な前方宣言だけを残す。State／Context・部品の実行interface・RunForwardは公開署名へ出さず、RunForwardはcpp内で既定／準備済みのBody・Head計算を共用する。

NetworkForwardExecutor は対象Networkの寿命を保持する。CreateForwardExecutorは名前付き設定と時計種別を部品の準備時に使い、実行器は準備済み処理とNetwork別の内部Stateを保持する。既定設定では部品を準備せず、State・時計種別を持たず、clock_kindとseedを使用しない。重みは既存Moduleを参照し、clone先は別に準備する。実行中の設定切替は行わない。

NetworkForwardCallInfoは時計値・episode_start・行対応だけを持つ不変クラスで、別文で生成してForward(input, info)へ渡す。WithRowMappingは全Paramsをコピーして行対応だけを置換する。内部ContextはStateポインタとCallInfoのconst参照を持ち、時計種別を含まない。Impl内のMakeContextはTensorコピー・抽選・State更新・演算をせず、参照のlifetimeは同期Forward内に限定する。Tensorストレージは呼び出し中に変更しない。

既定の Network::Forward(input)／ForwardUpTo(input, branch_key) は内部へ NetworkForwardContext::GetDefault() を伝え、Noisy は μ-only で動く。dummy 実行、追加診断、対象外 Agent、既存の TensorDictFunction も既定経路を使う。名前付き設定を Context で渡す経路は削除する。設定取得に RTTI・dynamic_cast・型トークンを使わず、State 内部の既存の型管理は維持する。参照存在は設定読込時、使用する項目内容と利用時制約は具象の準備処理、実データの shape/device・時計値は演算時に検証する。ValidateForwardReuse は各部品へ再利用検証を委譲する。

### 2.4 設定記法

`NetworkConfig`は主に次の情報を解決する。

- `net.block.[name].type`: 再利用可能なblock定義
- `net.body.[name].structure`: blockを`>`で接続する直列構造
- branchのbind term、`bind_concat_dim`、raw key、出力key
- block/branch単位のconfig profile

`bind`は`,`区切りのtermを入力し、各termでは`*`区切りのfactorをfeature-last elementwise productする。`*`は`,`より優先し、rank差は低rank側のbatch直後へsingleton次元を挿入して揃える。複数termはbranch単位の`bind_concat_dim`（既定1、負値可）で連結するが、batch次元0への連結は禁止する。

設定のJSON表現と`ToJson()`は`bind_terms`と`bind_concat_dim`を現行schemaとし、旧`bind_keys`は現行契約に含めない。

```properties
net.branch.[fusion].bind = main_feature * tau_embedding
net.branch.[merged].bind = fusion, context
net.branch.[merged].bind_concat_dim = -1
```

`(raw)`はfactorへ記述できるが、意味はkey-globalである。同じkeyを別branchから参照してもNetwork全体でraw扱いになる。structureでは、出力tag、入力tag参照、block反復などを表せる。実際の使用例は[apps/runner/config/nn.txt](../../apps/runner/config/nn.txt)と各Run設定を参照する。

## 3. コンポーネント定義

| コンポーネント | 定義 |
|---|---|
| `NetworkConfig` | block catalog、branch、出力key、config profileを保持するimmutableな構築情報 |
| `NetworkBuilder` | Config、入力spec、HeadFactory、deviceから完成したNetworkを構築する入口 |
| `NetworkBodyBuilder` | branch依存を解析・検証し、実行順を確定したBodyを構築する |
| `NetworkBody` | 入力整形後、branchを順に実行して特徴量TensorDictを作る |
| `NetworkBranch` | bind termのfactor積とterm連結を行い、1本のNetworkStructへ入力して結果を登録するDAG node |
| `NetworkStruct` | orderedなNetworkBlock列を実行する |
| `NetworkBlock` | 名前と1つのNetworkModuleを束ねる |
| `NetworkModule` | Tensorを受け取りTensorを返す多態的なNN module |
| `NetworkModuleRepository` | type名からNetworkModuleFactoryを解決するprocess registry |
| `NetworkHeadFactory` | dummy featureからAgent固有Headを構築する |
| `NetworkHead` | Body featureをAgent固有出力へ変換する |
| `Network` | BodyとHeadを所有し、forward、clone、hard/soft copy、GraphVizを公開する |
| `NetworkForwardExecutor` | 対象Networkの寿命と用途別の準備済み処理を保持する公開実行器 |
| `NetworkModuleExecutor`／`NetworkHeadExecutor` | 内部のModule／Head作者が満たす実行interface。CreateExecutorで生成し、不要なら本体を直接呼ぶ |

## 4. コードマップ

| 領域 | 主なファイル |
|---|---|
| 利用側の公開Config・CallInfo・Network・Executor・診断・Builder | [nn.hpp](../../core/anet-core/include/anet/nn.hpp) |
| Module／Head実装側のinterface・Body/Branch/Block/Repository | [nn_impl.hpp](../../core/anet-core/src/nn_impl.hpp)、[nn_impl.cpp](../../core/anet-core/src/nn_impl.cpp) |
| module実装と登録 | [nn_modules.cpp](../../core/anet-core/src/nn_modules.cpp) |
| Agent向けHead | [nn_heads.hpp](../../core/anet-core/src/nn_heads.hpp)、[nn_heads.cpp](../../core/anet-core/src/nn_heads.cpp) |
| DQN系Head | [dqn_based_heads.hpp](../../core/anet-core/src/dqn_based_heads.hpp)、[dqn_based_heads.cpp](../../core/anet-core/src/dqn_based_heads.cpp) |
| TensorDict共通処理 | [common.hpp](../../core/anet-core/include/anet/common.hpp)、[common.cpp](../../core/anet-core/src/common.cpp) |
| Tensor補助処理 | [tensor_util.hpp](../../core/anet-core/include/anet/tensor_util.hpp)、[tensor_check.hpp](../../core/anet-core/include/anet/tensor_check.hpp) |
| 設定例 | [nn.txt](../../apps/runner/config/nn.txt)、[nn_cnx.txt](../../apps/runner/config/nn_cnx.txt) |
| unit test | [nn_test.cpp](../../core/anet-core/src/nn_test.cpp) |

### 4.1 登録済みmoduleの分類

現行`InitNN()`は次の種類を`NetworkModuleRepository`へ登録する。

| 分類 | type例 |
|---|---|
| 形状・routing | `Flatten`、`Permute`、`Reshape`、`StackMerge`、`Dropout` |
| 活性化 | `ReLU`、`GELU`、`SiLU`、`Mish`、`LeakyReLU` |
| 正規化・pooling | `GroupNorm`、`LayerNorm`、`LayerNorm2d`、`BatchNorm2d`、`GAP1D`、`GAP2D`、`MaxPool2d` |
| embedding | `CosineEmbedding`、`HybridSpatialEmbedder`、`SpatialEmbedder`、`SpatialPositionalEmbedding2D` |
| layer | `Linear`、`Conv1d`、`Conv2d`、`ResBlock`、`CNBlock`、`TransformerEncoder` |
| token | `ClsAppend`、`ClsExtract` |

この表はConfigのコメントではなく、現行の`InitNN()`登録内容を基準としている。

`CosineEmbedding`は`cos.num_basis`（既定64）を使い、taus `(B,K)`を`cos(πiτ)`の基底 `(B,K,n)`へ変換する。後段の射影と活性化は既存の`Linear`、`ReLU`、`SiLU`などを設定で接続する。

## 5. 静的構造

```mermaid
classDiagram
direction LR

class NetworkBuilder
class NetworkConfig
class Network {
  +CreateForwardExecutor(config, clock_kind, seed)
  +Forward(input, callback)
  +Clone(device)
  +CopyTo(target)
  +SoftCopyTo(target, tau)
}
class NetworkBody
class NetworkBranch
class NetworkStruct
class NetworkBlock
class NetworkModule
class NetworkModuleRepository
class NetworkHead
class NetworkHeadFactory
class NetworkForwardExecutor {
  +Forward(input, info, callback)
}
class NetworkForwardState {
  <<internal>>
}
class NetworkForwardContext {
  <<internal>>
}
class NetworkForwardCallInfo
class NetworkModuleExecutor
class NetworkHeadExecutor

NetworkForwardExecutor --> Network : lifetime
NetworkForwardExecutor o-- NetworkForwardState : named Network/purpose
NetworkForwardExecutor ..> NetworkForwardCallInfo
NetworkForwardExecutor --> NetworkModuleExecutor : fixed calls
NetworkForwardExecutor --> NetworkHeadExecutor : fixed call
NetworkModule --|> NetworkModuleExecutor
NetworkHead --|> NetworkHeadExecutor
Network ..> NetworkForwardExecutor : prepares named/default
NetworkBuilder ..> NetworkConfig
NetworkBuilder ..> NetworkModuleRepository
NetworkBuilder ..> NetworkHeadFactory
NetworkBuilder ..> Network : 構築
Network *-- NetworkBody
Network *-- NetworkHead
NetworkBody *-- NetworkBranch
NetworkBranch *-- NetworkStruct
NetworkStruct *-- NetworkBlock
NetworkBlock *-- NetworkModule
NetworkHeadFactory ..> NetworkHead : dummy featureから生成
```

Repositoryが管理するのはfactoryであり、学習中のmodule instanceやparameterではない。実体のNetworkとparameterはAgent側のResourceとして所有される。

## 6. 処理フロー

### 6.1 Network構築

```mermaid
sequenceDiagram
    participant A as AgentFactory
    participant B as NetworkBuilder
    participant C as NetworkBodyBuilder
    participant R as NetworkModuleRepository
    participant D as NetworkBody
    participant H as NetworkHeadFactory

    A->>B: BuildNetwork(config, input_specs, head_factory, seed, device)
    B->>C: branch DAGを構築
    C->>R: block typeごとにfactoryを解決
    R-->>C: NetworkModuleFactory
    C-->>B: NetworkBody
    B->>D: dummy inputをforward
    D-->>B: dummy features
    B->>H: CreateHead(dummy_features)
    H-->>B: NetworkHead
    B-->>A: deviceへ配置したNetwork
```

dummy forwardにより、設定で省略可能な入出力次元を実Tensor shapeから確定し、Headの入力shapeを構築時に検証する。

`seed`はNetworkの構築情報として保持され、`Clone()`時にも再利用される。module固有乱数は1 Network 1個の`ModuleRandomSource`からpurpose名ごとに遅延生成される。spectral normalizationは`"spectral_norm"` streamだけを使うため、parameter初期化が使うglobal torch RNGを消費しない。DefaultDQN / Rainbow / ImageClsはAgent seedから`"network"`、MuZeroは`"network.rep"` / `"network.dyn"` / `"network.pred"`を導出する。

### 6.2 Forward の事前準備

```mermaid
sequenceDiagram
    participant C as Actor/Learner initialization
    participant N as Network
    participant M as Concrete Module/Head
    participant E as NetworkForwardExecutor
    C->>N: CreateForwardExecutor(config, clock_kind, seed)
    alt default setting
        N->>E: direct calls, own Network (no State)
    else named setting
        N->>M: CreateExecutor(config, clock_kind)
        M-->>N: private prepared execution or direct body
        N->>E: fix calls, own State and Network
    end
    E-->>C: execution
```

### 6.3 Forward

```mermaid
sequenceDiagram
    participant C as Actor/Learner
    participant E as NetworkForwardExecutor
    participant R as internal RunForward
    participant P as BoundaryPreprocessor
    participant B as NetworkBody
    participant H as prepared Head/body
    C->>C: construct immutable CallInfo
    C->>E: Forward(input TensorDict, info)
    E->>E: MakeContext(info)
    E->>R: RunForward(body, input, context, fixed calls)
    R->>B: Forward(input, context, fixed calls)
    B->>P: Format(raw input)
    P-->>B: formatted TensorDict
    B->>B: execute branches in dependency order
    B-->>R: feature TensorDict
    R->>H: Forward(features, context)
    H-->>R: Agent TensorDict
    R-->>E: output
    E-->>C: output
```

`TraceCallback`を渡した場合は、module/branchの中間Tensorを可視化・診断用に収集できる。

## 7. 設定・lifetime・エラー・性能特性

### 7.1 構築時の検証

- 未登録のmodule type、存在しないblock、解決不能なbind key、循環branchは構築時に失敗させる。
- input specのkeyが全bind factorにもdirect `net.body.output` mappingにも参照されない場合は、keyごとに1 build 1回だけWARNする。この診断は書き忘れの候補を示すだけで、最終出力への到達性や意味的寄与を保証しない。
- bind積はfactor間のbatch sizeを検証する。複数termの連結は先頭termのrankで`bind_concat_dim`を正規化し、範囲、batch次元でないこと、term間batch sizeを明示検証する。それ以外のbroadcast/concat shape不整合はlibtorchのエラーとして扱う。
- input specと実入力TensorDictのshape/dtypeが一致しない場合はNetwork境界で検出する。
- IQN Headはrank 3 `(B,K,D)`、IQN Dueling Headはvalue/advantageのB/K一致を局所入力契約としてdummy forward時に検証する。これは`taus`が最終出力へ意味的に寄与することの証明ではない。
- Weight初期化modeは`default`、`xavier`、`he`、`orthogonal`、`constant`、`trunc_normal`を受け付け、未知値をfail-fastする。
- `Linear`、`Conv1d`、`Conv2d`、`ResBlock`、`CNBlock`、`TransformerEncoder`の`weight_norm.mode`は`none`、`spectral`、`spectral_cap`だけを受け付ける。未知値はkey、指定値、許容値を含めてfail-fastする。
- `spectral`はzero-init weightを構築時に拒否する。zero-initを維持する残差blockには`init2.mode=he`等を指定するか、`spectral_cap`を使う。
- TransformerEncoderでSNを使う場合は`tf.use_sdpa=true`を要求する。
- config profileはblockの出現順へ値を展開する。profile名や補間設定の誤りを暗黙に無視しない。

### 7.2 cloneとmodel同期

- `Clone(device)`は同じ構築情報とparameterを持つ別Network instanceを作る。
- `CopyTo`はhard update、`SoftCopyTo`はtauによるsoft updateを行う。
- SNを含む`SoftCopyTo`は、source/targetを変更する前に`tau`を検証し、`0 <= tau <= 0.1`または`tau=1`だけを許可する。bufferをlerpした後、targetのu/vを単位normへ戻す。SNを含まないNetworkの既存tau契約は変えない。
- Policy/Target Networkの更新時期と所有権は具象Agentが決める。
- Eval用cloneはsnapshotの一貫性を得る代わりにcopy時間と追加memoryを使う。

### 7.3 dtypeとdevice

- Networkは構築後に指定deviceへ配置される。
- AMP/BF16の適用はActor/Learner側の実行contextとmodule設定の両方に依存する。
- BatchNormやLayerNormなど、数値安定性のためFP32実行を選べるmoduleがある。
- Observationの`uint8`から浮動小数への変換などはBoundaryPreprocessorのcontractとして扱う。

### 7.4 spectral normalization

SN対象はmoduleが所有するweightだけであり、Head、embedding、bias、normalization affine、layerscale、cls tokenは対象外である。ResBlockは`conv1` / `conv2` / `downsample`、CNBlockは`dwconv` / `pwconv1` / `pwconv2`、TransformerEncoderは各layerのQ / K / V / `out_proj` / `linear1` / `linear2`を独立に扱う。

u/vはnamed bufferで、専用RNGによる初期化後に15回power iterationする。SN計算はFP32・autocast OFFで行い、u/vの更新はtrain modeかつGradMode有効なforwardだけである。sigmaは毎forwardでweightから再計算し、weightへの勾配をdetachしない。`spectral`の実効weightは`W / sigma`、`spectral_cap`は`W / max(1, abs(sigma))`である。

`NetworkModule::GetSpectralNormEntries()`は既定空で、各対象moduleがweight、mode、u/vを公開する。Networkはbranch/block walkで完全layer名を付与する。parameter normは生L2に加えて、SN weightの寄与だけを実効weightへ置換した実効L2、feature/readout別の最大sigma、invalid-count device scalarを返す。SN layerがない場合、実効L2は生L2と一致し、sigmaは`NaN`である。

### 7.5 可視化と性能

- `MakeGraphViz`はNetworkのbranch、factorごとの依存edge、block、shape、parameter情報を構築情報から出力する。branch設定詳細を有効にした場合は`bind_concat_dim`も表示する。
- `GetTensorDictFunction`とTraceCallbackはConv2d可視化やprobeからNetwork内部へ到達するseamである。
- forward、attention、主要blockはprofile対象であり、module追加時はshape/batch sizeに応じたcostを実測する。
- moduleを細分化しすぎたprofile rangeは測定noiseになるため、意味のある処理境界を選ぶ。

### 7.6 forward captureと部分forward

`Network::Forward`、`NetworkForwardExecutor::Forward`、`Network::ForwardUpTo`はoptionalな`NetworkForwardCapture`を受け取る。呼出側が完全一致の鍵、または末尾`/`の接頭辞を要求し、`Wants`・`Add`・`Get`・`GetAll`・`Unfulfilled`・`Reset`で扱う。鍵1つにTensor1つを保持し、未要求鍵や同じ鍵への2回目の`Add`はfail-fastする。`Get`は完全一致の値（未記入はundefined Tensor）、`GetAll`は接頭辞に一致した鍵とTensorを記入順に返す。接頭辞`X/`は`X`自体に一致しない。

書き手は二つある。Bodyは各branch実行後に要求されたbranch名の出力をdetachして書く。moduleは`NetworkForwardContext`経由で素鍵の要求を確認し、診断値を書く。NetworkStructが与えた`<branch>/<block インスタンス名>`をmoduleの素鍵の後ろへ付けるため、`(*2)`の繰返しでも鍵は衝突しない。インスタンス名はcheckpointと同じ`<定義名>_<連番>`であり、traceの描画番号は使わない。moduleの直接呼出しでは位置が付かない。branch名の`/`は構築時に禁止する。

captureはforward入口で結果だけをResetし、要求を保持する。出口では完全一致要求が書かれていない場合、または接頭辞要求に1件も一致しない場合に、要求・到達branch・記入済み鍵・利用可能branch一覧を添えてfail-fastする。captureを渡さないforwardは位置文字列を生成せず、TraceCallbackとActorの数値経路も維持する。

`Network::ForwardUpTo(input, branch_key, capture)`は既定ContextでFormat済み入力と対象branchのancestor closureだけを既存のトポロジカル順で実行し、そのstateを返す。閉包は`ComputeDependencyClosure`で一元化する。bind factorがinput specにも同名branchにも存在する場合は、builderと同じくinput keyを優先して依存探索を終了する。未知branchは指定名と`GetBranchNames()`の一覧を含めてfail-fastする。

TransformerEncoderは公開定数`kTransformerEncoderAttentionEntropyCaptureKey`（`TransformerEncoder/attention_entropy`）を使い、要求時だけ各層のattention入力と実効Q/K重みを`nn_util`の純関数`ComputeAttentionEntropy`へ渡す。関数内はautocast無効・NoGrad・FP32で射影、softmax、エントロピー、head・query・sample平均、`ln n`による正規化まで完結する。系列長2未満はNaN。Encoderは層数長のFP32 Tensorを1回書き、確率行列は搬送しない。SDPAの学習forwardは変更しない。

可塑性統計はrank-2 `(N,D)` の特徴と指標要求集合を受け、detachしたFP32 CPU特徴から要求された統計だけを計算して呼出側でcacheする。srank系を要求しないstepではSVDを実行せず、δ=0.01 / 0.05 / 0.20を同時要求した場合も1回の`svdvals`とcumsumを共有する。結果の各値はoptionalであり、未計算フィールドを成立値として読めない。無印のsrankはδ=0.01を表す。

`ComputeParameterNormSplit(final_feature_branch_key)`は同じ依存閉包を使い、閉包内branchの学習parameterをfeature、閉包外branchとheadをreadoutとして、各群の生L2、実効L2、最大sigmaをFP32 device scalarで返す。forwardとRNGを使わず、`requires_grad`がfalseのparameterはnorm対象外とする。

### 7.7 Noisy Linear と実行状態

Body の `Linear` は `noisy.enabled=false`、`noisy.sigma0=0.5`、`force_fp32=false` を既定とする。Noisy は既存の weight/bias を μ として保ち、有効時だけ `sigma_weight` / `sigma_bias` を登録する。σ は符号付きの学習 parameter であり、初期値は weight/bias とも `sigma0 / sqrt(fan_in)`。初期 μ の名前・初期化・乱数順を維持する。

Body LinearとHeadが合成する具象LinearProjectionModuleは、そのprivateな設定型でnoisy.mode/hold/share/countを準備時に解釈する。利用する設定の型・列挙・値域・必要項目・episode/batch・時計種別の制約を準備時に検証する。未参照項目とNoisy OFFの設定群は休眠する。μ-onlyの未使用保持項目は必須にしないが、利用部品が認識する明示値は検証する。

名前付き実行オブジェクト内部の `NetworkForwardState` は module 実体毎の ε と専用 RNG を持つ不透明な箱。call は時計が変わると再抽選、count は Actor の N 回毎、episode は Actor の `episode_start` が立った lane のみ再抽選する。同じ時計の再 forward は同じ ε を使い、同じ episode 境界を二度反映しない。count は episode 境界でリセットしない。保持中の shape/device 不一致と時計の逆行は fail-fast する。row_mapping は不変な NetworkForwardRowMapping とし、CPU の int64 index と元行数を構築時に検証して使用 device へ一度だけ転送する。射影は内部 index を公開しない SelectRows を使い、実行時は行数・device だけを確認する。GetIndices は検査用の独立した複製を返す。

解釈済み値は具象の準備済み処理が所有する。NN内部の PreparedModuleChildren<N> は子の準備結果と本体への固定した呼び先を管理し、PreparedHeadExecutor<Head, ForwardMethod, N> は指定した計算関数をコンパイル時に結び付ける。共通部品はDQN/Noisy/具体設定を解釈せず、全子が既定動作ならHead本体を直接呼ぶ。通常 forward に設定検索・解釈・設定用ロックはなく、設定元の寿命・変更・同名キーから独立する。

sample は `Linear(x, μ)` と factorized ε を使う `Linear(x*ε_in, σ)*ε_out` を分けて計算し、batch 毎の重み `[B,out,in]` を実体化しない。ε は batch 以外の軸へ broadcast され、IQN の全 τ に同じ ε が届く。μ-only と `GetDefault()` は σ を計算 graph に入れず、ε/RNG を消費しない。SN は μ だけを既存の FP32 境界で正規化する。`force_fp32` は Noisy の ON/OFF と独立であり、通常の AMP 継承と Head の FP32 境界を維持する。

clone・hard/soft copy・保存読込は μ/σ を含む parameter と既存 buffer を扱い、箱の ε・時計・RNG を含めない。コピーや snapshot 同期は再抽選の境界にならない。`GetNoisyParameterEntries()` の μ/σ 対応・weight/bias・branch metadata は optimizer と診断だけに使用し、実行設定の解釈へ流用しない。

`ComputeNoisyParameterStatistics(feature_key)` は feature の依存閉包と readout に分類し、weight/bias 別の σ mean-abs と L2 ratio を返す。ratio の μ 分母は σ を持つ射影だけを集計し、SN は実効 μ を使う。対象なしは NaN、分母ゼロは ratio だけ NaN。既存 weight norm は σ を除外する。

## 8. テストと拡張時の確認事項

Networkを変更する場合は次を確認する。

1. Configからの正常構築と不正設定のfail-fastをtestする。
2. dummy forwardと実forwardでshape/dtype/deviceが一致する。
3. clone、hard copy、soft copy後のparameterと出力を確認する。
4. train/eval modeでDropout、Normalization、DropPathの意味が正しい。
5. AMP/BF16とFP32の両経路を、対応するAgent testから確認する。
6. 新しいmodule typeを`InitNN()`へ登録し、設定例と名称を一致させる。
7. GraphViz/Traceを壊さず、必要なら可視化可能性を公開する。
8. bindを拡張する場合はparse、依存順、循環、実行shape、ToJson、GraphViz、未使用入力WARNを同じterm/factor契約で確認する。

実行設定の責任境界・独自 Module / Head の拡張テストは [PRD 084 検証手順](../../core/anet-core/testdata/prd084/README.md)を参照する。

主なunit testは[nn_test.cpp](../../core/anet-core/src/nn_test.cpp)にあり、Agentとの結合はDQN/ImageClsのtestでも検証する。

## 9. 関連文書

- [フレームワーク全体概要](010_framework_overview.jp.md)
- [Agentと学習](110_agents_and_learning.jp.md)
- [可観測性](140_observability.jp.md)
- [ReplayBuffer](150_replay_buffer.jp.md)
- [DQN系Agent](200_dqn_agents.jp.md)
- [設定例: nn.txt](../../apps/runner/config/nn.txt)
- [TensorDict統一ADR](../adr/0002-tensordict-function-unify.md)
- [SDPA ADR](../adr/0004-sdpa-attention-via-aten.md)
- [dropout設定ADR](../adr/0007-nn-dropout-config-semantics.md)
- [WeightInit ADR](../adr/0008-weight-init-mode-string.md)
- [IQN bind積DAG ADR](../adr/0018-iqn-via-bind-product-dag.md)
- [NN Forward 事前準備 ADR](../adr/0051-nn-forward-prepares-concrete-calls-and-purpose-owned-execution.md)
