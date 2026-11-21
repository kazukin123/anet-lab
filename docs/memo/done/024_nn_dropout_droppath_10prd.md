# NN モジュール dropout / DropPath サポート

> 設計分担: Claude=設計/PRD、実装=Codex。本書は self-contained。
> 動機の上流は ImageCls 過学習対策（`022_imagecls_augmentation_10prd.md` 拡張 / `023_imagecls_gap_head_10prd.md` GAP2D）。本書は **モデル側の正則化レバー**（dropout / DropPath）を NN モジュール全般へ追加する。
> ①②③ は独立に実装・A/B 可能。効果切り分けのため **1 個ずつ** 入れて評価する。

## Context（背景・目的）

ImageCls（Food-101）の汎化検証で **Transformer 系・深い ResNet 系が過学習**（train 95%+ / eval 低、ユーザー確認）。改善案として dropout 強化が挙がったが、現状 NN モジュールは **内部 dropout を一切持たない**。唯一の dropout は独立ブロック `DropoutModule`（structure に `Drop` として挿す要素 dropout、`nn_modules.cpp:337`）のみで、特に `CustomTransformerEncoderLayer` は dropout 4 箇所すべて非対応。

目的: dropout を NN モジュールへ拡張する。対象は次の 3 グループ。

- **① ResBlock**: DropPath（残差枝の Stochastic Depth）＋ Dropout2d（conv 間の channel dropout）を config 駆動（既定 OFF）で追加。
- **② TransformerEncoder**: 要素 dropout（attention / FFN / residual）＋ DropPath を追加。
- **③ Embedder/PosEmbed**: **新規コード無し**。pos_drop は既存 `Drop` ブロックで構造記述として表現する（後述）。

**DropPath とは**: 残差 `out = x + F(x)` の **枝 F(x) のみ**をサンプル単位で確率的にゼロ化する。skip 接続が残るのでブロック出力は **恒等 x**（出力ゼロではない）。これにより「サンプルごとにネットの深さが変わる」= Stochastic Depth となり、深い ResNet/ViT の過学習に効く主力レバー。eval 時は全枝 ON（no-op）。

**前提（確定済み・本書では再検討しない）**: conv standalone 向け dropout モジュールは不要。Linear は既存 `DropoutModule` で対応。dropout が機能する train/eval mode 切替は整備済み（`TrainingModeGuard`、後述）。

## 確定した設計判断

1. **DropPath は ①② 横断**。共有 free 関数 `anet::nn::DropPath` を 1 つ用意し、ResBlock と Transformer 双方が残差枝に適用する。**枝出力のみ**を落とし、skip / downsample 射影は決して落とさない。
2. **粒度はサンプル単位**（mask shape `[N,1,1,...]`）。**深さ方向スケジュールは持たない**（ブロックごと定数）。ラダーが欲しければ config で別ブロックを定義して手動表現（`(*N)` 反復内は同率）。
3. **設定キー命名**: `_rate` 接尾辞で率を示し、`dropout` / `droppath` を各 1 語表記で統一。"2d" は conv 文脈で自明なので付けない。
   - `*.droppath_rate` … 残差枝 Stochastic Depth。
   - `res.dropout_rate` … ResBlock の conv1→conv2 間 Dropout2d/channel dropout。
   - `[Drop].dropout_rate` … 構造ブロック上の任意点要素 dropout。
   - `tf.hidden_dropout_rate` … Transformer の hidden activations / residual branches 用の要素 dropout。
   - `tf.attn_dropout_rate` … Transformer の attention weights dropout。
   - 値域 `[0.0, 1.0)`、既定すべて `0.0`（OFF）。
4. **② は timm ViT 流**: `tf.hidden_dropout_rate`（FFN内＋residual dropout1/2）＋ `tf.attn_dropout_rate`（attention）＋ `tf.droppath_rate`。残差は要素 dropout と DropPath を併用（枝順: `dropout → DropPath → add`）。
5. **① Dropout2d × BatchNorm は WARN して指定通り動かす**（AGENTS.md: 非推奨だが動作可能 → 実行＋警告）。不正値（範囲外）は `ANET_SYSTEM_ERROR`。
6. **③ はコード追加なし**。pos_drop は既存 `Drop` ブロックを構造に挿して表現（DropoutModule＝構造上の任意点 dropout という既存の役割と一致）。
7. **既存 `DropoutModule`（`Drop`）の設定キーを `p` → `dropout_rate` にハード改名**（統一性）。役割（構造上の任意点 dropout、モジュール内 dropout と直交）は不変。**後方互換・WARN は入れない**（ユーザー判断）。旧 `p` は読まれず、設定は利用者が責任を持って `dropout_rate` で入れる前提。in-repo の `ImageCls.txt` は本 PRD で更新する。

## 前提事実（実コード確認済み）

> 基準コミット: HEAD `f3f2150`。`nn_modules.cpp` / `nn_impl.hpp` / `nn_test.cpp` は未コミット変更なし（行番号は HEAD 基準で安定）。`nn_modules.cpp` は全 1958 行。

- **train/eval mode 切替は整備済み**（dropout/DropPath が効く前提）:
  - `anet::TrainingModeGuard`（`core/anet-core/include/anet/nn_util.hpp:71`）= RAII で `module.train(flag)`、デストラクタで復帰。
  - ImageCls: 学習区間で `TrainingModeGuard train_guard(*network_, true)`（`image_cls_agent.cpp:94`）。推論は `network_->eval()`（`:154`）。
  - DQN 系（DropMerge）: `ForwardOnlineWithTrain` が `TrainingModeGuard guard(*online_net_, true)`（`dqn_based_agent.cpp:135`）。
  - `register_module` 経由で登録した submodule に train/eval が再帰伝播するため、新規 dropout も `register_module` すれば自動追従。
- **DropoutModule**（`nn_modules.cpp:337-391`）: `p>0` のとき `torch::nn::Dropout` を `register_module("dropout", ...)`。`p<=0` は no-op。Factory が `[0,1)` 検証、登録名 `"Dropout"`（`:1925`）= structure の `Drop`。
- **ResBlock**:
  - `ResBlockConfig`（`nn_modules.cpp:630-642`）: `channels/kernel_size/stride/.../norm_type/...` を保持。`ResBlockModule` は `config_` を保持。
  - `ResBlockModule::Forward`（`:671-791`）。Lazy init（`if (!conv1_)`, `:676-728`）で conv1_/norm1_/conv2_/norm2_/（必要なら）downsample_conv_ を構築。
  - **pre-act（v2, `act_mode_==Pre`, :731-762）**: `out = conv1(pre_act); if(norm2_) out=norm2_(out); out=Activate(out);`（`:758`）`out=conv2_(out);`（`:759`）`return out + residual;`（`:762`）。`residual` は input か downsample 射影。
  - **post-act（v1, :763-790）**: `out=conv1(input); if(norm1_) out=norm1_(out); out=Activate(out);`（`:772`）`out=conv2_(out); if(norm2_) out=norm2_(out);`（`:775-776`）`out += residual;`（`:786`）`out=Activate(out);`（`:787`）。
  - メンバ（`:846-854`）に conv/norm holder。`GetCurrentConfigData`（`:793-811`）が config を全ダンプ。
  - Factory（`:858-906`）: `Config` が `ANET_READ_CONFIG(config_data, res.xxx)` で読込（`:872-882`）。`CreateModule`（`:901`）は検証無しで生成。
- **Transformer**:
  - `SdpaSelfAttention`（`nn_modules.cpp:1249-1293`）: `const double dropout_p = mha->is_training() ? mha->options.dropout() : 0.0;`（`:1287`）→ `at::scaled_dot_product_attention(q,k,v,{},dropout_p,false)`（`:1288`）。**MHA の dropout option を読む口が既にある**。検証は `_qkv_same_embed_dim` / `add_bias_kv` 等のみで、**dropout option は弾かない**（`nn_test.cpp:752` の reject テストは `add_bias_kv(true)` で発火）。
  - `CustomTransformerEncoderLayer`（`:1296-1403`）: ctor（`:1298`）が `MultiheadAttentionOptions(d_model,nhead)`（`:1305`、dropout 未指定=0）で `mha_` 構築。`linear1_/linear2_/norm1_/norm2_` を保持。**dropout メンバは無い**。
    - pre-LN（`:1329-1363`）: `x = x + attn_out;`（`:1349`）/ FFN は `linear1`（`:1355`）→ `gelu/relu`（`:1357`）→ `linear2`（`:1359`）→ `x = x + ffn_out;`（`:1363`）。
    - post-LN（`:1364-1389`）: `x = norm1_(x + attn_out);`（`:1379`）/ FFN `linear1`（`:1382`）→act（`:1384`）→`linear2`（`:1386`）→`x = norm2_(x + ffn_out);`（`:1388`）。
  - `TransformerConfig`（`:1405-1413`）: `d_model/nhead/num_layers/dim_feedforward/norm_first/use_sdpa/activation`。
  - `TransformerEncoderModule`（`:1416-1494`）: ctor（`:1418`）が層を生成、最終 `norm_`（pre-LN 時のみ）。`GetCurrentConfigData`（`:1477`）。Factory（`:1496-1517`）が `ANET_READ_CONFIG`（`:1502-1508`）。
- **Embedder（③）**:
  - `SpatialPositionalEmbedding2DModule`（`:991-1042`, 登録 `"SpatialPositionalEmbedding2D"`/構造名 `PosEmbed2D`）: `[B,C,H,W]` に学習可能 `y_embed/x_embed` を加算（`:1021`）し `out.flatten(2).transpose(1,2)` で `[B,H*W,C]` 系列化（`:1025`）。**本物の位置埋め込み**。
  - `HybridSpatialEmbedderModule`（`:1064`）/ `SpatialEmbedderModule`（`:1191`）: ベクトル観測→画像 `[B,C,H,W]` の**入力エンコーダ**（one-hot 化 / broadcast）。位置埋め込みではない。
  - ViT 構造例（`apps/runner/config/ImageCls.txt:240-243`）: `PatchEmbed16 > PosEmbed2D > ClsAppend > TransEnc_ViT > ClsExtract > LinearOut` 等。`ClsAppend`/`ClsExtract` 登録（`:1954-1955`）。

## 設計方針

### A. 共有 DropPath ヘルパ（`nn_impl.hpp` 宣言 / `nn_modules.cpp` 定義、`SdpaSelfAttention` と同パターン）

- 宣言（`nn_impl.hpp`、`SdpaSelfAttention` 宣言 `:15` の隣）:
  `torch::Tensor DropPath(const torch::Tensor& x, double drop_prob, bool training);`
- 定義（`nn_modules.cpp`、`SdpaSelfAttention` 定義付近）。参考実装:
  ```cpp
  torch::Tensor anet::nn::DropPath(const torch::Tensor& x, double drop_prob, bool training)
  {
      if (!training || drop_prob <= 0.0) return x;           // eval / 無効は完全 no-op
      const double keep_prob = 1.0 - drop_prob;
      std::vector<int64_t> shape(x.dim(), 1);
      shape[0] = x.size(0);                                  // [N,1,1,...] サンプル単位
      torch::Tensor mask = torch::empty(shape, x.options()).bernoulli_(keep_prob);
      return x / keep_prob * mask;                           // inverted scaling
  }
  ```
- mask 形状は `x.dim()` から動的生成（[N,C,H,W] と [N,S,D] 両対応）。RNG は torch 既定ジェネレータ（`manual_seed` / ADR 0006 `setDeterministicAlgorithms` 尊重）。**有効化すると RNG 消費が変わり dropout 無し run とは bit 一致しない**（同一 config では再現可能）。

### B. ① ResBlock（`nn_modules.cpp`）

1. `ResBlockConfig`（`:630`）にフィールド追加:
   ```cpp
   double droppath_rate = 0.0;  ///< 残差枝の Stochastic Depth ドロップ確率
   double dropout_rate   = 0.0;  ///< conv1→conv2 間 Dropout2d（channel dropout）確率
   ```
2. `ResBlockModule` メンバに `torch::nn::Dropout2d dropout2d_{ nullptr };` を追加（`:846` 付近）。Lazy init（`:711` の conv2 構築直後）で `config_.dropout_rate > 0` のとき
   `dropout2d_ = register_module("dropout2d", torch::nn::Dropout2d(torch::nn::Dropout2dOptions(config_.dropout_rate)));`。
3. **Dropout2d 挿入**（v1/v2 共通: 活性化済み特徴マップ、conv2 直前 = WideResNet 配置）。`out = Activate(out);` の直後・`out = conv2_->forward(out);` の直前に `if (dropout2d_) out = dropout2d_->forward(out);`（pre-act `:758→759` の間 / post-act `:772→775` の間）。
4. **DropPath 挿入**（枝 `out` のみ）:
   - pre-act（`:762`）: `return anet::nn::DropPath(out, config_.droppath_rate, is_training()) + residual;`
   - post-act（`:786` 直前）: `out = anet::nn::DropPath(out, config_.droppath_rate, is_training());` を入れてから `out += residual; out = Activate(out);`
   - downsample 有（stride>1/チャネル変化）でも一律適用。落ちた時の出力は射影済み shortcut（= 次元整合した恒等）。
5. `GetCurrentConfigData`（`:793`）に `cd.Set("droppath_rate", config_.droppath_rate);` / `cd.Set("dropout_rate", config_.dropout_rate);`。
6. Factory（`:858`）: `ANET_READ_CONFIG(config_data, res.droppath_rate);` / `ANET_READ_CONFIG(config_data, res.dropout_rate);` を追加。`CreateModule`（`:901`）で検証:
   - 両 rate が `[0.0, 1.0)` 外 → `ANET_SYSTEM_ERROR`（キー・指定値・期待範囲を含める）。
   - `res.dropout_rate > 0 && res.norm_type == "batch"` → `LOG::warn()`（英語）。キー `dropout_rate`、指定値、理由（BN×channel dropout の variance shift 不調和）、推奨代替（`droppath_rate` を使う / `norm_type` を `group`・`none` に）を出す。`ANET_LOG_WARN` ではなく `LOG::warn()`。
7. プロファイル: 既存 `ANET_PROFILE_SCOPE(pre_act/post_act)` 内。個別スコープ追加不要。

### C. ② TransformerEncoder（`nn_modules.cpp`）

1. `TransformerConfig`（`:1405`）に `double hidden_dropout_rate=0.0; double attn_dropout_rate=0.0; double droppath_rate=0.0;`。
2. `CustomTransformerEncoderLayer`:
   - ctor（`:1298`）に 3 rate を引数追加。`mha_opts.dropout(attn_dropout_rate)` を設定（`:1305`）→ attention weights dropout が SDPA（`:1287` が読む）と legacy MHA の両方で有効化。
   - `torch::nn::Dropout dropout_{ nullptr }` を `hidden_dropout_rate>0` 時に 1 つ register。各 site で同一インスタンスを呼べば呼ぶ度に独立マスク。`droppath_rate_` をメンバ保持。
   - **pre-LN**（`:1349` / `:1355-1363`）:
     - attn 枝: `attn_out = ...; if(dropout_) attn_out = dropout_->forward(attn_out); x = x + anet::nn::DropPath(attn_out, droppath_rate_, is_training());`
     - FFN 枝: `linear1 → act → (dropout_) → linear2 → (dropout_) → x = x + DropPath(ffn_out, droppath_rate_, is_training());`
   - **post-LN**（`:1379` / `:1382-1388`）:
     - attn: `attn_out=...; if(dropout_) attn_out=dropout_(attn_out); x = norm1_(x + DropPath(attn_out, droppath_rate_, is_training()));`
     - FFN: `linear1 → act → (dropout_) → linear2 → (dropout_) → x = norm2_(x + DropPath(ffn_out, droppath_rate_, is_training()));`
   - 最終 `norm_`（`TransformerEncoderModule` 側）に dropout は入れない。
3. `TransformerEncoderModule` ctor（`:1418`）: `config_` の 3 rate を各 `CustomTransformerEncoderLayer` に渡す。
4. `GetCurrentConfigData`（`:1477`）に 3 キー Set。
5. Factory（`:1496`）: 3 つ `ANET_READ_CONFIG` 追加。`CreateModule` で 3 rate を `[0,1)` 検証（範囲外 `ANET_SYSTEM_ERROR`）。
6. プロファイル: 既存 `self_attn` / `ffn_*` スコープ内。追加無し。

### D. ③ Embedder/PosEmbed — コード変更なし（運用で対応）

pos_drop は「系列への要素 dropout を構造の一点に挿す」だけ。既存 `Drop`（DropoutModule）をそのまま使う。

- ViT CLS: `PatchEmbed16 > PosEmbed2D > ClsAppend > Drop > TransEnc_ViT`（CLS を含めて pos_drop）。
- ViT GAP: `PatchEmbed16 > PosEmbed2D > Drop > TransEnc_ViT > GAP1D > LinearOut`。
- CLS の前後どちらに効かせるかは `Drop` の位置で制御。入力エンコーダ（Hybrid/SpatialEmbedder）は対象外。

### E. 命名と用語の明文化

実装時、`CustomTransformerEncoderLayer` / `ResBlockModule` のメンバ付近に短いコメントで「`res.dropout_rate` は ResBlock の Dropout2d、`tf.hidden_dropout_rate` は Transformer の要素 dropout」「DropPath は枝のみ・skip は落とさない」を残す。読者が `res.dropout_rate`(channel/2d) と 独立 `Drop` ブロックの `dropout_rate`(要素) と `tf.hidden_dropout_rate`(Transformer hidden/residual) の差を疑問に思わないようにする。

### F. 既存 DropoutModule の設定キー改名（`nn_modules.cpp` + config）

統一性のため独立ブロック `Drop`（`DropoutModule`）の設定キーを `p` → `dropout_rate` に**ハード改名**（後方互換・WARN は入れない＝ユーザー判断）:

- `DropoutModuleFactory`（`nn_modules.cpp:371`）の `Config`: `ANET_READ_CONFIG(config_data, dropout_rate)` のみ（旧 `p` は読まない）。範囲検証は従来通り `[0.0, 1.0)` 外で `ANET_SYSTEM_ERROR`。
- 範囲検証メッセージ（`:387` 付近の "Dropout p must be in [0.0, 1.0)"）を `dropout_rate` 表記へ。
- `DropoutModule::GetCurrentConfigData`（`:360`）のダンプキーを `p` → `dropout_rate`（内部メンバ名 `p_` は任意）。
- config 更新: `apps/runner/config/ImageCls.txt:101-103` の `[Drop].p` を `[Drop].dropout_rate` へ（committed 唯一の使用箇所）。
- 注意（意図的・将来の読者向け）: config 層は未読キーを検出しない（`config.hpp:149`）ため、移行漏れで `p` が残ると silent に dropout=0 になる。これは承知の上で後方互換 WARN を**敢えて入れない**設計判断なので、後からフォールバックを足さないこと。設定は利用者が `dropout_rate` で責任を持って入れる。

## 非対象（Out of Scope）

- conv standalone 向け Dropout2d/DropBlock 独立モジュール（不要で確定）。
- DropPath の深さ方向自動 linear decay（ブロックごと定数で対応）。
- 入力エンコーダ（Hybrid/SpatialEmbedder）への dropout。
- Linear への内部 dropout（既存 `Drop` ブロックで対応）。
- 過学習の A/B 評価そのもの（実装後にユーザーが seed 違い複数 run で実施）。

## 影響ファイル

| ファイル | 変更 |
|---|---|
| `core/anet-core/src/nn_impl.hpp` | `DropPath` 宣言追加（`SdpaSelfAttention` の隣） |
| `core/anet-core/src/nn_modules.cpp` | `DropPath` 定義／① ResBlock（Config・forward・Factory・dump）／② Transformer（Config・layer・Module・Factory・dump、`tf.hidden_dropout_rate`）／既存 `DropoutModule` の `p`→`dropout_rate` 改名＋後方互換なし |
| `core/anet-core/src/nn_test.cpp` | ① `[nn][resblock][dropout]` ② `[nn][transformer][dropout]` の単体テスト追加 |
| `apps/runner/config/ImageCls.txt` | `[Drop].p`（`:101-103`）を `[Drop].dropout_rate` に改名（committed 唯一の使用箇所） |
| `apps/runner/config/*.txt` | （任意・実験時）`res.droppath_rate` / `tf.droppath_rate` 等の付与、pos_drop 用 `Drop` 挿入。コードと別 PR/別作業で可 |

## 受け入れ基準

1. **ビルド緑**（x64-Debug、`anet-core-test`）、既存テスト緑（特に既存 `[transformer][sdpa]` 群が不変）。
2. **DropPath ヘルパ**: `training=false` または `drop_prob<=0` で入力一致（no-op）。`training=true` で形状不変・期待値スケール（`1/keep_prob`）・サンプル単位マスク。
3. **① ResBlock 単体**（`[resblock][dropout]`）:
   - 両 rate=0 で出力が現行一致（回帰）。
   - `eval()` では `droppath_rate>0` でも `branch + residual`（スケール無し）一致。
   - `train()` + `droppath_rate≈0.99` で出力がほぼ shortcut のみ。
   - `train()` + `dropout_rate>0` でチャネル単位の 0 平面（[H,W] 全面 0）が出る。
   - Factory が範囲外 rate で `ANET_SYSTEM_ERROR`、`dropout_rate>0 & norm_type=batch` で WARN。
4. **② Transformer 単体**（`[transformer][dropout]`）: `eval()` で各 rate>0 でも legacy/参照と一致（no-op）。`train()` で dropout により出力が割れる。pre-LN/post-LN・SDPA/legacy 両方。
5. **config dump**: `GetCurrentConfigData` 経由で `runs/<name>/config/config_data.txt` に `res.dropout_rate` / `tf.hidden_dropout_rate` / `tf.attn_dropout_rate` / `tf.droppath_rate` などの新キーが ground truth として載る。
6. **既存挙動不変**: 全 rate 既定 0.0 のとき、ビルド前と数値・チェックポイント名（`layer_*.self_attn.*` 等）が不変。
7. **DropoutModule 改名**: `[Drop].dropout_rate` が効く（旧 `p` は読まれない＝ハード改名）。config dump が `dropout_rate` を出す。`ImageCls.txt` が `dropout_rate` 表記に更新済み。

## 正直なリスク / 注意

- **attention dropout × determinism（要検証）**: `tf.attn_dropout_rate>0` は ATen SDPA（`:1288`）経由。ADR 0006 で既定 `setDeterministicAlgorithms(true)`。CUDA で SDPA+dropout の決定的 backward が無いと **throw** する可能性。検証で確認し、throw するなら本書/実装メモへ「attn_dropout は `deterministic_warn_only=true` 退避 or 当面 0」と注記。FFN/residual 要素 dropout・DropPath は単純 elementwise/bernoulli で determinism と両立。
- **二重正則化**: ② は residual に要素 dropout(dropout1/2) と DropPath を併用（timm 流で確定）。掛けすぎると学習が遅延。既定 0 + 1 個ずつ評価で回避。
- **BN×Dropout2d 不調和**: 現行 ResBlock は `norm_type=batch` 主流。`dropout_rate>0` は variance shift で逆効果になりやすい（WARN で通知済み）。ResNet 系は基本 `droppath_rate` を主レバーにする。
- **再現性**: dropout 系を有効化した run は無効 run と bit 一致しない（RNG 消費差）。同一 config 内では再現可。構成比較はブレ幅基準（seed 違い複数 run の終盤平均）で判断。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe "[resblock]"
core\anet-core\bin\Debug\anet-core-test.exe "[transformer]"
core\anet-core\bin\Debug\anet-core-test.exe
git diff --check
```

- 機能確認: runner online 起動で、`res.droppath_rate=0.1` 等を一時付与した branch が起動・学習し、`config_data.txt` に新キーが載ることを確認。
- ③ pos_drop: ViT 系構造に `Drop` を挿した config が従来通りパース・起動すること（既存 `Drop` 再利用の回帰のみ）。
- 効果（eval / gap）はユーザーが seed 違い複数 run の終盤平均で評価（本書範囲外）。

## 後続

1. 必要なら実装メモ `024_..._20impl.md` → Codex 実装 → 受け入れ緑。
2. 任意 ADR: 「dropout をモジュール内部に config 駆動で持たせる／Transformer は `hidden_dropout_rate` を使う」命名・配置規約。将来の読者が `res.dropout_rate`(channel) と `Drop.dropout_rate`(要素) の差を疑問に思い得るため記録価値あり（要否はユーザー判断）。
3. A/B: ① `droppath_rate`（ResNet18ish の深層から）→ ② `tf.droppath_rate`/`attn_dropout_rate`（ViT/Hybrid）→ ③ pos_drop の順で 1 個ずつ。
