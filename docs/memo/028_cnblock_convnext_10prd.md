# CNBlock（ConvNeXt v1）導入

> 設計分担: Claude=設計/PRD、実装=Codex。本書は self-contained。
> **前提（実装依存）**: 本書は次の2つが適用済みであることを前提とする。
> - `027_weight_init_mode_string_10prd.md`（`WeightInitConfig.mode` の文字列化）。本書は `init.mode` を最初から文字列（`"trunc_normal"` 等）で設計しており、数値modeは一切登場しない。
> - `029_config_profile_param_interp_10prd.md`（`net.config_profile` + `@`マーカーによるブロック群パラメータ補間機構）。本書は droppath の線形補間をこの機構に委ね、`cn.droppath_rate = @cn_dp01` とマーカーで書く。**029未実装だとマーカーが解決されず起動時エラーになる**。
> 027・029は互いに独立で先行実装可能。本書（028）は両者に依存し、3点が揃って初めて ConvNeXt-Tiny が実働評価できる（実装・評価は束ねて行う）。
> 動機: ImageCls（Food-101）で ResBlock系（ResNet18ish）・Transformer系（Hybrid/ViT）に続く第3の主力候補として ConvNeXt v1 を追加する。まず ImageCls で評価し、その後 DropMerge でも評価する（DropMerge向けbranch設計は後継PRD、本書スコープ外）。
> 設定デフォルトは可能な限りオリジナル論文（Liu et al. 2022, *A ConvNet for the 2020s*, ConvNeXt-Tiny）に合わせる。

## Context（背景・目的）

ImageCls の過学習対策（`022_imagecls_augmentation_10prd.md` データ拡張 / `023_imagecls_gap_head_10prd.md` GAP2D化 / `024_nn_dropout_droppath_10prd.md` dropout/DropPath）を経て、ResNet18ish系（容量増でeval 67%程度、ユーザー確認）が現状の主力構成になっている。ここに、ImageNet系で ResNet を上回る実績のある ConvNeXt を第3の選択肢として追加し、同一データセット・同一評価手順で比較できるようにする。

ConvNeXt の中核は「depthwise 7x7 conv + inverted bottleneck（4倍拡張1x1 conv）+ LayerNorm + LayerScale + Stochastic Depth」から成るブロック（本書では **CNBlock** と呼ぶ）。既存の `NetworkModuleFactory`/`NetworkModuleRepository`（`ResBlock` が参照実装）と同じ枠組みに新規ブロックタイプとして追加し、既存の `net.block`/`net.branch` DSL で ResNet18ish/Hybrid/ViT と同列に切り替え評価できるようにする。

**ConvNeXt v1 と v2 の違い（参考）**: v2 は LayerScale を廃し GRN（Global Response Normalization）を追加する。本書は v1（LayerScale方式）のみを対象とし、ブロック名に `V1` 接尾辞は付けない（v2対応が必要になった時点で別名を検討する）。

## 確定した設計判断

1. **ブロック名は "CNBlock"**（`ConvNeXtBlock` は長いため短縮）。V1/V2の接尾辞は付けない。
2. **既存 ResBlock と同じ `NetworkModuleFactory` パターン**で `NetworkModuleRepository` に登録する自己完結ブロックとして実装する。
3. **モデル規模は標準 ConvNeXt-Tiny**: `dims=[96,192,384,768]`, `depths=[3,3,9,3]`（計18ブロック）。
4. **Stemは論文準拠のPatchify**: 既存Conv2d（kernel=4, stride=4, padding=0, `init.mode="trunc_normal"`）→ 新規LayerNorm2d（eps=1e-6）。
5. **新規プリミティブ "LayerNorm2d"（channels-first）を新規登録**。既存の channels-last "LayerNorm"（Transformer用、LunarLanderの `main_feature` branchで実使用中）とは別クラス。フィールドは `BatchNorm2d`/既存`LayerNorm`と同じフラット命名（`num_channels`必須、`eps` default=1e-6）。なお **CNBlock 内での norm 適用は `cn.norm_type`（default=`layernorm2d`, `none`で無効化, 将来 batch/group 追加余地）で切替可能**にする。ResBlock の `res.norm_type` 前例に倣い、DropMerge で norm 無しが有効だった知見を CNBlock にも横展開できるようにする（デフォルトは論文準拠 layernorm2d を維持）。
6. **既存LayerNorm（channels-last）は削除しない**: `apps/runner/config/LunarLander.txt:120` の現行アクティブbranch `net.branch.[main_feature].structure = Flatten > MLP_FC1 > LN256 > SiLU > MLP_FC2 > LN128 > SiLU` で実使用中（1Dベクトル特徴量の正規化）。ConvNeXtが必要とする2D空間マップ(NCHW)の正規化とは用途が別物であり、2クラス共存とする。
7. **PRDスコープは ImageCls での CNBlock 動作確認まで**。DropMerge向け branch 設定は別PRD（後継）とする。DropMergeのgrid観測はImageClsの224x224より大幅に小さい（20x32〜40x64、`apps/runner/config/DropMerge.txt:539`）ため、stem設計を含め別途検討が必要。
8. **Head最終正規化は既存資産の再利用**: GAP2D後に `[B,768]` になった時点で、既存の channels-last "LayerNorm" ブロックを使う。論文はここも eps=1e-6 を使うため、既存 `LayerNormModuleFactory` に**後方互換な `eps` フィールド**（default=1e-5、既存Hybrid/ViT branchには影響なし、フラットフィールド）を追加し、ConvNeXt branchでは `eps=1e-6` を明示指定する。
9. **重み初期化はオリジナル論文に合わせ TruncNormal(std=0.02) をデフォルトにする**。ConvNeXt branch内の全 Conv2d/Linear（stem, downsample, dwconv, pwconv1, pwconv2, head Linear）は `init.mode="trunc_normal"` を使う。バイアスは `WeightInitializer::Initialize`（PRD027後）の仕様通り、`mode!="constant"` なら自動的にゼロ初期化される（論文のbias=0と一致、追加実装不要）。
10. **LayerScale gamma初期値は論文デフォルト1e-6**。CNBlock内で `register_parameter("gamma", torch::full({channels}, layerscale_init))` として直接生成する（`WeightInitializer`は`layer->weight`/`layer->bias`を持つtorch::nn Moduleハンドル用のため、生パラメータには使わない。`SpatialPositionalEmbedding2DModule`の`y_embed_`/`x_embed_`と同様の直接初期化パターン）。`layerscale_init<=0` でLayerScale自体を無効化できる。
11. **DropPathは既存共有関数を再利用**: `anet::nn::DropPath(x, drop_prob, training)`（サンプル単位stochastic depth、eval時no-op）。
12. **droppath_rateは論文のLinear Stochastic Depth Schedule（全18ブロック、0→0.1で線形補間）に厳密に合わせる**。ただし18個を手書きせず、**PRD029の `net.config_profile` + `@`マーカー機構で自動補間する**。CNBlock定義は channel別の4定義（CN96/CN192/CN384/CN768）に集約し `cn.droppath_rate = @cn_dp01` とマーカーで書き、structure では `CN96(*3)` 等の `(*N)` 記法で展開する。補間値は structure 内の CNBlock 出現順（ステージ跨ぎの通し番号）で `net.config_profile.[cn_dp01]`（type=linear, min=0, max=0.1）から自動計算される（具体は設計方針E参照）。
13. **GELUはCNBlock内部でinline実装**（ResBlockの `Activate()` 相当。`torch::gelu(x, "none")` 自由関数を直接呼ぶ、登録済み"GELU"ブロックは経由しない）。approximateモードは"none"固定でconfig公開しない。
14. **Depthwise 7x7 convはCNBlock内部で自己完結**（ResBlockのconv1_/conv2_と同様、`torch::nn::Conv2d(groups=channels)` を直接保持）。汎用Conv2dモジュールへの `groups` パラメータ追加は行わない。
15. **CNBlock自体はstride=1固定、in_channels==out_channels(=config.channels)固定**。ダウンサンプリング/チャンネル変更はブロック外（LayerNorm2d→Conv2d k2s2）が担当し、ResBlockのようなdownsample_conv_内蔵は持たない。初回forward時に `in_channels != config.channels` を検出したら `ANET_SYSTEM_ERROR` で即座に失敗させる。
16. **ADRは不要と判断**（LayerNorm2dの新設は既存BatchNorm2dパターン踏襲の拡張であり、「後戻りコスト大」「本物のトレードオフ」の水準に達しない。TruncNormal自体のADR相当の議論はPRD027側の `docs/adr/0008-weight-init-mode-string.md` に含まれている）。

## 前提事実（実コード確認済み）

> 基準コミット: HEAD `0598b1b`。`nn.hpp` / `nn_impl.hpp` / `nn_modules.cpp` / `nn_test.cpp` は未コミット変更なし。行番号は PRD027 適用**前**のHEAD基準（PRD027実装後は前後にズレる。実装時は近傍のクラス名で再検索すること）。

- **`ResBlockConfig`/`ResBlockModule`/`ResBlockModuleFactory`**（`nn_modules.cpp:651-951`）が参照実装。Lazy Init（`if (!conv1_)` で初回forward時に入力からin_channelsを取得しconv/norm群を構築、`register_module`で登録、`WeightInitializer::Initialize`で初期化）というパターンをCNBlockでも踏襲する。
- **`ValidateDropRate`**（`nn_modules.cpp:18-24`）:
  ```cpp
  static void ValidateDropRate(const std::string& key, double value)
  {
      if (value < 0.0 || value >= 1.0) {
          ANET_SYSTEM_ERROR("Invalid dropout rate. key=" << key
              << " value=" << value << " expected=[0.0, 1.0)");
      }
  }
  ```
  CNBlockFactoryでも `cn.droppath_rate` 検証にそのまま再利用する。
- **`DropPath`**（`nn_modules.cpp:48-62`）:
  ```cpp
  torch::Tensor anet::nn::DropPath(const torch::Tensor& x, double drop_prob, bool training)
  {
      if (!training || drop_prob <= 0.0) return x;
      ANET_CHECK_MSG(drop_prob < 1.0, "DropPath: drop_prob must be less than 1.0. actual=" << drop_prob);
      ANET_CHECK_MSG(x.dim() > 0, "DropPath: input must have a batch dimension.");
      const double keep_prob = 1.0 - drop_prob;
      std::vector<int64_t> shape(static_cast<size_t>(x.dim()), 1);
      shape[0] = x.size(0);
      torch::Tensor mask = torch::empty(shape, x.options()).bernoulli_(keep_prob);
      return x / keep_prob * mask;
  }
  ```
- **`BatchNorm2dModule`/`Factory`**（`nn_modules.cpp:537-582`）: `num_features` をフラット読込する最小形パターン。新設 `LayerNorm2d` の型として踏襲する。
- **既存 `LayerNormModule`/`Factory`**（`nn_modules.cpp:958-1009`）: `torch::nn::LayerNorm`（channels-last、`[..., normalized_shape]` を正規化）をラップ。`normalized_shape` をフラット読込（`net.block.[LN256].normalized_shape = 256` 、実例 `apps/runner/config/LunarLander.txt:91-92`）。eps未指定でtorchデフォルト1e-5のまま。`torch::nn::LayerNormOptions` は `.eps(double)` をサポート（`torch/csrc/api/include/torch/nn/options/normalization.h`、`TORCH_ARG(double, eps) = 1e-5;`）。
- **`SpatialPositionalEmbedding2DModule`** の生パラメータ直接初期化パターン（`WeightInitializer`非経由、`register_parameter`＋`torch::randn(...)`で直接構築）を、CNBlockのgamma・LayerNorm2dのweight/biasで同様に踏襲する。
- **`torch::gelu` 自由関数**（`torch/csrc/api/include/torch/nn/functional/activation.h:373-374`）: `torch::nn::functional::gelu(input, approximate)` は内部で `torch::gelu(input, approximate)` を呼ぶだけなので、CNBlock内で `torch::gelu(x, "none")` を直接呼んでよい（既存の`GELUModule`が使う`torch::nn::GELU`と数値的に同一）。
- **既存 `DropoutModule`/`Conv2dModule` 等**のGetCurrentConfigData/Factory設計（`nn_modules.cpp:361-411`, `188-247`）が、CNBlockModule/CNBlockModuleFactoryの実装パターンの参考になる。
- **libtorchに `torch::nn::init::trunc_normal_` は存在しない**（`torch/csrc/api/include/torch/nn/init.h`、確認済みバージョン: `libtorch-win-shared-with-deps-2.11.0+cu130`。`normal_`/`uniform_`/`kaiming_normal_`等は存在）。**PRD027側で自前実装済みの前提**（`WeightInitializer::TruncNormal_`、PyTorch/timm標準のerfinv法）。
- **`(*N)` 繰り返し記法**（`nn_impl.cpp:652-693`、`NetworkStructBuilder::Build`）は正規表現 `\(\*(\d+)\)` で repeat_count を抽出し、ループ内で毎回 **同一の** `block_cfg.config_data` から `factory->CreateModule(...)` を呼ぶ（インスタンス名は `block_def_name + "_" + idx` で連番、パラメータは独立だが**値は複製元と同一**）。droppath_rateを線形に変えたい場合は個別ブロック定義が必要（判断12の根拠）。
- **`NetworkModuleRepository::Instance().Register(...)` 登録一覧**: `nn_modules.cpp:2085-2128`（`InitNN()`内）。
- **config DSLのフィールド命名規則**: 単純な1~2フィールドのブロック（`BatchNorm2d`の`num_features`、既存`LayerNorm`の`normalized_shape`）はサブプレフィックスなしのフラット命名。多フィールドの複合ブロック（`ResBlock`→`res.`、`Conv2d`→`conv.`+`init.`、`TransformerEncoder`→`tf.`）はサブプレフィックス付き。新設する`LayerNorm2d`（`num_channels`+`eps`のみ）はフラット、`CNBlock`（6フィールド: channels/kernel_size/ffn_expand_ratio/layerscale_init/droppath_rate/norm_type + 3つのWeightInitConfig）は`cn.`プレフィックスに倣う。
- **テストヘルパ**: `MakeResBlockTestModule`（`nn_test.cpp:223-244`）、`CheckTensorClose`（`:115-122`）、`CopyModuleState`（`:158-177`）。ResBlock/Transformerテスト本体は`:916-1025`。
- **GAP2D**（`nn_modules.cpp:1632-1656`）: `input.mean({2,3}, /*keepdim=*/false)` で `[B,C,H,W]`→`[B,C]`。既存branchは `GAP2D > LinearOut` のように Flatten を挟まず直結している（`ImageCls.txt:349` 等）。ConvNeXt headも同様に `GAP2D > CNLayerNormHead > CNLinearOut` で直結できる。

## 設計方針

### A. `LayerNormModuleFactory` の `eps` 拡張、`LayerNorm2dModule`/`Factory` 新設（`nn_modules.cpp`、既存`LayerNormModule`直後に追記）

既存 `LayerNormModuleFactory::Config` に `double eps = 1e-5;` を追加（`ANET_READ_CONFIG(config_data, eps);` 1行、後方互換）。`LayerNormModule` コンストラクタに `eps` 引数を追加し `LayerNormOptions({normalized_shape}).eps(eps)` へ反映。

新設 `LayerNorm2dModule`（channels-first、公式ConvNeXt実装がpermuteコストを避けるために採る「NCHWのまま手動でmean/varを計算する方式」に倣う。`torch::nn::LayerNorm`は経由しない）:

```cpp
class LayerNorm2dModule : public NetworkModule {
public:
    LayerNorm2dModule(int64_t num_channels, double eps)
        : num_channels_(num_channels), eps_(eps) {}

    torch::Tensor Forward(torch::Tensor input) override
    {
        if (!weight_.defined()) {
            weight_ = register_parameter("weight", torch::ones({num_channels_}, input.options()));
            bias_ = register_parameter("bias", torch::zeros({num_channels_}, input.options()));
        }
        // input: [N, C, H, W]
        auto u = input.mean(/*dim=*/1, /*keepdim=*/true);
        auto s = (input - u).pow(2).mean(/*dim=*/1, /*keepdim=*/true);
        auto normalized = (input - u) / torch::sqrt(s + eps_);
        return weight_.view({1, num_channels_, 1, 1}) * normalized + bias_.view({1, num_channels_, 1, 1});
    }

    anet::ConfigData GetCurrentConfigData() const override
    {
        anet::ConfigData cd;
        cd.Set("num_channels", num_channels_);
        cd.Set("eps", eps_);
        return cd;
    }
private:
    int64_t num_channels_;
    double eps_;
    torch::Tensor weight_, bias_;
};

class LayerNorm2dModuleFactory final : public NetworkModuleFactory {
private:
    struct Config : anet::Config {
        int num_channels = 0;
        double eps = 1e-6;
        Config(const anet::ConfigData& config_data) : anet::Config("") {
            ANET_READ_CONFIG(config_data, num_channels);
            ANET_READ_CONFIG(config_data, eps);
        }
    };
public:
    std::shared_ptr<NetworkModule> CreateModule(const anet::ConfigData& config_data, const ModuleContext& context) const override
    {
        Config config(config_data);
        if (config.num_channels <= 0) {
            ANET_SYSTEM_ERROR("LayerNorm2dModule: 'num_channels' must be strictly positive.");
        }
        return std::make_shared<LayerNorm2dModule>(config.num_channels, config.eps);
    }
};
```

`weight_`/`bias_` は生パラメータのため `WeightInitializer` を経由しない（`torch::ones`/`torch::zeros` で直接初期化、標準LayerNormの初期値と同じ）。1x1 Conv2d（NCHW）によるpwconv1/pwconv2実装と組み合わせることで、公式実装（permute→channels-last→`nn.Linear`）とは実装経路が異なるが**数学的には同一**である。

### B. `CNBlockConfig`/`CNBlockModule`（`nn_modules.cpp`、ResBlock直後に追記）

```cpp
struct CNBlockConfig {
    int channels = 0;                    ///< in==out channels（stride/チャネル変更はブロック外が担当）
    int kernel_size = 7;                 ///< depthwise convのカーネルサイズ（論文デフォルト7）
    int ffn_expand_ratio = 4;            ///< pwconv1の拡張率（論文デフォルト4）
    double layerscale_init = 1e-6;       ///< LayerScale gammaの初期値（論文デフォルト、0以下で無効化）
    double droppath_rate = 0.0;          ///< 残差枝のStochastic Depthドロップ確率
    std::string norm_type = "layernorm2d"; ///< "layernorm2d"(論文準拠) or "none"(無効化)。将来 batch/group 追加余地
};

class CNBlockModule : public NetworkModule {
public:
    CNBlockModule(const CNBlockConfig& config,
                  const WeightInitConfig& init_dw, const WeightInitConfig& init_pw1, const WeightInitConfig& init_pw2)
        : config_(config), init_dw_(init_dw), init_pw1_(init_pw1), init_pw2_(init_pw2) {}

    torch::Tensor Forward(torch::Tensor input) override
    {
        ANET_PROFILE_FUNC();

        if (!dwconv_) {
            ANET_PROFILE_SCOPE(init);
            auto device = input.device();
            auto dtype = input.scalar_type();
            int64_t in_channels = input.size(1);
            if (in_channels != config_.channels) {
                ANET_SYSTEM_ERROR("CNBlock: in_channels(" << in_channels << ") != cn.channels(" << config_.channels
                    << "). CNBlock does not change channel count internally; insert a downsample block before it.");
            }

            const int padding = config_.kernel_size / 2;
            torch::nn::Conv2dOptions dw_opts(config_.channels, config_.channels, config_.kernel_size);
            dw_opts.stride(1).padding(padding).groups(config_.channels).bias(true);
            dwconv_ = register_module("dwconv", torch::nn::Conv2d(dw_opts));
            dwconv_->to(device, dtype);
            WeightInitializer::Initialize(dwconv_, init_dw_);

            // norm_type="layernorm2d" のときだけ生成。"none" は norm_ を nullptr のままにし Forward でスキップ
            // （ResBlockの CreateAndRegisterNorm が norm_type 分岐で nullptr を返すのと同型）
            if (config_.norm_type == "layernorm2d") {
                norm_ = register_module("norm", std::make_shared<LayerNorm2dModule>(config_.channels, /*eps=*/1e-6));
                norm_->to(device, dtype);
            } else if (config_.norm_type != "none") {
                ANET_SYSTEM_ERROR("CNBlock: unknown cn.norm_type='" << config_.norm_type
                    << "' expected one of: layernorm2d, none");
            }

            const int64_t hidden = config_.channels * config_.ffn_expand_ratio;
            pwconv1_ = register_module("pwconv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(config_.channels, hidden, 1)));
            pwconv1_->to(device, dtype);
            WeightInitializer::Initialize(pwconv1_, init_pw1_);

            pwconv2_ = register_module("pwconv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(hidden, config_.channels, 1)));
            pwconv2_->to(device, dtype);
            WeightInitializer::Initialize(pwconv2_, init_pw2_);

            if (config_.layerscale_init > 0.0) {
                gamma_ = register_parameter("gamma",
                    torch::full({config_.channels}, config_.layerscale_init, torch::TensorOptions().device(device).dtype(dtype)));
            }
        }

        torch::Tensor residual = input;
        torch::Tensor out = dwconv_->forward(input);
        if (norm_) out = norm_->Forward(out);   // norm_type="none" のときスキップ
        out = pwconv1_->forward(out);
        out = torch::gelu(out, "none");
        out = pwconv2_->forward(out);
        if (gamma_.defined()) {
            out = out * gamma_.view({1, config_.channels, 1, 1});
        }
        return DropPath(out, config_.droppath_rate, is_training()) + residual;
    }

    anet::ConfigData GetCurrentConfigData() const override
    {
        anet::ConfigData cd;
        cd.Set("channels", config_.channels);
        cd.Set("kernel_size", config_.kernel_size);
        cd.Set("ffn_expand_ratio", config_.ffn_expand_ratio);
        cd.Set("layerscale_init", config_.layerscale_init);
        cd.Set("droppath_rate", config_.droppath_rate);
        cd.Set("norm_type", config_.norm_type);
        if (dwconv_) cd.Set("in_channels", dwconv_->options.in_channels());
        return cd;
    }
private:
    CNBlockConfig config_;
    WeightInitConfig init_dw_, init_pw1_, init_pw2_;
    torch::nn::Conv2d dwconv_{ nullptr }, pwconv1_{ nullptr }, pwconv2_{ nullptr };
    std::shared_ptr<LayerNorm2dModule> norm_;
    torch::Tensor gamma_;
};
```

`kernel_size`/`stride`/`dilation` はResBlockのような個別config化をせず、`padding = kernel_size / 2` の固定計算のみ（stride=1・dilation=1固定は判断15の通り）。

### C. `CNBlockModuleFactory`

```cpp
class CNBlockModuleFactory final : public NetworkModuleFactory {
private:
    struct Config : anet::Config {
        CNBlockConfig cn;
        WeightInitConfig init_dw, init_pw1, init_pw2;

        Config(const anet::ConfigData& config_data) : anet::Config("")
        {
            // 論文準拠デフォルト: WeightInitConfig自体の構造体デフォルトは"xavier"のままなので、
            // ANET_READ_CONFIG呼び出し前に明示上書きする（ResBlockFactory::Configのinit1/init2/init_dsと同じ要領）。
            init_dw.mode = "trunc_normal";  init_dw.trunc_std = 0.02;
            init_pw1.mode = "trunc_normal"; init_pw1.trunc_std = 0.02;
            init_pw2.mode = "trunc_normal"; init_pw2.trunc_std = 0.02;

            ANET_READ_CONFIG(config_data, cn.channels);
            ANET_READ_CONFIG(config_data, cn.kernel_size);
            ANET_READ_CONFIG(config_data, cn.ffn_expand_ratio);
            ANET_READ_CONFIG(config_data, cn.layerscale_init);
            ANET_READ_CONFIG(config_data, cn.droppath_rate);
            ANET_READ_CONFIG(config_data, cn.norm_type);

            ANET_READ_CONFIG(config_data, init_dw.mode);
            ANET_READ_CONFIG(config_data, init_dw.trunc_std);
            ANET_READ_CONFIG(config_data, init_pw1.mode);
            ANET_READ_CONFIG(config_data, init_pw1.trunc_std);
            ANET_READ_CONFIG(config_data, init_pw2.mode);
            ANET_READ_CONFIG(config_data, init_pw2.trunc_std);
        }
    };
public:
    std::shared_ptr<NetworkModule> CreateModule(const anet::ConfigData& config_data, const ModuleContext& context) const override
    {
        Config config(config_data);
        if (config.cn.channels <= 0) {
            ANET_SYSTEM_ERROR("CNBlock: 'cn.channels' must be strictly positive.");
        }
        ValidateDropRate("cn.droppath_rate", config.cn.droppath_rate);
        return std::make_shared<CNBlockModule>(config.cn, config.init_dw, config.init_pw1, config.init_pw2);
    }
};
```

`InitNN()`（`nn_modules.cpp:2085-2128`）に以下を追加:
```cpp
repo.Register("LayerNorm2d", std::make_shared<LayerNorm2dModuleFactory>());
repo.Register("CNBlock", std::make_shared<CNBlockModuleFactory>());
```

**重要**: `kernel_size`/`ffn_expand_ratio`/`layerscale_init`/`norm_type`/`init_dw`/`init_pw1`/`init_pw2` は全て論文デフォルト値がC++側で既定されているため、`apps/runner/config/ImageCls.txt` 側の CNBlock 定義では `cn.channels` と droppath マーカー `cn.droppath_rate = @cn_dp01` の2フィールドのみ指定すればよい（norm を無効化したいときのみ `cn.norm_type = none` を追加する。設計方針Eを参照）。

### D. `TruncNormal` 初期化との整合性

PRD027（`docs/memo/027_weight_init_mode_string_10prd.md`）で `WeightInitConfig.mode="trunc_normal"` と `trunc_std`/`trunc_a`/`trunc_b` フィールド、`WeightInitizer::TruncNormal_` ヘルパが実装済みである前提。本書のCNBlock/stem/downsample/head Linearは全てこの `"trunc_normal"` モードを使う（判断9）。

### E. ImageCls.txt に追加する完成branch定義

`apps/runner/config/ImageCls.txt` の `# NN` セクション（既存ResBlock/Transformerブロック定義群の後）に、以下を**そのまま追加**する。

**Stem**:
```
net.block.[CNStemConv].type = Conv2d
net.block.[CNStemConv].conv.out_channels = 96
net.block.[CNStemConv].conv.kernel_size = 4
net.block.[CNStemConv].conv.stride = 4
net.block.[CNStemConv].conv.padding = 0
net.block.[CNStemConv].init.mode = trunc_normal

net.block.[CNStemLN].type = LayerNorm2d
net.block.[CNStemLN].num_channels = 96
net.block.[CNStemLN].eps = 1e-6
```

**Downsample（3箇所、LayerNorm2d→Conv2d k2s2）**:
```
net.block.[CNDownLN96].type = LayerNorm2d
net.block.[CNDownLN96].num_channels = 96
net.block.[CNDownLN96].eps = 1e-6

net.block.[CNDownConv96to192].type = Conv2d
net.block.[CNDownConv96to192].conv.out_channels = 192
net.block.[CNDownConv96to192].conv.kernel_size = 2
net.block.[CNDownConv96to192].conv.stride = 2
net.block.[CNDownConv96to192].conv.padding = 0
net.block.[CNDownConv96to192].init.mode = trunc_normal

net.block.[CNDownLN192].type = LayerNorm2d
net.block.[CNDownLN192].num_channels = 192
net.block.[CNDownLN192].eps = 1e-6

net.block.[CNDownConv192to384].type = Conv2d
net.block.[CNDownConv192to384].conv.out_channels = 384
net.block.[CNDownConv192to384].conv.kernel_size = 2
net.block.[CNDownConv192to384].conv.stride = 2
net.block.[CNDownConv192to384].conv.padding = 0
net.block.[CNDownConv192to384].init.mode = trunc_normal

net.block.[CNDownLN384].type = LayerNorm2d
net.block.[CNDownLN384].num_channels = 384
net.block.[CNDownLN384].eps = 1e-6

net.block.[CNDownConv384to768].type = Conv2d
net.block.[CNDownConv384to768].conv.out_channels = 768
net.block.[CNDownConv384to768].conv.kernel_size = 2
net.block.[CNDownConv384to768].conv.stride = 2
net.block.[CNDownConv384to768].conv.padding = 0
net.block.[CNDownConv384to768].init.mode = trunc_normal
```

**CNBlock（channel別4定義のみ）**（`cn.channels` と droppath マーカー `@cn_dp01` のみ指定。他は論文デフォルトを継承。18個手書きは PRD029 の `net.config_profile` 補間で不要）:
```
net.block.[CN96].type = CNBlock
net.block.[CN96].cn.channels = 96
net.block.[CN96].cn.droppath_rate = @cn_dp01

net.block.[CN192].type = CNBlock
net.block.[CN192].cn.channels = 192
net.block.[CN192].cn.droppath_rate = @cn_dp01

net.block.[CN384].type = CNBlock
net.block.[CN384].cn.channels = 384
net.block.[CN384].cn.droppath_rate = @cn_dp01

net.block.[CN768].type = CNBlock
net.block.[CN768].cn.channels = 768
net.block.[CN768].cn.droppath_rate = @cn_dp01
```

**droppath 補間ポリシー（PRD029 の `net.config_profile` 機構）**:
```
net.config_profile.[cn_dp01].type = linear
net.config_profile.[cn_dp01].min = 0.0
net.config_profile.[cn_dp01].max = 0.1
```
→ structure 内の CNBlock 出現順（全18個、ステージ跨ぎの通し番号）で `linspace(0, 0.1, 18)` が各インスタンスに自動配分される。手計算した参考値: 96ch群 `0.0000/0.0059/0.0118`、192ch群 `0.0176/0.0235/0.0294`、384ch群 `0.0353/0.0412/0.0471/0.0529/0.0588/0.0647/0.0706/0.0765/0.0824`、768ch群 `0.0882/0.0941/0.1000`（実効値は `config_data.txt` で確認）。

**Head**（既存の共有 `LinearOut`（`ImageCls.txt:90-92`, `init.mode`旧数値2）とは別名にして既存branchに影響を与えない）:
```
net.block.[CNLayerNormHead].type = LayerNorm
net.block.[CNLayerNormHead].normalized_shape = 768
net.block.[CNLayerNormHead].eps = 1e-6

net.block.[CNLinearOut].type = Linear
net.block.[CNLinearOut].linear.out_features = 101
net.block.[CNLinearOut].init.mode = trunc_normal
```

**Branch定義**:
```
net.branch.ConvNeXtT.bind = grid
net.branch.ConvNeXtT.structure = CNStemConv > CNStemLN > CN96(*3) > CNDownLN96 > CNDownConv96to192 > CN192(*3) > CNDownLN192 > CNDownConv192to384 > CN384(*9) > CNDownLN384 > CNDownConv384to768 > CN768(*3) > GAP2D > CNLayerNormHead > CNLinearOut
```

**有効化**: 既存の `#net.branch.[main_feature].$ = net.branch.XXX` 切り替え群（`ImageCls.txt:19-29`）に以下を追加し、他は全てコメントアウトのまま維持する:
```
net.branch.[main_feature].$ = net.branch.ConvNeXtT
```

### F. Head部の設計根拠

GAP2D（`[B,768,7,7]`→`[B,768]`）→LayerNorm（既存資産、eps拡張のみ）→Linear、という構成は論文のClassifierヘッド（`GlobalAvgPool → LayerNorm → Linear`）そのままである。

### G. 非対象（Out of Scope）

- DropMerge branch設定（後継PRD）。DropMergeのgrid観測（20x32〜40x64）はImageClsの224x224と大幅に異なり、stem（4x4 stride4）がそのまま使えない可能性が高い。
- GELU `approximate="tanh"` の公開（v1はconfig非公開で"none"固定）。
- CNBlock内蔵downsample（stride>1やchannel変更をCNBlock自身が扱う機能）。
- `head_init_scale`（論文のfine-tuning用パラメータ。スクラッチ学習では1.0固定=no-opのため省略）。
- 深さ方向のブロック数・チャンネル幅を config から動的に生成する仕組み（既存プロジェクトの流儀通り、個々のブロックインスタンスを手動列挙する）。

## 影響ファイル

| ファイル | 変更 |
|---|---|
| `core/anet-core/src/nn_modules.cpp` | 既存`LayerNormModuleFactory`に`eps`追加／`LayerNorm2dModule`・`Factory`新設／`CNBlockConfig`・`CNBlockModule`・`CNBlockModuleFactory`新設／`InitNN()`に`repo.Register`2件追加 |
| `core/anet-core/src/nn_test.cpp` | `[nn][layernorm2d]` `[nn][cnblock]` の単体テスト追加（ResBlockテスト`:916-974`直後が自然な挿入位置） |
| `apps/runner/config/ImageCls.txt` | 設計方針E の全ブロック定義＋`net.branch.ConvNeXtT`＋有効化切り替えを追加 |

## 受け入れ基準

1. **ビルド緑**（x64-Debug、`anet-core-test`）。
2. **LayerNorm2d**: forward後も `[N,C,H,W]` の shape不変。チャンネル軸のみ正規化されること（`input.mean(1)`が意図通り機能し、H/W方向では正規化されないこと）を数値確認。`num_channels`未指定または0以下で`ANET_SYSTEM_ERROR`。
3. **既存LayerNorm回帰なし**: `eps`未指定時に従来通り1e-5になり、LunarLander/Hybrid/ViT branchの既存挙動が変わらないこと。
4. **CNBlock**: forward後も `[N,C,H,W]` のshape不変（H,Wも不変、stride=1のため）。`GetCurrentConfigData`が`channels`/`kernel_size`/`ffn_expand_ratio`/`layerscale_init`/`droppath_rate`/`norm_type`を正しくダンプすること。
5. **CNBlock droppath**: `eval()`では`droppath_rate>0`でも`residual + gamma*branch`（スケール無し）と一致（no-op）。`train()` + `droppath_rate≈0.99`で出力がほぼ`residual`のみ。
6. **CNBlock LayerScale**: 初回forward直後の`gamma`パラメータが`layerscale_init`（既定1e-6）で埋まっていること。`layerscale_init<=0`のとき`gamma`が生成されず、出力に乗算が適用されないこと。
7. **CNBlock Factory検証**: `cn.channels<=0`または`cn.droppath_rate`が`[0.0,1.0)`外で`ANET_SYSTEM_ERROR`。
8. **CNBlock channel不一致**: 入力channel数が`cn.channels`と異なる場合、初回forwardで`ANET_SYSTEM_ERROR`（shape不一致でのtensor add破綻より先にfail-fastする）。
9. **CNBlock norm_type=none**: `cn.norm_type=none`で内部normステップがスキップされ、`layernorm2d`時と出力が変わること（norm有/無のcode-path確認）。未知の`norm_type`で`ANET_SYSTEM_ERROR`。
10. **ImageCls実branch起動確認**: `net.branch.[main_feature].$ = net.branch.ConvNeXtT` を有効化した状態でImageCls runnerが起動し、クラッシュせず学習ループが回ること（PRD027+029実装済み前提。`@cn_dp01`が解決され18個の補間droppathが`config_data.txt`に載る。精度評価は本書スコープ外、次のタスク）。

## テスト項目リスト

1. LayerNorm2d forward正規化軸確認（C軸のみ、H,W不変）
2. LayerNorm2d config必須検証（num_channels未指定/0でエラー）
3. 既存LayerNorm eps後方互換確認（未指定時1e-5、指定時に反映）
4. CNBlock forward shape不変（`[N,C,H,W]`→`[N,C,H,W]`）
5. CNBlock config dump（5キー確認）
6. CNBlock droppath eval no-op
7. CNBlock droppath train時のshortcut近似一致（droppath_rate≈0.99）
8. CNBlock LayerScale gamma初期値確認（layerscale_init反映、<=0で無効化）
9. CNBlock Factory範囲外エラー（channels<=0、droppath_rate範囲外）
10. CNBlock in_channels不一致エラー
11. CNBlock norm_type=noneでnormスキップ確認（layernorm2dと出力差、未知norm_typeでエラー）

## 正直なリスク / 注意

- **自前実装したTruncNormal（PRD027側）とPyTorch公式実装の数値的等価性は未検証**。アルゴリズム自体はPyTorch/timm公式と同一（erfinv法）だが、実装後に統計検証（生成テンソルのstd・範囲）が必要。これはPRD027側の受け入れ基準で担保する。
- **PRD029（config_profile機構）への依存**: droppath は `@cn_dp01` マーカーで書くため、029未実装だとマーカーが解決されず起動時エラーになる。027・029・028の3点セット実装が前提（実装・評価は束ねる）。CNBlock定義は channel別4定義 + `(*N)` に集約され、18個手書きは不要。
- **depthwise self-containedな実装は、将来汎用Conv2dモジュールに`groups`パラメータが追加された場合に重複する可能性がある**。現時点ではCNBlock専用の自己完結実装を優先する（判断14）。
- **ConvNeXt-Tinyはパラメータ数がResNet18ish系より大きい**（標準構成で約28M）。学習時間・メモリ使用量がResNet18ish比で増える見込みだが、具体的な比較は実行後にユーザーが判断する（本書スコープ外）。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe "[cnblock]"
core\anet-core\bin\Debug\anet-core-test.exe "[layernorm2d]"
core\anet-core\bin\Debug\anet-core-test.exe
git diff --check
```

- 機能確認: `net.branch.[main_feature].$ = net.branch.ConvNeXtT` を有効化したImageCls runnerを起動し、`runs/<name>/config/config_data.txt` に `cn.channels`/`cn.droppath_rate`等の新キーがground truthとして載ること、学習ループがクラッシュせず数ステップ回ることを確認。
- 精度評価（eval accuracy、ResNet18ish系との比較）はユーザーが別途実施する（本書スコープ外）。

## 後続

1. 本書の受け入れ基準達成後、ImageClsでの精度評価（ResNet18ish系との比較、[[feedback_compare_with_run_variance]]の通りseed違い複数runの終盤平均で判断）はユーザーが実施。
2. DropMerge向けCNBlock branch設計（stem/downsample調整を含む）を後継PRDとして着手する（判断7）。
3. 必要に応じてConvNeXt v2（GRN対応）を別PRDで検討する。
