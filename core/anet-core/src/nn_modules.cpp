// nn_module_impl.cpp

#include <numeric>
#include "nn_impl.hpp"
#include "anet/log.hpp"
#include "anet/profile.hpp"


using namespace anet::nn;
namespace LOG = anet::log;


torch::nn::init::NonlinearityType anet::nn::GetNonlinearityType(const std::string& name)
{
    if (name == "relu") return torch::kReLU;
    if (name == "linear") return torch::kLinear;
    if (name == "tanh") return torch::kTanh;
    if (name == "leaky_relu") return torch::kLeakyReLU;
    return torch::kReLU;
}


// ===========================================================================
// Standard Module Implementations
// ===========================================================================

// Lazy Linear Implementation
class LinearModule : public NetworkModule {
public:
    LinearModule(int64_t out_features, bool bias, const WeightInitConfig& init_config)
        : out_features_(out_features), with_bias_(bias), init_config_(init_config)
    {
    }

    torch::Tensor forward(torch::Tensor x)
    {
        anet::ProfileRange r("LinearModule::forward");

        if (!linear) {
            // 初回実行時に入力次元数を自動取得

            // 入力次元数
            const int64_t in_features = x.size(-1);
            ANET_LOG_DEBUG("in_features=" << in_features);

            // モジュール生成と登録
            torch::nn::LinearOptions opts(in_features, out_features_);
            opts.bias(with_bias_);
            linear = register_module("linear", torch::nn::Linear(opts));

            // デバイス同期 (入力と同じデバイスへ移動)
            linear->to(x.device(), x.scalar_type());

			// 重み初期化
            WeightInitializer::Initialize(linear, init_config_);
        }
        return linear->forward(x);
    }

    // NetworkModule interface override
    torch::Tensor Forward(torch::Tensor input) override
    {
        return forward(input);
    }
private:
    WeightInitConfig init_config_;
    torch::nn::Linear linear{ nullptr };
    int64_t out_features_;
    bool with_bias_;
};

// Lazy Conv1d Implementation
class Conv1dModule : public NetworkModule {
public:

    Conv1dModule(int64_t out_channels, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation, const WeightInitConfig& init_config)
        : out_channels_(out_channels), kernel_size_(kernel_size), stride_(stride), padding_(padding), dilation_(dilation), init_config_(init_config)
    {
    }

    torch::Tensor forward(torch::Tensor x)
    {
        anet::ProfileRange r("Conv1dModule::forward");

        if (!conv) {
            // 初回実行時に入力チャンネル数(in_channels)を自動取得

            // x: (Batch, Channel, H, W) -> in_channels is dim 1
            const int64_t in_channels = x.size(1);
            ANET_LOG_DEBUG("in_channels=" << in_channels);

            // モジュール生成と登録
            torch::nn::Conv1dOptions opts(in_channels, out_channels_, kernel_size_);
            opts.stride(stride_);
            opts.padding(padding_);
            opts.dilation(dilation_);
            conv = register_module("conv", torch::nn::Conv1d(opts));

            // 重み初期化
            conv->to(x.device(), x.scalar_type());

            // 重み初期化
            WeightInitializer::Initialize(conv, init_config_);
        }
        return conv->forward(x);
    }

    torch::Tensor Forward(torch::Tensor input) override {
        return forward(input);
    }
private:
    WeightInitConfig init_config_;
    torch::nn::Conv1d conv{ nullptr };
    int64_t out_channels_;
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
    int64_t dilation_;
};

class Conv2dModule : public NetworkModule {
public:
    Conv2dModule(int64_t out_channels, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation, const WeightInitConfig& init_config)
		: out_channels_(out_channels), kernel_size_(kernel_size), stride_(stride), padding_(padding), dilation_(dilation), init_config_(init_config)
    {
    }

    torch::Tensor forward(torch::Tensor x)
    {
        anet::ProfileRange r("Conv2dModule::forward");

        if (!conv_) {
            // 初回の処理で入力チャンネル数(in_channels)を自動取得

            // x: (Batch, Channel, Length) -> in_channels is dim 1
            const int64_t in_channels = x.size(1);

            // モジュール生成と登録
            torch::nn::Conv2dOptions opts(in_channels, out_channels_, kernel_size_);
            opts.stride(stride_);
            opts.padding(padding_);
			opts.dilation(dilation_);
            conv_ = register_module("conv2d", torch::nn::Conv2d(opts));

            // 重み初期化
            conv_->to(x.device(), x.scalar_type());

            // 重み初期化
            WeightInitializer::Initialize(conv_, init_config_);
        }
        return conv_->forward(x);
    }

    torch::Tensor Forward(torch::Tensor input) override {
        return forward(input);
    }
private:
    WeightInitConfig init_config_;
    torch::nn::Conv2d conv_{ nullptr };
    int64_t out_channels_;
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
    int64_t dilation_;
};

/// Permute Module (Transpose axes)
/// e.g. dims=[0, 2, 1] -> (Batch, Time, Feat) -> (Batch, Feat, Time)
class PermuteModule : public NetworkModule {
public:
    explicit PermuteModule(std::vector<int64_t> dims) : dims_(std::move(dims))
    {
    }

    torch::Tensor forward(torch::Tensor x)
    {
        anet::ProfileRange r("PermuteModule::forward");
        return x.permute(dims_);
    }
    torch::Tensor Forward(torch::Tensor input) override
    {
        return forward(input);
    }
private:
    std::vector<int64_t> dims_;
};

/// Flatten Module
class FlattenModule : public NetworkModule {
public:
    torch::Tensor forward(torch::Tensor x)
    {
        anet::ProfileRange r("FlattenModule::forward");

        return x.flatten(1);
    }
    torch::Tensor Forward(torch::Tensor input) override
    {
        return forward(input);
    }
};

// ===========================================================================
// 活性化関数Modules
// ===========================================================================

/// ReLU Module
class ReLUModule : public NetworkModule {
public:
    bool IsConv2dVisualizable() const override { return true; }

    torch::Tensor forward(torch::Tensor x)
    {
        anet::ProfileRange r("ReLUModule::forward");

        return torch::relu(x);
    }

    torch::Tensor Forward(torch::Tensor input) override
    {
        return forward(input);
    }
};

// --- GELU Module ---
class GELUModule : public NetworkModule {
public:
    explicit GELUModule(const std::string& approximate)
    {
        torch::nn::GELUOptions opts;
        opts.approximate(approximate); // "none" or "tanh"
        impl_ = register_module("gelu", torch::nn::GELU(opts));
    }

    bool IsConv2dVisualizable() const override { return true; }

    torch::Tensor Forward(torch::Tensor input) override
    {
        return impl_->forward(input);
    }
private:
    torch::nn::GELU impl_{ nullptr };
};

// --- SiLU (Swish) Module ---
class SiLUModule : public NetworkModule {
public:
    SiLUModule()
    {
        impl_ = register_module("silu", torch::nn::SiLU());
    }

    bool IsConv2dVisualizable() const override { return true; }

    torch::Tensor Forward(torch::Tensor input) override
    {
        return impl_->forward(input);
    }
private:
    torch::nn::SiLU impl_{ nullptr };
};

// --- Mish Module ---
class MishModule : public NetworkModule {
public:
    MishModule()
    {
        impl_ = register_module("mish", torch::nn::Mish());
    }

    bool IsConv2dVisualizable() const override { return true; }

    torch::Tensor Forward(torch::Tensor input) override
    {
        return impl_->forward(input);
    }
private:
    torch::nn::Mish impl_{ nullptr };
};

// --- LeakyReLU Module ---
class LeakyReLUModule : public NetworkModule {
public:
    explicit LeakyReLUModule(double negative_slope)
    {
        torch::nn::LeakyReLUOptions opts;
        opts.negative_slope(negative_slope);
        impl_ = register_module("leaky_relu", torch::nn::LeakyReLU(opts));
    }

    bool IsConv2dVisualizable() const override { return true; }

    torch::Tensor Forward(torch::Tensor input) override
    {
        return impl_->forward(input);
    }
private:
    torch::nn::LeakyReLU impl_{ nullptr };
};


// ===========================================================================
//  BatchNorm2d Module
// ===========================================================================

// BatchNorm2d Module
class BatchNorm2dModule : public NetworkModule {
public:
    explicit BatchNorm2dModule(int64_t num_features)
        : num_features_(num_features)
    {
    }

    torch::Tensor Forward(torch::Tensor input) override
    {
        // Lazy Init
        if (!bn_) {
            torch::nn::BatchNorm2dOptions opts(num_features_);
            bn_ = register_module("bn", torch::nn::BatchNorm2d(opts));
            bn_->to(input.device(), input.scalar_type());
        }
        return bn_->forward(input);
    }
private:
    int64_t num_features_;
    torch::nn::BatchNorm2d bn_{ nullptr };
};

// Factory
class BatchNorm2dModuleFactory final : public NetworkModuleFactory {
private:
    struct Config : anet::Config {
        int num_features = 0;
        Config(const anet::ConfigData& config_data) : anet::Config("")
        {
            ANET_READ_CONFIG(config_data, num_features);
        }
    };
public:
    std::shared_ptr<NetworkModule> CreateModule(const anet::ConfigData& config_data, const ModuleContext& context) const override
    {
        Config config(config_data);
        return std::make_shared<BatchNorm2dModule>(config.num_features);
    }
};


// ===========================================================================
//  GroupNormModule Module
// ===========================================================================

class GroupNormModule : public NetworkModule {
public:
    GroupNormModule(int64_t num_groups, int64_t num_channels)
        : num_groups_(num_groups), num_channels_(num_channels)
    {
    }

    torch::Tensor Forward(torch::Tensor input) override
    {
        // Lazy Init
        if (!impl_) {
            torch::nn::GroupNormOptions opts(num_groups_, num_channels_);
            // GroupNormは学習可能パラメータ(Affine)を持つのがデフォルト
            opts.affine(true);
            impl_ = register_module("gn", torch::nn::GroupNorm(opts));
            impl_->to(input.device(), input.scalar_type());
        }
        return impl_->forward(input);
    }
private:
    int64_t num_groups_;
    int64_t num_channels_;
    torch::nn::GroupNorm impl_{ nullptr };
};

class GroupNormModuleFactory final : public NetworkModuleFactory {
private:
    struct Config : anet::Config {
        int num_groups = 32;
        int num_channels = 0;

        Config(const anet::ConfigData& config_data) : anet::Config("")
        {
            ANET_READ_CONFIG(config_data, num_groups);
            ANET_READ_CONFIG(config_data, num_channels);
        }
    };
public:
    std::shared_ptr<NetworkModule> CreateModule(const anet::ConfigData& config_data, const ModuleContext& context) const override
    {
        Config config(config_data);
        if (config.num_channels <= 0) {
            // GroupNormはチャンネル数がグループ数で割り切れる必要がある。明示指定を必須とする
            ANET_SYSTEM_ERROR("GroupNormModule: num_channels is 0.");
        }
        return std::make_shared<GroupNormModule>(config.num_groups, config.num_channels);
    }
};


// ===========================================================================
//  ResBlock Module
// ===========================================================================

struct ResBlockConfig {
    int channels = 64;
    int kernel_size = 3;
    int padding = -1;
    int stride = 1;
    int dilation = 1;
    std::string activation = "silu"; // "relu" or "silu"(default)  / "swish"
    std::string activation_mode = "post"; // "post" (v1) or "pre" (v2)
    std::string norm_type = "none"; // "none", "batch", "group"
    int group_norm_groups = 32;
    bool conv1_bias = true;        // Norm無しならtrue必須。None有りならFalse推奨。
    bool conv2_bias = true;        // ZeroInitするならTrue推奨
};

/// ResNet Basic Block
class ResBlockModule : public NetworkModule {
private:
    enum class ActType { ReLU, SiLU };
    enum class ActMode { Post, Pre };
public:
    ResBlockModule(const ResBlockConfig& config, const WeightInitConfig& init1_config, const WeightInitConfig& init2_config, const WeightInitConfig& init_ds_config)
        : config_(config), init1_config_(init1_config), init2_config_(init2_config), init_ds_config_(init_ds_config)
    {
        // 活性化関数設定取得
        if (config_.activation == "SiLU" || config_.activation == "silu" ||
            config_.activation == "Swish" || config_.activation == "swish") {
            act_type_ = ActType::SiLU;
        } else {
            act_type_ = ActType::ReLU;
        }

        // モード設定取得
        if (config_.activation_mode == "pre" || config_.activation_mode == "Pre") {
            act_mode_ = ActMode::Pre;
        } else {
            act_mode_ = ActMode::Post;
        }
    }

    bool IsConv2dVisualizable() const override { return true; }

    torch::Tensor Forward(torch::Tensor input) override
    {
        anet::ProfileRange r("ResBlockModule::Forward");

        // Lazy Initialization
        if (!conv1_) {
            anet::ProfileRange r2("ResBlockModule::Forward.init");

            auto device = input.device();
            auto dtype = input.scalar_type();
            int64_t in_channels = input.size(1);
            int padding = config_.padding < 0 ? (config_.dilation * (config_.kernel_size - 1) / 2) : config_.padding;

            // ------------------------------------------------
            // Main Path (Conv1)
            // ------------------------------------------------
            torch::nn::Conv2dOptions conv1_opts(in_channels, config_.channels, config_.kernel_size);
            conv1_opts.stride(config_.stride);
            conv1_opts.padding(padding);
            conv1_opts.dilation(config_.dilation);
            conv1_opts.bias(config_.conv1_bias);
            conv1_ = register_module("conv1", torch::nn::Conv2d(conv1_opts));
            conv1_->to(device, dtype);
            WeightInitializer::Initialize(conv1_, init1_config_);

            norm1_ = CreateAndRegisterNorm("norm1", config_.channels);

            // ------------------------------------------------
            // Main Path (Conv2)
            // ------------------------------------------------
            torch::nn::Conv2dOptions conv2_opts(config_.channels, config_.channels, config_.kernel_size);
            conv2_opts.stride(1);
            conv2_opts.padding(padding);
            conv2_opts.dilation(config_.dilation);
            conv2_opts.bias(config_.conv2_bias);
            conv2_ = register_module("conv2", torch::nn::Conv2d(conv2_opts));
            conv2_->to(device, dtype);

            WeightInitializer::Initialize(conv2_, init2_config_);

            norm2_ = CreateAndRegisterNorm("norm2", config_.channels);

            // ------------------------------------------------
            // Shortcut Path
            // ------------------------------------------------
            if (config_.stride > 1 || in_channels != config_.channels) {
                torch::nn::Conv2dOptions ds_opts(in_channels, config_.channels, 1);
                ds_opts.stride(config_.stride);
                ds_opts.padding(0);
                ds_opts.bias(false); // Shortcutは通常Biasなし(直後にAddされるため)
                downsample_conv_ = register_module("ds_conv", torch::nn::Conv2d(ds_opts));
                downsample_conv_->to(device, dtype);
                WeightInitializer::Initialize(downsample_conv_, init_ds_config_);

                // Shortcut Norm (Conv1x1 -> Norm)
                norm_ds_ = CreateAndRegisterNorm("ds_norm", config_.channels);
            }
        }

        // --- Forwarding ---
        if (act_mode_ == ActMode::Pre) {

            // ==================================================
            // Pre-Activation (ResNet v2)
            // ==================================================
            anet::ProfileRange r3("ResBlockModule::Forward.pre_act");

            // 共通の Pre-Activation (Norm -> Act)
            torch::Tensor pre_act = input;
            if (norm1_) pre_act = norm1_->Forward(pre_act);
            pre_act = Activate(pre_act);

            // Shortcut Path
            torch::Tensor residual = input;
            if (downsample_conv_) {
                // 次元が変わる場合、Pre-Actされた値から1x1 Convで射影する（v2の標準）
                residual = downsample_conv_->forward(pre_act);
                if (norm_ds_) residual = norm_ds_->Forward(residual);
            }

            // Main Path

            // Conv1
            torch::Tensor out = conv1_->forward(pre_act);

            // Norm2 -> Act -> Conv2
            if (norm2_) out = norm2_->Forward(out);
            out = Activate(out);
            out = conv2_->forward(out);

            // Add (最後のアクティベーション無し)
            return out + residual;
        } else {
            // ==================================================
            // Post-Activation (ResNet v1)
            // ==================================================
            anet::ProfileRange r3("ResBlockModule::Forward.post_act");

            // Block 1: Conv -> Norm -> Act
            torch::Tensor out = conv1_->forward(input);
            if (norm1_) out = norm1_->Forward(out);
            out = Activate(out);

            // Block 2: Conv -> Norm
            out = conv2_->forward(out);
            if (norm2_) out = norm2_->Forward(out);

            // Down-sample
            torch::Tensor residual = input;
        if (downsample_conv_) { // 次元合わせが必要な場合の1x1Conv
                residual = downsample_conv_->forward(residual);
                if (norm_ds_) residual = norm_ds_->Forward(residual);
            }

            // Add & Act
            out += residual;
            out = Activate(out);

            return out;
        }
    }
private:
    std::shared_ptr<NetworkModule> CreateAndRegisterNorm(const std::string& name, int64_t channels)
    {
        std::shared_ptr<NetworkModule> mod = nullptr;

        if (config_.norm_type == "batch") {
            mod = std::make_shared<BatchNorm2dModule>(channels);
        } else if (config_.norm_type == "group") {
            mod = std::make_shared<GroupNormModule>(config_.group_norm_groups, channels);
        }

        if (mod) {
            // パラメータ登録のため register_module を経由させる
            register_module(name, mod);
        }
        return mod;
    }

    inline torch::Tensor Activate(const torch::Tensor& x) const
    {
        if (act_type_ == ActType::SiLU) {
            return torch::silu(x);
        }
        return torch::relu(x);
    }
private:
    ResBlockConfig config_;
    WeightInitConfig init1_config_;
    WeightInitConfig init2_config_;
    WeightInitConfig init_ds_config_;

    ActType act_type_ = ActType::ReLU;
	ActMode act_mode_ = ActMode::Post;

    // Conv2d
    torch::nn::Conv2d conv1_{ nullptr };
    torch::nn::Conv2d conv2_{ nullptr };
    torch::nn::Conv2d downsample_conv_{ nullptr };

    // Normalization Layers
    std::shared_ptr<NetworkModule> norm1_{ nullptr };
    std::shared_ptr<NetworkModule> norm2_{ nullptr };
    std::shared_ptr<NetworkModule> norm_ds_{ nullptr };
};

// ResBlockModuleFactory
class ResBlockModuleFactory final : public NetworkModuleFactory {
private:
    struct Config : anet::Config {
        ResBlockConfig res;
        WeightInitConfig init1;
        WeightInitConfig init2;
        WeightInitConfig init_ds;

        Config(const anet::ConfigData& config_data) : anet::Config("")
        {
            init1.mode = 2;			    // Default: He       
            init2.mode = 4;             // Default: ZeroInit
            init_ds.mode = 2;			// Default: He       

            ANET_READ_CONFIG(config_data, res.channels);
            ANET_READ_CONFIG(config_data, res.kernel_size);
            ANET_READ_CONFIG(config_data, res.stride);
            ANET_READ_CONFIG(config_data, res.padding);
            ANET_READ_CONFIG(config_data, res.dilation);
            ANET_READ_CONFIG(config_data, res.conv1_bias);
            ANET_READ_CONFIG(config_data, res.conv2_bias);
            ANET_READ_CONFIG(config_data, res.activation);
            ANET_READ_CONFIG(config_data, res.activation_mode);
            ANET_READ_CONFIG(config_data, res.norm_type);
            ANET_READ_CONFIG(config_data, res.group_norm_groups);

            ANET_READ_CONFIG(config_data, init1.mode);
            ANET_READ_CONFIG(config_data, init1.manual_gain);
            ANET_READ_CONFIG(config_data, init1.nonlinearity);
            ANET_READ_CONFIG(config_data, init1.constant_val);

            ANET_READ_CONFIG(config_data, init2.mode);
            ANET_READ_CONFIG(config_data, init2.manual_gain);
            ANET_READ_CONFIG(config_data, init2.nonlinearity);
            ANET_READ_CONFIG(config_data, init2.constant_val);

            ANET_READ_CONFIG(config_data, init_ds.mode);
            ANET_READ_CONFIG(config_data, init_ds.manual_gain);
            ANET_READ_CONFIG(config_data, init_ds.nonlinearity);
            ANET_READ_CONFIG(config_data, init_ds.constant_val);
        }
    };
public:
    std::shared_ptr<NetworkModule> CreateModule(const anet::ConfigData& config_data, const ModuleContext& context) const override
    {
        Config config(config_data);
        return std::make_shared<ResBlockModule>(config.res, config.init1, config.init2, config.init_ds);
    }
};


// ===========================================================================
//  LayerNorm Module
// ===========================================================================

class LayerNormModule : public NetworkModule {
public:
    explicit LayerNormModule(int64_t normalized_shape)
    {
        torch::nn::LayerNormOptions opts({ normalized_shape });
        ln_ = register_module("ln", torch::nn::LayerNorm(opts));
    }

    torch::Tensor Forward(torch::Tensor input) override
    {
        anet::ProfileRange r("LayerNormModule::Forward");

        // Lazy Init for device/dtype transfer
        if (!initialized_) {
            ln_->to(input.device(), input.scalar_type());
            initialized_ = true;
        }
        return ln_->forward(input);
    }
private:
    bool initialized_ = false;
    torch::nn::LayerNorm ln_{ nullptr };
};

class LayerNormModuleFactory final : public NetworkModuleFactory {
private:
    struct Config : anet::Config {
        int normalized_shape = 0;

        Config(const anet::ConfigData& config_data) : anet::Config("") {
            ANET_READ_CONFIG(config_data, normalized_shape);
        }
    };
public:
    std::shared_ptr<NetworkModule> CreateModule(const anet::ConfigData& config_data, const ModuleContext& context) const override
    {
        Config config(config_data);
        if (config.normalized_shape <= 0) {
            ANET_SYSTEM_ERROR("LayerNormModule: 'normalized_shape' must be strictly positive.");
        }
        return std::make_shared<LayerNormModule>(config.normalized_shape);
    }
};

// ===========================================================================
//  SpatialPositionalEmbedding2D Module
// ===========================================================================


/// @brief CNNの2次元特徴マップに空間位置情報(Positional Embedding)を付与し、
///        Transformer用の1次元シーケンスデータへ変換するブリッジモジュール。
///
/// 【入出力のテンソル形状】
/// - Input : [Batch, Channels, Height, Width]  (CNNの出力)
/// - Output: [Batch, SequenceLength, Channels] (Transformerの入力)
///           ※ SequenceLength = Height * Width
///
/// 【使用上の注意点】
/// 1. 直前の層（通常は 1x1 Conv 等）において、出力チャンネル数(Channels)を
///    後続の TransformerEncoder の `d_model` と完全に一致させておく必要がある。
///    (例: Transformerのd_modelが32なら、直前のConvのout_channelsも32にする)
/// 2. 本モジュールは Lazy Initialization（遅延初期化）を採用しています。
///    初回の順伝播時に入力テンソルの形状から Height, Width, Channels を自動取得し、
///    必要なサイズのパラメータを自己構築するため、Configでの設定値は一切不要。
///
/// 【内部処理】
/// X座標用とY座標用に独立した学習可能なベクトル(Embedding)を保持し、ブロードキャストに
/// よって特徴マップの各ピクセルへ一括加算。その後、空間次元を平坦化(Flatten)し、
/// 軸を入れ替える(Transpose)ことで、Transformerが読めるシーケンス配列を生成する。
class SpatialPositionalEmbedding2DModule : public NetworkModule {
public:
    SpatialPositionalEmbedding2DModule() = default;

    torch::Tensor Forward(torch::Tensor input) override
    {
        anet::ProfileRange r("SpatialPositionalEmbedding2DModule::Forward");

        // 初回実行時のLazy Initialization
        if (!initialized_) {
            // 入力: [Batch, d_model(Channels), Height, Width]
            int64_t d_model = input.size(1);
            int64_t height = input.size(2);
            int64_t width = input.size(3);

            // X座標用とY座標用のEmbeddingを独立して学習可能なパラメータとして登録
            y_embed_ = register_parameter("y_embed", torch::randn({ height, d_model }) * 0.02f);
            x_embed_ = register_parameter("x_embed", torch::randn({ width, d_model }) * 0.02f);

            // デバイス同期
            this->to(input.device(), input.scalar_type());
            initialized_ = true;

            LOG::info() << "SpatialPositionalEmbedding2D initialized with Height:" << height
                << " Width:" << width << " d_model:" << d_model;
        }

        // --- 位置情報の加算 ---
        auto y_emb = y_embed_.transpose(0, 1).unsqueeze(0).unsqueeze(-1);
        auto x_emb = x_embed_.transpose(0, 1).unsqueeze(0).unsqueeze(2);
        auto out = input + y_emb + x_emb;

        // --- Transformer用シーケンスへの変形 ---
        // [Batch, C, H, W] -> [Batch, H*W, C]
        return out.flatten(2).transpose(1, 2);
    }
private:
    bool initialized_ = false;
    torch::Tensor y_embed_;
    torch::Tensor x_embed_;
};

class SpatialPositionalEmbedding2DFactory final : public NetworkModuleFactory {
public:
    std::shared_ptr<NetworkModule> CreateModule(const anet::ConfigData& config_data, const ModuleContext& context) const override {
        // パラメータ不要のため即座に生成
        return std::make_shared<SpatialPositionalEmbedding2DModule>();
    }
};


// ===========================================================================
// SpatialEmbedderModule (Vector to Spatial Image)
// ===========================================================================

struct SpatialEmbedderConfig {
    int grid_width = 0;
    int grid_height = 0;
};

class SpatialEmbedderModule : public NetworkModule {
public:
    SpatialEmbedderModule(const SpatialEmbedderConfig& config)
        : config_(config)
    {
        ANET_CHECK(config_.grid_width > 0);
        ANET_CHECK(config_.grid_height > 0);
    }

    bool IsConv2dVisualizable() const override { return true; }

    torch::Tensor Forward(torch::Tensor input) override
    {
        anet::ProfileRange r("SpatialEmbedderModule::Forward");

        // Input: (Batch, Features) または FrameStack時 (Batch, Stack, Features)
        auto shape = input.sizes().vec();
        const int64_t batch_size = shape[0];
        const int64_t input_feature_dim = shape.back();

        bool has_stack = (input.dim() == 3);
        int64_t stack_count = has_stack ? shape[1] : 1;

        // 処理のために (TotalBatch, Feat) にFlatten
        torch::Tensor flat_input = has_stack ? input.reshape({ -1, input_feature_dim }) : input;

        // (N, Feat) -> (N, Feat, 1, 1) -> (N, Feat, H, W) へBroadcast
        torch::Tensor out_img = flat_input.view({ -1, input_feature_dim, 1, 1 })
            .expand({ -1, input_feature_dim, config_.grid_height, config_.grid_width });

        // Stack次元の統合 (Stack as Channel)
        if (has_stack) {
            // (B*S, C, H, W) -> (B, S, C, H, W) -> (B, S*C, H, W)
            out_img = out_img.view({ batch_size, stack_count, input_feature_dim, config_.grid_height, config_.grid_width });
            out_img = out_img.reshape({ batch_size, stack_count * input_feature_dim, config_.grid_height, config_.grid_width });
        }

        // float32 キャストして返す
        return out_img.to(torch::kFloat32);
    }

private:
    SpatialEmbedderConfig config_;
};


// ===========================================================================
//  TransformerEncoder Module
// ===========================================================================

/// libtorchの制約（Post-LN固定、[SeqLen, Batch, d_model] 形式の入力）を突破するためカスタムのTransformer層を用意
class CustomTransformerEncoderLayer : public torch::nn::Module {
public:
    CustomTransformerEncoderLayer(int64_t d_model, int64_t nhead, int64_t dim_feedforward, bool norm_first, const std::string& activation)
        : norm_first_(norm_first)
    {
        // Multihead Attention
        torch::nn::MultiheadAttentionOptions mha_opts(d_model, nhead);
        mha_ = register_module("self_attn", torch::nn::MultiheadAttention(mha_opts));

        // Feed Forward Network (FFN)
        linear1_ = register_module("linear1", torch::nn::Linear(d_model, dim_feedforward));
        linear2_ = register_module("linear2", torch::nn::Linear(dim_feedforward, d_model));

        // Layer Normalizations
        norm1_ = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ d_model })));
        norm2_ = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ d_model })));

        //  Activation Function
        std::string act_lower = activation;
        std::transform(act_lower.begin(), act_lower.end(), act_lower.begin(), ::tolower);
        use_gelu_ = (act_lower == "gelu");
    }

    torch::Tensor forward(torch::Tensor src)
    {
        // srcの期待形状: [Batch, SeqLen, d_model]
        torch::Tensor x = src;

        if (norm_first_) {
            // ==========================================
            // Pre-LN：強化学習では推奨（表現力が僅かに落ちるが学習が極めて安定。現代のデファクトスタンダード）
            // ==========================================

            // --- Attention Block ---
            torch::Tensor x_norm = norm1_->forward(x);

            // libtorchのMHAは [SeqLen, Batch, d_model] しか受け付けないため明示的に転置
            torch::Tensor x_norm_t = x_norm.transpose(0, 1);

            // MultiheadAttention (Query, Key, Value)
            auto mha_out = std::get<0>(mha_->forward(x_norm_t, x_norm_t, x_norm_t));

            // 転置して戻し、Skip Connection (Add)
            x = x + mha_out.transpose(0, 1);

            // --- FFN Block ---
            x_norm = norm2_->forward(x);
            torch::Tensor ffn_out = linear1_->forward(x_norm);
            ffn_out = use_gelu_ ? torch::gelu(ffn_out) : torch::relu(ffn_out);
            ffn_out = linear2_->forward(ffn_out);

            // Skip Connection (Add)
            x = x + ffn_out;
        } else {
            // ==========================================
            // Post-LN：オリジナルTransformer相当（最終的な性能は高いが不安定）
            // ==========================================

            torch::Tensor x_t = x.transpose(0, 1);
            auto mha_out = std::get<0>(mha_->forward(x_t, x_t, x_t));
            x = norm1_->forward(x + mha_out.transpose(0, 1));

            torch::Tensor ffn_out = linear1_->forward(x);
            ffn_out = use_gelu_ ? torch::gelu(ffn_out) : torch::relu(ffn_out);
            ffn_out = linear2_->forward(ffn_out);
            x = norm2_->forward(x + ffn_out);
        }

        return x;
    }

private:
    bool norm_first_;
    bool use_gelu_;
    torch::nn::MultiheadAttention mha_{ nullptr };
    torch::nn::Linear linear1_{ nullptr };
    torch::nn::Linear linear2_{ nullptr };
    torch::nn::LayerNorm norm1_{ nullptr };
    torch::nn::LayerNorm norm2_{ nullptr };
};

struct TransformerConfig {
    int d_model = 32;
    int nhead = 4;
    int num_layers = 2;
    int dim_feedforward = 128;
    bool norm_first = true;             /// Pre-LN default
    std::string activation = "gelu";    /// relu / gelu
};

// --- TransformerEncoderModule 本体 ---
class TransformerEncoderModule : public NetworkModule {
public:
    TransformerEncoderModule(const TransformerConfig& config)
        : config_(config)
    {
        ANET_CHECK_MSG(config_.d_model % config_.nhead == 0, "TransformerEncoder: d_model must be divisible by nhead.");

        // カスタムレイヤーをループで生成・登録
        for (int i = 0; i < config_.num_layers; ++i) {
            auto layer = std::make_shared<CustomTransformerEncoderLayer>(
                config_.d_model, config_.nhead, config_.dim_feedforward, config_.norm_first, config_.activation
            );
            layers_.push_back(register_module("layer_" + std::to_string(i), layer));
        }

        // Pre-LNの場合、ネットワークの最後を締める正規化
        if (config_.norm_first) {
            torch::nn::LayerNormOptions ln_opts({ config_.d_model });
            norm_ = register_module("norm", torch::nn::LayerNorm(ln_opts));
        }
    }

    torch::Tensor Forward(torch::Tensor input) override
    {
        anet::ProfileRange r("TransformerEncoderModule::Forward");

        if (!initialized_) {
            // 初回の形状チェック
            int64_t input_dim = input.size(2); // [B, SeqLen, d_model]
            if (input_dim != config_.d_model) {
                ANET_SYSTEM_ERROR("TransformerEncoder: Dimension mismatch! "
                    << "Configured d_model is " << config_.d_model << ", but received input with dimension " << input_dim << ". "
                    << "Please check the 'out_channels' of the preceding layer.");
            }
            this->to(input.device(), input.scalar_type());
            initialized_ = true;
        }

        torch::Tensor out = input;

        // レイヤーを順番に適用
        for (auto& layer : layers_) {
            out = layer->forward(out);
        }

        // 最終正規化
        if (norm_) {
            out = norm_->forward(out);
        }

        return out;
    }
private:
    TransformerConfig config_;
    bool initialized_ = false;
    std::vector<std::shared_ptr<CustomTransformerEncoderLayer>> layers_;
    torch::nn::LayerNorm norm_{ nullptr };
};

class TransformerEncoderModuleFactory final : public NetworkModuleFactory {
private:
    struct Config : anet::Config {
        TransformerConfig tf;

        Config(const anet::ConfigData& config_data) : anet::Config("") {
            ANET_READ_CONFIG(config_data, tf.d_model);
            ANET_READ_CONFIG(config_data, tf.nhead);
            ANET_READ_CONFIG(config_data, tf.num_layers);
            ANET_READ_CONFIG(config_data, tf.dim_feedforward);
            ANET_READ_CONFIG(config_data, tf.norm_first);
            ANET_READ_CONFIG(config_data, tf.activation);
        }
    };
public:
    std::shared_ptr<NetworkModule> CreateModule(const anet::ConfigData& config_data, const ModuleContext& context) const override
    {
        Config config(config_data);
        return std::make_shared<TransformerEncoderModule>(config.tf);
    }
};


// ===========================================================================
//  Global Average Pooling 1D Module (GAP1D)
// ===========================================================================

class GlobalAveragePooling1DModule : public NetworkModule {
public:
    torch::Tensor Forward(torch::Tensor input) override
    {
        anet::ProfileRange r("GlobalAveragePooling1DModule::Forward");
        // [Batch, SeqLen, d_model] の SeqLen (dim=1) を平均して潰す
        return input.mean(/*dim=*/1);
    }
};

class GlobalAveragePooling1DFactory final : public NetworkModuleFactory {
public:
    std::shared_ptr<NetworkModule> CreateModule(const anet::ConfigData& config_data, const ModuleContext& context) const override
    {
        return std::make_shared<GlobalAveragePooling1DModule>();
    }
};

// ===========================================================================
//  CLS Token Append Module
// ===========================================================================

class ClsTokenAppendModule : public NetworkModule {
public:
    torch::Tensor Forward(torch::Tensor input) override
    {
        anet::ProfileRange r("ClsTokenAppendModule::Forward");

        int64_t batch_size = input.size(0);
        int64_t d_model = input.size(2);

        if (!initialized_) {
            // ダミートークンを初期化
            cls_token_ = register_parameter("cls_token", torch::randn({ 1, 1, d_model }) * 0.02f);
            this->to(input.device(), input.scalar_type());
            initialized_ = true;
        }

        // [1, 1, d_model] -> [Batch, 1, d_model] に拡張
        auto cls_expanded = cls_token_.expand({ batch_size, -1, -1 });

        // 先頭にくっつけて出力: [Batch, 1 + SeqLen, d_model]
        return torch::cat({ cls_expanded, input }, /*dim=*/1);
    }
private:
    bool initialized_ = false;
    torch::Tensor cls_token_;
};

class ClsTokenAppendFactory final : public NetworkModuleFactory {
public:
    std::shared_ptr<NetworkModule> CreateModule(const anet::ConfigData& config_data, const ModuleContext& context) const override
    {
        return std::make_shared<ClsTokenAppendModule>();
    }
};

// ===========================================================================
//  CLS Token Extract Module
// ===========================================================================

class ClsTokenExtractModule : public NetworkModule {
public:
    torch::Tensor Forward(torch::Tensor input) override
    {
        anet::ProfileRange r("ClsTokenExtractModule::Forward");

        // [Batch, 1 + SeqLen, d_model] の 0番目 (先頭) を抽出する
        return input.select(/*dim=*/1, /*index=*/0);
    }
};

class ClsTokenExtractFactory final : public NetworkModuleFactory {
public:
    std::shared_ptr<NetworkModule> CreateModule(const anet::ConfigData& config_data, const ModuleContext& context) const override
    {
        return std::make_shared<ClsTokenExtractModule>();
    }
};


// ===========================================================================
//  NetworkModuleFactory
// ===========================================================================

struct LinearConfig {
    int out_features = 128;
    bool bias = true;
};

struct ConvConfig {
    int out_channels = 128;
    int kernel_size = 3;
    int stride = 1;
    int padding = 0;
    int dilation = 1;
};

struct PermuteConfig {
    std::vector<int64_t> dims;
};


class LinearModuleFactory final : public NetworkModuleFactory {
private:
    struct Config : anet::Config {
        LinearConfig linear;
        WeightInitConfig init;

        Config(const anet::ConfigData& config_data) : anet::Config("")
        {
            ANET_READ_CONFIG(config_data, linear.out_features);
            ANET_READ_CONFIG(config_data, linear.bias);
            ANET_READ_CONFIG(config_data, init.mode);
            ANET_READ_CONFIG(config_data, init.manual_gain);
            ANET_READ_CONFIG(config_data, init.nonlinearity);
        }
    };
public:
    std::shared_ptr<NetworkModule> CreateModule(const anet::ConfigData& config_data, const ModuleContext& context) const override
    {
        Config config(config_data);
        return std::make_shared<LinearModule>(config.linear.out_features, config.linear.bias, config.init);
    }
};

class Conv1dModuleFactory final : public NetworkModuleFactory {
private:
    struct Config : anet::Config {
        ConvConfig conv;
        WeightInitConfig init;

        Config(const anet::ConfigData& config_data) : anet::Config("")
        {
            init.mode = 2; // Default: He
            ANET_READ_CONFIG(config_data, conv.out_channels);
            ANET_READ_CONFIG(config_data, conv.kernel_size);
            ANET_READ_CONFIG(config_data, conv.stride);
            ANET_READ_CONFIG(config_data, conv.padding);
            ANET_READ_CONFIG(config_data, conv.dilation);
            ANET_READ_CONFIG(config_data, init.mode);
            ANET_READ_CONFIG(config_data, init.manual_gain);
            ANET_READ_CONFIG(config_data, init.nonlinearity);
        }
    };
public:
    std::shared_ptr<NetworkModule> CreateModule(const anet::ConfigData& config_data, const ModuleContext& context) const override
    {
        Config config(config_data);
        return std::make_shared<Conv1dModule>(
            config.conv.out_channels, config.conv.kernel_size, config.conv.stride, config.conv.padding, config.conv.dilation, config.init);
    }
};

class Conv2dModuleFactory final : public NetworkModuleFactory {
private:
    struct Config : anet::Config {
        ConvConfig conv;
        WeightInitConfig init;

        Config(const anet::ConfigData& config_data) : anet::Config("")
        {
			init.mode = 2; // Default: He
            ANET_READ_CONFIG(config_data, conv.out_channels);
            ANET_READ_CONFIG(config_data, conv.kernel_size);
            ANET_READ_CONFIG(config_data, conv.stride);
            ANET_READ_CONFIG(config_data, conv.padding);
            ANET_READ_CONFIG(config_data, conv.dilation);
            ANET_READ_CONFIG(config_data, init.mode);
            ANET_READ_CONFIG(config_data, init.manual_gain);
            ANET_READ_CONFIG(config_data, init.nonlinearity);
        }
    };
public:
    std::shared_ptr<NetworkModule> CreateModule(const anet::ConfigData& config_data, const ModuleContext& context) const override
    {
        Config config(config_data);
        return std::make_shared<Conv2dModule>(
            config.conv.out_channels, config.conv.kernel_size, config.conv.stride, config.conv.padding, config.conv.dilation, config.init);
    }
};

class PermuteModuleFactory final : public NetworkModuleFactory {
private:
    struct Config : anet::Config {
        PermuteConfig permute;
        Config(const anet::ConfigData& config_data) : anet::Config("") {
            ANET_READ_CONFIG(config_data, permute.dims);
        }
    };
public:
    std::shared_ptr<NetworkModule> CreateModule(const anet::ConfigData& config_data, const ModuleContext& context) const override
    {
        Config config(config_data);
        if (config.permute.dims.empty()) {
            ANET_SYSTEM_ERROR("PermuteModule: 'dims' is empty.");
        }
        return std::make_shared<PermuteModule>(config.permute.dims);
    }
};

class SpatialEmbedderModuleFactory final : public NetworkModuleFactory {
private:
    struct Config : anet::Config {
        SpatialEmbedderConfig embed;
        Config(const anet::ConfigData& config_data) : anet::Config("") {
            ANET_READ_CONFIG(config_data, embed.grid_width);
            ANET_READ_CONFIG(config_data, embed.grid_height);
        }
    };
public:
    std::shared_ptr<NetworkModule> CreateModule(const anet::ConfigData& config_data, const ModuleContext& context) const override
    {
        Config config(config_data);
        return std::make_shared<SpatialEmbedderModule>(config.embed);
    }
};

class FlattenModuleFactory final : public NetworkModuleFactory {
public:
    std::shared_ptr<NetworkModule> CreateModule(const anet::ConfigData& config_data, const ModuleContext& context) const override
    {
        return std::make_shared<FlattenModule>();
    }
};

class ReLUModuleFactory final : public NetworkModuleFactory {
public:
    std::shared_ptr<NetworkModule> CreateModule(const anet::ConfigData& config_data, const ModuleContext& context) const override
    {
        return std::make_shared<ReLUModule>();
    }
};
class GELUModuleFactory final : public NetworkModuleFactory {
private:

    struct Config : anet::Config {
        /// none:標準正規分布の累積分布関数（厳密解、デフォルト）
		/// tanh: 計算コストが抑えられた近似関数
        std::string approximate = "none";


        Config(const anet::ConfigData& config_data) : anet::Config("") {
            ANET_READ_CONFIG(config_data, approximate);
        }
    };

public:
    std::shared_ptr<NetworkModule> CreateModule(const anet::ConfigData& config_data, const ModuleContext& ctx) const override {
        Config config(config_data);
        return std::make_shared<GELUModule>(config.approximate);
    }
};

class SiLUModuleFactory final : public NetworkModuleFactory {
public:
    std::shared_ptr<NetworkModule> CreateModule(const anet::ConfigData& config_data, const ModuleContext& ctx) const override {
        return std::make_shared<SiLUModule>();
    }
};

class MishModuleFactory final : public NetworkModuleFactory {
public:
    std::shared_ptr<NetworkModule> CreateModule(const anet::ConfigData& config_data, const ModuleContext& ctx) const override {
        return std::make_shared<MishModule>();
    }
};

// --- LeakyReLU Factory ---
class LeakyReLUModuleFactory final : public NetworkModuleFactory {
private:

    struct Config : anet::Config {
        /// 負の領域(x < 0)における直線の傾き係数。 例: 0.01 (default) の場合、負の値は 0.01倍 されて出力される
        double negative_slope = 0.01;

        Config(const anet::ConfigData& config_data) : anet::Config("") {
            ANET_READ_CONFIG(config_data, negative_slope);
        }
    };

public:
    std::shared_ptr<NetworkModule> CreateModule(const anet::ConfigData& config_data, const ModuleContext& ctx) const override {
        Config config(config_data);
        return std::make_shared<LeakyReLUModule>(config.negative_slope);
    }
};

 void anet::nn::InitNN()
 {
    auto& repo = NetworkModuleRepository::Instance();

	// 基本モジュール登録
    repo.Register("Flatten", std::make_shared<FlattenModuleFactory>());
    repo.Register("Permute", std::make_shared<PermuteModuleFactory>());

	// 活性化関数モジュール登録
    repo.Register("ReLU", std::make_shared<ReLUModuleFactory>());
    repo.Register("GELU", std::make_shared<GELUModuleFactory>());
    repo.Register("SiLU", std::make_shared<SiLUModuleFactory>());
    repo.Register("Mish", std::make_shared<MishModuleFactory>());
    repo.Register("LeakyReLU", std::make_shared<LeakyReLUModuleFactory>());

    // 正規化・Pooling登録
    repo.Register("GroupNorm", std::make_shared<GroupNormModuleFactory>());
    repo.Register("LayerNorm", std::make_shared<LayerNormModuleFactory>());
    repo.Register("BatchNorm2d", std::make_shared<BatchNorm2dModuleFactory>());
    repo.Register("GAP1D", std::make_shared<GlobalAveragePooling1DFactory>());

    // データ加工系モジュール登録
    repo.Register("SpatialEmbedder", std::make_shared<SpatialEmbedderModuleFactory>());
    repo.Register("SpatialPositionalEmbedding2D", std::make_shared<SpatialPositionalEmbedding2DFactory>());
    
    // レイヤー系モジュール登録
    repo.Register("Linear", std::make_shared<LinearModuleFactory>());
    repo.Register("Conv1d", std::make_shared<Conv1dModuleFactory>());
    repo.Register("Conv2d", std::make_shared<Conv2dModuleFactory>());
    repo.Register("ResBlock", std::make_shared<ResBlockModuleFactory>());
    repo.Register("TransformerEncoder", std::make_shared<TransformerEncoderModuleFactory>());

    // Tokenモジュール登録
    repo.Register("ClsAppend", std::make_shared<ClsTokenAppendFactory>());
    repo.Register("ClsExtract", std::make_shared<ClsTokenExtractFactory>());

    //RegisterNetworkModuleFactory<Module>("Linear");
 }

