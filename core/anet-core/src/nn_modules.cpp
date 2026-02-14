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

    Conv1dModule(int64_t out_channels, int64_t kernel_size, int64_t stride, int64_t padding, const WeightInitConfig& init_config)
        : out_channels_(out_channels), kernel_size_(kernel_size), stride_(stride), padding_(padding), init_config_(init_config)
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
};

class Conv2dModule : public NetworkModule {
public:
    Conv2dModule(int64_t out_channels, int64_t kernel_size, int64_t stride, int64_t padding, const WeightInitConfig& init_config)
		: out_channels_(out_channels), kernel_size_(kernel_size), stride_(stride), padding_(padding), init_config_(init_config)
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
};

/// ElementwiseAdd (For Skip Connection via Tags)
class ElementwiseAddModule : public NetworkModule {
public:
    explicit ElementwiseAddModule(int split_count) : split_count_(split_count)
    {
        if (split_count_ < 2) {
			LOG::warn() << "ElementwiseAddModule: split_count should be >= 2. Given: " << split_count_;
        }
    }

    torch::Tensor forward(torch::Tensor x)
    {
        anet::ProfileRange r("ElementwiseAddModule::forward");

        // 入力 x は (Batch, TotalChannels, ...) と結合されている前提
        if (split_count_ <= 1) return x;

        // split_count_ で等分割する。
        // ※ 各要素のチャンネル数が同じであることが前提
        auto chunks = x.chunk(split_count_, 1); // dim=1
        if (chunks.size() != split_count_) {
            ANET_SYSTEM_ERROR(
                "ElementwiseAdd: Input channels cannot be split into "
                << split_count_ <<  " equal parts. Total channels=" << x.size(1));
        }

        // 加算
        torch::Tensor sum = chunks[0];
        for (size_t i = 1; i < chunks.size(); ++i) {
            sum = sum + chunks[i];
        }
        return sum;
    }

    torch::Tensor Forward(torch::Tensor input) override
    {
        return forward(input);
    }
private:
    int split_count_;
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
    torch::Tensor Forward(torch::Tensor input) override
    {
        return impl_->forward(input);
    }
private:
    torch::nn::LeakyReLU impl_{ nullptr };
};


// ===========================================================================
// SpatialEmbedderModule (For 2D Grid + Scalar Input)
// ===========================================================================

struct SpatialEmbedderConfig {
    int scalar_dim = 0;
    int grid_width = 0;
    int grid_height = 0;
    int num_classes = 0;
};

class SpatialEmbedderModule : public NetworkModule {
public:
    SpatialEmbedderModule(const SpatialEmbedderConfig& config)
        : config_(config)
    {
        ANET_CHECK(config_.scalar_dim >= 0);
        ANET_CHECK(config_.grid_width > 0);
        ANET_CHECK(config_.grid_height > 0);
        ANET_CHECK(config_.num_classes > 0);
    }

    torch::Tensor Forward(torch::Tensor input) override {
        anet::ProfileRange r("SpatialEmbedderModule::Forward");

        // Input: (Batch, Stack, Features) or (Batch, Features)
        // Features = scalar_dim + (grid_w * grid_h)

        auto shape = input.sizes().vec();
        const int64_t batch_size = shape[0];
        const int64_t input_feature_dim = shape.back();

        // 次元チェック
        const int64_t expected_grid_dim = (int64_t)config_.grid_width * config_.grid_height;
        const int64_t expected_total_dim = (int64_t)config_.scalar_dim + expected_grid_dim;

        // ※Input次元が一致しない場合はエラーにする（あるいは柔軟に対応するかだが、基本は厳密に）
        if (input_feature_dim != expected_total_dim) {
            ANET_SYSTEM_ERROR("SpatialEmbedder: Input dimension mismatch. Expected "
                << expected_total_dim << " (Scalar:" << config_.scalar_dim << " + Grid:" << expected_grid_dim << ")"
                << " but got " << input_feature_dim);
        }

        // Stack次元の有無を確認
        // dim=3なら (Batch, Stack, Feat)、dim=2なら (Batch, Feat)
        bool has_stack = (input.dim() == 3);
        int64_t stack_count = has_stack ? shape[1] : 1;

        // 処理のために (TotalBatch, Feat) にFlattenする
        // TotalBatch = Batch * Stack
        torch::Tensor flat_input = input;
        if (has_stack) {
            flat_input = input.reshape({ -1, input_feature_dim });
        }

        // 1. Split (Scalar / Grid)
        // scalar_part: (N, scalar_dim)
        // grid_part:   (N, grid_dim)
        std::vector<torch::Tensor> parts;
        if (config_.scalar_dim > 0) {
            parts = torch::split(flat_input, { (int64_t)config_.scalar_dim, expected_grid_dim }, /*dim=*/1);
        } else {
            // スカラーがない場合
            parts = { torch::Tensor(), flat_input };
        }
        auto& scalar_part = parts[0];
        auto& grid_part = parts[1];

        // 2. Grid -> One-Hot Image
        // (N, grid_dim) -> (N, H, W)
        auto grid_2d = grid_part.view({ -1, config_.grid_height, config_.grid_width });

        // float(ID) -> long -> one_hot
        // ※ IDが num_classes 以上だとクラッシュするので注意（Env側の保証が必要）
        auto grid_long = grid_2d.to(torch::kLong);
        auto grid_onehot = torch::one_hot(grid_long, config_.num_classes); // (N, H, W, Classes)

        // Permute: (N, H, W, C) -> (N, C, H, W)
        auto grid_img = grid_onehot.permute({ 0, 3, 1, 2 }).to(torch::kFloat32);

        // 3. Scalar -> Broadcast Image
        torch::Tensor scalar_img;
        if (config_.scalar_dim > 0) {
            // (N, scalar_dim) -> (N, scalar_dim, 1, 1) -> (N, scalar_dim, H, W)
            // expandはメモリコピーを行わないView操作なので高速
            scalar_img = scalar_part.view({ -1, config_.scalar_dim, 1, 1 })
                .expand({ -1, config_.scalar_dim, config_.grid_height, config_.grid_width });
        }

        // 4. Concat
        torch::Tensor out_img;
        if (config_.scalar_dim > 0) {
            out_img = torch::cat({ grid_img, scalar_img }, /*dim=*/1);
        } else {
            out_img = grid_img;
        }

        // 5. Stack次元の統合 (Stack as Channel)
        // 現在: (Batch*Stack, TotalChannels, H, W)
        // 目標: (Batch, Stack*TotalChannels, H, W)
        if (has_stack) {
            int64_t total_channels = config_.num_classes + config_.scalar_dim;

            // (B*S, C, H, W) -> (B, S, C, H, W)
            out_img = out_img.view({ batch_size, stack_count, total_channels, config_.grid_height, config_.grid_width });

            // (B, S, C, H, W) -> (B, S*C, H, W)
            out_img = out_img.reshape({ batch_size, stack_count * total_channels, config_.grid_height, config_.grid_width });
        }

        return out_img;
    }

private:
    SpatialEmbedderConfig config_;
};


// ===========================================================================
//  BatchNorm2d Module
// ===========================================================================

// BatchNorm2d Module
class BatchNorm2dModule : public NetworkModule {
public:
    explicit BatchNorm2dModule(int64_t num_features)
        : num_features_(num_features) {
    }

    torch::Tensor Forward(torch::Tensor input) override {
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
        Config(const anet::ConfigData& config_data) : anet::Config("") {
            ANET_READ_CONFIG(config_data, num_features);
        }
    };
public:
    std::shared_ptr<NetworkModule> CreateModule(const anet::ConfigData& config_data, const ModuleContext& context) const override {
        Config config(config_data);
        return std::make_shared<BatchNorm2dModule>(config.num_features);
    }
};


// ===========================================================================
//  ResBlock Module
// ===========================================================================

struct ResBlockConfig {
    int channels = 64;
    int kernel_size = 3;
    int stride = 1;
    std::string activation = "relu"; // "relu" (default) or "silu" / "swish"
};

class ResBlockModule : public NetworkModule {
private:
    enum class ActType { ReLU, SiLU };
public:
    ResBlockModule(const ResBlockConfig& config, const WeightInitConfig& init_config)
        : config_(config), init_config_(init_config)
    {
        if (config_.activation == "SiLU" || config_.activation == "silu" ||
            config_.activation == "Swish" || config_.activation == "swish") {
            act_type_ = ActType::SiLU;
        } else {
            act_type_ = ActType::ReLU;
        }
    }

    torch::Tensor Forward(torch::Tensor input) override {
        anet::ProfileRange r("ResBlockModule::Forward");

        // Lazy Initialization
        if (!conv1_) {
            anet::ProfileRange r2("ResBlockModule::Forward.init");

            int64_t in_channels = input.size(1);

            // --- Main Path (Conv1) ---

            torch::nn::Conv2dOptions conv1_opts(in_channels, config_.channels, config_.kernel_size);
            conv1_opts.stride(config_.stride);
            conv1_opts.padding(config_.kernel_size / 2);
            conv1_opts.bias(false);

            conv1_ = register_module("conv1", torch::nn::Conv2d(conv1_opts));
            bn1_ = register_module("bn1", torch::nn::BatchNorm2d(config_.channels));

            // --- Main Path (Conv2) ---

            // Conv2は常にStride=1 (サイズ維持)
            torch::nn::Conv2dOptions conv2_opts(config_.channels, config_.channels, config_.kernel_size);
            conv2_opts.stride(1);
            conv2_opts.padding(config_.kernel_size / 2);
            conv2_opts.bias(false);

            conv2_ = register_module("conv2", torch::nn::Conv2d(conv2_opts));
            bn2_ = register_module("bn2", torch::nn::BatchNorm2d(config_.channels));

            // --- Downsample Path (Projection Shortcut) ---

            // サイズが変わる(stride > 1) または チャンネルが変わる場合、
            // input側も合わせるために 1x1 Conv を通す
            if (config_.stride > 1 || in_channels != config_.channels) {
                torch::nn::Conv2dOptions ds_opts(in_channels, config_.channels, 1); // Kernel=1 (1x1 Conv)
                ds_opts.stride(config_.stride); // Main Pathと同じStrideで縮小
                ds_opts.padding(0);             // 1x1なのでPadding不要
                ds_opts.bias(false);

                downsample_conv_ = register_module("ds_conv", torch::nn::Conv2d(ds_opts));
                downsample_bn_ = register_module("ds_bn", torch::nn::BatchNorm2d(config_.channels));

                // Init Downsample
                // 1x1 Convへの初期化も適用
                downsample_conv_->to(input.device(), input.scalar_type());
                downsample_bn_->to(input.device(), input.scalar_type());
                WeightInitializer::Initialize(downsample_conv_, init_config_);
            }

            // Init Main Path
            auto device = input.device();
            auto dtype = input.scalar_type();
            conv1_->to(device, dtype); bn1_->to(device, dtype);
            conv2_->to(device, dtype); bn2_->to(device, dtype);

            WeightInitializer::Initialize(conv1_, init_config_);
            WeightInitializer::Initialize(conv2_, init_config_);
        }

        // --- Forwarding ---

        // Conv1
        anet::ProfileRange r3("ResBlockModule::Forward.conv1");
        torch::Tensor out = conv1_->forward(input);
        out = bn1_->forward(out);

        // Activation 1
        out = Activate(out);

        // Conv2
        anet::ProfileRange r4("ResBlockModule::Forward.conv2", r3);
        out = conv2_->forward(out);
        out = bn2_->forward(out);

        //  Shortcut Path
        anet::ProfileRange r5("ResBlockModule::Forward.downsample", r4);
        torch::Tensor residual = input;
        if (downsample_conv_) { // 次元合わせが必要な場合
            residual = downsample_conv_->forward(residual);
            residual = downsample_bn_->forward(residual);
        }

        // Shortcut-Connection & Activate
        anet::ProfileRange r6("ResBlockModule::Forward.fin", r5);
        out += residual;

        // Activation 2
        out = Activate(out);

        return out;
    }
private:
    inline torch::Tensor Activate(const torch::Tensor& x) const {
        if (act_type_ == ActType::SiLU) {
            return torch::silu(x);
        }
        return torch::relu(x);
    }
private:
    ResBlockConfig config_;
    WeightInitConfig init_config_;

    ActType act_type_ = ActType::ReLU;

    torch::nn::Conv2d conv1_{ nullptr };
    torch::nn::BatchNorm2d bn1_{ nullptr };
    torch::nn::Conv2d conv2_{ nullptr };
    torch::nn::BatchNorm2d bn2_{ nullptr };

    // Shortcut調整用 (不要な合はnullptr)
    torch::nn::Conv2d downsample_conv_{ nullptr };
    torch::nn::BatchNorm2d downsample_bn_{ nullptr };
};

// ResBlockModuleFactory
class ResBlockModuleFactory final : public NetworkModuleFactory {
private:
    struct Config : anet::Config {
        ResBlockConfig res;
        WeightInitConfig init;

        Config(const anet::ConfigData& config_data) : anet::Config("") {
            ANET_READ_CONFIG(config_data, res.channels);
            ANET_READ_CONFIG(config_data, res.kernel_size);
            ANET_READ_CONFIG(config_data, res.stride);
            ANET_READ_CONFIG(config_data, init.mode);
            ANET_READ_CONFIG(config_data, init.manual_gain);
            ANET_READ_CONFIG(config_data, init.nonlinearity);
        }
    };
public:
    std::shared_ptr<NetworkModule> CreateModule(const anet::ConfigData& config_data, const ModuleContext& context) const override {
        Config config(config_data);
        return std::make_shared<ResBlockModule>(config.res, config.init);
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
};

struct PermuteConfig {
    std::vector<int64_t> dims;
};


class LinearModuleFactory final : public NetworkModuleFactory {
private:
    struct Config : anet::Config {
        LinearConfig linear;
        WeightInitConfig init;

        Config(const anet::ConfigData& config_data) : anet::Config("") {
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

        Config(const anet::ConfigData& config_data) : anet::Config("") {
            ANET_READ_CONFIG(config_data, conv.out_channels);
            ANET_READ_CONFIG(config_data, conv.kernel_size);
            ANET_READ_CONFIG(config_data, conv.stride);
            ANET_READ_CONFIG(config_data, conv.padding);
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
            config.conv.out_channels, config.conv.kernel_size, config.conv.stride, config.conv.padding, config.init);
    }
};

class Conv2dModuleFactory final : public NetworkModuleFactory {
private:
    struct Config : anet::Config {
        ConvConfig conv;
        WeightInitConfig init;

        Config(const anet::ConfigData& config_data) : anet::Config("")
        {
            ANET_READ_CONFIG(config_data, conv.out_channels);
            ANET_READ_CONFIG(config_data, conv.kernel_size);
            ANET_READ_CONFIG(config_data, conv.stride);
            ANET_READ_CONFIG(config_data, conv.padding);
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
            config.conv.out_channels, config.conv.kernel_size, config.conv.stride, config.conv.padding, config.init);
    }
};

class ElementwiseAddModuleFactory final : public NetworkModuleFactory {
public:
    std::shared_ptr<NetworkModule> CreateModule(const anet::ConfigData& config_data, const ModuleContext& context) const override 
    {
        ANET_CHECK_MSG(!context.input_tags.empty(), "ElementwiseAdd: input_tags should not be epmty.");
		int split_count = (int)context.input_tags.size();   // 分割数として入力タグ数を取得

        return std::make_shared<ElementwiseAddModule>(split_count);
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
            ANET_READ_CONFIG(config_data, embed.scalar_dim);
            ANET_READ_CONFIG(config_data, embed.grid_width);
            ANET_READ_CONFIG(config_data, embed.grid_height);
            ANET_READ_CONFIG(config_data, embed.num_classes);
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
    repo.Register("Add", std::make_shared<ElementwiseAddModuleFactory>());
    repo.Register("Flatten", std::make_shared<FlattenModuleFactory>());
    repo.Register("Permute", std::make_shared<PermuteModuleFactory>());

	// 活性化関数モジュール登録
    repo.Register("ReLU", std::make_shared<ReLUModuleFactory>());
    repo.Register("GELU", std::make_shared<GELUModuleFactory>());
    repo.Register("SiLU", std::make_shared<SiLUModuleFactory>());
    repo.Register("Mish", std::make_shared<MishModuleFactory>());
    repo.Register("LeakyReLU", std::make_shared<LeakyReLUModuleFactory>());

	// その他モジュール登録
    repo.Register("Linear", std::make_shared<LinearModuleFactory>());
    repo.Register("Conv1d", std::make_shared<Conv1dModuleFactory>());
    repo.Register("Conv2d", std::make_shared<Conv2dModuleFactory>());
    repo.Register("SpatialEmbedder", std::make_shared<SpatialEmbedderModuleFactory>());
    repo.Register("BatchNorm2d", std::make_shared<BatchNorm2dModuleFactory>());
    repo.Register("ResBlock", std::make_shared<ResBlockModuleFactory>());

    //RegisterNetworkModuleFactory<Module>("Linear");
 }

