// nn_heads.cpp

#include "nn_heads.hpp"
#include "anet/profile.hpp"


using namespace anet::nn;


// ===========================================================================
// PassThroughHead
// ===========================================================================

PassThroughHead::PassThroughHead(const std::string& output_key)
    : output_key_(output_key)
{
    // 重みを持たないため register_module 等は不要
}

anet::TensorDict PassThroughHead::Forward(torch::Tensor feature_vector)
{
    anet::ProfileRange r("PassThroughHead::Forward");

    anet::TensorDict out;
    out.Set(output_key_, feature_vector);
    return out;
}

std::optional<anet::TensorFunction> PassThroughHead::GetTensorFunction(const std::string& key)
{
    if (key == "forward" || key == output_key_) {
        return [](const torch::Tensor& x) { return x; };
    }
    return std::nullopt;
}


// ===========================================================================
// PassThroughHeadFactory
// ===========================================================================

PassThroughHeadFactory::PassThroughHeadFactory(const std::string& output_key, int64_t expected_input_dim)
    : output_key_(output_key), expected_input_dim_(expected_input_dim)
{
}

std::shared_ptr<NetworkHead> PassThroughHeadFactory::CreateHead(int64_t input_dim) const
{
    ANET_ASSERT_MSG(input_dim == expected_input_dim_,
        "Head body's output dim does not match exptencted. expected=" << expected_input_dim_ << " actual=" << input_dim);

    // 次元数に依存しないため、引数は無視してキー名だけ渡す
    return std::make_shared<PassThroughHead>(output_key_);
}

// ===========================================================================
// ConfigurableHead
// ===========================================================================

ConfigurableHead::ConfigurableHead(int64_t feature_dim, std::map<std::string, std::shared_ptr<NetworkStruct>> branches)
    : branches_(std::move(branches))
{
    // 全てのブランチをサブモジュールとして登録（PyTorchにパラメータを管理させる）
    // ※ パススルー（空のNetworkStruct）の場合、パラメータを持たないが無害なのでそのまま登録する
    for (const auto& [key, branch] : branches_) {
        register_module("branch_" + key, branch);
    }
}

anet::TensorDict ConfigurableHead::Forward(torch::Tensor feature_vector)
{
    anet::ProfileRange r("ConfigurableHead::Forward");
    anet::TensorDict out;

    // すべてのブランチに特徴量を並行して流し込み、それぞれのキーでDictに詰める
    for (const auto& [key, branch] : branches_) {
        out.Set(key, branch->Forward(feature_vector));
    }

    return out;
}

std::optional<anet::TensorFunction> ConfigurableHead::GetTensorFunction(const std::string& key)
{
    auto it = branches_.find(key);
    if (it != branches_.end()) {
        // 特定のブランチだけを実行するラムダ関数を返す
        auto branch = it->second; // shared_ptr をコピー
        return [branch](const torch::Tensor& x) { return branch->Forward(x); }; 
    }
    return std::nullopt;
}


// ===========================================================================
// ConfigurableHeadFactory
// ===========================================================================

ConfigurableHeadFactory::ConfigurableHeadFactory(
    const NetworkConfig& config,
    const std::map<std::string, std::string>& branch_mapping)
    : config_(config), branch_mapping_(branch_mapping)
{
}

std::shared_ptr<NetworkHead> ConfigurableHeadFactory::CreateHead(int64_t input_dim) const
{
    std::map<std::string, std::shared_ptr<NetworkStruct>> branches;

    for (const auto& [out_key, conf_key] : branch_mapping_) {
        // Configの additional_structures の中から、指定されたキーの文字列を探す
        auto it = config_.additional_structures.find(conf_key);

        // フェイルファスト: 要求された設定キーが存在しない場合は即座に落とす
        ANET_ASSERT_MSG(it != config_.additional_structures.end(),
            ("Branch structure not found in config.additional_structures: " + conf_key).c_str());

        const std::string& struct_str = it->second;

        // ブランチのビルド
        branches[out_key] = NetworkStructBuilder::Build(config_, struct_str);
    }

    return std::make_shared<ConfigurableHead>(input_dim, branches);
}

// ===========================================================================
// LinearHead (Standard Q-Network Head)
// ===========================================================================
// 構造: Feature -> Linear -> Output
// 用途: DQN, QR-DQN(出力次元を調整して使用)

class LinearHead : public NetworkHead {
public:
    LinearHead(int64_t in_features, int64_t out_features, const WeightInitConfig& init_config)
    {
        // ヘッドは入力次元(in_features)が確定してから作られるため、Lazyである必要はない
        torch::nn::LinearOptions opts(in_features, out_features);
        opts.bias(true);

        linear_ = register_module("linear", torch::nn::Linear(opts));

        // 初期化
        WeightInitializer::Initialize(linear_, init_config);
    }

    anet::TensorDict Forward(torch::Tensor feature_vector) override
    {
        anet::ProfileRange r("LinearHead::Forward");

        anet::TensorDict out;
        out.Set("q", linear_->forward(feature_vector)); // (B, A)
        return out;
    }

    std::optional<anet::TensorFunction> GetTensorFunction(const std::string& key) override
    {
        // Q値 (B, A)
        if (key == "forward" || key == "forward.q" || key == "q_values") {
            return [this](const torch::Tensor& x) {
                torch::NoGradGuard no_grad;
                return linear_->forward(x);
                };
        }
        return std::nullopt;
    }

private:
    torch::nn::Linear linear_{ nullptr };
};

// ===========================================================================
// DuelingHead (V + A)
// ===========================================================================

class DuelingHead : public NetworkHead {
public:
    DuelingHead(int64_t in_features, int64_t action_dim, const WeightInitConfig& init_config)
        : action_dim_(action_dim)
    {
        // Value Stream (H -> 1)
        torch::nn::LinearOptions v_opts(in_features, 1);
        v_opts.bias(true);
        value_ = register_module("value", torch::nn::Linear(v_opts));

        // Advantage Stream (H -> A)
        torch::nn::LinearOptions a_opts(in_features, action_dim);
        a_opts.bias(true);
        adv_ = register_module("adv", torch::nn::Linear(a_opts));

        WeightInitializer::Initialize(value_, init_config);
        WeightInitializer::Initialize(adv_, init_config);
    }

    anet::TensorDict Forward(torch::Tensor x) override
    {
        anet::ProfileRange r("DuelingHead::Forward");

        // v,a,q を取得
        auto v = value_->forward(x); // (B, 1)
        auto a = adv_->forward(x);   // (B, A)
        auto a_mean = a.mean(/*dim=*/1, /*keepdim=*/true);
        auto q = v + (a - a_mean);

        // 結果を返す
        anet::TensorDict out;
        out.Set("q", q);
        out.Set("v", v);
        out.Set("a", a);
        return out;
    }

    std::optional<anet::TensorFunction> GetTensorFunction(const std::string& key) override
    {
        // Q値 (B, A)
        if (key == "forward" || key == "forward.q" || key == "q_values") {
            return [this](const torch::Tensor& x) {
                torch::NoGradGuard no_grad;
                auto v = value_->forward(x);
                auto a = adv_->forward(x);
                return v + (a - a.mean(1, true));
                };
        }
        // Value (B, 1)
        if (key == "forward.v" || key == "v_values") {
            return [this](const torch::Tensor& x) {
                torch::NoGradGuard no_grad;
                return value_->forward(x);
                };
        }
        // Advantage (B, A) - 生の値
        if (key == "forward.a" || key == "a_values") {
            return [this](const torch::Tensor& x) {
                torch::NoGradGuard no_grad;
                return adv_->forward(x);
                };
        }
        return std::nullopt;
    }
private:
    torch::nn::Linear value_{ nullptr };
    torch::nn::Linear adv_{ nullptr };
    int64_t action_dim_;
};

// ===========================================================================
// QuantileHead (QR-DQN Plain)
// ===========================================================================

class QuantileHead : public NetworkHead {
public:
    QuantileHead(int64_t in_features, int64_t action_dim, int64_t num_quantiles, const WeightInitConfig& init_config)
        : action_dim_(action_dim), num_quantiles_(num_quantiles)
    {
        // Output: Actions * Quantiles (Flat)
        int64_t out_features = action_dim * num_quantiles;

        torch::nn::LinearOptions opts(in_features, out_features);
        opts.bias(true);
        linear_ = register_module("linear", torch::nn::Linear(opts));

        WeightInitializer::Initialize(linear_, init_config);
    }

    anet::TensorDict Forward(torch::Tensor x) override
    {
        anet::ProfileRange r("QuantileHead::Forward");

        // 生の出力（B, A*N)
        auto flat = linear_->forward(x); // (B, A*N)

        // 分布に変形
        auto batch_size = flat.size(0);
        auto q_dist = flat.view({ batch_size, action_dim_, num_quantiles_ });

        // Q分布の平均としてQ値を算出
        auto q = q_dist.mean(2);

        // 結果を返す
        anet::TensorDict out;
        out.Set("q_dist", q_dist);  // (B, A, N)
        out.Set("q", q);            // (B, A)
        return out;
    }

    std::optional<anet::TensorFunction> GetTensorFunction(const std::string& key) override
    {
        // Q値 (B, A) - 分位点の平均
        if (key == "forward" || key == "forward.q" || key == "q_values") {
            return [this](const torch::Tensor& x) {
                torch::NoGradGuard no_grad;
                auto flat = linear_->forward(x);
                // (B, A*N) -> (B, A, N) -> mean -> (B, A)
                return flat.view({ flat.size(0), action_dim_, num_quantiles_ }).mean(2);
                };
        }
        // 分布 (B, A, N)
        if (key == "forward.dist" || key == "distributions") {
            return [this](const torch::Tensor& x) {
                torch::NoGradGuard no_grad;
                auto flat = linear_->forward(x);
                return flat.view({ flat.size(0), action_dim_, num_quantiles_ });
                };
        }
        return std::nullopt;
    }
private:
    torch::nn::Linear linear_{ nullptr };
    int64_t action_dim_;
    int64_t num_quantiles_;
};

// ===========================================================================
// QuantileDuelingHead (QR-DQN + Dueling)
// ===========================================================================

class QuantileDuelingHead : public NetworkHead {
public:
    QuantileDuelingHead(int64_t in_features, int64_t action_dim, int64_t num_quantiles, const WeightInitConfig& init_config)
        : action_dim_(action_dim), num_quantiles_(num_quantiles)
    {
        // Value Stream: H -> 1 * N
        value_ = register_module("value", torch::nn::Linear(in_features, num_quantiles));

        // Advantage Stream: H -> A * N
        adv_ = register_module("adv", torch::nn::Linear(in_features, action_dim * num_quantiles));

        WeightInitializer::Initialize(value_, init_config);
        WeightInitializer::Initialize(adv_, init_config);
    }

    anet::TensorDict Forward(torch::Tensor x) override
    {
        anet::ProfileRange r("QuantileDuelingHead::Forward");

        auto batch_size = x.size(0);

        // V: (B, N) -> (B, 1, N)
        auto v = value_->forward(x).view({ batch_size, 1, num_quantiles_ });

        // A: (B, A*N) -> (B, A, N)
        auto a = adv_->forward(x).view({ batch_size, action_dim_, num_quantiles_ });

        // Mean A over actions: (B, 1, N)
        auto a_mean = a.mean(1, true);

        // Q = V + (A - mean(A)) -> (B, A, N)
        auto q_dist = v + (a - a_mean);

        // Q分布の平均としてQ値を算出
        auto q = q_dist.mean(2); // (B, A)

        // 結果を返す
        anet::TensorDict out;
        out.Set("q_dist", q_dist);
        out.Set("q", q);
        out.Set("v_dist", v);
        out.Set("a_dist", a);
        return out;
    }

    std::optional<anet::TensorFunction> GetTensorFunction(const std::string& key) override
    {
        // Q値 (B, A) - 最終的なQ値の期待値
        if (key == "forward" || key == "forward.q" || key == "q_values") {
            return [this](const torch::Tensor& x) {
                torch::NoGradGuard no_grad;
                auto batch_size = x.size(0);
                auto v = value_->forward(x).view({ batch_size, 1, num_quantiles_ });
                auto a = adv_->forward(x).view({ batch_size, action_dim_, num_quantiles_ });
                auto q_dist = v + (a - a.mean(1, true));
                return q_dist.mean(2);
                };
        }
        // 分布 (B, A, N) - 最終的なQ分布
        if (key == "forward.dist" || key == "distributions") {
            return [this](const torch::Tensor& x) {
                torch::NoGradGuard no_grad;
                auto batch_size = x.size(0);
                auto v = value_->forward(x).view({ batch_size, 1, num_quantiles_ });
                auto a = adv_->forward(x).view({ batch_size, action_dim_, num_quantiles_ });
                return v + (a - a.mean(1, true));
                };
        }
        // Value分布 (B, 1, N)
        if (key == "forward.v" || key == "v_values") {
            return [this](const torch::Tensor& x) {
                torch::NoGradGuard no_grad;
                auto v = value_->forward(x);
                return v.view({ v.size(0), 1, num_quantiles_ });
                };
        }
        // Advantage分布 (B, A, N) - Centering前の生データ
        if (key == "forward.a" || key == "a_values") {
            return [this](const torch::Tensor& x) {
                torch::NoGradGuard no_grad;
                auto a = adv_->forward(x);
                return a.view({ a.size(0), action_dim_, num_quantiles_ });
                };
        }
        return std::nullopt;
    }
private:
    torch::nn::Linear value_{ nullptr };
    torch::nn::Linear adv_{ nullptr };
    int64_t action_dim_;
    int64_t num_quantiles_;
};


// ===========================================================================
// Factories
// ===========================================================================

LinearHeadFactory::LinearHeadFactory(int64_t action_dim, const WeightInitConfig& init_config)
    : action_dim_(action_dim), init_config_(init_config)
{
}

std::shared_ptr<NetworkHead> LinearHeadFactory::CreateHead(int64_t input_dim) const
{
    return std::make_shared<LinearHead>(input_dim, action_dim_, init_config_);
}


DuelingHeadFactory::DuelingHeadFactory(int64_t action_dim, const WeightInitConfig& init_config)
        : action_dim_(action_dim), init_config_(init_config)
{
}

std::shared_ptr<NetworkHead> DuelingHeadFactory::CreateHead(int64_t input_dim) const
{
    return std::make_shared<DuelingHead>(input_dim, action_dim_, init_config_);
}

QuantileHeadFactory::QuantileHeadFactory(int64_t action_dim, int64_t num_quantiles, const WeightInitConfig& init_config)
        : action_dim_(action_dim), num_quantiles_(num_quantiles), init_config_(init_config)
{
}

std::shared_ptr<NetworkHead> QuantileHeadFactory::CreateHead(int64_t input_dim) const
{
    return std::make_shared<QuantileHead>(input_dim, action_dim_, num_quantiles_, init_config_);
}

QuantileDuelingHeadFactory::QuantileDuelingHeadFactory(int64_t action_dim, int64_t num_quantiles, const WeightInitConfig& init_config)
        : action_dim_(action_dim), num_quantiles_(num_quantiles), init_config_(init_config)
{
}

std::shared_ptr<NetworkHead> QuantileDuelingHeadFactory::CreateHead(int64_t input_dim) const
{
    return std::make_shared<QuantileDuelingHead>(input_dim, action_dim_, num_quantiles_, init_config_);
}
