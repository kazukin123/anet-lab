// nn_heads.cpp

#include "nn_heads.hpp"
#include "anet/profile.hpp"


using namespace anet::nn;


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

    torch::Tensor Forward(torch::Tensor feature_vector) override
    {
        anet::ProfileRange r("LinearHead::Forward");

        return linear_->forward(feature_vector);
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

    torch::Tensor Forward(torch::Tensor x) override
    {
        anet::ProfileRange r("DuelingHead::Forward");

        auto v = value_->forward(x); // (B, 1)
        auto a = adv_->forward(x);   // (B, A)

        // Q = V + (A - mean(A))
        auto a_mean = a.mean(/*dim=*/1, /*keepdim=*/true);
        return v + (a - a_mean);
    }

    std::optional<anet::TensorFunction> GetTensorFunction(const std::string& key) override
    {
        // Q値 (B, A)
        if (key == "forward" || key == "forward.q" || key == "q_values") {
            return [this](const torch::Tensor& x) {
                torch::NoGradGuard no_grad;
                return this->Forward(x); // Q = V + (A - meanA)
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

    torch::Tensor Forward(torch::Tensor x) override
    {
        anet::ProfileRange r("QuantileHead::Forward");

        // Return Flat Tensor (B, A * N)
        // Note: Reshaping to (B, A, N) is handled by the Adapter (dqn::Network) or Learner
        return linear_->forward(x);
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

    torch::Tensor Forward(torch::Tensor x) override
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

        // Return Flat: (B, A * N)
        return q_dist.view({ batch_size, -1 });
    }

    std::optional<anet::TensorFunction> GetTensorFunction(const std::string& key) override
    {
        // Q値 (B, A) - 最終的なQ値の期待値
        if (key == "forward" || key == "forward.q" || key == "q_values") {
            return [this](const torch::Tensor& x) {
                torch::NoGradGuard no_grad;
                // Forward内部で V + (A - meanA) 計算済み、ただしFlatで返ってくる
                auto flat_q = this->Forward(x);
                // (B, A*N) -> (B, A, N) -> mean -> (B, A)
                return flat_q.view({ flat_q.size(0), action_dim_, num_quantiles_ }).mean(2);
                };
        }
        // 分布 (B, A, N) - 最終的なQ分布
        if (key == "forward.dist" || key == "distributions") {
            return [this](const torch::Tensor& x) {
                torch::NoGradGuard no_grad;
                auto flat_q = this->Forward(x);
                return flat_q.view({ flat_q.size(0), action_dim_, num_quantiles_ });
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

std::shared_ptr<NetworkHead> LinearHeadFactory::CreateHead(int64_t feature_dim) const
{
    return std::make_shared<LinearHead>(feature_dim, action_dim_, init_config_);
}


DuelingHeadFactory::DuelingHeadFactory(int64_t action_dim, const WeightInitConfig& init_config)
        : action_dim_(action_dim), init_config_(init_config)
{
}

std::shared_ptr<NetworkHead> DuelingHeadFactory::CreateHead(int64_t feature_dim) const
{
    return std::make_shared<DuelingHead>(feature_dim, action_dim_, init_config_);
}

QuantileHeadFactory::QuantileHeadFactory(int64_t action_dim, int64_t num_quantiles, const WeightInitConfig& init_config)
        : action_dim_(action_dim), num_quantiles_(num_quantiles), init_config_(init_config)
{
}

std::shared_ptr<NetworkHead> QuantileHeadFactory::CreateHead(int64_t feature_dim) const
{
    return std::make_shared<QuantileHead>(feature_dim, action_dim_, num_quantiles_, init_config_);
}

QuantileDuelingHeadFactory::QuantileDuelingHeadFactory(int64_t action_dim, int64_t num_quantiles, const WeightInitConfig& init_config)
        : action_dim_(action_dim), num_quantiles_(num_quantiles), init_config_(init_config)
{
}

std::shared_ptr<NetworkHead> QuantileDuelingHeadFactory::CreateHead(int64_t feature_dim) const
{
    return std::make_shared<QuantileDuelingHead>(feature_dim, action_dim_, num_quantiles_, init_config_);
}
