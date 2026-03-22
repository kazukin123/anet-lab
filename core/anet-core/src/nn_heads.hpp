// nn_heads.hpp

#pragma once

#include "nn_impl.hpp"

namespace anet::nn {


    /// @todo RL固有実装はanet::rlとかに移動?


    // ===========================================================================
    // PassThroughHead
    // ===========================================================================

    class PassThroughHead : public NetworkHead {
    public:
        explicit PassThroughHead(const std::string& output_key);
        anet::TensorDict Forward(torch::Tensor feature_vector) override;
        std::optional<anet::TensorFunction> GetTensorFunction(const std::string& key) override;
    private:
        std::string output_key_;
    };

    ///  Configの文字列で定義された複数の独立したブランチ（V, A, 報酬など）を実行する。
    // ブランチの構造が空("")の場合、NetworkStructの仕様により自動的にパススルーとなる。
    class ConfigurableHead : public NetworkHead {
    public:
        ConfigurableHead(
            int64_t feature_dim,
            std::map<std::string, std::shared_ptr<NetworkStruct>> branches
        );

        anet::TensorDict Forward(torch::Tensor feature_vector) override;
        std::optional<anet::TensorFunction> GetTensorFunction(const std::string& key) override;
    private:
        std::map<std::string, std::shared_ptr<NetworkStruct>> branches_;
    };


    // ===========================================================================
    // Factories
    // ===========================================================================

    class PassThroughHeadFactory : public NetworkHeadFactory {
    public:
        explicit PassThroughHeadFactory(const std::string& output_key, int64_t expected_input_dim = -1);
        std::shared_ptr<NetworkHead> CreateHead(int64_t input_dim) const override;
    private:
        const std::string output_key_;
        const int64_t expected_input_dim_;
    };

    class ConfigurableHeadFactory : public NetworkHeadFactory {
    public:
        // @param config         NetworkConfig (additional_structures に設定文字列が含まれている前提)
        // @param branch_mapping { "出力されるキー名" : "config.additional_structures内の検索キー" }
        ConfigurableHeadFactory(const NetworkConfig& config, const std::map<std::string, std::string>& branch_mapping);

        std::shared_ptr<NetworkHead> CreateHead(int64_t input_dim) const override;
    private:
        NetworkConfig config_;
        std::map<std::string, std::string> branch_mapping_;
    };

    class LinearHeadFactory : public NetworkHeadFactory {
    public:
        LinearHeadFactory(int64_t action_dim, const WeightInitConfig& init_config);
        std::shared_ptr<NetworkHead> CreateHead(int64_t input_dim) const override;
    private:
        int64_t action_dim_;
        WeightInitConfig init_config_;
    };

    class DuelingHeadFactory : public NetworkHeadFactory {
    public:
        DuelingHeadFactory(int64_t action_dim, const WeightInitConfig& init_config);
        std::shared_ptr<NetworkHead> CreateHead(int64_t input_dim) const override;
    private:
        int64_t action_dim_;
        WeightInitConfig init_config_;
    };

    class QuantileHeadFactory : public NetworkHeadFactory {
    public:
        QuantileHeadFactory(int64_t action_dim, int64_t num_quantiles, const WeightInitConfig& init_config);
        std::shared_ptr<NetworkHead> CreateHead(int64_t input_dim) const override;
    private:
        int64_t action_dim_;
        int64_t num_quantiles_;
        WeightInitConfig init_config_;
    };

    class QuantileDuelingHeadFactory : public NetworkHeadFactory {
    public:
        QuantileDuelingHeadFactory(int64_t action_dim, int64_t num_quantiles, const WeightInitConfig& init_config);
        std::shared_ptr<NetworkHead> CreateHead(int64_t input_dim) const override;
    private:
        int64_t action_dim_;
        int64_t num_quantiles_;
        WeightInitConfig init_config_;
    };

} // namespace anet::nn
