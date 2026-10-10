// nn_heads.hpp

#pragma once

#include "nn_impl.hpp"

#include <utility>

namespace anet::nn {


    // 子の実行interfaceをHead固有の計算へ渡し、準備結果の寿命だけを共通管理する。
    template <typename Head, auto ForwardMethod, size_t N>
    class PreparedHeadExecution final : public NetworkHeadExecution {
    public:
        PreparedHeadExecution(Head& owner, PreparedModuleChildren<N> children)
            : owner_(owner), children_(std::move(children)) {}
        anet::TensorDict Forward(const anet::TensorDict& input,
            const NetworkForwardContext& context) override
        {
            return ForwardImpl(input, context, std::make_index_sequence<N>{});
        }
    private:
        template <size_t... Indices>
        anet::TensorDict ForwardImpl(const anet::TensorDict& input,
            const NetworkForwardContext& context, std::index_sequence<Indices...>)
        {
            return (owner_.*ForwardMethod)(input, context, *children_.Get(Indices)...);
        }
        Head& owner_;
        const PreparedModuleChildren<N> children_;
    };

    template <typename Head, auto ForwardMethod, size_t N>
    std::unique_ptr<NetworkHeadExecution> MakePreparedHeadExecution(
        Head& owner, const std::array<NetworkModule*, N>& modules,
        const NetworkForwardConfig& config, NetworkForwardClock clock)
    {
        // 子を指定順に準備し、通常動作だけならHead本体への直接呼び出しを維持する。
        PreparedModuleChildren<N> children(modules, config, clock);
        if (!children.HasPrepared()) return nullptr;
        return std::make_unique<PreparedHeadExecution<Head, ForwardMethod, N>>(
            owner, std::move(children));
    }

    static constexpr const char* kKey_DefaultOutput = "features";
    static constexpr const char* kKey_Taus = "taus";

    // ===========================================================================
    // PassThroughHead
    // ===========================================================================

    class PassThroughHead : public NetworkHead {
    public:
        explicit PassThroughHead(const std::string& output_key);
        anet::TensorDict Forward(const anet::TensorDict& feature_dict, const anet::nn::NetworkForwardContext& context) override;
        std::optional<TensorDictFunction> GetTensorDictFunction(const std::string& key) override;
        HeadGraphVizInfo GetGraphVizInfo() const override;
    private:
        std::string output_key_;
    };


    // ===========================================================================
    // LinearHead
    // ===========================================================================

    class LinearHead : public NetworkHead {
    public:
        LinearHead(
            int64_t in_features,
            int64_t out_features,
            std::string output_key,
            const WeightInitConfig& init_config);

        anet::TensorDict Forward(const anet::TensorDict& feature_dict, const anet::nn::NetworkForwardContext& context) override;
        std::optional<TensorDictFunction> GetTensorDictFunction(const std::string& key) override;
        HeadGraphVizInfo GetGraphVizInfo() const override;
    private:
        torch::nn::Linear linear_{ nullptr };
        int64_t out_features_;
        std::string output_key_;
    };

    class LinearHeadFactory : public NetworkHeadFactory {
    public:
        LinearHeadFactory(
            int64_t out_features,
            std::string output_key,
            const WeightInitConfig& init_config);

        std::shared_ptr<NetworkHead> CreateHead(const anet::TensorDict& dummy_features) const override;
    private:
        int64_t out_features_;
        std::string output_key_;
        WeightInitConfig init_config_;
    };

} // namespace anet::nn
