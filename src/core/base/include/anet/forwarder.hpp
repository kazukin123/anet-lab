#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "anet/common.hpp"
#include "anet/random.hpp"

namespace anet::graphviz { class GraphViz; }

namespace anet {

    /// 鍵で要求した途中の値を、同期 Forward から持ち帰る器。
    class ForwardCapture {
    public:
        using Entry = std::pair<std::string, torch::Tensor>;
    public:
        ForwardCapture() = default;
        explicit ForwardCapture(std::vector<std::string> requests);
    public:
        bool Wants(std::string_view key) const;
        /// prefix 以下に一致しうる要求があるか。鍵を組み立てる前の判定に使う。
        bool WantsAny(std::string_view prefix) const;
        void Add(std::string_view key, torch::Tensor value);
        torch::Tensor Get(std::string_view key) const;
        std::vector<Entry> GetAll(std::string_view prefix) const;
        std::vector<std::string> Unfulfilled() const;
        void Reset();
    private:
        std::vector<std::string> requests_;
        std::vector<Entry> entries_;
    };

    /// 呼び出す側ごとの状態を持たず、TensorDict を変換する。
    class Forwarder {
    public:
        virtual ~Forwarder() = default;
        virtual TensorDict Forward(const TensorDict& input, ForwardCapture* capture = nullptr) = 0;
    };

    /// データに加えて、その変換を通したあとの spec を返す。
    class SpecForwarder : public Forwarder {
    public:
        virtual TensorSpecMap ForwardSpec(const TensorSpecMap& input_specs) const = 0;
    };

    enum class ForwardRequest {
        FrameWithStatsUpdate,
        PerceptionFromStack,
        PerceptionFromFrame,
    };

    /// 呼び出す側ごとの履歴を使って TensorDict を変換する。
    class ForwardExecutor {
    public:
        virtual ~ForwardExecutor() = default;
        virtual TensorDict Forward(const TensorDict& input, const torch::Tensor& episode_start,
            ForwardCapture* capture = nullptr) = 0;
    };

    /// 依頼によって処理が異なる部品が、依頼に合った変換と実行器を作る。
    class ForwarderProvider {
    public:
        virtual ~ForwarderProvider() = default;
        /// 空のポインタは、この依頼で何も処理しないことを表す。
        virtual std::shared_ptr<Forwarder> MakeForwarder(ForwardRequest request) const = 0;
        virtual std::unique_ptr<ForwardExecutor> CreateExecutor(int lane_count, torch::Device device, seed_t seed) const = 0;
    };

    /// 観測の spec と設定から、変換を行う部品の並びを組み立てる。
    class ForwarderChain final : public SpecForwarder {
    public:
        ForwarderChain(const ConfigData& config_data, const std::string& config_prefix,
            const TensorSpecMap& input_specs, seed_t seed);
        ~ForwarderChain() override;
    public:
        TensorDict Forward(const TensorDict& input, ForwardCapture* capture = nullptr) override;
        TensorSpecMap ForwardSpec(const TensorSpecMap& input_specs) const override;
    public:
        std::shared_ptr<Forwarder> GetForwarder(ForwardRequest request) const;
        std::unique_ptr<ForwardExecutor> CreateExecutor(int lane_count, torch::Device device, seed_t seed) const;
        int GetStackCount() const;
        /// 入力 spec と構成から選んだ、空でないキーの一覧。
        const std::vector<std::string>& GetStackKeys() const;
        std::optional<float> GetScalar(const std::string& key) const;
        std::optional<torch::Tensor> GetTensor(const std::string& key) const;
        std::unique_ptr<graphviz::GraphViz> MakeGraphViz(bool detail) const;
    private:
        struct Definition;
    private:
        std::unique_ptr<Definition> definition_;
    };

}
