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

    /// TensorDict を受け取り、変換した TensorDict を返す。
    class Forwarder {
    public:
        virtual ~Forwarder() = default;
        virtual TensorDict Forward(const TensorDict& input, ForwardCapture* capture = nullptr) = 0;
    };

    /// 依頼で決まった入力と出力の spec を持つ変換。
    class SpecForwarder : public Forwarder {
    public:
        virtual const TensorSpecMap& GetInputSpecs() const = 0;
        virtual const TensorSpecMap& GetOutputSpecs() const = 0;
        /// Network と異なり、図に対応しない実装は空を返す。
        virtual std::unique_ptr<graphviz::GraphViz> MakeGraphViz(bool detail) const;
        virtual std::vector<std::pair<std::string, std::string>> GetGraphVizDetails() const { return {}; }
    };

    using ForwardKind = std::string;

    class ForwarderRequest {
    public:
        explicit ForwarderRequest(ForwardKind forward_kind, ConfigData options = {});
    public:
        const ForwardKind& GetForwardKind() const { return forward_kind_; }
        const ConfigData& GetOptions() const { return options_; }
    private:
        ForwardKind forward_kind_;
        ConfigData options_;
    };

    class ForwarderBuilder final : public ModuleBase {
    public:
        ForwarderBuilder(const ConfigData& config_data, const std::string& config_prefix, seed_t seed);
        ~ForwarderBuilder() override;
    public:
        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index = -1) const override;
        std::shared_ptr<SpecForwarder> BuildForwarder(const ForwarderRequest& request,
            const TensorSpecMap& input_specs) const;
    private:
        struct Definition;
    private:
        std::unique_ptr<Definition> definition_;
    };

    // Forwarder の部品を定義・登録する側の API。

    class ForwarderModule : public ModuleBase {
    public:
        /// 設定値を検査したあと、並びで使える種類との組み合わせを検査する。
        virtual void ValidateForwardKinds(const std::optional<std::vector<ForwardKind>>&) const {}
        virtual std::shared_ptr<SpecForwarder> CreateForwarder(const ForwarderRequest& request,
            const TensorSpecMap& input_specs) const = 0;
    };

    std::shared_ptr<SpecForwarder> MakePassThroughForwarder(const TensorSpecMap& specs);

    struct ForwarderModuleContext {
        seed_t seed = 0;
        std::optional<std::vector<ForwardKind>> forward_kinds;
    };

    class ForwarderModuleFactory {
    public:
        virtual ~ForwarderModuleFactory() = default;
    public:
        /// 種類を問わない工場は std::nullopt を返す。
        virtual std::optional<std::vector<ForwardKind>> GetForwardKinds() const = 0;
        /// 受け付ける設定キーを文字列として読む。値の型と値域は CreateModule で検査する。
        virtual void ReadConfigKeys(const ConfigData&) const {}
        virtual std::shared_ptr<ForwarderModule> CreateModule(const ConfigData& config_data,
            const ForwarderModuleContext& context) const = 0;
    };

    void RegisterForwarderModuleFactory(const std::string& type_name,
        std::shared_ptr<ForwarderModuleFactory> factory);

}

namespace anet::rl {

    inline constexpr char kForwardKindFrameWithStatsUpdate[] = "FrameWithStatsUpdate";
    inline constexpr char kForwardKindPerceptionFromStack[] = "PerceptionFromStack";
    inline constexpr char kForwardKindPerceptionFromFrame[] = "PerceptionFromFrame";
    inline constexpr char kForwardKindPerceptionFromSequence[] = "PerceptionFromSequence";

    ForwarderRequest MakeFrameWithStatsUpdateRequest();
    ForwarderRequest MakePerceptionFromStackRequest(int stack_count, std::vector<std::string> stack_keys);
    ForwarderRequest MakePerceptionFromFrameRequest(int stack_count, std::vector<std::string> stack_keys);
    ForwarderRequest MakePerceptionFromSequenceRequest(int row_count, seed_t seed, std::string start_key,
        int stack_count, std::vector<std::string> stack_keys);

}
