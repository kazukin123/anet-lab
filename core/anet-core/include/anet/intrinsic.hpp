#pragma once

#include <map>
#include <memory>
#include <string>
#include "anet/config.hpp"
#include "anet/random.hpp"
#include "anet/serialize.hpp"

namespace anet::rl::intrinsic {

    /// 一生の新しさを倍率として返す。呼び出し側が報酬への合成を所有する。
    class LifelongNoveltyEstimator : public anet::Serializable {
    public:
        /// CPU の次観測を採点し、CPU float の lane ごとの倍率を返す。
        virtual torch::Tensor Multiplier(const anet::TensorDict& observations) = 0;
        /// 直前に採点した同じ観測で補助学習する。保存はこの操作の完了後に行う。
        virtual void Update(const anet::TensorDict& observations) = 0;
        /// 認識する診断キーを常に返し、未成立の値は NaN とする。
        virtual std::map<std::string, float> Diagnostics() const = 0;
        virtual ~LifelongNoveltyEstimator() = default;
    };

    /// 設定の葉の解釈と方式固有の構築を、呼び出し側から隠す。
    class LifelongNoveltyEstimatorFactory {
    public:
        static std::unique_ptr<LifelongNoveltyEstimator> Create(
            const anet::ConfigData& config_data, const std::string& prefix,
            const anet::TensorSpecMap& observation_spec, const torch::Device& device,
            const anet::SeedMaker& seed_maker);
    };

} // namespace anet::rl::intrinsic
