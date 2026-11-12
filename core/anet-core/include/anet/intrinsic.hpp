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
        /// 観測を採点して統計を更新し、入力と同じ device の float の lane ごとの倍率を返す。
        virtual torch::Tensor Estimate(const anet::TensorDict& observations) = 0;
        /// 任意のdeviceの観測すべてで補助学習を1 step進めたらtrueを返す。統計は更新しない。
        /// warm-up中だけは何も進めずfalseを返す。抽選と刻みは呼び出し側が持つ。
        /// Estimate と観測・順序・回数を揃える必要はなく、操作間ではいつでも Save/Load できる。
        virtual bool Learn(const anet::TensorDict& observations) = 0;
        /// 認識する診断キーを常に返し、未成立の値は NaN とする。
        virtual std::map<std::string, float> Diagnostics() const = 0;
        virtual ~LifelongNoveltyEstimator() = default;
    };

    /// 設定キーの解釈と方式固有の構築を、呼び出し側から隠す。
    class LifelongNoveltyEstimatorFactory {
    public:
        /// 推定器を作らない場合も認識する診断キーを、初期値NaNとともに返す。
        static std::map<std::string, float> InitialDiagnostics();
        static std::unique_ptr<LifelongNoveltyEstimator> Create(
            const anet::ConfigData& config_data, const std::string& prefix,
            const anet::TensorSpecMap& observation_spec, const torch::Device& device,
            const anet::SeedMaker& seed_maker);
    };

} // namespace anet::rl::intrinsic
