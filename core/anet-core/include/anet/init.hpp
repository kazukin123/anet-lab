// anet/init.hpp

#pragma once

#include <stdexcept>
#include <string>

#include "anet/config.hpp"

namespace anet::rl {

	struct BackendConfig : public anet::Config {
        bool use_tf32_cublas = true;
        bool use_tf32_cudnn = true;
        bool cudnn_deterministic = true;
        bool cudnn_benchmark = false;
        /// 全 ATen op を決定化して同 seed 再現性を確保する（cuDNN 外＝SDPA 等の非決定もカバー）。
        /// 決定版が無い op に当たると warn_only に従い throw/警告する。既定 true（cudnn_deterministic と同方針）。
        bool deterministic_algorithms = true;
        /// true: 決定版が無い op を例外でなく警告で素通りさせる（再現性は保証されない／throw 退避・診断用）。
        /// deterministic_algorithms=false のときは無視される。既定 false。
        bool deterministic_warn_only = false;
        int torch_num_threads = 1;
        std::string cuda_launch_blocking = "inherit";

        explicit BackendConfig(const ConfigData& config_data = EmptyConfigData, const std::string& prefix = "backend")
            : anet::Config(config_data, prefix)
        {
            ANET_READ_CONFIG(config_data, use_tf32_cublas);
            ANET_READ_CONFIG(config_data, use_tf32_cudnn);
            ANET_READ_CONFIG(config_data, cudnn_deterministic);
            ANET_READ_CONFIG(config_data, cudnn_benchmark);
            ANET_READ_CONFIG(config_data, deterministic_algorithms);
            ANET_READ_CONFIG(config_data, deterministic_warn_only);
            ANET_READ_CONFIG(config_data, torch_num_threads);
            ANET_READ_CONFIG(config_data, cuda_launch_blocking);

            cuda_launch_blocking = NormalizeCudaLaunchBlocking(cuda_launch_blocking);
            my_config_data_.Set("cuda_launch_blocking", cuda_launch_blocking);
            my_config_json_["cuda_launch_blocking"] = cuda_launch_blocking;
        }
    private:
        static std::string NormalizeCudaLaunchBlocking(const std::string& value)
        {
            const auto normalized = anet::ToLower(anet::TrimCopy(value));
            if (normalized == "on" || normalized == "true" || normalized == "1") return "on";
            if (normalized == "off" || normalized == "false" || normalized == "0") return "off";
            if (normalized == "inherit") return "inherit";

            throw std::invalid_argument(
                "Invalid backend.cuda_launch_blocking value: " + value +
                " (expected on, off, inherit, true, false, 1, or 0)");
        }
	};

	void InitRL(const BackendConfig& backend_config);
}

