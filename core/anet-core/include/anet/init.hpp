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
        int torch_num_threads = 1;
        std::string cuda_launch_blocking = "inherit";

        explicit BackendConfig(const ConfigData& config_data = EmptyConfigData, const std::string& prefix = "backend")
            : anet::Config(config_data, prefix)
        {
            ANET_READ_CONFIG(config_data, use_tf32_cublas);
            ANET_READ_CONFIG(config_data, use_tf32_cudnn);
            ANET_READ_CONFIG(config_data, cudnn_deterministic);
            ANET_READ_CONFIG(config_data, cudnn_benchmark);
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

