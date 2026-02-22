// anet/init.hpp

#pragma once

#include "anet/config.hpp"

namespace anet::rl {

	struct BackendConfig : public anet::Config {
        bool use_tf32_cublas = true;
        bool use_tf32_cudnn = true;
        bool cudnn_deterministic = true;
        bool cudnn_benchmark = false;

        explicit BackendConfig(const ConfigData& config_data = EmptyConfigData, const std::string& prefix = "backend")
            : anet::Config(config_data, prefix)
        {
            ANET_READ_CONFIG(config_data, use_tf32_cublas);
            ANET_READ_CONFIG(config_data, use_tf32_cudnn);
            ANET_READ_CONFIG(config_data, cudnn_deterministic);
            ANET_READ_CONFIG(config_data, cudnn_benchmark);
        }
	};

	void InitRL(const BackendConfig& backend_config);
}

