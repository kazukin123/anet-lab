#include "anet/init.hpp"
#include "anet/log.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/rainbow_agent.hpp"
#include "anet/default_dqn_agent.hpp"
#include "anet/muzero_proto_agent.hpp"
#include "anet/image_cls_agent.hpp"
#include "nn_impl.hpp"

#include <cstdlib>
#include <optional>
#include <string>

using namespace anet::rl;
namespace LOG = anet::log;

namespace {

constexpr const char* kCudaLaunchBlockingEnv = "CUDA_LAUNCH_BLOCKING";

std::optional<std::string> GetEnvironmentValue(const char* name)
{
	const char* value = std::getenv(name);
	if (value == nullptr) return std::nullopt;
	return std::string(value);
}

void SetEnvironmentValue(const char* name, const std::string& value)
{
#ifdef _WIN32
	const int result = _putenv_s(name, value.c_str());
#else
	const int result = setenv(name, value.c_str(), 1);
#endif
	if (result != 0) {
		ANET_SYSTEM_ERROR("Failed to set environment variable: " << name << "=" << value);
	}
}

std::string EnvValueToString(const std::optional<std::string>& value)
{
	return value.has_value() ? *value : "<unset>";
}

void WriteNullableEnvJson(anet::json& data, const std::string& key, const std::optional<std::string>& value)
{
	if (value.has_value()) {
		data[key] = *value;
	} else {
		data[key] = nullptr;
	}
}

void ApplyCudaLaunchBlockingConfig(const BackendConfig& backend_config)
{
	const auto previous_value = GetEnvironmentValue(kCudaLaunchBlockingEnv);
	std::optional<std::string> requested_env_value;

	if (backend_config.cuda_launch_blocking == "on") {
		requested_env_value = "1";
	} else if (backend_config.cuda_launch_blocking == "off") {
		requested_env_value = "0";
	}

	if (requested_env_value.has_value()) {
		SetEnvironmentValue(kCudaLaunchBlockingEnv, *requested_env_value);
	}

	const auto effective_value = GetEnvironmentValue(kCudaLaunchBlockingEnv);
	const bool changed = previous_value != effective_value;

	if (requested_env_value.has_value() && previous_value.has_value() && *previous_value != *requested_env_value) {
		LOG::warn() << "CUDA_LAUNCH_BLOCKING overridden by backend.cuda_launch_blocking."
			<< " previous=" << *previous_value
			<< " requested=" << backend_config.cuda_launch_blocking
			<< " effective=" << EnvValueToString(effective_value);
	} else {
		//LOG::info() << "CUDA_LAUNCH_BLOCKING status."
		//	<< " previous=" << EnvValueToString(previous_value)
		//	<< " requested=" << backend_config.cuda_launch_blocking
		//	<< " effective=" << EnvValueToString(effective_value);
	}
}

} // namespace

void anet::rl::InitRL(const BackendConfig& backend_config)
{
	// CUDA初期化前に近い位置で、設定値をプロセス環境へ反映する
	ApplyCudaLaunchBlockingConfig(backend_config);

	// CPUスレッド数設定
	if (backend_config.torch_num_threads > 0) {
		torch::set_num_threads(backend_config.torch_num_threads);
	}

	// バックエンド設定反映
	torch::Context& ctx = torch::globalContext();
	ctx.setAllowTF32CuBLAS(backend_config.use_tf32_cublas);
	ctx.setAllowTF32CuDNN(backend_config.use_tf32_cudnn);
	ctx.setDeterministicCuDNN(backend_config.cudnn_deterministic);  // 非決定論的である代わりに高速化
	ctx.setBenchmarkCuDNN(backend_config.cudnn_benchmark);		// サイズが変化しない場合に高速化

	// バックエンド設定ログ
	anet::MetricsLogger::Instance()->Log(backend_config);

	// NN初期化（モジュール登録等）
	anet::nn::InitNN();

	// Agent登録
	RegisterAgentFactory<anet::rl::dqn::RainbowAgentFactory>();
	RegisterAgentFactory<anet::rl::dqn::DefaultDQNAgentFactory>();
	RegisterAgentFactory<anet::rl::muzero_proto::MuZeroAgentFactory>();
	RegisterAgentFactory<anet::rl::img_cls::ImageClsAgentFactory>();

}
