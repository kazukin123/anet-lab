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
	// --- 決定論的アルゴリズム（同 seed 再現性） -------------------------------------------
	// setDeterministicAlgorithms はグローバルな「決定論ポリシー」スイッチ：
	//   (a) 決定版を持つ op（index_add / scatter_add 等）を決定版へ強制する。
	//   (b) 決定版が無い op は warn_only=false なら op 名付きで throw、true なら警告して
	//       非決定のまま実行する。→ warn_only=false が「真の決定モード」。true は再現性を
	//       保証しない（throw 退避・診断用）。
	//
	// なぜ cudnn_deterministic と別に要るか：
	//   cudnn_deterministic は cuDNN 畳み込み限定。SDPA（at::scaled_dot_product_attention）は
	//   ATen の flash / mem-efficient カーネルで cuDNN を通らず、その backward が atomic 加算で
	//   非決定。これが同 seed 非再現の真因だった（forward は決定的＝eval は再現・train だけ割れる）。
	//   mem-efficient は決定的 backward 変種を持ち、本フラグ(=true, warn_only=false)で atomic 回避
	//   経路に切替わる。よって SDPA backend の固定は不要、このフラグ 1 つで再現性が戻る（実測確認済み）。
	//
	// SDPA backend を明示制御したい場合（本コードでは未使用・将来用の参照）：
	//   at::Context の setSDPUseFlash / setSDPUseMemEfficient / setSDPUseMath / setSDPUseCuDNN(bool) で
	//   候補を on/off できる（選択は「有効 ∧ 入力対応」を優先順位 flash>efficient>cuDNN>math で先着）。
	//   math 固定でも決定化できるが遅く、mem-efficient の決定経路で足りるため採らない。
	//   将来 dtype/shape 変更で flash が選ばれ、その LibTorch 版に flash の決定 backward が無いと
	//   throw し得る → その時は setSDPUseFlash(false) で mem-efficient(決定)/math へ退避する。
	//   ※ これら setter 名は LibTorch 版差あり（新しめは setSDPPriorityOrder）。ATen/Context.h で要確認。
	//
	// CUBLAS_WORKSPACE_CONFIG：決定モードで cuBLAS GEMM が env(:4096:8 等)を要求する場合がある。
	//   失敗モードは silent ではなく throw。本環境では不要だった（throw せず再現）。将来 CUDA/cuBLAS/
	//   形状変更で要求 throw が出たら、CUDA 初期化前に env を設定する（ApplyCudaLaunchBlockingConfig と同じ枠）。
	//
	// コスト：決定 backward は atomic 回避で遅いが attention は本 Run のボトルネックでない（SDPA 前後で
	//   実時間ほぼ不変）ため無視可。他 op の決定版も僅かに遅くなる場合がある。
	ctx.setDeterministicAlgorithms(backend_config.deterministic_algorithms,
	                               backend_config.deterministic_warn_only);

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
