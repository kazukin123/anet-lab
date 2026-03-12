#include "anet/init.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/rainbow_agent.hpp"
#include "anet/default_dqn_agent.hpp"
#include "nn_impl.hpp"

using namespace anet::rl;

void anet::rl::InitRL(const BackendConfig& backend_config)
{
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
}
