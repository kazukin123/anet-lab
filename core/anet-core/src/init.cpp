#include "anet/init.hpp"
#include "anet/dqn_agent.hpp"
#include "anet/rainbow_agent.hpp"
#include "anet/default_dqn_agent.hpp"
#include "nn_impl.hpp"

using namespace anet::rl;

void anet::rl::InitRL(const BackendConfig& backend_config)
{
	// Torch関連設定反映
	torch::Context& ctx = torch::globalContext();
	ctx.setAllowTF32CuBLAS(backend_config.use_tf32_cublas);
	ctx.setAllowTF32CuDNN(backend_config.use_tf32_cudnn);
	ctx.setDeterministicCuDNN(backend_config.cudnn_deterministic);  // 非決定論的である代わりに高速化
	ctx.setBenchmarkCuDNN(backend_config.cudnn_benchmark);		// サイズが変化しない場合に高速化

	anet::nn::InitNN();
	RegisterAgentFactory<DQNAgentFactory>();
	RegisterAgentFactory<anet::rl::dqn::RainbowAgentFactory>();
	RegisterAgentFactory<anet::rl::dqn::DefaultDQNAgentFactory>();
}
