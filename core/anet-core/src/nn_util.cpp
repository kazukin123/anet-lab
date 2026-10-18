#include "anet/nn_util.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <ATen/autocast_mode.h>
#include <torch/csrc/autograd/autograd.h>
#include <ATen/ops/_amp_foreach_non_finite_check_and_unscale.h>
#include <ATen/ops/_foreach_add.h>
#include <ATen/ops/_foreach_mul.h>
#include <ATen/ops/_foreach_norm.h>
#include <ATen/ops/_fused_adamw.h>
#include "anet/common.hpp"
#include "anet/profile.hpp"


using namespace anet;


// ======================================================
// Gradient Utilities
// ======================================================

GradientNoiseEstimate anet::GradientNoiseEstimator::Update(const GradientNoiseEstimate& estimate)
{
    // 対の片方だけを更新せず、同じ測定点の履歴を保つ。
    if (!std::isfinite(estimate.signal) || !std::isfinite(estimate.variance)) return {};
    signal_.Update(estimate.signal);
    variance_.Update(estimate.variance);
    return { .signal = signal_.Value(), .variance = variance_.Value() };
}

GradientNoiseMeasurement anet::MeasureGradientNoise(const torch::Tensor& weighted_element_loss,
    const std::vector<GradientNoiseParameter>& parameters, const torch::optim::AdamW& optimizer, int num_micro_batches, double epsilon,
    const GradientNoiseRequest& request)
{
    ANET_PROFILE_FUNC();
    if (!weighted_element_loss.defined() || weighted_element_loss.dim() != 1 || weighted_element_loss.size(0) <= 0 ||
        num_micro_batches < 2 || weighted_element_loss.size(0) % num_micro_batches != 0) {
        ANET_SYSTEM_ERROR("Gradient noise requires weighted_element_loss [B], B>0, k>=2 dividing B; k=" << num_micro_batches);
    }
    const int64_t batch_size = weighted_element_loss.size(0);
    // 前回の.gradではなく今回のautograd結果を使い、未使用parameterを除く。
    std::vector<torch::Tensor> inputs;
    std::vector<bool> feature;
    for (const auto& entry : parameters) {
        if (!entry.parameter.requires_grad()) continue;
        inputs.push_back(entry.parameter);
        feature.push_back(entry.feature);
    }
    if (inputs.empty() || !weighted_element_loss.requires_grad()) return {};
    const bool adam_requested = std::ranges::any_of(request.adam, [](bool value) { return value; });
    std::unordered_map<void*, double> beta2;
    if (adam_requested) {
        for (const auto& group : optimizer.param_groups()) {
            const auto& options = static_cast<const torch::optim::AdamWOptions&>(group.options());
            for (const auto& parameter : group.params())
                beta2.emplace(parameter.unsafeGetTensorImpl(), std::get<1>(options.betas()));
        }
    }
    std::vector<torch::Tensor> sums(inputs.size()), preconditioners(inputs.size());
    std::vector<bool> used(inputs.size(), false);
    std::array<bool, 3> group_used{}, group_valid{ true, true, true };
    auto small = torch::zeros({ 4 }, inputs.front().options().requires_grad(false).dtype(torch::kFloat64));
    auto big = torch::zeros_like(small);
    for (int j = 0; j < num_micro_batches; ++j) {
        const auto loss = weighted_element_loss.narrow(0, j * (batch_size / num_micro_batches), batch_size / num_micro_batches).mean();
        const auto gradients = torch::autograd::grad({ loss }, inputs, {}, true, false, true);
        torch::NoGradGuard no_grad;
        for (size_t i = 0; i < gradients.size(); ++i) {
            if (!gradients[i].defined()) continue;
            const auto gradient = gradients[i].detach().to(torch::kFloat32);
            const size_t group = feature[i] ? 1 : 2;
            group_used[0] = group_used[group] = true;
            // 固定前処理Pを最初の使用時だけ作る。stateの作成や変更は行わない。
            if (!used[i] && adam_requested && (request.adam[0] || request.adam[group])) {
                const auto id = inputs[i].unsafeGetTensorImpl();
                const auto found = optimizer.state().find(id);
                if (found != optimizer.state().end() && beta2.contains(id)) {
                    const auto& state = static_cast<const torch::optim::AdamWParamState&>(*found->second);
                    const double correction = 1.0 - std::pow(beta2.at(id), state.step());
                    if (state.step() > 0 && correction > 0 && state.exp_avg_sq().defined()) {
                        preconditioners[i] = (state.exp_avg_sq().detach() / correction).sqrt().add(epsilon).reciprocal();
                    }
                }
                if (!preconditioners[i].defined()) group_valid[0] = group_valid[group] = false;
            }
            used[i] = true;
            if (!sums[i].defined())
                sums[i] = gradient.clone();
            else
                sums[i].add_(gradient);
            if (request.raw) small[0].add_(gradient.square().sum(torch::kFloat64));
            if (preconditioners[i].defined()) {
                const auto squared = (gradient * preconditioners[i]).square().sum(torch::kFloat64);
                if (request.adam[0]) small[1].add_(squared);
                if (request.adam[group]) small[group + 1].add_(squared);
            }
        }
    }
    // 大batch勾配は加算バッファから作り、取得済み勾配の全組を保持しない。
    torch::NoGradGuard no_grad;
    for (size_t i = 0; i < sums.size(); ++i) {
        if (!used[i]) continue;
        if (request.raw) big[0].add_(sums[i].square().sum(torch::kFloat64));
        const size_t group = feature[i] ? 1 : 2;
        if (preconditioners[i].defined()) {
            const auto squared = (sums[i] * preconditioners[i]).square().sum(torch::kFloat64);
            if (request.adam[0]) big[1].add_(squared);
            if (request.adam[group]) big[group + 1].add_(squared);
        }
    }
    const auto norms = torch::stack({ small / num_micro_batches, big / (static_cast<double>(num_micro_batches) * num_micro_batches) })
                           .to(torch::kCPU)
                           .contiguous();
    const auto values = norms.accessor<double, 2>();
    const auto estimate = [&](int index) -> GradientNoiseEstimate {
        const double s = values[0][index], b = values[1][index];
        if (!std::isfinite(s) || !std::isfinite(b)) return {};
        return { .signal = (num_micro_batches * b - s) / (num_micro_batches - 1),
            .variance = (s - b) * batch_size / (num_micro_batches - 1) };
    };
    GradientNoiseMeasurement result;
    if (request.raw && group_used[0]) result.raw = estimate(0);
    for (size_t group = 0; group < 3; ++group) {
        if (request.adam[group] && group_used[group] && group_valid[group]) result.adam[group] = estimate(static_cast<int>(group + 1));
    }
    return result;
}
std::vector<torch::Tensor> anet::CollectDefinedGrads(const std::vector<torch::Tensor>& parameters)
{
    // set_to_none後は未定義gradが混ざるため、foreach対象だけを毎回集める。
    std::vector<torch::Tensor> grads;
    grads.reserve(parameters.size());
    for (const auto& param : parameters) {
        if (param.grad().defined()) {
            grads.push_back(param.grad().detach());
        }
    }
    return grads;
}

torch::Tensor anet::ForeachGradNorm(const std::vector<torch::Tensor>& grads)
{
    // ノルム計算をforeachにまとめ、値化によるCPU同期を呼び出し側まで遅延する。
    if (grads.empty()) {
        return torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32));
    }

    auto norms = at::_foreach_norm(grads, 2);
    return torch::stack(norms).norm(2);
}

void anet::ForeachClipGradNorm_(const std::vector<torch::Tensor>& grads, const torch::Tensor& total_norm, float tau)
{
    // 既存の手動clipと同じ +1e-6 / clamp_max(1.0) のセマンティクスをforeachで適用する。
    if (grads.empty()) return;

    auto tau_tensor = torch::full({}, tau, total_norm.options());
    auto scale = (tau_tensor / (total_norm + 1e-6)).clamp_max(1.0);
    at::_foreach_mul_(grads, scale);
}


// ======================================================
// Autocast
// ======================================================

anet::Autocast::Autocast(torch::Device device, bool enabled, torch::ScalarType dtype)
    : device_type_(device.type())
    , prev_enabled_(at::autocast::is_autocast_enabled(device_type_))
    , prev_dtype_(at::autocast::get_autocast_dtype(device_type_))
    , prev_cache_enabled_(at::autocast::is_autocast_cache_enabled())
{
    // autocast状態はdevice indexではなくdevice type単位なので、anetのdevice指定からtypeだけを反映する。
    // libtorchのautocast scopeとして入れ子深度を進め、外側scopeとのcache lifetimeを揃える。
    at::autocast::increment_nesting();
    at::autocast::set_autocast_dtype(device_type_, dtype);
    at::autocast::set_autocast_enabled(device_type_, enabled);
}

anet::Autocast::~Autocast()
{
    at::autocast::set_autocast_enabled(device_type_, prev_enabled_);    // 元に戻す
    at::autocast::set_autocast_dtype(device_type_, prev_dtype_);        // 元に戻す
    if (at::autocast::decrement_nesting() == 0) {
        at::autocast::clear_cache();
    }
    at::autocast::set_autocast_cache_enabled(prev_cache_enabled_);      // 元に戻す
}


// ======================================================
// FusedAdamW
// ======================================================

struct anet::FusedAdamW::FusedAdamWStepGroup {
    std::vector<torch::Tensor> params;
    std::vector<torch::Tensor> grads;
    std::vector<torch::Tensor> exp_avgs;
    std::vector<torch::Tensor> exp_avg_sqs;
    std::vector<torch::Tensor> max_exp_avg_sqs;
    std::vector<torch::Tensor> state_steps;
};

static std::string MakeFusedAdamWStepGroupKey(const torch::Tensor& tensor)
{
    return tensor.device().str() + ":" + std::to_string(static_cast<int>(tensor.scalar_type()));
}

torch::Tensor anet::FusedAdamW::step(LossClosure closure)
{
    torch::Tensor loss;
    if (closure != nullptr) {
        at::AutoGradMode enable_grad(true);
        loss = closure();
    }

    torch::NoGradGuard no_grad;
    ANET_PROFILE_FUNC();

    for (auto& group : param_groups_) {
        auto& options = static_cast<torch::optim::AdamWOptions&>(group.options());
        const auto [beta1, beta2] = options.betas();
        const bool amsgrad = options.amsgrad();

        std::map<std::string, FusedAdamWStepGroup> step_groups;
        for (auto& param : group.params()) {
            if (!param.grad().defined()) continue;

            // gradはzero_grad(set_to_none=true)後に実体が変わるため、毎step再収集する。
            auto grad = param.grad().detach();
            if (grad.is_sparse()) {
                ANET_SYSTEM_ERROR("FusedAdamW does not support sparse gradients.");
            }

            void* key = param.unsafeGetTensorImpl();
            if (state_.find(key) == state_.end()) {
                auto state = std::make_unique<torch::optim::AdamWParamState>();
                state->step(0);
                state->exp_avg(torch::zeros_like(param, param.options(), at::MemoryFormat::Preserve));
                state->exp_avg_sq(torch::zeros_like(param, param.options(), at::MemoryFormat::Preserve));
                if (amsgrad) {
                    state->max_exp_avg_sq(torch::zeros_like(param, param.options(), at::MemoryFormat::Preserve));
                }
                state_[key] = std::move(state);
            }

            auto& state = static_cast<torch::optim::AdamWParamState&>(*state_[key]);
            if (amsgrad && !state.max_exp_avg_sq().defined()) {
                state.max_exp_avg_sq(torch::zeros_like(param, param.options(), at::MemoryFormat::Preserve));
            }

            if (step_tensors_.find(key) == step_tensors_.end()) {
                auto step_options = torch::TensorOptions().dtype(torch::kFloat32).device(param.device());
                step_tensors_[key] = torch::full({}, static_cast<float>(state.step()), step_options);
            }

            // fused adamwは同device・同dtypeのリスト前提なので、混在時も呼び出し単位を分ける。
            auto& step_group = step_groups[MakeFusedAdamWStepGroupKey(param)];
            step_group.params.push_back(param);
            step_group.grads.push_back(grad);
            step_group.exp_avgs.push_back(state.exp_avg());
            step_group.exp_avg_sqs.push_back(state.exp_avg_sq());
            if (amsgrad) {
                step_group.max_exp_avg_sqs.push_back(state.max_exp_avg_sq());
            }
            step_group.state_steps.push_back(step_tensors_[key]);

            // int64 stepはシリアライズ正本、fp32 step tensorはfusedカーネル用。
            // +1前の値でキャッシュを作り、両方を同じ順序で進めるので同期不要。
            state.step(state.step() + 1);
        }

        for (auto& kv : step_groups) {
            auto& step_group = kv.second;
            if (step_group.params.empty()) continue;

            at::_foreach_add_(step_group.state_steps, 1);
            at::_fused_adamw_(
                step_group.params,
                step_group.grads,
                step_group.exp_avgs,
                step_group.exp_avg_sqs,
                step_group.max_exp_avg_sqs,
                step_group.state_steps,
                options.lr(),
                beta1,
                beta2,
                options.weight_decay(),
                options.eps(),
                amsgrad,
                /*maximize=*/false,
                {},
                {});
        }
    }

    return loss;
}

void anet::FusedAdamW::load(torch::serialize::InputArchive& archive)
{
    torch::optim::AdamW::load(archive);

    // 親loadで復元されたint64 stepを正本とし、デバイスstep tensorは次回stepで再構築する。
    step_tensors_.clear();
}


// ======================================================
// GradScaler
// ======================================================

GradScaler::GradScaler(double init_scale, double growth_factor, double backoff_factor, int64_t growth_interval)
    : scale_(init_scale)
    , growth_factor_(growth_factor)
    , backoff_factor_(backoff_factor)
    , growth_interval_(growth_interval)
{
}

void anet::GradScaler::Step(torch::optim::Optimizer& optimizer)
{
    bool found_inf = false;
    if (found_inf_tensor_.defined()) {
        found_inf = found_inf_tensor_.item<float>() != 0.0f;
    }
    if (!found_inf) {
        optimizer.step();
    }

    // 次のupdateのためにフラグ保存
    found_inf_ = found_inf;
}

void anet::GradScaler::Step(torch::optim::Optimizer& optimizer, bool found_inf)
{
    if (!found_inf) {
        // 簡易チェック: まだUnscaleされていない場合などに備えて念のため
        // (外部でclip_grad_norm_前にチェック済みならここはfalseで来るはず)
        optimizer.step();
    }

    // 次のupdateのためにフラグ保存
    found_inf_ = found_inf;
}

void anet::GradScaler::Update(std::optional<bool> found_inf_override)
{
    bool has_inf = found_inf_override.value_or(found_inf_);

    if (has_inf) {
        scale_ *= backoff_factor_;
        growth_tracker_ = 0;
    } else {
        growth_tracker_++;
        if (growth_tracker_ >= growth_interval_) {
            scale_ *= growth_factor_;
            growth_tracker_ = 0;
        }
    }
    if (scale_ < 1.0) scale_ = 1.0;
    found_inf_ = false; // リセット
}

void anet::GradScaler::Unscale_(torch::optim::Optimizer& optimizer)
{
    // defined gradを集め、scale除算とInf/NaN検出を1つのforeach opにまとめる。
    std::vector<torch::Tensor> grads;
    for (auto& group : optimizer.param_groups()) {
        for (auto& param : group.params()) {
            if (param.grad().defined()) {
                grads.push_back(param.grad().detach());
            }
        }
    }

    if (grads.empty()) {
        found_inf_tensor_ = torch::zeros({ 1 }, torch::TensorOptions().dtype(torch::kFloat32));
        return;
    }

    auto device = grads.front().device();
    for (const auto& grad : grads) {
        if (grad.device() != device) {
            ANET_SYSTEM_ERROR("GradScaler::Unscale_ requires gradients on a single device.");
        }
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    found_inf_tensor_ = torch::zeros({ 1 }, options);
    auto inv_scale_tensor = torch::full({ 1 }, static_cast<float>(1.0 / scale_), options);
    at::_amp_foreach_non_finite_check_and_unscale_(grads, found_inf_tensor_, inv_scale_tensor);
}
