// cart1.cpp — Visual Studio + libtorch (CPU/GPU) / A2C & PPO / 並列CartPole / 簡易TensorBoardロガー

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <string>
#include <filesystem>

//==================== Logger (TSV for TensorBoard-like) ====================
struct EventWriter {
    std::ofstream ofs;
    explicit EventWriter(const std::string& path) {
        std::filesystem::create_directories(path);
        ofs.open(path + "/scalars.tsv", std::ios::out);
        ofs << "step\tname\tvalue\n";
    }
    void add_scalar(const std::string& name, double value, int64_t step) {
        ofs << step << "\t" << name << "\t" << value << "\n";
    }
};

//==================== CartPole (batched, device-aware) ====================
struct CartPoleEnv {
    int n_envs;
    torch::Device device;
    torch::Tensor x, x_dot, theta, theta_dot;

    CartPoleEnv(int n_envs_, torch::Device dev) : n_envs(n_envs_), device(dev) { reset(); }

    torch::Tensor reset() {
        torch::NoGradGuard _ng;
        auto f = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        x = torch::rand({ n_envs }, f).uniform_(-0.05f, 0.05f);
        x_dot = torch::zeros({ n_envs }, f);
        theta = torch::rand({ n_envs }, f).uniform_(-0.05f, 0.05f);
        theta_dot = torch::zeros({ n_envs }, f);
        return get_state();
    }

    torch::Tensor get_state() {
        torch::NoGradGuard _ng;
        return torch::stack({ x, x_dot, theta, theta_dot }, 1);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
        step(const torch::Tensor& action) {
        // ★ 環境更新は勾配不要
        torch::NoGradGuard _ng;

        auto f = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        auto force = torch::where(action > 0, torch::full_like(x, 10.0f, f), torch::full_like(x, -10.0f, f));
        auto costh = torch::cos(theta);
        auto sinth = torch::sin(theta);
        auto temp = (force + 0.05f * theta_dot * theta_dot * sinth) / 1.0f;
        auto thetaacc = (9.8f * sinth - costh * temp) / (0.5f * (4.0f / 3.0f - 0.5f * costh * costh));
        auto xacc = temp - 0.05f * thetaacc * costh / 1.0f;

        x = x + 0.02f * x_dot;
        x_dot = x_dot + 0.02f * xacc;
        theta = theta + 0.02f * theta_dot;
        theta_dot = theta_dot + 0.02f * thetaacc;

        auto done = (x.abs() > 2.4f) | (theta.abs() > 0.2095f);
        // 報酬：案③（角度＋位置）— 正規化寄りの安定形
        auto reward = torch::where(done, torch::zeros_like(x),
            0.5f * torch::cos(theta) +
            0.5f * (1.0f - (x.abs() / 2.4f)).clamp(0.0f, 1.0f));

        return { get_state(), reward, done.to(torch::kFloat32) };
    }

    void reset_done() {
        torch::NoGradGuard _ng;
        auto idx = torch::nonzero((x.abs() > 2.4f) | (theta.abs() > 0.2095f)).flatten();
        if (idx.numel() == 0) return;
        auto f = x.options();
        auto rx = torch::rand({ idx.size(0) }, f).uniform_(-0.05f, 0.05f);
        auto rxd = torch::zeros({ idx.size(0) }, f);
        auto rt = torch::rand({ idx.size(0) }, f).uniform_(-0.05f, 0.05f);
        auto rtd = torch::zeros({ idx.size(0) }, f);
        x.index_put_({ idx }, rx);
        x_dot.index_put_({ idx }, rxd);
        theta.index_put_({ idx }, rt);
        theta_dot.index_put_({ idx }, rtd);
    }
};

//==================== Policy + Value ====================
struct PolicyNetImpl : torch::nn::Module {
    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, pi{ nullptr }, v{ nullptr };
    PolicyNetImpl(int state_dim, int hidden_dim, int action_dim) {
        fc1 = register_module("fc1", torch::nn::Linear(state_dim, hidden_dim));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, hidden_dim));
        pi = register_module("pi", torch::nn::Linear(hidden_dim, action_dim));
        v = register_module("v", torch::nn::Linear(hidden_dim, 1));
    }
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        auto logits = pi->forward(x);
        auto value = v->forward(x);
        return { logits, value };
    }
};
TORCH_MODULE(PolicyNet);

//==================== RolloutBuffer (device-aware) ====================
struct RolloutBuffer {
    int T, N;
    torch::Device device;
    torch::Tensor obs, act, logp, rew, done, val, adv, ret;

    RolloutBuffer(int T_, int N_, torch::Device dev) : T(T_), N(N_), device(dev) {}

    void init(int state_dim) {
        torch::NoGradGuard _ng;
        auto f = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        auto i = torch::TensorOptions().dtype(torch::kInt64).device(device);
        obs = torch::zeros({ T, N, state_dim }, f);
        act = torch::zeros({ T, N }, i);
        logp = torch::zeros({ T, N }, f);
        rew = torch::zeros({ T, N }, f);
        done = torch::zeros({ T, N }, f);
        val = torch::zeros({ T, N }, f);
        adv = torch::zeros({ T, N }, f);
        ret = torch::zeros({ T, N }, f);
    }

    torch::Tensor flat(const torch::Tensor& t) { return t.view({ -1, t.size(-1) }); }

    void compute_gae(torch::Tensor last_value, double gamma, double lam) {
        // last_value: [N,1]（no_grad想定）
        torch::NoGradGuard _ng;
        auto gae = torch::zeros({ N }, last_value.options());
        for (int t = T - 1; t >= 0; --t) {
            auto mask = 1.0f - done[t]; // [N]
            auto delta = rew[t] + static_cast<float>(gamma) * last_value.squeeze(-1) * mask - val[t];
            gae = delta + static_cast<float>(gamma * lam) * mask * gae;
            adv[t] = gae;
            last_value = val[t].unsqueeze(-1);
        }
        ret = adv + val;
    }
};

//==================== helpers ====================
static inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
policy_act_logp_value(PolicyNet& net, const torch::Tensor& obs) {
    // ここは推論用途。呼び出し側で NoGradGuard する前提。
    auto [logits, value] = net->forward(obs);          // logits:[N,2], value:[N,1]
    auto logp_all = torch::log_softmax(logits, -1);    // [N,2]
    auto p_all = torch::softmax(logits, -1);        // [N,2]
    auto action = torch::multinomial(p_all, 1).squeeze(-1); // [N] in {0,1}
    auto logp = logp_all.gather(1, action.unsqueeze(1)).squeeze(1); // [N]
    return { action, logp, value.squeeze(-1) }; // value:[N]
}

//==================== A2C Updater ====================
struct A2CUpdater {
    double gamma = 0.99, lam = 0.95;
    double value_coef = 0.5, entropy_coef = 0.01;
    double grad_clip_maxnorm = 0.5;

    void update(PolicyNet& net, RolloutBuffer& buf, torch::optim::Optimizer& opt,
        EventWriter& logger, long long global_step)
    {
        // --- Bootstrap value for GAE (no grad) ---
        torch::NoGradGuard ng_boot;
        auto last_obs = buf.obs.index({ buf.T - 1 });
        auto last_val = std::get<1>(net->forward(last_obs)); // [N,1]
        buf.compute_gae(last_val, gamma, lam);
        // -----------------------------------------

        auto adv = buf.adv;
        adv = (adv - adv.mean()) / (adv.std() + 1e-8f);

        auto obs_f = buf.flat(buf.obs).contiguous();                  // [B,4]
        auto act_f = buf.act.reshape({ buf.T * buf.N });                // [B]
        auto adv_f = adv.reshape({ buf.T * buf.N });                    // [B]
        auto ret_f = buf.ret.reshape({ buf.T * buf.N });                // [B]

        opt.zero_grad();
        auto [logits, value] = net->forward(obs_f);                   // logits:[B,2] value:[B,1]
        auto logp_all = torch::log_softmax(logits, -1);               // [B,2]
        auto picked = logp_all.gather(1, act_f.unsqueeze(1)).squeeze(1); // [B]
        auto entropy = -(logp_all * logp_all.exp()).sum(-1).mean();

        auto policy_loss = -(picked * adv_f).mean();
        auto value_loss = torch::mse_loss(value.squeeze(-1), ret_f);
        auto loss = policy_loss + value_coef * value_loss - entropy_coef * entropy;

        loss.backward();

        // grad clip (double return — MSVC安定)
        double gn_val = torch::nn::utils::clip_grad_norm_(net->parameters(), grad_clip_maxnorm);
        logger.add_scalar("grad/norm", static_cast<float>(gn_val), global_step);

        opt.step();

        logger.add_scalar("loss/policy", policy_loss.item<double>(), global_step);
        logger.add_scalar("loss/value", value_loss.item<double>(), global_step);
        logger.add_scalar("loss/entropy", entropy.item<double>(), global_step);
    }
};

//==================== PPO Updater ====================
struct PPOUpdater {
    double gamma = 0.99, lam = 0.95;
    double value_coef = 0.5, entropy_coef = 0.02;
    double clip_range = 0.1, grad_clip_maxnorm = 0.5;
    int update_epochs = 2, num_minibatches = 8;

    void update(PolicyNet& net, RolloutBuffer& buf, torch::optim::Optimizer& opt,
        EventWriter& logger, long long global_step)
    {
        // --- Bootstrap value for GAE (no grad) ---
        {
            torch::NoGradGuard ng_boot;
            auto last_obs = buf.obs.index({ buf.T - 1 });
            auto last_val = std::get<1>(net->forward(last_obs));
            buf.compute_gae(last_val, gamma, lam);
        }
        // -----------------------------------------

        auto adv = buf.adv;
        adv = (adv - adv.mean()) / (adv.std() + 1e-8f);

        auto obs_f = buf.flat(buf.obs).contiguous();                // [B,4]
        auto act_f = buf.act.reshape({ buf.T * buf.N });              // [B]
        auto oldlp_f = buf.logp.reshape({ buf.T * buf.N });             // [B]
        auto adv_f = adv.reshape({ buf.T * buf.N });                  // [B]
        auto ret_f = buf.ret.reshape({ buf.T * buf.N });              // [B]

        const int64_t B = buf.T * buf.N;
        const int64_t MB = std::max<int64_t>(1, B / num_minibatches);

        std::vector<int64_t> indices(B);
        std::iota(indices.begin(), indices.end(), 0);

        for (int ep = 0; ep < update_epochs; ++ep) {
            std::shuffle(indices.begin(), indices.end(), std::mt19937{ std::random_device{}() });

            for (int64_t start = 0; start < B; start += MB) {
                int64_t end = std::min<int64_t>(start + MB, B);
                std::vector<int64_t> mb_idx(indices.begin() + start, indices.begin() + end);
                auto idx = torch::tensor(mb_idx, torch::TensorOptions().dtype(torch::kLong).device(obs_f.device()));

                // detach して毎ミニバッチで新グラフを構築
                auto mb_obs = obs_f.index_select(0, idx).detach();
                auto mb_act = act_f.index_select(0, idx).to(torch::kLong).detach();
                auto mb_oldlp = oldlp_f.index_select(0, idx).to(torch::kFloat32).detach();
                auto mb_adv = adv_f.index_select(0, idx).to(torch::kFloat32).detach();
                auto mb_ret = ret_f.index_select(0, idx).to(torch::kFloat32).detach();

                opt.zero_grad();
                auto [logits, value] = net->forward(mb_obs);
                auto logp_all = torch::log_softmax(logits, -1);
                auto newlp = logp_all.gather(1, mb_act.unsqueeze(1)).squeeze(1);

                auto ratio = (newlp - mb_oldlp).exp();
                auto clipped = torch::clamp(ratio, 1 - clip_range, 1 + clip_range);
                auto policy_loss = -torch::min(ratio * mb_adv, clipped * mb_adv).mean();
                auto value_loss = torch::mse_loss(value.squeeze(-1), mb_ret);
                auto entropy = -(logp_all * logp_all.exp()).sum(-1).mean();

                auto loss = policy_loss + value_coef * value_loss - entropy_coef * entropy;

                loss.backward(); // ミニバッチ毎に1回。retain_graph不要（都度新しいforward）

                double gn_val = torch::nn::utils::clip_grad_norm_(net->parameters(), grad_clip_maxnorm);
                logger.add_scalar("grad/norm", static_cast<float>(gn_val), global_step);

                opt.step();

                logger.add_scalar("loss/policy", policy_loss.item<double>(), global_step);
                logger.add_scalar("loss/value", value_loss.item<double>(), global_step);
                logger.add_scalar("loss/entropy", entropy.item<double>(), global_step);
            }
        }
    }
};

//==================== Main with correct rollout ====================
int main() {
    torch::manual_seed(0);
    const int N = 256;       // 並列環境数
    const int T = 128;       // ロールアウト長
    const int state_dim = 4;
    const int hidden = 128;
    const int action_dim = 2;

    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "device: " << (device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;

    PolicyNet net(state_dim, hidden, action_dim);
    net->to(device);
    torch::optim::Adam opt(net->parameters(), 1e-4);
    EventWriter logger("runs/cart1");

    CartPoleEnv env(N, device);
    auto obs = env.reset();  // [N,4] on device

    RolloutBuffer buf(T, N, device);
    buf.init(state_dim);

    A2CUpdater a2c;
    PPOUpdater ppo;

    // ---- アルゴリズム切替 ----
    const bool USE_PPO = true;

    for (int update = 0; update < 50; ++update) {
        // ===== 1) Rollout を T ステップ収集 =====
        for (int t = 0; t < T; ++t) {
            torch::NoGradGuard _ng;  // 推論＋保存は勾配不要

            auto [action, logp, value] = policy_act_logp_value(net, obs); // [N],[N],[N]

            auto [next_obs, reward, done] = env.step(action);
            env.reset_done();

            // detachして保存（index_put_もno_grad内）
            buf.obs.index_put_({ t }, obs.detach());
            buf.act.index_put_({ t }, action.detach());
            buf.logp.index_put_({ t }, logp.detach());
            buf.val.index_put_({ t }, value.detach());
            buf.rew.index_put_({ t }, reward.detach());
            buf.done.index_put_({ t }, done.detach());

            obs = next_obs;
        }

        // ===== 2) Update =====
        if (USE_PPO)  ppo.update(net, buf, opt, logger, update);
        else          a2c.update(net, buf, opt, logger, update);

        // 参考：平均報酬ログ
        double mean_rew = buf.rew.mean().item<double>();
        logger.add_scalar("train/return_mean", mean_rew, update);
        std::cout << (USE_PPO ? "[PPO]" : "[A2C]") << " update=" << update
            << " mean_rew=" << std::fixed << std::setprecision(4) << mean_rew << std::endl;
    }

    return 0;
}
