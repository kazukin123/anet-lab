// DropMergeEnv.hpp
#pragma once

#include <memory>
#include <vector>
#include <optional>
#include <set>
#include <box2d/box2d.h>
#include <torch/torch.h>
#include "anet/random.hpp"
#include "anet/config.hpp"
#include "anet/rl.hpp"

namespace anet::rl::env::drop_merge {

    constexpr int kActionNoop = 0;
    constexpr int kActionLeft = 1;
    constexpr int kActionDrop = 2;
    constexpr int kActionRight = 3;
    constexpr int kActionFastLeft = 4;
    constexpr int kActionFastRight = 5;

    constexpr int kFruitTypeCount = 11; // 11種類 (Rank 1..11)


    enum class SeedMode {
        Normal,         ///< 初期化時のみSeed指定（Factory由来）。Resetでは変更しない（現状通り）。
        Fixed,          ///< 初期化時のSeed（Factory由来）で、毎Reset時にRNGをリセットする。
        GlobalFixed     ///< 全環境で共通の設定値（global_seed）を使用し、毎Reset時にRNGをリセットする。
    };

    /// DropMerge 環境の設定
    struct DropMergeEnvConfig : public anet::Config {

        // --- Seed制御 ---
        std::string seed_mode = "normal";   ///< "normal", "fixed", "global_fixed"
        seed_t global_seed = -1;           ///< global_fixedモード時のSeed値 (-1: Auto)

        // --- 環境パラメータ ---
        int max_step = 3000;
        int no_drop_timeout_steps = 200;
        float box_width = 3.0f;
        float box_height = 4.0f;
        float ground_y = 0.5f;     // 箱の底の高さ
        float gravity = -10.0f;

        // --- グリッド観測パラメータ ---
        int grid_rows = 30;
        int grid_cols = 30;

        // --- ゲームプレイパラメータ ---
		bool use_fast_move = false;     ///< 高速移動モード
        bool use_instant_drop = false;  ///< 即時ドロップモード
        bool use_dropper_x_grid = true; ///< DropperのX座標をGridの最上段に描画する
        float dropper_speed = 0.05f;    ///< 1ステップあたりのDropper移動量
        float dropper_speed2 = 0.30f;   ///< 1ステップあたりのDropper移動量(FAST)
        float pop_force = 1.0f;         ///< 合体時の弾き飛ばし力
        int reload_min_steps = 20;      ///< Drop抑止ステップ数（物理判定が早くても必ず待つ時間）
        int reload_max_steps = 300;     ///< Drop抑止タイムアウトステップ数（物理判定が効かない場合の強制解除）
		bool noop_override = false;     ///< NOOPアクションを中央方向移動に上書きするか
		int game_over_grace_step = 60;  ///< 上端から溢れてからゲームオーバーとするまでの猶予ステップ数      
        float drop_noise = 0.01f;       ///< Drop時のX座標ノイズ 
        float spin_noise = 0.0f;        ///< Drop時の初期角速度ノイズ(rad/s)

        // --- 報酬調整用パラメータ ---
        float time_penalty = -0.0001f;     ///< 毎ステップ引かれる罰報酬
        float noop_penalty = -0.001f;     ///< NOOPアクションを選ぶ事による罰報酬
        float game_over_penalty = -5.0f;   ///< ゲームオーバー時の罰報酬

        // --- 果物パラメータ (Rank 1 -> 10) ---
		std::vector<float> fruit_radii;     ///< ランクごとの半径
		std::vector<float> fruit_densities; ///< ランクごとの密度
		std::vector<float> fruit_scores;    ///< ランクごとのスコア値
        std::vector<float> drop_probs;      ///< ランクの出現重み
        float restitution = 0.1f;           ///< 反発係数
        float friction = 0.5f;              ///< 摩擦係数
        float damping = 0.5f;               ///< 空気抵抗

        DropMergeEnvConfig(
            const anet::ConfigData& config_data = anet::EmptyConfigData,
            const std::string& config_prefix = "")
            : anet::Config(config_data, "DropMergeEnv", config_prefix)
        {
            ANET_READ_CONFIG(config_data, seed_mode);
            ANET_READ_CONFIG(config_data, global_seed);
            ANET_READ_CONFIG(config_data, max_step);
            ANET_READ_CONFIG(config_data, no_drop_timeout_steps);
            ANET_READ_CONFIG(config_data, box_width);
            ANET_READ_CONFIG(config_data, box_height);
            ANET_READ_CONFIG(config_data, ground_y);
            ANET_READ_CONFIG(config_data, gravity);
            ANET_READ_CONFIG(config_data, grid_rows);
            ANET_READ_CONFIG(config_data, grid_cols);
            ANET_READ_CONFIG(config_data, use_fast_move);
            ANET_READ_CONFIG(config_data, use_instant_drop);
            ANET_READ_CONFIG(config_data, use_dropper_x_grid);
            ANET_READ_CONFIG(config_data, dropper_speed);
            ANET_READ_CONFIG(config_data, dropper_speed2);
            ANET_READ_CONFIG(config_data, pop_force);
            ANET_READ_CONFIG(config_data, reload_min_steps);
            ANET_READ_CONFIG(config_data, reload_max_steps);
            ANET_READ_CONFIG(config_data, noop_override);
            ANET_READ_CONFIG(config_data, game_over_grace_step);
            ANET_READ_CONFIG(config_data, drop_noise);
            ANET_READ_CONFIG(config_data, spin_noise);
            ANET_READ_CONFIG(config_data, time_penalty);
            ANET_READ_CONFIG(config_data, noop_penalty);
            ANET_READ_CONFIG(config_data, game_over_penalty);
            ANET_READ_CONFIG(config_data, restitution);
            ANET_READ_CONFIG(config_data, friction);
            ANET_READ_CONFIG(config_data, damping);

            // デフォルト値 (Configファイルがない場合用)を定義
            std::vector<float> def_radii = {
                0.09f, 0.13f, 0.17f, 0.22f, 0.28f,
                0.35f, 0.43f, 0.52f, 0.62f, 0.72f,
                0.84f
            };
            std::vector<float> def_scores = {
                0.1f, 0.3f, 0.6f, 1.0f, 1.5f,
                2.1f, 2.8f, 3.6f, 4.5f, 5.5f,
                6.6f
            };
            std::vector<float> def_densities(11, 5.0f);
            std::vector<float> def_probs = { 20.0f, 20.0f, 20.0f, 20.0f, 20.0f };

            // デフォルト値を入れる
            fruit_radii = def_radii;
            fruit_densities = def_densities;
            fruit_scores = def_scores;
            drop_probs = def_probs;

			// Config読み込み
            ANET_READ_CONFIG(config_data, fruit_radii);
            ANET_READ_CONFIG(config_data, fruit_densities);
            ANET_READ_CONFIG(config_data, fruit_scores);
            ANET_READ_CONFIG(config_data, drop_probs);

            // サイズチェック
            if (fruit_radii.size() != kFruitTypeCount ||
                fruit_densities.size() != kFruitTypeCount ||
                fruit_scores.size() != kFruitTypeCount) {
                ANET_SYSTEM_ERROR("Invalid fruit config.");
            }
        }
    };

    /// DropMerge (Suika-like) 環境クラス
    class DropMergeEnv : public anet::rl::SingleDiscreteEnv, public anet::RandomHolder, public std::enable_shared_from_this<DropMergeEnv> {
    public:
        DropMergeEnv(
            const DropMergeEnvConfig& config,
            const torch::Device& device,
            const std::optional<anet::seed_t> seed = std::nullopt);

        ~DropMergeEnv() override;

        anet::rl::EnvSpec GetSpec() const override;
        std::shared_ptr<const anet::rl::SingleResetResult> Reset(anet::rl::RunMode mode = anet::rl::RunMode::Train) override;
        std::shared_ptr<const anet::rl::SingleStepResult> Step(int64_t action, anet::rl::RunMode mode = anet::rl::RunMode::Train) override;

    public:
        // デバッグ・可視化用
        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index = -1) const override;
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index = -1) const override;
        
    private:
        class Result;
        class ResetResult;
        class StepResult;

        struct MergeRequest {
            b2Body* bodyA;
            b2Body* bodyB;
            b2Vec2 center;
            int next_rank;
        };

        struct DropperState {
            float x = 0.0f;                 ///< 現在のX座標 (World座標)
            int current_rank = 1;           ///< 現在持っている果物のランク
            int next_rank = 1;              ///< 次の果物のランク
            int wait_timer = 0;             ///< max_stepsタイマー
			int min_wait_timer = 0;         ///< min_stepsタイマー
            bool is_busy = false;           ///< 投下動作中
            b2Body* pending_body = nullptr; ///< 落下判定待ちの果物
        };

        enum class TerminationReason {        ///< メトリクス集計用
            None,
            Timeout,
            SpawnBlocked,
            Overflow,
            MaxStep
        };

        // 衝突コールバック
        class ContactListener : public b2ContactListener {
        public:
            explicit ContactListener(DropMergeEnv& env) : env_(env) {}
            void BeginContact(b2Contact* contact) override;
        private:
            DropMergeEnv& env_;
        };
    private:
        void buildWorld();
        void destroyWorld();
        bool isSpawnAreaClear(float x, float y, float r) const;
        void updateDropperStatus();

        // ゲームロジック
        void notifyContact(b2Body* body);
        void processAction(int64_t action);
        //void updateDropper();
        b2Body* spawnFruit(float x, float y, int rank);
        void processMerges();
        void applyExplosion(const b2Vec2& center, float force);
        bool checkGameOver();
        int determineNextRank();

        // 観測・報酬
        anet::rl::SingleState makeState() const;
        std::pair<float, float> calcReward();
        anet::rl::AuxData CreateAuxData(float reward, float raw_reward) const;
    private:
        // 設定情報
        DropMergeEnvConfig config_;

        // Seed管理
        SeedMode seed_mode_ = SeedMode::Normal;
        anet::seed_t initial_seed_ = 0;

        // 使いまわし
        torch::TensorOptions float_opt_;
        torch::TensorOptions bool_opt_;

        // Box2d
        std::unique_ptr<b2World> world_;
        std::unique_ptr<ContactListener> contact_listener_;
        b2Body* ground_body_ = nullptr;

        // 状態管理
        DropperState dropper_;
        int step_count_ = 0;
        bool game_over_ = false;
        int game_over_timer_ = 0;
        int steps_since_last_drop_ = 0;
        torch::Tensor obs_buffer_;  ///< obsのバッファ
        float* obs_ptr_ = nullptr;  ///< obsのバッファのポインタ

        // マージ処理用
        std::vector<MergeRequest> merge_requests_;
        std::set<b2Body*> bodies_to_destroy_;
        float current_step_merge_score_ = 0.0f; ///< ステップ内で発生した合体スコア
        float episode_score_ = 0.0f;            ///< エピソード累積スコア

        // aux用 (Resetで初期化しない)
        float last_episode_score_ = -1.0f;
        int last_episode_step_ = -1;
        float episode_reward_ = 0.0f;           ///< エピソード累積報酬 (Penalty込み)
        float last_episode_reward_ = 0.0f;      ///< 前回のエピソード累積報酬

        // メトリクス用
        TerminationReason term_reason_ = TerminationReason::None;   ///< エピーソード終了要因
        bool episode_just_ended_ = false; ///< GetScalarで値を返す判定用
        int ep_max_rank_ = 0;             ///< エピソード中の最大ランク
        int ep_end_fruit_count_ = 0;      ///< エピソード終了時のフルーツ数

        // デバッグ・観測用キャッシュ
        mutable std::vector<float> grid_cache_;
    };

    class DropMergeEnvFactory : public anet::rl::SingleDiscreteEnvFactory {
    public:
        DropMergeEnvFactory() = default;
        std::string GetTargetEnvClassId() const override { return "DropMergeEnv"; }

        std::shared_ptr<anet::rl::SingleDiscreteEnv> CreateSingleEnv(
            const anet::ConfigData& config_data,
            const torch::Device& device,
            std::optional<anet::seed_t> seed = std::nullopt,
            const std::string& config_prefix = "") override;
    };

}
