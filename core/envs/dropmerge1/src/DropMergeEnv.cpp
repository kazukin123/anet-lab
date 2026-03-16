// DropMergeEnv.cpp
#include "DropMergeEnv.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include "anet/profile.hpp"
#include "anet/log.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/env.hpp"

using namespace anet::rl::env::drop_merge;
namespace LOG = anet::log;

constexpr int kNumScalarObsDim = 4;

// -------------------------------------------------------------
// Constants & UserData definition
// -------------------------------------------------------------

// Box2DのUserDataに格納する情報の定義
enum class BodyType : uintptr_t {
    Ground = 0,
    Fruit = 1
};

struct FruitUserData {
    BodyType type;
    int rank; // 1 to 10
};

// UserDataポインタを管理するための簡易プール等は省略し、
// ここでは b2BodyUserData.pointer に直接キャストした値を埋め込む方式はとらず、
// 各Bodyごとにnewして管理する。（削除時にdeleteが必要）
// あるいは、rank情報は小さいので pointer 値そのものにエンコードする。
// pointer = (type << 16) | rank 
// type: 1=Fruit, rank: 1~10.  => Fruit(Rank N) = 0x10000 + N

static uintptr_t EncodeUserData(BodyType type, int rank = 0) {
    return (static_cast<uintptr_t>(type) << 16) | static_cast<uintptr_t>(rank);
}

static std::pair<BodyType, int> DecodeUserData(uintptr_t val) {
    BodyType type = static_cast<BodyType>(val >> 16);
    int rank = static_cast<int>(val & 0xFFFF);
    return { type, rank };
}


// -------------------------------------------------------------
// Result Classes
// -------------------------------------------------------------

class DropMergeEnv::Result : virtual public anet::rl::SingleEnvResult {
public:
    Result(std::shared_ptr<const DropMergeEnv> env, float reward, float raw_reward, bool capture_now)
        : env_(env), reward_(reward), raw_reward_(raw_reward)
    {
        if (capture_now) {
            cached_aux_ = env->CreateAuxData(reward, raw_reward);
            has_cache_ = true;
        }
    }
    anet::rl::AuxData GetAuxData() const override
    {
        // キャッシュがあればそれを返す（Reset後でも大丈夫）
        if (has_cache_) {
            return cached_aux_;
        }

        // キャッシュがなければ、今のEnvから生成する（通常時はこちら）
        return env_->CreateAuxData(reward_, raw_reward_);
    }
protected:
    std::shared_ptr<const DropMergeEnv> env_;
    float reward_;
    float raw_reward_;
    bool has_cache_ = false;
    anet::rl::AuxData cached_aux_;
};

class DropMergeEnv::ResetResult : public anet::rl::SingleResetResult, public DropMergeEnv::Result {
public:
    ResetResult(std::shared_ptr<const DropMergeEnv> env, const anet::rl::SingleState state)
        : Result(env, 0.0f, 0.0f, false)
        , SingleResetResult(std::move(state))
    {
    }
};

class DropMergeEnv::StepResult : public anet::rl::SingleStepResult, public DropMergeEnv::Result {
public:
    StepResult(std::shared_ptr<const DropMergeEnv> env, float reward, float raw_reward, anet::rl::SingleState next_state)
//		: Result(env, reward, raw_reward, next_state.done || next_state.truncated)  // エピソード終了時は断面キャプチャ
		: Result(env, reward, raw_reward, false)
        , SingleStepResult(reward, std::move(next_state))
    {
    }
};


// -------------------------------------------------------------
// DropMergeEnv Implementation
// -------------------------------------------------------------

DropMergeEnv::DropMergeEnv(
    const DropMergeEnvConfig& config,
    const torch::Device& device,
    const std::optional<anet::seed_t> seed)
    : anet::RandomHolder(std::nullopt)
    , config_(config)
{
    // メトリクスログなど
    anet::MetricsLogger::Instance()->Log("DropMergeEnv", config_);

    // --- Seed Mode の解析 ---
    std::string mode_str = anet::ToLower(config_.seed_mode);
    if (mode_str == "fixed") {
        seed_mode_ = SeedMode::Fixed;
    } else if (mode_str == "global_fixed") {
        seed_mode_ = SeedMode::GlobalFixed;
        anet::json seed_info = {
            {"mode", "global_fixed"},
            {"global_seed", config_.global_seed}
		};
        anet::MetricsLogger::Instance()->Log("DropMergeEnv_seed", seed_info);
    } else {
        seed_mode_ = SeedMode::Normal;
    }

    // --- Initial Seed の決定 ---
    if (seed_mode_ == SeedMode::GlobalFixed) {
        // GlobalFixed: 設定ファイルの値を優先。未設定(-1)ならオート生成だが、
        // 「固定」なのでオート生成した値を保存して使い回す。
        if (config_.global_seed >= 0) {
            initial_seed_ = static_cast<anet::seed_t>(config_.global_seed);
        } else {
            initial_seed_ = anet::SeedMaker::MakeAutoSeed();
        }
    } else {
        // Normal / Fixed: Factoryから渡されたSeedを使用 (なければオート)
        initial_seed_ = seed.value_or(anet::SeedMaker::MakeAutoSeed());
    }

    // RNGの初期化 (RandomHolder::rnd_ はコンストラクタで生成済み)
    // RandomHolderのコンストラクタにnulloptを渡したので、ここでSetSeedする
    SetSeed(initial_seed_);

	// TensorOptionsの初期化
    float_opt_ = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    bool_opt_ = torch::TensorOptions().dtype(torch::kBool).device(device);

    // Obsバッファ初期化
    int total_dim = kNumScalarObsDim + config_.grid_rows * config_.grid_cols;
    obs_buffer_ = torch::empty({ total_dim }, float_opt_);
    obs_ptr_ = obs_buffer_.data_ptr<float>();

    // ActionMode設定を解釈
    std::string am = anet::ToLower(config_.action_mode);
    if (am == "move") action_mode_ = ActionMode::Move;
    else if (am == "direct") action_mode_ = ActionMode::Direct;
    else if (am == "direct_noop") action_mode_ = ActionMode::DirectNoop;
    else action_mode_ = ActionMode::MoveFast;

    // DROP座標数を設定から生成
    num_drop_actions_ = (config_.drop_divisions > 0) ? config_.drop_divisions : config_.grid_cols;

    // 世界初期構築
    buildWorld();
}

DropMergeEnv::~DropMergeEnv()
{
    destroyWorld();
}

anet::rl::EnvSpec DropMergeEnv::GetSpec() const
{
    // Dropper Info (Fixed size)
    //    [0]: dropper_x (normalized -1~1)
    //    [1]: current_rank (normalized 0~1)
    //    [2]: next_rank (normalized 0~1)
    //    [3]: is_busy (0 or 1)
    // Grid Info (Variable size: rows * cols)
    //    [4...]: grid cell value (0.0=Empty, 0.1=Rank1 ... 1.0=Rank10)

    //状態空間定義
    int grid_dim = config_.grid_rows * config_.grid_cols;
    anet::rl::StateSpec state_spec;
    state_spec.shape = { static_cast<int64_t>(kNumScalarObsDim + grid_dim) };

    state_spec.dims = {
        { {0}, -1.0f, 1.0f, "dropper_x", "Dropper X position" },
        { {1},  0.0f, 1.0f, "current_rank", "Current fruit rank" },
        { {2},  0.0f, 1.0f, "next_rank", "Next fruit rank" },
        { {3},  0.0f, 1.0f, "is_busy", "Dropper busy flag" },
        /// @todo Grid部分は省略
    };

    // 離散アクション
    anet::rl::ActionSpec action_spec;
    action_spec.is_discrete = true;

    // モードに応じたアクションラベルの動的生成
    if (action_mode_ == ActionMode::Move) {
        action_spec.value_labels = { "NOOP", "LEFT", "DROP", "RIGHT" };
    } else if (action_mode_ == ActionMode::MoveFast) {
        action_spec.value_labels = { "NOOP", "LEFT", "DROP", "RIGHT", "F_LEFT", "F_RIGHT" };
    } else if (action_mode_ == ActionMode::Direct) {
        for (int i = 0; i < num_drop_actions_; ++i) {
            action_spec.value_labels.push_back("DROP_" + std::to_string(i));
        }
    } else if (action_mode_ == ActionMode::DirectNoop) {
        action_spec.value_labels.push_back("NOOP");
        for (int i = 0; i < num_drop_actions_; ++i) {
            action_spec.value_labels.push_back("DROP_" + std::to_string(i));
        }
    }

    anet::rl::EnvSpec env_spec;
    env_spec.state_spec = state_spec;
    env_spec.action_spec = action_spec;
    env_spec.reward_range = { 0.0f, 10000.0f }; /// @todo スコア青天井

    return env_spec;
}

void DropMergeEnv::destroyWorld()
{
    if (world_) {
        // UserDataのポインタ管理はしていない（EncodeUserData使用）ので単にWorldを破棄すればOK
        contact_listener_.reset();
        world_.reset();
    }
    bodies_to_destroy_.clear();
    merge_requests_.clear();
}

void DropMergeEnv::buildWorld()
{
    destroyWorld();

    b2Vec2 gravity(0.0f, config_.gravity);
    world_ = std::make_unique<b2World>(gravity);
    world_->SetContinuousPhysics(false);  // 性能確保のために連続衝突判定(CCD) をオフに（すり抜けが発生したらtrueにする）
//world_->SetContinuousPhysics(true);  // 性能確保のために連続衝突判定(CCD) をオフに（すり抜けが発生したらtrueにする）
    world_->SetAllowSleeping(true);     // 動かなくなった果物の物理演算をスキップ(デフォルトで有効のはずだが念の為）

    contact_listener_ = std::make_unique<ContactListener>(*this);
    world_->SetContactListener(contact_listener_.get());

    // --- コンテナ（箱）の作成 ---
    {
        b2BodyDef bd;
        bd.type = b2_staticBody;
        bd.position.Set(0.0f, config_.ground_y);
        ground_body_ = world_->CreateBody(&bd);
        ground_body_->GetUserData().pointer = EncodeUserData(BodyType::Ground);

        b2FixtureDef fd;
        fd.density = 0.0f;
        fd.friction = config_.box_friction < 0 ? config_.friction : config_.box_friction;  // 摩擦係数
        fd.restitution = config_.box_restitution < 0 ? config_.restitution : config_.box_restitution; // 反発係数

        float half_w = config_.box_width * 0.5f;
        float h = config_.box_height;
        float wall_thick = 0.2f;

        // 底
        b2PolygonShape shape_bottom;
        shape_bottom.SetAsBox(half_w, wall_thick, b2Vec2(0.0f, -wall_thick), 0.0f);
        fd.shape = &shape_bottom;
        ground_body_->CreateFixture(&fd);

        // 左壁
        b2PolygonShape shape_left;
        shape_left.SetAsBox(wall_thick, h, b2Vec2(-half_w - wall_thick, h), 0.0f);
        fd.shape = &shape_left;
        ground_body_->CreateFixture(&fd);

        // 右壁
        b2PolygonShape shape_right;
        shape_right.SetAsBox(wall_thick, h, b2Vec2(half_w + wall_thick, h), 0.0f);
        fd.shape = &shape_right;
        ground_body_->CreateFixture(&fd);

        // ゲームオーバー判定ラインより少し上(+2.0f)に蓋を設置して
        // 「一瞬跳ねただけ」ならセーフ、「積み上がって詰まった」ならアウトになる余地を作る
        //float ceiling_y = h + 2.0f;
        //b2PolygonShape shape_top;
        //shape_top.SetAsBox(half_w, wall_thick, b2Vec2(0.0f, ceiling_y), 0.0f);
        //fd.shape = &shape_top;
        //ground_body_->CreateFixture(&fd);
    }
}

int DropMergeEnv::determineNextRank()
{
    const auto& probs = config_.drop_probs;
    if (probs.empty()) return 1;

    // 重みの合計を計算
    float total = 0.0f;
    for (float p : probs) total += p;
    if (total <= 0.0001f) return 1; // 安全策

    // 0 ~ total の乱数
    float r = rnd_->Uniform(0.0f, total);

	// 確率分布に基づいてランク決定(ルーレット法)
    float current = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        current += probs[i];
        if (r < current) {
            return (int)(i + 1); // Rankは1始まり
        }
    }

    return (int)probs.size();
}

std::shared_ptr<const anet::rl::SingleResetResult> DropMergeEnv::Reset(anet::rl::RunMode mode)
{
    anet::ProfileRange r("DropMergeEnv::Reset");

    // --- Seed Reset Logic ---
    // Normal: 何もしない (継続性維持)
    // Fixed / GlobalFixed: 毎回同じSeedに戻す (完全再現)
    if (seed_mode_ != SeedMode::Normal) {
        SetSeed(initial_seed_);
    }

    buildWorld();

    step_count_ = 0;
    steps_since_last_drop_ = 0;
    game_over_ = false;
    game_over_timer_ = 0;
    episode_score_ = 0.0f;
    episode_reward_ = 0.0f;

    // Dropper初期化
    dropper_.x = 0.0f;
    dropper_.current_rank = determineNextRank();
    dropper_.next_rank = determineNextRank();
    dropper_.wait_timer = 0;
    dropper_.is_busy = false;

    auto state = makeState();
    state.episode_start = true;

    return std::make_shared<ResetResult>(this->shared_from_this(), std::move(state));
}

bool DropMergeEnv::isSpawnAreaClear(float x, float y, float r) const
{
    constexpr float kOverlapMargin = 0.95f;

    for (b2Body* b = world_->GetBodyList(); b; b = b->GetNext()) {
        if (b->GetType() != b2_dynamicBody) continue;

        // 自分自身（生成前なので本来ないはずだが念のため）は除外
        if (b == dropper_.pending_body) continue;

        b2Vec2 pos = b->GetPosition();

        // UserDataから半径を取得
        auto data = DecodeUserData(b->GetUserData().pointer);
        if (data.first != BodyType::Fruit) continue;

        float r_other = config_.fruit_radii[data.second - 1];
        float dist_sq = (pos.x - x) * (pos.x - x) + (pos.y - y) * (pos.y - y);
        float radius_sum = (r + r_other) * kOverlapMargin;

        // 接触（重なり）判定
        if (dist_sq < radius_sum * radius_sum) {
            return false; // 埋まっている
        }
    }
    return true;
}

b2Body* DropMergeEnv::spawnFruit(float x, float y, int rank)
{
    if (rank < 1 || rank > kFruitTypeCount) return nullptr;

    b2BodyDef bd;
    bd.type = b2_dynamicBody;
    bd.position.Set(x, y);
    bd.linearDamping = config_.damping;
    bd.angularDamping = config_.damping;
    b2Body* body = world_->CreateBody(&bd);

    // UserData設定
    body->GetUserData().pointer = EncodeUserData(BodyType::Fruit, rank);

    b2CircleShape shape;
    shape.m_radius = config_.fruit_radii[rank - 1];

    b2FixtureDef fd;
    fd.shape = &shape;
    fd.density = config_.fruit_densities[rank - 1];
    fd.restitution = config_.restitution;
    fd.friction = config_.friction;
    

    body->CreateFixture(&fd);

    if (config_.spin_noise > 0.0f) {
        float spin = rnd_->Uniform(-config_.spin_noise, config_.spin_noise);
        body->SetAngularVelocity(spin);
    }
    return body;
}

void DropMergeEnv::processAction(int64_t action)
{
    bool execute_drop = false;

    if (action_mode_ == ActionMode::Direct || action_mode_ == ActionMode::DirectNoop) {
        // --- 座標直接指定モード ---

        int drop_col = -1;
        if (action_mode_ == ActionMode::Direct) {
            drop_col = action;
        } else { // DirectNoop
            if (action == 0) return; // NOOP
            drop_col = action - 1;
        }

        if (drop_col >= 0 && drop_col < num_drop_actions_) {
            if (dropper_.is_busy) return; // 落下中なら無視

            // 箱の有効幅の中央へ座標をマッピング
            float min_x = -config_.box_width * 0.5f;
            float max_x = config_.box_width * 0.5f;
            float cell_w = (max_x - min_x) / num_drop_actions_;
            dropper_.x = min_x + (drop_col + 0.5f) * cell_w;

            execute_drop = true;
        }
    } else {
        // --- 移動モード ---

        if (config_.noop_override && action == kActionNoop) {
            if (dropper_.x > 0.0f) action = kActionLeft;
            else action = kActionRight;
        }

        // 移動範囲の計算準備
        float half_w = config_.box_width * 0.5f;
        int check_rank = (dropper_.current_rank > 0) ? dropper_.current_rank : dropper_.next_rank;
        if (check_rank < 1) check_rank = 1;
        float r = config_.fruit_radii[check_rank - 1];
        float margin = 0.05f;
        float base_limit = half_w - r - margin; // 壁にめり込まない範囲

        // 左の方が移動範囲が少し広い
        float r_cherry = config_.fruit_radii[0]; // Rank 1 radius
        float asymmetry_offset = r_cherry / 3.0f;
        float limit_left = base_limit + asymmetry_offset;
        float limit_right = base_limit;

        // 移動処理
        if (action == kActionLeft) dropper_.x -= config_.dropper_speed;
        else if (action == kActionRight) dropper_.x += config_.dropper_speed;
        else if (action == kActionFastLeft) dropper_.x -= config_.dropper_speed2;
        else if (action == kActionFastRight) dropper_.x += config_.dropper_speed2;
        else if (action == kActionDrop) execute_drop = true;

        // 端ワープ (移動モードのみ)
        float total_width = limit_right - (-limit_left);
        if (dropper_.x > limit_right) {// 右にはみ出した場合
            while (dropper_.x > limit_right) dropper_.x -= total_width; // 超えた分だけ、左端(-limit_left)から右に進める
        } else if (dropper_.x < -limit_left) {  // 左にはみ出した場合
            while (dropper_.x < -limit_left) dropper_.x += total_width; // 超えた分だけ、右端(limit_left)から左に進める
        }
    }

    // --- 共通のDROP処理 ---
    if (execute_drop) {
        if (dropper_.is_busy) return;

        float spawn_y = config_.ground_y + config_.box_height;
        float r_drop = config_.fruit_radii[dropper_.current_rank - 1];

        // ノイズ計算
        float noise = (config_.drop_noise > 0.0f) ? rnd_->Uniform(-config_.drop_noise, config_.drop_noise) : 0.0f;
        float actual_x = dropper_.x + noise;

        // 壁めり込み防止クランプ
        float half_w = config_.box_width * 0.5f;
        float limit = half_w - r_drop - 0.01f;
        actual_x = std::clamp(actual_x, -limit, limit);

        // 置けない状態でDROPしたらGameOver
        if (!isSpawnAreaClear(actual_x, spawn_y, r_drop)) {
            game_over_ = true;
            term_reason_ = TerminationReason::SpawnBlocked;
            LOG::verbose() << "Game Over: Spawn area blocked. episode_score=" << episode_score_ << " step_count=" << step_count_ << " x=" << dropper_.x ;
            return;
        }

        // 落とした果物で最大ランクを更新
        ep_max_rank_ = std::max(ep_max_rank_, dropper_.current_rank);

        // 新しい果物つくる
        dropper_.pending_body = spawnFruit(actual_x, spawn_y, dropper_.current_rank);

        // リロード待ちに入る（タイマーセット）
        dropper_.wait_timer = config_.reload_max_steps;
        dropper_.min_wait_timer = config_.reload_min_steps;
        dropper_.is_busy = true;
        dropper_.current_rank = 0;  // 手持ちを空に（表示用）
    }
}

/// 衝突通知
void DropMergeEnv::notifyContact(b2Body* body)
{
    // 今待っている果物が何かにぶつかったらDrop待ち状態を解除
    if (dropper_.pending_body == body) {
        dropper_.pending_body = nullptr;
    }
}

void DropMergeEnv::processMerges()
{
    current_step_merge_score_ = 0.0f;

    // マージリクエスト処理
    // 複数のペアが登録されているが、すでに削除済み（bodies_to_destroy_入り）のものはスキップ
    for (const auto& req : merge_requests_) {
        bool a_removed = (bodies_to_destroy_.count(req.bodyA) > 0);
        bool b_removed = (bodies_to_destroy_.count(req.bodyB) > 0);

        if (a_removed || b_removed) {
            continue; // すでにどちらかがマージ処理された
        }

        if (req.next_rank <= kFruitTypeCount) {
            // 新しい果物を生成 
            spawnFruit(req.center.x, req.center.y, req.next_rank);

            // 合体後のランクで最大ランクを更新
            ep_max_rank_ = std::max(ep_max_rank_, req.next_rank);

			// スコア加算
            float s = config_.fruit_scores[req.next_rank - 1];
            current_step_merge_score_ += s;
            episode_score_ += s;

            // ログ
            if (req.next_rank >= kFruitTypeCount) { // スイカが出来たらログ＆音
                LOG::info() << "Merged fruits into Rank [ " << req.next_rank << " ] episode_score_=" << episode_score_ << " current_step_merge_score_=" << current_step_merge_score_;
                wxBell();       /// @todo wxBell()はスレッドセーフじゃないのでwxSoundを使うべき
            }

            // 小爆発
            applyExplosion(req.center, config_.pop_force);
        } else {
            // 最大ランクを更新
            ep_max_rank_ = std::max(ep_max_rank_, req.next_rank);

            // スイカ同士が消えた場合はSpawnしない（Rank 12相当）
            LOG::info() << "Merged fruits into Rank [ " << req.next_rank << " ] episode_score_=" << episode_score_ << " current_step_merge_score_=" << current_step_merge_score_;
            wxBell();       /// @todo wxBell()はスレッドセーフじゃないのでwxSoundを使うべき

            // スコア加算
            float s = config_.fruit_scores[kFruitTypeCount - 1];
            current_step_merge_score_ += s;
            episode_score_ += s;

            // スイカ消滅時の爆発…しない？
            //applyExplosion(req.center, config_.pop_force * 2.0f);
        }

        // 古い果物を削除予約
        bodies_to_destroy_.insert(req.bodyA);
        bodies_to_destroy_.insert(req.bodyB);
    }

	// マージ処理終わったのでクリア
    merge_requests_.clear();

    // 予約されていた古い果物削除を実行
    for (b2Body* b : bodies_to_destroy_) {
        world_->DestroyBody(b);
    }
    bodies_to_destroy_.clear();
}

void DropMergeEnv::applyExplosion(const b2Vec2& center, float force)
{
    // 周囲のBodyにインパルスを与える
    float blast_radius = 2.0f;

    // QueryAABB で効率化できるが、ここでは全Body走査で実装
    for (b2Body* b = world_->GetBodyList(); b; b = b->GetNext()) {
        if (b->GetType() != b2_dynamicBody) continue;

        b2Vec2 body_pos = b->GetWorldCenter();
        b2Vec2 diff = body_pos - center;
        float dist_sq = diff.LengthSquared();

        if (dist_sq < blast_radius * blast_radius && dist_sq > 0.0001f) {
            float dist = std::sqrt(dist_sq);
            b2Vec2 dir = diff;
            dir.Normalize();

            // 距離が近いほど強く
            float strength = force * (1.0f - dist / blast_radius);
            if (strength > 0.0f) {
                b->ApplyLinearImpulseToCenter(strength * dir, true);
            }
        }
    }
}

bool DropMergeEnv::checkGameOver()
{
    // 箱の上端判定
    float dead_line_y = config_.ground_y + config_.box_height;

	// overferlow中判定
    bool is_overflowing = false;

    for (b2Body* b = world_->GetBodyList(); b; b = b->GetNext()) {
        // 果物以外は無視
        if (b->GetType() != b2_dynamicBody) continue;

        // 落下中の果物は判定から除外
        if (b == dropper_.pending_body) continue;

        // 位置取得
        b2Vec2 pos = b->GetWorldCenter();

        // 横にはみ出した（壁抜けバグ）
        if (std::abs(pos.x) > config_.box_width * 0.6f) {
            LOG::warn() << "Fruit out of bounds (x=" << pos.x << ")";   // ログだけ
            ANET_ASSERT_MSG(false, "Fruit out of bounds");
        }

        if (pos.y > dead_line_y) {
            // 少し猶予を持たせるため、速度を見る
            //if (b->GetLinearVelocity().LengthSquared() < 0.1f) {
            //    continue;
            //}

            is_overflowing = true;
            break;
        }
    }

    // オーバーフロー処理
    if (is_overflowing) {
		game_over_timer_++;   // 超えていたらタイマー加算
    } else {
        game_over_timer_ = 0; // 超えてなければリセット（回復）
    }

    // 60step以上オーバーフローが続いたらゲームオーバー
    if (game_over_timer_ > config_.game_over_grace_step) {
        term_reason_ = TerminationReason::Overflow;
        LOG::verbose() << "Game Over: overflow timeout. episode_score=" << episode_score_ << " step_count=" << step_count_ << " x=" << dropper_.x;
        return true;
    }

    return false;
}

void DropMergeEnv::updateDropperStatus()
{
    if (dropper_.is_busy) {
        // タイマー減算
        if (dropper_.wait_timer > 0) dropper_.wait_timer--;
        if (dropper_.min_wait_timer > 0) dropper_.min_wait_timer--;

        // 解除条件A: 物理的に着地した OR タイムアウト
        bool physics_ready = (dropper_.pending_body == nullptr || dropper_.wait_timer <= 0);

        // 解除条件B: アニメーション最小時間を経過した
        // InstantDrop時は、物理判定さえ終わればアニメーション時間は無視する(待たない)
        bool anim_ready = (config_.use_instant_drop) ? true : (dropper_.min_wait_timer <= 0);

        // A&Bならリロード完了
        if (physics_ready && anim_ready) {
            dropper_.current_rank = dropper_.next_rank;
            dropper_.next_rank = determineNextRank();
            dropper_.is_busy = false;
            dropper_.pending_body = nullptr;
        }
    }
}

bool DropMergeEnv::isWorldSettled() const
{
    // 落下中であれば無条件で安定していないとみなす
    if (dropper_.is_busy) return false;
    if (dropper_.pending_body != nullptr) return false;

    // マージ予約や削除予定のBodyが残っていればまだ安定していない
    if (!merge_requests_.empty()) return false;
    if (!bodies_to_destroy_.empty()) return false;

    float v_sq_thresh = config_.settle_velocity_threshold * config_.settle_velocity_threshold;
    float a_thresh = config_.settle_angular_threshold;

    for (b2Body* b = world_->GetBodyList(); b; b = b->GetNext()) {
        if (b->GetType() != b2_dynamicBody) continue;


        // 線速度または角速度が閾値を超えていたら「未安定」
        if (b->GetLinearVelocity().LengthSquared() > v_sq_thresh) return false;
        if (std::abs(b->GetAngularVelocity()) > a_thresh) return false;
    }

    return true; // 全て静止＆マージ完了
}

std::shared_ptr<const anet::rl::SingleStepResult> DropMergeEnv::Step(int64_t action, anet::rl::RunMode mode)
{
    anet::ProfileRange r("DropMergeEnv::Step");

    // エピーソード開始してる
    episode_just_ended_ = false;

    // 新しいエピソードの最初のステップでメトリクスをリセット
    if (step_count_ == 0) {
        ep_max_rank_ = 0;
        ep_end_fruit_count_ = 0;
        term_reason_ = TerminationReason::None;
        ep_settle_steps_sum_ = 0;
        ep_settle_count_ = 0;
        ep_settle_steps_max_ = 0;
    }

    // エピソードstepインクリメント
    step_count_++;

    // アクションからDROP/NOOPを判定
    bool is_drop_action = false;
    bool is_noop_action = false;
    if (action_mode_ == ActionMode::Direct) {
        is_drop_action = true;
        is_noop_action = false;
    } else if (action_mode_ == ActionMode::DirectNoop) {
        is_drop_action = (action > 0);
        is_noop_action = (action == 0);
    } else {
        is_drop_action = (action == kActionDrop);
        is_noop_action = (action == kActionNoop);
    }

    // DROP無しカウント更新
    if (is_drop_action) {
        steps_since_last_drop_ = 0; // DROPしたらリセット
    } else {
        steps_since_last_drop_++;   // それ以外ならカウント
    }

    // アクション処理
    processAction(action);

    // 物理ステップ実行 (通常は1回、InstantDropやSettleモード時は条件を満たすまで回す)
    float accumulated_reward = 0.0f;
    float accumulated_raw_reward = 0.0f;

    if (game_over_) {
        //  スポーン位置ブロックで即死した場合、物理演算は行わず、即座に罰報酬のみを計算する
        auto rewards = calcReward();
        accumulated_reward += rewards.first;
        accumulated_raw_reward += rewards.second;
        last_step_sim_steps_ = 0; // 即死時は0
    } else {
        // 生存している場合、物理演算ループを回す
        int sim_steps = 0;

        // 無限ループ防止用の最大ステップ数
        int max_sim_steps = config_.reload_max_steps + 10;
        if (config_.use_settle_after_drop) {
            max_sim_steps = config_.settle_max_steps;
        }

        // 連続静止フレームのカウンター
        int settled_frames = 0;

        // このステップ中に一度でも不安定になったか
        bool was_unsettled_this_step = is_drop_action;

        // 最低1回は回す
        do {
            // Box2D Step
            float time_step = 1.0f / 60.0f;
            int32 velocity_iterations = 3;  // 6 3
            int32 position_iterations = 1;  // 2 1
            world_->Step(time_step, velocity_iterations, position_iterations);
            sim_steps++;

            // マージ処理 (スコア加算はこの中で current_step_merge_score_ に入る)
            processMerges();

            // ゲームオーバー判定
            if (!game_over_) {
                game_over_ = checkGameOver();
            }

            // 報酬計算 (1ステップ分)
            auto rewards = calcReward();
            accumulated_reward += rewards.first;
            accumulated_raw_reward += rewards.second;

            // リロード判定ロジック
            updateDropperStatus();

            // ゲームオーバーになったら即抜ける
            if (game_over_) break;

            // 静止状態ならカウンターを増やし、動いていたらリセットする
            bool currently_settled = isWorldSettled();
            if (currently_settled) {
                settled_frames++;
            } else {
                settled_frames = 0;
                was_unsettled_this_step = true; // 一度でも不安定になったらフラグを立てる
            }

            bool keep_simulating = false;

            if (config_.use_settle_after_drop) {
                // 安定待ちモード
                if (was_unsettled_this_step || dropper_.is_busy) {
                    // 不安定な状態を経験した、またはDROP中なら厳密に10フレーム待つ
                    keep_simulating = dropper_.is_busy || (settled_frames < 10);
                } else {
                    // ずっと安定したまま（ただの移動やNOOP）なら、無駄に回さず1ステップで抜ける
                    keep_simulating = false;
                }
            } else if (config_.use_instant_drop) {
                // 即時落下モード
                keep_simulating = dropper_.is_busy;
            }

            // 継続条件を満たさなければ抜ける（通常の1ステップ動作含む）
            if (!keep_simulating) break;

            // 無限ループ防止のため強制脱出
            if (sim_steps >= max_sim_steps) {
                if (config_.use_settle_after_drop) {
                    LOG::verbose() << "World did not settle within " << max_sim_steps << " steps. Forcing exit.";
                }
                break;
            }
        } while (true);

        // 今STEPで物理スキップしたステップ数を記録
        last_step_sim_steps_ = sim_steps;

        // エピーソード統計用データを更新
        if (action == kActionDrop) {
            ep_settle_steps_sum_ += sim_steps;
            ep_settle_count_++;
            ep_settle_steps_max_ = std::max(ep_settle_steps_max_, sim_steps);
        }
    }

    // --- ペナルティの計算（RL 1ステップにつき1回だけ） ---

    // DROP以外なら時間罰を与える
    if (!is_drop_action) {
        accumulated_reward += config_.time_penalty;
    }

    //  ゲームオーバー罰
    if (game_over_) {
        accumulated_reward += config_.game_over_penalty;
    }

    // エピソード完了判定
    bool done = game_over_;
    bool truncated = (step_count_ >= config_.max_step);

    // 最大ステップ数到達による打ち切りを終了理由としてセット
    if (!done && truncated) {
        term_reason_ = TerminationReason::MaxStep;
    }

    // ショットクロック判定
    if (config_.no_drop_timeout_steps > 0 && steps_since_last_drop_ >= config_.no_drop_timeout_steps) {
        truncated = true;
        if (!done) {
            term_reason_ = TerminationReason::Timeout;
        }
        LOG::info() << "Episode truncated due to inactivity (No DROP). episode_score=" << episode_score_ << " step_count=" << step_count_ << " x=" << dropper_.x;
    }

    // エピソード終了時のフルーツ数カウント＆フラグ立て
    if (done || truncated) {
        // エピソード終了時情報を返せる状態、だよ
        episode_just_ended_ = true;

        // 果物の数を記録
        ep_end_fruit_count_ = 0;
        for (b2Body* b = world_->GetBodyList(); b; b = b->GetNext()) {
            if (b->GetType() == b2_dynamicBody) {
                auto data = DecodeUserData(b->GetUserData().pointer);
                if (data.first == BodyType::Fruit) {
                    ep_end_fruit_count_++;
                }
            }
        }

        // 前回エピソード情報を記録
        last_episode_score_ = episode_score_;
        last_episode_step_ = step_count_;
        last_episode_reward_ = episode_reward_;
        last_ep_max_settle_steps_ = ep_settle_steps_max_;
        last_ep_mean_settle_steps_ = (ep_settle_count_ > 0) ? (static_cast<float>(ep_settle_steps_sum_) / ep_settle_count_) : 0.0f;
    }

    // State生成
    auto state = makeState();
    state.done = done;
    state.truncated = truncated;

    // NOOPペナルティ
    if (action == kActionNoop) {
        accumulated_reward += config_.noop_penalty;
    }

    // 累積報酬更新
    episode_reward_ += accumulated_reward;
     
    return std::make_shared<StepResult>(
        this->shared_from_this(), accumulated_reward, accumulated_raw_reward, std::move(state));
}

// -------------------------------------------------------------
// Contact Listener
// -------------------------------------------------------------

void DropMergeEnv::ContactListener::BeginContact(b2Contact* contact)
{
    b2Fixture* fa = contact->GetFixtureA();
    b2Fixture* fb = contact->GetFixtureB();

    auto dataA = DecodeUserData(fa->GetBody()->GetUserData().pointer);
    auto dataB = DecodeUserData(fb->GetBody()->GetUserData().pointer);

    // 両方とも果物の場合
    if (dataA.first == BodyType::Fruit && dataB.first == BodyType::Fruit) {
        // 同じランクなら合体
        if (dataA.second == dataB.second) {
            // 重複登録防止は processMerges で行うが、ここでも簡易チェック可
            // 中点を計算
            b2Vec2 posA = fa->GetBody()->GetWorldCenter();
            b2Vec2 posB = fb->GetBody()->GetWorldCenter();
            b2Vec2 center = 0.5f * (posA + posB);

            env_.merge_requests_.push_back({
                fa->GetBody(),
                fb->GetBody(),
                center,
                dataA.second + 1 // Next Rank
                });
        }
    }

    env_.notifyContact(contact->GetFixtureA()->GetBody());
    env_.notifyContact(contact->GetFixtureB()->GetBody());
}

// -------------------------------------------------------------
// Observation & Reward
// -------------------------------------------------------------

anet::rl::SingleState DropMergeEnv::makeState() const
{
    anet::ProfileRange r("DropMergeEnv::makeState");

    // --- 定数・範囲の定義 ---
    const float min_x = -config_.box_width * 0.5f;
    const float max_x = config_.box_width * 0.5f;
    const float min_y = config_.ground_y;
    const float max_y = config_.ground_y + config_.box_height;
    const float cell_w = (max_x - min_x) / config_.grid_cols;
    const float cell_h = (max_y - min_y) / config_.grid_rows;

    // --- スカラー部を充填 ---

    // Dropper X 正規化
    if (action_mode_ == ActionMode::Direct || action_mode_ == ActionMode::DirectNoop) {
        obs_ptr_[0] = 0.0f; // 座標指定モードの時は完全無効化（DropperX座標を使わないので）
    } else {
        obs_ptr_[0] = std::clamp(dropper_.x / (config_.box_width * 0.5f), -1.0f, 1.0f);
    }

    // Rank 正規化
    const float norm_scale = 1.0f / (float)kFruitTypeCount;
    obs_ptr_[1] = dropper_.current_rank * norm_scale;
    obs_ptr_[2] = dropper_.next_rank * norm_scale;

    // Busy フラグ (instant_dropモード時は常に0)
    const bool is_busy = (config_.use_instant_drop) ? false : dropper_.is_busy;
    obs_ptr_[3] = is_busy ? 1.0f : 0.0f;

    // --- グリッド情報のクリア ---
    const int grid_size = config_.grid_rows * config_.grid_cols;
    std::fill(obs_ptr_ + kNumScalarObsDim, obs_ptr_ + kNumScalarObsDim + grid_size, 0.0f);

    // --- グリッド充填 ---
    for (b2Body* b = world_->GetBodyList(); b; b = b->GetNext()) {
        if (b->GetType() != b2_dynamicBody) continue;

        auto data = DecodeUserData(b->GetUserData().pointer);
        if (data.first != BodyType::Fruit) continue;

        const b2Vec2 pos = b->GetPosition();
        const float r_fruit = config_.fruit_radii[data.second - 1];
        const float val = static_cast<float>(data.second);

        // バウンディングボックスからインデックス範囲を算出
        int c_min = static_cast<int>((pos.x - r_fruit - min_x) / cell_w);
        int c_max = static_cast<int>((pos.x + r_fruit - min_x) / cell_w);
        int r_min = static_cast<int>((pos.y - r_fruit - min_y) / cell_h);
        int r_max = static_cast<int>((pos.y + r_fruit - min_y) / cell_h);

        c_min = std::max(0, c_min);
        c_max = std::min(config_.grid_cols - 1, c_max);
        r_min = std::max(0, r_min);
        r_max = std::min(config_.grid_rows - 1, r_max);

        const float r_sq = r_fruit * r_fruit;

        for (int iy = r_min; iy <= r_max; ++iy) {
            const float cell_y1 = min_y + iy * cell_h;
            const float cell_y2 = cell_y1 + cell_h;
            const float closest_y = std::clamp(pos.y, cell_y1, cell_y2);
            const float dy_sq = (pos.y - closest_y) * (pos.y - closest_y);
            const int row_offset = kNumScalarObsDim + (iy * config_.grid_cols);

            for (int ix = c_min; ix <= c_max; ++ix) {
                const float cell_x1 = min_x + ix * cell_w;
                const float cell_x2 = cell_x1 + cell_w;
                const float closest_x = std::clamp(pos.x, cell_x1, cell_x2);
                const float dx = pos.x - closest_x;

                if (dx * dx + dy_sq <= r_sq) {
                    const int idx = row_offset + ix;
                    // 重なっている場合はランクが低い（小さい）方を優先
                    if (obs_ptr_[idx] == 0.0f || val < obs_ptr_[idx]) {
                        obs_ptr_[idx] = val;
                    }
                }
            }
        }
    }

    // --- Dropper X グリッドの描画---
    bool draw_dropper = config_.use_dropper_x_grid && (action_mode_ != ActionMode::Direct) && (action_mode_ != ActionMode::DirectNoop);
    if (draw_dropper) {
        int target_c = static_cast<int>((dropper_.x - min_x) / cell_w);
        target_c = std::clamp(target_c, 0, config_.grid_cols - 1);
        const int target_idx = kNumScalarObsDim + ((config_.grid_rows - 1) * config_.grid_cols) + target_c;
        obs_ptr_[target_idx] = static_cast<float>(kFruitTypeCount + 1);
    }

    // デバッグ表示用キャッシュ更新（vectorが必要な場合のみ）
    grid_cache_.assign(obs_ptr_ + kNumScalarObsDim, obs_ptr_ + kNumScalarObsDim + grid_size);

    // --- 返却（Designated Initializers） ---
    return anet::rl::SingleState {
        .obs = obs_buffer_.clone(),  // バッファのクローンを渡す
        .done = false,            // Step/Reset側で後ほど上書きされる
        .truncated = false,
        .episode_start = false
    };
}

std::pair<float, float> DropMergeEnv::calcReward()
{
    float reward = 0.0f;
    float raw_reward = 0.0f;

    // 合体スコア報酬
    if (current_step_merge_score_ > 0.0f) {
        reward += current_step_merge_score_;
        raw_reward += current_step_merge_score_;
    }

    // 時間経過ペナルティ
    //reward += config_.time_penalty;

    //  ゲームオーバー罰
    //if (game_over_) {
        //reward += config_.game_over_penalty;
    //}

    return { reward, raw_reward };
}

anet::rl::AuxData DropMergeEnv::CreateAuxData(float reward, float raw_reward) const
{
	anet::ProfileRange r("DropMergeEnv::CreateAuxData");

    anet::rl::AuxData aux;

    // --- Dropper情報 ---
    float r_curr = (dropper_.current_rank > 0) ? config_.fruit_radii[dropper_.current_rank - 1] : 0.0f;
    float r_next = (dropper_.next_rank > 0) ? config_.fruit_radii[dropper_.next_rank - 1] : 0.0f;
    aux.emplace("dropper", torch::tensor({
        dropper_.x,
        (float)dropper_.current_rank,
        (float)dropper_.next_rank,
		r_curr, // 手元の果物半径
		r_next, // 次の果物半径
        }, float_opt_));

    // --- 果物リスト (表示用) ---
    std::vector<float> fruits_data;
    for (b2Body* b = world_->GetBodyList(); b; b = b->GetNext()) {
        if (b->GetType() != b2_dynamicBody) continue;
        auto data = DecodeUserData(b->GetUserData().pointer);
        if (data.first == BodyType::Fruit) {
            b2Vec2 p = b->GetPosition();
            float r = config_.fruit_radii[data.second - 1];
            float angle = b->GetAngle();

            fruits_data.push_back(p.x);
            fruits_data.push_back(p.y);
            fruits_data.push_back(r);
            fruits_data.push_back((float)data.second); // Rank
            fruits_data.push_back(angle);
        }
    }
    // N x 5 tensor
    if (!fruits_data.empty()) {
        auto t = torch::from_blob(fruits_data.data(), { (long)fruits_data.size() / 5, 5 }, torch::kFloat).clone();
        aux.emplace("fruits", t);
    }

    // グリッド (デバッグ表示用)
    // makeStateで計算した grid_cache_ を利用
    if (!grid_cache_.empty()) {
        auto t = torch::from_blob((float*)grid_cache_.data(), { config_.grid_rows, config_.grid_cols }, torch::kFloat).clone();
        aux.emplace("grid", t);
    }

    // Config情報 (枠描画用)
    aux.emplace("world_info", torch::tensor({
        config_.box_width, config_.box_height, config_.ground_y
        }, float_opt_));

    // 報酬
    aux.emplace("rewards", torch::tensor({
        reward,
        raw_reward,
        episode_reward_,
        last_episode_reward_
        }, float_opt_));

    // ステップ数情報
    aux.emplace("step", torch::tensor({
        (float)step_count_,
        (float)last_step_sim_steps_
        }, float_opt_));
    aux.emplace("last_step", torch::tensor({ (float)last_episode_step_ }, float_opt_));

    // スコア
    aux.emplace("score", torch::tensor({ episode_score_ }, float_opt_));
    aux.emplace("last_score", torch::tensor({ last_episode_score_ }, float_opt_));

    // AcionMode
    float mode_id = static_cast<float>(action_mode_);
    aux.emplace("action_mode", torch::tensor({ mode_id, (float)num_drop_actions_ }, float_opt_));

    return aux;
}

std::optional<float> DropMergeEnv::GetScalar(const std::string& key, int64_t index) const
{
    const float nan = std::numeric_limits<float>::quiet_NaN();

    // --- 基本情報 ---

    if (key == "ep_step") {
        if (!episode_just_ended_) return nan;
        return static_cast<float>(last_episode_step_);
    }

    // --- 成果・盤面状態 ---
    if (key == "ep_max_rank") {
        if (!episode_just_ended_) return nan;
        return static_cast<float>(ep_max_rank_);
    }
    if (key == "ep_end_fruit_count") {
        if (!episode_just_ended_) return nan;
        return static_cast<float>(ep_end_fruit_count_);
    }

    // --- Settle関連統計 ---
    if (key == "ep_mean_settle_steps") {
        if (!episode_just_ended_) return nan;
        return last_ep_mean_settle_steps_;
    }
    if (key == "ep_max_settle_steps") {
        if (!episode_just_ended_) return nan;
        return static_cast<float>(last_ep_max_settle_steps_);
    }

    // --- 死因（One-hot表現） ---
    if (key == "term_reason_timeout") {
        if (!episode_just_ended_) return nan;
        return (term_reason_ == TerminationReason::Timeout) ? 1.0f : 0.0f;
    }
    if (key == "term_reason_spawn_blocked") {
        if (!episode_just_ended_) return nan;
        return (term_reason_ == TerminationReason::SpawnBlocked) ? 1.0f : 0.0f;
    }
    if (key == "term_reason_overflow") {
        if (!episode_just_ended_) return nan;
        return (term_reason_ == TerminationReason::Overflow) ? 1.0f : 0.0f;
    }
    if (key == "term_reason_maxstep") {
        if (!episode_just_ended_) return nan;
        return (term_reason_ == TerminationReason::MaxStep) ? 1.0f : 0.0f;
    }

    return std::nullopt;
}

std::optional<torch::Tensor> DropMergeEnv::GetTensor(const std::string& key, int64_t index) const
{
    return std::nullopt;
}

std::optional<std::vector<torch::Tensor>>
DropMergeEnv::GetTensorVector(const std::string& key, int64_t index) const
{
    return std::nullopt;
}


// -------------------------------------------------------------
// DropMergeEnvFactory
// -------------------------------------------------------------

std::shared_ptr<anet::rl::SingleDiscreteEnv>
DropMergeEnvFactory::CreateSingleEnv(const anet::ConfigData& config_data, const torch::Device& device,
    std::optional<anet::seed_t> seed, const std::string& config_prefix)
{
    DropMergeEnvConfig config(config_data, config_prefix);
    return std::make_shared<DropMergeEnv>(config, device, seed);
}

ANET_REGISTER_ENV_FACTORY(DropMergeEnvFactory);
