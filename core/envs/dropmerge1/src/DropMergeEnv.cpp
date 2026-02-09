// DropMergeEnv.cpp
#include "DropMergeEnv.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include "anet/profile.hpp"
#include "anet/log.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/env.hpp"

using namespace anet::rl::env;

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
    : anet::RandomHolder(seed)
    , config_(config)
{
    // メトリクスログなど
    anet::MetricsLogger::Instance()->Log("DropMergeEnv", config_.ToJson());

    float_opt_ = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    bool_opt_ = torch::TensorOptions().dtype(torch::kBool).device(device);

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

    int fixed_dim = 4;
    int grid_dim = config_.grid_rows * config_.grid_cols;

    anet::rl::StateSpec state_spec;
    state_spec.shape = { static_cast<int64_t>(fixed_dim + grid_dim) };

    // Dim infoは詳細記述可能だが、ここでは代表的なもののみ
    state_spec.dims = {
        { {0}, -1.0f, 1.0f, "dropper_x", "Dropper X position" },
        { {1}, 0.0f, 1.0f, "current_rank", "Current fruit rank" },
        { {2}, 0.0f, 1.0f, "next_rank", "Next fruit rank" },
        { {3}, 0.0f, 1.0f, "is_busy", "Dropper busy flag" },
        /// @todo Grid部分は省略
    };

    anet::rl::ActionSpec action_spec;
    action_spec.is_discrete = true;
    action_spec.value_labels = { "NOOP", "LEFT", "DROP", "RIGHT" };

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
        fd.friction = config_.friction;
        fd.restitution = 0.0f; // 床は弾まない

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
    buildWorld();

    step_count_ = 0;
    game_over_ = false;
    episode_score_ = 0.0f;

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

// -------------------------------------------------------------
// Action & Step Logic
// -------------------------------------------------------------

void DropMergeEnv::processAction(int64_t action)
{
    // リロード判定ロジック
    if (dropper_.is_busy) {
        // タイマー減算
        if (dropper_.wait_timer > 0) dropper_.wait_timer--;
        if (dropper_.min_wait_timer > 0) dropper_.min_wait_timer--;

        // 解除条件A: 物理的に着地した OR タイムアウト
        bool physics_ready = (dropper_.pending_body == nullptr || dropper_.wait_timer <= 0);

        // 解除条件B: アニメーション最小時間を経過した
        bool anim_ready = (dropper_.min_wait_timer <= 0);

        // A&Bならリロード完了
        if (physics_ready && anim_ready) {
            dropper_.current_rank = dropper_.next_rank;
            dropper_.next_rank = determineNextRank();
            dropper_.is_busy = false;
            dropper_.pending_body = nullptr;
        }
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
    if (action == kActionLeft) {
        dropper_.x -= config_.dropper_speed;
    } else if (action == kActionRight) {
        dropper_.x += config_.dropper_speed;
    }

    // 範囲制限
    if (dropper_.x < -limit_left) {
		dropper_.x = limit_right;   // 左端から右端にワープ
    } else if (dropper_.x > limit_right) {
		dropper_.x = -limit_left;   // 右端から左端にワープ
    }

    // DROP処理
    if (action == kActionDrop && !dropper_.is_busy) {
        // 果物生成
        float spawn_y = config_.ground_y + config_.box_height;// -0.5f;
        dropper_.pending_body = spawnFruit(dropper_.x, spawn_y, dropper_.current_rank);

        // リロード待ちに入る
        dropper_.wait_timer = config_.reload_max_steps;
        dropper_.min_wait_timer = config_.reload_min_steps;
        dropper_.is_busy = true;
        dropper_.current_rank = 0;  // 手持ちを空に（表示用）
    }
}

b2Body* DropMergeEnv::spawnFruit(float x, float y, int rank)
{
    if (rank < 1 || rank > kFruitTypeCount) return nullptr;

    b2BodyDef bd;
    bd.type = b2_dynamicBody;
    bd.position.Set(x, y);
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

	return body;
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

			// スコア加算
            float s = config_.fruit_scores[req.next_rank - 1];
            current_step_merge_score_ += s;
            episode_score_ += s;

            // 小爆発
            applyExplosion(req.center, config_.pop_force);
        } else {
            // スイカ同士が消えた場合はSpawnしない（Rank 11相当）

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

    for (b2Body* b = world_->GetBodyList(); b; b = b->GetNext()) {
        if (b->GetType() != b2_dynamicBody) continue;


        b2Vec2 pos = b->GetWorldCenter();
        if (pos.y > dead_line_y) {
            // 少し猶予を持たせるため、速度を見る
            if (b->GetLinearVelocity().LengthSquared() < 0.1f) {
                return true;
            }

            // 完全に画面外
            if (pos.y > dead_line_y + 1.0f) {
                return true;
            }
        }
        // 横にはみ出した（壁抜けバグ対策）
        if (std::abs(pos.x) > config_.box_width * 0.6f) {
            return true;
        }
    }
    return false;
}

std::shared_ptr<const anet::rl::SingleStepResult> DropMergeEnv::Step(int64_t action, anet::rl::RunMode mode)
{
    step_count_++;

    processAction(action);

    // Box2D Step
    float time_step = 1.0f / 60.0f;
    int32 velocity_iterations = 6;
    int32 position_iterations = 2;
    world_->Step(time_step, velocity_iterations, position_iterations);

	// マージ処理
    processMerges();

	// ゲームオーバー判定
    game_over_ = checkGameOver();

    // エピソード完了判定
    bool done = game_over_;
    bool truncated = (step_count_ >= config_.max_step);
    if (truncated) done = true;

    // State生成
    auto state = makeState();
    state.done = done;
    state.truncated = truncated;

	// エピソード終了時の情報記録
    if (done || truncated) {
        last_episode_score_ = episode_score_;
        last_episode_step_ = step_count_;
    }

    // 報酬計算
    auto rewards = calcReward();

    return std::make_shared<StepResult>(
        this->shared_from_this(), rewards.first, rewards.second, std::move(state));
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
    anet::rl::SingleState s;

    // Dropper Info (4 dims)
    std::vector<float> fixed_obs;
    fixed_obs.reserve(4);

    // -1.0 ～ 1.0 正規化X
    float norm_x = dropper_.x / (config_.box_width * 0.5f);
    fixed_obs.push_back(std::clamp(norm_x, -1.0f, 1.0f));

    // Rank
    float norm_scale = 1.0f / (float)kFruitTypeCount;
    fixed_obs.push_back(dropper_.current_rank * norm_scale);
    fixed_obs.push_back(dropper_.next_rank * norm_scale);

    // Busy
    fixed_obs.push_back(dropper_.is_busy ? 1.0f : 0.0f);

    // ---- Grid Info ----

    // GridのセルごとにQueryPointして一番手前の果物のRankを取得
    // 毎回Queryするのは重い可能性があるため、全Bodyを走査してグリッドを埋める方式

    std::vector<float> grid_obs(config_.grid_rows * config_.grid_cols, 0.0f);

    // グリッド範囲定義
    float min_x = -config_.box_width * 0.5f;
    float max_x = config_.box_width * 0.5f;
    float min_y = config_.ground_y;
    float max_y = config_.ground_y + config_.box_height;

    float cell_w = (max_x - min_x) / config_.grid_cols;
    float cell_h = (max_y - min_y) / config_.grid_rows;

    for (b2Body* b = world_->GetBodyList(); b; b = b->GetNext()) {
        if (b->GetType() != b2_dynamicBody) continue;

        auto data = DecodeUserData(b->GetUserData().pointer);
        if (data.first != BodyType::Fruit) continue;

        b2Vec2 pos = b->GetPosition();
        float r = config_.fruit_radii[data.second - 1];
		//if (data.second == 1) r *= 2.0f; // 1.3さくらんぼハック:小さすぎて見えくなる対策（物理挙動には影響しない）
        float val = data.second;

        // この果物がカバーするグリッド範囲を計算
        // バウンディングボックスからインデックス範囲を割り出す
        int c_min = static_cast<int>((pos.x - r - min_x) / cell_w);
        int c_max = static_cast<int>((pos.x + r - min_x) / cell_w);
        int r_min = static_cast<int>((pos.y - r - min_y) / cell_h);
        int r_max = static_cast<int>((pos.y + r - min_y) / cell_h);

        c_min = std::max(0, c_min);
        c_max = std::min(config_.grid_cols - 1, c_max);
        r_min = std::max(0, r_min);
        r_max = std::min(config_.grid_rows - 1, r_max);

        // 円形判定
        for (int iy = r_min; iy <= r_max; ++iy) {
            // Y方向の範囲
            float cell_y1 = min_y + iy * cell_h;
            float cell_y2 = min_y + (iy + 1) * cell_h;

            for (int ix = c_min; ix <= c_max; ++ix) {
                // Circle-AABB Intersection

                // X方向の範囲
                float cell_x1 = min_x + ix * cell_w;
                float cell_x2 = min_x + (ix + 1) * cell_w;

                // セル矩形上の、円の中心に最も近い点を求める
                float closest_x = std::clamp(pos.x, cell_x1, cell_x2);
                float closest_y = std::clamp(pos.y, cell_y1, cell_y2);

                // その点と円の中心との距離をチェック
                float dx = pos.x - closest_x;
                float dy = pos.y - closest_y;

                if (dx * dx + dy * dy <= r * r) {
                    // グリッド座標変換 (上から順か下から順か...ここでは下から順、row=0が底)
                    int idx = iy * config_.grid_cols + ix;

                    // 重なっている場合はランクが高い方を優先
                    //if (val > grid_obs[idx]) {
                    //    grid_obs[idx] = val;
                    //}

                    // 重なっている場合はランクが低い方を優先
                    if (grid_obs[idx] == 0.0f || val < grid_obs[idx]) {
                        grid_obs[idx] = val;
                    }
                }
            }
        }
    }

    // Tensor結合
    int total_dim = fixed_obs.size() + grid_obs.size();
    auto t = torch::empty({ total_dim }, float_opt_);

    // コピー
    float* ptr = t.data_ptr<float>();
    std::memcpy(ptr, fixed_obs.data(), fixed_obs.size() * sizeof(float));
    std::memcpy(ptr + fixed_obs.size(), grid_obs.data(), grid_obs.size() * sizeof(float));

    s.obs = t;
    s.done = false;
    s.truncated = false;
    s.episode_start = false;

    // デバッグ表示用にキャッシュしておく
    grid_cache_ = grid_obs;

    return s;
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
    reward += config_.time_penalty;

    //  ゲームオーバー罰
    if (game_over_) {
        reward += config_.game_over_penalty;
    }

    return { reward, raw_reward };
}

anet::rl::AuxData DropMergeEnv::CreateAuxData(float reward, float raw_reward) const
{
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

    // 報酬・ステップ
    aux.emplace("rewards", torch::tensor({ reward, raw_reward }, float_opt_));
    aux.emplace("step", torch::tensor({ (float)step_count_ }, float_opt_));
    aux.emplace("last_step", torch::tensor({ (float)last_episode_step_ }, float_opt_));

    // スコア
    aux.emplace("score", torch::tensor({ episode_score_ }, float_opt_));
    aux.emplace("last_score", torch::tensor({ last_episode_score_ }, float_opt_));

    return aux;
}

std::optional<float> DropMergeEnv::GetScalar(const std::string& key, int64_t index) const
{
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
// Factory
// -------------------------------------------------------------

std::shared_ptr<anet::rl::SingleDiscreteEnv>
DropMergeEnvFactory::CreateSingleEnv(const anet::ConfigData& config_data, const torch::Device& device,
    std::optional<anet::seed_t> seed, const std::string& config_prefix)
{
    DropMergeEnvConfig config(config_data, config_prefix);
    return std::make_shared<DropMergeEnv>(config, device, seed);
}

ANET_REGISTER_ENV_FACTORY(DropMergeEnvFactory);
