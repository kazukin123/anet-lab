#pragma once

#include <deque>
#include <memory>
#include <optional>
#include <span>
#include <vector>
#include <c10/util/SmallVector.h>

namespace anet::rl {

    /// ReplayBuffer内部でcompletionまで運ぶ、schema非依存のhint行。
    using ReplayInitialPriorityHintRow = c10::SmallVector<float, 5>;

    /// Storage上の重いデータ（Obs等）とは分離された、N-Step計算用の軽量メタデータ
    struct QueueRecord {
        int64_t time_idx;       ///< Storage上のインデックス (Facadeが管理するためのKey)
        int64_t logical_time_idx = 0;
        float reward;           ///< 即時報酬
        bool done;              ///< エピソード終了フラグ
        bool truncated;         ///< タイムアップ等による打ち切りフラグ
        bool is_dummy = false;  ///< Truncated時の終端計算用ダミーフラグ
        std::optional<ReplayInitialPriorityHintRow> replay_initial_priority_hint;
        std::vector<float> extra_component_rewards; ///< 2成分目以降。1成分なら確保しない。
    };

    /// 各環境(Env)ごとの未処理メタデータを一時保持するキュー
    class ExperienceQueue {
    public:
        void Push(const QueueRecord& record);
        void Pop(size_t k);
        std::vector<QueueRecord> Peek(size_t k) const;
        size_t Size() const;
        void Clear();
    private:
        std::deque<QueueRecord> buffer_;
    };


    // ======================================================
    // Queue Controller & Builder
    // ======================================================

    /// N-Step計算用に切り出された時系列レコード
    using ExperienceSequence = std::vector<QueueRecord>;

    class ExperienceQueueController {
    public:
        /// Queueを操作し、計算準備が整った系列のリストを返す。使用済み要素はPopする。
        virtual std::vector<ExperienceSequence> ExtractSequences(ExperienceQueue& queue) = 0;
        virtual ~ExperienceQueueController() = default;
    };

    /// Storageに後から上書き(Update)するための、構築済み経験データ
    struct ReplayExperience {
        float target_return;    ///< 計算された割引報酬和など (Value)
        bool terminal;          ///< 最終的な終了フラグ
        int actual_n_steps;     ///< 実際に進んだステップ数
        bool truncated = false; ///< true terminalではないtruncation境界
        std::vector<float> extra_component_returns; ///< 2成分目以降。
    };

    class ReplayExperienceBuilder {
    public:
        /// 系列(Sequence)を受け取り、アルゴリズムに応じた ReplayExperience を構築する
        virtual ReplayExperience Build(const ExperienceSequence& sequence, std::span<const float> gammas) const = 0;
        virtual ~ReplayExperienceBuilder() = default;
    };
   

    /// replayとstep診断で同じ系列抽出・収益の演算順序を使う。
    std::unique_ptr<ExperienceQueueController> CreateNStepQueueController(int n_step);
    std::unique_ptr<ReplayExperienceBuilder> CreateDefaultExperienceBuilder();

} // namespace anet::rl
