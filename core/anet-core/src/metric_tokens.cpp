#include "metric_tokens.hpp"
#include "anet/str_util.hpp"

using namespace anet::rl;
using namespace anet::rl::metric_tokens;

// 宣言順と各指定の出現を残し、既定値・後勝ち・診断はチャネル側へ委ねる。
std::vector<Token> anet::rl::metric_tokens::ParseMetricTokens(const std::string& definition)
{
    std::vector<Token> tokens;
    for (const auto& raw : anet::Split(definition, { " " }, true)) {
        Token token{ .raw = raw, .value = raw };
        if (raw == "@train" || raw == "@learn" || raw == "@episode_end" || raw == "@session_end") {
            token.kind = Kind::EVENT;
            token.value = raw.substr(1);
        } else if (raw == "$train_step" || raw == "$learn_step" || raw == "$episode_step"
            || raw == "$exp_step" || raw == "$update_step" || raw == "$sim_step") {
            token.kind = Kind::STEP;
            token.value = raw.substr(1);
        } else if (raw == "$actor" || raw == "$agent" || raw == "$env" || raw == "$env_step" || raw == "$runner"
            || raw == "$batch_experience" || raw == "$exp" || raw == "$batch_update_result"
            || raw == "$update_result" || raw == "$result" || raw == "$action" || raw == "$action_info") {
            token.kind = Kind::TARGET;
            token.value = raw.substr(1);
        } else if (raw == "$train" || anet::StartsWith(raw, "$eval.[")) {
            token.kind = Kind::SCOPE;
            token.value = raw == "$train" ? "train" : anet::ExtractBetween(raw, "$eval.[", "]");
        } else if (raw == "$ema") {
            token.kind = Kind::EMA;
        } else {
            const auto attr = anet::Split(raw, { ":" }, true);
            if (attr.size() == 2) {
                token.kind = Kind::ATTRIBUTE;
                token.attribute = attr[0];
                token.value = attr[1];
                if (attr[0] == "event") token.kind = Kind::EVENT;
                else if (attr[0] == "step" || attr[0] == "step_axis") token.kind = Kind::STEP;
                else if (attr[0] == "target") token.kind = Kind::TARGET;
            }
        }
        if (token.kind == Kind::EVENT) {
            if (token.value == "train") token.event = EventType::TRAIN;
            else if (token.value == "learn") token.event = EventType::LEARN;
            else if (token.value == "episode_end") token.event = EventType::EPISODE_END;
            else if (token.value == "session_end") token.event = EventType::SESSION_END;
        } else if (token.kind == Kind::STEP) {
            // 属性形は従来の短縮名を受理し、$ 形だけ train_step / learn_step を受理する。
            if (token.value == "train" || (token.attribute.empty() && token.value == "train_step")) token.step = StepAxis::TRAIN;
            else if (token.value == "learn" || (token.attribute.empty() && token.value == "learn_step")) token.step = StepAxis::LEARN;
            else if (token.value == "episode" || token.value == "episode_step") token.step = StepAxis::EPISODE;
            else if (token.value == "exp" || token.value == "exp_step") token.step = StepAxis::EXP;
            else if (token.value == "update" || token.value == "update_step") token.step = StepAxis::UPDATE;
            else if (token.value == "sim" || token.value == "sim_step") token.step = StepAxis::SIM;
        } else if (token.kind == Kind::TARGET) {
            if (token.value == "actor") token.field = EventField::ACTOR;
            else if (token.value == "agent") token.field = EventField::AGENT;
            else if (token.value == "env") token.field = EventField::ENV;
            else if (token.value == "env_step") token.field = EventField::ENV_STEP;
            else if (token.value == "runner") token.field = EventField::RUNNER;
            else if (token.value == "exp" || token.value == "batch_experience") token.field = EventField::EXPERIENCE;
            else if (token.value == "update_result" || token.value == "batch_update_result" || token.value == "result") token.field = EventField::UPDATE_RESULT;
            else if (token.value == "action_info" || token.value == "action") token.field = EventField::ACTION_INFO;
        }
        tokens.push_back(std::move(token));
    }
    return tokens;
}
