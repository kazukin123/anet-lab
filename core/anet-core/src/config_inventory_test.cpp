#include "anet/catch_test.hpp"
#include <chrono>
#include "anet/agent.hpp"
#include "anet/env.hpp"
#include "anet/test_util.hpp"

namespace {

    class InventoryAgentFactory final : public anet::rl::AgentFactory {
    public:
        InventoryAgentFactory(std::string id, std::vector<std::string> prefixes)
            : id_(std::move(id)), prefixes_(std::move(prefixes)) {}
        std::string GetTargetAgentClassId() const override { return id_; }
        std::vector<std::string> GetConfigPrefixes() const override { return prefixes_; }
        std::shared_ptr<anet::rl::Agent> CreateAgent(const anet::rl::EnvSpec&,
            const anet::rl::BatchEnvSpec&, const torch::Device&, const anet::ConfigData&,
            std::shared_ptr<anet::rl::Notifier>, std::optional<anet::seed_t>) const override
        {
            FAIL("Inventory classification must not construct an Agent");
            return nullptr;
        }
    private:
        const std::string id_;
        const std::vector<std::string> prefixes_;
    };

    class InventorySingleEnvFactory final : public anet::rl::SingleDiscreteEnvFactory {
    public:
        explicit InventorySingleEnvFactory(std::string id) : id_(std::move(id)) {}
        std::string GetTargetEnvClassId() const override { return id_; }
        std::string GetDefaultConfigPrefix() const override { return "inventory.single"; }
        std::shared_ptr<anet::rl::SingleDiscreteEnv> CreateSingleEnv(const anet::ConfigData&,
            const torch::Device&, const std::string&, std::optional<anet::seed_t>,
            anet::rl::RunMode, const std::string&) override
        {
            FAIL("Inventory classification must not construct a Single Env");
            return nullptr;
        }
    private:
        const std::string id_;
    };

    class InventoryBatchEnvFactory final : public anet::rl::BatchEnvFactory {
    public:
        explicit InventoryBatchEnvFactory(std::string id) : id_(std::move(id)) {}
        std::string GetTargetEnvClassId() const override { return id_; }
        std::vector<std::string> GetConfigPrefixes() const override { return {"inventory.batch", "inventory.shared.active"}; }
        std::shared_ptr<anet::rl::BatchEnv> CreateBatchEnv(const anet::ConfigData&,
            const torch::Device&, const std::string&, std::optional<anet::seed_t>, int,
            anet::rl::RunMode, const std::string&) override
        {
            FAIL("Inventory classification must not construct a Batch Env");
            return nullptr;
        }
    private:
        const std::string id_;
    };

} // namespace

TEST_CASE("Inventory prefix exclusions retain active descendants", "[prd117][config_inventory]")
{
    anet::ConfigData data;
    data.Set("inventory.shared.active.typo", "kept");
    data.Set("inventory.shared.inactive.value", "stock");
    data.MarkInventory("inventory.shared", {"inventory.shared.active"});
    const auto unread = data.GetUnreadKeys();
    REQUIRE(unread.size() == 1);
    CHECK(unread.front().first == "inventory.shared.active.typo");
}

TEST_CASE("Agent repositories classify factory-owned prefixes without constructing resources", "[prd117][config_inventory]")
{
    const auto suffix = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto active = "InventoryActiveAgent" + suffix;
    auto& repository = anet::rl::AgentRepository::Instance();
    repository.Register(std::make_shared<InventoryAgentFactory>(active,
        std::vector<std::string>{"inventory.shared.active"}));
    repository.Register(std::make_shared<InventoryAgentFactory>("InventoryInactiveAgent" + suffix,
        std::vector<std::string>{"inventory.shared", "inventory.extra"}));
    anet::ConfigData data;
    data.Set("inventory.shared.active.typo", "kept");
    data.Set("inventory.shared.inactive.value", "stock");
    data.Set("inventory.extra.value", "stock");
    repository.MarkUnusedConfigInventory(data, active);
    const auto unread = data.GetUnreadKeys();
    REQUIRE(unread.size() == 1);
    CHECK(unread.front().first == "inventory.shared.active.typo");
}

TEST_CASE("Env repositories respect Single prefixes and Batch-owned extra prefixes", "[prd117][config_inventory]")
{
    const auto suffix = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto active = "InventoryActiveBatch" + suffix;
    auto& repository = anet::rl::EnvRepository::Instance();
    repository.Regist(std::make_shared<InventoryBatchEnvFactory>(active));
    repository.Regist(std::make_shared<InventorySingleEnvFactory>("InventoryInactiveSingle" + suffix));
    anet::ConfigData data;
    data.Set("inventory.single.value", "stock");
    data.Set("inventory.batch.typo", "kept");
    data.Set("inventory.shared.active.typo", "kept");
    repository.MarkUnusedConfigInventory(data, active);
    const auto unread = data.GetUnreadKeys();
    REQUIRE(unread.size() == 2);
    CHECK(unread.at(0).first == "inventory.batch.typo");
    CHECK(unread.at(1).first == "inventory.shared.active.typo");
}
