#include "catch.hpp"

#include "anet/profile.hpp"

#include <string_view>

namespace {

constexpr auto kFreeFunctionName =
    anet::profile_detail::MakeProfileFunctionName<"void __cdecl FreeFunction(void)">();
static_assert(std::string_view(kFreeFunctionName.data()) == "FreeFunction");

constexpr auto kClassMethodName =
    anet::profile_detail::MakeProfileFunctionName<"void __cdecl anet::RunnerBase::DoUpdateFrame(void)">();
static_assert(std::string_view(kClassMethodName.data()) == "RunnerBase::DoUpdateFrame");

constexpr auto kConstMethodName =
    anet::profile_detail::MakeProfileFunctionName<"int __cdecl anet::Network::Forward(int) const">();
static_assert(std::string_view(kConstMethodName.data()) == "Network::Forward");

constexpr auto kNestedNamespaceName =
    anet::profile_detail::MakeProfileFunctionName<"void __cdecl anet::detail::Learner::Optimize(void)">();
static_assert(std::string_view(kNestedNamespaceName.data()) == "Learner::Optimize");

constexpr auto kPhaseName = anet::profile_detail::MakeProfilePhaseName(kClassMethodName, "step");
static_assert(std::string_view(kPhaseName.data()) == "RunnerBase::DoUpdateFrame.step");

void RunProfileMacroSmoke()
{
    ANET_PROFILE_FUNC();
    ANET_PROFILE_SCOPE(load);
    ANET_PROFILE_SCOPE_NEXT(update, load);
    ANET_PROFILE_SCOPE_END(update);
    ANET_PROFILE_SCOPE_NAMED(named_load, "namedLoad");
    ANET_PROFILE_SCOPE_NEXT_NAMED(named_update, named_load, "namedUpdate");
    ANET_PROFILE_SCOPE_END_NAMED(named_update);
    ANET_PROFILE_SCOPE_FULL(save, "ProfileTest::RunProfileMacroSmoke.save");
}

} // namespace

TEST_CASE("Profile macros compile and support explicit end", "[profile]")
{
    RunProfileMacroSmoke();
}
