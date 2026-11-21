#include "anet/algo/imagecls.hpp"
#include "anet/init.hpp"
#include "anet/agent.hpp"
#include "image_cls_agent.hpp"

void anet::rl::algo::InitImageCls()
{
    static const auto failure = anet::rl::detail::CaptureRegistrationFailure([] {
        anet::rl::RegisterAgentFactory<anet::rl::img_cls::ImageClsAgentFactory>();
    });
    if (failure) std::rethrow_exception(failure);
}
