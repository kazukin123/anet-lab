#include "anet/algo/imagecls.hpp"
#include "anet/agent.hpp"
#include "image_cls_agent.hpp"

void anet::rl::algo::InitImageCls()
{
    anet::rl::RegisterAgentFactory<anet::rl::img_cls::ImageClsAgentFactory>();
}
