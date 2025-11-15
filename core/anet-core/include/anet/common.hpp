
#include <torch/torch.h>

//#ifndef ANET_ASSERT
//#define ANET_ASSERT(cond, ...)                \
//  if (!(cond)) {       \
//    throw std::runtime_error(( \
//        cond,                                 \
//        "",                                   \
//        __func__,                             \
//        ", ",                                 \
//        __FILE__,                             \
//        ":",                                  \
//        __LINE__,                             \
//        ", ",                                 \
//        ##__VA_ARGS__));                     \
//  }
//#endif

#ifndef ANET_ASSERT
#define ANET_ASSERT(cond, ...)  assert(##__VA_ARGS__)
#endif

namespace anet {

    inline void CheckDevice(const torch::Tensor& t, const torch::Device& expected, const char* name = "tensor")
    {
        if (t.device() != expected) {
            std::stringstream ss;
            ss << "Device mismatch in " << name << ": tensor=" << t.device()
                << " expected=" << expected;

            wxASSERT_MSG(false, ss.str());
        }
    }
    inline void CheckDeviceCPU(const torch::Tensor& t, const char* name = "tensor")
    {
        CheckDevice(t, torch::kCPU, name);
    }
    inline void CheckDeviceCUDA(const torch::Tensor& t, const char* name = "tensor")
    {
        CheckDevice(t, torch::kCUDA, name);
    }

}
