#pragma once
#include <torch/torch.h>
#include <vector>
#include "anet/common.hpp"

//------------------------------------------------------
// Enable switches（デフォルトON）
//------------------------------------------------------

#ifndef ANET_ENABLE_TENSOR_ASSERT
#define ANET_ENABLE_TENSOR_ASSERT 1
#endif

//------------------------------------------------------
// 内部実装関数（.cpp に実体あり）
//------------------------------------------------------
#if ANET_ENABLE_TENSOR_ASSERT

void _anet_check_device_impl(const torch::Tensor& t,
    const torch::Device& expect,
    const char* msg,
    const char* file,
    int line);

void _anet_check_shape_impl(const torch::Tensor& t,
    const std::vector<int64_t>& expect,
    const char* msg,
    const char* file,
    int line);

void _anet_check_shape_or_impl(const torch::Tensor& t,
    const std::vector<std::vector<int64_t>>& expects,
    const char* msg,
    const char* file,
    int line);
#endif

//------------------------------------------------------
// Device チェック系マクロ
//------------------------------------------------------

#if ANET_ENABLE_TENSOR_ASSERT

// msg を自動生成（#tensor）
#define ANET_CHECK_DEVICE(tensor, expected_dev)                                \
    do {                                                                       \
        _anet_check_device_impl((tensor), (expected_dev), #tensor,             \
                                __FILE__, __LINE__);                           \
    } while (0)

// msg 明示版（従来動作）
#define ANET_CHECK_DEVICE_MSG(tensor, expected_dev, msg)                       \
    do {                                                                       \
        _anet_check_device_impl((tensor), (expected_dev), (msg),               \
                                __FILE__, __LINE__);                           \
    } while (0)

#define ANET_CHECK_DEVICE_CPU(tensor)                                                 \
    do {                                                                       \
        _anet_check_device_impl((tensor), torch::Device(torch::kCPU), #tensor, \
                                __FILE__, __LINE__);                           \
    } while (0)

#define ANET_CHECK_DEVICE_CPU_MSG(tensor, msg)                                        \
    do {                                                                       \
        _anet_check_device_impl((tensor), torch::Device(torch::kCPU), (msg),   \
                                __FILE__, __LINE__);                           \
    } while (0)

#define ANET_CHECK_DEVICE_CUDA(tensor)                                                \
    do {                                                                       \
        _anet_check_device_impl((tensor), torch::Device(torch::kCUDA), #tensor,\
                                __FILE__, __LINE__);                           \
    } while (0)

#define ANET_CHECK_DEVICE_CUDA_MSG(tensor, msg)                                       \
    do {                                                                       \
        _anet_check_device_impl((tensor), torch::Device(torch::kCUDA), (msg),  \
                                __FILE__, __LINE__);                           \
    } while (0)

#else

#define ANET_CHECK_DEVICE(t, d)                     do {} while(0)
#define ANET_CHECK_DEVICE_MSG(t, d, msg)            do {} while(0)
#define ANET_CHECK_DEVICE_CPU(t)                           do {} while(0)
#define ANET_CHECK_DEVICE_CPU_MSG(t, msg)                  do {} while(0)
#define ANET_CHECK_DEVICE_CUDA(t)                          do {} while(0)
#define ANET_CHECK_DEVICE_CUDA_MSG(t, msg)                 do {} while(0)

#endif


//------------------------------------------------------
// Shape チェック（複数 shape 許可）
//------------------------------------------------------

// ANY: 任意次元（shape のワイルドカード）
static constexpr int64_t ANET_SHAPE_ANY = -1;

#if ANET_ENABLE_TENSOR_ASSERT

// msg = #tensor
#define ANET_CHECK_SHAPE(tensor, ...)                                           \
    do {                                                                        \
        std::vector<std::vector<int64_t>> _anet_shapes = { __VA_ARGS__ };       \
        _anet_check_shape_or_impl((tensor), _anet_shapes, #tensor,              \
                                  __FILE__, __LINE__);                          \
    } while (0)

// 明示 msg 版
#define ANET_CHECK_SHAPE_MSG(tensor, msg, ...)                                  \
    do {                                                                        \
        std::vector<std::vector<int64_t>> _anet_shapes = { __VA_ARGS__ };       \
        _anet_check_shape_or_impl((tensor), _anet_shapes, (msg),                \
                                  __FILE__, __LINE__);                          \
    } while (0)

#else

#define ANET_CHECK_SHAPE(tensor, ...)             do {} while(0)
#define ANET_CHECK_SHAPE_MSG(tensor, msg, ...)    do {} while(0)

#endif
