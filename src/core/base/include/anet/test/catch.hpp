#pragma once

#ifndef ANET_TEST_SUPPORT
#error "anet/test/catch.hpp requires ANET_TEST_SUPPORT"
#endif

// libtorch の CHECK を先に定義・解除し、テストコードでは Catch2 の CHECK を使用する。
#include <torch/torch.h>

#ifdef CHECK
#undef CHECK
#endif

#include <catch.hpp>
