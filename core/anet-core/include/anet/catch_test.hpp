// anet/catch_test.hpp

#pragma once

// libtorch の CHECK を先に定義・解除し、テストコードでは Catch2 の CHECK を使用する。
#include <torch/torch.h>

#ifdef CHECK
#undef CHECK
#endif

#include <catch.hpp>
