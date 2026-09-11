#include "anet/catch_test.hpp"

#include "anet/exception.hpp"

#include <cstddef>
#include <new>

TEST_CASE("InstallAllocationFailureLogger records the requested size of a failed new", "[exception][alloc]")
{
    // handler はプロセス全体で 1 つなので、多重登録が無視されることも同時に確認する。
    anet::InstallAllocationFailureLogger();
    anet::InstallAllocationFailureLogger();

    const auto before = anet::GetAllocationFailureInfo();

    // 実メモリを消費しない巨大要求で operator new を確実に失敗させる。
    // SIZE_MAX を避けるのは bad_array_new_length ではなく bad_alloc を通すため。
    // volatile 経由にして、確保そのものが最適化で消えないようにする。
    constexpr size_t kHugeRequest = static_cast<size_t>(1) << 62;
    volatile size_t huge = kHugeRequest;
    CHECK_THROWS_AS(operator new(static_cast<size_t>(huge)), std::bad_alloc);

    const auto after = anet::GetAllocationFailureInfo();
    CHECK(after.failure_count == before.failure_count + 1);
    CHECK(after.last_request_bytes == kHugeRequest);
}
