# anet-lab/src/env/cartpole/CMakeLists.txt


# ---------------------------------------------------------------------
# プロジェクト定義
# ---------------------------------------------------------------------

cmake_minimum_required(VERSION 3.20)
project(anet-env-cartpole)


# ---------------------------------------------------------------------
# ファイル収集
# ---------------------------------------------------------------------

file(GLOB_RECURSE ANET_PRIVATE_SOURCES "src/*.cpp" "src/*.hpp")
list(FILTER ANET_PRIVATE_SOURCES EXCLUDE REGEX "_test\\.cpp$")


# ---------------------------------------------------------------------
# ターゲット定義
# ---------------------------------------------------------------------

add_library(anet-env-cartpole STATIC)


# ---------------------------------------------------------------------
# ソース設定
# ---------------------------------------------------------------------

target_sources(anet-env-cartpole PRIVATE
    ${ANET_PRIVATE_SOURCES}
)

target_include_directories(anet-env-cartpole
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_precompile_headers(anet-env-cartpole PRIVATE "src/pch.hpp")

target_link_libraries(anet-env-cartpole PRIVATE
    ${TORCH_LIBRARIES}
    anet-wx
    anet-core-base
)

add_executable(anet-env-cartpole-test src/CartPoleEnv_test.cpp)
target_link_libraries(anet-env-cartpole-test PRIVATE anet-env-cartpole anet-core-base anet-core-testing)
add_test(NAME anet-env-cartpole-test COMMAND anet-env-cartpole-test)
anet_set_test_runtime(anet-env-cartpole-test)


# ---------------------------------------------------------------------
#  IDE用設定
# ---------------------------------------------------------------------

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "src" FILES ${ANET_PRIVATE_SOURCES})
source_group("src" REGULAR_EXPRESSION "cmake_pch.*")
