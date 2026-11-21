# anet-lab/src/env/gridmaze/CMakeLists.txt


# ---------------------------------------------------------------------
# プロジェクト定義
# ---------------------------------------------------------------------

cmake_minimum_required(VERSION 3.20)
project(anet-env-gridmaze)


# ---------------------------------------------------------------------
# ファイル収集
# ---------------------------------------------------------------------

file(GLOB_RECURSE ANET_PRIVATE_SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.hpp")
list(FILTER ANET_PRIVATE_SOURCES EXCLUDE REGEX ".*test\\.cpp$")


# ---------------------------------------------------------------------
# ターゲット定義
# ---------------------------------------------------------------------

add_library(anet-env-gridmaze STATIC)
add_executable(anet-env-gridmaze-test src/GridMazeEnv_test.cpp)
target_include_directories(anet-env-gridmaze-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(anet-env-gridmaze-test PRIVATE anet-env-gridmaze anet-core-base anet-core-testing)
add_test(NAME anet-env-gridmaze-test COMMAND anet-env-gridmaze-test)


# ---------------------------------------------------------------------
# ソース設定
# ---------------------------------------------------------------------

target_sources(anet-env-gridmaze PRIVATE
    ${ANET_PRIVATE_SOURCES})

target_include_directories(anet-env-gridmaze
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_precompile_headers(anet-env-gridmaze PRIVATE "src/pch.hpp")

target_link_libraries(anet-env-gridmaze PRIVATE
    ${TORCH_LIBRARIES}
    anet-wx
    anet-core-base
)


anet_set_test_runtime(anet-env-gridmaze-test)


# ---------------------------------------------------------------------
#  IDE用設定
# ---------------------------------------------------------------------

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "src" FILES ${ANET_PRIVATE_SOURCES})
source_group("src" REGULAR_EXPRESSION "cmake_pch.*")
