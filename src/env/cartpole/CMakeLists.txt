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


# ---------------------------------------------------------------------
#  IDE用設定
# ---------------------------------------------------------------------

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "src" FILES ${ANET_PRIVATE_SOURCES})
source_group("src" REGULAR_EXPRESSION "cmake_pch.*")
