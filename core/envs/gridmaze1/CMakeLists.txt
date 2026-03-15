# anet-lab/core/envs/gridmaze1/CMakeLists.txt


# ---------------------------------------------------------------------
# プロジェクト定義
# ---------------------------------------------------------------------

cmake_minimum_required(VERSION 3.20)
project(GridMazeEnv)


# ---------------------------------------------------------------------
# ファイル収集
# ---------------------------------------------------------------------

file(GLOB_RECURSE ANET_PRIVATE_SOURCES "src/*.cpp" "src/*.hpp")


# ---------------------------------------------------------------------
# ターゲット定義
# ---------------------------------------------------------------------

add_library(GridMazeEnv STATIC)


# ---------------------------------------------------------------------
# ソース設定
# ---------------------------------------------------------------------

target_sources(GridMazeEnv PRIVATE
    ${ANET_PRIVATE_SOURCES})

target_include_directories(GridMazeEnv
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_precompile_headers(GridMazeEnv PRIVATE "src/pch.hpp")

target_link_libraries(GridMazeEnv PRIVATE
    ${TORCH_LIBRARIES}
    anet-wx
    anet-core
)


# ---------------------------------------------------------------------
# 出力ディレクトリ設定
# ---------------------------------------------------------------------

set_target_properties(GridMazeEnv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_CURRENT_SOURCE_DIR}/bin/Debug
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_SOURCE_DIR}/bin/RelWithDebInfo
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/bin/Release
)


# ---------------------------------------------------------------------
#  IDE用設定
# ---------------------------------------------------------------------

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "src" FILES ${ANET_PRIVATE_SOURCES})
source_group("src" REGULAR_EXPRESSION "cmake_pch.*")
