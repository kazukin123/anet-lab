# anet-lab/src/env/dropmerge/CMakeLists.txt


# ---------------------------------------------------------------------
# プロジェクト定義
# ---------------------------------------------------------------------

cmake_minimum_required(VERSION 3.20)
project(anet-env-dropmerge)


# ---------------------------------------------------------------------
# ファイル収集
# ---------------------------------------------------------------------

file(GLOB_RECURSE ANET_PRIVATE_SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.hpp")
file(GLOB_RECURSE DROPMERGE_TEST_SOURCES CONFIGURE_DEPENDS "src/*test.cpp")
list(FILTER ANET_PRIVATE_SOURCES EXCLUDE REGEX ".*test\\.cpp$")


# ---------------------------------------------------------------------
# ターゲット定義
# ---------------------------------------------------------------------

add_library(anet-env-dropmerge STATIC)
add_executable(anet-env-dropmerge-test)


# ---------------------------------------------------------------------
# ソース設定
# ---------------------------------------------------------------------

target_sources(anet-env-dropmerge PRIVATE
    ${ANET_PRIVATE_SOURCES})

target_sources(anet-env-dropmerge-test PRIVATE
    ${DROPMERGE_TEST_SOURCES})

target_include_directories(anet-env-dropmerge
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_include_directories(anet-env-dropmerge-test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_precompile_headers(anet-env-dropmerge PRIVATE "src/pch.hpp")

target_link_libraries(anet-env-dropmerge PRIVATE
    ${TORCH_LIBRARIES}
    anet-wx
    box2d
    anet-core-base
)

target_link_libraries(anet-env-dropmerge-test PRIVATE
    anet-env-dropmerge
    anet-core-base
    box2d
    anet-core-testing
)

add_test(NAME anet-env-dropmerge-test COMMAND anet-env-dropmerge-test)


anet_set_test_runtime(anet-env-dropmerge-test)


# ---------------------------------------------------------------------
#  IDE用設定
# ---------------------------------------------------------------------

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "src" FILES ${ANET_PRIVATE_SOURCES})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "test" FILES ${DROPMERGE_TEST_SOURCES})
source_group("src" REGULAR_EXPRESSION "cmake_pch.*")
