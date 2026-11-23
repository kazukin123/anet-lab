# anet-lab/src/env/lunarlander/CMakeLists.txt


# ---------------------------------------------------------------------
# プロジェクト定義
# ---------------------------------------------------------------------

cmake_minimum_required(VERSION 3.20)
project(anet-env-lunarlander)


# ---------------------------------------------------------------------
# ファイル収集
# ---------------------------------------------------------------------

file(GLOB_RECURSE ANET_PRIVATE_SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.hpp")
file(GLOB_RECURSE LUNARLANDER_TEST_SOURCES CONFIGURE_DEPENDS "src/*test.cpp")
list(FILTER ANET_PRIVATE_SOURCES EXCLUDE REGEX ".*test\\.cpp$")


# ---------------------------------------------------------------------
# ターゲット定義
# ---------------------------------------------------------------------

add_library(anet-env-lunarlander STATIC)
add_executable(anet-env-lunarlander-test)


# ---------------------------------------------------------------------
# ソース設定
# ---------------------------------------------------------------------

target_sources(anet-env-lunarlander PRIVATE
    ${ANET_PRIVATE_SOURCES}
)

target_sources(anet-env-lunarlander-test PRIVATE
    ${LUNARLANDER_TEST_SOURCES}
)

target_include_directories(anet-env-lunarlander
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_include_directories(anet-env-lunarlander-test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_precompile_headers(anet-env-lunarlander PRIVATE "src/pch.hpp")

target_link_libraries(anet-env-lunarlander PRIVATE
    ${TORCH_LIBRARIES}
    anet-wx
    box2d
    anet-core-base
)

target_link_libraries(anet-env-lunarlander-test PRIVATE
    anet-env-lunarlander
    anet-core-base
    box2d
    anet-core-testing
)

add_test(NAME anet-env-lunarlander-test COMMAND anet-env-lunarlander-test)


anet_set_test_runtime(anet-env-lunarlander-test)


# ---------------------------------------------------------------------
#  IDE用設定
# ---------------------------------------------------------------------

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "src" FILES ${ANET_PRIVATE_SOURCES})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "test" FILES ${LUNARLANDER_TEST_SOURCES})
source_group("src" REGULAR_EXPRESSION "cmake_pch.*")
