# anet-lab/src/env/imagecls/CMakeLists.txt


# ---------------------------------------------------------------------
# プロジェクト定義
# ---------------------------------------------------------------------

cmake_minimum_required(VERSION 3.20)
project(anet-env-imagecls)


# ---------------------------------------------------------------------
# ファイル収集
# ---------------------------------------------------------------------

file(GLOB_RECURSE ANET_PRIVATE_SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.hpp")
file(GLOB_RECURSE IMAGECLS_TEST_SOURCES CONFIGURE_DEPENDS "src/*test.cpp")
list(FILTER ANET_PRIVATE_SOURCES EXCLUDE REGEX ".*test\\.cpp$")


# ---------------------------------------------------------------------
# ターゲット定義
# ---------------------------------------------------------------------

add_library(anet-env-imagecls STATIC)
add_executable(anet-env-imagecls-test)


# ---------------------------------------------------------------------
# ソース設定
# ---------------------------------------------------------------------

target_sources(anet-env-imagecls PRIVATE
    ${ANET_PRIVATE_SOURCES})

target_sources(anet-env-imagecls-test PRIVATE
    ${IMAGECLS_TEST_SOURCES}
)

target_include_directories(anet-env-imagecls
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_include_directories(anet-env-imagecls-test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_precompile_headers(anet-env-imagecls PRIVATE "src/pch.hpp")

target_link_libraries(anet-env-imagecls PRIVATE
    ${TORCH_LIBRARIES}
    anet-wx
    anet-core-base
)

target_link_libraries(anet-env-imagecls-test PRIVATE
    anet-env-imagecls
    anet-core-base
    anet-core-testing
)

add_test(NAME anet-env-imagecls-test COMMAND anet-env-imagecls-test)


anet_set_test_runtime(anet-env-imagecls-test)


# ---------------------------------------------------------------------
#  IDE用設定
# ---------------------------------------------------------------------

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "src" FILES ${ANET_PRIVATE_SOURCES})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "test" FILES ${IMAGECLS_TEST_SOURCES})
source_group("src" REGULAR_EXPRESSION "cmake_pch.*")
