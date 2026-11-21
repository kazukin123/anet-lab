# anet-lab/src/core/base/CMakeLists.txt


# ---------------------------------------------------------------------
# プロジェクト定義
# ---------------------------------------------------------------------

cmake_minimum_required(VERSION 3.20)
project(anet-core-base)


# ---------------------------------------------------------------------
# ファイル収集
# ---------------------------------------------------------------------

file(GLOB_RECURSE ANET_PUBLIC_HEADERS CONFIGURE_DEPENDS "include/*.hpp")
file(GLOB_RECURSE ANET_PRIVATE_SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.hpp")
file(GLOB_RECURSE ANET_TEST_SOURCES CONFIGURE_DEPENDS "src/*test.cpp")

list(FILTER ANET_PRIVATE_SOURCES EXCLUDE REGEX ".*test\\.cpp$")


# ---------------------------------------------------------------------
# ターゲット定義
# ---------------------------------------------------------------------
add_library(anet-core-base STATIC)
add_executable(anet-core-base-test)

file(GLOB ANET_TEST_SUPPORT_SOURCES CONFIGURE_DEPENDS "src_testing/*.cpp")
add_library(anet-core-testing STATIC ${ANET_TEST_SUPPORT_SOURCES})
target_include_directories(anet-core-testing PRIVATE src)
target_compile_definitions(anet-core-testing PUBLIC ANET_TEST_SUPPORT)
target_link_libraries(anet-core-testing PUBLIC anet-core-base catch2)


# ---------------------------------------------------------------------
# ソース設定
# ---------------------------------------------------------------------

target_sources(anet-core-base PRIVATE
    ${ANET_PUBLIC_HEADERS}
    ${ANET_PRIVATE_SOURCES}
)

target_sources(anet-core-base-test PRIVATE
    ${ANET_TEST_SOURCES}
)

target_include_directories(anet-core-base
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if (MSVC)
    target_compile_definitions(anet-core-base PUBLIC NOMINMAX)
endif()

target_precompile_headers(anet-core-base PRIVATE "src/pch.hpp")

target_link_libraries(anet-core-base PUBLIC
    ${TORCH_LIBRARIES}
    tracy
    anet-wx
)

target_include_directories(anet-core-base-test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(anet-core-base-test PRIVATE
    anet-core-base
    anet-core-testing
)

add_test(NAME anet-core-base-test COMMAND anet-core-base-test)
# 現用設定を相対パスで読むテストは、直接実行と同じrepoルートで動かす。
set_tests_properties(anet-core-base-test PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")


anet_set_test_runtime(anet-core-base-test)


# ---------------------------------------------------------------------
#  IDE用設定
# ---------------------------------------------------------------------

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/include" PREFIX "include" FILES ${ANET_PUBLIC_HEADERS})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "src" FILES ${ANET_PRIVATE_SOURCES})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "test" FILES ${ANET_TEST_SOURCES})
source_group("src" REGULAR_EXPRESSION "cmake_pch.*")
