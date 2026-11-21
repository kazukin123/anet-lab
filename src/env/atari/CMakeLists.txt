# ---------------------------------------------------------------------
# プロジェクト定義
# ---------------------------------------------------------------------

cmake_minimum_required(VERSION 3.20)
project(anet-env-atari)

# ---------------------------------------------------------------------
# ファイル収集
# ---------------------------------------------------------------------

file(GLOB_RECURSE ANET_PRIVATE_SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.hpp")
file(GLOB_RECURSE ATARI_TEST_SOURCES CONFIGURE_DEPENDS "src/*test.cpp")
list(FILTER ANET_PRIVATE_SOURCES EXCLUDE REGEX ".*test\\.cpp$")

# ---------------------------------------------------------------------
# ターゲット定義
# ---------------------------------------------------------------------

add_library(anet-env-atari STATIC)
add_executable(anet-env-atari-test)

# ---------------------------------------------------------------------
# ソース設定
# ---------------------------------------------------------------------

target_sources(anet-env-atari PRIVATE ${ANET_PRIVATE_SOURCES})
target_sources(anet-env-atari-test PRIVATE ${ATARI_TEST_SOURCES})

target_include_directories(anet-env-atari
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        "${ANET_ALE_ROOT}/src"
        "${ANET_ALE_ROOT}/src/ale"
        "${ANET_ALE_ROOT}/build/src/ale"
)
# RAM fixtureの初期条件設定に必要なALEヘッダをテストtargetだけへ公開する。
target_include_directories(anet-env-atari-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    "${ANET_ALE_ROOT}/src"
    "${ANET_ALE_ROOT}/src/ale"
    "${ANET_ALE_ROOT}/build/src/ale"
)

target_precompile_headers(anet-env-atari PRIVATE "src/pch.hpp")

# ---------------------------------------------------------------------
# 依存関係
# ---------------------------------------------------------------------

list(APPEND CMAKE_PREFIX_PATH "${ANET_ALE_ROOT}/build/vcpkg_installed/x64-windows-static-md")
find_package(ZLIB REQUIRED)

target_link_libraries(anet-env-atari PRIVATE
    "${ANET_ALE_ROOT}/build/src/ale/$<IF:$<CONFIG:Debug>,Debug,Release>/ale.lib"
    ZLIB::ZLIB
    ${TORCH_LIBRARIES}
    anet-wx
    anet-core-base
)
target_link_libraries(anet-env-atari-test PRIVATE anet-env-atari anet-core-base anet-core-testing)
target_compile_definitions(anet-env-atari-test PRIVATE ANET_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# ---------------------------------------------------------------------
# テスト登録
# ---------------------------------------------------------------------

add_test(NAME anet-env-atari-test COMMAND anet-env-atari-test)

anet_set_test_runtime(anet-env-atari-test)


# ---------------------------------------------------------------------
# IDE 用設定
# ---------------------------------------------------------------------

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "src" FILES ${ANET_PRIVATE_SOURCES})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "test" FILES ${ATARI_TEST_SOURCES})
source_group("src" REGULAR_EXPRESSION "cmake_pch.*")
