# ---------------------------------------------------------------------
# プロジェクト定義
# ---------------------------------------------------------------------

cmake_minimum_required(VERSION 3.20)
project(AtariEnv)

# ---------------------------------------------------------------------
# ファイル収集
# ---------------------------------------------------------------------

file(GLOB_RECURSE ANET_PRIVATE_SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.hpp")
file(GLOB_RECURSE ATARI_TEST_SOURCES CONFIGURE_DEPENDS "src/*test.cpp")
list(FILTER ANET_PRIVATE_SOURCES EXCLUDE REGEX ".*test\\.cpp$")

# ---------------------------------------------------------------------
# ターゲット定義
# ---------------------------------------------------------------------

add_library(AtariEnv STATIC)
add_executable(AtariEnv-test)

# ---------------------------------------------------------------------
# ソース設定
# ---------------------------------------------------------------------

target_sources(AtariEnv PRIVATE ${ANET_PRIVATE_SOURCES})
target_sources(AtariEnv-test PRIVATE ${ATARI_TEST_SOURCES})

target_include_directories(AtariEnv
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        "${ANET_ALE_ROOT}/src"
        "${ANET_ALE_ROOT}/src/ale"
        "${ANET_ALE_ROOT}/build/src/ale"
)
target_include_directories(AtariEnv-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_precompile_headers(AtariEnv PRIVATE "src/pch.hpp")

# ---------------------------------------------------------------------
# 依存関係
# ---------------------------------------------------------------------

list(APPEND CMAKE_PREFIX_PATH "${ANET_ALE_ROOT}/build/vcpkg_installed/x64-windows-static-md")
find_package(ZLIB REQUIRED)

target_link_libraries(AtariEnv PRIVATE
    "${ANET_ALE_ROOT}/build/src/ale/$<IF:$<CONFIG:Debug>,Debug,Release>/ale.lib"
    ZLIB::ZLIB
    ${TORCH_LIBRARIES}
    anet-wx
    anet-core
)
target_link_libraries(AtariEnv-test PRIVATE AtariEnv anet-core catch2)
target_compile_definitions(AtariEnv-test PRIVATE ANET_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# ---------------------------------------------------------------------
# テスト登録
# ---------------------------------------------------------------------

add_test(NAME AtariEnv-test COMMAND AtariEnv-test)

# ---------------------------------------------------------------------
# 出力ディレクトリ設定
# ---------------------------------------------------------------------

set_target_properties(AtariEnv AtariEnv-test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/bin/Debug
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_SOURCE_DIR}/bin/RelWithDebInfo
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/bin/Release
)

# ---------------------------------------------------------------------
# MSVC ランタイム設定
# ---------------------------------------------------------------------

if(MSVC)
    get_filename_component(TORCH_ROOT "${Torch_DIR}/../../../" ABSOLUTE)
    file(GLOB TORCH_DLLS "${TORCH_ROOT}/lib/*.dll")
    if(TORCH_DLLS)
        add_custom_command(TARGET AtariEnv-test POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TORCH_DLLS} $<TARGET_FILE_DIR:AtariEnv-test>
            COMMENT "Copying libtorch DLLs")
    endif()
endif()

# ---------------------------------------------------------------------
# IDE 用設定
# ---------------------------------------------------------------------

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "src" FILES ${ANET_PRIVATE_SOURCES})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "test" FILES ${ATARI_TEST_SOURCES})
source_group("src" REGULAR_EXPRESSION "cmake_pch.*")
