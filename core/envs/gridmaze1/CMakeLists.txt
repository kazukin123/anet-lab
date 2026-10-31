# anet-lab/core/envs/gridmaze1/CMakeLists.txt


# ---------------------------------------------------------------------
# プロジェクト定義
# ---------------------------------------------------------------------

cmake_minimum_required(VERSION 3.20)
project(GridMazeEnv)


# ---------------------------------------------------------------------
# ファイル収集
# ---------------------------------------------------------------------

file(GLOB_RECURSE ANET_PRIVATE_SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.hpp")
list(FILTER ANET_PRIVATE_SOURCES EXCLUDE REGEX ".*test\\.cpp$")


# ---------------------------------------------------------------------
# ターゲット定義
# ---------------------------------------------------------------------

add_library(GridMazeEnv STATIC)
add_executable(GridMazeEnv-test src/GridMazeEnv_test.cpp)
target_include_directories(GridMazeEnv-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(GridMazeEnv-test PRIVATE GridMazeEnv anet-core catch2)
add_test(NAME GridMazeEnv-test COMMAND GridMazeEnv-test)


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

set_target_properties(GridMazeEnv GridMazeEnv-test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_CURRENT_SOURCE_DIR}/bin/Debug
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_SOURCE_DIR}/bin/RelWithDebInfo
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/bin/Release
)


# ---------------------------------------------------------------------
#  IDE用設定
# ---------------------------------------------------------------------

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "src" FILES ${ANET_PRIVATE_SOURCES})
source_group("src" REGULAR_EXPRESSION "cmake_pch.*")

# 他のEnvテストと同じDLLを配置し、単独実行とCTestの経路をそろえる。
if(MSVC)
    get_filename_component(TORCH_ROOT "${Torch_DIR}/../../../" ABSOLUTE)
    file(GLOB TORCH_DLLS "${TORCH_ROOT}/lib/*.dll")
    if(TORCH_DLLS)
        add_custom_command(TARGET GridMazeEnv-test POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TORCH_DLLS} $<TARGET_FILE_DIR:GridMazeEnv-test>
            COMMENT "Copying libtorch DLLs")
    endif()
    anet_copy_wx_dlls(GridMazeEnv-test)
endif()
