# anet-lab/core/envs/dropmerge1/CMakeLists.txt


# ---------------------------------------------------------------------
# プロジェクト定義
# ---------------------------------------------------------------------

cmake_minimum_required(VERSION 3.20)
project(DropMergeEnv)


# ---------------------------------------------------------------------
# ファイル収集
# ---------------------------------------------------------------------

file(GLOB_RECURSE ANET_PRIVATE_SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.hpp")
file(GLOB_RECURSE DROPMERGE_TEST_SOURCES CONFIGURE_DEPENDS "src/*test.cpp")
list(FILTER ANET_PRIVATE_SOURCES EXCLUDE REGEX ".*test\\.cpp$")


# ---------------------------------------------------------------------
# ターゲット定義
# ---------------------------------------------------------------------

add_library(DropMergeEnv STATIC)
add_executable(DropMergeEnv-test)


# ---------------------------------------------------------------------
# ソース設定
# ---------------------------------------------------------------------

target_sources(DropMergeEnv PRIVATE
    ${ANET_PRIVATE_SOURCES})

target_sources(DropMergeEnv-test PRIVATE
    ${DROPMERGE_TEST_SOURCES})

target_include_directories(DropMergeEnv
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_include_directories(DropMergeEnv-test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_precompile_headers(DropMergeEnv PRIVATE "src/pch.hpp")

target_link_libraries(DropMergeEnv PRIVATE
    ${TORCH_LIBRARIES}
    anet-wx
    box2d
    anet-core
)

target_link_libraries(DropMergeEnv-test PRIVATE
    DropMergeEnv
    anet-core
    box2d
    catch2
)

add_test(NAME DropMergeEnv-test COMMAND DropMergeEnv-test)


# ---------------------------------------------------------------------
# 出力ディレクトリ設定
# ---------------------------------------------------------------------

set_target_properties(DropMergeEnv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_CURRENT_SOURCE_DIR}/bin/Debug
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_SOURCE_DIR}/bin/RelWithDebInfo
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/bin/Release
)

set_target_properties(DropMergeEnv-test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_CURRENT_SOURCE_DIR}/bin/Debug
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_SOURCE_DIR}/bin/RelWithDebInfo
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/bin/Release
)

if (MSVC)
  get_filename_component(TORCH_ROOT "${Torch_DIR}/../../../" ABSOLUTE)
  set(TORCH_DLL_DIR "${TORCH_ROOT}/lib")
  file(GLOB TORCH_DLLS "${TORCH_DLL_DIR}/*.dll")

  if(TORCH_DLLS)
    add_custom_command(TARGET DropMergeEnv-test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Copying libtorch DLLs from ${TORCH_DLL_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TORCH_DLLS}
                $<TARGET_FILE_DIR:DropMergeEnv-test>
        COMMENT "Copying libtorch DLLs"
    )
  endif()
endif()


# ---------------------------------------------------------------------
#  IDE用設定
# ---------------------------------------------------------------------

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "src" FILES ${ANET_PRIVATE_SOURCES})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "test" FILES ${DROPMERGE_TEST_SOURCES})
source_group("src" REGULAR_EXPRESSION "cmake_pch.*")
