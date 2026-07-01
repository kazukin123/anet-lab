# anet-lab/core/envs/imagecls/CMakeLists.txt


# ---------------------------------------------------------------------
# プロジェクト定義
# ---------------------------------------------------------------------

cmake_minimum_required(VERSION 3.20)
project(ImageClsEnv)


# ---------------------------------------------------------------------
# ファイル収集
# ---------------------------------------------------------------------

file(GLOB_RECURSE ANET_PRIVATE_SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.hpp")
file(GLOB_RECURSE IMAGECLS_TEST_SOURCES CONFIGURE_DEPENDS "src/*test.cpp")
list(FILTER ANET_PRIVATE_SOURCES EXCLUDE REGEX ".*test\\.cpp$")


# ---------------------------------------------------------------------
# ターゲット定義
# ---------------------------------------------------------------------

add_library(ImageClsEnv STATIC)
add_executable(ImageClsEnv-test)


# ---------------------------------------------------------------------
# ソース設定
# ---------------------------------------------------------------------

target_sources(ImageClsEnv PRIVATE
    ${ANET_PRIVATE_SOURCES})

target_sources(ImageClsEnv-test PRIVATE
    ${IMAGECLS_TEST_SOURCES}
)

target_include_directories(ImageClsEnv
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_include_directories(ImageClsEnv-test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_precompile_headers(ImageClsEnv PRIVATE "src/pch.hpp")

target_link_libraries(ImageClsEnv PRIVATE
    ${TORCH_LIBRARIES}
    anet-wx
    anet-core
)

target_link_libraries(ImageClsEnv-test PRIVATE
    ImageClsEnv
    anet-core
    catch2
)

add_test(NAME ImageClsEnv-test COMMAND ImageClsEnv-test)


# ---------------------------------------------------------------------
# 出力ディレクトリ設定
# ---------------------------------------------------------------------

set_target_properties(ImageClsEnv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_CURRENT_SOURCE_DIR}/bin/Debug
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_SOURCE_DIR}/bin/RelWithDebInfo
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/bin/Release
)

set_target_properties(ImageClsEnv-test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_CURRENT_SOURCE_DIR}/bin/Debug
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_SOURCE_DIR}/bin/RelWithDebInfo
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/bin/Release
)

if (MSVC)
  get_filename_component(TORCH_ROOT "${Torch_DIR}/../../../" ABSOLUTE)
  set(TORCH_DLL_DIR "${TORCH_ROOT}/lib")
  file(GLOB TORCH_DLLS "${TORCH_DLL_DIR}/*.dll")

  if(TORCH_DLLS)
    add_custom_command(TARGET ImageClsEnv-test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Copying libtorch DLLs from ${TORCH_DLL_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TORCH_DLLS}
                $<TARGET_FILE_DIR:ImageClsEnv-test>
        COMMENT "Copying libtorch DLLs"
    )
  endif()
endif()


# ---------------------------------------------------------------------
#  IDE用設定
# ---------------------------------------------------------------------

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "src" FILES ${ANET_PRIVATE_SOURCES})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "test" FILES ${IMAGECLS_TEST_SOURCES})
source_group("src" REGULAR_EXPRESSION "cmake_pch.*")
