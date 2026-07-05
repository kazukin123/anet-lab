# anet-lab/core/envs/lunarlander1/CMakeLists.txt


# ---------------------------------------------------------------------
# プロジェクト定義
# ---------------------------------------------------------------------

cmake_minimum_required(VERSION 3.20)
project(LunarLanderEnv)


# ---------------------------------------------------------------------
# ファイル収集
# ---------------------------------------------------------------------

file(GLOB_RECURSE ANET_PRIVATE_SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.hpp")
file(GLOB_RECURSE LUNARLANDER_TEST_SOURCES CONFIGURE_DEPENDS "src/*test.cpp")
list(FILTER ANET_PRIVATE_SOURCES EXCLUDE REGEX ".*test\\.cpp$")


# ---------------------------------------------------------------------
# ターゲット定義
# ---------------------------------------------------------------------

add_library(LunarLanderEnv STATIC)
add_executable(LunarLanderEnv-test)


# ---------------------------------------------------------------------
# ソース設定
# ---------------------------------------------------------------------

target_sources(LunarLanderEnv PRIVATE
    ${ANET_PRIVATE_SOURCES}
)

target_sources(LunarLanderEnv-test PRIVATE
    ${LUNARLANDER_TEST_SOURCES}
)

target_include_directories(LunarLanderEnv
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_include_directories(LunarLanderEnv-test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_precompile_headers(LunarLanderEnv PRIVATE "src/pch.hpp")

target_link_libraries(LunarLanderEnv PRIVATE
    ${TORCH_LIBRARIES}
    anet-wx
    box2d
    anet-core
)

target_link_libraries(LunarLanderEnv-test PRIVATE
    LunarLanderEnv
    anet-core
    box2d
    catch2
)

add_test(NAME LunarLanderEnv-test COMMAND LunarLanderEnv-test)


# ---------------------------------------------------------------------
# 出力ディレクトリ設定
# ---------------------------------------------------------------------

set_target_properties(LunarLanderEnv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_CURRENT_SOURCE_DIR}/bin/Debug
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_SOURCE_DIR}/bin/RelWithDebInfo
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/bin/Release
)

set_target_properties(LunarLanderEnv-test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_CURRENT_SOURCE_DIR}/bin/Debug
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_SOURCE_DIR}/bin/RelWithDebInfo
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/bin/Release
)

if (MSVC)
  get_filename_component(TORCH_ROOT "${Torch_DIR}/../../../" ABSOLUTE)
  set(TORCH_DLL_DIR "${TORCH_ROOT}/lib")
  file(GLOB TORCH_DLLS "${TORCH_DLL_DIR}/*.dll")

  if(TORCH_DLLS)
    add_custom_command(TARGET LunarLanderEnv-test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Copying libtorch DLLs from ${TORCH_DLL_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TORCH_DLLS}
                $<TARGET_FILE_DIR:LunarLanderEnv-test>
        COMMENT "Copying libtorch DLLs"
    )
  endif()
endif()


# ---------------------------------------------------------------------
#  IDE用設定
# ---------------------------------------------------------------------

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "src" FILES ${ANET_PRIVATE_SOURCES})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "test" FILES ${LUNARLANDER_TEST_SOURCES})
source_group("src" REGULAR_EXPRESSION "cmake_pch.*")
