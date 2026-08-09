# anet-lab/core/anet-core/CMakeLists.txt


# ---------------------------------------------------------------------
# プロジェクト定義
# ---------------------------------------------------------------------

cmake_minimum_required(VERSION 3.20)
project(anet-core)


# ---------------------------------------------------------------------
# ファイル収集
# ---------------------------------------------------------------------

file(GLOB_RECURSE ANET_PUBLIC_HEADERS CONFIGURE_DEPENDS "include/*.hpp")
file(GLOB_RECURSE ANET_PRIVATE_SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.hpp")
file(GLOB_RECURSE ANET_TEST_SOURCES CONFIGURE_DEPENDS "src/*test.cpp")

set(ANET_TEST_MAIN "${CMAKE_CURRENT_SOURCE_DIR}/src/main_test.cpp")
list(FILTER ANET_PRIVATE_SOURCES EXCLUDE REGEX ".*test\\.cpp$")
list(REMOVE_ITEM ANET_TEST_SOURCES "${ANET_TEST_MAIN}")


# ---------------------------------------------------------------------
# ターゲット定義
# ---------------------------------------------------------------------
add_library(anet-core STATIC)
add_executable(anet-core-test)


# ---------------------------------------------------------------------
# ソース設定
# ---------------------------------------------------------------------

target_sources(anet-core PRIVATE
    ${ANET_PUBLIC_HEADERS}
    ${ANET_PRIVATE_SOURCES}
)

target_sources(anet-core-test PRIVATE
    ${ANET_TEST_MAIN}
    ${ANET_TEST_SOURCES}
)

target_include_directories(anet-core
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if (MSVC)
    target_compile_definitions(anet-core PUBLIC NOMINMAX)
endif()

target_precompile_headers(anet-core PRIVATE "src/pch.hpp")

target_link_libraries(anet-core PUBLIC
    ${TORCH_LIBRARIES}
    tracy
    anet-wx
)

target_include_directories(anet-core-test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(anet-core-test PRIVATE
    anet-core
    catch2
)

add_test(NAME anet-core-test COMMAND anet-core-test)


# ---------------------------------------------------------------------
# 出力ディレクトリ設定
# ---------------------------------------------------------------------

set_target_properties(anet-core PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_CURRENT_SOURCE_DIR}/lib/Debug
    ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_SOURCE_DIR}/lib/RelWithDebInfo
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release
)

set_target_properties(anet-core-test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_CURRENT_SOURCE_DIR}/bin/Debug
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_SOURCE_DIR}/bin/RelWithDebInfo
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/bin/Release
)


# ---------------------------------------------------------------------
#  DLLコピー
# ---------------------------------------------------------------------

if (MSVC)
  # --- Torchルート決定（上位Torch_DIRを利用） ---
  get_filename_component(TORCH_ROOT "${Torch_DIR}/../../../" ABSOLUTE)
  set(TORCH_DLL_DIR "${TORCH_ROOT}/lib")

  message(STATUS "Torch root: ${TORCH_ROOT}")
  message(STATUS "Torch DLL dir: ${TORCH_DLL_DIR}")

  # --- DLL一覧を取得（構成切替関係なく同一libフォルダ使用） ---
  file(GLOB TORCH_DLLS "${TORCH_DLL_DIR}/*.dll")

  if(TORCH_DLLS)
    add_custom_command(TARGET anet-core-test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Copying libtorch DLLs from ${TORCH_DLL_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TORCH_DLLS}
                $<TARGET_FILE_DIR:anet-core-test>
        COMMENT "Copying libtorch DLLs"
    )
  else()
    message(WARNING "No DLLs found in ${TORCH_DLL_DIR}")
  endif()
endif()


# ---------------------------------------------------------------------
#  IDE用設定
# ---------------------------------------------------------------------

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/include" PREFIX "include" FILES ${ANET_PUBLIC_HEADERS})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "src" FILES ${ANET_PRIVATE_SOURCES})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "test" FILES ${ANET_TEST_MAIN} ${ANET_TEST_SOURCES})
source_group("src" REGULAR_EXPRESSION "cmake_pch.*")
