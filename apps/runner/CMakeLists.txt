# anet-lab/apps/runner/CMakeLists.txt


# ---------------------------------------------------------------------
# プロジェクト定義
# ---------------------------------------------------------------------

cmake_minimum_required(VERSION 3.20)
project(AnetRLRunner)


# ---------------------------------------------------------------------
# ファイル収集
# ---------------------------------------------------------------------

file(GLOB_RECURSE ANET_PRIVATE_SOURCES "src/*.cpp" "src/*.hpp")
file(GLOB ANET_CONFIGS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/config/*.txt")

if(WIN32)
    set(ANET_WIN_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/runner.manifest"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/runner.rc"
    )
endif()


# ---------------------------------------------------------------------
# ターゲット定義
# ---------------------------------------------------------------------
add_executable(AnetRLRunner)


# ---------------------------------------------------------------------
# ソース設定
# ---------------------------------------------------------------------

target_sources(AnetRLRunner PRIVATE
    ${ANET_CONFIGS}
    ${ANET_PRIVATE_SOURCES}
    ${ANET_WIN_SOURCES}
)

target_precompile_headers(AnetRLRunner PRIVATE "src/pch.hpp")

target_link_libraries(AnetRLRunner PRIVATE
    ${TORCH_LIBRARIES}
    anet-wx
    box2d
    tracy
    anet-core
    LunarLanderEnv
    CartPoleEnv
    DropMergeEnv
)

if (WIN32)
  set_target_properties(AnetRLRunner PROPERTIES
      WIN32_EXECUTABLE TRUE
      LINK_FLAGS "/SUBSYSTEM:WINDOWS"
  )
endif()


# ---------------------------------------------------------------------
# 出力ディレクトリ設定
# ---------------------------------------------------------------------

set_target_properties(AnetRLRunner PROPERTIES
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
    add_custom_command(TARGET AnetRLRunner POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Copying libtorch DLLs from ${TORCH_DLL_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TORCH_DLLS}
                $<TARGET_FILE_DIR:AnetRLRunner>
        COMMENT "Copying libtorch DLLs"
    )
  else()
    message(WARNING "No DLLs found in ${TORCH_DLL_DIR}")
  endif()
endif()


# ---------------------------------------------------------------------
#  IDE用設定
# ---------------------------------------------------------------------

#source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "src" FILES ${ANET_PRIVATE_SOURCES})
#source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/config" PREFIX "config" FILES ${ANET_CONFIGS})
source_group("config" FILES ${ANET_CONFIGS})
source_group("src" FILES ${ANET_PRIVATE_SOURCES})
source_group("src" REGULAR_EXPRESSION "cmake_pch.*")
if(WIN32)
    source_group("src" FILES ${ANET_WIN_SOURCES})
endif()
