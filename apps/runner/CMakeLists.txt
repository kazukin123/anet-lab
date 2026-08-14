# anet-lab/apps/runner/CMakeLists.txt


# ---------------------------------------------------------------------
# プロジェクト定義
# ---------------------------------------------------------------------

cmake_minimum_required(VERSION 3.20)
project(AnetRLRunner)


# ---------------------------------------------------------------------
# ファイル収集
# ---------------------------------------------------------------------

file(GLOB_RECURSE ANET_PRIVATE_SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.hpp")
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
    GridMazeEnv
    ImageClsEnv
)

if(ANET_ATARI_ENABLED)
    target_link_libraries(AnetRLRunner PRIVATE AtariEnv)
    target_compile_definitions(AnetRLRunner PRIVATE ANET_HAS_ATARI)
endif()

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

  # --- wxWidgets DLLコピー（Configモード＝wxソースビルド時のみ） ---
  # Moduleモード（ローカルのvcpkg toolchain）ではapplocal配備がDLLコピーを担うため対象外
  if(TARGET wx::core)
    get_target_property(ANET_WX_CORE_TYPE wx::core TYPE)
    if(ANET_WX_CORE_TYPE STREQUAL "SHARED_LIBRARY")
      add_custom_command(TARGET AnetRLRunner POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different
                  $<TARGET_FILE:wx::base>
                  $<TARGET_FILE:wx::core>
                  $<TARGET_FILE:wx::gl>
                  $<TARGET_FILE:wx::aui>
                  $<TARGET_FILE_DIR:AnetRLRunner>
          COMMENT "Copying wxWidgets DLLs"
      )
    endif()
  endif()

  # --- 汎用 runtime DLLコピー（third_party/runtime_dlls に置いた DLL を配備） ---
  # 実行時ロードされる DLL（SDL2.dll 等）の置き場。詳細は third_party/runtime_dlls/README.md。
  # DLL 追加・削除後は再 configure が必要（GLOB は configure 時評価）。
  set(ANET_RUNTIME_DLL_DIR "${PROJECT_SOURCE_DIR}/third_party/runtime_dlls")
  file(GLOB ANET_RUNTIME_DLLS "${ANET_RUNTIME_DLL_DIR}/*.dll")

  if(ANET_RUNTIME_DLLS)
    add_custom_command(TARGET AnetRLRunner POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ANET_RUNTIME_DLLS}
                $<TARGET_FILE_DIR:AnetRLRunner>
        COMMENT "Copying runtime DLLs"
    )
  else()
    message(STATUS "No runtime DLLs in ${ANET_RUNTIME_DLL_DIR} (skip)")
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
