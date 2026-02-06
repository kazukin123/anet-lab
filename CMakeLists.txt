# anet-lab/CMakeLists.txt


# ---------------------------------------------------------------------
# プロジェクト定義
# ---------------------------------------------------------------------

cmake_minimum_required(VERSION 3.20)
project(anet-lab LANGUAGES CXX)


# ---------------------------------------------------------------------
# 共通設定
# ---------------------------------------------------------------------

set(CMAKE_CONFIGURATION_TYPES "Debug;RelWithDebInfo;Release" CACHE STRING "" FORCE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
#set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#set(CMAKE_COLOR_MAKEFILE ON)
#set(CMAKE_VERBOSE_MAKEFILE ON)


# ---------------------------------------------------------------------
# ビルド定数定義
# ---------------------------------------------------------------------

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG -DDEBUG -DANET_ENABLE_DEBUGINFO")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DNDEBUG -DANET_ENABLE_DEBUGINFO")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(_DEBUG ANET_ENABLE_DEBUGINFO)
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebugInfo")
    add_compile_definitions(NDEBUG ANET_ENABLE_DEBUGINFO)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_definitions(NDEBUG)
endif()

add_compile_definitions(WXINTL_NO_GETTEXT_MACRO) # N_()マクロ競合対策


# ---------------------------------------------------------------------
# コンパイルオプション
# ---------------------------------------------------------------------
if (MSVC)
    #add_compile_definitions(_HAS_CONDITIONAL_EXPLICIT=0)
    #add_compile_options(/MP8)
    #add_link_options(/DEBUG:FULL)
    #add_link_options(/DEBUG:FASTLINK)
    add_link_options(/INCREMENTAL)
    #add_compile_options(/diagnostics:color)
    #add_compile_options(/wd4267)  # warning C4267を無効化

    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")  # mtd
    add_compile_options(/utf-8)
endif()


# ---------------------------------------------------------------------
# libtorch のパス検出
# ---------------------------------------------------------------------

# Multi-config (Visual Studio 等)
if(CMAKE_CONFIGURATION_TYPES)
    # 現在の構成を得る
    if(NOT DEFINED CMAKE_BUILD_TYPE)
        # VS では CMAKE_BUILD_TYPE は空なのでジェネレータ表現で判定
        set(_cfg "$<CONFIG>")
    else()
        set(_cfg "${CMAKE_BUILD_TYPE}")
    endif()

    message(STATUS "Active configuration: ${_cfg}")

    # Debug 構成
    if(_cfg STREQUAL "Debug")
        if(DEFINED ENV{Torch_DIR_DEBUG})
            set(Torch_DIR "$ENV{Torch_DIR_DEBUG}")
            message(STATUS "Using Torch_DIR_DEBUG: ${Torch_DIR}")
        elseif(DEFINED ENV{Torch_DIR})
            set(Torch_DIR "$ENV{Torch_DIR}")
            message(STATUS "Using Torch_DIR (env) for Debug: ${Torch_DIR}")
        else()
            set(Torch_DIR "${CMAKE_SOURCE_DIR}/third_party/libtorch/debug/share/cmake/Torch")
            message(WARNING "Fallback: local libtorch debug")
        endif()

    # RelWithDebInfo と Release は同じ扱い
    else()
        if(DEFINED ENV{Torch_DIR_RELEASE})
            set(Torch_DIR "$ENV{Torch_DIR_RELEASE}")
            message(STATUS "Using Torch_DIR_RELEASE: ${Torch_DIR}")
        elseif(DEFINED ENV{Torch_DIR})
            set(Torch_DIR "$ENV{Torch_DIR}")
            message(STATUS "Using Torch_DIR (env) for Release: ${Torch_DIR}")
        else()
            set(Torch_DIR "${CMAKE_SOURCE_DIR}/third_party/libtorch/release/share/cmake/Torch")
            message(WARNING "Fallback: local libtorch release")
        endif()
    endif()

# Single-config (MinGW / Ninja 単体)
else()
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(Torch_DIR "${CMAKE_SOURCE_DIR}/third_party/libtorch/debug/share/cmake/Torch")
    else()
        set(Torch_DIR "${CMAKE_SOURCE_DIR}/third_party/libtorch/release/share/cmake/Torch")
    endif()
endif()

message(STATUS "Final Torch_DIR: ${Torch_DIR}")
list(APPEND CMAKE_PREFIX_PATH "${Torch_DIR}")


# ---------------------------------------------------------------------
# 外部ライブラリ
# ---------------------------------------------------------------------
find_package(Torch REQUIRED)
find_package(wxWidgets REQUIRED COMPONENTS core base gl aui)


# ---------------------------------------------------------------------
# wxWidgets 互換性レイヤー
# ---------------------------------------------------------------------
add_library(anet-wx INTERFACE)
if(TARGET wx::core)
    # --- 【CI環境 / Configモード】 ---
    # wx::core などのターゲットが見つかった場合
    message(STATUS "wxWidgets: Config mode detected (Targets available)")
    target_link_libraries(anet-wx INTERFACE wx::core wx::base wx::gl wx::aui)
else()
    # --- 【ローカル環境 / Moduleモード】 ---
    # ターゲットがなく、変数(wxWidgets_LIBRARIES)のみの場合
    message(STATUS "wxWidgets: Module mode detected (Variables available)")
    
    # Moduleモードではこのファイルをincludeして定義やパスを一括設定するのが定石
    if(wxWidgets_USE_FILE)
        include(${wxWidgets_USE_FILE})
    endif()

    # 変数を仲介ターゲットに紐付ける
    target_link_libraries(anet-wx INTERFACE ${wxWidgets_LIBRARIES})
    
    # includeパスも明示的に継承させる (USE_FILEで足りない場合用)
    if(wxWidgets_INCLUDE_DIRS)
        target_include_directories(anet-wx INTERFACE ${wxWidgets_INCLUDE_DIRS})
    endif()
endif()


# ---------------------------------------------------------------------
# third_party
# ---------------------------------------------------------------------
include_directories("${PROJECT_SOURCE_DIR}/third_party/nvtx3/include")
include_directories("${PROJECT_SOURCE_DIR}/third_party/nlohmann/include")


# ---------------------------------------------------------------------
# サブプロジェクト
# ---------------------------------------------------------------------
add_subdirectory(third_party/box2d-2.4.2)

add_subdirectory(core/anet-core)
add_subdirectory(core/envs/lunarlander1)
add_subdirectory(core/envs/cartpole2)
add_subdirectory(core/envs/dropmerge1)
#add_subdirectory(apps/cartpole1)
#add_subdirectory(apps/cartpole2)
#add_subdirectory(apps/lunarlander1)
add_subdirectory(apps/runner)


# ---------------------------------------------------------------------
# Doxygen
# ---------------------------------------------------------------------
find_package(Doxygen)

if(DOXYGEN_FOUND)
    set(DOXYGEN_IN  ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile)
    add_custom_target(doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_IN}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/docs
        COMMENT "Generating full documentation with Doxygen"
    )
else()
    message(STATUS "Doxygen not found. Documentation target will be skipped.")
endif()

