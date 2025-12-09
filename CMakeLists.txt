cmake_minimum_required(VERSION 3.20)
project(anet-lab LANGUAGES CXX)
set(CMAKE_CONFIGURATION_TYPES "Debug;RelWithDebInfo;Release" CACHE STRING "" FORCE)

# ---------------------------------------------------------------------
# 共通設定
# ---------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)

# =====================================================================
# libtorch のパス検出
# =====================================================================

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
find_package(wxWidgets REQUIRED COMPONENTS core base)
#find_package(wxWidgets COMPONENTS core base OPTIONAL_COMPONENTS gl)

include_directories("${PROJECT_SOURCE_DIR}/third_party/nvtx3")

# ---------------------------------------------------------------------
# サブプロジェクト
# ---------------------------------------------------------------------
add_subdirectory(core/anet-core)

if(wxWidgets_FOUND)
    message(STATUS "wxWidgets found: building apps-wx/")
    add_subdirectory(apps-wx/cartpole2)
else()
    message(WARNING "wxWidgets not found: skipping apps-wx/")
endif()

#add_subdirectory(apps/cartpole1)

# ---------------------------------------------------------------------
# Doxygen
# ---------------------------------------------------------------------
find_package(Doxygen REQUIRED)
set(DOXYGEN_IN  ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile)
add_custom_target(doc
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_IN}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/docs
    COMMENT "Generating full documentation with Doxygen"
)
