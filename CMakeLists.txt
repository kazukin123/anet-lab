cmake_minimum_required(VERSION 3.20)
project(anet-lab LANGUAGES CXX)

# ---------------------------------------------------------------------
# 共通設定
# ---------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# =====================================================================
# libtorch のパス検出
# =====================================================================

# 手動で Torch_DIR が設定されていればそのまま使用
if(DEFINED Torch_DIR)
    message(STATUS "Torch_DIR provided explicitly: ${Torch_DIR}")

else()
    # Debug 構成？
    if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_CONFIGURATION_TYPES MATCHES "Debug")
        # Torch_DIR_DEBUG（環境変数） があれば最優先
        if(DEFINED ENV{Torch_DIR_DEBUG})
            set(Torch_DIR "$ENV{Torch_DIR_DEBUG}")
            message(STATUS "Using Torch_DIR_DEBUG from environment: ${Torch_DIR}")

        # 次に Torch_DIR（環境変数）があれば使用
        elseif(DEFINED ENV{Torch_DIR})
            set(Torch_DIR "$ENV{Torch_DIR}")
            message(STATUS "Using Torch_DIR (env) for Debug: ${Torch_DIR}")

        # フォールバック
        else()
            set(Torch_DIR "${CMAKE_SOURCE_DIR}/third_party/libtorch/debug/share/cmake/Torch")
            message(WARNING "Torch_DIR_DEBUG and Torch_DIR env not set. Fallback to local debug libtorch: ${Torch_DIR}")
        endif()

    # Release 構成？
    else()
        # Torch_DIR_RELEASE（環境変数）  があれば最優先
        if(DEFINED ENV{Torch_DIR_RELEASE})
            set(Torch_DIR "$ENV{Torch_DIR_RELEASE}")
            message(STATUS "Using Torch_DIR_RELEASE from environment: ${Torch_DIR}")

        # 次に Torch_DIR（環境変数）があれば使用
        elseif(DEFINED ENV{Torch_DIR})
            set(Torch_DIR "$ENV{Torch_DIR}")
            message(STATUS "Using Torch_DIR (env) for Release: ${Torch_DIR}")

        # フォールバック
        else()
            set(Torch_DIR "${CMAKE_SOURCE_DIR}/third_party/libtorch/release/share/cmake/Torch")
            message(WARNING "Torch_DIR_RELEASE and Torch_DIR env not set. Fallback to local release libtorch: ${Torch_DIR}")
        endif()
    endif()
endif()

# 最終的に設定された Torch_DIR を確認
message(STATUS "Final Torch_DIR: ${Torch_DIR}")
list(APPEND CMAKE_PREFIX_PATH "${Torch_DIR}")


# ---------------------------------------------------------------------
# 外部ライブラリ
# ---------------------------------------------------------------------
find_package(Torch REQUIRED)
find_package(wxWidgets COMPONENTS core base OPTIONAL_COMPONENTS gl)

# ---------------------------------------------------------------------
# サブプロジェクト
# ---------------------------------------------------------------------
add_subdirectory(core/anet-core)

if(wxWidgets_FOUND)
    message(STATUS "wxWidgets found: building wx-apps/")
    add_subdirectory(wx-apps/cartpole2)
else()
    message(WARNING "wxWidgets not found: skipping wx-apps/")
endif()

add_subdirectory(apps/cartpole1)

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
