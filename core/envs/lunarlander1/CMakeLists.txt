# anet-lab/core/envs/lunarlander1/CMakeLists.txt


cmake_minimum_required(VERSION 3.18)
project(LunarLanderEnv)

set(CMAKE_CONFIGURATION_TYPES "Debug;RelWithDebInfo;Release" CACHE STRING "" FORCE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG -DDEBUG -DANET_ENABLE_DEBUGINFO")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DNDEBUG -DANET_ENABLE_DEBUGINFO")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(_DEBUG DEBUG ANET_ENABLE_DEBUGINFO)
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebugInfo")
    add_compile_definitions(NDEBUG ANET_ENABLE_DEBUGINFO)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_definitions(NDEBUG)
endif()

add_library(LunarLanderEnv STATIC
    "src/LunarLanderEnv.cpp"
    "src/LunarLanderView.cpp"
    "src/LunarLander.cpp"
)

target_include_directories(LunarLanderEnv
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_precompile_headers(LunarLanderEnv PRIVATE "src/pch.hpp")

target_compile_definitions(LunarLanderEnv PUBLIC WXINTL_NO_GETTEXT_MACRO)    # N_()マクロ競合対策

target_link_libraries(LunarLanderEnv PRIVATE
    anet-core
    ${TORCH_LIBRARIES}
    anet_wx
    box2d
)

set_target_properties(LunarLanderEnv PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_CURRENT_SOURCE_DIR}/bin/Debug
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_SOURCE_DIR}/bin/RelWithDebInfo
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/bin/Release
)



