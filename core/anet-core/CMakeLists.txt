# anet-core

set(CMAKE_CONFIGURATION_TYPES "Debug;RelWithDebInfo;Release" CACHE STRING "" FORCE)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)


if (MSVC)
    add_compile_options(/wd4267)  # warning C4267を無効化
    add_compile_options(
        $<$<C_COMPILER_ID:MSVC>:/utf-8>
        $<$<CXX_COMPILER_ID:MSVC>:/utf-8>
    )
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

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

find_package(wxWidgets REQUIRED COMPONENTS core base)

include(${wxWidgets_USE_FILE})

add_library(anet-core STATIC
    "src/rl.cpp"
    "src/heat_map.cpp"
    "src/metrics_logger.cpp"
    "src/dqn_agent.cpp"
    "src/config.cpp"
    "src/tensor_check.cpp"
    "src/tensor_util.cpp"
    "src/random.cpp"
    "src/replay_buffer.cpp"
    "src/observers.cpp"
    "src/eval_env.cpp"
    "src/probe.cpp"
    "src/env.cpp"
    "src/profile.cpp"
    "src/thread.cpp"
    "src/trainer.cpp"
    "src/agent.cpp"
    "src/init.cpp"
    "src/rainbow_agent.cpp"
    "src/dqn_based_agent.cpp"
    "src/replay_buffer_impl.cpp"
    "src/default_dqn_agent.cpp"
    "src/scaler_impl.cpp")

target_include_directories(anet-core
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_precompile_headers(anet-core PRIVATE "src/pch.hpp")
target_link_libraries(anet-core PUBLIC ${TORCH_LIBRARIES})

set_target_properties(anet-core PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_CURRENT_SOURCE_DIR}/lib/Debug
    ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_SOURCE_DIR}/lib/RelWithDebugInfo
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release
)

