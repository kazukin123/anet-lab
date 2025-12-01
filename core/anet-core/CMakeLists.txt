# anet-core

set(CMAKE_CONFIGURATION_TYPES "Debug;RelWithDebInfo;Release" CACHE STRING "" FORCE)

set(CMAKE_VERBOSE_MAKEFILE ON)

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

find_package(wxWidgets REQUIRED core base OPTIONAL_COMPONENTS gl)
#find_package(CUDAToolkit REQUIRED)

include(${wxWidgets_USE_FILE})
#include_directories(${CUDAToolkit_INCLUDE_DIRS})
#include_directories("$ENV{CUDA_PATH}/extras/CUPTI/include")
#include_directories("$ENV{CUDA_PATH}/include")
#include_directories("C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.3.2/target-windows-x64/nvtx/include")

add_library(anet-core STATIC
    "src/rl.cpp"
    "src/heat_map.cpp"
    "src/metrics_logger.cpp"
    "src/dqn_agent.cpp"
    "src/config.cpp" "src/tensor_check.cpp" "src/tensor_utils.cpp" "src/random.cpp" 
    "src/replay_buffer.cpp" "src/observers.cpp" "src/eval_env.cpp" "src/probe.cpp")

target_include_directories(anet-core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(anet-core
    PUBLIC ${TORCH_LIBRARIES}
)

set_property(TARGET anet-core PROPERTY CXX_STANDARD 17)
