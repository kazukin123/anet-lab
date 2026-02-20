cmake_minimum_required(VERSION 3.20)
project(tracy-internal LANGUAGES CXX)

set(TRACY_PUBLIC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/public")

add_library(tracy STATIC "${TRACY_PUBLIC_DIR}/TracyClient.cpp")

target_include_directories(tracy PUBLIC "${TRACY_PUBLIC_DIR}" "${TRACY_PUBLIC_DIR}/tracy")

target_compile_definitions(tracy PUBLIC 
    TRACY_ENABLE 
    TRACY_ON_DEMAND
    TRACY_CALLSTACK=32
)

# 依存ライブラリ
find_package(Threads REQUIRED)
target_link_libraries(tracy PRIVATE Threads::Threads)
if(WIN32)
    target_link_libraries(tracy PRIVATE ws2_32 dbghelp)
endif()
