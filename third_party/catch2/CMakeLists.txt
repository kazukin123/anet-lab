cmake_minimum_required(VERSION 3.20)
project(catch2-internal LANGUAGES CXX)

add_library(catch2 STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/catch.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/catch.hpp"
)

target_include_directories(catch2
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_compile_definitions(catch2 PRIVATE CATCH_AMALGAMATED_CUSTOM_MAIN)
target_compile_features(catch2 PUBLIC cxx_std_14)

source_group("include" FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/catch.hpp")
source_group("src" FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/catch.cpp")
