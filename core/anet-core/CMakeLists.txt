
find_package(wxWidgets COMPONENTS core base OPTIONAL_COMPONENTS gl)

if (MSVC)
    add_compile_options(/wd4267)  # warning C4267を無効化
endif()

include(${wxWidgets_USE_FILE})

add_library(anet-core STATIC
    "src/rl.cpp"
    "src/heat_map.cpp"
    "src/metrics_logger.cpp"
    "src/dqn_agent.cpp"
    "src/anet_rl_samples.cpp"
)

target_include_directories(anet-core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(anet-core
    PUBLIC ${TORCH_LIBRARIES}
)

set_property(TARGET anet-core PROPERTY CXX_STANDARD 17)
