add_executable(cartpole1 main.cpp)

target_link_libraries(cartpole1
    PRIVATE
        anet-core
        ${TORCH_LIBRARIES}
)

set_target_properties(cartpole1 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/bin/Release
    RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_CURRENT_SOURCE_DIR}/bin/Debug
)
