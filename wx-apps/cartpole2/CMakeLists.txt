cmake_minimum_required(VERSION 3.18)
project(CartPoleRLGUI)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_COLOR_MAKEFILE ON)

find_package(wxWidgets REQUIRED COMPONENTS core base)
find_package(Torch REQUIRED)

if (MSVC)
  add_compile_options(/std:c++17 /permissive- /Zc:__cplusplus)
  add_definitions(-D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)
  add_compile_definitions(_HAS_CONDITIONAL_EXPLICIT=0)
  add_compile_options(/wd4267)  # warning C4267を無効化
  add_compile_options(/MP8)
  add_link_options(/DEBUG:FULL)
  #add_link_options(/DEBUG:FASTLINK)
  add_link_options(/INCREMENTAL)

#  add_compile_options(/diagnostics:color)
#  add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
endif()

include(${wxWidgets_USE_FILE})


add_executable(CartPoleRLGUI
    "src/app.cpp"
    "src/CartPoleEnv.cpp"
    "src/CartPoleFrame.cpp"
    "src/CartPoleCanvas.cpp"
    "src/PlotPanel.cpp"
)

target_precompile_headers(CartPoleRLGUI PRIVATE "src/pch.hpp")
target_link_libraries(CartPoleRLGUI PRIVATE
    anet-core
    ${TORCH_LIBRARIES}
    ${wxWidgets_LIBRARIES}
)

set_target_properties(CartPoleRLGUI PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/bin/Release
    RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_CURRENT_SOURCE_DIR}/bin/Debug
)

if (WIN32)
  set_target_properties(CartPoleRLGUI PROPERTIES
      WIN32_EXECUTABLE TRUE
      LINK_FLAGS "/SUBSYSTEM:WINDOWS"
  )
endif()

if (MSVC)
   file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
   add_custom_command(TARGET CartPoleRLGUI
                      POST_BUILD
                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
                      ${TORCH_DLLS}
                      $<TARGET_FILE_DIR:CartPoleRLGUI>)
endif (MSVC)


