cmake_minimum_required(VERSION 3.18)
project(CartPoleRLGUI)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(wxWidgets COMPONENTS core base REQUIRED)
find_package(Torch REQUIRED)

if (MSVC)
  add_compile_options(/std:c++17 /permissive- /Zc:__cplusplus)
  add_definitions(-D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)
  add_compile_definitions(_HAS_CONDITIONAL_EXPLICIT=0)
  add_compile_options(/wd4267)  # warning C4267を無効化
  add_compile_options(/Zi) 
endif()

include(${wxWidgets_USE_FILE})
add_executable(CartPoleRLGUI
    app.cpp
    CartPoleEnv.cpp
    RLAgent.cpp
    CartPoleFrame.cpp
    CartPoleCanvas.cpp
    PlotPanel.cpp)

target_link_libraries(CartPoleRLGUI PRIVATE ${wxWidgets_LIBRARIES} ${TORCH_LIBRARIES})
# GUIアプリとしてリンク（Windows専用）
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

