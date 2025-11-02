cmake_minimum_required(VERSION 3.18)
project(CartPoleRLGUI)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_COLOR_MAKEFILE ON)

find_package(wxWidgets COMPONENTS core base REQUIRED)
find_package(Torch REQUIRED)
find_package(GTest CONFIG REQUIRED)

if (MSVC)
  add_compile_options(/std:c++17 /permissive- /Zc:__cplusplus)
  add_definitions(-D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)
  add_compile_definitions(_HAS_CONDITIONAL_EXPLICIT=0)
  add_compile_options(/wd4267)  # warning C4267を無効化
  add_compile_options(/Zi) 

  add_compile_options(/MP8)
  add_link_options(/DEBUG:FULL)
  #add_link_options(/DEBUG:FASTLINK)
  add_link_options(/INCREMENTAL)

#  add_compile_options(/diagnostics:color)
#  add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
endif()

include(${wxWidgets_USE_FILE})

# ==========================================================
# 1. GUIアプリ
# ==========================================================
add_executable(CartPoleRLGUI
    "src/app.cpp"
    "src/CartPoleEnv.cpp"
    "src/RLAgent.cpp"
    "src/CartPoleFrame.cpp"
    "src/CartPoleCanvas.cpp"
    "src/PlotPanel.cpp"
    "src/anet_rl_samples.cpp"
    "src/HeatMap.cpp" "src/rl.cpp" "src/MetricsLogger.cpp")

target_precompile_headers(CartPoleRLGUI PRIVATE "src/pch.hpp")
target_include_directories(CartPoleRLGUI PRIVATE "include")
target_link_libraries(CartPoleRLGUI PRIVATE ${wxWidgets_LIBRARIES} ${TORCH_LIBRARIES})
set_target_properties(CartPoleRLGUI PROPERTIES UNITY_BUILD $<CONFIG:Release>)

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

# ==========================================================
# 2. GoogleTestを使った単体テストEXE
# ==========================================================
enable_testing()

add_executable(CartPoleRLTests
        "src/HeatMap.cpp"
         "src/test_heat_map.cpp")

target_include_directories(CartPoleRLTests PRIVATE "include")
target_precompile_headers(CartPoleRLTests PRIVATE "src/pch.hpp")
target_link_libraries(CartPoleRLTests
    PRIVATE
    ${wxWidgets_LIBRARIES}
    ${TORCH_LIBRARIES}
    GTest::gtest_main
)

# CTest / Visual Studio テストエクスプローラー対応
#include(GoogleTest)
#gtest_discover_tests(CartPoleRLTests)
##    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
##    PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
##)

#gtest_add_tests(TARGET CartPoleRLTests
#    TEST_PREFIX "GTest."
#    TEST_LIST all_tests
#)
#gtest_discover_tests(CartPoleRLTests
#    DISCOVERY_MODE PRE_TEST   # ← 再構成せず、実行直前に自動検出
#)

target_link_libraries(CartPoleRLTests PRIVATE GTest::gtest)

# 実行ファイル出力ディレクトリを統一

set_target_properties(CartPoleRLTests 
    PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:CartPoleRLTests>"
    VS_DEBUGGER_COMMAND           "$<TARGET_FILE:CartPoleRLTests>"
    VS_DEBUGGER_ENVIRONMENT       "PATH=%PATH%;${CMAKE_BINARY_DIR}")

#set_target_properties(CartPoleRLGUI CartPoleRLTests PROPERTIES
#    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
#)

if (MSVC)
   file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
   add_custom_command(TARGET CartPoleRLTests
                      PRE_BUILD
                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
                      ${TORCH_DLLS}
                      $<TARGET_FILE_DIR:CartPoleRLTests>)
endif (MSVC)
