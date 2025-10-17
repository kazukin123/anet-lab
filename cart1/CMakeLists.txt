cmake_minimum_required(VERSION 3.21)
project(cart1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

# ============================================================
# プロジェクト設定
# ============================================================

add_executable(cart1 "cart1.cpp")
target_link_libraries(cart1 PRIVATE "${TORCH_LIBRARIES}")
set_property(TARGET cart1 PROPERTY CXX_STANDARD 17)

if (MSVC)
   file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
   add_custom_command(TARGET cart1
                      POST_BUILD
                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
                      ${TORCH_DLLS}
                      $<TARGET_FILE_DIR:cart1>)
endif (MSVC)
