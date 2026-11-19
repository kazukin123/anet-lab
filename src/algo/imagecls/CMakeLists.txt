file(GLOB_RECURSE ANET_ALGO_SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.hpp" "include/*.hpp")
file(GLOB_RECURSE ANET_ALGO_TESTS CONFIGURE_DEPENDS "src/*_test.cpp")
list(REMOVE_ITEM ANET_ALGO_SOURCES ${ANET_ALGO_TESTS})

add_library(anet-algo-imagecls STATIC ${ANET_ALGO_SOURCES})
target_include_directories(anet-algo-imagecls PUBLIC include PRIVATE src)
target_link_libraries(anet-algo-imagecls PUBLIC anet-core-base)

add_executable(anet-algo-imagecls-test ${ANET_ALGO_TESTS})
target_include_directories(anet-algo-imagecls-test PRIVATE src)
target_link_libraries(anet-algo-imagecls-test PRIVATE anet-algo-imagecls anet-core-base anet-core-testing)
anet_set_test_runtime(anet-algo-imagecls-test)
add_test(NAME anet-algo-imagecls-test COMMAND anet-algo-imagecls-test)
set_tests_properties(anet-algo-imagecls-test PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
