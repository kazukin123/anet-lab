file(GLOB_RECURSE ANET_ALGO_SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.hpp" "include/*.hpp")
file(GLOB_RECURSE ANET_ALGO_TESTS CONFIGURE_DEPENDS "src/*_test.cpp")
list(REMOVE_ITEM ANET_ALGO_SOURCES ${ANET_ALGO_TESTS})

add_library(anet-algo-muzero STATIC ${ANET_ALGO_SOURCES})
target_include_directories(anet-algo-muzero PUBLIC include PRIVATE src)
target_link_libraries(anet-algo-muzero PUBLIC anet-core-base)

add_executable(anet-algo-muzero-test ${ANET_ALGO_TESTS})
target_include_directories(anet-algo-muzero-test PRIVATE src)
target_link_libraries(anet-algo-muzero-test PRIVATE anet-algo-muzero anet-core-base anet-core-testing)
anet_set_test_runtime(anet-algo-muzero-test)
add_test(NAME anet-algo-muzero-test COMMAND anet-algo-muzero-test)
set_tests_properties(anet-algo-muzero-test PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
