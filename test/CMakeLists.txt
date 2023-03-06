find_package(GTest REQUIRED)

enable_testing()

add_executable(test_graph ${CMAKE_CURRENT_SOURCE_DIR}/test_graph.cpp)

# target_include_directories(test_graph PUBLIC ${GTest_INCLUDE_DIR})
target_link_libraries(test_graph ${PROJECT_NAME} GTest::gtest_main)
# target_link_directories(test_graph PUBLIC ${PROJECT_SOURCE_DIR}/build)

include(GoogleTest)
gtest_discover_tests(test_graph)