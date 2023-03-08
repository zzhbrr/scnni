find_package(GTest REQUIRED)

enable_testing()

add_executable(test_graph ${CMAKE_CURRENT_SOURCE_DIR}/test_graph.cpp)
target_link_libraries(test_graph ${PROJECT_NAME} GTest::gtest_main)

# add_executable(test_tensor ${CMAKE_CURRENT_SOURCE_DIR}/test_tensor.cpp)
# target_link_libraries(test_tensor ${PROJECT_NAME} GTest::gtest_main)
# target_link_libraries(test_tensor PUBLIC Eigen3::Eigen GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_graph)