cmake_minimum_required(VERSION 3.7)
project(SCNNI LANGUAGES CXX)

# Presettings
set(CMAKE_CXX_STANDARD 14)
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to `Debug` as none was specified.")
    set(CMAKE_BUILD_TYPE "Debug")
endif ()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find 3rdparty
# find_package(OpenCV REQUIRED) 
find_package(Eigen3 REQUIRED)

# Add Build library & execuable

# file(GLOB_RECURSE SRCS CONFIGURE_DEPENDS src/*.cpp)
# add_library(${PROJECT_NAME} SHARED ${SRCS})
# target_include_directories(${PROJECT_NAME} PUBLIC
#     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#     $<INSTALL_INTERFACE:include>
# )
# target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)

add_executable(tmp src/tmp.cpp)
# add_executable(tmp src/tmp.cpp src/tensor.cpp)
target_link_libraries(tmp PUBLIC Eigen3::Eigen)

# 单元测试
# add_subdirectory(test)
