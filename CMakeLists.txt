cmake_minimum_required (VERSION 3.1.0)
project(tensorflow_cpp_wrapper)

set(CMAKE_CXX_STANDARD 11)
set(tensorflow_path /home/clark/Documents/tensorflow)

include_directories(${tensorflow_path})
link_directories(${tensorflow_path}/bazel-bin/tensorflow/)

# build tensorflow library
add_library(tensorflow_cpp_wrapper STATIC tensorflow_cpp_wrapper.cpp)
target_link_libraries(tensorflow_cpp_wrapper libtensorflow.so)

# build test executable
add_executable(test_exe test.cpp)
target_link_libraries(test_exe tensorflow_cpp_wrapper)


