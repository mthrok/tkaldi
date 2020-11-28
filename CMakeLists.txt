cmake_minimum_required(VERSION 3.18)

project(tkaldi)

set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ standard whose features are requested to build this target.")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory(src/libtkaldi)
