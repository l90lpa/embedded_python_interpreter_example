cmake_minimum_required(VERSION 3.26)
project(cpp_python)

find_package(pybind11 CONFIG REQUIRED)

add_executable(cpp_python main.cpp)
target_link_libraries(cpp_python PRIVATE pybind11::embed)