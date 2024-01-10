cmake_minimum_required(VERSION 3.22)
project(cpp_python)

find_package(pybind11 CONFIG REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})
set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})

add_executable(cpp_python main.cpp)
target_link_libraries(cpp_python PRIVATE pybind11::embed)
target_link_libraries(cpp_python PRIVATE ${MPI_LIBRARIES})