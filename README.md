# Embedded Python Interpreter Example

This project contains an example of using `pybind11` to embedded the Python interpreter into a C++ program, and call python functons from C++.

## Set-up Dev Environment

### Set-up Python dependencies
The Python dependenices are listed in `requirements.txt` (this project has been tested with Python 3.10.2). To use `venv` to set-up the dependencies:
- Set-up virtual environment: `python3 -m venv .venv`
- Activiate the virtual environment: `source ./.venv/bin/activate`
- Install Python dependencies: `pip install -r requirements.txt`

### Set-up C++ dependencies
The only C++ dependency is `pybind11`. To use `vcpkg` to set-up the dependencies:
- Get the vcpkg repo: `git clone https://github.com/Microsoft/vcpkg.git`
- Set-up the vcpkg repo: `./vcpkg/bootstrap-vcpkg.sh`
- Install pybind11: `./vcpkg/vcpkg install pybind11`

## Build

### CMake configure
- `mkdir build`
- `cd build`
- `cmake ../src -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake`

### Cmake build
- `cmake --build .`

## Run

To run the program, one should start from a terminal with the virtual environment activated, because by default embedded Python interpreter will have access to the terminals environment (environment variables). Also we run the program with the current working directory set to the root of the project, because by default the current working directory is added to the python module search path. Thus we use the following to run the program: `./build/cpp_python`
