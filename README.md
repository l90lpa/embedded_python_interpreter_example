# Embedded Python Interpreter Example

This project contains an example of using `pybind11` to embedded the Python interpreter into a C++ program, and call python functons from C++.

## Set-up Dev Environment

### MPI
To satisfy the MPI dependency one could install OpenMPI as follows:
- `sudo apt install openmpi-bin libopenmpi-dev`

### Set-up Python dependencies (ensure that the MPI is installed first)
The Python dependenices are listed in `requirements.txt` (this project has been tested with Python 3.10.2). To use `venv` to set-up the dependencies:
- Ensure that you have Python Virtual Environment installed: `sudo apt install python3.10-venv`
- Set-up virtual environment: `python3 -m venv .venv`
- Activiate the virtual environment: `source ./.venv/bin/activate`
- Install Python dependencies: `pip install -r requirements.txt`

## Build

### Cmake Config Search Path - PyBind11
With pybind11 installed through pip, its CMake `.config` may not be in the config search paths. To find we can use the command: `pybind11-config --cmakedir`.

### CMake configure
- `mkdir build`
- `cd build`
- `cmake ../src -DCMAKE_PREFIX_PATH=$(pybind11-config --cmakedir)`

#### Errors:
- If you have an error like:
```
CMake Error in CMakeLists.txt:
Imported target "pybind11::module" includes non-existent path
  "/usr/include/python3.10"
in its INTERFACE_INCLUDE_DIRECTORIES.  Possible reasons include:
* The path was deleted, renamed, or moved to another location.
* An install or uninstall procedure did not complete successfully.
* The installation package was faulty and references files it does not
provide.
```
Then this suggests that the Python development headers might not be installed. You can install the headers using apt, `sudo apt install python3-dev`.

### Cmake build
- `cmake --build .`

## Run

The build generates a executable artifact, `cpp_python`. By default the embedded Python interpreter will have access to the terminals environment (environment variables), therefore it will have access to the Python search path list `PYTHONPATH`. Hence if our Python dependencies are installed in some virtual environment, that is active when running the program `cpp_python` then the import statement `` while work. We can manually add to this searhc path list as follows `export PYTHONPATH=$PYTHONPATH:/path/to/some/directory`. Additionally, the current working directory is included in `sys.path` when using the embedded interpreter, see [pybind11 importing modules](https://pybind11.readthedocs.io/en/latest/advanced/embedding.html#importing-modules). 

Since we are loading a module `python_module` that is not going to be in the search paths by default, we either need to run the executable from the project root (to take advantage of the current working directory being a search path), or modify `PYTHONPATH`
