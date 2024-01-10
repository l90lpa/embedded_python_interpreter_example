#include <pybind11/embed.h> // everything needed for embedding
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <mpi.h>
#include <vector>
#include <iostream>
#include <cassert>

namespace py = pybind11;

template<typename T>
std::optional<py::array_t<T>> to_array(py::object obj) {
    auto array = py::array_t<T, py::array::c_style>::ensure(obj);
    return !array ? std::nullopt : std::make_optional(std::move(array));
}

template<typename T>
std::vector<T> copy_array(const py::array_t<T>& array) {
    return std::vector<T>{array.data(), array.data()+array.nbytes()/array.itemsize()};
}

py::object create_constant_height_initial_condition(py::module_& funcs, int nx, int ny) {
    py::array_t<double> u{{nx+2, ny+2}};
    py::array_t<double> v{{nx+2, ny+2}};
    py::array_t<double> h{{nx+2, ny+2}};
    auto view_u = u.mutable_unchecked<2>();
    auto view_v = v.mutable_unchecked<2>();
    auto view_h = h.mutable_unchecked<2>();
    for(py::ssize_t i = 0; i < view_u.shape(0); ++i) {
        for(py::ssize_t j = 0; j < view_u.shape(1); ++j) {
            view_u(i,j) = 0;
            view_v(i,j) = 0;
            view_h(i,j) = 5000;
        }
    }
    return funcs.attr("State")(u, v, h);
}

int main(int argc, char** argv) {
    bool create_ic_on_cpp_side = false;
    bool zero_ic_h_field = false;

    // start the interpreter and keep it alive
    py::scoped_interpreter guard{}; 

    // start the MPI runtime and create a comm handle.
    MPI_Init(&argc, &argv);
    auto comm = MPI_COMM_WORLD;
    auto comm_int = py::int_{MPI_Comm_c2f(comm)};

    // import the py_swe_interface
    py::module_ funcs = py::module_::import("python_module.py_swe_interface");

    // create the geometry
    int cxx_nx = 100, cxx_ny = 100;
    auto ny = py::int_(cxx_nx);
    auto nx = py::int_(cxx_ny);
    auto xmax = py::float_(100000.0);
    auto ymax = py::float_(100000.0);
    auto geometry = funcs.attr("create_geometry")(comm_int, nx, ny, xmax, ymax);

    // create the initial condition
    auto s0 = [&]() {
        if(create_ic_on_cpp_side) {
            std::cout << "creating initial condition with constant 5000 height field\n";
            return create_constant_height_initial_condition(funcs, cxx_nx, cxx_ny);
        } else {
            std::cout << "creating initial condition with tsunami pulse height field\n";
            return funcs.attr("create_tsunami_pulse_initial_condition")(geometry);
        }
    }();
    
    // Optionally modify the h field to be zero
    if(zero_ic_h_field) {
        // Alternatively could do, `auto h_field = s0.attr("h").cast<py::array_t<double>>();`
        auto maybe_h_field = to_array<double>(s0.attr("h"));
        assert(maybe_h_field);

        auto h_field = maybe_h_field.value();
        auto view = h_field.mutable_unchecked<2>();
        std::cout << "Zeroing h field (of shape << " << view.shape(0) << ", " << view.shape(1) << ") ...\n";
        for(py::ssize_t i = 0; i < view.shape(0); ++i) {
            for(py::ssize_t j = 0; j < view.shape(1); ++j) {
                view(i,j) = 0;
            }
        }
    }

    // run the model
    auto root = py::int_(0);
    funcs.attr("step_model")(geometry, s0, comm_int, root);

    MPI_Finalize();

    return 0;
}