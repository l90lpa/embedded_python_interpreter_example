#include <pybind11/embed.h> // everything needed for embedding
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

template <typename Sequence>
inline py::array_t<typename Sequence::value_type> as_pyarray(Sequence &&seq) {
    auto size = seq.size();
    auto data = seq.data();
    std::unique_ptr<Sequence> seq_ptr = std::make_unique<Sequence>(std::move(seq));
    auto capsule = py::capsule(seq_ptr.get(), [](void *p) { std::unique_ptr<Sequence>(reinterpret_cast<Sequence*>(p)); });
    seq_ptr.release();
    return py::array(size, data, capsule);
}

template <typename Sequence>
inline py::array_t<typename Sequence::value_type> to_pyarray(const Sequence &seq) {
    return py::array(seq.size(), seq.data());
}

int main() {
    py::scoped_interpreter guard{}; // start the interpreter and keep it alive

    py::print("Hello, World!"); // use the Python API

    py::module_ funcs = py::module_::import("python_module.python_functions");
    funcs.attr("print_env")();

    std::vector<float> cpp_array({1,2,3,4,5});

    auto np_array = as_pyarray(std::move(cpp_array));
    funcs.attr("flushed_print")(np_array);

    return 0;
}