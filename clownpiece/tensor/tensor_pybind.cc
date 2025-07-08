#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <sstream>
#include "tensor.h"
#include "meta.h"
#include <string>

// Macro to simplify GIL release for compute-heavy functions
#define RELEASE_GIL py::call_guard<py::gil_scoped_release>()

// "Idiot-proof" macros for automatic GIL release on compute functions
#define DEF_COMPUTE_METHOD(name, func, ...) \
    .def(name, func, ##__VA_ARGS__, RELEASE_GIL)

#define DEF_COMPUTE_LAMBDA(name, lambda_expr, ...) \
    .def(name, lambda_expr, ##__VA_ARGS__, RELEASE_GIL)

#define MODULE_DEF_COMPUTE(name, func, ...) \
    m.def(name, func, ##__VA_ARGS__, RELEASE_GIL)

// Macro for non-compute functions (keeps GIL)
#define DEF_SIMPLE(name, func, ...) \
    .def(name, func, ##__VA_ARGS__)

namespace py = pybind11;
using shape_t = at::shape_t;
using Tensor = at::Tensor;
using stride_t = at::stride_t;
using dtype = at::dtype;
using Storage = at::Storage;
using slice_t = at::slice_t;

void parse_nested_list(const py::handle& obj, std::vector<at::dtype>& data, shape_t& shape, int depth);
slice_t py_slice_to_slice_t(const py::slice& s, ssize_t dim_size);
template <typename Tensor>
std::vector<slice_t> parse_index(const Tensor& self, const py::object &idx);
Tensor tensor_reshape_wrapper(const Tensor &self, py::args args, bool copy);
Tensor tensor_view_wrapper(const Tensor &self, py::args args);
py::object tensor_to_list(const at::Tensor& tensor);

PYBIND11_MODULE(tensor_impl, m) {
    py::class_<at::Tensor, std::shared_ptr<at::Tensor>>(m, "TensorBaseImpl")
        .def(py::init<>())
        .def("__repr__",
            [](const at::Tensor &t) {
                std::ostringstream oss;
                // oss << "I AM CLOWNPIECE TENSOR!\n";
                oss << t;
                return oss.str();
            }
        )
        /*** Part Ø: Bindings for graderlib ***/
        .def_property_readonly("shape", &at::Tensor::get_shape)
        .def("data_at", &at::Tensor::get_data_at, py::arg("index"),
            "Get the data at the specified index. This is for graderlib use only.")
        .def("change_data_at", [](at::Tensor &self, int index, dtype value) {
            self.get_data_at(index) = value;
        }, py::arg("index"), py::arg("value"),
            "Change the data at the specified index. This is for graderlib use only.")
        .def("tolist", &tensor_to_list)
        .def(py::pickle(
            [](const at::Tensor& t) {
                std::cout<<"Serializing TensorBaseImpl to Python tuple..."<<std::endl;
                shape_t shape = t.get_shape();
                std::vector<at::dtype> data;
                std::cout<<"Shape=";
                for (const auto& dim : shape) {
                    std::cout<<dim<<" ";
                }
                std::cout<<std::endl;
                if (t.is_contiguous()) {
                    std::cout<<"Tensor is contiguous, using storage directly."<<std::endl;
                    const Storage& storage = t.get_storage();
                    data.resize(storage.size);
                    std::memcpy(data.data(), storage.data.get(), storage.size * sizeof(at::dtype));
                    std::cout<<"Data=";
                    for (const auto& val : data) {
                        std::cout<<val<<" ";
                    }
                    std::cout<<std::endl;
                    
                } else {
                    std::cout<<"Tensor is not contiguous, sampling elements."<<std::endl;
                    // 逐元素采样
                    size_t ndim = shape.size();
                    std::vector<size_t> indices(ndim, 0);
                    size_t total = 1;
                    for (auto d : shape) total *= d;
                    data.reserve(total);
                    while (true) {
                        // indices: vector<size_t> -> vector<int>
                        std::vector<int> int_indices;
                        int_indices.reserve(indices.size());
                        for (auto idx : indices) int_indices.push_back(static_cast<int>(idx));
                        data.push_back(t[int_indices].item()); // 用你的标量提取方法
                        // 增加 indices
                        int dim = ndim - 1;
                        while (dim >= 0) {
                            indices[dim]++;
                            if (indices[dim] < static_cast<size_t>(shape[dim])) break;
                            indices[dim] = 0;
                            dim--;
                        }
                        if (dim < 0) break;
                    }
                }
                return py::make_tuple(shape, data);
            },
            [](py::tuple t) {
                if (t.size() != 2)
                    throw std::runtime_error("Invalid state for TensorBase!");
                shape_t shape = t[0].cast<shape_t>();
                std::cout<<"Deserializing TensorBaseImpl from Python tuple..."<<std::endl;
                std::cout<<"Shape=";
                for (const auto& dim : shape) {
                    std::cout<<dim<<" ";
                }
                std::cout<<std::endl;
                std::vector<at::dtype> data = t[1].cast<std::vector<at::dtype>>();
                std::cout<<"Data=";
                for (const auto& val : data) {
                    std::cout<<val<<" ";
                }
                std::cout<<std::endl;
                return at::Tensor(shape, data);
            }
        ))

        /*** Part I: constructors, assignments, destructors, and item() ***/
        /* constructors */
        .def(py::init<at::dtype>(), py::arg("value"))
        // .def(py::init<const shape_t &>(), py::arg("shape"))
        .def(py::init<const shape_t &, at::dtype>(), py::arg("shape"), py::arg("value"))
        .def(py::init<const shape_t &, std::function<at::dtype()>>(), py::arg("shape"), py::arg("generator"))
        .def(py::init<const shape_t &, at::vec<at::dtype>>(), py::arg("shape"), py::arg("data"))
        .def(py::init<const shape_t &, at::stride_t, int, at::Storage>(), py::arg("shape"), py::arg("stride"), py::arg("offset"), py::arg("storage"))
        .def(py::init<const at::Tensor&>(), py::arg("other"))

        // Initialize from a Python list given shape
        .def(py::init([](const py::list& list) {
            at::vec<at::dtype> data;
            shape_t shape;
            parse_nested_list(list, data, shape, 0);
            return new at::Tensor(shape, data);
        }), py::arg("data"))

        /* assignments */
        .def("__copy__", &at::Tensor::clone, RELEASE_GIL, "Create a copy of the tensor")
        .def("__deepcopy__", &at::Tensor::clone, RELEASE_GIL, "Create a deep copy of the tensor")

        /* item() */
        .def("item", &at::Tensor::item, "Get the single value of a singleton tensor")



        /*** Part II: utils, clone, make contiguous and copy_ ***/
        .def("dim", &at::Tensor::dim)
        .def("size", [](const Tensor &self) {
            return self.size();
        }, "Get the size of the tensor")
        .def("size", [](const Tensor &self, int dim) {
            return self.size(dim);
        }, py::arg("dim"), "Get the size of a specific dimension")
        .def("is_contiguous", &at::Tensor::is_contiguous, "Check if the tensor is contiguous")

        .def("clone", &at::Tensor::clone, RELEASE_GIL)
        .def("contiguous", &at::Tensor::contiguous, RELEASE_GIL, "Make the tensor contiguous")
        .def("copy_", &at::Tensor::copy_, py::arg("other"), RELEASE_GIL, "Copy data from another tensor")
        .def("scatter_", &at::Tensor::scatter_,
             py::arg("dim"), py::arg("index"), py::arg("src"), RELEASE_GIL,
             "Scatter values from src tensor to this tensor along a specified dimension using index tensor")



        /*** Part III: subscriptor ***/
        .def("__getitem__", [](const at::Tensor &self, const at::veci &index) {
            // std::cout<<"Getting tensor with vector of indices: ";
            return self[index];
        }, "Get a tensor using a vector of indices")
        .def("__getitem__", [](const at::Tensor &self, int index) {
            // std::cout<<"Getting tensor with single index: "<<index<<std::endl;
            return self[index];
        }, "Get a tensor using a single index")
        .def("__getitem__", [](const at::Tensor &self, py::object idx) {
            // std::cout<<"Getting tensor with index object: "<<py::str(idx)<<std::endl;
            
            // Check for mixed indexing case
            if (py::isinstance<py::tuple>(idx)) {
                auto t = idx.cast<py::tuple>();
                bool has_int = false, has_slice = false;
                
                for (size_t i = 0; i < t.size(); ++i) {
                    auto item = t[i];
                    if (py::isinstance<py::int_>(item)) has_int = true;
                    else if (py::isinstance<py::slice>(item) || py::isinstance<py::tuple>(item)) has_slice = true;
                }
                
                if (has_int && has_slice){                                                                   
                    // Mixed indexing case: convert to pure slicing
                    std::vector<slice_t> slices;
                    
                    for (size_t i = 0; i < t.size(); ++i) {
                        auto item = t[i];
                        if (py::isinstance<py::int_>(item)) {
                            int index = item.cast<int>();
                            slices.emplace_back(index, index + 1);
                        } else if (py::isinstance<py::slice>(item)) {
                            slice_t slice = py_slice_to_slice_t(item.cast<py::slice>(), self.size(i));
                            slices.push_back(slice);
                        } else if (py::isinstance<py::tuple>(item)) {
                            auto subt = item.cast<py::tuple>();
                            if (subt.size() == 2) {
                                slices.emplace_back(subt[0].cast<int>(), subt[1].cast<int>());
                            } else {
                                throw std::runtime_error("Invalid tuple size for slice");
                            }
                        } else {
                            throw std::runtime_error("Invalid index type in tuple");
                        }
                    }
                    
                    // Apply all slices together
                    at::Tensor result = self[slices];
                    
                    // Now squeeze out the dimensions that were indexed with integers
                    for (size_t i = 0; i < t.size(); ++i) {
                        if (py::isinstance<py::int_>(t[i])) {
                            // Need to squeeze dimension i, but account for already squeezed dimensions
                            int dim_to_squeeze = i;
                            for (size_t j = 0; j < i; ++j) {
                                if (py::isinstance<py::int_>(t[j])) {
                                    dim_to_squeeze--;
                                }
                            }
                            result = result.squeeze(dim_to_squeeze);
                        }
                    }
                    return result;
                }
            }
            
            // Normal case: use parse_index
            auto slices = parse_index(self, idx);
            // std::cout<<"Parsed slices: ";
            // for(auto & slice : slices) {
            //     std::cout<<"slice="<<slice.first<<":"<<slice.second<<std::endl;
            // }
            
            // Special case: if there's only one slice, use the single slice operator
            if (slices.size() == 1) {
                auto p = self[slices[0]];
                return p;
            }
            
            auto p = self[slices];
            // std::cout<<"new_shape=";
            // for (const auto& dim : p.get_shape()) {
            //     std::cout<<dim<<" ";
            // }
            // std::cout<<std::endl;
            // std::cout<<"stride_=";
            // for (const auto& dim : p.get_stride()) {
            //     std::cout<<dim<<" ";
            // }
            // std::cout<<std::endl;
            // std::cout<<"storage.size="<<p.numel()<<std::endl;
            // for(int i = 0; i < p.numel(); i++) {
            //   std::cout<<p[i]<<" ";
            // }
            // std::cout<<std::endl;
            return p;
            
        }, "Get a sliced tensor (supports int, slice, tuple of int/slice)")

        .def("__setitem__", [](at::Tensor& self, const at::veci& index, float value) {
            self[index] = value;
        }, "Set a tensor value using a vector of indices")
        .def("__setitem__", [](at::Tensor& self, int index, float value) {
            self[index] = value;
        }, "Set a tensor value using a single index")
        .def("__setitem__", [](at::Tensor& self, py::object idx, float value) {
            auto slices = parse_index(self, idx);
            self[slices] = value;
        }, "Set a sliced tensor (supports int, slice, tuple of int/slice)")
        .def("__setitem__", [](at::Tensor& self, int index, const at::Tensor& other) {
            for(int i = 0; i < other.numel(); i++) {
                self[index].data_at(i) = other.data_at(i);
            }
        }, "Set a sliced tensor using a another sliced tensor")
        .def("__setitem__", [](at::Tensor& self, py::object idx, const at::Tensor& other){
            auto slices = parse_index(self, idx);
            self[slices] = other;
        }, "Set a sliced tensor with another tensor (supports int, slice, tuple of int/slice)")



        /*** Part IV: Element-wise Binary and Unary Operators ***/
        .def("__gt__", [](const at::Tensor &a, const at::Tensor &b) { return a > b; }, RELEASE_GIL)
        .def("__gt__", [](const at::Tensor &a, at::dtype b) { return a > at::Tensor(b); }, RELEASE_GIL)
        .def("__gt__", [](at::dtype a, const at::Tensor &b) { return at::Tensor(a) > b; }, RELEASE_GIL)
        .def("__ge__", [](const at::Tensor &a, const at::Tensor &b) { return a >= b; }, RELEASE_GIL)
        .def("__ge__", [](const at::Tensor &a, at::dtype b) { return a >= at::Tensor(b); }, RELEASE_GIL)
        .def("__ge__", [](at::dtype a, const at::Tensor &b) { return at::Tensor(a) >= b; }, RELEASE_GIL)
        .def("__lt__", [](const at::Tensor &a, const at::Tensor &b) { return a < b; }, RELEASE_GIL)
        .def("__lt__", [](const at::Tensor &a, at::dtype b) { return a < at::Tensor(b); }, RELEASE_GIL)
        .def("__lt__", [](at::dtype a, const at::Tensor &b) { return at::Tensor(a) < b; }, RELEASE_GIL)
        .def("__le__", [](const at::Tensor &a, const at::Tensor &b) { return a <= b; }, RELEASE_GIL)
        .def("__le__", [](const at::Tensor &a, at::dtype b) { return a <= at::Tensor(b); }, RELEASE_GIL)
        .def("__le__", [](at::dtype a, const at::Tensor &b) { return at::Tensor(a) <= b; }, RELEASE_GIL)
        .def("__eq__", [](const at::Tensor &a, const at::Tensor &b) { return a == b; }, RELEASE_GIL)
        .def("__eq__", [](const at::Tensor &a, at::dtype b) { return a == at::Tensor(b); }, RELEASE_GIL)
        .def("__eq__", [](at::dtype a, const at::Tensor &b) { return at::Tensor(a) == b; }, RELEASE_GIL)
        .def("__ne__", [](const at::Tensor &a, const at::Tensor &b) { return a != b; }, RELEASE_GIL)
        .def("__ne__", [](const at::Tensor &a, at::dtype b) { return a != at::Tensor(b); }, RELEASE_GIL)
        .def("__ne__", [](at::dtype a, const at::Tensor &b) { return at::Tensor(a) != b; }, RELEASE_GIL)

        .def("__neg__", [](const at::Tensor &a) {return -a;}, RELEASE_GIL)
        .def("__add__", [](const at::Tensor &a, const at::Tensor &b) {return a + b;}, RELEASE_GIL)
        .def("__add__", [](const at::Tensor &a, at::dtype &b) {return a + b;}, RELEASE_GIL)
        .def("__radd__", [](const at::Tensor &a, at::dtype &b) {return a + b;}, RELEASE_GIL)
        .def("__sub__", [](const at::Tensor &a, const at::Tensor &b) {return a - b;}, RELEASE_GIL)
        .def("__sub__", [](const at::Tensor &a, at::dtype &b) {return a - b;}, RELEASE_GIL)
        .def("__rsub__", [](const at::Tensor &a, at::dtype &b) {   return b - a;}, RELEASE_GIL)
        .def("__mul__", [](const at::Tensor &a, const at::Tensor &b) {return a * b;}, RELEASE_GIL)
        .def("__mul__", [](const at::Tensor &a, at::dtype &b) {return a * b;}, RELEASE_GIL)
        .def("__rmul__", [](const at::Tensor &a, at::dtype &b) {return a * b;}, RELEASE_GIL)
        .def("__truediv__", [](const at::Tensor &a, const at::Tensor &b) {return a / b;}, RELEASE_GIL)
        .def("__truediv__", [](const at::Tensor &a, at::dtype &b) {return a / b;}, RELEASE_GIL)
        .def("__rtruediv__", [](const at::Tensor &a, at::dtype &b) {return b / a;}, RELEASE_GIL)

        .def("sign", &at::Tensor::sign, RELEASE_GIL)
        .def("abs", &at::Tensor::abs, RELEASE_GIL)
        .def("__abs__", &at::Tensor::abs, RELEASE_GIL)
        .def("sin", &at::Tensor::sin, RELEASE_GIL)
        .def("cos", &at::Tensor::cos, RELEASE_GIL)
        .def("tanh", &at::Tensor::tanh, RELEASE_GIL)
        .def("clamp", &at::Tensor::clamp, py::arg("min"), py::arg("max"), RELEASE_GIL)
        .def("log", &at::Tensor::log, RELEASE_GIL)
        .def("exp", &at::Tensor::exp, RELEASE_GIL)
        .def("pow", &at::Tensor::pow, py::arg("exponent"), RELEASE_GIL)
        .def("sqrt", &at::Tensor::sqrt, RELEASE_GIL)



        /*** Part V: Matrix Multiplication ***/
        .def("matmul", &at::matmul, py::arg("other"), RELEASE_GIL, "Matrix multiplication of two tensors")
        .def("__matmul__", [](const at::Tensor &a, const at::Tensor &b) {
            return a ^ b;
        }, RELEASE_GIL)


        /*** Part VI: Reduction and Normalization Operations ***/
        .def("sum", &at::Tensor::sum, py::arg("dim"), py::arg("keepdims") = false, RELEASE_GIL, "Sum over a dimension")
        .def("max", &at::Tensor::max, py::arg("dim"), py::arg("keepdims") = false, RELEASE_GIL, "Get the maximum value and its index over a dimension")
        .def("softmax", &at::Tensor::softmax, py::arg("dim"), RELEASE_GIL, "Compute the softmax over a dimension")



        /*** Part VII: Shape Manipulation ***/
        .def("reshape", &tensor_reshape_wrapper, py::kw_only(), py::arg("copy") = false,
             "Reshape the tensor to a new shape, optionally copying data")
        .def("view", &tensor_view_wrapper,
             "Return a view of the tensor with a new shape")
        .def("transpose", &at::Tensor::transpose, py::arg("dim0"), py::arg("dim1"), RELEASE_GIL,
         "Transpose the tensor along two dimensions")
        .def("split", [](const at::Tensor &self, int dim, int split_size) {
            // 分割 tensor
            return self.split(dim, split_size);
        }, py::arg("dim"), py::arg("split_size"), RELEASE_GIL, "Split the tensor into chunks along a dimension")
        .def("split", [](const at::Tensor &self, int dim, const std::vector<int> &split_sections) {
            // 分割 tensor
            return self.split(dim, split_sections);
        }, py::arg("dim"), py::arg("split_sections"), RELEASE_GIL, "Split the tensor into sections along a dimension")
        .def_static("stack", [](const std::vector<at::Tensor> &tensors, int dim = 0) {
            // 堆叠 tensor
            return at::Tensor().stack(tensors, dim);
        }, py::arg("tensors"), py::arg("dim") = 0, RELEASE_GIL, "Stack a list of tensors along a new dimension")
        .def_static("cat", [](const std::vector<at::Tensor> &tensors, int dim = 0) {
            // 拼接 tensor
            return at::Tensor().cat(tensors, dim);
        }, py::arg("tensors"), py::arg("dim") = 0, RELEASE_GIL, "Concatenate a list of tensors along a dimension")
        .def_static("broadcast", [](const at::Tensor &lhs, const at::Tensor &rhs) {
            // 广播两个 tensor
            return at::Tensor().broadcast(lhs, rhs);
        }, py::arg("lhs"), py::arg("rhs"), RELEASE_GIL, "Broadcast two tensors to a common shape")
        .def_static("broadcast", [](const std::vector<at::Tensor> &tensors) {
            // 广播多个 tensor
            return at::Tensor().broadcast(tensors);
        }, py::arg("tensors"), RELEASE_GIL, "Broadcast a list of tensors to a common shape")
        // 增加.T属性
        .def_property_readonly("T", [](const at::Tensor &self) {
            // 通常 2D tensor 的 .T 是 swap 0,1 轴；更高维可按需扩展
            return self.transpose(0, 1);
        }, "Matrix transpose (swap axes 0 and 1)")
        .def_static("ones", [](std::vector<int> shape){
            return at::ones(shape);
        }, RELEASE_GIL)
        .def_static("ones_like", [](const at::Tensor &self) {
            return at::ones_like(self);
        }, RELEASE_GIL, "Create a tensor of ones with the same shape and type as another tensor")
        .def_static("zeros", [](std::vector<int> shape){
            return at::zeros(shape);
        }, RELEASE_GIL)
        .def_static("zeros_like", [](const at::Tensor &self) {
            return at::zeros_like(self);
        }, RELEASE_GIL, "Create a tensor of zeros with the same shape and type as another tensor")
        .def_static("empty", [](std::vector<int> shape){
            return at::empty(shape);
        }, RELEASE_GIL, "Create an empty tensor with the specified shape and uninitialized data")
        .def_static("empty_like", [](const at::Tensor &self) {
            return at::empty_like(self);
        }, RELEASE_GIL, "Create an empty tensor with the same shape and type as another tensor")
        .def_static("randn", [](std::vector<int> shape){
            return at::randn(shape);
        }, RELEASE_GIL, "Create a tensor with random values from a normal distribution")
        .def_static("randn_like", [](const at::Tensor &self) {
            return at::randn_like(self);
        }, RELEASE_GIL, "Create a tensor with random values from a normal distribution with the same shape and type as another tensor")

        /* week 3 adds on*/

        .def("mean", &at::Tensor::mean, py::arg("dim"), py::arg("keepdims") = false, RELEASE_GIL, "Calculate the mean along a dimension")
        .def("var", &at::Tensor::var, py::arg("dim"), py::arg("keepdims") = false, py::arg("unbiased") = true, RELEASE_GIL, "Calculate the variance along a dimension")
        ;

    /*** Part II: utils, clone, make contiguous and copy_ ***/
    m.def("numel", [](at::Tensor& self) {
        return self.numel();
    }, py::arg("self"), "Calculate the number of elements in a tensor given its shape");

    /*** Part IV: Element-wise Binary and Unary Operators ***/
    // __eq__ & __ne__
    m.def("__eq__", [](const at::Tensor &a, const at::Tensor &b) {
        return a == b;
    }, py::arg("a"), py::arg("b"), RELEASE_GIL, "Element-wise equality comparison");
    m.def("__eq__", [](const at::Tensor &a, at::dtype b) {
        return a == at::Tensor(b);
    }, py::arg("a"), py::arg("b"), RELEASE_GIL, "Element-wise equality comparison with scalar");
    m.def("__eq__", [](at::dtype a, const at::Tensor &b) {
        return at::Tensor(a) == b;
    }, py::arg("a"), py::arg("b"), RELEASE_GIL, "Element-wise equality comparison with scalar");
    m.def("__ne__", [](const at::Tensor &a, const at::Tensor &b) {
        return a != b;
    }, py::arg("a"), py::arg("b"), RELEASE_GIL, "Element-wise inequality comparison");
    m.def("__ne__", [](const at::Tensor &a, at::dtype b) {
        return a != at::Tensor(b);
    }, py::arg("a"), py::arg("b"), RELEASE_GIL, "Element-wise inequality comparison with scalar");
    m.def("__ne__", [](at::dtype a, const at::Tensor &b) {
        return at::Tensor(a) != b;
    }, py::arg("a"), py::arg("b"), RELEASE_GIL, "Element-wise inequality comparison with scalar");

    // __lt__ & __le__
    m.def("__lt__", [](const at::Tensor &a, const at::Tensor &b) {
        return a < b;
    }, py::arg("a"), py::arg("b"), RELEASE_GIL, "Element-wise less than comparison");
    m.def("__lt__", [](const at::Tensor &a, at::dtype b) {
        return a < at::Tensor(b);
    }, py::arg("a"), py::arg("b"), RELEASE_GIL, "Element-wise less than comparison with scalar");
    m.def("__lt__", [](at::dtype a, const at::Tensor &b) {
        return at::Tensor(a) < b;
    }, py::arg("a"), py::arg("b"), RELEASE_GIL, "Element-wise less than comparison with scalar");
    m.def("__le__", [](const at::Tensor &a, const at::Tensor &b) {
        return a <= b;
    }, py::arg("a"), py::arg("b"), RELEASE_GIL, "Element-wise less than or equal to comparison");
    m.def("__le__", [](const at::Tensor &a, at::dtype b) {
        return a <= at::Tensor(b);
    }, py::arg("a"), py::arg("b"), RELEASE_GIL, "Element-wise less than or equal to comparison with scalar");
    m.def("__le__", [](at::dtype a, const at::Tensor &b) {
        return at::Tensor(a) <= b;
    }, py::arg("a"), py::arg("b"), RELEASE_GIL, "Element-wise less than or equal to comparison with scalar");

    // __gt__ & __ge__
    m.def("__gt__", [](const at::Tensor &a, const at::Tensor &b) { 
        return a > b;
    }, py::arg("a"), py::arg("b"), RELEASE_GIL, "Element-wise greater than comparison");
    m.def("__gt__", [](const at::Tensor &a, at::dtype b) {
        return a > at::Tensor(b);
    }, py::arg("a"), py::arg("b"), RELEASE_GIL, "Element-wise greater than comparison with scalar");
    m.def("__gt__", [](at::dtype a, const at::Tensor &b) {
        return at::Tensor(a) > b;
    }, py::arg("a"), py::arg("b"), RELEASE_GIL, "Element-wise greater than comparison with scalar");
    m.def("__ge__", [](const at::Tensor &a, const at::Tensor &b) {
        return a >= b;
    }, py::arg("a"), py::arg("b"), RELEASE_GIL, "Element-wise greater than or equal to comparison");
    m.def("__ge__", [](const at::Tensor &a, at::dtype b) {
        return a >= at::Tensor(b);
    }, py::arg("a"), py::arg("b"), RELEASE_GIL, "Element-wise greater than or equal to comparison with scalar");
    m.def("__ge__", [](at::dtype a, const at::Tensor &b) {
        return at::Tensor(a) >= b;
    }, py::arg("a"), py::arg("b"), RELEASE_GIL, "Element-wise greater than or equal to comparison with scalar");


    m.def("sign", [](const at::Tensor& t) {
        return t.sign();
    }, RELEASE_GIL);
    m.def("sin", [](const at::Tensor& t) {
        return t.sin();
    }, RELEASE_GIL);
    m.def("cos", [](const at::Tensor& t) {
        return t.cos();
    }, RELEASE_GIL);
    m.def("tanh", [](const at::Tensor& t) {
        return t.tanh();
    }, RELEASE_GIL);
    m.def("clamp", [](const at::Tensor& t, const at::dtype& min, const at::dtype& max) {
        return t.clamp(min, max);
    }, py::arg("t"), py::arg("min"), py::arg("max"), RELEASE_GIL);
    m.def("log", [](const at::Tensor& t) {
        return t.log();
    }, py::arg("t"), RELEASE_GIL);
    m.def("exp", [](const at::Tensor& t) {
        return t.exp();
    }, RELEASE_GIL);
    m.def("pow", [](const at::Tensor& t, at::dtype exponent) {
        return t.pow(exponent);
    }, py::arg("t"), py::arg("exponent"), RELEASE_GIL);
    m.def("sqrt", [](const at::Tensor& t) {
        return t.sqrt();
    }, RELEASE_GIL);

    /*** Part VI: Reduction and Normalization Operations ***/
    m.def("dot", [](const at::Tensor &a, const at::Tensor &b) {
        return a ^ b;
    }, py::arg("a"), py::arg("b"), RELEASE_GIL, "Dot product of two tensors");
    m.def("sum", &at::Tensor::sum, py::arg("t"), py::arg("dim"), py::arg("keepdims") = false, RELEASE_GIL, "Sum over a dimension");
    m.def("max", &at::Tensor::max, py::arg("t"), py::arg("dim"), py::arg("keepdims") = false, RELEASE_GIL, "Get the maximum value and its index over a dimension");
    m.def("softmax", &at::Tensor::softmax, py::arg("t"), py::arg("dim"), RELEASE_GIL, "Compute the softmax over a dimension");

    /*** Part VII: Shape Manipulation ***/
    m.def("permute", &at::Tensor::permute, py::arg("t"), py::arg("dims"), RELEASE_GIL, "Permute the dimensions of the tensor");
    m.def("transpose", &at::Tensor::transpose, py::arg("t"), py::arg("dim0"), py::arg("dim1"), RELEASE_GIL, "Transpose the tensor along two dimensions");
    m.def("narrow", &at::Tensor::narrow, py::arg("t"), py::arg("dim"), py::arg("start"), py::arg("length"), py::arg("copy") = false, RELEASE_GIL, "Narrow the tensor along a dimension");
    m.def("chunk", &at::Tensor::chunk, py::arg("t"), py::arg("chunks"), py::arg("dim"), RELEASE_GIL, "Split the tensor into chunks along a dimension");
    m.def("split", [](const at::Tensor &self, int split_size, int dim = 0) {
        return self.split(dim, split_size);
    }, py::arg("t"), py::arg("split_size"), py::arg("dim") = 0, RELEASE_GIL, "Split the tensor into chunks of a given size along a dimension");
    m.def("split", [](const at::Tensor &self, const shape_t &split_sizes, int dim = 0) {
        return self.split(dim, split_sizes);
    }, py::arg("t"), py::arg("split_sizes"), py::arg("dim") = 0, RELEASE_GIL, "Split the tensor into chunks of specified sizes along a dimension");
    m.def("stack", [](const std::vector<at::Tensor> &tensors, int dim = 0) {
        return at::Tensor().stack(tensors, dim);
    }, py::arg("tensors"), py::arg("dim") = 0, RELEASE_GIL, "Stack a list of tensors along a new dimension");
    m.def("cat", [](const std::vector<at::Tensor> &tensors, int dim = 0) {
        return at::Tensor().cat(tensors, dim);
    }, py::arg("tensors"), py::arg("dim") = 0, RELEASE_GIL, "Concatenate a list of tensors along a dimension");
    m.def("squeeze", [](const at::Tensor &self, int dim = -1) {
        return self.squeeze(dim);
    }, py::arg("t"), py::arg("dim"), RELEASE_GIL, "Remove a dimension of size 1 from the tensor");
    m.def("unsqueeze", [](const at::Tensor &self, int dim = 0) {
        return self.unsqueeze(dim);
    }, py::arg("t"), py::arg("dim"), RELEASE_GIL, "Add a dimension of size 1 to the tensor");
    m.def("broadcast_to", [](const at::Tensor &self, const shape_t &shape) {
        return self.broadcast_to(shape);
    }, py::arg("t"), py::arg("shape"), RELEASE_GIL, "Broadcast the tensor to a new shape");
    m.def("broadcast_tensors", [](const at::Tensor &lhs, const at::Tensor &rhs) {
        return at::Tensor().broadcast(lhs, rhs);
    }, py::arg("lhs"), py::arg("rhs"), RELEASE_GIL, "Broadcast two tensors to a common shape");
    m.def("broadcast_tensors", [](const std::vector<at::Tensor> &tensors) {
        return at::Tensor().broadcast(tensors);
    }, py::arg("tensors"), RELEASE_GIL, "Broadcast a list of tensors to a common shape");
    

    /*** Part VIII: Other Helper Constuctors: ***/
    m.def("to_singleton_tensor", [](const at::dtype &value, int dim) {
        return at::to_singleton_tensor(value, dim);
    }, py::arg("value"), py::arg("dim"), "Create a singleton tensor with a single value and specified dimension");
    m.def("ones", [](const shape_t &shape) {
        return at::ones(shape);
    }, py::arg("shape"), RELEASE_GIL);
    m.def("ones_like", [](const at::Tensor &self) {
        return at::ones_like(self);
    }, py::arg("self"), RELEASE_GIL, "Create a tensor of ones with the same shape and type as another tensor");
    m.def("zeros", [](const shape_t &shape) {
        return at::zeros(shape);
    }, py::arg("shape"), RELEASE_GIL);
    m.def("zeros_like", [](const at::Tensor &self) {
        return at::zeros_like(self);
    }, py::arg("self"), RELEASE_GIL, "Create a tensor of zeros with the same shape and type as another tensor");
    m.def("randn", [](const shape_t &shape) {
        return at::randn(shape);
    }, py::arg("shape"), RELEASE_GIL);
    m.def("randn_like", [](const at::Tensor &self) {
        return at::randn_like(self);
    }, py::arg("self"), RELEASE_GIL, "Create a tensor of random values with the same shape and type as another tensor");
    m.def("empty", [](const shape_t &shape) {
        return at::empty(shape);
    }, py::arg("shape"), RELEASE_GIL, "Create an empty tensor with the specified shape");
    m.def("empty_like", [](const at::Tensor &self) {
        return at::empty_like(self);
    }, py::arg("self"), RELEASE_GIL, "Create an empty tensor with the same shape and type as another tensor");
    m.def("arange", [](at::dtype start, at::dtype end, int step = 1) {
        return at::arange(start, end, step);
    }, py::arg("start"), py::arg("end"), py::arg("step") = 1, RELEASE_GIL, "Create a tensor with a range of values from start to end with a specified step");
    m.def("range", [](at::dtype start, at::dtype end, int step = 1) {
        return at::range(start, end, step);
    }, py::arg("start"), py::arg("end"), py::arg("step") = 1, RELEASE_GIL, "Create a tensor with a range of values from start to end with a specified step");
    m.def("linspace", [](at::dtype start, at::dtype end, int num_steps) {
        return at::linspace(start, end, num_steps);
    }, py::arg("start"), py::arg("end"), py::arg("num_steps"), RELEASE_GIL, "Create a tensor with linearly spaced values from start to end with a specified number of steps");
}

/* Utils */

// 递归解析嵌套列表的工具函数
void parse_nested_list(const py::handle& obj, std::vector<at::dtype>& data, shape_t& shape, int depth = 0) {
    if (py::isinstance<py::list>(obj)) {
        py::list list = py::cast<py::list>(obj);
        if (list.empty()) {
            throw std::runtime_error("Empty lists are not supported for tensor initialization.");
        }

        // Only add a new dimension if we're going deeper than before
        if (depth >= shape.size()) {
            shape.push_back(list.size());
        } else if (list.size() != static_cast<size_t>(shape[depth])) {
            throw std::runtime_error("Inconsistent list lengths in nested lists.");
        }

        // Recursively process each item
        for (const auto& item : list) {
            parse_nested_list(item, data, shape, depth + 1);
        }
    } else {
        // Leaf node: add the scalar value
        data.push_back(py::cast<at::dtype>(obj));
    }
}

slice_t py_slice_to_slice_t(const py::slice& s, ssize_t dim_size) {
    py::ssize_t start, stop, step, slicelength;
    if (!s.compute(dim_size, &start, &stop, &step, &slicelength)) {
        throw std::runtime_error("Invalid Python slice");
    }
    if (step != 1) {
        throw std::runtime_error("Only step=1 is supported");
    }
    return slice_t(static_cast<int>(start), static_cast<int>(stop));
}

// 递归处理索引对象
template <typename Tensor>
std::vector<slice_t> parse_index(const Tensor& self, const py::object &idx) {
    std::vector<slice_t> slices;

    if (py::isinstance<py::tuple>(idx)) {
        auto t = idx.cast<py::tuple>();
        
        // For pure slice/int cases (non-mixed indexing)
        for (size_t i = 0; i < t.size(); ++i) {
            auto item = t[i];
            if (py::isinstance<py::int_>(item)) {
                int index = item.cast<int>();
                slices.emplace_back(index, index + 1);
            } else if (py::isinstance<py::slice>(item)) {
                slices.push_back(py_slice_to_slice_t(item.cast<py::slice>(), self.size(i)));
            } else if (py::isinstance<py::tuple>(item)) {
                auto subt = item.cast<py::tuple>();
                if (subt.size() == 2) {
                    slices.emplace_back(subt[0].cast<int>(), subt[1].cast<int>());
                } else {
                    throw std::runtime_error("Invalid tuple size for slice");
                }
            } else {
                throw std::runtime_error("Invalid index type in tuple");
            }
        }
    } else if (py::isinstance<py::slice>(idx)) {
        slices.push_back(py_slice_to_slice_t(idx.cast<py::slice>(), self.size(0)));
    } else if (py::isinstance<py::int_>(idx)) {
        int index = idx.cast<int>();
        slices.emplace_back(index, index + 1);
    } else if (py::isinstance<py::tuple>(idx)) {
        auto subt = idx.cast<py::tuple>();
        if (subt.size() == 2) {
            slices.emplace_back(subt[0].cast<int>(), subt[1].cast<int>());
        } else {
            throw std::runtime_error("Invalid tuple size for slice");
        }
    } else {
        throw std::runtime_error("Unsupported index type");
    }
    return slices;
}

Tensor tensor_reshape_wrapper(const Tensor &self, py::args args, bool copy = false) {
    // std::cout<<"Reshaping tensor"<<std::endl;
    std::vector<int> shape;
    if (args.size() == 1 && py::isinstance<py::sequence>(args[0])) {
        shape = args[0].cast<std::vector<int>>();
    } else {
        for (auto item : args) {
            shape.push_back(item.cast<int>());
        }
    }
    return self.reshape(shape, copy);
}

Tensor tensor_view_wrapper(const Tensor &self, py::args args) {
    std::vector<int> shape;
    if (args.size() == 1 && py::isinstance<py::sequence>(args[0])) {
        shape = args[0].cast<std::vector<int>>();
    } else {
        for (auto item : args) {
            shape.push_back(item.cast<int>());
        }
    }
    return self.view(shape);
}

// 递归辅助函数
py::object tensor_to_list_recursive(const at::Tensor& tensor, int dim, std::vector<int> indices) {
    // 如果是最后一维
    if (dim == tensor.dim() - 1) {
        py::list l;
        int len = tensor.size(dim);
        for (int i = 0; i < len; ++i) {
            indices.push_back(i);
            // 计算flat index
            int flat_idx = tensor.get_offset();
            const auto& stride = tensor.get_stride();
            for (int d = 0; d < tensor.dim(); ++d)
                flat_idx += indices[d] * stride[d];
            l.append(py::float_(tensor.get_storage().data[flat_idx]));
            indices.pop_back();
        }
        return l;
    } else {
        py::list l;
        int len = tensor.size(dim);
        for (int i = 0; i < len; ++i) {
            indices.push_back(i);
            l.append(tensor_to_list_recursive(tensor, dim + 1, indices));
            indices.pop_back();
        }
        return l;
    }
}

// 对外接口
py::object tensor_to_list(const at::Tensor& tensor) {
    if (tensor.dim() == 0) {
        // 标量
        return py::float_(tensor.item());
    }
    return tensor_to_list_recursive(tensor, 0, {});
}