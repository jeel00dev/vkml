// Python extension entry point.
//
// Exposes Tensor, dtypes, devices and the operator surface. Deliberately does
// NOT expose Node, Backend, Storage or Allocator: per guardrail 1 in
// docs/adr/0001-graph-ownership-and-ir.md, the internal representation must be
// replaceable without breaking Python callers.
//
// NumPy and DLPack interop both live here, in one place, because of a trap
// worth stating once loudly: DLPack (and therefore nanobind's ndarray) measures
// strides in ELEMENTS, while NumPy and vkml measure them in BYTES. The
// conversion happens at this boundary and nowhere else.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include "vkml/api/ops.h"
#include "vkml/api/tensor.h"
#include "vkml/autograd/autograd.h"
#include "vkml/backend/api/backend.h"
#ifdef VKML_HAS_VULKAN
#    include "vkml/backend/vulkan/vulkan_backend.h"
#endif
#include "vkml/dispatch/executor.h"
#include "vkml/graph/op.h"
#include "vkml/util/coverage.h"
#include "vkml/util/error.h"
#include "vkml/util/log.h"
#include "vkml/util/decisions.h"
#include "vkml/util/env.h"

#include <cstring>
#include <memory>
#include <optional>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
using vkml::Device;
using vkml::DType;
using vkml::Tensor;

namespace {

// ---------------------------------------------------------------------------
// dtype bridging
// ---------------------------------------------------------------------------

nb::dlpack::dtype to_dlpack_dtype(DType dt) {
    switch (dt) {
        case DType::F32: return nb::dtype<float>();
        case DType::F16:
            return nb::dlpack::dtype{static_cast<uint8_t>(nb::dlpack::dtype_code::Float), 16, 1};
        case DType::I32: return nb::dtype<int32_t>();
        case DType::I64: return nb::dtype<int64_t>();
        case DType::Bool: return nb::dtype<bool>();
    }
    throw vkml::DTypeError("unknown dtype");
}

DType from_dlpack_dtype(nb::dlpack::dtype dt) {
    const auto code = static_cast<nb::dlpack::dtype_code>(dt.code);
    if (dt.lanes != 1) {
        throw vkml::DTypeError("vectorised dlpack dtypes are not supported");
    }
    if (code == nb::dlpack::dtype_code::Float) {
        if (dt.bits == 32)
            return DType::F32;
        if (dt.bits == 16)
            return DType::F16;
    } else if (code == nb::dlpack::dtype_code::Int) {
        if (dt.bits == 32)
            return DType::I32;
        if (dt.bits == 64)
            return DType::I64;
    } else if (code == nb::dlpack::dtype_code::Bool) {
        return DType::Bool;
    }
    throw vkml::DTypeError("unsupported array dtype; vkml handles f32, f16, i32, i64 and bool");
}

// ---------------------------------------------------------------------------
// NumPy / DLPack bridge
// ---------------------------------------------------------------------------

using AnyArray = nb::ndarray<nb::c_contig, nb::device::cpu>;

Tensor tensor_from_array(const AnyArray& arr, std::optional<DType> dtype, Device device) {
    std::vector<int64_t> dims;
    dims.reserve(arr.ndim());
    for (size_t i = 0; i < arr.ndim(); ++i) {
        dims.push_back(static_cast<int64_t>(arr.shape(i)));
    }

    const DType src = from_dlpack_dtype(arr.dtype());
    Tensor t = Tensor::from_host(arr.data(), dims, src, device);
    if (dtype.has_value() && *dtype != src) {
        t = t.to(*dtype);
    }
    return t;
}

/// Copies a tensor out into a fresh buffer that Python owns.
///
/// The copy is not avoidable in general: a vkml tensor may be a strided view,
/// may live on a device, and may not be evaluated yet. Materialising a
/// contiguous host buffer and handing over ownership via a capsule is the
/// simple, always-correct answer, and export is not on any hot path.
nb::ndarray<nb::numpy> tensor_to_numpy(const Tensor& t) {
    const Tensor c = t.contiguous();
    c.realize();

    const size_t n = static_cast<size_t>(c.numel());
    const size_t esz = vkml::dtype_size(c.dtype());
    const size_t nbytes = n * esz;

    auto* buf = new std::byte[nbytes == 0 ? 1 : nbytes];
    if (nbytes > 0) {
        c.to_host(buf);
    }

    nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<std::byte*>(p); });

    const std::vector<int64_t> shape_i64 = c.shape();
    std::vector<size_t> shape(shape_i64.begin(), shape_i64.end());

    // Strides omitted: the buffer above is contiguous by construction, so
    // nanobind derives them. This is exactly where a byte/element stride mix-up
    // would otherwise creep in.
    return nb::ndarray<nb::numpy>(buf, shape.size(), shape.data(), owner, nullptr,
                                  to_dlpack_dtype(c.dtype()), nb::device::cpu::value, 0);
}

// ---------------------------------------------------------------------------
// Indexing
// ---------------------------------------------------------------------------

/// Basic __getitem__: integers, slices, and tuples of them.
///
/// Integers drop the axis, slices keep it -- NumPy's rule. Advanced indexing
/// (boolean masks, index arrays) is deliberately absent: it needs gather
/// kernels that do not exist yet, and silently supporting a subset would be
/// worse than a clear error.
Tensor tensor_getitem(const Tensor& self, nb::object key) {
    std::vector<nb::object> items;
    if (nb::isinstance<nb::tuple>(key)) {
        for (nb::handle h : nb::cast<nb::tuple>(key)) {
            items.emplace_back(nb::borrow(h));
        }
    } else {
        items.push_back(key);
    }

    if (static_cast<int>(items.size()) > self.ndim()) {
        throw vkml::IndexError("too many indices for tensor of rank " +
                               std::to_string(self.ndim()));
    }

    Tensor out = self;
    int axis = 0;  // tracks the axis in `out`, which shifts as integers drop axes

    for (nb::object& item : items) {
        if (nb::isinstance<nb::slice>(item)) {
            const auto sl = nb::cast<nb::slice>(item);
            const auto [start, stop, step, len] = sl.compute(static_cast<size_t>(out.size(axis)));
            if (static_cast<int64_t>(step) <= 0) {
                throw vkml::IndexError("negative or zero slice steps are not supported");
            }
            out = out.slice(axis, static_cast<int64_t>(start), static_cast<int64_t>(stop),
                            static_cast<int64_t>(step));
            ++axis;
        } else if (nb::isinstance<nb::int_>(item)) {
            int64_t i = nb::cast<int64_t>(item);
            const int64_t extent = out.size(axis);
            if (i < 0) {
                i += extent;
            }
            if (i < 0 || i >= extent) {
                throw vkml::IndexError("index " + std::to_string(nb::cast<int64_t>(item)) +
                                       " is out of range for axis " + std::to_string(axis) +
                                       " with extent " + std::to_string(extent));
            }
            out = out.slice(axis, i, i + 1).squeeze(axis);
            // axis is NOT incremented: squeezing removed it, so the next index
            // applies to what is now this position.
        } else if (item.is_none()) {
            out = out.unsqueeze(axis);
            ++axis;
        } else {
            throw vkml::IndexError(
                "unsupported index type; vkml supports integers, slices and None "
                "(advanced indexing is not implemented)");
        }
    }
    return out;
}

std::vector<int> to_axes(const nb::object& obj, int ndim) {
    if (obj.is_none()) {
        return {};
    }
    if (nb::isinstance<nb::int_>(obj)) {
        return {nb::cast<int>(obj)};
    }
    std::vector<int> axes;
    for (nb::handle h : nb::cast<nb::sequence>(obj)) {
        axes.push_back(nb::cast<int>(h));
    }
    (void)ndim;
    return axes;
}

}  // namespace

NB_MODULE(_vkml_core, m) {
    m.doc() = "vkml core (C++20 / Vulkan-native deep learning framework)";
    m.attr("__version__") = "0.1.0";

    // Registration order matters and is counter-intuitive: nanobind tries
    // exception translators in REVERSE order of registration, so the LAST one
    // registered is tried FIRST. The base class must therefore go first, or it
    // catches every derived type and Python only ever sees a bare RuntimeError.
    nb::exception<vkml::Error>(m, "Error", PyExc_RuntimeError);
    nb::exception<vkml::InternalError>(m, "InternalError", PyExc_RuntimeError);
    nb::exception<vkml::NotImplementedError>(m, "NotImplementedError", PyExc_NotImplementedError);
    nb::exception<vkml::OutOfMemoryError>(m, "OutOfMemoryError", PyExc_MemoryError);
    nb::exception<vkml::DeviceError>(m, "DeviceError", PyExc_RuntimeError);
    nb::exception<vkml::IndexError>(m, "IndexError", PyExc_IndexError);
    nb::exception<vkml::DTypeError>(m, "DTypeError", PyExc_TypeError);
    nb::exception<vkml::ShapeError>(m, "ShapeError", PyExc_ValueError);

    // -- logging ------------------------------------------------------------
    nb::enum_<vkml::LogLevel>(m, "LogLevel")
        .value("TRACE", vkml::LogLevel::Trace)
        .value("DEBUG", vkml::LogLevel::Debug)
        .value("INFO", vkml::LogLevel::Info)
        .value("WARN", vkml::LogLevel::Warn)
        .value("ERROR", vkml::LogLevel::Error)
        .value("OFF", vkml::LogLevel::Off);

    m.def("set_log_level", &vkml::set_log_level, "level"_a);
    m.def("log_level", &vkml::log_level);
    m.def(
        "set_log_callback",
        [](std::function<void(vkml::LogLevel, std::string)> fn) {
            if (!fn) {
                vkml::set_log_callback(nullptr);
                return;
            }
            vkml::set_log_callback([fn = std::move(fn)](vkml::LogLevel lvl, std::string_view msg) {
                const nb::gil_scoped_acquire gil;
                fn(lvl, std::string(msg));
            });
        },
        "callback"_a.none());

    // -- dtype --------------------------------------------------------------
    nb::enum_<DType>(m, "dtype")
        .value("float32", DType::F32)
        .value("float16", DType::F16)
        .value("int32", DType::I32)
        .value("int64", DType::I64)
        .value("bool", DType::Bool);

    m.def("dtype_size", &vkml::dtype_size, "dtype"_a);
    m.def("dtype_name", [](DType d) { return std::string(vkml::dtype_name(d)); }, "dtype"_a);

    // -- device -------------------------------------------------------------
    nb::class_<Device>(m, "device")
        .def(nb::init<>())
        .def(
            "__init__",
            [](Device* self, std::string_view spec) { new (self) Device(Device::parse(spec)); },
            "spec"_a)
        .def_prop_ro("index", &Device::index)
        .def_prop_ro("is_cpu", &Device::is_cpu)
        .def("__repr__", [](const Device& d) { return "device('" + d.str() + "')"; })
        .def("__str__", &Device::str)
        .def(
            "__eq__", [](const Device& a, const Device& b) { return a == b; }, nb::is_operator())
        .def("__hash__", [](const Device& d) {
            return static_cast<size_t>(d.index()) * 31U + static_cast<size_t>(d.kind());
        });

    m.def("cpu_device", &Device::cpu);
    m.def("available_devices", &vkml::available_devices);

    // -- execution mode -----------------------------------------------------
    m.def("set_eager", &vkml::set_eager, "enabled"_a,
          "Realize after every operation. Same results, easier debugging.");
    m.def("is_eager", &vkml::eager);

    // -- coverage recording -------------------------------------------------
    // Flushed explicitly rather than at process exit: static destruction order
    // across a Python extension is not something to rely on for a file write.
    m.def("_coverage_enabled", &vkml::coverage::enabled,
          "True when VKML_COVERAGE is set and execution is being recorded.");
    m.def("_coverage_dump", &vkml::coverage::dump, "path"_a,
          "Write every distinct observation to a file. Returns the line count.");
    m.def("_coverage_clear", &vkml::coverage::clear);

    // The operator inventory and its categories, read out of the enum itself.
    // The coverage report needs a denominator and needs to know which operators
    // the executor can even observe; parsing the header for either would drift
    // silently the first time an operator is added.
    m.def("_op_names", [] {
        std::vector<std::string> names;
        names.reserve(static_cast<size_t>(vkml::kNumOps));
        for (int i = 0; i < vkml::kNumOps; ++i) {
            names.emplace_back(vkml::op_name(static_cast<vkml::OpKind>(i)));
        }
        return names;
    });

    m.def("_op_categories", [] {
        std::vector<std::pair<std::string, std::string>> out;
        out.reserve(static_cast<size_t>(vkml::kNumOps));
        for (int i = 0; i < vkml::kNumOps; ++i) {
            const auto op = static_cast<vkml::OpKind>(i);
            const char* category = vkml::is_leaf_op(op)   ? "leaf"
                                   : vkml::is_view_op(op) ? "view"
                                                          : "compute";
            out.emplace_back(vkml::op_name(op), category);
        }
        return out;
    });

    // -- Tensor -------------------------------------------------------------
    nb::class_<Tensor> tensor(m, "Tensor");

    tensor.def(nb::init<>())
        .def_prop_ro("shape", [](const Tensor& t) { return nb::tuple(nb::cast(t.shape())); })
        .def_prop_ro("ndim", &Tensor::ndim)
        .def_prop_ro("size", &Tensor::numel)
        .def_prop_ro("dtype", &Tensor::dtype)
        .def_prop_ro("device", &Tensor::device)
        .def_prop_ro("strides", &Tensor::strides,
                     "Strides in BYTES, following NumPy. DLPack uses elements; the "
                     "conversion happens only at the DLPack boundary.")
        .def_prop_ro("is_contiguous", &Tensor::is_contiguous)
        .def("defined", &Tensor::defined,
             "False for a default-constructed / cleared tensor (e.g. after zero_grad).")
        .def("__bool__", &Tensor::defined)
        .def("__len__",
             [](const Tensor& t) {
                 if (t.ndim() == 0) {
                     throw vkml::IndexError("len() of a 0-dimensional tensor");
                 }
                 return t.size(0);
             })
        .def("__repr__", &Tensor::str)
        .def("__str__", &Tensor::str);

    // evaluation
    tensor
        .def("realize",
             [](const Tensor& t) {
                 t.realize();
                 return t;
             })
        .def("item", &Tensor::item)
        .def("numpy", &tensor_to_numpy, "Copy to a new NumPy array.")
        .def(
            "__dlpack__", [](const Tensor& t, nb::kwargs) { return tensor_to_numpy(t); },
            "Export via DLPack. Note strides are reported in elements, not bytes.")
        .def("__dlpack_device__",
             [](const Tensor&) { return nb::make_tuple(nb::device::cpu::value, 0); });

    // views
    tensor
        .def(
            "reshape", [](const Tensor& t, std::vector<int64_t> d) { return t.reshape(d); },
            "shape"_a)
        .def(
            "view", [](const Tensor& t, std::vector<int64_t> d) { return t.reshape(d); }, "shape"_a)
        .def(
            "permute", [](const Tensor& t, std::vector<int> p) { return t.permute(p); }, "dims"_a)
        .def("transpose", &Tensor::transpose, "dim0"_a, "dim1"_a)
        .def_prop_ro("T", [](const Tensor& t) { return t.transpose(-2, -1); })
        .def("squeeze", &Tensor::squeeze, "dim"_a)
        .def("unsqueeze", &Tensor::unsqueeze, "dim"_a)
        .def(
            "broadcast_to",
            [](const Tensor& t, std::vector<int64_t> d) { return t.broadcast_to(d); }, "shape"_a)
        .def("contiguous", &Tensor::contiguous)
        .def("to", &Tensor::to, "dtype"_a)
        .def("astype", &Tensor::to, "dtype"_a)
        .def("__getitem__", &tensor_getitem, "key"_a);

    // autograd
    tensor
        .def_prop_rw("requires_grad", &Tensor::requires_grad,
                     [](Tensor& t, bool v) { t.set_requires_grad(v); })
        .def_prop_rw("grad", &Tensor::grad, [](Tensor& t, const Tensor& g) { t.set_grad(g); })
        .def(
            "backward", [](const Tensor& t) { vkml::backward(t); },
            "Accumulate gradients into every leaf that requires grad.")
        .def("assign_", &Tensor::assign_, "src"_a,
             "In-place overwrite. Used by optimizers; see the C++ docs for the "
             "mid-graph hazard.")
        .def(
            "detach", [](const Tensor& t) { return vkml::detach(t); },
            "A view of the same data that carries no gradient history.");

    // arithmetic
    tensor.def(
              "__add__", [](const Tensor& a, const Tensor& b) { return a + b; }, nb::is_operator())
        .def(
            "__add__", [](const Tensor& a, double s) { return a + s; }, nb::is_operator())
        .def(
            "__radd__", [](const Tensor& a, double s) { return a + s; }, nb::is_operator())
        .def(
            "__sub__", [](const Tensor& a, const Tensor& b) { return a - b; }, nb::is_operator())
        .def(
            "__sub__", [](const Tensor& a, double s) { return a - s; }, nb::is_operator())
        .def(
            "__rsub__", [](const Tensor& a, double s) { return vkml::add(vkml::neg(a), s); },
            nb::is_operator())
        .def(
            "__mul__", [](const Tensor& a, const Tensor& b) { return a * b; }, nb::is_operator())
        .def(
            "__mul__", [](const Tensor& a, double s) { return a * s; }, nb::is_operator())
        .def(
            "__rmul__", [](const Tensor& a, double s) { return a * s; }, nb::is_operator())
        .def(
            "__truediv__", [](const Tensor& a, const Tensor& b) { return a / b; },
            nb::is_operator())
        .def(
            "__truediv__", [](const Tensor& a, double s) { return a / s; }, nb::is_operator())
        .def(
            "__rtruediv__",
            [](const Tensor& a, double s) { return vkml::mul(vkml::reciprocal(a), s); },
            nb::is_operator())
        .def(
            "__pow__", [](const Tensor& a, double s) { return vkml::pow(a, s); }, nb::is_operator())
        .def(
            "__neg__", [](const Tensor& a) { return -a; }, nb::is_operator())
        .def(
            "__matmul__", [](const Tensor& a, const Tensor& b) { return vkml::matmul(a, b); },
            nb::is_operator())
        .def(
            "__lt__", [](const Tensor& a, const Tensor& b) { return vkml::less(a, b); },
            nb::is_operator())
        .def(
            "__gt__", [](const Tensor& a, const Tensor& b) { return vkml::greater(a, b); },
            nb::is_operator())
        .def(
            "__le__", [](const Tensor& a, const Tensor& b) { return vkml::less_equal(a, b); },
            nb::is_operator())
        .def(
            "__ge__", [](const Tensor& a, const Tensor& b) { return vkml::greater_equal(a, b); },
            nb::is_operator());

    // method forms of common ops, so `x.relu()` works like `vkml.relu(x)`
    tensor
        .def(
            "sum",
            [](const Tensor& t, nb::object ax, bool keep) {
                return vkml::sum(t, to_axes(ax, t.ndim()), keep);
            },
            "dim"_a = nb::none(), "keepdim"_a = false)
        .def(
            "mean",
            [](const Tensor& t, nb::object ax, bool keep) {
                return vkml::mean(t, to_axes(ax, t.ndim()), keep);
            },
            "dim"_a = nb::none(), "keepdim"_a = false)
        .def(
            "max",
            [](const Tensor& t, nb::object ax, bool keep) {
                return vkml::max(t, to_axes(ax, t.ndim()), keep);
            },
            "dim"_a = nb::none(), "keepdim"_a = false)
        .def(
            "min",
            [](const Tensor& t, nb::object ax, bool keep) {
                return vkml::min(t, to_axes(ax, t.ndim()), keep);
            },
            "dim"_a = nb::none(), "keepdim"_a = false)
        .def("relu", &vkml::relu)
        .def("exp", &vkml::exp)
        .def("log", &vkml::log)
        .def("sqrt", &vkml::sqrt)
        .def("tanh", &vkml::tanh)
        .def("sigmoid", &vkml::sigmoid)
        .def("gelu", &vkml::gelu)
        .def("silu", &vkml::silu)
        .def("abs", &vkml::abs)
        .def("softmax", &vkml::softmax, "dim"_a = -1)
        .def("log_softmax", &vkml::log_softmax, "dim"_a = -1)
        .def("matmul", &vkml::matmul, "other"_a);

    // -- construction -------------------------------------------------------
    m.def("from_numpy", &tensor_from_array, "array"_a, "dtype"_a = nb::none(),
          "device"_a = Device::cpu(), "Create a tensor by copying a C-contiguous CPU array.");

    m.def(
        "zeros",
        [](std::vector<int64_t> d, DType dt, Device dev) { return Tensor::zeros(d, dt, dev); },
        "shape"_a, "dtype"_a = DType::F32, "device"_a = Device::cpu());
    m.def(
        "ones",
        [](std::vector<int64_t> d, DType dt, Device dev) { return Tensor::ones(d, dt, dev); },
        "shape"_a, "dtype"_a = DType::F32, "device"_a = Device::cpu());
    m.def(
        "full",
        [](std::vector<int64_t> d, double v, DType dt, Device dev) {
            return Tensor::full(d, v, dt, dev);
        },
        "shape"_a, "value"_a, "dtype"_a = DType::F32, "device"_a = Device::cpu());
    // Forwarded to the C++ free functions rather than reimplemented here, so the
    // rule for which properties are inherited from the input lives in exactly
    // one place and the two surfaces cannot drift apart.
    m.def("zeros_like", &vkml::zeros_like, "input"_a);
    m.def("ones_like", &vkml::ones_like, "input"_a);
    m.def("full_like", &vkml::full_like, "input"_a, "value"_a);
    m.def("arange", &Tensor::arange, "start"_a, "stop"_a, "step"_a = 1.0, "dtype"_a = DType::F32,
          "device"_a = Device::cpu());

    // -- free functions -----------------------------------------------------
    m.def("add", nb::overload_cast<const Tensor&, const Tensor&>(&vkml::add));
    m.def("sub", nb::overload_cast<const Tensor&, const Tensor&>(&vkml::sub));
    m.def("mul", nb::overload_cast<const Tensor&, const Tensor&>(&vkml::mul));
    m.def("div", nb::overload_cast<const Tensor&, const Tensor&>(&vkml::div));
    m.def("pow", nb::overload_cast<const Tensor&, const Tensor&>(&vkml::pow));

    // Scalar right-hand sides, registered AFTER the tensor-tensor forms so an
    // exact Tensor match is always found first. The operators above already
    // work this way (`__mul__` is bound twice); these are the same overloads
    // reached by name, which is what a caller writing `V.mul(t, 2.0)` expects
    // and what every expression in ops.cpp already uses internally.
    m.def("add", nb::overload_cast<const Tensor&, double>(&vkml::add));
    m.def("sub", nb::overload_cast<const Tensor&, double>(&vkml::sub));
    m.def("mul", nb::overload_cast<const Tensor&, double>(&vkml::mul));
    m.def("div", nb::overload_cast<const Tensor&, double>(&vkml::div));
    m.def("pow", nb::overload_cast<const Tensor&, double>(&vkml::pow));
    m.def("maximum", &vkml::maximum);
    m.def("minimum", &vkml::minimum);

    m.def("equal", &vkml::equal);
    m.def("less", &vkml::less);
    m.def("greater", &vkml::greater);
    m.def("less_equal", &vkml::less_equal);
    m.def("greater_equal", &vkml::greater_equal);
    m.def("not_equal", &vkml::not_equal);

    m.def("neg", &vkml::neg);
    m.def("abs", &vkml::abs);
    m.def("sign", &vkml::sign);
    m.def("square", &vkml::square);
    m.def("sqrt", &vkml::sqrt);
    m.def("rsqrt", &vkml::rsqrt);
    m.def("reciprocal", &vkml::reciprocal);
    m.def("exp", &vkml::exp);
    m.def("log", &vkml::log);
    // erf has been in the C++ API since M0 but was never exposed here, so the
    // validation suite could not reach it at all.
    m.def("erf", &vkml::erf);
    m.def("erfc", &vkml::erfc);
    m.def("sin", &vkml::sin);
    m.def("cos", &vkml::cos);
    m.def("tanh", &vkml::tanh);
    m.def("sigmoid", &vkml::sigmoid);
    m.def("relu", &vkml::relu);
    m.def("gelu", &vkml::gelu);
    m.def("silu", &vkml::silu);
    m.def("clamp", &vkml::clamp, "input"_a, "min"_a, "max"_a);
    m.def("clamp_min", &vkml::clamp_min, "input"_a, "min"_a);
    m.def("clamp_max", &vkml::clamp_max, "input"_a, "max"_a);
    m.def("where", &vkml::where, "condition"_a, "x"_a, "y"_a);
    m.def(
        "cat", [](const std::vector<vkml::Tensor>& ts, int axis) { return vkml::cat(ts, axis); },
        "tensors"_a, "axis"_a = 0);
    m.def(
        "realize",
        [](const std::vector<vkml::Tensor>& ts) {
            std::vector<vkml::NodePtr> roots;
            roots.reserve(ts.size());
            for (const vkml::Tensor& t : ts) {
                if (!t.defined()) {
                    // Caught here rather than left to the executor: a null root
                    // is silently SKIPPED by topological_order, so the call
                    // would appear to succeed having evaluated nothing.
                    throw vkml::Error("realize() on an undefined tensor");
                }
                roots.push_back(t.node());
            }
            vkml::realize(roots);
        },
        "tensors"_a,
        "Evaluate several tensors together, as ONE unit of work.\n\n"
        "Equivalent in result to realizing each in turn, and cheaper: the whole\n"
        "set is scheduled once and reaches the backend as a single submission,\n"
        "rather than one submission per tensor. All must be on the same device.");
    nb::enum_<vkml::Reduction>(m, "Reduction")
        .value("mean", vkml::Reduction::Mean)
        .value("sum", vkml::Reduction::Sum)
        .value("none", vkml::Reduction::None);
    m.def("mse_loss", &vkml::mse_loss, "input"_a, "target"_a,
          "reduction"_a = vkml::Reduction::Mean);
    m.def("cross_entropy", &vkml::cross_entropy, "logits"_a, "target"_a,
          "reduction"_a = vkml::Reduction::Mean);
    m.def("binary_cross_entropy_with_logits", &vkml::binary_cross_entropy_with_logits, "logits"_a,
          "target"_a, "reduction"_a = vkml::Reduction::Mean);
    m.def("kl_div", &vkml::kl_div, "input"_a, "target"_a, "reduction"_a = vkml::Reduction::Mean,
          "log_target"_a = false);
    m.def("huber_loss", &vkml::huber_loss, "input"_a, "target"_a,
          "reduction"_a = vkml::Reduction::Mean, "delta"_a = 1.0);
    m.def(
        "rand",
        [](const std::vector<int64_t>& dims, uint64_t seed, uint64_t offset, vkml::Device device) {
            return vkml::rand(dims, seed, offset, device);
        },
        "shape"_a, "seed"_a, "offset"_a = 0, "device"_a = vkml::Device::cpu());
    m.def("dropout", &vkml::dropout, "input"_a, "p"_a, "seed"_a, "offset"_a = 0,
          "training"_a = true);
    m.def("batch_norm", &vkml::batch_norm, "input"_a, "mean"_a, "variance"_a,
          "weight"_a = vkml::Tensor{}, "bias"_a = vkml::Tensor{}, "eps"_a = 1e-5);
    m.def("max_pool2d", &vkml::max_pool2d, "input"_a, "kernel"_a,
          "stride"_a = std::array<int, 2>{0, 0}, "padding"_a = std::array<int, 2>{0, 0},
          "dilation"_a = std::array<int, 2>{1, 1});
    m.def("avg_pool2d", &vkml::avg_pool2d, "input"_a, "kernel"_a,
          "stride"_a = std::array<int, 2>{0, 0}, "padding"_a = std::array<int, 2>{0, 0});
    m.def("conv2d", &vkml::conv2d, "input"_a, "weight"_a, "bias"_a = vkml::Tensor{},
          "stride"_a = std::array<int, 2>{1, 1}, "padding"_a = std::array<int, 2>{0, 0},
          "dilation"_a = std::array<int, 2>{1, 1});
    m.def("im2col", &vkml::im2col, "input"_a, "kernel"_a, "stride"_a = std::array<int, 2>{1, 1},
          "padding"_a = std::array<int, 2>{0, 0}, "dilation"_a = std::array<int, 2>{1, 1});
    m.def("col2im", &vkml::col2im, "cols"_a, "image"_a, "kernel"_a,
          "stride"_a = std::array<int, 2>{1, 1}, "padding"_a = std::array<int, 2>{0, 0},
          "dilation"_a = std::array<int, 2>{1, 1});
    m.def("index_select", &vkml::index_select, "input"_a, "axis"_a, "index"_a);
    m.def("scatter_add", &vkml::scatter_add, "src"_a, "axis"_a, "index"_a, "dim_size"_a);
    m.def("layer_norm", &vkml::layer_norm, "input"_a, "normalized_axes"_a = 1, "eps"_a = 1e-5);
    m.def("rms_norm", &vkml::rms_norm, "input"_a, "normalized_axes"_a = 1, "eps"_a = 1e-5);
    m.def("masked_fill", &vkml::masked_fill, "input"_a, "mask"_a, "value"_a);
    m.def("triu", &vkml::triu, "input"_a, "diagonal"_a = 0);
    m.def("tril", &vkml::tril, "input"_a, "diagonal"_a = 0);

    m.def(
        "sum",
        [](const Tensor& t, nb::object ax, bool k) {
            return vkml::sum(t, to_axes(ax, t.ndim()), k);
        },
        "input"_a, "dim"_a = nb::none(), "keepdim"_a = false);
    m.def(
        "mean",
        [](const Tensor& t, nb::object ax, bool k) {
            return vkml::mean(t, to_axes(ax, t.ndim()), k);
        },
        "input"_a, "dim"_a = nb::none(), "keepdim"_a = false);
    m.def(
        "prod",
        [](const Tensor& t, nb::object ax, bool k) {
            return vkml::prod(t, to_axes(ax, t.ndim()), k);
        },
        "input"_a, "dim"_a = nb::none(), "keepdim"_a = false);
    m.def(
        "amax",
        [](const Tensor& t, nb::object ax, bool k) {
            return vkml::max(t, to_axes(ax, t.ndim()), k);
        },
        "input"_a, "dim"_a = nb::none(), "keepdim"_a = false);
    m.def(
        "amin",
        [](const Tensor& t, nb::object ax, bool k) {
            return vkml::min(t, to_axes(ax, t.ndim()), k);
        },
        "input"_a, "dim"_a = nb::none(), "keepdim"_a = false);
    m.def("argmax", &vkml::argmax, "input"_a, "dim"_a, "keepdim"_a = false);
    m.def("argmin", &vkml::argmin, "input"_a, "dim"_a, "keepdim"_a = false);

    m.def("softmax", &vkml::softmax, "input"_a, "dim"_a = -1);
    m.def("log_softmax", &vkml::log_softmax, "input"_a, "dim"_a = -1);
    m.def("matmul", &vkml::matmul, "a"_a, "b"_a);

    // -- vulkan -------------------------------------------------------------
    //
    // Compiled out entirely when the backend is not built, so the Python layer
    // can ask `has_vulkan()` and skip rather than fail on a CPU-only build.
#ifdef VKML_HAS_VULKAN
    m.attr("has_vulkan") = true;
    m.def("vulkan_available", &vkml::vulkan_available);
    m.def("vulkan_device_count", &vkml::vulkan_device_count);
    m.def("vulkan_device_names", &vkml::vulkan_device_names);
    m.def("vulkan_unavailable_reason", &vkml::vulkan_unavailable_reason);
    m.def(
        "vulkan_device_reports",
        [] {
            nb::list out;
            for (const vkml::DeviceReport& r : vkml::vulkan_device_reports()) {
                nb::dict d;
                d["name"] = r.name;
                d["driver_name"] = r.driver_name;
                d["device_type"] = r.device_type;
                d["api_version"] = r.api_version;
                d["driver_version"] = r.driver_version;
                d["vendor_id"] = r.vendor_id;
                d["device_id"] = r.device_id;

                // Derived here rather than stored, so the flag and the reason
                // cannot disagree.
                d["supported"] = r.missing_requirement.empty();
                d["missing_requirement"] = r.missing_requirement;

                d["buffer_device_address"] = r.buffer_device_address;
                d["scalar_block_layout"] = r.scalar_block_layout;
                d["timeline_semaphore"] = r.timeline_semaphore;
                d["synchronization2"] = r.synchronization2;
                d["subgroup_size_control"] = r.subgroup_size_control;
                d["shader_float16"] = r.shader_float16;
                d["shader_int8"] = r.shader_int8;
                d["shader_int16"] = r.shader_int16;
                d["storage_buffer_16bit"] = r.storage_buffer_16bit;
                d["global_float_atomic_add"] = r.global_float_atomic_add;
                d["shared_float_atomic_add"] = r.shared_float_atomic_add;
                d["cooperative_matrix"] = r.cooperative_matrix;

                d["subgroup_size"] = r.subgroup_size;
                d["min_subgroup_size"] = r.min_subgroup_size;
                d["max_subgroup_size"] = r.max_subgroup_size;
                d["max_workgroup_invocations"] = r.max_workgroup_invocations;
                d["max_workgroup_count_x"] = r.max_workgroup_count_x;
                d["shader_core_count"] = r.shader_core_count;
                d["max_shared_memory"] = r.max_shared_memory;
                d["max_push_constants"] = r.max_push_constants;
                d["max_allocation_size"] = r.max_allocation_size;
                d["device_local_bytes"] = r.device_local_bytes;
                d["host_visible_device_local_bytes"] = r.host_visible_device_local_bytes;
                d["timestamp_period"] = r.timestamp_period;
                out.append(d);
            }
            return out;
        },
        "Describe every visible Vulkan device without creating a logical device.");
    m.def(
        "init_vulkan",
        [](int index) {
            // Creating the backend also registers it, so device('vulkan:N')
            // resolves afterwards.
            vkml::Backend& b = vkml::vulkan_backend(index);
            return std::string(b.name());
        },
        "index"_a = 0, "Create and register the Vulkan backend for a device.");
    m.def(
        "vulkan_stats",
        [](int index) {
            auto& b = static_cast<vkml::VulkanBackend&>(vkml::vulkan_backend(index));
            const vkml::VulkanStats s = b.stats();
            nb::dict d;
            d["reserved_bytes"] = s.reserved_bytes;
            d["in_use_bytes"] = s.in_use_bytes;
            d["peak_in_use_bytes"] = s.peak_in_use_bytes;
            d["block_count"] = s.block_count;
            d["live_allocations"] = s.live_allocations;
            d["total_allocations"] = s.total_allocations;
            d["device_allocations"] = s.device_allocations;
            d["fragmentation"] = s.fragmentation;
            d["submissions"] = s.submissions;
            d["dispatches"] = s.dispatches;
            d["pipelines"] = s.pipelines;
            d["gpu_ms"] = s.gpu_ms;
            return d;
        },
        "index"_a = 0);
    m.def(
        "vulkan_timestamps_supported",
        [](int index) {
            return static_cast<vkml::VulkanBackend&>(vkml::vulkan_backend(index))
                .timestamps_supported();
        },
        "index"_a = 0,
        "False when the device's compute queue reports no timestamp bits, in which case "
        "every profile reads 0.000 ms regardless of the work done.");
    m.def(
        "vulkan_set_profiling",
        [](bool on, int index) {
            static_cast<vkml::VulkanBackend&>(vkml::vulkan_backend(index)).set_profiling(on);
        },
        "enabled"_a, "index"_a = 0);
    m.def(
        "vulkan_last_profile",
        [](int index) {
            return static_cast<vkml::VulkanBackend&>(vkml::vulkan_backend(index)).last_profile();
        },
        "index"_a = 0);
    m.def(
        "vulkan_set_subgroup_override",
        [](uint32_t size, int index) {
            static_cast<vkml::VulkanBackend&>(vkml::vulkan_backend(index))
                .set_subgroup_override(size);
        },
        "size"_a, "index"_a = 0);
    // The decision recorder. Deliberately NOT prefixed `vulkan_`: decisions are
    // published from any layer and a CPU-only build has them too, so naming it
    // after one backend would misdescribe what it observes.
    m.def(
        "configuration",
        [] {
            nb::list out;
            for (const vkml::ObservedSwitch& s : vkml::observed_environment()) {
                nb::dict e;
                e["name"] = s.name;
                e["value"] = s.value;
                e["set"] = s.set;
                out.append(e);
            }
            return out;
        },
        "Every environment switch this process has consulted, and what it saw. "
        "Observed at the point of reading, so it cannot disagree with the code.");
    m.def(
        "record_decisions",
        [](size_t capacity) { vkml::observe::start_recording(capacity); }, "capacity"_a = 256,
        "Begin recording decision facts into a bounded window, oldest dropped.");
    m.def(
        "stop_recording_decisions", [] { vkml::observe::stop_recording(); },
        "Stop recording and release the window.");
    m.def(
        "decisions",
        [] {
            nb::list out;
            for (const vkml::observe::RecordedDecision& d : vkml::observe::recorded()) {
                nb::dict e;
                e["site"] = d.site;
                e["op"] = d.op;
                e["chose"] = d.chose;
                e["instead_of"] = d.instead_of;
                e["because"] = d.because;
                e["required"] = d.required;
                e["available"] = d.available;
                e["seq"] = d.seq;
                out.append(e);
            }
            return out;
        },
        "What the engine recently chose, and instead of what. Oldest first.");
    m.def(
        "decisions_published", [] { return vkml::observe::published(); },
        "Decisions published since recording began, INCLUDING any the window "
        "evicted. Compare with len(decisions()) to detect a truncated history.");
    m.def(
        "vulkan_pipeline_stats",
        [](int index) {
            auto& b = static_cast<vkml::VulkanBackend&>(vkml::vulkan_backend(index));
            nb::list out;
            for (const vkml::PipelineStats& p : b.pipeline_stats()) {
                nb::dict d;
                d["name"] = p.name;
                d["available"] = p.available;
                d["vgprs"] = p.vgprs;
                d["sgprs"] = p.sgprs;
                d["spilled_vgprs"] = p.spilled_vgprs;
                d["spilled_sgprs"] = p.spilled_sgprs;
                d["scratch_bytes"] = p.scratch_bytes;
                d["lds_bytes"] = p.lds_bytes;
                d["waves_per_simd"] = p.waves_per_simd;
                d["instructions"] = p.instructions;
                d["code_bytes"] = p.code_bytes;
                out.append(d);
            }
            return out;
        },
        "index"_a = 0,
        "Compiler-reported resource usage per compiled pipeline. Empty entries have "
        "available=False when the driver does not report statistics.");
    m.def(
        "vulkan_capabilities",
        [](int index) {
            const vkml::DeviceCapabilities& c = vkml::vulkan_backend(index).capabilities();
            nb::dict d;
            d["fp16_compute"] = c.fp16_compute;
            d["subgroup_size"] = c.subgroup_size;
            d["can_pin_subgroup_size"] = c.can_pin_subgroup_size;
            d["min_subgroup_size"] = c.min_subgroup_size;
            d["max_subgroup_size"] = c.max_subgroup_size;
            d["global_float_atomics"] = c.global_float_atomics;
            d["shared_float_atomics"] = c.shared_float_atomics;
            d["cooperative_matrix"] = c.cooperative_matrix;
            d["max_workgroup_invocations"] = c.max_workgroup_invocations;
            d["max_shared_memory_bytes"] = c.max_shared_memory_bytes;
            d["total_memory_bytes"] = c.total_memory_bytes;
            return d;
        },
        "index"_a = 0);
#else
    m.attr("has_vulkan") = false;
    m.def("vulkan_available", [] { return false; });
    m.def("vulkan_device_count", [] { return 0; });
    // Present even here: a hardware report from a CPU-only build should say
    // "built without Vulkan" rather than fail with AttributeError.
    m.def("vulkan_device_reports", [] { return nb::list(); });
    // Same reason, and the one the README's own verification step calls:
    // `python -c "import vkml; print(vkml.vulkan_device_names())"` is what a user
    // runs straight after installing, including after the CPU-only install the
    // README documents. An empty list is the honest answer there; AttributeError
    // reads as a broken install (issue #9).
    m.def("vulkan_device_names", [] { return std::vector<std::string>{}; });
    m.def("vulkan_unavailable_reason",
          [] { return std::string("this build was compiled without the Vulkan backend"); });
#endif

    // -- autograd -----------------------------------------------------------
    m.def("backward", [](const Tensor& t) { vkml::backward(t); }, "tensor"_a);
    m.def(
        "backward", [](const Tensor& t, const Tensor& g) { vkml::backward(t, g); }, "tensor"_a,
        "gradient"_a);
    m.def("detach", &vkml::detach, "tensor"_a);
    m.def("set_grad_enabled", &vkml::set_grad_enabled, "enabled"_a);
    m.def("is_grad_enabled", &vkml::grad_enabled);
}
