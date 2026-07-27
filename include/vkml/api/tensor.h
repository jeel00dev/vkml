#pragma once

#include "vkml/core/device.h"
#include "vkml/core/dtype.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace vkml {

/// Forward declaration only.
///
/// Per guardrail 1 in docs/adr/0001-graph-ownership-and-ir.md, the graph node
/// type must not be visible through the public API. Callers and the binding
/// layer see an opaque handle, so the internal representation can change
/// without breaking either. Tensor's special members are therefore declared
/// here and defined in tensor.cpp, where Node is complete.
struct Node;

/// A handle to a value in the computation graph.
///
/// Cheap to copy -- two Tensors sharing a node are two names for one value, as
/// in PyTorch. A Tensor is not necessarily computed yet: operations build graph
/// nodes and evaluation is deferred until something observes the data
/// (`to_host`, `item`, `backward`) or `realize()` is called explicitly. Set
/// eager mode to collapse that distinction while debugging.
///
/// LAYOUT: row-major. `shape()[0]` is the outermost axis, matching NumPy,
/// PyTorch and DLPack. Strides are reported in bytes by `strides()`, as NumPy
/// does; the DLPack bridge converts to element strides at the boundary and
/// nowhere else.
class Tensor {
public:
    Tensor();
    ~Tensor();

    Tensor(const Tensor&);
    Tensor& operator=(const Tensor&);
    Tensor(Tensor&&) noexcept;
    Tensor& operator=(Tensor&&) noexcept;

    /// Wraps an existing graph node. Internal; not part of the stable surface.
    explicit Tensor(std::shared_ptr<Node> node);

    // -- construction -------------------------------------------------------

    [[nodiscard]] static Tensor full(std::span<const int64_t> dims, double value,
                                     DType dtype = DType::F32, Device device = Device::cpu());

    [[nodiscard]] static Tensor zeros(std::span<const int64_t> dims, DType dtype = DType::F32,
                                      Device device = Device::cpu());

    [[nodiscard]] static Tensor ones(std::span<const int64_t> dims, DType dtype = DType::F32,
                                     Device device = Device::cpu());

    [[nodiscard]] static Tensor arange(double start, double stop, double step = 1.0,
                                       DType dtype = DType::F32, Device device = Device::cpu());

    /// Copies `nbytes` of host data into a new tensor.
    [[nodiscard]] static Tensor from_host(const void* data, std::span<const int64_t> dims,
                                          DType dtype = DType::F32, Device device = Device::cpu());

    // Braced-list overloads.
    //
    // std::span deliberately does not bind to `{2, 3}`, which would make every
    // call site write out a named array. These forwarders exist purely so the
    // C++ API reads the way the Python one will. They are inline and add no
    // indirection.

    [[nodiscard]] static Tensor full(std::initializer_list<int64_t> dims, double value,
                                     DType dtype = DType::F32, Device device = Device::cpu()) {
        return full(std::span<const int64_t>{dims.begin(), dims.size()}, value, dtype, device);
    }

    [[nodiscard]] static Tensor zeros(std::initializer_list<int64_t> dims, DType dtype = DType::F32,
                                      Device device = Device::cpu()) {
        return zeros(std::span<const int64_t>{dims.begin(), dims.size()}, dtype, device);
    }

    [[nodiscard]] static Tensor ones(std::initializer_list<int64_t> dims, DType dtype = DType::F32,
                                     Device device = Device::cpu()) {
        return ones(std::span<const int64_t>{dims.begin(), dims.size()}, dtype, device);
    }

    [[nodiscard]] static Tensor from_host(const void* data, std::initializer_list<int64_t> dims,
                                          DType dtype = DType::F32, Device device = Device::cpu()) {
        return from_host(data, std::span<const int64_t>{dims.begin(), dims.size()}, dtype, device);
    }

    // -- introspection ------------------------------------------------------

    [[nodiscard]] bool defined() const noexcept { return node_ != nullptr; }

    [[nodiscard]] std::vector<int64_t> shape() const;

    [[nodiscard]] int ndim() const;

    [[nodiscard]] int64_t numel() const;

    [[nodiscard]] int64_t size(int axis) const;

    /// Strides in BYTES, following NumPy.
    [[nodiscard]] std::vector<int64_t> strides() const;

    [[nodiscard]] DType dtype() const;

    [[nodiscard]] Device device() const;

    [[nodiscard]] bool is_contiguous() const;

    [[nodiscard]] std::string str() const;

    // -- evaluation ---------------------------------------------------------

    /// Forces evaluation. Idempotent.
    const Tensor& realize() const;

    /// Copies the elements out to host memory in row-major order, realising
    /// first if needed. `dst` must have room for numel() * itemsize bytes.
    void to_host(void* dst) const;

    /// Value of a single-element tensor.
    [[nodiscard]] float item() const;

    // -- views (no copy) ----------------------------------------------------

    [[nodiscard]] Tensor reshape(std::span<const int64_t> dims) const;
    [[nodiscard]] Tensor permute(std::span<const int> perm) const;
    [[nodiscard]] Tensor transpose(int a, int b) const;
    [[nodiscard]] Tensor squeeze(int axis) const;
    [[nodiscard]] Tensor unsqueeze(int axis) const;
    [[nodiscard]] Tensor slice(int axis, int64_t start, int64_t stop, int64_t step = 1) const;
    [[nodiscard]] Tensor broadcast_to(std::span<const int64_t> dims) const;

    [[nodiscard]] Tensor reshape(std::initializer_list<int64_t> dims) const {
        return reshape(std::span<const int64_t>{dims.begin(), dims.size()});
    }

    [[nodiscard]] Tensor permute(std::initializer_list<int> perm) const {
        return permute(std::span<const int>{perm.begin(), perm.size()});
    }

    [[nodiscard]] Tensor broadcast_to(std::initializer_list<int64_t> dims) const {
        return broadcast_to(std::span<const int64_t>{dims.begin(), dims.size()});
    }

    // -- copies -------------------------------------------------------------

    /// Materialises a contiguous copy. Returns *this when already contiguous.
    [[nodiscard]] Tensor contiguous() const;

    [[nodiscard]] Tensor to(DType dtype) const;

    /// Overwrites this tensor's storage with `src`'s values, in place.
    ///
    /// The one deliberate escape from the otherwise functional graph, and it
    /// exists for exactly one reason: optimizers must update parameters that
    /// modules already hold references to. Rebinding a new Tensor would leave
    /// every Module still pointing at the old one, so PyTorch mutates in place
    /// and so does this.
    ///
    /// HAZARD: any already-computed node that read this tensor keeps its old
    /// result, while any node computed afterwards sees the new values. That is
    /// harmless in the intended use -- the training graph is rebuilt each step
    /// -- but it means assign_ must not be used mid-graph.
    ///
    /// Requires matching shape, dtype and device, and a contiguous destination.
    void assign_(const Tensor& src);

    // -- autograd -----------------------------------------------------------

    [[nodiscard]] bool requires_grad() const;

    /// Marks a leaf as trainable. Throws for non-float dtypes and for
    /// non-leaves, matching PyTorch.
    void set_requires_grad(bool value);

    /// Accumulated gradient, or an undefined Tensor if there is none.
    [[nodiscard]] Tensor grad() const;

    void set_grad(const Tensor& g);

    /// Internal accessor for layers above the API boundary.
    [[nodiscard]] const std::shared_ptr<Node>& node() const noexcept { return node_; }

private:
    std::shared_ptr<Node> node_;
};

}  // namespace vkml
