#pragma once

#include "vkml/backend/api/capabilities.h"
#include "vkml/core/allocator.h"
#include "vkml/graph/node.h"

#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace vkml {

/// One byte range to copy within a backend. See `copy_device_to_device`.
///
/// Raw pointers rather than references so a span of these is buildable; they
/// are borrowed for the duration of the call and never stored.
struct BufferCopy {
    Storage* dst = nullptr;
    int64_t dst_offset = 0;
    const Storage* src = nullptr;
    int64_t src_offset = 0;
    size_t nbytes = 0;
};

/// A device that can allocate memory and evaluate graph nodes.
///
/// Three responsibilities only: describe yourself, hand out memory, compute
/// nodes. Deliberately narrower than ggml's four-level
/// reg/device/backend/buffer_type split, which exists to support ~20 backends,
/// dynamic .so loading and multi-GPU -- none of which are goals here
/// (docs/ARCHITECTURE.md §3 Fork 3).
///
/// CONTRACT for compute()
/// ----------------------
/// The caller (dispatch/executor) guarantees, for every node passed in:
///   - every source is realised, i.e. has storage;
///   - the node itself has storage allocated and large enough;
///   - nodes appear in dependency order;
///   - view ops never appear -- the executor resolves them by aliasing, so a
///     backend never needs a kernel for reshape/permute/slice/broadcast.
///
/// The backend guarantees it fills each node's storage with the correct values.
/// Splitting allocation (caller) from computation (backend) is what lets the
/// M5 memory planner assign offsets without any backend knowing about it.
///
/// A whole span is passed rather than one node at a time so that a GPU backend
/// can record one command buffer for the batch. The CPU backend just loops.
class Backend {
public:
    Backend() = default;
    virtual ~Backend() = default;

    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;
    Backend(Backend&&) = delete;
    Backend& operator=(Backend&&) = delete;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;

    [[nodiscard]] virtual Device device() const noexcept = 0;

    [[nodiscard]] virtual const DeviceCapabilities& capabilities() const noexcept = 0;

    [[nodiscard]] virtual Allocator& allocator() = 0;

    /// Whether this backend can evaluate `node` as configured (op, dtype,
    /// rank, layout). Returning false is how the executor decides to fall back
    /// to the CPU, so it must be honest rather than optimistic.
    [[nodiscard]] virtual bool supports(const Node& node) const = 0;

    /// Evaluates `nodes` in order. See the contract above.
    virtual void compute(std::span<Node* const> nodes) = 0;

    /// Copies host bytes into device storage.
    virtual void copy_from_host(Storage& dst, int64_t dst_offset, const void* src,
                                size_t nbytes) = 0;

    /// Copies device storage out to host bytes.
    virtual void copy_to_host(void* dst, const Storage& src, int64_t src_offset, size_t nbytes) = 0;

    /// Copies several disjoint byte ranges between storages of THIS backend,
    /// as ONE unit of work.
    ///
    /// The regions must not overlap. Callers that cannot rule that out route
    /// through the host instead, which is always correct; requiring it here
    /// keeps the implementations to one primitive each rather than making
    /// every backend reimplement an overlap policy.
    ///
    /// **A span rather than a single copy, because a copy costs a submission.**
    /// Eight independent parameter assignments were eight submissions at a
    /// measured ~80 µs of host time each, against ~20 µs of GPU time for the
    /// kernels around them. Batching is not expressible above this interface:
    /// only the backend knows what a submission is. See
    /// docs/adr/0006-lazy-assign-and-submission-batching.md.
    ///
    /// Pure virtual on purpose. A default that staged through the host would
    /// be correct and quietly slow, and that is exactly the defect this method
    /// exists to remove -- vkML moved every assignment through host memory for
    /// months without anyone noticing. A new backend should have to answer
    /// this question rather than inherit a silent answer to it.
    virtual void copy_device_to_device(std::span<const BufferCopy> copies) = 0;

    /// One copy. Not virtual: a backend implements the span form and gets this
    /// for free, so the two cannot disagree about what a copy means.
    void copy_device_to_device(Storage& dst, int64_t dst_offset, const Storage& src,
                               int64_t src_offset, size_t nbytes) {
        const BufferCopy one{&dst, dst_offset, &src, src_offset, nbytes};
        copy_device_to_device(std::span<const BufferCopy>{&one, 1});
    }

    /// Blocks until all previously submitted work has completed. A no-op for
    /// synchronous backends.
    virtual void synchronize() {}
};

/// Returns the process-wide CPU backend. Always available.
[[nodiscard]] Backend& cpu_backend();

/// Returns the backend for `device`, or throws DeviceError if there is none.
/// Only the CPU exists until M1.
[[nodiscard]] Backend& backend_for(Device device);

/// Registers a backend for a device. Used by the Vulkan backend at M1; the CPU
/// backend registers itself. The registry does not take ownership -- backends
/// are process-lifetime objects.
void register_backend(Backend& backend);

/// Devices with a registered backend.
[[nodiscard]] std::vector<Device> available_devices();

}  // namespace vkml
