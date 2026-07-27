#pragma once

#include "vkml/core/device.h"

#include <cstddef>
#include <functional>
#include <memory>

namespace vkml {

/// Byte alignment of every CPU allocation.
///
/// 64 bytes is one cache line on every CPU this will run on, and is also the
/// alignment AVX-512 wants. The CPU backend is a correctness oracle and will
/// not be hand-vectorised, but aligning costs nothing here and removes a
/// variable if any part of it is ever vectorised by the compiler.
inline constexpr size_t kCpuAlignment = 64;

/// A refcounted block of device memory.
///
/// Storage deliberately does not know how to allocate. It holds a deleter
/// supplied by whoever created it, which is what lets `core` (layer 1) own the
/// type while `backend/vulkan` (layer 4) supplies Vulkan-specific freeing --
/// inverting a dependency that would otherwise be a layering violation.
///
/// Always held through shared_ptr: several Tensors may view one Storage, and a
/// view must keep the underlying block alive. This is the same reason ggml
/// tracks `view_src` on its tensors.
class Storage {
public:
    using Deleter = std::function<void(void* data, size_t nbytes)>;

    Storage(void* data, size_t nbytes, Device device, Deleter deleter);

    ~Storage();

    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;
    Storage(Storage&&) = delete;
    Storage& operator=(Storage&&) = delete;

    [[nodiscard]] void* data() noexcept { return data_; }

    [[nodiscard]] const void* data() const noexcept { return data_; }

    [[nodiscard]] size_t nbytes() const noexcept { return nbytes_; }

    [[nodiscard]] Device device() const noexcept { return device_; }

private:
    void* data_ = nullptr;
    size_t nbytes_ = 0;
    Device device_;
    Deleter deleter_;
};

/// Convenience wrapper over `cpu_allocator().allocate()`.
/// See vkml/core/allocator.h for the general path.
[[nodiscard]] std::shared_ptr<Storage> make_cpu_storage(size_t nbytes);

/// Live-allocation accounting, used by the leak tests required in
/// docs/ARCHITECTURE.md §7.4 tier 8.
///
/// This is process-global mutable state, which the project otherwise avoids.
/// It is justified here because leak detection is inherently global, it is
/// confined to three atomics, and it is diagnostics rather than behaviour --
/// nothing reads these counters to make a decision.
namespace storage_stats {

[[nodiscard]] size_t live_bytes() noexcept;

[[nodiscard]] size_t live_blocks() noexcept;

/// Monotonic count of allocations since process start.
[[nodiscard]] size_t total_allocations() noexcept;

}  // namespace storage_stats
}  // namespace vkml
