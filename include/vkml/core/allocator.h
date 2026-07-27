#pragma once

#include "vkml/core/device.h"
#include "vkml/core/storage.h"

#include <cstddef>
#include <memory>
#include <string_view>

namespace vkml {

/// Source of owned device memory.
///
/// Deliberately minimal -- allocate, identify, report a device. It models
/// "ask this thing for memory", not a general allocator framework. Alignment
/// policies, memory kinds, streams and async free lists are all properties of
/// a *concrete* allocator and can be added without touching this interface.
///
/// The same seam exists in ggml as `ggml_backend_buffer_type`, and for the same
/// reason: memory has to be requestable independently of who computes on it, so
/// that host-visible staging memory usable by a GPU can be obtained without
/// pretending it is ordinary device memory. That case is not hypothetical here
/// -- the target GPU exposes only 256 MiB of host-visible device-local memory
/// (docs/ARCHITECTURE.md §1.1), so every upload at M1 goes through a separate
/// staging allocator on the same device.
///
/// Allocators are long-lived and outlive the Storages they hand out. They are
/// owned by whoever owns the device: the process for CPU, the backend for GPUs.
///
/// See docs/adr/0002-allocator-abstraction.md.
class Allocator {
public:
    Allocator() = default;
    virtual ~Allocator() = default;

    Allocator(const Allocator&) = delete;
    Allocator& operator=(const Allocator&) = delete;
    Allocator(Allocator&&) = delete;
    Allocator& operator=(Allocator&&) = delete;

    /// Allocates at least `nbytes`. A zero-size request is legal and yields a
    /// Storage with a null pointer, so that empty tensors need no special case.
    /// Throws OutOfMemoryError on failure.
    [[nodiscard]] virtual std::shared_ptr<Storage> allocate(size_t nbytes) = 0;

    /// Short identifier, for logs and leak reports.
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;

    [[nodiscard]] virtual Device device() const noexcept = 0;

    /// Bytes currently handed out and not yet released by this allocator.
    [[nodiscard]] virtual size_t live_bytes() const noexcept = 0;
};

/// Host memory, 64-byte aligned (see kCpuAlignment).
class CpuAllocator final : public Allocator {
public:
    [[nodiscard]] std::shared_ptr<Storage> allocate(size_t nbytes) override;

    [[nodiscard]] std::string_view name() const noexcept override { return "cpu"; }

    [[nodiscard]] Device device() const noexcept override { return Device::cpu(); }

    [[nodiscard]] size_t live_bytes() const noexcept override;
};

/// The process-wide host allocator.
[[nodiscard]] CpuAllocator& cpu_allocator() noexcept;

}  // namespace vkml
