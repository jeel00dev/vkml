#pragma once

#include "vkml/backend/api/backend.h"

#include <span>

namespace vkml {

/// The reference backend.
///
/// Its job is to be obviously correct, deterministic and readable -- not fast.
/// Every operator must exist here before it exists anywhere else, and when a
/// Vulkan kernel and this disagree, this one is assumed right until proven
/// otherwise (docs/ARCHITECTURE.md §7.1).
///
/// Deterministic by construction: single-threaded, fixed reduction order,
/// no atomics. That means a golden-hash regression test can be exact-match
/// rather than tolerance-based.
class CpuBackend final : public Backend {
public:
    CpuBackend();

    [[nodiscard]] std::string_view name() const noexcept override { return "cpu"; }

    [[nodiscard]] Device device() const noexcept override { return Device::cpu(); }

    [[nodiscard]] const DeviceCapabilities& capabilities() const noexcept override { return caps_; }

    [[nodiscard]] Allocator& allocator() override { return cpu_allocator(); }

    [[nodiscard]] bool supports(const Node& node) const override;

    void compute(std::span<Node* const> nodes) override;

    void copy_from_host(Storage& dst, int64_t dst_offset, const void* src, size_t nbytes) override;

    void copy_to_host(void* dst, const Storage& src, int64_t src_offset, size_t nbytes) override;
    void copy_device_to_device(std::span<const BufferCopy> copies) override;
    using Backend::copy_device_to_device;  // keep the single-copy convenience visible

private:
    DeviceCapabilities caps_;
};

}  // namespace vkml
