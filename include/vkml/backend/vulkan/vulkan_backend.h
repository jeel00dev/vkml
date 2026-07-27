#pragma once

#include "vkml/backend/api/backend.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vkml {

/// Compiler-reported resource usage for one compiled pipeline.
///
/// Driver-independent by construction: vendor statistic names are normalised
/// inside the Vulkan backend and never surface above it. `available` is false
/// when the driver does not implement VK_KHR_pipeline_executable_properties,
/// which callers must treat as "unknown", not as zero.
///
/// These counters belong in benchmark metadata alongside timings because they
/// are far more reproducible: a 1024^3 GEMM timing varies by 13% run to run on
/// this machine, while VGPR count is exact and identical every time. The Stage 5
/// regression -- an optimisation that made the kernel slower -- was a register
/// allocation problem, invisible to timing and obvious in these numbers.
struct PipelineStats {
    std::string name;  ///< pipeline key, e.g. "gemm_reg:wg256_..."
    bool available = false;
    uint32_t vgprs = 0;
    uint32_t sgprs = 0;
    uint32_t spilled_vgprs = 0;
    uint32_t spilled_sgprs = 0;
    uint64_t scratch_bytes = 0;
    uint64_t lds_bytes = 0;
    uint32_t waves_per_simd = 0;
    uint64_t instructions = 0;
    uint64_t code_bytes = 0;
};

/// Diagnostics the Vulkan backend exposes for tests and reports.
struct VulkanStats {
    uint64_t reserved_bytes = 0;
    uint64_t in_use_bytes = 0;
    uint64_t peak_in_use_bytes = 0;
    uint32_t block_count = 0;
    uint32_t live_allocations = 0;
    uint64_t total_allocations = 0;
    uint64_t device_allocations = 0;
    double fragmentation = 0.0;
    uint64_t submissions = 0;
    uint64_t dispatches = 0;
    size_t pipelines = 0;

    [[nodiscard]] std::string describe() const;
};

/// GPU backend. Created on demand; never at static initialisation.
class VulkanBackend final : public Backend {
public:
    explicit VulkanBackend(int device_index, bool enable_validation);
    ~VulkanBackend() override;

    [[nodiscard]] std::string_view name() const noexcept override { return name_; }

    [[nodiscard]] Device device() const noexcept override { return device_; }

    [[nodiscard]] const DeviceCapabilities& capabilities() const noexcept override;
    [[nodiscard]] Allocator& allocator() override;
    [[nodiscard]] bool supports(const Node& node) const override;

    void compute(std::span<Node* const> nodes) override;
    void copy_from_host(Storage& dst, int64_t dst_offset, const void* src, size_t nbytes) override;
    void copy_to_host(void* dst, const Storage& src, int64_t src_offset, size_t nbytes) override;
    void synchronize() override;

    [[nodiscard]] VulkanStats stats() const;

    /// Resource usage of every pipeline compiled so far.
    ///
    /// The same API benchmark tooling and developers use; there is no separate
    /// debug path.
    [[nodiscard]] std::vector<PipelineStats> pipeline_stats() const;

    /// Enables per-dispatch timestamp queries. Zero cost when off.
    void set_profiling(bool enabled);

    /// GPU intervals from the most recent compute() call, in milliseconds.
    [[nodiscard]] std::vector<std::pair<std::string, double>> last_profile() const;

    /// Overrides the subgroup width kernels request, for measurement.
    ///
    /// 0 restores the default (driver-selected for elementwise, 64 for
    /// reductions). Exists so wave32 and wave64 can be compared in one process
    /// rather than by rebuilding -- the comparison is the point, and a
    /// hardware-specific constant should never be adopted without it.
    void set_subgroup_override(uint32_t size);

    /// Releases empty memory blocks back to the driver.
    void trim();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Device device_;
    std::string name_;
};

/// True when at least one Vulkan device is present.
[[nodiscard]] bool vulkan_available();

[[nodiscard]] int vulkan_device_count();

[[nodiscard]] std::vector<std::string> vulkan_device_names();

/// Returns the process-wide backend for `index`, creating it on first call and
/// registering it so `backend_for(Device::vulkan(index))` resolves.
///
/// Validation layers default on and can be disabled with VKML_VULKAN_VALIDATION=0.
/// Defaulting them ON is deliberate for M1: correctness and observability come
/// before performance, and a silent undefined-behaviour bug in a shader is far
/// more expensive than the validation overhead.
[[nodiscard]] Backend& vulkan_backend(int index = 0);

/// Destroys every Vulkan backend.
///
/// Backends are otherwise deliberately leaked at process exit -- see the note in
/// vulkan_backend.cpp. Call this only while the Vulkan loader is still alive
/// (i.e. not from a static destructor or atexit handler); it exists so tests can
/// assert the allocator ends with no live blocks.
void vulkan_shutdown();

}  // namespace vkml
