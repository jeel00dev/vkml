#pragma once

#include "vkml/backend/api/backend.h"

#include <cstdint>
#include <memory>
#include <span>
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

/// One enumerated Vulkan device, described without creating a logical device.
///
/// Deliberately free of Vulkan types. `vulkan.h` is an implementation detail of
/// backend/vulkan and must not reach a public header -- the same rule that
/// keeps DeviceCapabilities Vulkan-free -- so the device type and API version
/// are decoded to strings inside the backend, where those enums live.
///
/// Exists so a portability report can be produced on hardware the backend
/// cannot use. Every other route to capability data goes through backend
/// creation, which throws on precisely the device a report is wanted for.
struct DeviceReport {
    std::string name;
    std::string driver_name;
    std::string device_type;  ///< "discrete", "integrated", "virtual", "cpu", "other"
    std::string api_version;  ///< decoded, e.g. "1.3.275"

    /// Vendor-defined packing. Compare it between machines; do not decode it.
    uint32_t driver_version = 0;
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;

    /// Empty when the backend can use this device; otherwise the first
    /// requirement it fails, e.g. "bufferDeviceAddress".
    std::string missing_requirement;

    bool buffer_device_address = false;
    bool scalar_block_layout = false;
    bool timeline_semaphore = false;
    bool synchronization2 = false;
    bool subgroup_size_control = false;
    bool shader_float16 = false;
    bool shader_int8 = false;
    bool shader_int16 = false;
    bool storage_buffer_16bit = false;
    bool global_float_atomic_add = false;
    bool shared_float_atomic_add = false;
    bool cooperative_matrix = false;

    uint32_t subgroup_size = 0;
    uint32_t min_subgroup_size = 0;
    uint32_t max_subgroup_size = 0;
    uint32_t max_workgroup_invocations = 0;
    /// 0 means the driver did not report it, never "no compute units".
    uint32_t shader_core_count = 0;
    uint64_t max_shared_memory = 0;
    uint64_t max_push_constants = 0;
    uint64_t max_allocation_size = 0;
    uint64_t device_local_bytes = 0;
    uint64_t host_visible_device_local_bytes = 0;

    /// Nanoseconds per timestamp tick. Zero means GPU timings are unobtainable
    /// on this device however many ticks elapse, since the conversion
    /// multiplies by it -- worth reporting, because the symptom is a profile
    /// full of 0.000 ms that reads as an impossibly fast kernel.
    float timestamp_period = 0.0F;
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

    /// Whether this device's compute queue can produce meaningful timestamps.
    ///
    /// False means every profile will read 0.000 ms however much work ran, so a
    /// caller measuring something must check this rather than believe the
    /// number. The Vulkan spec permits a queue family to report no timestamp
    /// bits at all, and at least one driver does.
    [[nodiscard]] bool timestamps_supported() const;

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

/// Describes every visible device WITHOUT creating a logical device.
///
/// Empty when no Vulkan loader or no device is present, rather than throwing:
/// "this machine has no Vulkan" is an answer a report needs to be able to give.
[[nodiscard]] std::vector<DeviceReport> vulkan_device_reports();

/// Empty when a device is visible; otherwise why none is, in a form a person
/// reading a bug report can act on.
///
/// "No devices" has two quite different causes -- the loader cannot create an
/// instance at all, or it creates one that enumerates nothing -- and telling
/// them apart is most of the diagnosis on hardware nobody here can inspect.
[[nodiscard]] std::string vulkan_unavailable_reason();

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
