#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace vkml {

/// What a backend can actually do.
///
/// The point of this struct is to keep backend-dependent behaviour from
/// scattering into `if (device.is_cpu())` checks across the runtime. Higher
/// layers ask "does this backend have global float atomics?", never "is this
/// Vulkan?". When a second GPU backend appears, nothing above `backend/` needs
/// to learn about it.
///
/// It is a plain value type, filled in once at backend construction. No virtual
/// queries, no per-call cost.
///
/// Every field here corresponds to a decision recorded in
/// docs/ARCHITECTURE.md §1.1, measured on the target GPU. The defaults describe
/// a minimal backend; each backend overwrites what it actually supports.
struct DeviceCapabilities {
    // -- arithmetic --------------------------------------------------------

    /// fp16 *arithmetic*, not merely fp16 storage. Storage/conversion is always
    /// available via vkml::Half; this flag means kernels may compute in fp16.
    bool fp16_compute = false;

    /// fp64 arithmetic. Available on the CPU, and on the target GPU only at a
    /// heavy rate penalty, so kernels should not assume it.
    bool fp64_compute = false;

    /// bf16 arithmetic. Measured absent on the target GPU
    /// (docs/ARCHITECTURE.md §1.1), so it is off by default everywhere.
    bool bf16_compute = false;

    // -- parallel execution primitives -------------------------------------

    /// Subgroup/wave collective operations (shuffle, ballot, reductions).
    bool subgroup_ops = false;

    /// Native subgroup width, 0 when not applicable. 32 on the target GPU.
    uint32_t subgroup_size = 0;

    /// Whether a compute pipeline may pin its subgroup width at all. The range
    /// below is only selectable when this is true.
    ///
    /// Two separate things on Vulkan, and conflating them is a bug: the device
    /// may expose VK_EXT_subgroup_size_control yet name no shader stage in
    /// requiredSubgroupSizeStages, in which case the range is reported but no
    /// width is pinnable. Observed on RADV RENOIR, which reports 64..64 and an
    /// empty stage mask.
    bool can_pin_subgroup_size = false;

    /// Minimum and maximum selectable subgroup width, when
    /// `can_pin_subgroup_size`. RDNA1 wants wave64 for reductions and wave32
    /// elsewhere, so this is load-bearing.
    uint32_t min_subgroup_size = 0;
    uint32_t max_subgroup_size = 0;

    /// atomicAdd on floats in *global* memory.
    ///
    /// Measured FALSE on the target GPU. This is the single most consequential
    /// capability in the struct: it forces every cross-workgroup gradient
    /// accumulation (embedding backward, scatter-add, conv weight gradients)
    /// onto a deterministic two-pass reduction instead of atomics
    /// (docs/ARCHITECTURE.md §3 Fork 5).
    bool global_float_atomics = false;

    /// atomicAdd on floats in workgroup-shared memory. TRUE on the target GPU,
    /// so intra-workgroup accumulation is still cheap.
    bool shared_float_atomics = false;

    /// Cooperative-matrix / tensor-core instructions. Measured absent on the
    /// target GPU, which is why GEMM must be hand-tiled.
    bool cooperative_matrix = false;

    // -- memory ------------------------------------------------------------

    /// 64-bit buffer addresses usable directly in kernels. Available on the
    /// target GPU, and the reason the Vulkan backend can skip descriptor sets
    /// entirely (docs/ARCHITECTURE.md §5.3).
    bool buffer_device_address = false;

    /// Host and device share one address space, so no staging copy is needed.
    /// True for CPU; false for this discrete GPU.
    bool unified_memory = false;

    /// Whether host code may dereference this backend's buffers directly.
    /// True for CPU. False for Vulkan device-local memory, which is why the
    /// CPU-fallback path has to copy rather than alias.
    bool host_accessible_buffers = false;

    /// Required alignment, in bytes, for a buffer used by a kernel.
    size_t min_buffer_alignment = 1;

    /// Largest single allocation. 4 GiB - 4 B on the target GPU.
    size_t max_allocation_bytes = SIZE_MAX;

    /// Total device memory, 0 when not meaningfully bounded.
    size_t total_memory_bytes = 0;

    // -- execution ---------------------------------------------------------

    /// Largest number of invocations in one workgroup, 0 when not applicable.
    uint32_t max_workgroup_invocations = 0;
    /// Independent compute units; 0 when the backend cannot determine it.
    uint32_t shader_core_count = 0;

    /// Workgroup-shared scratch, in bytes. 65536 on the target GPU.
    size_t max_shared_memory_bytes = 0;

    /// True when work is submitted asynchronously and needs an explicit
    /// synchronize() before results are readable from the host.
    bool asynchronous = false;

    [[nodiscard]] std::string summary() const;
};

}  // namespace vkml
