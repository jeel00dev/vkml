#pragma once

#include "vk_device.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vkml::vk {

/// Per-kernel launch configuration.
///
/// Exists so the runtime never hardcodes a workgroup or subgroup decision.
/// llama.cpp demonstrates why this matters on exactly this GPU: its RDNA1 table
/// pins wave64 for soft_max/argmax/mul_mat_vec and wave32 everywhere else,
/// because the right width differs per kernel and the device's *default*
/// (64 here) is not the right answer for most of them.
///
/// A plain value type with sensible defaults. Autotuning is explicitly not
/// implemented -- but when it is, it becomes a search over these fields rather
/// than a redesign, because nothing above reads a hardcoded constant.
struct KernelConfig {
    /// Invocations per workgroup, x dimension. 256 is a reasonable default on
    /// AMD: it is 8 wave32s or 4 wave64s, enough to hide memory latency without
    /// starving occupancy.
    uint32_t workgroup_size = 256;

    /// 0 means "let the driver choose". Otherwise the pipeline is created with
    /// VkPipelineShaderStageRequiredSubgroupSizeCreateInfo, which needs
    /// subgroupSizeControl (present on the target GPU).
    uint32_t required_subgroup_size = 0;

    /// Declared shared-memory use, in bytes, validated against
    /// maxComputeSharedMemorySize before the pipeline is created.
    ///
    /// DECLARED, not derived: the footprint follows from specialisation
    /// constants, so the caller choosing them is the only thing that knows it.
    /// The consequence is that leaving this at 0 makes the check pass vacuously
    /// rather than fail loudly -- a caller that uses shared memory and forgets to
    /// say so hands the driver an over-budget pipeline, which is undefined
    /// behaviour and crashed one of them (issue #14). Set it wherever the shader
    /// declares a `shared` array.
    uint32_t shared_memory_bytes = 0;

    /// Elements each invocation loads at once. 1 for scalar; 4 lets a shader
    /// use vec4 loads, which matters for bandwidth-bound elementwise kernels.
    /// Recorded now, used when the vectorised variants land.
    uint32_t load_vector_width = 1;

    /// Specialisation constants, applied in order from constant_id 0.
    /// Preferred over #define variants: one SPIR-V module, many pipelines.
    std::vector<uint32_t> spec_constants;

    [[nodiscard]] std::string key() const;
};

/// Owns shader modules and compute pipelines, keyed by name plus configuration.
///
/// Pipelines are created lazily on first use and cached for the process
/// lifetime. A VkPipelineCache is kept alongside so the driver's SPIR-V -> ISA
/// compilation is paid once even across pipeline variants; it is not yet
/// serialised to disk, which would additionally amortise it across runs.
class PipelineCache {
public:
    explicit PipelineCache(Context& ctx);
    ~PipelineCache();

    PipelineCache(const PipelineCache&) = delete;
    PipelineCache& operator=(const PipelineCache&) = delete;
    PipelineCache(PipelineCache&&) = delete;
    PipelineCache& operator=(PipelineCache&&) = delete;

    /// What the driver reports about a compiled pipeline.
    ///
    /// This is measurement infrastructure, not diagnostics: register and
    /// scratch usage explain performance that timestamps alone cannot. Stage 5
    /// showed a kernel getting slower after an optimisation, and the cause was
    /// register allocation -- invisible to any timing measurement.
    struct Stats {
        bool available = false;
        uint32_t vgprs = 0;
        uint32_t sgprs = 0;
        uint32_t spilled_vgprs = 0;
        uint32_t spilled_sgprs = 0;
        uint64_t scratch_bytes = 0;
        uint64_t lds_bytes = 0;
        uint32_t max_waves = 0;
        uint64_t code_bytes = 0;
        uint64_t instructions = 0;
        /// Kept for diagnostics only; never consumed above this layer.
        std::vector<std::pair<std::string, int64_t>> raw;
    };

    struct Pipeline {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        uint32_t workgroup_size = 0;
        Stats stats;
    };

    /// Returns a pipeline for `name`, creating it on first request.
    ///
    /// @param spirv           embedded SPIR-V words
    /// @param spirv_bytes     size of the module in bytes
    /// @param push_constant_bytes  size of the shader's push constant block
    [[nodiscard]] const Pipeline& get(const std::string& name, const uint32_t* spirv,
                                      size_t spirv_bytes, uint32_t push_constant_bytes,
                                      const KernelConfig& config);

    [[nodiscard]] size_t pipeline_count() const noexcept { return pipelines_.size(); }

    /// Every compiled pipeline, keyed by name, with its statistics.
    [[nodiscard]] std::vector<std::pair<std::string, Stats>> all_stats() const;

private:
    [[nodiscard]] VkShaderModule module_for(const std::string& name, const uint32_t* spirv,
                                            size_t bytes);
    [[nodiscard]] Stats query_stats(VkPipeline pipeline) const;

    Context& ctx_;
    VkPipelineCache cache_ = VK_NULL_HANDLE;
    std::unordered_map<std::string, VkShaderModule> modules_;
    std::unordered_map<std::string, Pipeline> pipelines_;
};

}  // namespace vkml::vk
