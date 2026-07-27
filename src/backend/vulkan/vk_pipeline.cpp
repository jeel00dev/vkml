#include "vk_pipeline.h"

#include <cstdlib>

#include "vkml/util/assert.h"
#include "vkml/util/log.h"

#include <format>

namespace vkml::vk {

std::string KernelConfig::key() const {
    std::string k =
        std::format("wg{}_sg{}_lv{}", workgroup_size, required_subgroup_size, load_vector_width);
    for (const uint32_t c : spec_constants) {
        k += std::format("_{}", c);
    }
    return k;
}

PipelineCache::PipelineCache(Context& ctx) : ctx_(ctx) {
    VkPipelineCacheCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    check(vkCreatePipelineCache(ctx_.device(), &ci, nullptr, &cache_), "vkCreatePipelineCache");
}

PipelineCache::~PipelineCache() {
    for (auto& [name, p] : pipelines_) {
        vkDestroyPipeline(ctx_.device(), p.pipeline, nullptr);
        vkDestroyPipelineLayout(ctx_.device(), p.layout, nullptr);
    }
    for (auto& [name, m] : modules_) {
        vkDestroyShaderModule(ctx_.device(), m, nullptr);
    }
    if (cache_ != VK_NULL_HANDLE) {
        vkDestroyPipelineCache(ctx_.device(), cache_, nullptr);
    }
}

VkShaderModule PipelineCache::module_for(const std::string& name, const uint32_t* spirv,
                                         size_t bytes) {
    if (const auto it = modules_.find(name); it != modules_.end()) {
        return it->second;
    }

    VKML_ASSERT(bytes % 4 == 0, "SPIR-V module '{}' is {} bytes, not a multiple of 4", name, bytes);
    VKML_ASSERT(spirv[0] == 0x07230203, "SPIR-V module '{}' has a bad magic number 0x{:08x}", name,
                spirv[0]);

    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = bytes;
    ci.pCode = spirv;

    VkShaderModule module = VK_NULL_HANDLE;
    check(vkCreateShaderModule(ctx_.device(), &ci, nullptr, &module),
          std::format("vkCreateShaderModule({})", name).c_str());

    modules_.emplace(name, module);
    return module;
}

const PipelineCache::Pipeline& PipelineCache::get(const std::string& name, const uint32_t* spirv,
                                                  size_t spirv_bytes, uint32_t push_constant_bytes,
                                                  const KernelConfig& config) {
    const std::string key = name + ":" + config.key();
    if (const auto it = pipelines_.find(key); it != pipelines_.end()) {
        return it->second;
    }

    VKML_CHECK(push_constant_bytes <= ctx_.info().max_push_constants, DeviceError,
               "kernel '{}' needs {} bytes of push constants but the device allows {}", name,
               push_constant_bytes, ctx_.info().max_push_constants);
    VKML_CHECK(config.workgroup_size <= ctx_.info().max_workgroup_invocations, DeviceError,
               "kernel '{}' requests a workgroup of {} but the device allows {}", name,
               config.workgroup_size, ctx_.info().max_workgroup_invocations);
    VKML_CHECK(config.shared_memory_bytes <= ctx_.info().max_shared_memory, DeviceError,
               "kernel '{}' requests {} bytes of shared memory but the device allows {}", name,
               config.shared_memory_bytes, ctx_.info().max_shared_memory);

    // No descriptor set layouts at all: every buffer reaches the shader as a
    // device address in the push constant block (docs/ARCHITECTURE.md 5.3).
    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.offset = 0;
    range.size = push_constant_bytes;

    VkPipelineLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    lci.setLayoutCount = 0;
    lci.pSetLayouts = nullptr;
    lci.pushConstantRangeCount = push_constant_bytes > 0 ? 1 : 0;
    lci.pPushConstantRanges = push_constant_bytes > 0 ? &range : nullptr;

    Pipeline p;
    p.workgroup_size = config.workgroup_size;
    check(vkCreatePipelineLayout(ctx_.device(), &lci, nullptr, &p.layout),
          "vkCreatePipelineLayout");

    // The workgroup size is a specialisation constant (local_size_id), so one
    // SPIR-V module serves every width without recompilation. maintenance4,
    // enabled at device creation, is what permits this.
    std::vector<VkSpecializationMapEntry> entries;
    std::vector<uint32_t> values = config.spec_constants;
    for (uint32_t i = 0; i < values.size(); ++i) {
        entries.push_back(VkSpecializationMapEntry{i, i * 4U, sizeof(uint32_t)});
    }

    VkSpecializationInfo spec{};
    spec.mapEntryCount = static_cast<uint32_t>(entries.size());
    spec.pMapEntries = entries.data();
    spec.dataSize = values.size() * sizeof(uint32_t);
    spec.pData = values.data();

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup{};
    subgroup.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    subgroup.requiredSubgroupSize = config.required_subgroup_size;

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = module_for(name, spirv, spirv_bytes);
    stage.pName = "main";
    stage.pSpecializationInfo = entries.empty() ? nullptr : &spec;

    if (config.required_subgroup_size != 0) {
        VKML_CHECK(ctx_.info().subgroup_size_control, DeviceError,
                   "kernel '{}' pins subgroup size {} but the device lacks subgroupSizeControl",
                   name, config.required_subgroup_size);
        VKML_CHECK(config.required_subgroup_size >= ctx_.info().min_subgroup_size &&
                       config.required_subgroup_size <= ctx_.info().max_subgroup_size,
                   DeviceError,
                   "kernel '{}' pins subgroup size {}, outside the device range {}..{}", name,
                   config.required_subgroup_size, ctx_.info().min_subgroup_size,
                   ctx_.info().max_subgroup_size);
        stage.pNext = &subgroup;
    }

    VkComputePipelineCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pci.stage = stage;
    pci.layout = p.layout;
    // CAPTURE_STATISTICS must be requested at creation; the driver cannot
    // report statistics for a pipeline it did not instrument.
    //
    // It is also a MEASUREMENT-VALIDITY question, which is why it can be turned
    // off. The Vulkan spec permits an implementation to compile differently
    // when asked to retain statistics, and every performance conclusion in this
    // project has been drawn from pipelines carrying this flag. If the flag
    // changed codegen, the whole measurement chain would be biased and nothing
    // in the project would have detected it -- the statistics cannot report on
    // the pipeline that would have existed without them.
    //
    // VKML_VULKAN_NO_PIPELINE_STATS=1 builds pipelines without the flag, so the
    // two can be compared by timing. Measured on this driver over three paired
    // trials: <= 0.8 % slower with the flag at 1024^3 (consistent in direction,
    // so probably real) and indistinguishable at the other two shapes. That is
    // an order of magnitude below the smallest effect the project has ever
    // claimed, so the flag stays on (docs/MEASUREMENT-AUDIT.md 2).
    static const bool suppress_stats = [] {
        const char* v = std::getenv("VKML_VULKAN_NO_PIPELINE_STATS");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    if (ctx_.info().pipeline_executable_properties && !suppress_stats) {
        pci.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
    }

    const VkResult r =
        vkCreateComputePipelines(ctx_.device(), cache_, 1, &pci, nullptr, &p.pipeline);
    if (r != VK_SUCCESS) {
        vkDestroyPipelineLayout(ctx_.device(), p.layout, nullptr);
        check(r, std::format("vkCreateComputePipelines({})", key).c_str());
    }

    p.stats = query_stats(p.pipeline);

    VKML_LOG_DEBUG("vulkan pipeline '{}' created (workgroup {}, subgroup {})", key,
                   config.workgroup_size,
                   config.required_subgroup_size == 0 ? 0 : config.required_subgroup_size);
    if (p.stats.available) {
        VKML_LOG_DEBUG("  resources: vgpr={} sgpr={} spilled_vgpr={} scratch={}B lds={}B "
                       "max_waves={}",
                       p.stats.vgprs, p.stats.sgprs, p.stats.spilled_vgprs, p.stats.scratch_bytes,
                       p.stats.lds_bytes, p.stats.max_waves);
    }

    return pipelines_.emplace(key, p).first->second;
}

std::vector<std::pair<std::string, PipelineCache::Stats>> PipelineCache::all_stats() const {
    std::vector<std::pair<std::string, Stats>> out;
    out.reserve(pipelines_.size());
    for (const auto& [key, pipe] : pipelines_) {
        out.emplace_back(key, pipe.stats);
    }
    return out;
}

PipelineCache::Stats PipelineCache::query_stats(VkPipeline pipeline) const {
    Stats out;
    if (!ctx_.info().pipeline_executable_properties) {
        return out;
    }

    auto get_props = reinterpret_cast<PFN_vkGetPipelineExecutablePropertiesKHR>(
        vkGetDeviceProcAddr(ctx_.device(), "vkGetPipelineExecutablePropertiesKHR"));
    auto get_stats = reinterpret_cast<PFN_vkGetPipelineExecutableStatisticsKHR>(
        vkGetDeviceProcAddr(ctx_.device(), "vkGetPipelineExecutableStatisticsKHR"));
    if (get_props == nullptr || get_stats == nullptr) {
        return out;
    }

    VkPipelineInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR;
    info.pipeline = pipeline;

    uint32_t exec_count = 0;
    get_props(ctx_.device(), &info, &exec_count, nullptr);
    if (exec_count == 0) {
        return out;
    }

    // A compute pipeline has one executable; index 0 is it.
    VkPipelineExecutableInfoKHR exec{};
    exec.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR;
    exec.pipeline = pipeline;
    exec.executableIndex = 0;

    uint32_t stat_count = 0;
    get_stats(ctx_.device(), &exec, &stat_count, nullptr);
    if (stat_count == 0) {
        return out;
    }

    std::vector<VkPipelineExecutableStatisticKHR> stats(stat_count);
    for (auto& st : stats) {
        st.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
    }
    get_stats(ctx_.device(), &exec, &stat_count, stats.data());

    out.available = true;
    for (const auto& st : stats) {
        const std::string name = st.name;
        int64_t value = 0;
        switch (st.format) {
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
                value = static_cast<int64_t>(st.value.u64);
                break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR: value = st.value.i64; break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
                value = st.value.b32 != VK_FALSE ? 1 : 0;
                break;
            default: value = static_cast<int64_t>(st.value.f64); break;
        }
        out.raw.emplace_back(name, value);

        // Statistic names are driver-defined, so match loosely rather than
        // assuming RADV's exact spelling.
        const auto has = [&name](const char* needle) {
            return name.find(needle) != std::string::npos;
        };
        const auto uv = static_cast<uint64_t>(value);
        if (has("SGPRs") && has("Spilled")) {
            out.spilled_sgprs = static_cast<uint32_t>(uv);
        } else if (has("VGPRs") && has("Spilled")) {
            out.spilled_vgprs = static_cast<uint32_t>(uv);
        } else if (has("SGPRs")) {
            out.sgprs = static_cast<uint32_t>(uv);
        } else if (has("VGPRs") && !has("Private")) {
            out.vgprs = static_cast<uint32_t>(uv);
        } else if (has("Scratch") || has("scratch") || has("Private")) {
            out.scratch_bytes = uv;
        } else if (has("LDS") || has("Workgroup Memory")) {
            out.lds_bytes = uv;
        } else if (has("Waves") || has("Subgroups per SIMD")) {
            // RADV names this "Subgroups per SIMD"; other drivers say "Waves".
            // Matching both is why normalisation lives here rather than at the
            // call sites.
            out.max_waves = static_cast<uint32_t>(uv);
        } else if (has("Code size")) {
            out.code_bytes = uv;
        } else if (has("Instructions")) {
            out.instructions = uv;
        }
    }
    return out;
}

}  // namespace vkml::vk
