#pragma once

#include "vkml/backend/api/capabilities.h"

#include <cstdint>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace vkml::vk {

/// Everything the backend needs to know about the chosen GPU.
///
/// Populated once, from real queries. The values are then copied into a
/// backend-agnostic DeviceCapabilities so nothing above `backend/` ever sees a
/// Vulkan type -- which is the whole point of the capability abstraction.
struct DeviceInfo {
    std::string name;
    uint32_t api_version = 0;
    uint32_t driver_version = 0;
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;
    VkPhysicalDeviceType type = VK_PHYSICAL_DEVICE_TYPE_OTHER;
    std::string driver_name;

    // Features that gate whole implementation strategies.
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
    /// VK_KHR_pipeline_executable_properties: lets the driver report a
    /// compiled pipeline's register, scratch and LDS usage. Profiling
    /// infrastructure, not a rendering feature.
    bool pipeline_executable_properties = false;

    /// Independent compute units, from VK_AMD_shader_core_properties(2).
    /// 0 means the device did not report it; treat that as "decline any
    /// occupancy-based decision", never as a default guess.
    uint32_t shader_core_count = 0;

    uint32_t subgroup_size = 0;
    uint32_t min_subgroup_size = 0;
    uint32_t max_subgroup_size = 0;
    uint32_t max_workgroup_invocations = 0;
    uint32_t max_workgroup_count[3] = {0, 0, 0};
    uint32_t max_workgroup_size[3] = {0, 0, 0};
    size_t max_shared_memory = 0;
    size_t max_push_constants = 0;
    size_t min_storage_buffer_offset_alignment = 0;
    uint64_t max_allocation_size = 0;
    uint64_t max_storage_buffer_range = 0;
    float timestamp_period = 0.0F;

    /// Device-local heap size, i.e. VRAM.
    uint64_t device_local_bytes = 0;

    /// Host-visible AND device-local, i.e. the BAR window.
    ///
    /// Tracked separately because it is the number that decides the upload
    /// strategy. On the target GPU it is 256 MiB against 5.75 GiB of VRAM, so
    /// device memory cannot simply be mapped and every upload needs staging.
    uint64_t host_visible_device_local_bytes = 0;

    [[nodiscard]] std::string describe() const;
};

/// Owns the instance, the chosen physical device, the logical device and the
/// compute queue. One per process.
class Context {
public:
    /// @param device_index  which enumerated GPU to use
    /// @param enable_validation  install the Khronos validation layer if present
    Context(int device_index, bool enable_validation);
    ~Context();

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) = delete;
    Context& operator=(Context&&) = delete;

    [[nodiscard]] VkInstance instance() const noexcept { return instance_; }
    [[nodiscard]] VkPhysicalDevice physical() const noexcept { return physical_; }
    [[nodiscard]] VkDevice device() const noexcept { return device_; }
    [[nodiscard]] VkQueue queue() const noexcept { return queue_; }
    [[nodiscard]] uint32_t queue_family() const noexcept { return queue_family_; }
    [[nodiscard]] const DeviceInfo& info() const noexcept { return info_; }
    [[nodiscard]] bool validation_enabled() const noexcept { return debug_messenger_ != nullptr; }

    /// Fills in the backend-agnostic capability struct.
    [[nodiscard]] DeviceCapabilities capabilities() const;

private:
    void create_instance(bool enable_validation);
    void select_physical_device(int device_index);
    void query_info();
    void create_logical_device();

    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = 0;
    DeviceInfo info_;
};

/// Number of Vulkan devices visible, without creating a logical device.
[[nodiscard]] int enumerate_device_count();

/// Names of the visible devices, for diagnostics and device selection.
[[nodiscard]] std::vector<std::string> enumerate_device_names();

/// Turns a VkResult into a readable string.
[[nodiscard]] const char* result_string(VkResult r) noexcept;

/// Throws DeviceError with the call site and decoded result on failure.
void check(VkResult r, const char* what);

}  // namespace vkml::vk
