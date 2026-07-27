#include "vk_device.h"

#include "vkml/util/assert.h"
#include "vkml/util/log.h"

#include <algorithm>
#include <cstring>
#include <format>

namespace vkml::vk {
namespace {

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                              VkDebugUtilsMessageTypeFlagsEXT /*types*/,
                                              const VkDebugUtilsMessengerCallbackDataEXT* data,
                                              void* /*user*/) {
    // Routed through vkml's logger rather than stderr so the Python layer can
    // capture validation output alongside everything else.
    if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0) {
        VKML_LOG_ERROR("[validation] {}", data->pMessage);
    } else if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0) {
        VKML_LOG_WARN("[validation] {}", data->pMessage);
    } else {
        VKML_LOG_DEBUG("[validation] {}", data->pMessage);
    }
    return VK_FALSE;  // never abort the offending call
}

[[nodiscard]] bool has_layer(const char* name) {
    uint32_t count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());
    return std::any_of(layers.begin(), layers.end(), [name](const VkLayerProperties& l) {
        return std::strcmp(l.layerName, name) == 0;
    });
}

[[nodiscard]] bool has_instance_extension(const char* name) {
    uint32_t count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> exts(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, exts.data());
    return std::any_of(exts.begin(), exts.end(), [name](const VkExtensionProperties& e) {
        return std::strcmp(e.extensionName, name) == 0;
    });
}

[[nodiscard]] std::vector<VkExtensionProperties> device_extensions(VkPhysicalDevice dev) {
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> exts(count);
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &count, exts.data());
    return exts;
}

[[nodiscard]] bool has_device_extension(const std::vector<VkExtensionProperties>& exts,
                                        const char* name) {
    return std::any_of(exts.begin(), exts.end(), [name](const VkExtensionProperties& e) {
        return std::strcmp(e.extensionName, name) == 0;
    });
}

[[nodiscard]] std::vector<VkPhysicalDevice> physical_devices(VkInstance instance) {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());
    return devices;
}

/// Creates a bare instance for enumeration only.
[[nodiscard]] VkInstance make_probe_instance() {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "vkml-probe";
    app.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app;

    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&ci, nullptr, &instance) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return instance;
}

}  // namespace

const char* result_string(VkResult r) noexcept {
    switch (r) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_NOT_READY: return "VK_NOT_READY";
        case VK_TIMEOUT: return "VK_TIMEOUT";
        case VK_INCOMPLETE: return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
        case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
        case VK_ERROR_INVALID_EXTERNAL_HANDLE: return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
        case VK_ERROR_UNKNOWN: return "VK_ERROR_UNKNOWN";
        default: return "VK_ERROR_<unrecognised>";
    }
}

void check(VkResult r, const char* what) {
    if (r == VK_SUCCESS) {
        return;
    }
    // Device loss is unrecoverable and worth naming explicitly: it usually
    // means a shader hung or wrote out of bounds, and the message should say so
    // rather than leaving the reader to look up the enum.
    if (r == VK_ERROR_DEVICE_LOST) {
        throw DeviceError(std::format(
            "{} failed with VK_ERROR_DEVICE_LOST -- the GPU was reset, usually because a "
            "shader hung or wrote out of bounds. Re-run with validation enabled.",
            what));
    }
    if (r == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
        throw OutOfMemoryError(std::format("{} failed: out of device memory", what));
    }
    throw DeviceError(std::format("{} failed: {}", what, result_string(r)));
}

std::string DeviceInfo::describe() const {
    return std::format(
        "{} (vendor 0x{:04x} device 0x{:04x}, driver {}, Vulkan {}.{}.{})\n"
        "  VRAM ................ {:.2f} GiB device-local, {:.0f} MiB host-visible+device-local\n"
        "  subgroup ............ size {} (controllable {}..{})\n"
        "  workgroup ........... max {} invocations, {} KiB shared memory\n"
        "  compute units ....... {}\n"
        "  push constants ...... {} bytes\n"
        "  max allocation ...... {:.2f} GiB\n"
        "  buffer_device_address {}   scalar_block_layout {}   timeline_semaphore {}\n"
        "  fp16 {}   int8 {}   int16 {}   16-bit storage {}\n"
        "  global float atomicAdd {}   shared float atomicAdd {}   cooperative matrix {}",
        name, vendor_id, device_id, driver_name, VK_API_VERSION_MAJOR(api_version),
        VK_API_VERSION_MINOR(api_version), VK_API_VERSION_PATCH(api_version),
        static_cast<double>(device_local_bytes) / (1024.0 * 1024.0 * 1024.0),
        static_cast<double>(host_visible_device_local_bytes) / (1024.0 * 1024.0), subgroup_size,
        min_subgroup_size, max_subgroup_size, max_workgroup_invocations, max_shared_memory / 1024,
        shader_core_count, max_push_constants,
        static_cast<double>(max_allocation_size) / (1024.0 * 1024.0 * 1024.0),
        buffer_device_address, scalar_block_layout, timeline_semaphore, shader_float16, shader_int8,
        shader_int16, storage_buffer_16bit, global_float_atomic_add, shared_float_atomic_add,
        cooperative_matrix);
}

int enumerate_device_count() {
    VkInstance probe = make_probe_instance();
    if (probe == VK_NULL_HANDLE) {
        return 0;
    }
    const int n = static_cast<int>(physical_devices(probe).size());
    vkDestroyInstance(probe, nullptr);
    return n;
}

std::vector<std::string> enumerate_device_names() {
    std::vector<std::string> names;
    VkInstance probe = make_probe_instance();
    if (probe == VK_NULL_HANDLE) {
        return names;
    }
    for (VkPhysicalDevice dev : physical_devices(probe)) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(dev, &props);
        names.emplace_back(props.deviceName);
    }
    vkDestroyInstance(probe, nullptr);
    return names;
}

Context::Context(int device_index, bool enable_validation) {
    create_instance(enable_validation);
    select_physical_device(device_index);
    query_info();
    create_logical_device();

    VKML_LOG_INFO("vulkan device selected:\n{}", info_.describe());
}

Context::~Context() {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
        vkDestroyDevice(device_, nullptr);
    }
    if (debug_messenger_ != VK_NULL_HANDLE) {
        auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroy != nullptr) {
            destroy(instance_, debug_messenger_, nullptr);
        }
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
    }
}

void Context::create_instance(bool enable_validation) {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "vkml";
    app.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app.pEngineName = "vkml";
    // 1.3 is the floor: buffer_device_address, timeline semaphores,
    // synchronization2 and subgroup size control are all core there, which is
    // what lets the backend skip descriptor sets entirely.
    app.apiVersion = VK_API_VERSION_1_3;

    std::vector<const char*> layers;
    std::vector<const char*> extensions;

    const bool want_validation = enable_validation && has_layer("VK_LAYER_KHRONOS_validation") &&
                                 has_instance_extension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    if (enable_validation && !want_validation) {
        VKML_LOG_WARN("validation requested but VK_LAYER_KHRONOS_validation is unavailable");
    }
    if (want_validation) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app;
    ci.enabledLayerCount = static_cast<uint32_t>(layers.size());
    ci.ppEnabledLayerNames = layers.data();
    ci.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    ci.ppEnabledExtensionNames = extensions.data();

    // Attached via pNext so that instance creation and destruction are
    // themselves covered by validation.
    VkDebugUtilsMessengerCreateInfoEXT dbg{};
    dbg.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    dbg.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                          VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    dbg.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                      VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                      VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    dbg.pfnUserCallback = debug_callback;
    if (want_validation) {
        ci.pNext = &dbg;
    }

    check(vkCreateInstance(&ci, nullptr, &instance_), "vkCreateInstance");

    if (want_validation) {
        auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
        if (create != nullptr) {
            check(create(instance_, &dbg, nullptr, &debug_messenger_),
                  "vkCreateDebugUtilsMessengerEXT");
            VKML_LOG_INFO("vulkan validation layers enabled");
        }
    }
}

void Context::select_physical_device(int device_index) {
    const std::vector<VkPhysicalDevice> devices = physical_devices(instance_);
    VKML_CHECK(!devices.empty(), DeviceError, "no Vulkan devices found");
    VKML_CHECK(device_index >= 0 && device_index < static_cast<int>(devices.size()), DeviceError,
               "vulkan device index {} is out of range ({} device(s) present)", device_index,
               devices.size());
    physical_ = devices[static_cast<size_t>(device_index)];
}

void Context::query_info() {
    const std::vector<VkExtensionProperties> exts = device_extensions(physical_);

    // -- properties --------------------------------------------------------
    VkPhysicalDeviceSubgroupSizeControlProperties subgroup_size_props{};
    subgroup_size_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;

    VkPhysicalDeviceVulkan11Properties props11{};
    props11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES;
    props11.pNext = &subgroup_size_props;

    VkPhysicalDeviceVulkan12Properties props12{};
    props12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES;
    props12.pNext = &props11;

    VkPhysicalDeviceMaintenance3Properties maint3{};
    maint3.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES;
    maint3.pNext = &props12;

    // Compute-unit count. Not a core Vulkan property -- there is no portable
    // way to ask "how many independent schedulers does this GPU have" -- so it
    // comes from vendor extensions, exactly as llama.cpp does it
    // (ggml-vulkan.cpp:6139-6145). Chained only when the extension is present;
    // a physical-device property may be queried without enabling the extension
    // on the logical device, but the struct must not be chained if the driver
    // does not advertise it.
    //
    // Zero means "unknown", and every consumer must treat it as a reason to
    // decline an occupancy-based decision rather than to guess.
    const bool has_core_props2 =
        has_device_extension(exts, VK_AMD_SHADER_CORE_PROPERTIES_2_EXTENSION_NAME);
    const bool has_core_props =
        has_device_extension(exts, VK_AMD_SHADER_CORE_PROPERTIES_EXTENSION_NAME);

    VkPhysicalDeviceShaderCoreProperties2AMD core2{};
    core2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_2_AMD;
    VkPhysicalDeviceShaderCorePropertiesAMD core1{};
    core1.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_AMD;

    void* props_chain = &maint3;
    if (has_core_props2) {
        core2.pNext = props_chain;
        props_chain = &core2;
    }
    if (has_core_props) {
        core1.pNext = props_chain;
        props_chain = &core1;
    }

    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = props_chain;
    vkGetPhysicalDeviceProperties2(physical_, &props2);

    if (has_core_props2 && core2.activeComputeUnitCount != 0) {
        info_.shader_core_count = core2.activeComputeUnitCount;
    } else if (has_core_props) {
        // The older extension reports the topology rather than the active
        // count, so a harvested part reports its physical, not usable, CUs.
        info_.shader_core_count = core1.shaderEngineCount * core1.shaderArraysPerEngineCount *
                                  core1.computeUnitsPerShaderArray;
    }

    const VkPhysicalDeviceLimits& limits = props2.properties.limits;

    info_.name = props2.properties.deviceName;
    info_.api_version = props2.properties.apiVersion;
    info_.driver_version = props2.properties.driverVersion;
    info_.vendor_id = props2.properties.vendorID;
    info_.device_id = props2.properties.deviceID;
    info_.type = props2.properties.deviceType;
    info_.driver_name = props12.driverName;

    info_.subgroup_size = props11.subgroupSize;
    info_.min_subgroup_size = subgroup_size_props.minSubgroupSize;
    info_.max_subgroup_size = subgroup_size_props.maxSubgroupSize;
    info_.max_workgroup_invocations = limits.maxComputeWorkGroupInvocations;
    for (int i = 0; i < 3; ++i) {
        info_.max_workgroup_count[i] = limits.maxComputeWorkGroupCount[i];
        info_.max_workgroup_size[i] = limits.maxComputeWorkGroupSize[i];
    }
    info_.max_shared_memory = limits.maxComputeSharedMemorySize;
    info_.max_push_constants = limits.maxPushConstantsSize;
    info_.min_storage_buffer_offset_alignment = limits.minStorageBufferOffsetAlignment;
    info_.max_storage_buffer_range = limits.maxStorageBufferRange;
    info_.max_allocation_size = maint3.maxMemoryAllocationSize;
    info_.timestamp_period = limits.timestampPeriod;

    // -- features ----------------------------------------------------------
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float{};
    atomic_float.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;

    VkPhysicalDeviceVulkan13Features feats13{};
    feats13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    feats13.pNext = has_device_extension(exts, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME)
                        ? static_cast<void*>(&atomic_float)
                        : nullptr;

    VkPhysicalDeviceVulkan12Features feats12{};
    feats12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    feats12.pNext = &feats13;

    VkPhysicalDeviceVulkan11Features feats11{};
    feats11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    feats11.pNext = &feats12;

    VkPhysicalDeviceFeatures2 feats2{};
    feats2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    feats2.pNext = &feats11;
    vkGetPhysicalDeviceFeatures2(physical_, &feats2);

    info_.buffer_device_address = feats12.bufferDeviceAddress != VK_FALSE;
    info_.scalar_block_layout = feats12.scalarBlockLayout != VK_FALSE;
    info_.timeline_semaphore = feats12.timelineSemaphore != VK_FALSE;
    info_.shader_float16 = feats12.shaderFloat16 != VK_FALSE;
    info_.shader_int8 = feats12.shaderInt8 != VK_FALSE;
    info_.shader_int16 = feats2.features.shaderInt16 != VK_FALSE;
    info_.storage_buffer_16bit = feats11.storageBuffer16BitAccess != VK_FALSE;
    info_.synchronization2 = feats13.synchronization2 != VK_FALSE;
    info_.subgroup_size_control = feats13.subgroupSizeControl != VK_FALSE;

    info_.global_float_atomic_add = atomic_float.shaderBufferFloat32AtomicAdd != VK_FALSE;
    info_.shared_float_atomic_add = atomic_float.shaderSharedFloat32AtomicAdd != VK_FALSE;

    // Cooperative matrix is detected by extension rather than by feature bit,
    // because its feature struct only exists when the extension does.
    info_.cooperative_matrix = has_device_extension(exts, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    info_.pipeline_executable_properties =
        has_device_extension(exts, VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);

    // -- memory heaps ------------------------------------------------------
    VkPhysicalDeviceMemoryProperties mem{};
    vkGetPhysicalDeviceMemoryProperties(physical_, &mem);

    for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
        const VkMemoryType& type = mem.memoryTypes[i];
        const VkMemoryHeap& heap = mem.memoryHeaps[type.heapIndex];

        const bool device_local = (type.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
        const bool host_visible = (type.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;

        if (device_local && !host_visible) {
            info_.device_local_bytes = std::max(info_.device_local_bytes, heap.size);
        }
        if (device_local && host_visible) {
            // The BAR window. Small on a discrete GPU without resizable BAR,
            // which is what forces staged uploads.
            info_.host_visible_device_local_bytes =
                std::max(info_.host_visible_device_local_bytes, heap.size);
        }
    }
    if (info_.device_local_bytes == 0) {
        // Integrated GPU: everything is one shared heap.
        for (uint32_t i = 0; i < mem.memoryHeapCount; ++i) {
            if ((mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
                info_.device_local_bytes =
                    std::max(info_.device_local_bytes, mem.memoryHeaps[i].size);
            }
        }
    }
}

void Context::create_logical_device() {
    VKML_CHECK(info_.buffer_device_address, DeviceError,
               "device '{}' does not support bufferDeviceAddress, which the vkml Vulkan "
               "backend requires (docs/ARCHITECTURE.md 5.3)",
               info_.name);
    VKML_CHECK(info_.scalar_block_layout, DeviceError,
               "device '{}' does not support scalarBlockLayout", info_.name);
    VKML_CHECK(info_.timeline_semaphore, DeviceError,
               "device '{}' does not support timelineSemaphore", info_.name);

    // A compute-capable queue family. Prefer one WITHOUT graphics: on AMD that
    // is an async compute engine, which does not contend with any desktop
    // compositor work on the same queue.
    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &family_count, families.data());

    int chosen = -1;
    for (uint32_t i = 0; i < family_count; ++i) {
        const bool compute = (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
        const bool graphics = (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
        if (compute && !graphics) {
            chosen = static_cast<int>(i);
            break;
        }
    }
    if (chosen < 0) {
        for (uint32_t i = 0; i < family_count; ++i) {
            if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                chosen = static_cast<int>(i);
                break;
            }
        }
    }
    VKML_CHECK(chosen >= 0, DeviceError, "device '{}' has no compute queue family", info_.name);
    queue_family_ = static_cast<uint32_t>(chosen);

    const float priority = 1.0F;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = queue_family_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    std::vector<const char*> device_exts;
    const std::vector<VkExtensionProperties> available = device_extensions(physical_);
    if (has_device_extension(available, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME)) {
        device_exts.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    }
    if (info_.pipeline_executable_properties) {
        device_exts.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
    }

    // Enable exactly what is used. Requesting a feature the device lacks fails
    // device creation, so each is gated on the query above.
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float{};
    atomic_float.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
    atomic_float.shaderSharedFloat32AtomicAdd = info_.shared_float_atomic_add ? VK_TRUE : VK_FALSE;
    atomic_float.shaderBufferFloat32AtomicAdd = info_.global_float_atomic_add ? VK_TRUE : VK_FALSE;

    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR exec_props{};
    exec_props.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR;
    exec_props.pipelineExecutableInfo = info_.pipeline_executable_properties ? VK_TRUE : VK_FALSE;
    exec_props.pNext = device_exts.empty() ? nullptr : static_cast<void*>(&atomic_float);

    VkPhysicalDeviceVulkan13Features feats13{};
    feats13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    feats13.synchronization2 = info_.synchronization2 ? VK_TRUE : VK_FALSE;
    feats13.subgroupSizeControl = info_.subgroup_size_control ? VK_TRUE : VK_FALSE;
    feats13.computeFullSubgroups = info_.subgroup_size_control ? VK_TRUE : VK_FALSE;
    feats13.maintenance4 = VK_TRUE;
    feats13.pNext = info_.pipeline_executable_properties
                        ? static_cast<void*>(&exec_props)
                        : (device_exts.empty() ? nullptr : static_cast<void*>(&atomic_float));

    VkPhysicalDeviceVulkan12Features feats12{};
    feats12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    feats12.bufferDeviceAddress = VK_TRUE;
    feats12.scalarBlockLayout = VK_TRUE;
    feats12.timelineSemaphore = VK_TRUE;
    feats12.shaderFloat16 = info_.shader_float16 ? VK_TRUE : VK_FALSE;
    feats12.shaderInt8 = info_.shader_int8 ? VK_TRUE : VK_FALSE;
    feats12.storageBuffer8BitAccess = info_.shader_int8 ? VK_TRUE : VK_FALSE;
    feats12.uniformAndStorageBuffer8BitAccess = info_.shader_int8 ? VK_TRUE : VK_FALSE;
    feats12.hostQueryReset = VK_TRUE;
    feats12.pNext = &feats13;

    VkPhysicalDeviceVulkan11Features feats11{};
    feats11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    feats11.storageBuffer16BitAccess = info_.storage_buffer_16bit ? VK_TRUE : VK_FALSE;
    feats11.uniformAndStorageBuffer16BitAccess = info_.storage_buffer_16bit ? VK_TRUE : VK_FALSE;
    feats11.pNext = &feats12;

    VkPhysicalDeviceFeatures2 feats2{};
    feats2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    feats2.features.shaderInt64 = VK_TRUE;  // buffer device addresses are 64-bit
    feats2.features.shaderInt16 = info_.shader_int16 ? VK_TRUE : VK_FALSE;
    feats2.pNext = &feats11;

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &feats2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = static_cast<uint32_t>(device_exts.size());
    dci.ppEnabledExtensionNames = device_exts.data();

    check(vkCreateDevice(physical_, &dci, nullptr, &device_), "vkCreateDevice");
    vkGetDeviceQueue(device_, queue_family_, 0, &queue_);

    VKML_LOG_DEBUG("vulkan queue family {} ({} families available)", queue_family_, family_count);
}

DeviceCapabilities Context::capabilities() const {
    DeviceCapabilities c;
    c.fp16_compute = info_.shader_float16;
    c.fp64_compute = false;  // available but slow on consumer GPUs; not used
    c.bf16_compute = false;

    c.subgroup_ops = true;
    c.subgroup_size = info_.subgroup_size;
    c.min_subgroup_size = info_.min_subgroup_size;
    c.max_subgroup_size = info_.max_subgroup_size;

    c.global_float_atomics = info_.global_float_atomic_add;
    c.shared_float_atomics = info_.shared_float_atomic_add;
    c.cooperative_matrix = info_.cooperative_matrix;

    c.buffer_device_address = info_.buffer_device_address;
    c.unified_memory = info_.type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    c.host_accessible_buffers = false;  // device-local memory is not host-mappable
    c.min_buffer_alignment = std::max<size_t>(info_.min_storage_buffer_offset_alignment, 16);
    c.max_allocation_bytes = info_.max_allocation_size;
    c.total_memory_bytes = info_.device_local_bytes;

    c.max_workgroup_invocations = info_.max_workgroup_invocations;
    c.shader_core_count = info_.shader_core_count;
    c.max_shared_memory_bytes = info_.max_shared_memory;
    c.asynchronous = true;
    return c;
}

}  // namespace vkml::vk
