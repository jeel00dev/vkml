#include "vkml/backend/api/backend.h"

// Deliberately does NOT include backend/cpu: that would be a layering
// violation (layer 3 reaching into layer 4). `cpu_backend()` is declared in
// this layer's header and defined in the CPU backend, so the dependency points
// the right way and the linker joins them.
#include "vkml/util/assert.h"

#include <format>
#include <mutex>
#include <string>
#include <vector>

namespace vkml {
namespace {

struct Registry {
    std::mutex mutex;
    std::vector<Backend*> backends;
};

Registry& registry() {
    static Registry r;
    return r;
}

}  // namespace

std::string DeviceCapabilities::summary() const {
    return std::format(
        "fp16={} fp64={} subgroup={}({}..{}) global_float_atomics={} coopmat={} bda={} "
        "unified={} host_accessible={} align={} max_alloc={}",
        fp16_compute, fp64_compute, subgroup_ops, min_subgroup_size, max_subgroup_size,
        global_float_atomics, cooperative_matrix, buffer_device_address, unified_memory,
        host_accessible_buffers, min_buffer_alignment,
        max_allocation_bytes == SIZE_MAX
            ? std::string("unbounded")
            : std::format("{}MB", max_allocation_bytes / (1024 * 1024)));
}

void register_backend(Backend& backend) {
    Registry& r = registry();
    const std::lock_guard<std::mutex> lock(r.mutex);
    for (Backend* existing : r.backends) {
        if (existing->device() == backend.device()) {
            return;  // idempotent: re-registering the same device is harmless
        }
    }
    r.backends.push_back(&backend);
}

Backend& backend_for(Device device) {
    // The CPU backend registers itself on first use rather than at static
    // initialisation, which keeps it out of the static init order fiasco.
    Backend& cpu = cpu_backend();
    if (device == cpu.device()) {
        return cpu;
    }

    Registry& r = registry();
    const std::lock_guard<std::mutex> lock(r.mutex);
    for (Backend* b : r.backends) {
        if (b->device() == device) {
            return *b;
        }
    }
    // Deliberately says nothing Vulkan-specific. This is backend/api, LAYER 3;
    // backend/vulkan is layer 4, so the reason a Vulkan device failed to
    // register -- vulkan_unavailable_reason() -- is out of reach from here and
    // is added by the Python layer, where both are visible.
    throw DeviceError(std::format(
        "no backend registered for device '{}'. A device must be initialised before use: "
        "call vkml.init_vulkan(index) for a Vulkan device. If that fails, the device is "
        "unavailable -- vkml.best_device() picks a usable one and explains its choice.",
        device.str()));
}

std::vector<Device> available_devices() {
    std::vector<Device> out{cpu_backend().device()};

    Registry& r = registry();
    const std::lock_guard<std::mutex> lock(r.mutex);
    for (Backend* b : r.backends) {
        if (b->device() != Device::cpu()) {
            out.push_back(b->device());
        }
    }
    return out;
}

}  // namespace vkml
