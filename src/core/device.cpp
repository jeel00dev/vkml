#include "vkml/core/device.h"

#include "vkml/util/assert.h"

#include <charconv>
#include <format>

namespace vkml {

std::string Device::str() const {
    if (kind_ == DeviceKind::CPU) {
        return "cpu";
    }
    return std::format("{}:{}", device_kind_name(kind_), index_);
}

Device Device::parse(std::string_view spec) {
    VKML_CHECK(!spec.empty(), DeviceError, "empty device string");

    std::string_view name = spec;
    int index = 0;

    if (const auto colon = spec.find(':'); colon != std::string_view::npos) {
        name = spec.substr(0, colon);
        const std::string_view digits = spec.substr(colon + 1);
        VKML_CHECK(!digits.empty(), DeviceError, "device '{}' has no index after ':'", spec);

        const auto* first = digits.data();
        const auto* last = digits.data() + digits.size();
        const auto res = std::from_chars(first, last, index);
        VKML_CHECK(res.ec == std::errc{} && res.ptr == last, DeviceError,
                   "device '{}' has a malformed index", spec);
        VKML_CHECK(index >= 0, DeviceError, "device index must be non-negative, got {}", index);
    }

    if (name == "cpu") {
        VKML_CHECK(index == 0, DeviceError, "cpu has no index, got '{}'", spec);
        return Device::cpu();
    }
    if (name == "vulkan") {
        return Device::vulkan(index);
    }

    throw DeviceError(std::format("unknown device '{}' (expected 'cpu' or 'vulkan[:N]')", spec));
}

}  // namespace vkml
