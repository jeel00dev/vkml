#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace vkml {

enum class DeviceKind : uint8_t {
    CPU = 0,
    Vulkan = 1,
};

[[nodiscard]] constexpr std::string_view device_kind_name(DeviceKind kind) noexcept {
    switch (kind) {
        case DeviceKind::CPU: return "cpu";
        case DeviceKind::Vulkan: return "vulkan";
    }
    return "?";
}

/// A (kind, index) pair identifying where a tensor's storage lives.
///
/// Value type, trivially copyable, cheap to pass around and to compare. The
/// index is meaningful only for multi-device kinds; CPU always uses index 0.
class Device {
public:
    constexpr Device() noexcept = default;

    constexpr Device(DeviceKind kind, int index) noexcept : kind_(kind), index_(index) {}

    [[nodiscard]] static constexpr Device cpu() noexcept { return {DeviceKind::CPU, 0}; }

    [[nodiscard]] static constexpr Device vulkan(int index = 0) noexcept {
        return {DeviceKind::Vulkan, index};
    }

    /// Parses "cpu", "vulkan", or "vulkan:1". Throws DeviceError on anything else.
    [[nodiscard]] static Device parse(std::string_view spec);

    [[nodiscard]] constexpr DeviceKind kind() const noexcept { return kind_; }

    [[nodiscard]] constexpr int index() const noexcept { return index_; }

    [[nodiscard]] constexpr bool is_cpu() const noexcept { return kind_ == DeviceKind::CPU; }

    [[nodiscard]] std::string str() const;

    friend constexpr bool operator==(const Device& a, const Device& b) noexcept {
        return a.kind_ == b.kind_ && a.index_ == b.index_;
    }

private:
    DeviceKind kind_ = DeviceKind::CPU;
    int index_ = 0;
};

}  // namespace vkml
