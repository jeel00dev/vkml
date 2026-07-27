#include "vkml/graph/grad_mode.h"

#include <atomic>

namespace vkml {
namespace {

std::atomic<bool>& flag() noexcept {
    static std::atomic<bool> enabled{true};
    return enabled;
}

}  // namespace

void set_grad_enabled(bool enabled) noexcept { flag().store(enabled, std::memory_order_relaxed); }

bool grad_enabled() noexcept { return flag().load(std::memory_order_relaxed); }

}  // namespace vkml
