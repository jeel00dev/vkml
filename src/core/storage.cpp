#include "vkml/core/storage.h"

#include "vkml/core/allocator.h"
#include "vkml/util/assert.h"
#include "vkml/util/log.h"

#include <atomic>
#include <format>

namespace vkml {
namespace {

std::atomic<size_t>& live_bytes_counter() noexcept {
    static std::atomic<size_t> v{0};
    return v;
}

std::atomic<size_t>& live_blocks_counter() noexcept {
    static std::atomic<size_t> v{0};
    return v;
}

std::atomic<size_t>& total_allocations_counter() noexcept {
    static std::atomic<size_t> v{0};
    return v;
}

}  // namespace

Storage::Storage(void* data, size_t nbytes, Device device, Deleter deleter)
    : data_(data), nbytes_(nbytes), device_(device), deleter_(std::move(deleter)) {
    VKML_ASSERT(data != nullptr || nbytes == 0, "non-empty storage with a null pointer");
    live_bytes_counter().fetch_add(nbytes, std::memory_order_relaxed);
    live_blocks_counter().fetch_add(1, std::memory_order_relaxed);
    total_allocations_counter().fetch_add(1, std::memory_order_relaxed);
}

Storage::~Storage() {
    // Destructors must not throw, so failures here are logged rather than
    // raised. A throwing deleter would terminate the process during unwinding.
    if (deleter_) {
        try {
            deleter_(data_, nbytes_);
        } catch (const std::exception& e) {
            VKML_LOG_ERROR("storage deleter threw during destruction: {}", e.what());
        } catch (...) {
            VKML_LOG_ERROR("storage deleter threw an unknown exception during destruction");
        }
    }
    live_bytes_counter().fetch_sub(nbytes_, std::memory_order_relaxed);
    live_blocks_counter().fetch_sub(1, std::memory_order_relaxed);
}

std::shared_ptr<Storage> make_cpu_storage(size_t nbytes) {
    return cpu_allocator().allocate(nbytes);
}

namespace storage_stats {

size_t live_bytes() noexcept {
    return live_bytes_counter().load(std::memory_order_relaxed);
}

size_t live_blocks() noexcept {
    return live_blocks_counter().load(std::memory_order_relaxed);
}

size_t total_allocations() noexcept {
    return total_allocations_counter().load(std::memory_order_relaxed);
}

}  // namespace storage_stats
}  // namespace vkml
