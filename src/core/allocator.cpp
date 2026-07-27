#include "vkml/core/allocator.h"

#include "vkml/util/assert.h"

#include <atomic>
#include <cstdlib>
#include <format>

namespace vkml {
namespace {

std::atomic<size_t>& cpu_live_bytes() noexcept {
    static std::atomic<size_t> v{0};
    return v;
}

}  // namespace

std::shared_ptr<Storage> CpuAllocator::allocate(size_t nbytes) {
    if (nbytes == 0) {
        // An empty tensor is legal (torch allows shape (0, 5)). Handing back a
        // real Storage with a null pointer keeps every downstream path free of
        // null-storage special cases.
        return std::make_shared<Storage>(nullptr, 0, Device::cpu(), Storage::Deleter{});
    }

    // std::aligned_alloc requires the size to be a multiple of the alignment.
    const size_t rounded = (nbytes + kCpuAlignment - 1) / kCpuAlignment * kCpuAlignment;

    void* p = std::aligned_alloc(kCpuAlignment, rounded);
    if (p == nullptr) {
        throw OutOfMemoryError(std::format("cpu allocator: failed to allocate {} bytes", rounded));
    }

    cpu_live_bytes().fetch_add(rounded, std::memory_order_relaxed);

    return std::make_shared<Storage>(p, nbytes, Device::cpu(), [rounded](void* data, size_t) {
        std::free(data);
        cpu_live_bytes().fetch_sub(rounded, std::memory_order_relaxed);
    });
}

size_t CpuAllocator::live_bytes() const noexcept {
    return cpu_live_bytes().load(std::memory_order_relaxed);
}

CpuAllocator& cpu_allocator() noexcept {
    static CpuAllocator instance;
    return instance;
}

}  // namespace vkml
