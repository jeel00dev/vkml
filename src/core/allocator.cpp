#include "vkml/core/allocator.h"

#include "vkml/util/assert.h"

#include <atomic>
#include <cstdlib>
#include <format>

#ifdef _MSC_VER
#    include <malloc.h>
#endif

namespace vkml {
namespace {

std::atomic<size_t>& cpu_live_bytes() noexcept {
    static std::atomic<size_t> v{0};
    return v;
}

// std::aligned_alloc is C11, adopted into C++17 -- and absent on MSVC, whose
// CRT deliberately does not provide it because its free() cannot release such a
// pointer. _aligned_malloc is the supported spelling there, and it MUST be
// released by _aligned_free rather than free().
//
// So the two are defined as a pair, next to each other: the failure mode if
// they ever drift apart is a heap corruption that only appears on one platform,
// which is precisely the kind of bug this project cannot debug remotely.

void* aligned_allocate(size_t alignment, size_t nbytes) noexcept {
#ifdef _MSC_VER
    // Note the reversed argument order against std::aligned_alloc.
    return _aligned_malloc(nbytes, alignment);
#else
    return std::aligned_alloc(alignment, nbytes);
#endif
}

void aligned_release(void* p) noexcept {
#ifdef _MSC_VER
    _aligned_free(p);
#else
    std::free(p);
#endif
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

    void* p = aligned_allocate(kCpuAlignment, rounded);
    if (p == nullptr) {
        throw OutOfMemoryError(std::format("cpu allocator: failed to allocate {} bytes", rounded));
    }

    cpu_live_bytes().fetch_add(rounded, std::memory_order_relaxed);

    return std::make_shared<Storage>(p, nbytes, Device::cpu(), [rounded](void* data, size_t) {
        aligned_release(data);
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
