#include "vk_allocator.h"

#include "vkml/util/assert.h"
#include "vkml/util/log.h"

#include <algorithm>
#include <format>

namespace vkml::vk {
namespace {

/// Default block size.
///
/// 64 MiB balances two failure modes: too small and the driver sees thousands
/// of allocations (they are ~100x a malloc and capped in number); too large and
/// a 5.75 GiB card wastes a meaningful fraction on a partly-used block. A
/// request larger than this gets its own exactly-sized block instead.
constexpr uint64_t kDefaultBlockSize = 64ULL * 1024 * 1024;

[[nodiscard]] constexpr uint64_t round_up(uint64_t value, uint64_t align) noexcept {
    return (value + align - 1) / align * align;
}

}  // namespace

const char* memory_kind_name(MemoryKind kind) noexcept {
    switch (kind) {
        case MemoryKind::DeviceLocal: return "device";
        case MemoryKind::HostStaging: return "staging";
    }
    return "?";
}

double AllocatorStats::fragmentation() const noexcept {
    const uint64_t total_free = reserved_bytes - in_use_bytes;
    if (total_free == 0) {
        return 0.0;
    }
    return 1.0 - static_cast<double>(largest_free_run) / static_cast<double>(total_free);
}

std::string AllocatorStats::describe() const {
    const double mib = 1024.0 * 1024.0;
    return std::format(
        "reserved {:.1f} MiB in {} block(s) | in use {:.1f} MiB ({} live, peak {:.1f} MiB) | "
        "waste {:.2f} MiB | largest free run {:.1f} MiB | fragmentation {:.1f}% | "
        "{} allocations, {} device allocations",
        static_cast<double>(reserved_bytes) / mib, block_count,
        static_cast<double>(in_use_bytes) / mib, live_allocations,
        static_cast<double>(peak_in_use_bytes) / mib,
        static_cast<double>(in_use_bytes - requested_bytes) / mib,
        static_cast<double>(largest_free_run) / mib, fragmentation() * 100.0,
        total_allocations,
        device_allocations);
}

uint64_t Allocator::Block::free_bytes() const noexcept {
    uint64_t total = 0;
    for (const FreeRun& r : free_runs) {
        total += r.size;
    }
    return total;
}

uint64_t Allocator::Block::largest_run() const noexcept {
    uint64_t best = 0;
    for (const FreeRun& r : free_runs) {
        best = std::max(best, r.size);
    }
    return best;
}

Allocator::Allocator(Context& ctx) : ctx_(ctx) {
    // Every suballocation must satisfy the strictest alignment any consumer
    // needs. 256 covers minStorageBufferOffsetAlignment on every device the
    // project targets and is a cache-line multiple, so it costs nothing to be
    // generous here.
    alignment_ = std::max<uint64_t>(ctx.info().min_storage_buffer_offset_alignment, 256);

    default_block_size_ = std::min<uint64_t>(kDefaultBlockSize, ctx.info().max_allocation_size);

    VKML_LOG_DEBUG("vulkan allocator: block size {} MiB, alignment {} B",
                   default_block_size_ / (1024 * 1024), alignment_);
}

Allocator::~Allocator() {
    const AllocatorStats s = stats();
    if (s.live_allocations != 0) {
        // Not thrown: a destructor must not throw, and tearing the device down
        // with allocations outstanding is survivable. It is always a bug in
        // vkml, so it is logged loudly.
        VKML_LOG_ERROR("vulkan allocator destroyed with {} live allocation(s), {} bytes leaked",
                       s.live_allocations, s.in_use_bytes);
    }
    for (Block& b : blocks_) {
        destroy_block(b);
    }
}

uint32_t Allocator::find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags want,
                                     VkMemoryPropertyFlags avoid,
                                     VkMemoryPropertyFlags prefer) const {
    VkPhysicalDeviceMemoryProperties mem{};
    vkGetPhysicalDeviceMemoryProperties(ctx_.physical(), &mem);

    // A preferred-flags pass first. This is not cosmetic: on this device the
    // FIRST host-visible+coherent type is write-combined (no HOST_CACHED), and
    // CPU *reads* from write-combined memory are roughly two orders of
    // magnitude slower than from cached memory. Downloads memcpy out of the
    // staging buffer, so picking the first match cost ~23 ms per 4 MiB.
    if (prefer != 0) {
        for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
            if ((type_bits & (1U << i)) == 0) {
                continue;
            }
            const VkMemoryPropertyFlags flags = mem.memoryTypes[i].propertyFlags;
            if ((flags & want) != want || (flags & prefer) != prefer) {
                continue;
            }
            if (avoid != 0 && (flags & avoid) != 0) {
                continue;
            }
            return i;
        }
    }

    for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
        if ((type_bits & (1U << i)) == 0) {
            continue;
        }
        const VkMemoryPropertyFlags flags = mem.memoryTypes[i].propertyFlags;
        if ((flags & want) != want) {
            continue;
        }
        if (avoid != 0 && (flags & avoid) != 0) {
            continue;
        }
        return i;
    }

    // Retry without the avoid-set. This matters on integrated GPUs, where every
    // memory type is both device-local and host-visible, so "device-local but
    // NOT host-visible" matches nothing.
    if (avoid != 0) {
        return find_memory_type(type_bits, want, 0);
    }
    throw DeviceError(std::format(
        "no memory type satisfies flags 0x{:x} (typeBits 0x{:x})", want, type_bits));
}

uint32_t Allocator::create_block(uint64_t min_size, MemoryKind kind) {
    const uint64_t size = std::max(round_up(min_size, alignment_), default_block_size_);
    VKML_CHECK(size <= ctx_.info().max_allocation_size, OutOfMemoryError,
               "requested {} bytes exceeds the device's {} byte allocation limit", size,
               ctx_.info().max_allocation_size);

    Block block;
    block.size = size;
    block.kind = kind;

    // One buffer spanning the block. SHADER_DEVICE_ADDRESS is what makes the
    // descriptor-less model work; TRANSFER_SRC/DST cover staging copies.
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(ctx_.device(), &bci, nullptr, &block.buffer), "vkCreateBuffer");

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(ctx_.device(), block.buffer, &req);

    const VkMemoryPropertyFlags want =
        kind == MemoryKind::DeviceLocal
            ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            : (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    // For device-local, actively avoid the host-visible types: on this GPU that
    // is the 256 MiB BAR window, and burning it on ordinary tensors would
    // exhaust it immediately.
    const VkMemoryPropertyFlags avoid =
        kind == MemoryKind::DeviceLocal ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT : 0;

    // Staging is read back by the host, so cached memory matters a great deal.
    const VkMemoryPropertyFlags prefer =
        kind == MemoryKind::HostStaging ? VK_MEMORY_PROPERTY_HOST_CACHED_BIT : 0;
    block.memory_type = find_memory_type(req.memoryTypeBits, want, avoid, prefer);

    VkMemoryAllocateFlagsInfo flags{};
    flags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.pNext = &flags;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = block.memory_type;

    const VkResult r = vkAllocateMemory(ctx_.device(), &mai, nullptr, &block.memory);
    if (r != VK_SUCCESS) {
        vkDestroyBuffer(ctx_.device(), block.buffer, nullptr);
        check(r, std::format("vkAllocateMemory({} MiB, {})", size / (1024 * 1024),
                             memory_kind_name(kind))
                     .c_str());
    }
    ++device_allocations_;

    check(vkBindBufferMemory(ctx_.device(), block.buffer, block.memory, 0), "vkBindBufferMemory");

    VkBufferDeviceAddressInfo addr_info{};
    addr_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addr_info.buffer = block.buffer;
    block.address = vkGetBufferDeviceAddress(ctx_.device(), &addr_info);
    VKML_ASSERT(block.address != 0, "buffer device address query returned 0");

    if (kind == MemoryKind::HostStaging) {
        // Mapped once for the block's lifetime. Repeated map/unmap is pure
        // overhead, and coherent memory needs no explicit flush.
        check(vkMapMemory(ctx_.device(), block.memory, 0, VK_WHOLE_SIZE, 0, &block.mapped),
              "vkMapMemory");
    }

    block.free_runs.push_back(FreeRun{0, size});
    blocks_.push_back(block);

    VKML_LOG_DEBUG("vulkan allocator: new {} block #{} of {} MiB", memory_kind_name(kind),
                   blocks_.size() - 1, size / (1024 * 1024));
    return static_cast<uint32_t>(blocks_.size() - 1);
}

void Allocator::destroy_block(Block& block) {
    if (block.mapped != nullptr) {
        vkUnmapMemory(ctx_.device(), block.memory);
        block.mapped = nullptr;
    }
    if (block.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(ctx_.device(), block.buffer, nullptr);
        block.buffer = VK_NULL_HANDLE;
    }
    if (block.memory != VK_NULL_HANDLE) {
        vkFreeMemory(ctx_.device(), block.memory, nullptr);
        block.memory = VK_NULL_HANDLE;
    }
}

Allocation Allocator::allocate(uint64_t size, MemoryKind kind, std::string_view debug_name) {
    const std::lock_guard<std::mutex> lock(mutex_);

    // A zero-size tensor is legal upstream. Reserve one aligned unit so the
    // allocation still has a distinct, valid device address.
    const uint64_t padded = round_up(std::max<uint64_t>(size, 1), alignment_);

    // First fit rather than best fit. Best fit leaves a trail of unusably small
    // gaps; first fit over an offset-sorted list keeps large runs intact at the
    // end of the block, which is what the next big tensor needs. Revisit only
    // if fragmentation() reports a problem.
    for (uint32_t bi = 0; bi < blocks_.size(); ++bi) {
        Block& block = blocks_[bi];
        if (block.kind != kind) {
            continue;
        }
        for (size_t ri = 0; ri < block.free_runs.size(); ++ri) {
            FreeRun& run = block.free_runs[ri];
            if (run.size < padded) {
                continue;
            }

            const uint64_t offset = run.offset;
            run.offset += padded;
            run.size -= padded;
            if (run.size == 0) {
                block.free_runs.erase(block.free_runs.begin() + static_cast<ptrdiff_t>(ri));
            }
            ++block.live;

            in_use_ += padded;
            requested_ += size;
            peak_in_use_ = std::max(peak_in_use_, in_use_);
            ++live_allocations_;
            ++total_allocations_;

            Allocation a;
            a.block = bi;
            a.offset = offset;
            a.size = size;
            a.padded_size = padded;
            a.address = block.address + offset;
            a.mapped = block.mapped != nullptr
                           ? static_cast<std::byte*>(block.mapped) + offset
                           : nullptr;
            a.kind = kind;

            if (!debug_name.empty()) {
                VKML_LOG_TRACE("vulkan alloc '{}': {} B at block {} offset {}", debug_name, size,
                               bi, offset);
            }
            return a;
        }
    }

    const uint32_t bi = create_block(padded, kind);
    Block& block = blocks_[bi];

    block.free_runs[0].offset = padded;
    block.free_runs[0].size -= padded;
    if (block.free_runs[0].size == 0) {
        block.free_runs.clear();
    }
    ++block.live;

    in_use_ += padded;
    requested_ += size;
    peak_in_use_ = std::max(peak_in_use_, in_use_);
    ++live_allocations_;
    ++total_allocations_;

    Allocation a;
    a.block = bi;
    a.offset = 0;
    a.size = size;
    a.padded_size = padded;
    a.address = block.address;
    a.mapped = block.mapped;
    a.kind = kind;
    return a;
}

void Allocator::insert_free_run(Block& block, uint64_t offset, uint64_t size) {
    // Keep the list sorted by offset so that coalescing only ever has to look
    // at the immediate neighbours.
    auto pos = std::lower_bound(block.free_runs.begin(), block.free_runs.end(), offset,
                                [](const FreeRun& r, uint64_t o) { return r.offset < o; });
    auto it = block.free_runs.insert(pos, FreeRun{offset, size});

    // Merge forward, then backward. Without this, a block degrades into a list
    // of exactly-sized holes and every subsequent allocation of a different
    // size has to create a new block.
    auto next = std::next(it);
    if (next != block.free_runs.end() && it->offset + it->size == next->offset) {
        it->size += next->size;
        block.free_runs.erase(next);
    }
    if (it != block.free_runs.begin()) {
        auto prev = std::prev(it);
        if (prev->offset + prev->size == it->offset) {
            prev->size += it->size;
            block.free_runs.erase(it);
        }
    }
}

void Allocator::free(const Allocation& alloc) {
    if (!alloc.valid()) {
        return;
    }
    const std::lock_guard<std::mutex> lock(mutex_);

    VKML_ASSERT(alloc.block < blocks_.size(), "free() with block index {} of {}", alloc.block,
                blocks_.size());
    Block& block = blocks_[alloc.block];
    VKML_ASSERT(block.live > 0, "free() on a block with no live allocations");
    VKML_ASSERT(alloc.offset + alloc.padded_size <= block.size,
                "free() range [{}, {}) exceeds block size {}", alloc.offset,
                alloc.offset + alloc.padded_size, block.size);

    insert_free_run(block, alloc.offset, alloc.padded_size);
    --block.live;

    in_use_ -= alloc.padded_size;
    requested_ -= alloc.size;
    --live_allocations_;
}

VkBuffer Allocator::buffer_of(const Allocation& alloc) const {
    const std::lock_guard<std::mutex> lock(mutex_);
    VKML_ASSERT(alloc.block < blocks_.size(), "buffer_of() with an out-of-range block index");
    return blocks_[alloc.block].buffer;
}

AllocatorStats Allocator::stats() const {
    const std::lock_guard<std::mutex> lock(mutex_);

    AllocatorStats s;
    s.in_use_bytes = in_use_;
    s.requested_bytes = requested_;
    s.peak_in_use_bytes = peak_in_use_;
    s.live_allocations = live_allocations_;
    s.total_allocations = total_allocations_;
    s.device_allocations = device_allocations_;
    s.block_count = static_cast<uint32_t>(blocks_.size());

    for (const Block& b : blocks_) {
        s.reserved_bytes += b.size;
        s.largest_free_run = std::max(s.largest_free_run, b.largest_run());
    }
    return s;
}

void Allocator::trim() {
    const std::lock_guard<std::mutex> lock(mutex_);

    // Erase-and-compact would invalidate every Allocation's block index, so
    // empty blocks are torn down in place and left as zero-sized holes that
    // allocate() skips (kind never matches a destroyed block because its
    // free_runs are empty).
    uint32_t freed = 0;
    for (Block& b : blocks_) {
        if (b.live == 0 && b.memory != VK_NULL_HANDLE) {
            destroy_block(b);
            b.free_runs.clear();
            b.size = 0;
            ++freed;
        }
    }
    if (freed != 0) {
        VKML_LOG_DEBUG("vulkan allocator: trimmed {} empty block(s)", freed);
    }
}

}  // namespace vkml::vk
