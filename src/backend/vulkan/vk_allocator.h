#pragma once

#include "vk_device.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace vkml::vk {

/// Where an allocation lives.
enum class MemoryKind : uint8_t {
    /// VRAM. Fast for the GPU, not mappable by the host on a discrete card.
    DeviceLocal,
    /// Host-coherent system memory, mappable. Used for staging transfers.
    ///
    /// Not optional on this GPU: only 256 MiB of VRAM is host-visible
    /// (measured, docs/ARCHITECTURE.md 1.1), so device memory cannot simply be
    /// mapped and every upload has to be staged through here.
    HostStaging,
};

[[nodiscard]] const char* memory_kind_name(MemoryKind kind) noexcept;

/// A suballocation handed out by the Allocator.
///
/// Deliberately NOT a VkBuffer. Each memory block owns one VkBuffer spanning
/// its whole extent, and an allocation is an offset into it. That means:
///
///   - no per-tensor vkCreateBuffer / vkBindBufferMemory, which are expensive
///     and subject to a driver limit on live objects;
///   - one buffer device address per block, so an allocation's address is just
///     `block_address + offset` -- no per-tensor address query;
///   - the descriptor-less model falls out naturally, since a shader only ever
///     needs that 64-bit number.
struct Allocation {
    uint32_t block = 0;          ///< index into the owning pool
    uint64_t offset = 0;         ///< byte offset within the block
    uint64_t size = 0;           ///< usable bytes (the request, not the padded reservation)
    uint64_t padded_size = 0;    ///< bytes actually reserved, after alignment rounding
    VkDeviceAddress address = 0; ///< block address + offset; what shaders receive
    void* mapped = nullptr;      ///< host pointer, HostStaging only
    MemoryKind kind = MemoryKind::DeviceLocal;

    [[nodiscard]] bool valid() const noexcept { return size > 0 || padded_size > 0; }
};

/// Snapshot of allocator state, for diagnostics and leak assertions.
struct AllocatorStats {
    uint64_t reserved_bytes = 0;    ///< total VkDeviceMemory obtained from the driver
    uint64_t in_use_bytes = 0;      ///< currently handed out (padded)
    uint64_t requested_bytes = 0;   ///< currently handed out (unpadded); the difference is waste
    uint64_t peak_in_use_bytes = 0;
    uint32_t block_count = 0;
    uint32_t live_allocations = 0;
    uint64_t total_allocations = 0;  ///< monotonic
    uint64_t device_allocations = 0; ///< monotonic count of vkAllocateMemory calls

    /// Largest single free run, across all blocks.
    uint64_t largest_free_run = 0;

    /// 0.0 when free space is one contiguous run, approaching 1.0 as it
    /// shatters. Defined as 1 - largest_free_run / total_free.
    [[nodiscard]] double fragmentation() const noexcept;

    [[nodiscard]] std::string describe() const;
};

/// Suballocating device memory allocator.
///
/// One vkAllocateMemory per BLOCK, not per tensor. The measured cost of a
/// device allocation is on the order of tens of microseconds against ~50 ns for
/// a malloc (docs/BASELINE-CPU.md), and drivers cap the number of live
/// allocations at a few thousand, so per-tensor allocation is not viable.
///
/// Free-list per block, sorted by offset, coalescing neighbours on release.
/// This is the same structure as ggml's `ggml_dyn_tallocr`, but managing real
/// memory rather than planning offsets -- the two are complementary: this
/// allocator serves persistent tensors, while the M5 execution planner will
/// assign offsets for transient activations inside one large allocation taken
/// from here.
///
/// Thread-safe: a mutex guards every mutation. Contention is irrelevant because
/// allocation happens per tensor, not per element.
class Allocator {
public:
    explicit Allocator(Context& ctx);
    ~Allocator();

    Allocator(const Allocator&) = delete;
    Allocator& operator=(const Allocator&) = delete;
    Allocator(Allocator&&) = delete;
    Allocator& operator=(Allocator&&) = delete;

    /// Reserves `size` bytes. Throws OutOfMemoryError if the device refuses.
    [[nodiscard]] Allocation allocate(uint64_t size, MemoryKind kind,
                                      std::string_view debug_name = {});

    void free(const Allocation& alloc);

    /// The VkBuffer backing an allocation, needed for copies and barriers.
    [[nodiscard]] VkBuffer buffer_of(const Allocation& alloc) const;

    [[nodiscard]] AllocatorStats stats() const;

    /// Releases blocks with no live allocations. Called on idle; never
    /// automatically, because returning memory to the driver only to ask for it
    /// again next step would be worse than holding it.
    void trim();

private:
    struct FreeRun {
        uint64_t offset;
        uint64_t size;
    };

    struct Block {
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceAddress address = 0;
        void* mapped = nullptr;
        uint64_t size = 0;
        uint32_t memory_type = 0;
        MemoryKind kind = MemoryKind::DeviceLocal;
        std::vector<FreeRun> free_runs;
        uint32_t live = 0;

        [[nodiscard]] uint64_t free_bytes() const noexcept;
        [[nodiscard]] uint64_t largest_run() const noexcept;
    };

    [[nodiscard]] uint32_t find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags want,
                                            VkMemoryPropertyFlags avoid,
                                            VkMemoryPropertyFlags prefer = 0) const;
    [[nodiscard]] uint32_t create_block(uint64_t min_size, MemoryKind kind);
    void destroy_block(Block& block);
    void insert_free_run(Block& block, uint64_t offset, uint64_t size);

    Context& ctx_;
    mutable std::mutex mutex_;
    std::vector<Block> blocks_;

    uint64_t alignment_ = 256;
    uint64_t default_block_size_ = 0;

    // Diagnostics.
    uint64_t in_use_ = 0;
    uint64_t requested_ = 0;
    uint64_t peak_in_use_ = 0;
    uint32_t live_allocations_ = 0;
    uint64_t total_allocations_ = 0;
    uint64_t device_allocations_ = 0;
};

}  // namespace vkml::vk
