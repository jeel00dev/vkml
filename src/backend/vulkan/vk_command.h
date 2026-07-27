#pragma once

#include "vk_allocator.h"
#include "vk_device.h"
#include "vk_pipeline.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vkml::vk {

/// One measured GPU interval, resolved from timestamp queries.
struct ProfileEntry {
    std::string label;
    double gpu_ms = 0.0;
};

/// Records GPU work and submits it.
///
/// SEPARATION OF CONCERNS
/// ----------------------
/// This class records and submits. It makes no scheduling decisions: it does
/// not choose an order, does not allocate, does not decide which kernel runs.
/// Those belong to the executor above it, and eventually to the M5 execution
/// graph. Keeping the split means the lowered graph can drive this recorder
/// unchanged -- it will simply call `dispatch` in a different order.
///
/// SYNCHRONISATION STRATEGY
/// ------------------------
/// Two primitives, both deliberately the simplest correct choice:
///
/// 1. A GLOBAL MEMORY BARRIER between dependent dispatches. Not a per-buffer
///    barrier: a single vkCmdPipelineBarrier with shaderRead|shaderWrite on
///    both sides. This is what llama.cpp does, and the reason is that tracking
///    per-buffer hazards costs more CPU time than the barrier costs GPU time,
///    for a graph where almost every node depends on its predecessor anyway.
///    It is conservative -- it orders more than strictly necessary -- which is
///    the right default when correctness comes first.
///
///    The barrier is NOT optional. Without it a dispatch may read a buffer
///    another dispatch is still writing; the GPU does not serialise dispatches
///    on its own. When the M5 planner knows which nodes alias, this can become
///    selective, and that is the one place skipping it will be justified.
///
/// 2. A TIMELINE SEMAPHORE for host-side completion. One monotonically
///    increasing counter per stream, incremented per submit; waiting for value
///    N means "everything up to submit N has finished". This replaces fences
///    entirely -- no fence pool, no reset, no per-submit object -- and is why
///    the device requires timelineSemaphore.
class Recorder {
public:
    Recorder(Context& ctx, Allocator& allocator);
    ~Recorder();

    Recorder(const Recorder&) = delete;
    Recorder& operator=(const Recorder&) = delete;
    Recorder(Recorder&&) = delete;
    Recorder& operator=(Recorder&&) = delete;

    /// Starts recording. Must be paired with submit().
    void begin();

    /// Binds a pipeline, pushes constants and dispatches enough workgroups to
    /// cover `element_count` invocations.
    void dispatch(const PipelineCache::Pipeline& pipeline, const void* push_constants,
                  uint32_t push_constant_bytes, uint64_t element_count);

    /// As dispatch(), but with an explicit workgroup count.
    ///
    /// Reductions need this: they launch one workgroup per OUTPUT element,
    /// which has no relation to the total element count.
    void dispatch_groups(const PipelineCache::Pipeline& pipeline, const void* push_constants,
                         uint32_t push_constant_bytes, uint64_t group_count);

    /// Orders every prior write against every subsequent read. See the class
    /// comment for why this is global rather than per-buffer.
    void barrier();

    /// Records a buffer-to-buffer copy, used for staged uploads and downloads.
    void copy(VkBuffer src, uint64_t src_offset, VkBuffer dst, uint64_t dst_offset,
              uint64_t bytes);

    /// Abandons an in-progress recording.
    ///
    /// Needed because a kernel that throws mid-recording would otherwise leave
    /// the recorder permanently in the recording state, and every subsequent
    /// begin() would fail with a misleading assertion instead of the original
    /// error. compute() drives this from a scope guard.
    void abort_recording() noexcept;

    [[nodiscard]] bool recording() const noexcept { return recording_; }

    /// Ends recording and submits. Returns the timeline value to wait on.
    [[nodiscard]] uint64_t submit();

    /// Blocks until the given timeline value has been reached.
    void wait(uint64_t value);

    /// Blocks until every submission has completed.
    void wait_idle();

    /// Enables timestamp queries around each dispatch.
    ///
    /// Off by default and genuinely zero-cost when off: no query pool is
    /// created, and the record path takes a single predictable branch. Wall
    /// clock cannot separate GPU execution from upload, submission and
    /// synchronisation, which is what makes this necessary rather than nice --
    /// every performance number gathered before it existed was suspect.
    void set_profiling(bool enabled);

    [[nodiscard]] bool profiling() const noexcept { return profiling_; }

    /// Intervals from the most recent submission. Valid after wait().
    [[nodiscard]] const std::vector<ProfileEntry>& profile() const noexcept { return profile_; }

    /// Names the next dispatch, for the profile report.
    void set_label(std::string label) { next_label_ = std::move(label); }

    [[nodiscard]] uint64_t submitted_count() const noexcept { return timeline_value_; }
    [[nodiscard]] uint64_t dispatch_count() const noexcept { return dispatch_count_; }

private:
    void reset_command_buffer();

    Context& ctx_;
    Allocator& allocator_;

    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkSemaphore timeline_ = VK_NULL_HANDLE;

    uint64_t timeline_value_ = 0;
    uint64_t dispatch_count_ = 0;
    bool recording_ = false;

    // Profiling. All inert unless set_profiling(true) has been called.
    bool profiling_ = false;
    VkQueryPool query_pool_ = VK_NULL_HANDLE;
    uint32_t query_index_ = 0;
    std::string next_label_;
    std::vector<std::pair<std::string, uint32_t>> pending_;  // label, first query slot
    std::vector<ProfileEntry> profile_;

    void begin_timestamp(const char* label);
    void end_timestamp();
    void resolve_timestamps();
};

/// Host-visible scratch used to move data to and from device-local memory.
///
/// Necessary rather than convenient: the target GPU exposes only 256 MiB of
/// host-visible device-local memory against 5.75 GiB of VRAM (measured,
/// docs/ARCHITECTURE.md 1.1), so tensors cannot simply be mapped and written.
///
/// Transfers larger than the staging buffer are chunked. The buffer is
/// allocated once and reused; a fresh staging allocation per upload would cost
/// a device allocation each time.
class StagingBuffer {
public:
    StagingBuffer(Context& ctx, Allocator& allocator, Recorder& recorder, uint64_t capacity);
    ~StagingBuffer();

    StagingBuffer(const StagingBuffer&) = delete;
    StagingBuffer& operator=(const StagingBuffer&) = delete;
    StagingBuffer(StagingBuffer&&) = delete;
    StagingBuffer& operator=(StagingBuffer&&) = delete;

    /// Host -> device. Blocking.
    void upload(const void* src, const Allocation& dst, uint64_t dst_offset, uint64_t bytes);

    /// Device -> host. Blocking.
    void download(void* dst, const Allocation& src, uint64_t src_offset, uint64_t bytes);

    [[nodiscard]] uint64_t capacity() const noexcept { return staging_.size; }

private:
    Context& ctx_;
    Allocator& allocator_;
    Recorder& recorder_;
    Allocation staging_;
};

}  // namespace vkml::vk
