#pragma once

#include "vk_allocator.h"
#include "vk_device.h"
#include "vk_pipeline.h"

#include <cstdint>
#include <deque>
#include <string>
#include <string_view>
#include <vector>

namespace vkml::vk {

/// One measured GPU interval, resolved from timestamp queries.
struct ProfileEntry {
    std::string label;
    double gpu_ms = 0.0;

    /// Where the interval STARTS, in milliseconds after its submission's own
    /// window opened. With `gpu_ms` this makes the entry an interval rather
    /// than a duration, and that is the difference between an attribution a
    /// consumer can trust and one it cannot.
    ///
    /// Durations alone cannot distinguish dispatches that ran one after another
    /// from dispatches that ran at the same time, so a consumer summing them
    /// silently multiply-counts the concurrent case. Measured: split-K's
    /// sixteen partitions each report ~0.065 ms and sum to 1.056 ms against a
    /// true submission window of 0.105 ms, giving a REMAINDER OF -0.95 ms. A
    /// negative remainder is the visible symptom; the invisible one is every
    /// per-kernel share being wrong by the same factor.
    ///
    /// With intervals the consumer takes the UNION and the arithmetic closes.
    /// Only meaningful against entries from the same `submission` -- each
    /// submission's window is its own origin.
    double start_ms = 0.0;

    /// Which submission this interval belongs to: the timeline value the
    /// recorder signalled for it. IDENTITY, NOT DESCRIPTION, exactly as
    /// `dispatch` below.
    ///
    /// Required to read `start_ms` at all, and required by rule 3: whole-submit
    /// windows may be summed across submissions because submissions are serial,
    /// while the intervals inside one may not. A consumer that loses track of
    /// which submission an entry came from cannot honour either rule.
    uint64_t submission = 0;

    /// Which dispatch this interval measured, or 0 when the interval is not a
    /// dispatch (the whole-submission entry is the case that exists today).
    ///
    /// IDENTITY, NOT DESCRIPTION. This says WHICH dispatch, never which kernel
    /// or why it was chosen -- those are the Decision's, and duplicating them
    /// here would give kernel selection two owners
    /// (docs/OBSERVABILITY-ARCHITECTURE.md 4b). A consumer joins the two on
    /// this field; neither producer reads the other.
    ///
    /// Compare for EQUALITY only. It is deliberately opaque: code that sorts by
    /// it or computes `id + 1` has taken a dependency on today's counter and
    /// will break when this widens to (queue, ordinal) for multiple queues.
    uint64_t dispatch = 0;
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
    void copy(VkBuffer src, uint64_t src_offset, VkBuffer dst, uint64_t dst_offset, uint64_t bytes);

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

    /// Retains the last `max_submissions` submissions' intervals instead of
    /// only the most recent. 0 -- the default -- disables retention and frees
    /// what was held.
    ///
    /// A training step is many submissions (twelve, measured, for an MLP), and
    /// `profile()` holds one. Anything reasoning about a STEP rather than a
    /// dispatch therefore had nothing to read, which is why per-kernel cost
    /// across a step could not be computed from outside. This is retention
    /// only: it stores what is already produced and interprets none of it.
    void set_profile_history(size_t max_submissions);

    /// Every retained submission's intervals, oldest first, grouped by
    /// `ProfileEntry::submission`. Empty unless retention was asked for.
    [[nodiscard]] const std::vector<ProfileEntry>& profile_history() const noexcept {
        return history_;
    }

    /// Submissions offered to the window since retention began, INCLUDING any
    /// it dropped. Compare with the distinct submissions in
    /// `profile_history()` to detect truncation.
    ///
    /// The parallel to `decisions_published()`, and it exists for the same
    /// reason: a bounded window that silently drops its oldest entries
    /// produces a report that is quietly wrong rather than visibly short.
    ///
    /// Counts SUBMISSIONS WITH DISPATCHES, which is not every submission --
    /// a download is a copy with nothing to time. `submitted_count()` is the
    /// other number, and the difference between the two is itself worth
    /// reading.
    [[nodiscard]] uint64_t profile_submissions_resolved() const noexcept {
        return history_resolved_;
    }

    /// Names subsequent dispatches, for the profile report. Applies until
    /// changed or until the next begin(), so an op that issues SEVERAL
    /// dispatches -- split-K GEMM, a multi-level reduction -- names all of
    /// them. Labelling here rather than pairing profile entries with nodes
    /// afterwards is what keeps the report correct when the two are not 1:1.
    void set_label(std::string_view label) { label_ = label; }

    /// GPU milliseconds summed over every submission so far. 0 unless
    /// profiling is on.
    ///
    /// Summing the WHOLE-SUBMIT windows is the one summation rule 3 permits:
    /// submissions are serial, so their windows cannot overlap. The
    /// per-dispatch entries inside a submission can, which is why they are not
    /// what accumulates here. This exists because `profile()` holds only the
    /// LAST submission, so a workload that submits repeatedly -- a training
    /// step -- had no admissible GPU total, and rule 1b could not be checked
    /// for it at all.
    [[nodiscard]] double total_gpu_ms() const noexcept { return total_gpu_ms_; }

    [[nodiscard]] uint64_t submitted_count() const noexcept { return timeline_value_; }

    [[nodiscard]] uint64_t dispatch_count() const noexcept { return dispatch_count_; }

    /// The DispatchId the next recorded dispatch will carry.
    ///
    /// The recorder is the only thing that records dispatches, so it is the
    /// only thing that may number them. A decision published just before a
    /// dispatch names it through this; the profile entry written just after
    /// carries the same value, and a consumer joins them.
    [[nodiscard]] uint64_t next_dispatch_id() const noexcept { return dispatch_count_ + 1; }

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
    std::string label_;

    struct Pending {
        std::string label;
        uint32_t slot = 0;      ///< first query slot of the pair
        uint64_t dispatch = 0;  ///< 0 when the interval is not a dispatch
    };

    std::vector<Pending> pending_;
    std::vector<ProfileEntry> profile_;
    double total_gpu_ms_ = 0.0;

    // Retention. `history_spans_` holds one entry count per retained
    // submission, so the oldest can be dropped without re-scanning for
    // submission boundaries.
    size_t history_limit_ = 0;
    uint64_t history_resolved_ = 0;
    std::vector<ProfileEntry> history_;
    std::deque<size_t> history_spans_;

    void begin_timestamp(const char* label, uint64_t dispatch_id = 0);
    void end_timestamp();
    void resolve_timestamps();
    void retain(const std::vector<ProfileEntry>& entries);
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
