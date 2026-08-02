#include "vk_command.h"

#include "vk_dispatch_grid.h"

#include "vkml/util/assert.h"
#include "vkml/util/log.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <limits>

namespace vkml::vk {

Recorder::Recorder(Context& ctx, Allocator& allocator) : ctx_(ctx), allocator_(allocator) {
    VkCommandPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    // RESET_COMMAND_BUFFER lets one buffer be re-recorded per submission rather
    // than allocating a new one each time.
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = ctx.queue_family();
    check(vkCreateCommandPool(ctx_.device(), &pci, nullptr, &pool_), "vkCreateCommandPool");

    std::vector<VkCommandBuffer> buffers(kRingSize);
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = pool_;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = kRingSize;
    check(vkAllocateCommandBuffers(ctx_.device(), &cbai, buffers.data()),
          "vkAllocateCommandBuffers");

    ring_.resize(kRingSize);
    for (uint32_t i = 0; i < kRingSize; ++i) {
        ring_[i].cmd = buffers[i];
    }

    VkSemaphoreTypeCreateInfo tci{};
    tci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    tci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    tci.initialValue = 0;

    VkSemaphoreCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sci.pNext = &tci;
    check(vkCreateSemaphore(ctx_.device(), &sci, nullptr, &timeline_), "vkCreateSemaphore");
}

Recorder::~Recorder() {
    wait_idle();
    if (query_pool_ != VK_NULL_HANDLE) {
        vkDestroyQueryPool(ctx_.device(), query_pool_, nullptr);
    }
    if (timeline_ != VK_NULL_HANDLE) {
        vkDestroySemaphore(ctx_.device(), timeline_, nullptr);
    }
    if (pool_ != VK_NULL_HANDLE) {
        // Destroying the pool frees its command buffers.
        vkDestroyCommandPool(ctx_.device(), pool_, nullptr);
    }
}

namespace {
/// Query slots across the whole ring. Two per dispatch (start, end), so 511
/// dispatches per submission plus the whole-submit pair -- far beyond anything
/// the current executor produces, and the same figure as before the ring.
///
/// Partitioned per ring slot rather than shared, because several submissions
/// are in flight at once now: a shared cursor lets one submission reset query
/// slots another is still writing, which is not a wrong number but a
/// VK_ERROR_DEVICE_LOST. That is the negative control on the ring test.
constexpr uint32_t kMaxQueries = Recorder::kRingSize * Recorder::kQueriesPerSlot;
}  // namespace

void Recorder::set_profiling(bool enabled) {
    if (enabled == profiling_) {
        return;
    }
    profiling_ = enabled;

    if (enabled && query_pool_ == VK_NULL_HANDLE) {
        VkQueryPoolCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qci.queryCount = kMaxQueries;
        check(vkCreateQueryPool(ctx_.device(), &qci, nullptr, &query_pool_), "vkCreateQueryPool");
    }
}

void Recorder::set_profile_history(size_t max_submissions) {
    history_limit_ = max_submissions;
    history_resolved_ = 0;
    history_.clear();
    history_spans_.clear();
    if (max_submissions == 0) {
        history_.shrink_to_fit();
    }
}

void Recorder::retain(const std::vector<ProfileEntry>& entries) {
    if (history_limit_ == 0 || entries.empty()) {
        return;
    }
    ++history_resolved_;
    history_.insert(history_.end(), entries.begin(), entries.end());
    history_spans_.push_back(entries.size());

    while (history_spans_.size() > history_limit_) {
        const size_t oldest = history_spans_.front();
        history_spans_.pop_front();
        history_.erase(history_.begin(), history_.begin() + static_cast<ptrdiff_t>(oldest));
    }
}

void Recorder::begin_timestamp(const char* label, uint64_t dispatch_id) {
    if (!profiling_ || query_index_ + 2 > query_base(current_) + kQueriesPerSlot) {
        return;
    }
    ring_[current_].pending.push_back(
        Pending{label_.empty() ? label : label_, query_index_, dispatch_id});
    // TOP_OF_PIPE for the start: the earliest point the dispatch can be said to
    // have begun.
    vkCmdWriteTimestamp2(cmd(), VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, query_pool_, query_index_);
    query_index_ += 2;
}

void Recorder::end_timestamp() {
    if (!profiling_ || ring_[current_].pending.empty()) {
        return;
    }
    // ALL_COMMANDS for the end: the dispatch is only finished once its writes
    // have drained, and BOTTOM_OF_PIPE would report early on some drivers.
    vkCmdWriteTimestamp2(cmd(), VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, query_pool_,
                         ring_[current_].pending.back().slot + 1);
}

void Recorder::resolve_slot(Slot& slot) {
    slot.resolved = true;

    // Deliberately does NOT clear profile_ when there is nothing pending. A
    // download submits a copy with no dispatches and then waits, and that wait
    // would otherwise wipe the profile of the compute submission that produced
    // the data being downloaded -- which is exactly what happened, reporting
    // 0.000 ms for every kernel.
    if (!profiling_ || slot.pending.empty()) {
        slot.pending.clear();
        return;
    }
    profile_.clear();

    // Only this slot's range is read back. Reading the whole pool would return
    // other slots' timestamps, which belong to submissions that may still be
    // running -- and WAIT_BIT would then block on them.
    //
    // The range is contiguous: `submit` occupies the slot's first pair and each
    // dispatch takes the next, so the queries run from `first` in steps of two.
    const auto count = static_cast<uint32_t>(slot.pending.size() * 2);
    const uint32_t first = slot.pending.front().slot;
    std::vector<uint64_t> raw(count, 0);
    const VkResult r = vkGetQueryPoolResults(
        ctx_.device(), query_pool_, first, count, raw.size() * sizeof(uint64_t), raw.data(),
        sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

    if (r == VK_SUCCESS) {
        // Ticks are device-specific; timestampPeriod converts to nanoseconds.
        const double ns_per_tick = static_cast<double>(ctx_.info().timestamp_period);

        // Every interval is reported relative to one origin, so that a consumer
        // can place them on a common line and take their union. The earliest
        // start is that origin -- normally the whole-submit window's, which is
        // written before any other command, but taken as a minimum rather than
        // assumed so that a profile without a submit entry is still coherent.
        uint64_t origin = std::numeric_limits<uint64_t>::max();
        for (const auto& p : slot.pending) {
            origin = std::min(origin, raw[p.slot - first]);
        }

        for (const auto& [label, query, dispatch_id] : slot.pending) {
            const uint64_t start = raw[query - first];
            const uint64_t end = raw[query - first + 1];
            const double ms =
                end > start ? static_cast<double>(end - start) * ns_per_tick / 1e6 : 0.0;
            const double start_ms = static_cast<double>(start - origin) * ns_per_tick / 1e6;
            profile_.push_back(ProfileEntry{label, ms, start_ms, slot.value, dispatch_id});
            // Only the whole-submit window accumulates; see total_gpu_ms().
            if (label == "submit") {
                total_gpu_ms_ += ms;
            }
        }
        retain(profile_);
    }
    slot.pending.clear();
}

void Recorder::resolve_completed() {
    if (!profiling_) {
        // Nothing to read back, but the flags must still be cleared or a slot
        // profiled earlier would be resolved again after profiling was turned
        // off, against queries that had since been reset.
        for (Slot& slot : ring_) {
            slot.resolved = true;
            slot.pending.clear();
        }
        return;
    }

    // Oldest first, so `profile()` ends up holding the most recently COMPLETED
    // submission rather than an arbitrary one. Retention receives them in
    // execution order for the same reason.
    const uint64_t completed = completed_value();
    std::vector<Slot*> ready;
    for (Slot& slot : ring_) {
        if (!slot.resolved && slot.value <= completed) {
            ready.push_back(&slot);
        }
    }
    std::sort(ready.begin(), ready.end(),
              [](const Slot* a, const Slot* b) { return a->value < b->value; });
    for (Slot* slot : ready) {
        resolve_slot(*slot);
    }
}

void Recorder::begin() {
    VKML_ASSERT(!recording_, "Recorder::begin() called while already recording");

    // Take the next slot in the ring and wait only for ITS OWN previous
    // submission -- not for the most recent one, which is the whole point. With
    // a single buffer this was `wait(timeline_value_)` and serialised the host
    // against the GPU on every realise (docs/adr/0012 sec1).
    current_ = (current_ + 1) % kRingSize;
    Slot& slot = ring_[current_];

    const bool anything_outstanding = work_outstanding();

    if (slot.value != 0) {
        // Blocks only when the GPU is kRingSize submissions behind.
        wait(slot.value);
    }
    resolve_completed();

    check(vkResetCommandBuffer(slot.cmd, 0), "vkResetCommandBuffer");

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(slot.cmd, &bi), "vkBeginCommandBuffer");
    recording_ = true;
    label_.clear();  // a label must not leak from the previous submission

    if (profiling_) {
        query_index_ = query_base(current_);
        slot.pending.clear();
        vkCmdResetQueryPool(cmd(), query_pool_, query_base(current_), kQueriesPerSlot);

        // Slots 0 and 1 are reserved for the WHOLE-SUBMIT window.
        //
        // Per-dispatch timestamps end at ALL_COMMANDS, which is a GLOBAL drain
        // point: a dispatch's end timestamp fires only once every concurrently
        // running dispatch has also finished. Independent dispatches therefore
        // each report a window stretching to the end of the whole group, and
        // summing them multiply-counts the same elapsed time. Split-K's eight
        // partitions each report ~0.84 ms and sum to 7.2 ms, against a true
        // concurrent total of 0.93 ms (docs/MEASUREMENT-AUDIT.md 3).
        //
        // Execution is NOT altered -- profiled and unprofiled wall clock agree
        // to within host readback overhead. Only the attribution is wrong.
        //
        // This outer pair brackets the entire command buffer, so it is the one
        // timing that stays valid when dispatches overlap. It is written here
        // but only REPORTED if the submit contained at least one dispatch --
        // see submit(), which preserves the download-profile guard below.
        vkCmdWriteTimestamp2(cmd(), VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, query_pool_,
                             query_base(current_));
        query_index_ = query_base(current_) + 2;
    }

    // ORDERING AGAINST EARLIER SUBMISSIONS.
    //
    // Submissions to one queue are ordered, and a barrier's first scope covers
    // every command submitted earlier in submission order -- including commands
    // in earlier BATCHES. One barrier here therefore makes every prior write on
    // this queue visible to everything recorded below.
    //
    // Needed because the host no longer waits between submissions. It is not
    // enough that the compute path happens to end each submission with a
    // barrier: copy_device_to_device ends with copies and none, so the property
    // would hold by accident of one caller rather than by construction
    // (docs/adr/0012 sec4b).
    //
    // Emitted only when something is actually in flight, so the common
    // quiescent case -- the first submission after a wait -- pays nothing. The
    // cost when it does fire is ~2.4 us against the ~42 us not blocking saves.
    if (anything_outstanding) {
        barrier();
    }
}

void Recorder::dispatch(const PipelineCache::Pipeline& pipeline, const void* push_constants,
                        uint32_t push_constant_bytes, uint64_t element_count) {
    VKML_ASSERT(recording_, "dispatch() outside begin()/submit()");
    if (element_count == 0) {
        return;
    }

    vkCmdBindPipeline(cmd(), VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
    if (push_constant_bytes > 0) {
        vkCmdPushConstants(cmd(), pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           push_constant_bytes, push_constants);
    }

    // The geometry lives in vk_dispatch_grid.h, as a pure function taking the
    // limits as parameters. It is not inlined here because the device this runs
    // on reports 2^32-1 workgroups in x against a guaranteed 65535, so the
    // folding path is unreachable locally -- and a rule that cannot be exercised
    // on the machine that writes it is how issue #20 shipped. Separated, it is
    // tested against the guaranteed minimum in every build.
    const DispatchGrid grid = choose_dispatch_grid(element_count, pipeline.workgroup_size,
                                                   ctx_.info().max_workgroup_count[0],
                                                   ctx_.info().max_workgroup_count[1]);

    begin_timestamp("dispatch", dispatch_count_ + 1);
    vkCmdDispatch(cmd(), grid.groups_x, grid.groups_y, 1);
    end_timestamp();
    ++dispatch_count_;
}

void Recorder::dispatch_groups(const PipelineCache::Pipeline& pipeline, const void* push_constants,
                               uint32_t push_constant_bytes, uint64_t group_count) {
    VKML_ASSERT(recording_, "dispatch_groups() outside begin()/submit()");
    if (group_count == 0) {
        return;
    }
    vkCmdBindPipeline(cmd(), VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
    if (push_constant_bytes > 0) {
        vkCmdPushConstants(cmd(), pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           push_constant_bytes, push_constants);
    }
    // Same ceiling as dispatch(), reached sooner: this path dispatches one group
    // per output ROW, so a tensor with more than maxComputeWorkGroupCount[x] rows
    // exceeds a one-dimensional grid. 32 images of 3x64x64 average-pooled is
    // 98,304 rows against a guaranteed 65,535 (issue #20).
    //
    // The kernels on this path reconstruct the flat group index with
    // global_group_index() in common.glsl, which is identically
    // gl_WorkGroupID.x whenever y holds a single group.
    const uint64_t max_x = ctx_.info().max_workgroup_count[0];
    const uint64_t groups_x = group_count <= max_x ? group_count : max_x;
    const uint64_t groups_y = (group_count + groups_x - 1) / groups_x;

    VKML_CHECK(groups_y <= ctx_.info().max_workgroup_count[1], DeviceError,
               "reduction needs {} workgroups ({} x {}) but the device allows {} x {}", group_count,
               groups_x, groups_y, max_x, ctx_.info().max_workgroup_count[1]);
    begin_timestamp("dispatch", dispatch_count_ + 1);
    vkCmdDispatch(cmd(), static_cast<uint32_t>(groups_x), static_cast<uint32_t>(groups_y), 1);
    end_timestamp();
    ++dispatch_count_;
}

void Recorder::barrier() {
    VKML_ASSERT(recording_, "barrier() outside begin()/submit()");

    // One global memory barrier. Conservative by design -- see the strategy
    // note in vk_command.h. Removing it is only safe once the planner can prove
    // two dispatches touch disjoint memory.
    VkMemoryBarrier2 mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT |
                       VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &mb;

    vkCmdPipelineBarrier2(cmd(), &dep);
}

void Recorder::copy(VkBuffer src, uint64_t src_offset, VkBuffer dst, uint64_t dst_offset,
                    uint64_t bytes) {
    VKML_ASSERT(recording_, "copy() outside begin()/submit()");
    if (bytes == 0) {
        return;
    }
    VkBufferCopy region{};
    region.srcOffset = src_offset;
    region.dstOffset = dst_offset;
    region.size = bytes;
    vkCmdCopyBuffer(cmd(), src, dst, 1, &region);
}

void Recorder::abort_recording() noexcept {
    if (!recording_) {
        return;
    }
    // End the buffer so it is in a legal state, then drop it. Nothing is
    // submitted, so no GPU work results.
    vkEndCommandBuffer(cmd());
    recording_ = false;
}

uint64_t Recorder::submit() {
    VKML_ASSERT(recording_, "submit() without begin()");

    Slot& slot = ring_[current_];

    // Close the whole-submit window, but ONLY when this submit actually
    // dispatched something. A download is a copy with no dispatches, and
    // reporting a profile entry for it would make the resolve clear the compute
    // profile it is meant to preserve -- the 0.000 ms bug. Keying on the
    // pending list being non-empty reuses that exact guard rather than
    // inventing a second one that could drift from it.
    if (profiling_ && !slot.pending.empty()) {
        vkCmdWriteTimestamp2(cmd(), VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, query_pool_,
                             query_base(current_) + 1);
        // dispatch id 0: the whole-submit window is not a dispatch, and a
        // consumer joining on identity must not match it to one.
        slot.pending.insert(slot.pending.begin(), Pending{"submit", query_base(current_), 0});
    }

    check(vkEndCommandBuffer(cmd()), "vkEndCommandBuffer");
    recording_ = false;

    const uint64_t signal_value = timeline_value_ + 1;

    VkCommandBufferSubmitInfo cbsi{};
    cbsi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    cbsi.commandBuffer = cmd();

    VkSemaphoreSubmitInfo ssi{};
    ssi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    ssi.semaphore = timeline_;
    ssi.value = signal_value;
    ssi.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

    VkSubmitInfo2 si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &cbsi;
    si.signalSemaphoreInfoCount = 1;
    si.pSignalSemaphoreInfos = &ssi;

    check(vkQueueSubmit2(ctx_.queue(), 1, &si, VK_NULL_HANDLE), "vkQueueSubmit2");

    timeline_value_ = signal_value;
    slot.value = signal_value;
    slot.resolved = !profiling_ || slot.pending.empty();

    // Returns WITHOUT waiting. The caller decides when it needs the results,
    // and the ones that need them at all are the host reads.
    return signal_value;
}

uint64_t Recorder::completed_value() const {
    uint64_t value = 0;
    // A poll. vkGetSemaphoreCounterValue does not block, which is what lets the
    // allocator retire memory opportunistically without ever paying the ~40 us
    // a real wait costs.
    if (vkGetSemaphoreCounterValue(ctx_.device(), timeline_, &value) != VK_SUCCESS) {
        return 0;
    }
    return value;
}

void Recorder::wait(uint64_t value) {
    if (value == 0) {
        return;
    }
    VkSemaphoreWaitInfo wi{};
    wi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    wi.semaphoreCount = 1;
    wi.pSemaphores = &timeline_;
    wi.pValues = &value;

    // A finite timeout rather than UINT64_MAX: a hung shader should surface as
    // a diagnosable error instead of an unkillable process.
    constexpr uint64_t kTimeoutNs = 30ULL * 1000 * 1000 * 1000;
    const VkResult r = vkWaitSemaphores(ctx_.device(), &wi, kTimeoutNs);
    if (r == VK_TIMEOUT) {
        throw DeviceError(std::format(
            "timed out after 30s waiting for GPU timeline value {} -- a shader has most likely "
            "hung. Re-run with validation enabled.",
            value));
    }
    check(r, "vkWaitSemaphores");
    resolve_completed();
}

void Recorder::wait_idle() {
    if (timeline_value_ > 0) {
        wait(timeline_value_);
    }
}

// ---------------------------------------------------------------------------

StagingBuffer::StagingBuffer(Context& ctx, Allocator& allocator, Recorder& recorder,
                             uint64_t capacity)
    : ctx_(ctx), allocator_(allocator), recorder_(recorder) {
    staging_ = allocator_.allocate(capacity, MemoryKind::HostStaging, "staging");
    VKML_ASSERT(staging_.mapped != nullptr, "staging allocation is not host-mapped");
    VKML_LOG_DEBUG("vulkan staging buffer: {} MiB", capacity / (1024 * 1024));
}

StagingBuffer::~StagingBuffer() { allocator_.free(staging_); }

void StagingBuffer::upload(const void* src, const Allocation& dst, uint64_t dst_offset,
                           uint64_t bytes) {
    if (bytes == 0) {
        return;
    }
    const auto* bytes_in = static_cast<const std::byte*>(src);
    const VkBuffer dst_buffer = allocator_.buffer_of(dst);
    const VkBuffer staging_buffer = allocator_.buffer_of(staging_);

    // Chunked so an arbitrarily large tensor can be uploaded through a fixed
    // staging buffer.
    uint64_t done = 0;
    while (done < bytes) {
        const uint64_t chunk = std::min(bytes - done, staging_.size);

        // Wait for the PREVIOUS use of the staging memory, immediately before
        // overwriting it -- not after submitting, which is where this wait used
        // to be.
        //
        // The distinction did not matter while every submission blocked. It
        // matters now: the timeline is monotonic, so waiting on this copy's own
        // ticket drains EVERY submission made before it, and an upload in the
        // middle of a training step therefore drained the whole step. Measured:
        // one 200 KiB upload cost 109 us on an idle device and 477 us behind a
        // step's queued work.
        //
        // Moving it here is not a weaker guarantee. What the wait protects is
        // the staging buffer against being rewritten while a copy still reads
        // it, and that is exactly what waiting before the memcpy establishes. A
        // single-chunk upload -- every tensor smaller than the staging buffer,
        // which is the common case -- now blocks not at all.
        recorder_.wait(staging_ticket_);
        std::memcpy(staging_.mapped, bytes_in + done, chunk);

        recorder_.begin();
        recorder_.copy(staging_buffer, staging_.offset, dst_buffer, dst.offset + dst_offset + done,
                       chunk);
        // The copy must be visible to any shader that reads this buffer next.
        recorder_.barrier();
        staging_ticket_ = recorder_.submit();
        done += chunk;
    }
}

void StagingBuffer::download(void* dst, const Allocation& src, uint64_t src_offset,
                             uint64_t bytes) {
    if (bytes == 0) {
        return;
    }
    auto* bytes_out = static_cast<std::byte*>(dst);
    const VkBuffer src_buffer = allocator_.buffer_of(src);
    const VkBuffer staging_buffer = allocator_.buffer_of(staging_);

    uint64_t done = 0;
    while (done < bytes) {
        const uint64_t chunk = std::min(bytes - done, staging_.size);

        recorder_.begin();
        // Ordered against any shader still writing the source.
        recorder_.barrier();
        recorder_.copy(src_buffer, src.offset + src_offset + done, staging_buffer, staging_.offset,
                       chunk);
        const uint64_t ticket = recorder_.submit();

        // THE synchronisation point, and the only one that cannot be deferred:
        // the memcpy below reads host memory the GPU is filling. Submission is
        // asynchronous, so the data may be produced by a submission nobody has
        // waited on; this copy's ticket covers all of them, because the timeline
        // is monotonic and this copy was submitted last.
        //
        // It therefore also covers the staging buffer's own previous use, which
        // is what `staging_ticket_` guards on the upload path.
        recorder_.wait(ticket);
        staging_ticket_ = ticket;

        // Host-coherent memory needs no invalidate before reading.
        std::memcpy(bytes_out + done, staging_.mapped, chunk);
        done += chunk;
    }
}

}  // namespace vkml::vk
