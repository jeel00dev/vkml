#pragma once

#include <cstdint>
#include <string>
#include <string_view>

/// Records what the test suite actually executes, so coverage can be measured
/// rather than assumed.
///
/// WHY THIS EXISTS
/// ---------------
/// Every milestone added tests for the thing it built. That answers "is the new
/// code tested" and never answers "what is *not* tested" -- and the second
/// question is the one that finds defects. A green suite of a thousand tests
/// says nothing about which operator has only ever been run at rank 2, on f32,
/// with contiguous inputs, on one backend.
///
/// The honest way to answer it is to record what runs. Reading test names
/// cannot: a test called `test_add` may exercise one shape, and a composite
/// operator like conv2d dispatches im2col and matmul under a name that mentions
/// neither. Recording at the point where nodes are actually evaluated sees
/// through both.
///
/// REJECTED ALTERNATIVE. `VKML_VULKAN_DEBUG` already logs one line per Vulkan
/// dispatch, so the log could be parsed instead of adding this. It was not,
/// because that trace is Vulkan-only -- it cannot see the CPU backend, which is
/// the oracle half of the correctness chain -- and it carries the kernel and its
/// tuning, not the dtype, rank or input layout that decide whether a code path
/// was reached.
///
/// WHAT THIS DOES NOT MEASURE, stated because a coverage number that overstates
/// itself is worse than none: it records what was EXECUTED, not what was
/// ASSERTED. A test that dispatches an operator and checks nothing still counts
/// here. That gap is covered by two other gates, and only the three together
/// mean anything -- `scripts/mutation_check.py` shows the suite can fail, and
/// the assertion audit (docs/MILESTONE-B-REVIEW.md 3.1) shows no test asserts
/// nothing. This one shows what the suite reaches.
///
/// Off unless `VKML_COVERAGE` is set, so it costs one relaxed atomic load per
/// evaluated node when unused.
namespace vkml::coverage {

/// One evaluated node, reduced to the properties that decide whether a distinct
/// code path was taken. Deliberately plain values rather than a `Node`: this
/// header sits in `util`, the lowest layer, so that both `dispatch` and
/// `autograd` may include it (scripts/check_layering.py).
struct Dispatch {
    std::string_view op;
    std::string_view backend;  ///< "cpu", "vulkan", or "view" for a zero-copy op
    std::string_view dtype;
    int ndim = 0;

    /// Element count of the LARGEST tensor the operator touched, output or
    /// input. Not the output alone: a reduction over 1517 elements produces 37,
    /// and classifying it by its output would report a kernel that spans six
    /// workgroups as a single-group run -- which is exactly backwards, since the
    /// span is what the size axis exists to detect.
    int64_t work_numel = 0;

    /// The output's own element count, which is what "empty" and "scalar" mean.
    int64_t numel = 0;
    int n_src = 0;                 ///< distinguishes creation ops, which cannot have strided input
    bool broadcast_input = false;  ///< some source has a zero stride, so aliases
    bool strided_input = false;    ///< some source is non-contiguous
    bool strided_output = false;   ///< this node's own layout is non-contiguous; only views can be
};

[[nodiscard]] bool enabled() noexcept;

/// Records one evaluated node. Observations are deduplicated in memory and
/// counted, so a suite that dispatches millions of nodes writes a small file.
void record(const Dispatch& d);

/// Records that a backward RULE fired, which is a different question from which
/// kernels ran. Because every backward rule is written in terms of forward
/// operations (`autograd.h`), a gradient test shows up in `record` as more of
/// the same forward ops -- so the rules need their own axis or an untested one
/// is invisible.
void record_backward_rule(std::string_view op);

/// Writes every distinct observation to `path`, one per line, tab separated.
/// Returns the number of lines written. Overwrites.
size_t dump(const std::string& path);

void clear() noexcept;

}  // namespace vkml::coverage
