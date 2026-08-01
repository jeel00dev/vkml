#pragma once

#include <cstdint>
#include <string>
#include <vector>

/// The decision recorder: a bounded window on what the engine recently chose.
///
/// WHY THIS EXISTS, and why it is not part of `observe.h`. `observe.h` is the
/// fact model — decision sites publish into it and it holds nothing. Somebody
/// still has to keep the facts so a consumer can read them, and without this
/// every consumer would install its own subscriber and maintain its own buffer:
/// the test suite, the benchmark harness and the Python API would each grow a
/// private copy of the same ring. That is the duplication this removes.
///
/// WHICH PATTERN IT EXTENDS. `util/coverage.h` — a layer-0 consumer that records
/// what actually happened, stores it in memory, and exposes it for inspection.
/// Same layer, same shape, different fact type. This is deliberately not a new
/// mechanism (docs/OBSERVABILITY-ARCHITECTURE.md 3-4).
///
/// IT IS A CONSUMER, AND NOTHING KNOWS IT EXISTS. No decision site includes this
/// header; publication goes through `observe::publish` and may go nowhere. The
/// architectural test is that deleting this file and its binding leaves the
/// library compiling and the decision sites untouched (section 4a).
///
/// BOUNDED, ALWAYS. A ring of fixed capacity, oldest dropped. Unbounded history
/// would make this a tracing system, which section 9 forbids — and the boundary
/// rule there is that observability records decisions, never data.
///
/// OFF BY DEFAULT. Recording is started explicitly. The design argues in section
/// 7.2 that a user asks "why did that happen" only AFTER being surprised, which
/// wants an always-on ring; whether that is affordable is a measurement not yet
/// taken (tracker #117), so the conservative default stands until it is.
namespace vkml::observe {

/// A decision, copied. `Decision` in observe.h carries `string_view`s that point
/// at the caller's literals; anything outliving the publish call must own its
/// strings, which is the one thing this type adds.
struct RecordedDecision {
    std::string site;
    std::string op;
    std::string chose;
    std::string instead_of;
    std::string because;
    int64_t required = 0;
    int64_t available = 0;

    /// Publication order, from 1. Survives eviction, so a reader can tell "the
    /// oldest I kept" from "the first that happened" — the distinction that
    /// makes a bounded window honest about what it dropped.
    uint64_t seq = 0;
};

/// Begins recording, replacing any previous window. Installs the subscriber.
void start_recording(size_t capacity = 256);

/// Stops recording and releases the window. Safe when not recording.
void stop_recording();

[[nodiscard]] bool recording() noexcept;

/// The window, oldest first. Empty when not recording.
[[nodiscard]] std::vector<RecordedDecision> recorded();

/// How many decisions were published since `start_recording`, INCLUDING any the
/// ring evicted. `dropped() == published() - recorded().size()`.
///
/// Reported rather than inferred, because a consumer that cannot tell a full
/// window from a complete one will silently draw conclusions from a truncated
/// history — the same failure as a gate that cannot say "I could not verify".
[[nodiscard]] uint64_t published() noexcept;

}  // namespace vkml::observe
