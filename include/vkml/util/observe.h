#pragma once

#include <cstdint>
#include <functional>
#include <string_view>

/// Decision facts: what the engine chose, instead of what, and why.
///
/// WHY THIS EXISTS
/// ---------------
/// The engine already knows why it does what it does, and cannot say so. When
/// the register-blocked GEMM does not fit, `vulkan_backend.cpp` computes the
/// reason and the numbers, formats them as English, and throws them at stderr —
/// where nothing can assert on them, aggregate them, attribute them to an
/// operation, or read them at all under pytest, which captures stderr.
///
/// The cost of that is not hypothetical. `tests/python/vkvalidate.py` cannot ask
/// the backend whether it fell back, so it reimplements the backend's selection
/// rule as `max_workgroup_invocations < 256` — a second copy of a decision the
/// backend already owns, which the two can silently disagree about. That is
/// category 3 in docs/ENGINEERING-PRINCIPLES.md, inside the test suite, and it
/// was the only option available.
///
/// THE ONE RULE THIS HEADER ENFORCES BY ITS SHAPE
/// ---------------------------------------------
/// A decision site publishes a fact and learns nothing in return. It does not
/// know whether anyone is listening, where the fact goes, or how it is rendered.
/// There is no recorder type here, no handle to pass, no consumer to include:
///
///     Decision site -> Fact -> Recorder -> snapshot / log / CI / user queries
///
/// and NOT `Decision site -> Recorder -> everything else`. If a backend ever
/// includes a recorder header or branches on whether recording is on, the
/// architecture has been inverted (docs/OBSERVABILITY-ARCHITECTURE.md 4a).
///
/// The practical test: deleting every consumer must still compile. `publish` is
/// a call into layer 0 that may go nowhere.
///
/// Modelled on `util/log.h` (host-settable sink) and `util/coverage.h`
/// (atomic-gated publication at the point of execution). Those two already prove
/// the shape works across the layer boundary; this is the third fact type, not a
/// third mechanism.
///
/// COST, stated accurately and not optimistically. `publish` takes a mutex and
/// runs the library's own notification policy on every call — there is no
/// subscriber-absent fast path, because the default renderer is always a
/// consumer. An earlier revision of this comment claimed "one relaxed atomic
/// load when nothing is subscribed"; that was true for exactly one increment and
/// became false the moment a default consumer existed. It is corrected here
/// rather than left to be discovered, since a comment that overstates cheapness
/// is how an unmeasured cost gets accepted.
///
/// Whether that price is acceptable is a MEASUREMENT, not a claim: it is
/// baselined and gated like any other benchmark, because being internal is a
/// reason for more discipline rather than less
/// (docs/OBSERVABILITY-ARCHITECTURE.md 8, tracker #117).
namespace vkml::observe {

/// One choice, reduced to what makes it answerable and checkable.
///
/// Deliberately plain values and `string_view`s: this header sits in `util`, the
/// lowest layer, so `backend/vulkan`, `dispatch` and `api` may all include it
/// (scripts/check_layering.py). Nothing here owns storage — a subscriber that
/// keeps a fact past the call must copy it.
struct Decision {
    /// Where the choice was made: "matmul.kernel", "allocator.memory_kind".
    /// Dotted, subsystem-first, so decisions group by the thing that owns them.
    std::string_view site{};

    /// The operation this was decided FOR, in the user's vocabulary — "matmul",
    /// not a pipeline key. Empty for process-wide choices. A user asks about the
    /// operation they wrote; the internal name belongs in `chose`, underneath.
    std::string_view op{};

    std::string_view chose{};       ///< "gemm_naive"
    std::string_view instead_of{};  ///< "gemm_reg" — empty when nothing was rejected
    std::string_view because{};     ///< "needs more invocations than the device allows"

    /// The numbers that forced it, and the reason a Decision is falsifiable
    /// rather than merely present. `required` 256 against `available` 128 is
    /// checkable against what the driver independently reports about the
    /// compiled pipeline; a prose reason is not.
    ///
    /// Both zero when the choice was not driven by a limit.
    int64_t required = 0;
    int64_t available = 0;
};

/// True when at least one subscriber is installed. Decision sites should not
/// call this — `publish` is already cheap and checking twice invites a site to
/// grow a branch, which is how coupling starts. It exists for tests.
[[nodiscard]] bool enabled() noexcept;

/// Publishes one decision. Costs a relaxed atomic load when nobody is listening.
///
/// Safe to call from any thread and during static destruction. Never throws: a
/// subscriber that throws would turn observation into a failure mode of the
/// thing observed, which is the opposite of the point.
void publish(const Decision& d) noexcept;

/// Installs the subscriber. One at a time, replacing any previous: fan-out is a
/// consumer's problem, not this layer's, and a subscriber list here would be the
/// first step towards the logging framework that
/// docs/OBSERVABILITY-ARCHITECTURE.md 9 exists to forbid.
///
/// Passing `nullptr` removes it, which is what makes "delete every consumer and
/// it still compiles" true rather than aspirational.
using Subscriber = std::function<void(const Decision&)>;
void subscribe(Subscriber s);

}  // namespace vkml::observe
