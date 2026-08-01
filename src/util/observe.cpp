#include "vkml/util/observe.h"

#include "vkml/util/log.h"

#include <atomic>
#include <mutex>
#include <set>
#include <string>

namespace vkml::observe {
namespace {

/// Two pieces of state, deliberately, and the atomic is the reason.
///
/// `publish` is called from the dispatch path. Taking a mutex there when nobody
/// is listening would make observation cost something in the overwhelmingly
/// common case, and the cost of observation is a measured property of this file
/// (docs/OBSERVABILITY-ARCHITECTURE.md 8) rather than something to be casual
/// about. So the fast path is one relaxed load of a bool, exactly as
/// `coverage::enabled()` does.
///
/// The mutex only guards installation and the call itself, which happens when a
/// subscriber exists and has therefore already accepted the cost.
std::atomic<bool>& active() noexcept {
    static std::atomic<bool> flag{false};
    return flag;
}

std::mutex& lock() noexcept {
    static std::mutex m;
    return m;
}

Subscriber& sink() noexcept {
    static Subscriber s;
    return s;
}

/// The default consumer: renders a decision to the log, once per distinct fact.
///
/// It is a CONSUMER, not part of the model — `observe.h` knows nothing about
/// logging, and this function could be deleted without the header changing.
/// It lives here for the same reason `log.cpp` carries its own `default_sink`:
/// a facility whose default behaviour requires a separate library to link is a
/// facility that silently does nothing in the build that needed it most.
///
/// DEDUPLICATION IS PRESENTATION, and this is where it belongs. The backend
/// used to carry a `reported_gemm_fallback` bool because logging the fallback
/// per dispatch produced 4,785 identical lines in one MNIST epoch. That is a
/// true observation about *reading logs*, not about the decision — the decision
/// really is made on every dispatch. Keeping the flag at the decision site made
/// the site responsible for how often a reader wants to hear about it, which is
/// not its business. The site now publishes every time; this decides what is
/// worth saying.
void log_once(const Decision& d) {
    static std::set<std::string> said;
    std::string key(d.site);
    key += '\v';
    key += d.chose;
    key += '\v';
    key += d.instead_of;
    if (!said.insert(key).second) {
        return;
    }
    if (d.required != 0 || d.available != 0) {
        VKML_LOG_INFO("{}: chose {} instead of {} — {} (needs {}, device allows {})", d.site,
                      d.chose, d.instead_of, d.because, d.required, d.available);
    } else {
        VKML_LOG_INFO("{}: chose {} instead of {} — {}", d.site, d.chose, d.instead_of, d.because);
    }
}

}  // namespace

bool enabled() noexcept { return active().load(std::memory_order_relaxed); }

void publish(const Decision& d) noexcept {
    // BOTH consumers run, and an earlier version of this ran only one.
    //
    // The first draft treated a subscriber as REPLACING the default renderer,
    // which meant that installing the recorder silenced the log. Two renderings
    // of one fact are independent by design (OBSERVABILITY-ARCHITECTURE 6) and
    // making them exclusive is the same mistake as the log being load-bearing:
    // a consumer's presence would change what a user sees.
    //
    // Fan-out stays trivial -- one policy renderer plus one installed consumer
    // -- because a general subscriber list here is the first step towards the
    // logging framework section 9 forbids.
    // noexcept, and this is where that is earned. A subscriber that throws must
    // not become a failure mode of the operation being observed -- a matmul that
    // fails because something was watching it is worse than no observability.
    try {
        std::lock_guard<std::mutex> g(lock());
        log_once(d);
        if (sink()) {
            sink()(d);
        }
    } catch (...) {
    }
}

void subscribe(Subscriber s) {
    std::lock_guard<std::mutex> g(lock());
    sink() = std::move(s);
    // Ordering matters: the flag is set only after the sink is in place, and
    // cleared before it is torn down, so publish() never sees a live flag with a
    // dead sink.
    active().store(static_cast<bool>(sink()), std::memory_order_relaxed);
}

}  // namespace vkml::observe
