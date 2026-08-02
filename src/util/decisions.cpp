#include "vkml/util/decisions.h"

#include "vkml/util/observe.h"

#include <deque>
#include <mutex>

namespace vkml::observe {
namespace {

struct Window {
    std::deque<RecordedDecision> ring;
    size_t capacity = 0;
    uint64_t published = 0;
    bool on = false;
};

// One mutex, and it is NOT observe.cpp's. Sharing that one would make this file
// aware of the publisher's locking, and the two would have to be reasoned about
// together forever. `publish` already holds its own lock when it calls the
// subscriber, so this is only ever contended by a reader.
std::mutex& lock() noexcept {
    static std::mutex m;
    return m;
}

Window& window() noexcept {
    static Window w;
    return w;
}

}  // namespace

void start_recording(size_t capacity) {
    {
        std::lock_guard<std::mutex> g(lock());
        Window& w = window();
        w.ring.clear();
        w.capacity = capacity == 0 ? 1 : capacity;
        w.published = 0;
        w.on = true;
    }
    // Subscribing LAST, so the window can never receive a decision before it is
    // ready to hold one. The mirror of observe::subscribe clearing its flag
    // before tearing the sink down.
    subscribe([](const Decision& d) {
        std::lock_guard<std::mutex> g(lock());
        Window& w = window();
        if (!w.on) {
            return;
        }
        ++w.published;
        if (w.ring.size() == w.capacity) {
            w.ring.pop_front();
        }
        w.ring.push_back(RecordedDecision{
            std::string(d.site), std::string(d.op), std::string(d.chose), std::string(d.instead_of),
            std::string(d.because), d.required, d.available, w.published, d.dispatch});
    });
}

void stop_recording() {
    subscribe(nullptr);
    std::lock_guard<std::mutex> g(lock());
    Window& w = window();
    w.on = false;
    w.ring.clear();
    w.capacity = 0;
}

bool recording() noexcept {
    std::lock_guard<std::mutex> g(lock());
    return window().on;
}

std::vector<RecordedDecision> recorded() {
    std::lock_guard<std::mutex> g(lock());
    const Window& w = window();
    return std::vector<RecordedDecision>(w.ring.begin(), w.ring.end());
}

uint64_t published() noexcept {
    std::lock_guard<std::mutex> g(lock());
    return window().published;
}

}  // namespace vkml::observe
