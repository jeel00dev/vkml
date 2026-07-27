#include "vkml/util/log.h"

#include <atomic>
#include <cstdio>
#include <mutex>

namespace vkml {
namespace {

/// Function-local statics rather than namespace-scope globals: this guarantees
/// initialisation on first use and sidesteps the static initialisation order
/// fiasco if anything logs from a constructor during static init.
std::atomic<LogLevel>& level_storage() noexcept {
    static std::atomic<LogLevel> level{
#ifdef NDEBUG
        LogLevel::Info
#else
        LogLevel::Debug
#endif
    };
    return level;
}

std::mutex& callback_mutex() noexcept {
    static std::mutex m;
    return m;
}

LogCallback& callback_storage() noexcept {
    static LogCallback cb;  // empty == use the default stderr sink
    return cb;
}

void default_sink(LogLevel level, std::string_view message) noexcept {
    // Single fwrite-style call per message so that concurrent writers interleave
    // whole lines rather than fragments.
    std::fprintf(stderr, "[vkml %.*s] %.*s\n", static_cast<int>(to_string(level).size()),
                 to_string(level).data(), static_cast<int>(message.size()), message.data());
}

}  // namespace

std::string_view to_string(LogLevel level) noexcept {
    switch (level) {
        case LogLevel::Trace: return "trace";
        case LogLevel::Debug: return "debug";
        case LogLevel::Info:  return "info";
        case LogLevel::Warn:  return "warn";
        case LogLevel::Error: return "error";
        case LogLevel::Off:   return "off";
    }
    return "?";
}

void set_log_level(LogLevel level) noexcept {
    level_storage().store(level, std::memory_order_relaxed);
}

LogLevel log_level() noexcept {
    return level_storage().load(std::memory_order_relaxed);
}

void set_log_callback(LogCallback cb) {
    const std::lock_guard<std::mutex> lock(callback_mutex());
    callback_storage() = std::move(cb);
}

namespace detail {

bool log_enabled(LogLevel level) noexcept {
    return static_cast<int>(level) >= static_cast<int>(log_level());
}

void log_write(LogLevel level, std::string_view message) {
    const std::lock_guard<std::mutex> lock(callback_mutex());
    if (callback_storage()) {
        callback_storage()(level, message);
    } else {
        default_sink(level, message);
    }
}

}  // namespace detail
}  // namespace vkml
