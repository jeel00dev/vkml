#pragma once

#include <format>
#include <functional>
#include <string_view>

// Logging
// -------
// A leveled logger with a host-settable sink, modelled on ggml_log_set. The
// callback indirection exists for one concrete reason: when vkml is driven from
// Python, log output should be routable into the `logging` module rather than
// splattered onto stderr from a background thread. Tests also use it to capture
// and assert on log output.
//
// On global mutable state: a logger is inherently process-global. It is
// confined here to two atomics and one callback behind accessor functions, and
// nothing else in the codebase is permitted to hold mutable globals.
//
// Log levels below VKML_LOG_MIN_LEVEL are removed by the preprocessor, so
// Trace/Debug calls cost literally nothing in Release. Argument expressions are
// not evaluated when a level is disabled at runtime either.

#ifndef VKML_LOG_MIN_LEVEL
#    ifdef NDEBUG
#        define VKML_LOG_MIN_LEVEL 2  // Info
#    else
#        define VKML_LOG_MIN_LEVEL 0  // Trace
#    endif
#endif

namespace vkml {

enum class LogLevel : int {
    Trace = 0,
    Debug = 1,
    Info  = 2,
    Warn  = 3,
    Error = 4,
    Off   = 5,
};

[[nodiscard]] std::string_view to_string(LogLevel level) noexcept;

/// Messages below this level are dropped. Default: Info (Release) / Debug (Debug).
void set_log_level(LogLevel level) noexcept;

[[nodiscard]] LogLevel log_level() noexcept;

/// Receives every message that passes the level filter. Must be thread-safe.
using LogCallback = std::function<void(LogLevel, std::string_view)>;

/// Replaces the sink. Passing nullptr restores the default stderr sink.
void set_log_callback(LogCallback callback);

namespace detail {

[[nodiscard]] bool log_enabled(LogLevel level) noexcept;

void log_write(LogLevel level, std::string_view message);

}  // namespace detail
}  // namespace vkml

#define VKML_LOG_AT(level, ...)                                                                    \
    do {                                                                                           \
        if (::vkml::detail::log_enabled(level)) {                                                  \
            ::vkml::detail::log_write(level, std::format(__VA_ARGS__));                            \
        }                                                                                          \
    } while (0)

#if VKML_LOG_MIN_LEVEL <= 0
#    define VKML_LOG_TRACE(...) VKML_LOG_AT(::vkml::LogLevel::Trace, __VA_ARGS__)
#else
#    define VKML_LOG_TRACE(...) ((void)0)
#endif

#if VKML_LOG_MIN_LEVEL <= 1
#    define VKML_LOG_DEBUG(...) VKML_LOG_AT(::vkml::LogLevel::Debug, __VA_ARGS__)
#else
#    define VKML_LOG_DEBUG(...) ((void)0)
#endif

#if VKML_LOG_MIN_LEVEL <= 2
#    define VKML_LOG_INFO(...) VKML_LOG_AT(::vkml::LogLevel::Info, __VA_ARGS__)
#else
#    define VKML_LOG_INFO(...) ((void)0)
#endif

#if VKML_LOG_MIN_LEVEL <= 3
#    define VKML_LOG_WARN(...) VKML_LOG_AT(::vkml::LogLevel::Warn, __VA_ARGS__)
#else
#    define VKML_LOG_WARN(...) ((void)0)
#endif

#if VKML_LOG_MIN_LEVEL <= 4
#    define VKML_LOG_ERROR(...) VKML_LOG_AT(::vkml::LogLevel::Error, __VA_ARGS__)
#else
#    define VKML_LOG_ERROR(...) ((void)0)
#endif
