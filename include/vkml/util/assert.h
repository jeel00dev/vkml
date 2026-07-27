#pragma once

#include "vkml/util/error.h"

#include <format>

// Assertion vocabulary
// --------------------
// Three macros, three distinct meanings. Keeping them distinct matters because
// they have different costs and different audiences:
//
//   VKML_CHECK        user error. Active in every build. Throws the exception
//                     type you name, so Python sees ValueError/TypeError/... .
//                     Use for anything reachable from a public API with bad
//                     arguments: shape mismatch, wrong dtype, bad axis.
//
//   VKML_ASSERT       internal invariant. Active in every build, including
//                     Release, and throws InternalError. Use where a failure
//                     means vkml itself is broken. Kept on in Release
//                     deliberately: a silently corrupted tensor is far more
//                     expensive to debug than a branch is to execute.
//
//   VKML_DEBUG_ASSERT internal invariant on a hot path. Compiled out unless
//                     NDEBUG is absent. Use inside per-element loops, where an
//                     always-on check would actually show up in a profile.
//
// ggml has a single GGML_ASSERT that aborts. Splitting user error from internal
// error is what lets the Python layer raise something meaningful instead of a
// generic RuntimeError for every failure.

#define VKML_CHECK(cond, ExcType, ...)                                                             \
    do {                                                                                           \
        if (!(cond)) [[unlikely]] {                                                                \
            throw ::vkml::ExcType(std::format(__VA_ARGS__));                                       \
        }                                                                                          \
    } while (0)

#define VKML_ASSERT(cond, ...)                                                                     \
    do {                                                                                           \
        if (!(cond)) [[unlikely]] {                                                                \
            throw ::vkml::InternalError(::vkml::detail::format_internal(                           \
                __FILE__, __LINE__, static_cast<const char*>(__func__), #cond,                     \
                std::format(__VA_ARGS__)));                                                        \
        }                                                                                          \
    } while (0)

#ifdef NDEBUG
#    define VKML_DEBUG_ASSERT(cond, ...) ((void)0)
#else
#    define VKML_DEBUG_ASSERT(cond, ...) VKML_ASSERT(cond, __VA_ARGS__)
#endif

/// Marks a branch that should be unreachable given the surrounding invariants.
#define VKML_UNREACHABLE(...)                                                                      \
    throw ::vkml::InternalError(::vkml::detail::format_internal(                                   \
        __FILE__, __LINE__, static_cast<const char*>(__func__), "unreachable",                     \
        std::format(__VA_ARGS__)))

/// Marks a path that is designed but not yet built.
#define VKML_NOT_IMPLEMENTED(...) throw ::vkml::NotImplementedError(std::format(__VA_ARGS__))
