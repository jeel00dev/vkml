#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace vkml {

/// Reading vkml's environment switches, in one place.
///
/// WHY THIS EXISTS. Eighteen call sites each wrote their own three-line
/// `std::getenv` idiom. That duplicated a rule eighteen times, and it hid a
/// lifetime bug: one site stored the RETURNED POINTER in a function-local
/// static and dereferenced it on every later dispatch. `getenv` returns a
/// pointer into the environment block, which a `putenv` from any thread may
/// invalidate — vkml is embedded in Python, where `os.environ[...] = ...` does
/// exactly that. Returning an owned `std::string` makes the question
/// structurally impossible rather than merely unlikely (issue #23).
///
/// It is also the single place MSVC's C4996 has to be answered. `std::getenv`
/// is standard C++ and is not deprecated by ISO; MSVC deprecates it under its
/// secure-CRT policy and suggests `getenv_s` or `_dupenv_s`, both of which are
/// Windows-only. Rather than take a `#ifdef` into eighteen files, `getenv` is
/// called once, here, and the suppression sits beside the reasoning.
///
/// READ-ONCE, BY CONTRACT. Every caller reads its switch into a
/// function-local static at first use, so these are CONFIGURATION fixed for the
/// life of the process, not a control interface. Changing one mid-run has no
/// defined effect: pipelines are already selected and cached from the earlier
/// value. Documented in README.md alongside the variables themselves.

/// The raw value of `name`, or `nullopt` when the variable is not set.
///
/// An empty variable returns an empty string rather than `nullopt`, preserving
/// `getenv`'s own distinction between unset and set-but-empty — one caller
/// tests presence alone, and collapsing the two would change it.
[[nodiscard]] std::optional<std::string> env_value(const char* name);

/// Whether `raw` reads as "on": set, non-empty, and not starting with '0'.
///
/// Split from the lookup so the rule is testable without touching the
/// environment, which is otherwise process-global state that tests would have
/// to mutate and restore. `nullptr` means unset.
[[nodiscard]] bool parse_env_flag(const char* raw, bool fallback) noexcept;

/// `raw` as a base-10 integer, or `fallback` when unset or empty.
///
/// Deliberately does NOT clamp: the callers' valid ranges differ (one caps at
/// 256, another requires more than 1), so each keeps its own check where the
/// reason for it is visible.
[[nodiscard]] int64_t parse_env_int(const char* raw, int64_t fallback) noexcept;

/// `parse_env_flag` applied to `name`'s value.
/// One environment switch, as actually read. NOT a declared list of switches:
/// this is populated by `env_value` at the moment each one is consulted, so it
/// cannot drift from the code that reads them. A hand-maintained registry would
/// be a second model of vkml's configuration and would go stale the first time
/// somebody added a switch without updating it.
struct ObservedSwitch {
    std::string name;
    std::string value;  ///< empty when unset
    bool set = false;   ///< distinguishes unset from set-to-empty
};

/// Every switch this process has consulted, sorted by name.
///
/// WHY THIS EXISTS. Eighteen variables change what vkML does and none of them
/// was visible in a running process. `VKML_GEMM_NOVEC=1` costs a measured 14.6%
/// on a 1024-cubed matmul and left no trace anywhere — the exact signature of
/// issue #76, where a "regression" turned out to be the execution environment
/// rather than the code. A baseline recorded with one exported is wrong forever
/// and says nothing about why (docs/ENGINEERING-PRINCIPLES.md 4).
///
/// Observed, not declared: an entry appears because something read it.
[[nodiscard]] std::vector<ObservedSwitch> observed_environment();

[[nodiscard]] bool env_flag(const char* name, bool fallback = false);

/// `parse_env_int` applied to `name`'s value.
[[nodiscard]] int64_t env_int(const char* name, int64_t fallback);

}  // namespace vkml
