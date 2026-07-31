#include "vkml/util/env.h"

#include <cstdlib>

namespace vkml {

std::optional<std::string> env_value(const char* name) {
    if (name == nullptr) {
        return std::nullopt;
    }

    // THE ONLY std::getenv IN THE PROJECT. Everything else goes through the
    // helpers above, which return owned strings.
    //
    // MSVC reports C4996 here: its secure-CRT policy deprecates getenv and
    // suggests getenv_s or _dupenv_s, both Windows-only. std::getenv is
    // standard C++ and is not deprecated by ISO, so the portable code stays and
    // the suppression is confined to this one call rather than spread across
    // eighteen files or applied project-wide, where it would also silence a
    // future C4996 that did matter.
    //
    // The hazard the warning names is real and is answered by the copy below,
    // not by the suppression: the returned pointer is into the environment
    // block and a putenv from any thread may invalidate it, so the value is
    // copied before returning and the pointer never escapes this function.
#ifdef _MSC_VER
#    pragma warning(push)
#    pragma warning(disable : 4996)
#endif
    const char* raw = std::getenv(name);
#ifdef _MSC_VER
#    pragma warning(pop)
#endif

    if (raw == nullptr) {
        return std::nullopt;
    }
    return std::string(raw);
}

bool parse_env_flag(const char* raw, bool fallback) noexcept {
    if (raw == nullptr || raw[0] == '\0') {
        return fallback;
    }
    // Only the first character is examined, which is what every call site did
    // before this was centralised: "0" is off, anything else is on. Kept rather
    // than tightened, because a switch that started rejecting "false" or "no"
    // would change behaviour for anyone already setting them.
    return raw[0] != '0';
}

int64_t parse_env_int(const char* raw, int64_t fallback) noexcept {
    if (raw == nullptr || raw[0] == '\0') {
        return fallback;
    }
    // strtol rather than atoi: atoi has undefined behaviour on overflow, and
    // these values come from a user's shell.
    return static_cast<int64_t>(std::strtol(raw, nullptr, 10));
}

bool env_flag(const char* name, bool fallback) {
    const std::optional<std::string> value = env_value(name);
    return parse_env_flag(value ? value->c_str() : nullptr, fallback);
}

int64_t env_int(const char* name, int64_t fallback) {
    const std::optional<std::string> value = env_value(name);
    return parse_env_int(value ? value->c_str() : nullptr, fallback);
}

}  // namespace vkml
