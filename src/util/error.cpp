#include "vkml/util/error.h"

#include <filesystem>

namespace vkml::detail {

std::string format_internal(std::string_view file, int line, std::string_view func,
                            std::string_view cond, std::string_view msg) {
    // Trim to the basename: absolute build paths are noise in an exception
    // message, and the repo-relative name is enough to locate the check.
    const std::string_view name = [file] {
        const auto pos = file.find_last_of("/\\");
        return pos == std::string_view::npos ? file : file.substr(pos + 1);
    }();

    if (msg.empty()) {
        return std::format("internal error at {}:{} in {}: assertion failed: {}", name, line, func,
                           cond);
    }
    return std::format("internal error at {}:{} in {}: {} (assertion failed: {})", name, line, func,
                       msg, cond);
}

}  // namespace vkml::detail
