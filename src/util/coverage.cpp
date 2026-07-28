#include "vkml/util/coverage.h"

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>

namespace vkml::coverage {
namespace {

/// Size classes, chosen against real boundaries in this codebase rather than
/// round numbers. A kernel that has only ever run inside one workgroup has not
/// exercised its cross-group indexing, which is where off-by-ones live.
///
/// 256 is the default compute workgroup size (src/backend/vulkan/vk_pipeline.h).
/// If that default changes, this classification should follow it; the boundary
/// is the point, not the number.
constexpr int64_t kWorkgroup = 256;

[[nodiscard]] std::string_view size_class(int64_t numel) noexcept {
    if (numel == 0) {
        return "empty";
    }
    if (numel == 1) {
        return "scalar";
    }
    if (numel <= kWorkgroup) {
        return "one_group";
    }
    return "many_groups";
}

/// The observation store. A map rather than a hash set so the dump is sorted
/// and therefore diffable between runs -- a coverage report that reorders itself
/// is unreadable as a diff, and comparing two runs is the main way this gets
/// used after the first time.
struct Store {
    std::mutex mutex;
    std::map<std::string, uint64_t> counts;
};

Store& store() {
    static Store s;
    return s;
}

void bump(std::string key) {
    Store& s = store();
    const std::lock_guard<std::mutex> lock(s.mutex);
    ++s.counts[std::move(key)];
}

}  // namespace

bool enabled() noexcept {
    // Read once. The suite sets this before the process starts, and re-reading
    // the environment per node would cost more than the recording does.
    static const std::atomic<bool> flag{[] {
        const char* env = std::getenv("VKML_COVERAGE");
        return env != nullptr && env[0] != '\0';
    }()};
    return flag.load(std::memory_order_relaxed);
}

void record(const Dispatch& d) {
    std::string key = "dispatch\t";
    key += d.op;
    key += '\t';
    key += d.backend;
    key += '\t';
    key += d.dtype;
    key += "\trank";
    key += static_cast<char>('0' + (d.ndim < 0 || d.ndim > 9 ? 9 : d.ndim));
    key += '\t';
    // "empty" and "scalar" are properties of the OUTPUT -- an empty result is
    // the edge case, whatever its inputs were. The two larger classes are
    // properties of the WORK, so that a reduction is classified by what it
    // reads rather than by what it returns.
    key += (d.numel <= 1) ? size_class(d.numel) : size_class(d.work_numel);
    key += "\tsrc";
    key += static_cast<char>('0' + (d.n_src < 0 || d.n_src > 9 ? 9 : d.n_src));
    key += d.broadcast_input ? "\tbroadcast" : "\t-";
    key += d.strided_input ? "\tstrided_in" : "\t-";
    key += d.strided_output ? "\tstrided_out" : "\t-";
    bump(std::move(key));
}

void record_backward_rule(std::string_view op) {
    std::string key = "backward\t";
    key += op;
    bump(std::move(key));
}

size_t dump(const std::string& path) {
    Store& s = store();
    const std::lock_guard<std::mutex> lock(s.mutex);

    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        return 0;
    }

    for (const auto& [key, count] : s.counts) {
        out << key << '\t' << count << '\n';
    }
    return s.counts.size();
}

void clear() noexcept {
    Store& s = store();
    const std::lock_guard<std::mutex> lock(s.mutex);
    s.counts.clear();
}

}  // namespace vkml::coverage
