#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

// Minimal measurement harness.
//
// Deliberately not Google Benchmark. What this project needs from a benchmark
// is a stable, diffable table of numbers that CI can compare against a recorded
// baseline; what Google Benchmark adds beyond that (statistical modelling,
// complexity fitting, a large dependency) is not currently used by anything.
// This is ~150 lines and emits both a human table and JSON.
//
// Methodology, stated so the numbers mean something:
//   - the body is run untimed until a warm-up budget elapses, so allocation and
//     first-touch page faults do not land in the sample;
//   - iteration count is auto-tuned to reach a target wall time, so cheap and
//     expensive cases get comparable statistical weight;
//   - repeats are taken and the MINIMUM reported. For a deterministic
//     single-threaded workload the minimum is the estimate least polluted by
//     scheduler noise; the mean would report the machine's background load.
//     Both are printed so a large spread is visible.

namespace vkml::bench {

struct Result {
    std::string category;
    std::string name;
    double ns_min = 0.0;
    double ns_mean = 0.0;
    int64_t iterations = 0;
    double work = 0.0;         ///< bytes, FLOPs or elements; 0 if not meaningful
    std::string work_unit;     ///< "B", "FLOP", "elem"
};

class Registry {
public:
    static Registry& instance() {
        static Registry r;
        return r;
    }

    void add(Result r) { results_.push_back(std::move(r)); }

    [[nodiscard]] const std::vector<Result>& results() const { return results_; }

    void print_table() const {
        std::printf("\n%-14s %-38s %14s %14s %16s\n", "category", "benchmark", "min", "mean",
                    "throughput");
        std::printf("%s\n", std::string(100, '-').c_str());

        std::string last;
        for (const Result& r : results_) {
            if (r.category != last) {
                last = r.category;
            }
            std::string thr = "-";
            if (r.work > 0.0 && r.ns_min > 0.0) {
                const double per_s = r.work / (r.ns_min * 1e-9);
                if (r.work_unit == "B") {
                    thr = format_scaled(per_s / 1e9, "GB/s");
                } else if (r.work_unit == "FLOP") {
                    thr = format_scaled(per_s / 1e9, "GFLOP/s");
                } else {
                    thr = format_scaled(per_s / 1e6, "Melem/s");
                }
            }
            std::printf("%-14s %-38s %14s %14s %16s\n", r.category.c_str(), r.name.c_str(),
                        format_time(r.ns_min).c_str(), format_time(r.ns_mean).c_str(),
                        thr.c_str());
        }
        std::printf("\n");
    }

    void print_json() const {
        std::printf("[\n");
        for (size_t i = 0; i < results_.size(); ++i) {
            const Result& r = results_[i];
            std::printf(R"(  {"category":"%s","name":"%s","ns_min":%.3f,"ns_mean":%.3f,)"
                        R"("iterations":%lld,"work":%.1f,"work_unit":"%s"}%s)"
                        "\n",
                        r.category.c_str(), r.name.c_str(), r.ns_min, r.ns_mean,
                        static_cast<long long>(r.iterations), r.work, r.work_unit.c_str(),
                        i + 1 < results_.size() ? "," : "");
        }
        std::printf("]\n");
    }

private:
    static std::string format_time(double ns) {
        char buf[64];
        if (ns < 1e3) {
            std::snprintf(buf, sizeof(buf), "%.1f ns", ns);
        } else if (ns < 1e6) {
            std::snprintf(buf, sizeof(buf), "%.2f us", ns / 1e3);
        } else {
            std::snprintf(buf, sizeof(buf), "%.2f ms", ns / 1e6);
        }
        return buf;
    }

    static std::string format_scaled(double v, const char* unit) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.2f %s", v, unit);
        return buf;
    }

    std::vector<Result> results_;
};

/// Prevents the optimiser from discarding a computed value.
template <typename T>
inline void keep(T&& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

struct Options {
    double warmup_seconds = 0.05;
    double target_seconds = 0.20;
    int repeats = 5;
};

/// Times `body`, auto-tuning the iteration count.
inline void run(const std::string& category, const std::string& name,
                const std::function<void()>& body, double work = 0.0,
                const std::string& work_unit = "", Options opt = {}) {
    using clock = std::chrono::steady_clock;
    auto elapsed = [](clock::time_point a, clock::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };

    // Warm up untimed: first-touch faults and lazy allocation belong to setup,
    // not to the measurement.
    const auto warm_start = clock::now();
    int64_t warm_iters = 0;
    do {
        body();
        ++warm_iters;
    } while (elapsed(warm_start, clock::now()) < opt.warmup_seconds);

    const double per_iter = elapsed(warm_start, clock::now()) / static_cast<double>(warm_iters);
    auto iters = static_cast<int64_t>(opt.target_seconds / std::max(per_iter, 1e-9));
    iters = std::clamp<int64_t>(iters, 1, 100'000'000);

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(opt.repeats));
    for (int r = 0; r < opt.repeats; ++r) {
        const auto t0 = clock::now();
        for (int64_t i = 0; i < iters; ++i) {
            body();
        }
        const auto t1 = clock::now();
        samples.push_back(elapsed(t0, t1) * 1e9 / static_cast<double>(iters));
    }

    const double min = *std::min_element(samples.begin(), samples.end());
    double mean = 0.0;
    for (const double s : samples) {
        mean += s;
    }
    mean /= static_cast<double>(samples.size());

    Registry::instance().add(Result{category, name, min, mean, iters, work, work_unit});
}

}  // namespace vkml::bench
