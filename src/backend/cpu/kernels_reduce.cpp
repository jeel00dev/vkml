#include "iterate.h"
#include "kernels.h"
#include "reduce.h"

#include "vkml/util/assert.h"

#include <cmath>
#include <limits>

namespace vkml::cpu {
namespace {

/// Reads element `k` of the reduced axes, at output position `base`.
///
/// ALWAYS RETURNS float, whatever `T` is stored. That is the fp32-accumulation
/// half of `ARCHITECTURE.md` §7.3: an f16 reduction of any length would lose
/// far more than its 1e-3 tolerance allows if the running sum were 16-bit, and
/// every fold below is written against `float` precisely so it cannot be.
///
/// Templated on the storage type rather than branching on a dtype field,
/// because this is the innermost loop of every reduction in the library and a
/// template costs nothing to read here.
template <typename T>
struct Reader {
    const std::byte* data;
    int64_t base;
    const Shape* reduced;

    [[nodiscard]] float operator()(int64_t k) const noexcept {
        T v{};
        std::memcpy(&v, data + base + linear_to_offset(k, *reduced), sizeof(T));
        return widen(v);
    }
};

/// Shared skeleton for every reduction: walk output positions, fold the
/// collapsed axes into each one. `fold` receives a Reader and the reduction
/// length and returns the output value.
///
/// `In` and `Out` are separate because for argmax and argmin they differ: the
/// input is floating and the result is an index. Collapsing them into one
/// parameter reads f32 input bytes as i64, which is a wrong answer rather than
/// a compile error -- it was caught by the existing argmax tests.
template <typename In, typename Out, typename Fold>
void reduce_generic(Node& out, Fold&& fold) {
    const Node& in = *out.src[0];
    const auto p = out.params.get<ReduceParams>();
    const uint32_t mask = p.axes_mask != 0 ? p.axes_mask : all_axes_mask(in.shape.ndim());

    const ReducePlan plan = make_reduce_plan(in.shape, mask);
    const auto* base_bytes = static_cast<const std::byte*>(in.data());

    const int64_t n_out = plan.kept.numel();
    const int64_t n_red = plan.reduced.numel();

    // Reduction outputs are always freshly allocated by the executor and so are
    // contiguous; a flat index is then correct and much cheaper than
    // recomputing an offset per element.
    VKML_ASSERT(out.shape.is_contiguous(), "reduction output must be contiguous");

    for (int64_t o = 0; o < n_out; ++o) {
        const Reader<In> read{base_bytes, linear_to_offset(o, plan.kept), &plan.reduced};
        // static_cast, because Half's constructor from float is explicit -- the
        // narrowing back to 16 bits has to be visible.
        static_cast<Out*>(out.data())[o] = static_cast<Out>(fold(read, n_red));
    }
}

/// Runs `fold` over whichever floating storage type the output carries.
///
/// Every caller's fold is written against `float` and stays unchanged, which is
/// what keeps the f16 path from quietly acquiring different numerics from the
/// f32 one -- they are the same code.
template <typename Fold>
void reduce_float(Node& out, Fold&& fold) {
    switch (out.dtype) {
        case DType::F32: reduce_generic<float, float>(out, fold); return;
        case DType::F16: reduce_generic<Half, Half>(out, fold); return;
        default: unsupported_dtype(out);
    }
}

/// Reductions that consume a float and produce an index, so the dtype that
/// selects the reader is the INPUT's rather than the output's.
template <typename Fold>
void reduce_to_index(Node& out, Fold&& fold) {
    switch (out.src[0]->dtype) {
        case DType::F32: reduce_generic<float, int64_t>(out, fold); return;
        case DType::F16: reduce_generic<Half, int64_t>(out, fold); return;
        default: unsupported_dtype(out, out.src[0]->dtype);
    }
}

void k_sum(Node& out) {
    reduce_float(out, [](const auto& read, int64_t n) { return pairwise_sum<float>(read, 0, n); });
}

void k_mean(Node& out) {
    reduce_float(out, [](const auto& read, int64_t n) {
        if (n == 0) {
            return std::numeric_limits<float>::quiet_NaN();  // matches torch.mean of empty
        }
        // Sum then divide once, rather than accumulating scaled values: one
        // rounding at the end instead of n of them.
        return pairwise_sum<float>(read, 0, n) / static_cast<float>(n);
    });
}

void k_prod(Node& out) {
    reduce_float(out, [](const auto& read, int64_t n) {
        float acc = 1.0F;
        for (int64_t i = 0; i < n; ++i) {
            acc *= read(i);
        }
        return acc;
    });
}

void k_max(Node& out) {
    reduce_float(out, [](const auto& read, int64_t n) {
        float best = -std::numeric_limits<float>::infinity();
        for (int64_t i = 0; i < n; ++i) {
            const float v = read(i);
            if (std::isnan(v)) {
                return v;  // torch.max propagates NaN
            }
            if (v > best) {
                best = v;
            }
        }
        return best;
    });
}

void k_min(Node& out) {
    reduce_float(out, [](const auto& read, int64_t n) {
        float best = std::numeric_limits<float>::infinity();
        for (int64_t i = 0; i < n; ++i) {
            const float v = read(i);
            if (std::isnan(v)) {
                return v;
            }
            if (v < best) {
                best = v;
            }
        }
        return best;
    });
}

void k_argmax(Node& out) {
    VKML_ASSERT(out.dtype == DType::I64, "argmax must produce I64");
    reduce_to_index(out, [](const auto& read, int64_t n) -> int64_t {
        int64_t best_i = 0;
        float best = -std::numeric_limits<float>::infinity();
        for (int64_t i = 0; i < n; ++i) {
            const float v = read(i);
            // Strict > keeps the FIRST maximum, which is what torch.argmax
            // documents. Using >= would silently return the last one.
            if (v > best) {
                best = v;
                best_i = i;
            }
        }
        return best_i;
    });
}

void k_argmin(Node& out) {
    VKML_ASSERT(out.dtype == DType::I64, "argmin must produce I64");
    reduce_to_index(out, [](const auto& read, int64_t n) -> int64_t {
        int64_t best_i = 0;
        float best = std::numeric_limits<float>::infinity();
        for (int64_t i = 0; i < n; ++i) {
            const float v = read(i);
            if (v < best) {
                best = v;
                best_i = i;
            }
        }
        return best_i;
    });
}

/// Shared body for softmax and log_softmax over one axis.
///
/// Both use the max-subtraction trick: exp(x - max) instead of exp(x). Without
/// it, exp() overflows to inf for x > 88 in fp32 and the result is NaN after
/// the division. The subtraction is mathematically a no-op (it cancels in the
/// ratio) and costs one extra pass, which is the standard trade and the same
/// one PyTorch makes.
/// `T` is the STORAGE type; every intermediate below is float, for the reason
/// given on Reader. Templated rather than gated because this had no dtype check
/// at all: an f16 tensor was read and written as f32, silently, which is the
/// same defect the comparisons carried.
template <bool Log, typename T>
void softmax_impl(Node& out) {
    const Node& in = *out.src[0];
    const auto p = out.params.get<AxisParams>();
    const uint32_t mask = 1U << static_cast<uint32_t>(p.axis);

    const ReducePlan plan = make_reduce_plan(in.shape, mask);
    // The output is a fresh allocation, so its strides need not match the
    // input's -- the input may well be a transposed or broadcast view. Build a
    // second plan over the output's own layout rather than reusing the input's
    // offsets, which would scatter the results.
    const ReducePlan out_plan = make_reduce_plan(out.shape, mask);
    const auto* base_bytes = static_cast<const std::byte*>(in.data());

    const int64_t n_out = plan.kept.numel();
    const int64_t n_axis = plan.reduced.numel();

    for (int64_t o = 0; o < n_out; ++o) {
        const int64_t base = linear_to_offset(o, plan.kept);
        const int64_t out_base = linear_to_offset(o, out_plan.kept);
        const Reader<T> read{base_bytes, base, &plan.reduced};

        float peak = -std::numeric_limits<float>::infinity();
        for (int64_t k = 0; k < n_axis; ++k) {
            peak = std::max(peak, read(k));
        }

        const float denom = pairwise_sum<float>(
            [&read, peak](int64_t k) { return std::exp(read(k) - peak); }, 0, n_axis);

        for (int64_t k = 0; k < n_axis; ++k) {
            const float shifted = read(k) - peak;
            // log_softmax uses log(sum) rather than log(exp(x)/sum): the latter
            // loses all precision once exp underflows to zero, which is exactly
            // where log_softmax is most needed (very negative logits).
            const T v = static_cast<T>(Log ? shifted - std::log(denom) : std::exp(shifted) / denom);
            const int64_t off = out_base + linear_to_offset(k, out_plan.reduced);
            std::memcpy(static_cast<std::byte*>(out.data()) + off, &v, sizeof(T));
        }
    }
}

template <bool Log>
void softmax_dispatch(Node& out) {
    switch (out.dtype) {
        case DType::F32: softmax_impl<Log, float>(out); return;
        case DType::F16: softmax_impl<Log, Half>(out); return;
        default: unsupported_dtype(out);
    }
}

void k_softmax(Node& out) { softmax_dispatch<false>(out); }

void k_log_softmax(Node& out) { softmax_dispatch<true>(out); }

}  // namespace

void register_reduce_kernels(KernelTable& t) {
    t[static_cast<size_t>(OpKind::Sum)] = k_sum;
    t[static_cast<size_t>(OpKind::Mean)] = k_mean;
    t[static_cast<size_t>(OpKind::Prod)] = k_prod;
    t[static_cast<size_t>(OpKind::Max)] = k_max;
    t[static_cast<size_t>(OpKind::Min)] = k_min;
    t[static_cast<size_t>(OpKind::ArgMax)] = k_argmax;
    t[static_cast<size_t>(OpKind::ArgMin)] = k_argmin;
    t[static_cast<size_t>(OpKind::Softmax)] = k_softmax;
    t[static_cast<size_t>(OpKind::LogSoftmax)] = k_log_softmax;
}

}  // namespace vkml::cpu
