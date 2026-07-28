#include "iterate.h"
#include "kernels.h"
#include "reduce.h"

#include "vkml/util/assert.h"

#include <cstring>

namespace vkml::cpu {
namespace {

/// Batched matrix multiply.
///
/// Operands arrive as rank-4 views [B0, B1, M, K] x [B0, B1, K, N] -> [B0, B1, M, N];
/// lower-rank cases are unsqueezed to rank 4 during graph construction, and any
/// batch broadcasting is already expressed as stride-0 axes. That keeps this
/// kernel to one shape rather than a nest of rank special cases.
///
/// NUMERICS: the K-reduction uses pairwise summation, not a running scalar
/// accumulator. At K = 784 (MNIST) a sequential dot product has a worst-case
/// relative error of ~9.3e-5, which misses the 1e-5 gate against PyTorch by
/// nearly an order of magnitude. See reduce.h for the full analysis. This is
/// the single most important numerical decision in the CPU backend.
///
/// PERFORMANCE: this is the naive triple loop with no blocking or
/// vectorisation, which is deliberate -- the CPU backend is the correctness
/// oracle and the project standard puts performance last for it. The tiled
/// version lives in the Vulkan backend at M3.
///
/// `T` is the STORAGE type. The dot product accumulates in float regardless,
/// which is the fp32-accumulation half of ARCHITECTURE.md 7.3 and matters far
/// more here than anywhere else: an f16 accumulator over K = 784 would lose
/// roughly three decimal digits, well outside the 1e-3 the f16 tolerance allows.
template <typename T>
void matmul_impl(Node& out) {
    const Node& a = *out.src[0];
    const Node& b = *out.src[1];

    VKML_ASSERT(out.shape.ndim() == 4 && a.shape.ndim() == 4 && b.shape.ndim() == 4,
                "matmul kernel expects rank-4 operands, got out={} a={} b={}", out.shape.ndim(),
                a.shape.ndim(), b.shape.ndim());
    VKML_ASSERT(out.shape.is_contiguous(), "matmul output must be contiguous");

    const int64_t b0 = out.shape.dim(0);
    const int64_t b1 = out.shape.dim(1);
    const int64_t m = out.shape.dim(2);
    const int64_t n = out.shape.dim(3);
    const int64_t k = a.shape.dim(3);

    VKML_ASSERT(b.shape.dim(2) == k, "matmul inner dimensions disagree: {} vs {}", k,
                b.shape.dim(2));

    const auto* a_bytes = static_cast<const std::byte*>(a.data());
    const auto* b_bytes = static_cast<const std::byte*>(b.data());
    auto* out_p = static_cast<T*>(out.data());

    for (int64_t i0 = 0; i0 < b0; ++i0) {
        for (int64_t i1 = 0; i1 < b1; ++i1) {
            const int64_t a_batch = i0 * a.shape.stride(0) + i1 * a.shape.stride(1);
            const int64_t b_batch = i0 * b.shape.stride(0) + i1 * b.shape.stride(1);
            const int64_t o_batch = ((i0 * b1) + i1) * m * n;

            for (int64_t r = 0; r < m; ++r) {
                const int64_t a_row = a_batch + r * a.shape.stride(2);

                for (int64_t c = 0; c < n; ++c) {
                    const int64_t b_col = b_batch + c * b.shape.stride(3);

                    const int64_t a_k_stride = a.shape.stride(3);
                    const int64_t b_k_stride = b.shape.stride(2);

                    const float dot = pairwise_sum<float>(
                        [&](int64_t idx) {
                            T av{};
                            T bv{};
                            std::memcpy(&av, a_bytes + a_row + idx * a_k_stride, sizeof(T));
                            std::memcpy(&bv, b_bytes + b_col + idx * b_k_stride, sizeof(T));
                            return widen(av) * widen(bv);
                        },
                        0, k);

                    out_p[o_batch + r * n + c] = static_cast<T>(dot);
                }
            }
        }
    }
}

void k_matmul(Node& out) {
    switch (out.dtype) {
        case DType::F32: matmul_impl<float>(out); return;
        case DType::F16: matmul_impl<Half>(out); return;
        default: unsupported_dtype(out);
    }
}

}  // namespace

void register_matmul_kernels(KernelTable& t) { t[static_cast<size_t>(OpKind::Matmul)] = k_matmul; }

}  // namespace vkml::cpu
