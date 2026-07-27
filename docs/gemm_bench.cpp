// Micro-benchmark: large compute-bound GEMM throughput on this machine's Vulkan device.
// Purpose: establish a realistic performance ceiling for a hand-written Vulkan GEMM
// before committing to performance targets in the framework roadmap.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static void bench(ggml_backend_t backend, ggml_type ta, int M, int N, int K, int iters) {
    struct ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 16 + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(ip);

    // ggml_mul_mat(a[K,M], b[K,N]) -> dst[M,N]
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, ta, K, M);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    struct ggml_tensor * c = ggml_mul_mat(ctx, a, b);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        printf("  alloc failed for M=%d N=%d K=%d\n", M, N, K);
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return;
    }

    // warmup
    for (int i = 0; i < 3; i++) ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);

    const double t0 = now_ms();
    for (int i = 0; i < iters; i++) ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
    const double t1 = now_ms();

    const double ms     = (t1 - t0) / iters;
    const double gflop  = 2.0 * (double)M * N * K / 1e9;
    // effective bandwidth counts only the weight matrix, which dominates when N is small
    const double gbps   = ((double)M * K * ggml_type_size(ta) / 1e9) / (ms / 1000.0);
    printf("  %-5s M=%-5d N=%-5d K=%-5d  %8.3f ms  %8.1f GFLOPS  %7.1f GB/s\n",
           ggml_type_name(ta), M, N, K, ms, gflop / (ms / 1000.0), gbps);

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
}

int main() {
    ggml_backend_load_all();

    ggml_backend_t backend = NULL;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            printf("device: %s (%s)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
            backend = ggml_backend_dev_init(dev, NULL);
            break;
        }
    }
    if (!backend) { printf("no GPU backend\n"); return 1; }

    const int sizes[] = { 512, 1024, 2048, 4096 };
    printf("\n-- square GEMM, weights f32 (training-relevant) --\n");
    for (int s : sizes) bench(backend, GGML_TYPE_F32, s, s, s, s >= 2048 ? 20 : 50);
    printf("\n-- square GEMM, weights f16 --\n");
    for (int s : sizes) bench(backend, GGML_TYPE_F16, s, s, s, s >= 2048 ? 20 : 50);

    printf("\n-- MLP-ish shapes (batch x features) --\n");
    bench(backend, GGML_TYPE_F32, 4096, 256, 4096, 30);
    bench(backend, GGML_TYPE_F16, 4096, 256, 4096, 30);
    bench(backend, GGML_TYPE_F32, 1024, 1024, 1024, 50);
    bench(backend, GGML_TYPE_F16, 1024, 1024, 1024, 50);

    // Small-batch / GEMV: relevant to RNN steps and batch-1 inference.
    // Reported as effective weight-read bandwidth, since these are memory bound.
    printf("\n-- small N (memory bound; GB/s = weight bytes / time) --\n");
    for (int n : { 1, 4, 16, 64 }) {
        bench(backend, GGML_TYPE_F32, 4096, n, 4096, 30);
        bench(backend, GGML_TYPE_F16, 4096, n, 4096, 30);
    }

    ggml_backend_free(backend);
    return 0;
}
