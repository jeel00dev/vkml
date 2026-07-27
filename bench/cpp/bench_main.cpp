// Baseline measurements for the CPU backend.
//
// Categories match the list M1 asks for. The ones that need a GPU (command
// recording, dispatch, upload/download over PCIe) are stubbed with their CPU
// equivalents so the table has the same shape once the Vulkan backend lands and
// the two can be diffed directly.

#include "harness.h"

#include "vkml/api/ops.h"
#include "vkml/api/tensor.h"
#include "vkml/autograd/autograd.h"
#include "vkml/backend/api/backend.h"
#include "vkml/core/allocator.h"
#include "vkml/dispatch/executor.h"
#include "vkml/graph/graph.h"
#include "vkml/util/log.h"

#include <cstring>
#include <string>
#include <vector>

using namespace vkml;
namespace b = vkml::bench;

namespace {

Tensor make(int64_t rows, int64_t cols) {
    std::vector<float> data(static_cast<size_t>(rows * cols), 1.0F);
    return Tensor::from_host(data.data(), {rows, cols});
}

void bench_allocation() {
    for (const size_t bytes : {size_t{1024}, size_t{1024 * 1024}, size_t{64 * 1024 * 1024}}) {
        const std::string name = "alloc " + std::to_string(bytes / 1024) + " KiB";
        b::run("memory", name, [bytes] {
            auto s = cpu_allocator().allocate(bytes);
            b::keep(s->data());
        });
    }
}

void bench_transfer() {
    // The CPU analogue of an upload/download. On Vulkan these become staging
    // copies over PCIe, and the target GPU has only 256 MiB of host-visible
    // device-local memory, so every transfer goes through a staging buffer.
    for (const size_t mib : {size_t{1}, size_t{16}}) {
        const size_t bytes = mib * 1024 * 1024;
        const size_t n = bytes / sizeof(float);
        std::vector<float> host(n, 2.0F);

        b::run(
            "transfer", "upload " + std::to_string(mib) + " MiB",
            [&host, n] {
                const Tensor t = Tensor::from_host(host.data(), {static_cast<int64_t>(n)});
                b::keep(t.numel());
            },
            static_cast<double>(bytes), "B");

        const Tensor device = Tensor::from_host(host.data(), {static_cast<int64_t>(n)});
        std::vector<float> out(n);
        b::run(
            "transfer", "download " + std::to_string(mib) + " MiB",
            [&device, &out] { device.to_host(out.data()); }, static_cast<double>(bytes), "B");
    }
}

void bench_graph_construction() {
    // Pure graph building, no evaluation: the cost the M5 lowering is meant to
    // amortise away (see docs/adr/0001).
    const bool prev = eager();
    set_eager(false);

    const Tensor a = make(64, 64);

    for (const int depth : {10, 100, 1000}) {
        b::run("graph", "build chain depth " + std::to_string(depth), [&a, depth] {
            Tensor t = a;
            for (int i = 0; i < depth; ++i) {
                t = relu(t);
            }
            b::keep(t.ndim());
        });
    }

    for (const int depth : {100, 1000}) {
        Tensor t = a;
        for (int i = 0; i < depth; ++i) {
            t = relu(t);
        }
        b::run("graph", "topo order depth " + std::to_string(depth),
               [&t] { b::keep(topological_order(t.node()).size()); });
    }

    set_eager(prev);
}

void bench_dispatch_overhead() {
    // Per-node executor cost, isolated by using the smallest possible tensor:
    // whatever remains is scheduling, binding and the kernel call itself.
    const bool prev = eager();
    set_eager(false);

    const Tensor tiny = make(1, 1);
    b::run("dispatch", "realize 1 node (1x1)", [&tiny] {
        const Tensor t = relu(tiny);
        t.realize();
        b::keep(t.numel());
    });

    b::run("dispatch", "realize 10 nodes (1x1)", [&tiny] {
        Tensor t = tiny;
        for (int i = 0; i < 10; ++i) {
            t = relu(t);
        }
        t.realize();
        b::keep(t.numel());
    });

    set_eager(prev);
}

void bench_kernels() {
    const bool prev = eager();
    set_eager(false);

    for (const int64_t n : {int64_t{256}, int64_t{1024}}) {
        const Tensor a = make(n, n);
        const Tensor c = make(n, n);
        const double elems = static_cast<double>(n * n);
        const std::string sz = std::to_string(n) + "x" + std::to_string(n);

        b::run("kernel", "add " + sz, [&a, &c] { (a + c).realize(); }, elems, "elem");
        b::run("kernel", "relu " + sz, [&a] { relu(a).realize(); }, elems, "elem");
        b::run("kernel", "exp " + sz, [&a] { exp(a).realize(); }, elems, "elem");
        b::run("kernel", "sum " + sz, [&a] { sum(a).realize(); }, elems, "elem");
        b::run("kernel", "softmax " + sz, [&a] { softmax(a, -1).realize(); }, elems, "elem");
        b::run(
            "kernel", "contiguous(T) " + sz, [&a] { a.transpose(0, 1).contiguous().realize(); },
            elems, "elem");
    }

    // Strided vs contiguous, to quantify what the fast path in iterate.h buys.
    const Tensor big = make(1024, 1024);
    b::run(
        "kernel", "relu 1024x1024 strided", [&big] { relu(big.transpose(0, 1)).realize(); },
        1024.0 * 1024.0, "elem");

    set_eager(prev);
}

void bench_matmul() {
    const bool prev = eager();
    set_eager(false);

    for (const int64_t n : {int64_t{64}, int64_t{128}, int64_t{256}}) {
        const Tensor a = make(n, n);
        const Tensor c = make(n, n);
        const double flops =
            2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
        b::run(
            "matmul", "sgemm " + std::to_string(n) + "^3", [&a, &c] { matmul(a, c).realize(); },
            flops, "FLOP");
    }

    set_eager(prev);
}

void bench_training_step() {
    const bool prev = eager();
    set_eager(false);

    // An MNIST-shaped MLP step: forward, backward, and the loss reduction.
    // This is the end-to-end number the M0 gate exercises.
    const Tensor x = make(64, 784);
    Tensor w1 = make(784, 128);
    Tensor w2 = make(128, 10);
    w1.set_requires_grad(true);
    w2.set_requires_grad(true);

    b::run("training", "MLP 784-128-10 fwd (batch 64)", [&] {
        const Tensor h = relu(matmul(x, w1));
        const Tensor y = matmul(h, w2);
        y.realize();
        b::keep(y.numel());
    });

    b::run("training", "MLP 784-128-10 fwd+bwd (batch 64)", [&] {
        const Tensor h = relu(matmul(x, w1));
        const Tensor y = matmul(h, w2);
        const Tensor loss = mean(y);
        backward(loss);
        b::keep(loss.numel());
    });

    set_eager(prev);
}

}  // namespace

int main(int argc, char** argv) {
    set_log_level(LogLevel::Warn);

    bool json = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--json") == 0) {
            json = true;
        }
    }

    if (!json) {
        std::printf("vkml CPU baseline\n");
        std::printf("backend: %s   %s\n", std::string(cpu_backend().name()).c_str(),
                    cpu_backend().capabilities().summary().c_str());
    }

    bench_allocation();
    bench_transfer();
    bench_graph_construction();
    bench_dispatch_overhead();
    bench_kernels();
    bench_matmul();
    bench_training_step();

    if (json) {
        b::Registry::instance().print_json();
    } else {
        b::Registry::instance().print_table();
    }
    return 0;
}
