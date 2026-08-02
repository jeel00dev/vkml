#include "vkml/backend/cpu/cpu_backend.h"

#include "kernels.h"

#include "vkml/util/assert.h"
#include "vkml/util/log.h"

#include <cstring>
#include <format>

namespace vkml {
namespace {

const cpu::KernelTable& kernel_table() {
    // Built once, on first use. A function-local static keeps it out of the
    // static initialisation order fiasco, which matters because the backend
    // registry may be touched during another translation unit's static init.
    static const cpu::KernelTable table = [] {
        cpu::KernelTable t{};
        cpu::register_movement_kernels(t);
        cpu::register_elementwise_kernels(t);
        cpu::register_reduce_kernels(t);
        cpu::register_matmul_kernels(t);
        return t;
    }();
    return table;
}

}  // namespace

CpuBackend::CpuBackend() {
    caps_.fp16_compute = false;  // F16 is storage-only here; conversion via vkml::Half
    caps_.fp64_compute = true;
    caps_.bf16_compute = false;

    caps_.subgroup_ops = false;
    caps_.global_float_atomics = true;  // trivially, since execution is serial
    caps_.shared_float_atomics = true;
    caps_.cooperative_matrix = false;

    caps_.buffer_device_address = true;  // host pointers are addresses
    caps_.unified_memory = true;
    caps_.host_accessible_buffers = true;
    caps_.min_buffer_alignment = kCpuAlignment;
    caps_.max_allocation_bytes = SIZE_MAX;

    caps_.asynchronous = false;
}

bool CpuBackend::supports(const Node& node) const {
    // View ops never reach a backend: the executor resolves them by aliasing
    // storage, so there is deliberately no kernel for them.
    if (is_view_op(node.op) || node.is_leaf()) {
        return true;
    }
    const auto slot = static_cast<size_t>(node.op);
    return slot < kernel_table().size() && kernel_table()[slot] != nullptr;
}

void CpuBackend::compute(std::span<Node* const> nodes) {
    const cpu::KernelTable& table = kernel_table();

    for (Node* node : nodes) {
        VKML_ASSERT(node != nullptr, "null node in compute list");

        if (is_view_op(node->op) || node->is_leaf()) {
            continue;  // resolved by the executor, nothing to compute
        }

        const auto slot = static_cast<size_t>(node->op);
        const cpu::Kernel fn = slot < table.size() ? table[slot] : nullptr;
        if (fn == nullptr) {
            throw NotImplementedError(
                std::format("cpu backend: no kernel for op '{}'", op_name(node->op)));
        }

        // Binding, not computedness, on both counts -- including the sources.
        // A source's value is usually produced by an EARLIER NODE IN THIS SAME
        // batch, and kFlagComputed is only set once the whole batch returns, so
        // asserting is_computed() here would fire on every ordinary chain. What
        // is assertable at this point is that the memory exists; that it holds
        // the right value is the executor's ordering guarantee, not a local one.
        VKML_ASSERT(node->is_bound(), "node '{}' has no storage before compute", op_name(node->op));
        for (int i = 0; i < node->n_src; ++i) {
            VKML_ASSERT(node->src[static_cast<size_t>(i)]->is_bound(),
                        "source {} of '{}' has no storage", i, op_name(node->op));
        }

        fn(*node);
    }
}

void CpuBackend::copy_from_host(Storage& dst, int64_t dst_offset, const void* src, size_t nbytes) {
    if (nbytes == 0) {
        return;
    }
    VKML_CHECK(dst_offset >= 0 && static_cast<size_t>(dst_offset) + nbytes <= dst.nbytes(),
               IndexError, "copy_from_host writes {} bytes at offset {} into a {}-byte storage",
               nbytes, dst_offset, dst.nbytes());
    std::memcpy(static_cast<std::byte*>(dst.data()) + dst_offset, src, nbytes);
}

void CpuBackend::copy_to_host(void* dst, const Storage& src, int64_t src_offset, size_t nbytes) {
    if (nbytes == 0) {
        return;
    }
    VKML_CHECK(src_offset >= 0 && static_cast<size_t>(src_offset) + nbytes <= src.nbytes(),
               IndexError, "copy_to_host reads {} bytes at offset {} from a {}-byte storage",
               nbytes, src_offset, src.nbytes());
    std::memcpy(dst, static_cast<const std::byte*>(src.data()) + src_offset, nbytes);
}

void CpuBackend::copy_device_to_device(std::span<const BufferCopy> copies) {
    // A loop, and that is the honest implementation rather than a stub: on the
    // CPU there is no submission to amortise, so batching buys nothing here.
    // The interface takes a span because the VULKAN backend needs it, which is
    // the shape of most of this project's backend methods.
    for (const BufferCopy& c : copies) {
        if (c.nbytes == 0) {
            continue;
        }
        VKML_ASSERT(c.dst != nullptr && c.src != nullptr, "null storage in copy_device_to_device");
        VKML_CHECK(c.dst_offset >= 0 &&
                       static_cast<size_t>(c.dst_offset) + c.nbytes <= c.dst->nbytes(),
                   IndexError,
                   "copy_device_to_device writes {} bytes at offset {} into a {}-byte "
                   "storage",
                   c.nbytes, c.dst_offset, c.dst->nbytes());
        VKML_CHECK(c.src_offset >= 0 &&
                       static_cast<size_t>(c.src_offset) + c.nbytes <= c.src->nbytes(),
                   IndexError,
                   "copy_device_to_device reads {} bytes at offset {} from a {}-byte "
                   "storage",
                   c.nbytes, c.src_offset, c.src->nbytes());
        // The interface forbids overlap, so memcpy would be legal. memmove is
        // used anyway: it costs nothing measurable at these sizes, and it means
        // a caller that gets the overlap check wrong reads stale bytes rather
        // than entering undefined behaviour. "CPU backend" here means host
        // memory on both sides.
        std::memmove(static_cast<std::byte*>(c.dst->data()) + c.dst_offset,
                     static_cast<const std::byte*>(c.src->data()) + c.src_offset, c.nbytes);
    }
}

Backend& cpu_backend() {
    static CpuBackend instance;
    return instance;
}

}  // namespace vkml
