#include "vkml/backend/vulkan/vulkan_backend.h"

#include "vk_allocator.h"
#include "vk_command.h"
#include "vk_device.h"
#include "vk_pipeline.h"

#include "vkml/spv/binary.h"
#include "vkml/spv/cast.h"
#include "vkml/spv/cat.h"
#include "vkml/spv/col2im.h"
#include "vkml/spv/im2col.h"
#include "vkml/spv/max_pool2d.h"
#include "vkml/spv/rand.h"
#include "vkml/spv/index_select.h"
#include "vkml/spv/scatter_add.h"
#include "vkml/spv/slice_backward.h"
#include "vkml/spv/fill.h"
#include "vkml/spv/gemm_naive.h"
#include "vkml/spv/gemm_db.h"
#include "vkml/spv/gemm_reg.h"
#include "vkml/spv/gemm_split_k_reduce.h"
#include "vkml/spv/gemm_tiled.h"
#include "vkml/spv/gemv.h"
#include "vkml/spv/reduce.h"
#include "vkml/spv/softmax.h"
#include "vkml/spv/tri.h"
#include "vkml/spv/unary.h"
#include "vkml/spv/where.h"

#include "vkml/util/assert.h"
#include "vkml/util/log.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <string_view>
#include <cstring>
#include <format>
#include <limits>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace vkml {
namespace {

/// Mirrors the Operand struct in shaders/common.glsl.
///
/// scalar_block_layout is what lets this be a plain struct on both sides with
/// no padding rules to reconcile -- the single reason the push constant blocks
/// below can be written as ordinary C++.
struct GpuOperand {
    std::array<uint32_t, 4> ne{1, 1, 1, 1};
    std::array<uint32_t, 4> nb{0, 0, 0, 0};
};

/// Converts a Shape into the shader's view of it.
///
/// Two conversions happen here and nowhere else:
///   - extents are right-padded to rank 4 with 1s, so the shader's unrolled
///     loop needs no rank parameter;
///   - strides go from BYTES (host convention, matching NumPy) to ELEMENTS
///     (shader convention, because it indexes a typed buffer reference).
[[nodiscard]] GpuOperand to_gpu_operand(const Shape& shape, size_t itemsize) {
    GpuOperand op;
    const int nd = shape.ndim();
    const int pad = kMaxDims - nd;

    for (int i = 0; i < nd; ++i) {
        op.ne[static_cast<size_t>(pad + i)] = static_cast<uint32_t>(shape.dim(i));
        VKML_ASSERT(shape.stride(i) % static_cast<int64_t>(itemsize) == 0,
                    "stride {} is not a multiple of the element size {}", shape.stride(i),
                    itemsize);
        op.nb[static_cast<size_t>(pad + i)] =
            static_cast<uint32_t>(shape.stride(i) / static_cast<int64_t>(itemsize));
    }
    // Leading padded axes have extent 1, so their stride is never used; 0 is
    // the honest value and matches how broadcasting is expressed elsewhere.
    return op;
}

struct FillPush {
    uint64_t dst;
    uint32_t n;
    float value;
    float step;  ///< 0 for fill; arange is the same kernel with a slope
};

struct UnaryPush {
    uint64_t src;
    uint64_t dst;
    uint32_t n;
    // Clamp bounds only. Carried unconditionally rather than behind a separate
    // pipeline: eight bytes in a 256-byte budget costs nothing, and a spec
    // constant per bound would double the variant count of every other op.
    float clamp_lo;
    float clamp_hi;
    GpuOperand in_op;
    GpuOperand out_op;
};

struct BinaryPush {
    uint64_t a;
    uint64_t b;
    uint64_t dst;
    uint32_t n;
    GpuOperand a_op;
    GpuOperand b_op;
    GpuOperand out_op;
};

static_assert(sizeof(BinaryPush) <= 256, "binary push constants exceed the device budget");

struct WherePush {
    uint64_t cond;
    uint64_t a;
    uint64_t b;
    uint64_t dst;
    uint32_t n;
    GpuOperand cond_op;
    GpuOperand a_op;
    GpuOperand b_op;
    GpuOperand out_op;
};

static_assert(sizeof(WherePush) <= 256, "where push constants exceed the device budget");

struct TriPush {
    uint64_t src;
    uint64_t dst;
    uint32_t n;
    uint32_t height;
    uint32_t width;
    int32_t diagonal;
    GpuOperand in_op;
    GpuOperand out_op;
};

static_assert(sizeof(TriPush) <= 256, "tri push constants exceed the device budget");

struct CatPush {
    uint64_t a;
    uint64_t b;
    uint64_t dst;
    uint32_t n;
    uint32_t inner;
    uint32_t out_extent;
    uint32_t a_extent;
    uint32_t b_extent;
    GpuOperand a_op;
    GpuOperand b_op;
    GpuOperand out_op;
};

static_assert(sizeof(CatPush) <= 256, "cat push constants exceed the device budget");

/// Shared by index_select and scatter_add: the two are adjoints and remap the
/// same axis, so they need the same description of it.
struct GatherPush {
    uint64_t src;
    uint64_t index;
    uint64_t dst;
    uint32_t n;
    uint32_t inner;
    uint32_t out_extent;
    uint32_t src_extent;
    GpuOperand src_op;
    GpuOperand out_op;
};

static_assert(sizeof(GatherPush) <= 256, "gather push constants exceed the device budget");

/// Shared by im2col and col2im: adjoints over the same window geometry, so
/// they need the same description of it. The window counts and row total are
/// derived on the host rather than in the shader, because both kernels need
/// them and neither should re-derive a value the other already has.
struct UnfoldPush {
    uint64_t src;
    uint64_t dst;
    uint32_t n;
    int32_t kernel_h, kernel_w;
    int32_t stride_h, stride_w;
    int32_t pad_h, pad_w;
    int32_t dilation_h, dilation_w;
    int32_t image_h, image_w;
    int32_t out_h, out_w;
    int32_t channels;
    int32_t rows;
    GpuOperand in_op;
};

static_assert(sizeof(UnfoldPush) <= 256, "unfold push constants exceed the device budget");

/// Max pooling and its adjoint. `input` is used only by the adjoint, which
/// recomputes the argmax rather than reading a stored index.
struct RandPush {
    uint64_t dst;
    uint32_t n;
    uint64_t seed;
    uint64_t offset;
};

static_assert(sizeof(RandPush) <= 256, "rand push constants exceed the device budget");

struct SliceBackwardPush {
    uint64_t src;
    uint64_t dst;
    uint32_t n;
    int32_t axis;
    int32_t start;
    int32_t stop;
    int32_t step;
    GpuOperand grad_op;
    GpuOperand out_op;
};

static_assert(sizeof(SliceBackwardPush) <= 256,
              "slice_backward push constants exceed the device budget");

struct PoolPush {
    uint64_t src;
    uint64_t image;
    uint64_t dst;
    uint32_t n;
    int32_t kernel_h, kernel_w;
    int32_t stride_h, stride_w;
    int32_t pad_h, pad_w;
    int32_t dilation_h, dilation_w;
    int32_t image_h, image_w;
    int32_t out_h, out_w;
};

static_assert(sizeof(PoolPush) <= 256, "pool push constants exceed the device budget");

struct ReducePush {
    uint64_t src;
    uint64_t dst;
    uint32_t n_out;
    uint32_t n_red;
    GpuOperand kept;
    GpuOperand reduced;
};

static_assert(sizeof(ReducePush) <= 256, "reduce push constants exceed the device budget");

/// Splits a shape into the axes a reduction keeps and the axes it collapses.
///
/// Mirrors vkml::cpu::make_reduce_plan. Duplicated rather than shared because
/// backend/vulkan and backend/cpu are sibling layers and must not include each
/// other; the two are kept in step by the validation suite, which compares
/// their results directly.
struct SplitShape {
    Shape kept;
    Shape reduced;
};

[[nodiscard]] SplitShape split_for_reduce(const Shape& in, uint32_t axes_mask) {
    std::vector<int64_t> kd;
    std::vector<int64_t> ks;
    std::vector<int64_t> rd;
    std::vector<int64_t> rs;

    for (int i = 0; i < in.ndim(); ++i) {
        const bool reduced = (axes_mask & (1U << static_cast<uint32_t>(i))) != 0;
        if (reduced) {
            rd.push_back(in.dim(i));
            rs.push_back(in.stride(i));
        } else {
            kd.push_back(in.dim(i));
            ks.push_back(in.stride(i));
        }
    }
    return SplitShape{Shape::strided(kd, ks, in.itemsize()), Shape::strided(rd, rs, in.itemsize())};
}

struct SoftmaxPush {
    uint64_t src;
    uint64_t dst;
    uint32_t n_out;
    uint32_t n_axis;
    GpuOperand in_kept;
    GpuOperand in_axis;
    GpuOperand out_kept;
    GpuOperand out_axis;
};

static_assert(sizeof(SoftmaxPush) <= 256, "softmax push constants exceed the device budget");

struct GemmPush {
    uint64_t a;
    uint64_t b;
    uint64_t d;
    uint32_t n_out;
    uint32_t b1;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    GpuOperand op_a;
    GpuOperand op_b;
};

static_assert(sizeof(GemmPush) <= 256, "gemm push constants exceed the device budget");

struct SplitKReducePush {
    uint64_t src;
    uint64_t dst;
    uint32_t ne;
    uint32_t splits;
};

static_assert(sizeof(SplitKReducePush) <= 256,
              "split-k reduce push constants exceed the device budget");

/// GEMV dispatch mode. AUTO is identical to OFF: M4-R1 implements and measures
/// the kernel; enabling it by default is a separate decision with its own
/// evidence requirement.
enum class GemvMode : uint8_t { Auto, Off, Forced };

[[nodiscard]] GemvMode gemv_mode() {
    static const GemvMode mode = [] {
        const char* v = std::getenv("VKML_GEMV");
        const std::string_view sel = v != nullptr ? v : "";
        if (sel == "FORCED" || sel == "forced") {
            return GemvMode::Forced;
        }
        if (sel == "OFF" || sel == "off") {
            return GemvMode::Off;
        }
        return GemvMode::Auto;
    }();
    return mode;
}

/// How split-K was requested. AUTO is deliberately identical to OFF: M3-03
/// implements the mechanism, and the occupancy heuristic that would drive AUTO
/// is a separate stage with its own evidence requirement.
enum class SplitKMode : uint8_t { Auto, Off, Forced };

[[nodiscard]] SplitKMode split_k_mode() {
    static const SplitKMode mode = [] {
        const char* v = std::getenv("VKML_GEMM_SPLITK");
        const std::string_view sel = v != nullptr ? v : "";
        if (sel == "FORCED" || sel == "forced") {
            return SplitKMode::Forced;
        }
        if (sel == "OFF" || sel == "off") {
            return SplitKMode::Off;
        }
        return SplitKMode::Auto;
    }();
    return mode;
}

/// Requested partition count when split-K is forced. The chunk is derived from
/// it and rounded DOWN to a power of two, so the effective split count is
/// generally >= this; correctness does not depend on which value is chosen
/// (docs/SPLIT_K_DESIGN.md 2.4), only on the chunk being a power of two.
[[nodiscard]] uint32_t split_k_requested() {
    static const uint32_t n = [] {
        const char* v = std::getenv("VKML_GEMM_SPLITK_SPLITS");
        if (v == nullptr || v[0] == '\0') {
            return 4U;
        }
        const int parsed = std::atoi(v);
        return parsed > 1 ? static_cast<uint32_t>(parsed) : 4U;
    }();
    return n;
}

/// The split-K decision, isolated from dispatch so its rationale lives in one
/// place and can be exercised without a GPU.
struct SplitKPlan {
    uint32_t splits = 1;  ///< 1 means "do not split"
    uint32_t chunk = 0;   ///< K-tiles per partition; always a power of two
};

/// Chooses whether and how to partition K.
///
/// THE CORRECTNESS CONSTRAINT, first, because it is not negotiable: `chunk`
/// must be a power of two. The alignment lemma (docs/SPLIT_K_DESIGN.md 2.3)
/// needs every partition boundary at a multiple of 2^q so that no carry-stack
/// fold crosses it; that is what makes split-K bit-identical to the unsplit
/// kernel. This inverts the production ordering -- llama.cpp picks the split
/// count and derives the chunk -- and the inversion is deliberate.
///
/// THE PROFITABILITY RULE: `ktiles >= tiles`.
///
/// Split-K trades reduction traffic and one extra dispatch for occupancy, so it
/// pays only when output-tile parallelism is the scarce dimension and K is the
/// abundant one. Measured over 16 shape/K combinations spanning tiles 4..2304
/// and ktiles 8..512, this rule enables every case that gains more than 1.15x
/// and declines every case that would lose, including the harmful ones
/// (256x256x256 at 0.59x, 256x512x256 at 0.84x). Everything it forgoes is
/// <= 1.23x. A tiles-only threshold cannot do this: at tiles=64 the outcome
/// ranges from 0.59x to 1.99x depending on K alone.
///
/// THE PARTITION COUNT targets enough workgroups to fill the machine --
/// `CU x 8` concurrent slots, the figure the fill curve in
/// docs/PERFORMANCE-MODEL.md 5g is built on -- then clamps to [2, 16]. The cap
/// follows llama.cpp's ("unless k is huge this is a lot of overhead") and is
/// reinforced here by measurement: 32 partitions beat 8 on one shape and lost
/// on another, so a larger cap buys nothing reliable.
[[nodiscard]] SplitKPlan plan_split_k(uint32_t tiles, uint32_t ktiles, uint64_t out_elems,
                                      uint32_t cu_count, uint32_t requested) {
    // A single K-tile cannot be divided; a single output tile still can.
    if (ktiles < 2 || tiles == 0) {
        return {};
    }
    // No compute-unit count means no occupancy judgement is possible. Declining
    // is the honest response -- never guess a device property.
    if (cu_count == 0 && requested == 0) {
        return {};
    }

    uint32_t target = requested;
    if (target == 0) {
        // AUTO. Profitability first.
        if (ktiles < tiles) {
            return {};
        }
        const uint32_t slots = cu_count * 8;
        target = std::clamp(slots / std::max(tiles, 1U), 2U, 16U);
    }

    // Largest power-of-two chunk that yields at least `target` partitions.
    uint32_t chunk = 1;
    while (chunk * 2 <= ktiles / target) {
        chunk *= 2;
    }
    // Below four K-tiles a partition is dominated by its own prologue, so raise
    // the chunk and accept fewer partitions rather than declining. An earlier
    // version rejected outright here and silently forfeited two measured wins
    // (128x1024x128 at 2.10x, 64x512x64 at 2.05x), where the target partition
    // count was simply more ambitious than K could support.
    if (chunk < 4 && requested == 0) {
        chunk = 4;
    }
    if (chunk >= ktiles) {
        return {};
    }

    const uint32_t splits = (ktiles + chunk - 1) / chunk;
    if (splits < 2) {
        return {};
    }
    // Workspace is `splits` copies of the output. Bound it so a large-output
    // shape cannot quietly allocate hundreds of megabytes; such shapes have
    // ample tiles already and are declined by the profitability rule anyway.
    constexpr uint64_t kMaxWorkspaceBytes = 64ULL * 1024 * 1024;
    if (out_elems * splits * sizeof(float) > kMaxWorkspaceBytes) {
        return {};
    }
    return SplitKPlan{splits, chunk};
}

struct CastPush {
    uint64_t src;
    uint64_t dst;
    uint32_t n;
};

static_assert(sizeof(UnaryPush) <= 256, "unary push constants exceed the device budget");

/// Mirrors the OP_* codes in shaders/unary.comp. Codes 0-4 are frozen; adding
/// an operation appends.
enum class UnaryOp : uint32_t {
    Copy = 0,
    Relu = 1,
    Neg = 2,
    Abs = 3,
    Exp = 4,
    Sign = 5,
    Square = 6,
    Sqrt = 7,
    Rsqrt = 8,
    Reciprocal = 9,
    Log = 10,
    Erf = 11,
    Sin = 12,
    Cos = 13,
    Tanh = 14,
    Sigmoid = 15,
    Gelu = 16,
    Silu = 17,
    Clamp = 18,
};

/// Mirrors the OP_* codes in shaders/binary.comp. Codes from Equal upward are
/// comparisons, which the shader keys on to pick the destination element type,
/// so they must stay contiguous and last.
enum class BinaryOp : uint32_t {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    Pow = 4,
    Maximum = 5,
    Minimum = 6,
    Equal = 7,
    Less = 8,
    Greater = 9,
    LessEqual = 10,
    GreaterEqual = 11,
    NotEqual = 12,
};

[[nodiscard]] std::optional<BinaryOp> to_binary_op(OpKind k) {
    switch (k) {
        case OpKind::Add: return BinaryOp::Add;
        case OpKind::Sub: return BinaryOp::Sub;
        case OpKind::Mul: return BinaryOp::Mul;
        case OpKind::Div: return BinaryOp::Div;
        case OpKind::Pow: return BinaryOp::Pow;
        case OpKind::Maximum: return BinaryOp::Maximum;
        case OpKind::Minimum: return BinaryOp::Minimum;
        case OpKind::Equal: return BinaryOp::Equal;
        case OpKind::Less: return BinaryOp::Less;
        case OpKind::Greater: return BinaryOp::Greater;
        case OpKind::LessEqual: return BinaryOp::LessEqual;
        case OpKind::GreaterEqual: return BinaryOp::GreaterEqual;
        case OpKind::NotEqual: return BinaryOp::NotEqual;
        default: return std::nullopt;
    }
}

/// Both operands present, floating, and agreeing. Comparisons narrow the
/// *output* to Bool but still consume floats, so this is asked of the sources
/// independently of the node's own dtype.
///
/// Agreement is required rather than assumed: `check_same_dtype` in api/ops.cpp
/// already rejects mixed operands, and this is the backend-side statement of
/// the same contract -- one DTYPE constant selects the width for both.
[[nodiscard]] bool binary_srcs_are_float(const Node& node) {
    return node.src[0] != nullptr && node.src[1] != nullptr && is_floating(node.src[0]->dtype) &&
           node.src[0]->dtype == node.src[1]->dtype;
}

/// A dtype as the shader's DTYPE specialisation constant.
///
/// The enum values are the shader's T_* codes by construction (cast.comp
/// mirrors vkml::DType). This is that cast with the precondition attached: the
/// elementwise, reduction and softmax shaders handle only the two floating
/// types, and an integer arriving here would silently select the f32 path and
/// read the wrong width. `supports()` is what guarantees it cannot.
[[nodiscard]] uint32_t spec_dtype(DType dt) {
    VKML_ASSERT(is_floating(dt), "shader DTYPE constant expects a floating dtype, got {}",
                dtype_name(dt));
    return static_cast<uint32_t>(dt);
}

/// OpKind -> shader code. Returns nullopt for anything this shader does not
/// implement, so a caller cannot silently get a copy: the previous form used a
/// `default: Copy` arm, which would have turned a forgotten entry here into
/// wrong output rather than a loud failure.
[[nodiscard]] std::optional<UnaryOp> to_unary_op(OpKind k) {
    switch (k) {
        case OpKind::Contiguous: return UnaryOp::Copy;
        case OpKind::Relu: return UnaryOp::Relu;
        case OpKind::Neg: return UnaryOp::Neg;
        case OpKind::Abs: return UnaryOp::Abs;
        case OpKind::Exp: return UnaryOp::Exp;
        case OpKind::Sign: return UnaryOp::Sign;
        case OpKind::Square: return UnaryOp::Square;
        case OpKind::Sqrt: return UnaryOp::Sqrt;
        case OpKind::Rsqrt: return UnaryOp::Rsqrt;
        case OpKind::Reciprocal: return UnaryOp::Reciprocal;
        case OpKind::Log: return UnaryOp::Log;
        case OpKind::Erf: return UnaryOp::Erf;
        case OpKind::Sin: return UnaryOp::Sin;
        case OpKind::Cos: return UnaryOp::Cos;
        case OpKind::Tanh: return UnaryOp::Tanh;
        case OpKind::Sigmoid: return UnaryOp::Sigmoid;
        case OpKind::Gelu: return UnaryOp::Gelu;
        case OpKind::Silu: return UnaryOp::Silu;
        case OpKind::Clamp: return UnaryOp::Clamp;
        default: return std::nullopt;
    }
}

/// Per-dispatch tracing, enabled with VKML_VULKAN_DEBUG=1.
///
/// Zero cost when off: the flag is read once into a static, and every call site
/// is a single predictable branch on it. Nothing is formatted unless enabled --
/// which matters, because std::format of a dozen fields per dispatch would be
/// far more expensive than the dispatch itself for small tensors.
[[nodiscard]] bool debug_dispatch_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("VKML_VULKAN_DEBUG");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return on;
}

/// Dumps small tensors after a dispatch, with VKML_VULKAN_DUMP=<max_elements>.
/// Off unless set; capped so a stray setting cannot print a 4M-element tensor.
[[nodiscard]] int64_t debug_dump_limit() {
    static const int64_t limit = []() -> int64_t {
        const char* v = std::getenv("VKML_VULKAN_DUMP");
        if (v == nullptr || v[0] == '\0') {
            return 0;
        }
        const long parsed = std::strtol(v, nullptr, 10);
        return parsed > 0 ? std::min<long>(parsed, 256) : 0;
    }();
    return limit;
}

void trace_dispatch(const Node& node, const char* kernel, const vk::KernelConfig& cfg,
                    uint32_t push_bytes, uint64_t groups, uint32_t subgroup_default) {
    std::string spec;
    for (size_t i = 0; i < cfg.spec_constants.size(); ++i) {
        spec += std::format("{}{}", i ? "," : "", cfg.spec_constants[i]);
    }
    VKML_LOG_INFO(
        "dispatch op={} kernel={} groups={}x1x1 wg={} subgroup={} spec=[{}] push={}B shared={}B "
        "shape={} dtype={}",
        op_name(node.op), kernel, groups, cfg.workgroup_size,
        cfg.required_subgroup_size != 0 ? std::to_string(cfg.required_subgroup_size)
                                        : std::format("driver({})", subgroup_default),
        spec, push_bytes, cfg.shared_memory_bytes, node.shape.str(), dtype_name(node.dtype));
}

[[nodiscard]] bool env_flag(const char* name, bool fallback) {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return fallback;
    }
    return v[0] != '0';
}

}  // namespace

std::string VulkanStats::describe() const {
    const double mib = 1024.0 * 1024.0;
    return std::format(
        "memory: {:.1f} MiB reserved in {} block(s), {:.1f} MiB in use (peak {:.1f}), "
        "{} live / {} total allocations, {} device allocations, fragmentation {:.1f}%\n"
        "execution: {} submissions, {} dispatches, {} pipelines",
        static_cast<double>(reserved_bytes) / mib, block_count,
        static_cast<double>(in_use_bytes) / mib, static_cast<double>(peak_in_use_bytes) / mib,
        live_allocations, total_allocations, device_allocations, fragmentation * 100.0, submissions,
        dispatches, pipelines);
}

// ---------------------------------------------------------------------------

struct VulkanBackend::Impl {
    vk::Context ctx;
    vk::Allocator allocator;
    vk::PipelineCache pipelines;
    vk::Recorder recorder;
    vk::StagingBuffer staging;
    DeviceCapabilities caps;

    /// Adapts the Vulkan allocator to the backend-agnostic Allocator interface.
    class StorageAllocator final : public vkml::Allocator {
    public:
        StorageAllocator(Impl& impl, Device dev) : impl_(impl), device_(dev) {}

        [[nodiscard]] std::shared_ptr<Storage> allocate(size_t nbytes) override {
            const vk::Allocation a = impl_.allocator.allocate(nbytes, vk::MemoryKind::DeviceLocal);

            // The device address is what a Storage's `data()` reports. It is
            // NOT a host pointer and must never be dereferenced on the CPU --
            // which is exactly what capabilities().host_accessible_buffers
            // being false tells every layer above.
            void* handle = reinterpret_cast<void*>(static_cast<uintptr_t>(a.address));

            // A Storage carries only the address, but vkCmdCopyBuffer needs the
            // block and offset behind it. Registering here is what lets
            // copy_from_host/copy_to_host recover the full Allocation.
            {
                const std::lock_guard<std::mutex> lock(impl_.map_mutex);
                impl_.live.emplace(a.address, a);
            }

            return std::make_shared<Storage>(handle, nbytes, device_, [this, a](void*, size_t) {
                {
                    const std::lock_guard<std::mutex> lock(impl_.map_mutex);
                    impl_.live.erase(a.address);
                }
                impl_.allocator.free(a);
            });
        }

        [[nodiscard]] std::string_view name() const noexcept override { return "vulkan"; }

        [[nodiscard]] Device device() const noexcept override { return device_; }

        [[nodiscard]] size_t live_bytes() const noexcept override {
            return impl_.allocator.stats().in_use_bytes;
        }

    private:
        Impl& impl_;
        Device device_;
    };

    std::unique_ptr<StorageAllocator> storage_allocator;

    /// 0 = use each kernel's default.
    uint32_t subgroup_override = 0;

    // Allocation lookup by device address, so a Storage (which only carries the
    // address) can be turned back into a block+offset for buffer copies.
    std::mutex map_mutex;
    std::unordered_map<uint64_t, vk::Allocation> live;

    /// Scratch for split-K partials: `splits` consecutive copies of the output.
    ///
    /// Backend-internal by design. It is not a tensor, never escapes to the
    /// graph or the planner, and needs no public API -- which is what keeps
    /// split-K from leaking into layers that have no business knowing about it.
    /// Grown on demand and reused, so a training loop pays the allocation once;
    /// the same lifetime strategy as llama.cpp's prealloc_split_k.
    vk::Allocation splitk_ws{};

    /// Returns a device address for at least `bytes` of split-K workspace.
    [[nodiscard]] uint64_t splitk_workspace(uint64_t bytes) {
        if (splitk_ws.valid() && splitk_ws.size >= bytes) {
            return splitk_ws.address;
        }
        if (splitk_ws.valid()) {
            allocator.free(splitk_ws);
            splitk_ws = {};
        }
        splitk_ws = allocator.allocate(bytes, vk::MemoryKind::DeviceLocal);
        return splitk_ws.address;
    }

    Impl(int index, bool validation, uint64_t staging_bytes)
        : ctx(index, validation), allocator(ctx), pipelines(ctx), recorder(ctx, allocator),
          staging(ctx, allocator, recorder, staging_bytes), caps(ctx.capabilities()) {}
};

VulkanBackend::VulkanBackend(int device_index, bool enable_validation) {
    // 32 MiB of staging. Large enough that a typical tensor moves in one chunk,
    // small enough not to waste host memory; transfers larger than this are
    // chunked automatically.
    constexpr uint64_t kStagingBytes = 32ULL * 1024 * 1024;
    impl_ = std::make_unique<Impl>(device_index, enable_validation, kStagingBytes);
    device_ = Device::vulkan(device_index);
    name_ = std::format("vulkan:{}", device_index);
    impl_->storage_allocator = std::make_unique<Impl::StorageAllocator>(*impl_, device_);
}

VulkanBackend::~VulkanBackend() {
    if (impl_) {
        impl_->recorder.wait_idle();
    }
}

const DeviceCapabilities& VulkanBackend::capabilities() const noexcept { return impl_->caps; }

Allocator& VulkanBackend::allocator() { return *impl_->storage_allocator; }

bool VulkanBackend::supports(const Node& node) const {
    if (is_view_op(node.op) || node.is_leaf()) {
        return true;
    }
    // Anything not listed cannot run on this backend AT ALL: Executor::realize
    // throws NotImplementedError rather than routing the node elsewhere.
    //
    // An earlier version of this comment claimed unported ops "fall back to the
    // CPU, slow but not wrong". That describes the design in
    // docs/ARCHITECTURE.md 3 Fork 3, not the code. supports() is the predicate
    // that fallback would need, but the graph splitting that would consume it
    // was never built -- see the executor, which says as much.
    //
    // The practical consequence, verified: a model touching one unported op
    // cannot run on Vulkan, it does not run slowly. Do not restore the old
    // wording without making it true first.
    switch (node.op) {
        // Same kernel: fill is arange with a zero slope. F32 only, because the
        // shader writes through F32Buf -- an I64 arange still falls to the CPU.
        case OpKind::Full:
        case OpKind::Arange: return is_floating(node.dtype);
        // Counter-based, so the value depends only on the element index and the
        // push constants -- nothing to seed per invocation, nothing shared.
        case OpKind::Rand: return node.dtype == DType::F32;
        // The output is freshly allocated and contiguous; the gradient may be
        // any view, which its own strides carry.
        case OpKind::SliceBackward:
            return node.dtype == DType::F32 && node.src[0] != nullptr &&
                   node.src[0]->dtype == DType::F32 && node.shape.is_contiguous();
        // Unary elementwise: one shader, one specialisation constant per op.
        case OpKind::Contiguous:
        case OpKind::Relu:
        case OpKind::Neg:
        case OpKind::Abs:
        case OpKind::Exp:
        case OpKind::Sign:
        case OpKind::Square:
        case OpKind::Sqrt:
        case OpKind::Rsqrt:
        case OpKind::Reciprocal:
        case OpKind::Log:
        case OpKind::Erf:
        case OpKind::Sin:
        case OpKind::Cos:
        case OpKind::Tanh:
        case OpKind::Sigmoid:
        case OpKind::Gelu:
        case OpKind::Silu:
        // Contiguous is here too: it reaches unary.comp as a copy, so it gains
        // f16 with the rest rather than needing its own path.
        case OpKind::Clamp:
            return is_floating(node.dtype) && node.src[0] != nullptr &&
                   node.src[0]->dtype == node.dtype;
        // Binary elementwise arithmetic: floating in, the same floating out.
        case OpKind::Add:
        case OpKind::Sub:
        case OpKind::Mul:
        case OpKind::Div:
        case OpKind::Pow:
        case OpKind::Maximum:
        case OpKind::Minimum:
            return is_floating(node.dtype) && binary_srcs_are_float(node) &&
                   node.src[0]->dtype == node.dtype;
        // Comparisons: floating in, Bool out. The output dtype differs from the
        // inputs', so both ends are checked rather than inferring one -- getting
        // that backwards is precisely the defect the CPU kernel carried.
        case OpKind::Equal:
        case OpKind::Less:
        case OpKind::Greater:
        case OpKind::LessEqual:
        case OpKind::GreaterEqual:
        case OpKind::NotEqual: return node.dtype == DType::Bool && binary_srcs_are_float(node);
        // Ternary select. The condition is Bool while the values are F32, so
        // all three sources are checked rather than assumed uniform.
        case OpKind::Where:
            return is_floating(node.dtype) && node.src[0] != nullptr &&
                   node.src[0]->dtype == DType::Bool && node.src[1] != nullptr &&
                   node.src[1]->dtype == node.dtype && node.src[2] != nullptr &&
                   node.src[2]->dtype == node.dtype;
        // Triangular masks. Rank >= 2 is guaranteed by the API, but the shader
        // indexes the last two extents directly, so it is re-checked here
        // rather than trusted across a layer boundary.
        case OpKind::Triu:
        case OpKind::Tril: return node.dtype == DType::F32 && node.shape.ndim() >= 2;
        // The CPU kernel is byte-wise and so dtype-generic; the shader reads
        // through F32Buf, so it claims only F32 and everything else falls back.
        case OpKind::Cat: return node.dtype == DType::F32 && binary_srcs_are_float(node);
        // Values are F32, the index is I64. Both shaders read the index through
        // I64Buf, so the dtype is checked rather than assumed.
        // Window geometry is carried in params and the output is freshly
        // allocated, so only the element type needs checking.
        case OpKind::Im2Col:
        case OpKind::Col2Im:
            return node.dtype == DType::F32 && node.src[0] != nullptr &&
                   node.src[0]->dtype == DType::F32;
        // The shaders index planes directly rather than through an Operand, so
        // both operands must be contiguous as well as F32.
        case OpKind::MaxPool2d:
            return node.dtype == DType::F32 && node.src[0] != nullptr &&
                   node.src[0]->dtype == DType::F32 && node.src[0]->shape.is_contiguous();
        case OpKind::MaxPool2dBackward:
            return node.dtype == DType::F32 && node.src[0] != nullptr &&
                   node.src[0]->dtype == DType::F32 && node.src[0]->shape.is_contiguous() &&
                   node.src[1] != nullptr && node.src[1]->shape.is_contiguous();
        case OpKind::IndexSelect:
        case OpKind::ScatterAdd:
            return node.dtype == DType::F32 && node.src[0] != nullptr &&
                   node.src[0]->dtype == DType::F32 && node.src[1] != nullptr &&
                   node.src[1]->dtype == DType::I64;
        case OpKind::Cast: return true;
        // Value reductions keep their input's dtype. The shader accumulates in
        // shared floats whatever it reads, so fp32 accumulation holds for f16
        // input without the kernel changing (ARCHITECTURE.md 7.3).
        case OpKind::Sum:
        case OpKind::Mean:
        case OpKind::Max:
        case OpKind::Min:
            return is_floating(node.dtype) && node.src[0] != nullptr &&
                   node.src[0]->dtype == node.dtype;
        // Index reductions: floating in, I64 out, so the checked dtype is the
        // source's and not this node's.
        case OpKind::ArgMax:
        case OpKind::ArgMin:
            return node.dtype == DType::I64 && node.src[0] != nullptr &&
                   is_floating(node.src[0]->dtype);
        case OpKind::Softmax:
        case OpKind::LogSoftmax:
            return is_floating(node.dtype) && node.src[0] != nullptr &&
                   node.src[0]->dtype == node.dtype;
        case OpKind::Matmul:
            // Operands arrive normalised to rank 4 by the graph builder, and
            // the output is always freshly allocated and contiguous.
            //
            // Shared tiles, register accumulators and split-K partials all stay
            // f32 whatever the operands are; only the global loads and the
            // final store narrow (ARCHITECTURE.md 7.3).
            return is_floating(node.dtype) && binary_srcs_are_float(node) &&
                   node.src[0]->dtype == node.dtype && node.shape.ndim() == 4 &&
                   node.shape.is_contiguous();
        default: return false;
    }
}

void VulkanBackend::compute(std::span<Node* const> nodes) {
    vk::Recorder& rec = impl_->recorder;
    vk::PipelineCache& pipes = impl_->pipelines;

    // Subgroup width is left to the driver for these kernels: they are purely
    // bandwidth-bound elementwise work with no cross-lane communication, so the
    // width does not matter. Reductions at M2 will pin it (llama.cpp uses
    // wave64 for exactly those on RDNA1).
    const uint32_t wg = 256;

    std::vector<Node*> traced;

    rec.begin();

    // If a kernel throws, the recording must not be left open -- otherwise the
    // next begin() asserts and hides the original error.
    struct RecordingGuard {
        vk::Recorder& rec;

        ~RecordingGuard() { rec.abort_recording(); }
    } guard{rec};

    for (Node* node : nodes) {
        if (node == nullptr || is_view_op(node->op) || node->is_leaf()) {
            continue;
        }
        VKML_ASSERT(node->is_realized(), "node '{}' has no storage", op_name(node->op));

        const auto address_of = [](const Node& n) {
            return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(n.data()));
        };
        const auto n_elems = static_cast<uint32_t>(node->shape.numel());
        if (n_elems == 0) {
            continue;
        }

        switch (node->op) {
            case OpKind::Full:
            case OpKind::Arange: {
                vk::KernelConfig cfg;
                cfg.workgroup_size = wg;
                cfg.spec_constants = {wg, spec_dtype(node->dtype)};

                // A fill is an arange whose slope is zero, so both reach the
                // same kernel and differ only in what the push constants say.
                float base = 0.0F;
                float step = 0.0F;
                if (node->op == OpKind::Arange) {
                    const auto ap = node->params.get<ArangeParams>();
                    base = static_cast<float>(ap.start);
                    step = static_cast<float>(ap.step);
                } else {
                    base = static_cast<float>(node->params.get<FullParams>().value);
                }

                const FillPush push{address_of(*node), n_elems, base, step};
                if (debug_dispatch_enabled()) {
                    trace_dispatch(*node, "fill", cfg, sizeof(FillPush), (n_elems + wg - 1) / wg,
                                   impl_->caps.subgroup_size);
                }
                rec.dispatch(pipes.get("fill", spv::fill, spv::fill_size, sizeof(FillPush), cfg),
                             &push, sizeof(push), n_elems);
                break;
            }

            case OpKind::Contiguous:
            case OpKind::Relu:
            case OpKind::Neg:
            case OpKind::Abs:
            case OpKind::Exp:
            case OpKind::Sign:
            case OpKind::Square:
            case OpKind::Sqrt:
            case OpKind::Rsqrt:
            case OpKind::Reciprocal:
            case OpKind::Log:
            case OpKind::Erf:
            case OpKind::Sin:
            case OpKind::Cos:
            case OpKind::Tanh:
            case OpKind::Sigmoid:
            case OpKind::Gelu:
            case OpKind::Silu:
            case OpKind::Clamp: {
                const Node& src = *node->src[0];
                const size_t esz = dtype_size(node->dtype);

                const std::optional<UnaryOp> op = to_unary_op(node->op);
                VKML_ASSERT(op.has_value(),
                            "unary dispatch reached for '{}', which to_unary_op "
                            "does not map -- supports() and this switch disagree",
                            op_name(node->op));

                const bool contiguous = src.shape.is_contiguous() && node->shape.is_contiguous();

                vk::KernelConfig cfg;
                cfg.workgroup_size = wg;
                cfg.spec_constants = {wg, static_cast<uint32_t>(*op), contiguous ? 1U : 0U,
                                      spec_dtype(node->dtype)};

                UnaryPush push{};
                push.src = address_of(src);
                push.dst = address_of(*node);
                push.n = n_elems;
                // An absent bound becomes an infinity, so the shader needs no
                // has_lo/has_hi flags and NaN still passes through: neither
                // `NaN < -inf` nor `NaN > +inf` is true.
                push.clamp_lo = -std::numeric_limits<float>::infinity();
                push.clamp_hi = std::numeric_limits<float>::infinity();
                if (node->op == OpKind::Clamp) {
                    const auto cp = node->params.get<ClampParams>();
                    if (cp.has_lo) {
                        push.clamp_lo = cp.lo;
                    }
                    if (cp.has_hi) {
                        push.clamp_hi = cp.hi;
                    }
                }
                push.in_op = to_gpu_operand(src.shape, esz);
                push.out_op = to_gpu_operand(node->shape, esz);

                if (debug_dispatch_enabled()) {
                    trace_dispatch(*node, "unary", cfg, sizeof(UnaryPush), (n_elems + wg - 1) / wg,
                                   impl_->caps.subgroup_size);
                }
                rec.dispatch(
                    pipes.get("unary", spv::unary, spv::unary_size, sizeof(UnaryPush), cfg), &push,
                    sizeof(push), n_elems);
                break;
            }

            case OpKind::Add:
            case OpKind::Sub:
            case OpKind::Mul:
            case OpKind::Div:
            case OpKind::Pow:
            case OpKind::Maximum:
            case OpKind::Minimum:
            case OpKind::Equal:
            case OpKind::Less:
            case OpKind::Greater:
            case OpKind::LessEqual:
            case OpKind::GreaterEqual:
            case OpKind::NotEqual: {
                const Node& a = *node->src[0];
                const Node& b = *node->src[1];

                const std::optional<BinaryOp> op = to_binary_op(node->op);
                VKML_ASSERT(op.has_value(),
                            "binary dispatch reached for '{}', which to_binary_op does not "
                            "map -- supports() and this switch disagree",
                            op_name(node->op));

                // Inputs are always F32; comparisons narrow the output to Bool,
                // so the destination element size differs from the source one
                // and the two operands cannot share an itemsize.
                const size_t in_esz = dtype_size(a.dtype);
                const size_t out_esz = dtype_size(node->dtype);

                const bool contiguous = a.shape.is_contiguous() && b.shape.is_contiguous() &&
                                        node->shape.is_contiguous();

                vk::KernelConfig cfg;
                cfg.workgroup_size = wg;
                cfg.spec_constants = {wg, static_cast<uint32_t>(*op), contiguous ? 1U : 0U,
                                      spec_dtype(a.dtype)};

                BinaryPush push{};
                push.a = address_of(a);
                push.b = address_of(b);
                push.dst = address_of(*node);
                push.n = n_elems;
                // Both sources arrive already broadcast to the output shape, so
                // a broadcast axis is simply one with stride 0 and the shader
                // needs no shape reconciliation of its own.
                push.a_op = to_gpu_operand(a.shape, in_esz);
                push.b_op = to_gpu_operand(b.shape, in_esz);
                push.out_op = to_gpu_operand(node->shape, out_esz);

                if (debug_dispatch_enabled()) {
                    trace_dispatch(*node, "binary", cfg, sizeof(BinaryPush),
                                   (n_elems + wg - 1) / wg, impl_->caps.subgroup_size);
                }
                rec.dispatch(
                    pipes.get("binary", spv::binary, spv::binary_size, sizeof(BinaryPush), cfg),
                    &push, sizeof(push), n_elems);
                break;
            }

            case OpKind::SliceBackward: {
                const Node& grad = *node->src[0];
                const size_t esz = dtype_size(node->dtype);
                const auto sp = node->params.get<SliceParams>();

                SliceBackwardPush push{};
                push.src = address_of(grad);
                push.dst = address_of(*node);
                push.n = n_elems;
                // Operands are right-padded to rank 4, so the axis moves with
                // them; using the unpadded index would address the wrong extent.
                push.axis = static_cast<int32_t>(kMaxDims - node->shape.ndim() + sp.axis);
                push.start = static_cast<int32_t>(sp.start);
                push.stop = static_cast<int32_t>(sp.stop);
                push.step = static_cast<int32_t>(sp.step);
                push.grad_op = to_gpu_operand(grad.shape, esz);
                push.out_op = to_gpu_operand(node->shape, esz);

                vk::KernelConfig cfg;
                cfg.workgroup_size = wg;
                cfg.spec_constants = {wg};

                if (debug_dispatch_enabled()) {
                    trace_dispatch(*node, "slice_backward", cfg, sizeof(SliceBackwardPush),
                                   (n_elems + wg - 1) / wg, impl_->caps.subgroup_size);
                }
                rec.dispatch(pipes.get("slice_backward", spv::slice_backward,
                                       spv::slice_backward_size, sizeof(SliceBackwardPush), cfg),
                             &push, sizeof(push), n_elems);
                break;
            }

            case OpKind::Rand: {
                vk::KernelConfig cfg;
                cfg.workgroup_size = wg;
                cfg.spec_constants = {wg};

                const auto rp = node->params.get<RandParams>();
                const RandPush push{address_of(*node), n_elems, rp.seed, rp.offset};

                if (debug_dispatch_enabled()) {
                    trace_dispatch(*node, "rand", cfg, sizeof(RandPush), (n_elems + wg - 1) / wg,
                                   impl_->caps.subgroup_size);
                }
                rec.dispatch(pipes.get("rand", spv::rand, spv::rand_size, sizeof(RandPush), cfg),
                             &push, sizeof(push), n_elems);
                break;
            }

            case OpKind::MaxPool2d:
            case OpKind::MaxPool2dBackward: {
                const bool backward = node->op == OpKind::MaxPool2dBackward;
                const Node& src = *node->src[0];
                const auto up = node->params.get<UnfoldParams>();

                // Forward reads the input directly; the adjoint reads the
                // gradient from src[0] and the original input from src[1],
                // which is what the argmax is recomputed from.
                const Node& image = backward ? *node->src[1] : src;

                // The pooled extent lives on whichever operand is pooled-shaped:
                // the output going forward, the gradient coming back.
                const Shape& pooled = backward ? src.shape : node->shape;

                PoolPush push{};
                push.src = address_of(src);
                push.image = address_of(image);
                push.dst = address_of(*node);
                push.n = n_elems;
                push.kernel_h = up.kernel_h;
                push.kernel_w = up.kernel_w;
                push.stride_h = up.stride_h;
                push.stride_w = up.stride_w;
                push.pad_h = up.pad_h;
                push.pad_w = up.pad_w;
                push.dilation_h = up.dilation_h;
                push.dilation_w = up.dilation_w;
                push.image_h = up.image_h;
                push.image_w = up.image_w;
                push.out_h = static_cast<int32_t>(pooled.dim(2));
                push.out_w = static_cast<int32_t>(pooled.dim(3));

                vk::KernelConfig cfg;
                cfg.workgroup_size = wg;
                cfg.spec_constants = {wg, backward ? 1U : 0U};

                if (debug_dispatch_enabled()) {
                    trace_dispatch(*node, "max_pool2d", cfg, sizeof(PoolPush),
                                   (n_elems + wg - 1) / wg, impl_->caps.subgroup_size);
                }
                rec.dispatch(pipes.get("max_pool2d", spv::max_pool2d, spv::max_pool2d_size,
                                       sizeof(PoolPush), cfg),
                             &push, sizeof(push), n_elems);
                break;
            }

            case OpKind::Im2Col:
            case OpKind::Col2Im: {
                const bool extracting = node->op == OpKind::Im2Col;
                const Node& src = *node->src[0];
                const size_t esz = dtype_size(node->dtype);
                const auto up = node->params.get<UnfoldParams>();

                const auto span = [](int64_t extent, int32_t k, int32_t stride, int32_t pad,
                                     int32_t dilation) {
                    return static_cast<int32_t>(
                        (extent + 2LL * pad - (static_cast<int64_t>(dilation) * (k - 1) + 1)) /
                            stride +
                        1);
                };

                // Rows are the channel-patch axis, which lives on the column
                // side: it is the source for col2im and the output for im2col.
                const int32_t rows =
                    static_cast<int32_t>(extracting ? node->shape.dim(1) : src.shape.dim(1));

                UnfoldPush push{};
                push.src = address_of(src);
                push.dst = address_of(*node);
                push.n = n_elems;
                push.kernel_h = up.kernel_h;
                push.kernel_w = up.kernel_w;
                push.stride_h = up.stride_h;
                push.stride_w = up.stride_w;
                push.pad_h = up.pad_h;
                push.pad_w = up.pad_w;
                push.dilation_h = up.dilation_h;
                push.dilation_w = up.dilation_w;
                push.image_h = up.image_h;
                push.image_w = up.image_w;
                push.out_h = span(up.image_h, up.kernel_h, up.stride_h, up.pad_h, up.dilation_h);
                push.out_w = span(up.image_w, up.kernel_w, up.stride_w, up.pad_w, up.dilation_w);
                push.rows = rows;
                push.channels = rows / (up.kernel_h * up.kernel_w);
                push.in_op = to_gpu_operand(src.shape, esz);

                vk::KernelConfig cfg;
                cfg.workgroup_size = wg;
                cfg.spec_constants = {wg};

                const char* name = extracting ? "im2col" : "col2im";
                const uint32_t* code = extracting ? spv::im2col : spv::col2im;
                const size_t code_size = extracting ? spv::im2col_size : spv::col2im_size;

                if (debug_dispatch_enabled()) {
                    trace_dispatch(*node, name, cfg, sizeof(UnfoldPush), (n_elems + wg - 1) / wg,
                                   impl_->caps.subgroup_size);
                }
                rec.dispatch(pipes.get(name, code, code_size, sizeof(UnfoldPush), cfg), &push,
                             sizeof(push), n_elems);
                break;
            }

            case OpKind::IndexSelect:
            case OpKind::ScatterAdd: {
                const bool gather = node->op == OpKind::IndexSelect;
                const Node& src = *node->src[0];
                const Node& index = *node->src[1];
                const size_t esz = dtype_size(node->dtype);
                const int axis = node->params.get<AxisParams>().axis;
                const int nd = node->shape.ndim();

                uint32_t inner = 1;
                for (int i = axis + 1; i < nd; ++i) {
                    inner *= static_cast<uint32_t>(node->shape.dim(i));
                }

                vk::KernelConfig cfg;
                cfg.workgroup_size = wg;
                cfg.spec_constants = {wg};

                GatherPush push{};
                push.src = address_of(src);
                push.index = address_of(index);
                push.dst = address_of(*node);
                push.n = n_elems;
                push.inner = inner;
                // For the gather, `out_extent` is the index length and
                // `src_extent` the axis being read. The scatter swaps them:
                // it owns a destination row and scans the index instead.
                push.out_extent = static_cast<uint32_t>(node->shape.dim(axis));
                push.src_extent = static_cast<uint32_t>(src.shape.dim(axis));
                push.src_op = to_gpu_operand(src.shape, esz);
                push.out_op = to_gpu_operand(node->shape, esz);

                const char* name = gather ? "index_select" : "scatter_add";
                const uint32_t* code = gather ? spv::index_select : spv::scatter_add;
                const size_t code_size = gather ? spv::index_select_size : spv::scatter_add_size;

                if (debug_dispatch_enabled()) {
                    trace_dispatch(*node, name, cfg, sizeof(GatherPush), (n_elems + wg - 1) / wg,
                                   impl_->caps.subgroup_size);
                }
                rec.dispatch(pipes.get(name, code, code_size, sizeof(GatherPush), cfg), &push,
                             sizeof(push), n_elems);
                break;
            }

            case OpKind::Cat: {
                const Node& a = *node->src[0];
                const Node& b = *node->src[1];
                const size_t esz = dtype_size(node->dtype);
                const int axis = node->params.get<AxisParams>().axis;
                const int nd = node->shape.ndim();

                uint32_t inner = 1;
                for (int i = axis + 1; i < nd; ++i) {
                    inner *= static_cast<uint32_t>(node->shape.dim(i));
                }

                const bool contiguous = a.shape.is_contiguous() && b.shape.is_contiguous() &&
                                        node->shape.is_contiguous();

                vk::KernelConfig cfg;
                cfg.workgroup_size = wg;
                cfg.spec_constants = {wg, contiguous ? 1U : 0U, spec_dtype(node->dtype)};

                CatPush push{};
                push.a = address_of(a);
                push.b = address_of(b);
                push.dst = address_of(*node);
                push.n = n_elems;
                push.inner = inner;
                push.out_extent = static_cast<uint32_t>(node->shape.dim(axis));
                push.a_extent = static_cast<uint32_t>(a.shape.dim(axis));
                push.b_extent = static_cast<uint32_t>(b.shape.dim(axis));
                push.a_op = to_gpu_operand(a.shape, esz);
                push.b_op = to_gpu_operand(b.shape, esz);
                push.out_op = to_gpu_operand(node->shape, esz);

                if (debug_dispatch_enabled()) {
                    trace_dispatch(*node, "cat", cfg, sizeof(CatPush), (n_elems + wg - 1) / wg,
                                   impl_->caps.subgroup_size);
                }
                rec.dispatch(pipes.get("cat", spv::cat, spv::cat_size, sizeof(CatPush), cfg), &push,
                             sizeof(push), n_elems);
                break;
            }

            case OpKind::Triu:
            case OpKind::Tril: {
                const Node& src = *node->src[0];
                const size_t esz = dtype_size(node->dtype);
                const auto tp = node->params.get<TriParams>();

                const int nd = node->shape.ndim();
                const bool contiguous = src.shape.is_contiguous() && node->shape.is_contiguous();

                vk::KernelConfig cfg;
                cfg.workgroup_size = wg;
                cfg.spec_constants = {wg, node->op == OpKind::Triu ? 1U : 0U, contiguous ? 1U : 0U};

                TriPush push{};
                push.src = address_of(src);
                push.dst = address_of(*node);
                push.n = n_elems;
                push.height = static_cast<uint32_t>(node->shape.dim(nd - 2));
                push.width = static_cast<uint32_t>(node->shape.dim(nd - 1));
                push.diagonal = tp.diagonal;
                push.in_op = to_gpu_operand(src.shape, esz);
                push.out_op = to_gpu_operand(node->shape, esz);

                if (debug_dispatch_enabled()) {
                    trace_dispatch(*node, "tri", cfg, sizeof(TriPush), (n_elems + wg - 1) / wg,
                                   impl_->caps.subgroup_size);
                }
                rec.dispatch(pipes.get("tri", spv::tri, spv::tri_size, sizeof(TriPush), cfg), &push,
                             sizeof(push), n_elems);
                break;
            }

            case OpKind::Where: {
                const Node& cond = *node->src[0];
                const Node& a = *node->src[1];
                const Node& b = *node->src[2];

                const size_t cond_esz = dtype_size(cond.dtype);  // Bool: 1 byte
                const size_t val_esz = dtype_size(node->dtype);  // F32: 4 bytes

                const bool contiguous = cond.shape.is_contiguous() && a.shape.is_contiguous() &&
                                        b.shape.is_contiguous() && node->shape.is_contiguous();

                vk::KernelConfig cfg;
                cfg.workgroup_size = wg;
                cfg.spec_constants = {wg, contiguous ? 1U : 0U, spec_dtype(node->dtype)};

                WherePush push{};
                push.cond = address_of(cond);
                push.a = address_of(a);
                push.b = address_of(b);
                push.dst = address_of(*node);
                push.n = n_elems;
                push.cond_op = to_gpu_operand(cond.shape, cond_esz);
                push.a_op = to_gpu_operand(a.shape, val_esz);
                push.b_op = to_gpu_operand(b.shape, val_esz);
                push.out_op = to_gpu_operand(node->shape, val_esz);

                if (debug_dispatch_enabled()) {
                    trace_dispatch(*node, "where", cfg, sizeof(WherePush), (n_elems + wg - 1) / wg,
                                   impl_->caps.subgroup_size);
                }
                rec.dispatch(
                    pipes.get("where", spv::where, spv::where_size, sizeof(WherePush), cfg), &push,
                    sizeof(push), n_elems);
                break;
            }

            case OpKind::Cast: {
                const Node& src = *node->src[0];
                VKML_CHECK(src.shape.is_contiguous() && node->shape.is_contiguous(), ShapeError,
                           "vulkan cast requires contiguous operands");

                vk::KernelConfig cfg;
                cfg.workgroup_size = wg;
                cfg.spec_constants = {wg, static_cast<uint32_t>(src.dtype),
                                      static_cast<uint32_t>(node->dtype)};

                const CastPush push{address_of(src), address_of(*node), n_elems};
                rec.dispatch(pipes.get("cast", spv::cast, spv::cast_size, sizeof(CastPush), cfg),
                             &push, sizeof(push), n_elems);
                break;
            }

            case OpKind::Sum:
            case OpKind::Mean:
            case OpKind::Max:
            case OpKind::Min:
            case OpKind::ArgMax:
            case OpKind::ArgMin: {
                const Node& src = *node->src[0];
                const auto rp = node->params.get<ReduceParams>();
                const uint32_t mask =
                    rp.axes_mask != 0
                        ? rp.axes_mask
                        : (src.shape.ndim() <= 0
                               ? 0U
                               : (1U << static_cast<uint32_t>(src.shape.ndim())) - 1U);

                const SplitShape split = split_for_reduce(src.shape, mask);
                const auto n_out = static_cast<uint32_t>(split.kept.numel());
                const auto n_red = static_cast<uint32_t>(split.reduced.numel());
                if (n_out == 0 || n_red == 0) {
                    break;  // empty reduction; nothing defined to write
                }

                uint32_t op_id = 0;
                switch (node->op) {
                    case OpKind::Sum: op_id = 0; break;
                    case OpKind::Mean: op_id = 1; break;
                    case OpKind::Max: op_id = 2; break;
                    case OpKind::Min: op_id = 3; break;
                    case OpKind::ArgMax: op_id = 4; break;
                    default: op_id = 5; break;
                }

                // wave64 is pinned here. Reductions are the one kernel family
                // where subgroup width matters, because the shared-memory tree
                // is cross-lane work; llama.cpp's RDNA1 table selects 64 for
                // exactly these ops on this chip. Left to the driver where
                // subgroup size control is unavailable.
                vk::KernelConfig cfg;
                cfg.workgroup_size = wg;
                cfg.shared_memory_bytes = wg * (sizeof(float) + sizeof(uint32_t));
                // Subgroup width is left to the driver by default.
                //
                // wave64 was pinned here initially, inherited from llama.cpp's
                // RDNA1 table. Measuring it on OUR kernels (timestamp queries,
                // 12 workload/op combinations) showed wave32 winning 8 of 12
                // with every difference inside +/-7% -- not enough to justify a
                // hardware-specific constant. The override remains for
                // experimentation and future autotuning.
                if (impl_->subgroup_override != 0) {
                    cfg.required_subgroup_size = impl_->subgroup_override;
                }
                cfg.spec_constants = {wg, op_id, wg, spec_dtype(src.dtype)};

                const size_t esz = dtype_size(src.dtype);
                ReducePush push{};
                push.src = address_of(src);
                push.dst = address_of(*node);
                push.n_out = n_out;
                push.n_red = n_red;
                push.kept = to_gpu_operand(split.kept, esz);
                push.reduced = to_gpu_operand(split.reduced, esz);

                if (debug_dispatch_enabled()) {
                    trace_dispatch(*node, "reduce", cfg, sizeof(ReducePush), n_out,
                                   impl_->caps.subgroup_size);
                }
                rec.dispatch_groups(
                    pipes.get("reduce", spv::reduce, spv::reduce_size, sizeof(ReducePush), cfg),
                    &push, sizeof(push), n_out);
                break;
            }

            case OpKind::Softmax:
            case OpKind::LogSoftmax: {
                const Node& src = *node->src[0];
                const auto ap = node->params.get<AxisParams>();
                const uint32_t mask = 1U << static_cast<uint32_t>(ap.axis);

                // The output is freshly allocated and so may not share the
                // input's strides -- the input can be a transposed or broadcast
                // view. Both layouts are split, and the shader indexes each
                // with its own operands.
                const SplitShape in_split = split_for_reduce(src.shape, mask);
                const SplitShape out_split = split_for_reduce(node->shape, mask);

                const auto n_out = static_cast<uint32_t>(in_split.kept.numel());
                const auto n_axis = static_cast<uint32_t>(in_split.reduced.numel());
                if (n_out == 0 || n_axis == 0) {
                    break;
                }

                vk::KernelConfig cfg;
                cfg.workgroup_size = wg;
                cfg.shared_memory_bytes = wg * sizeof(float);
                if (impl_->subgroup_override != 0) {
                    cfg.required_subgroup_size = impl_->subgroup_override;  // as reductions
                }
                // M4-R4 theory probe: raises LDS only, so P1'' can be tested on a
                // kernel it was not derived from. 0 in production.
                static const uint32_t sm_pad = [] {
                    const char* v = std::getenv("VKML_SOFTMAX_PAD_KB");
                    return v == nullptr ? 0U : static_cast<uint32_t>(std::atoi(v)) * 256U;
                }();
                cfg.shared_memory_bytes += sm_pad * sizeof(float);
                cfg.spec_constants = {wg, node->op == OpKind::LogSoftmax ? 1U : 0U, wg, sm_pad,
                                      spec_dtype(node->dtype)};

                const size_t esz = dtype_size(src.dtype);
                SoftmaxPush push{};
                push.src = address_of(src);
                push.dst = address_of(*node);
                push.n_out = n_out;
                push.n_axis = n_axis;
                push.in_kept = to_gpu_operand(in_split.kept, esz);
                push.in_axis = to_gpu_operand(in_split.reduced, esz);
                push.out_kept = to_gpu_operand(out_split.kept, esz);
                push.out_axis = to_gpu_operand(out_split.reduced, esz);

                if (debug_dispatch_enabled()) {
                    trace_dispatch(*node, "softmax", cfg, sizeof(SoftmaxPush), n_out,
                                   impl_->caps.subgroup_size);
                }
                rec.dispatch_groups(
                    pipes.get("softmax", spv::softmax, spv::softmax_size, sizeof(SoftmaxPush), cfg),
                    &push, sizeof(push), n_out);
                break;
            }

            case OpKind::Matmul: {
                const Node& a = *node->src[0];
                const Node& b = *node->src[1];

                const auto b0 = static_cast<uint32_t>(node->shape.dim(0));
                const auto b1 = static_cast<uint32_t>(node->shape.dim(1));
                const auto m = static_cast<uint32_t>(node->shape.dim(2));
                const auto n = static_cast<uint32_t>(node->shape.dim(3));
                const auto k = static_cast<uint32_t>(a.shape.dim(3));

                VKML_ASSERT(static_cast<uint32_t>(b.shape.dim(2)) == k,
                            "matmul inner dimensions disagree: {} vs {}", k, b.shape.dim(2));

                const uint32_t total = b0 * b1 * m * n;
                if (total == 0 || k == 0) {
                    break;
                }

                // -- GEMV ---------------------------------------------------
                //
                // One workgroup per OUTPUT ELEMENT. The tiled kernel collapses
                // to ceil(M/32) workgroups at N=1 -- 128 against this GPU's 288
                // concurrent slots, i.e. 44 % fill, where 5g says throughput
                // falls off a cliff. This restores the grid to M*N.
                //
                // Bit-identical to the tiled kernel by the alignment lemma
                // (docs/SPLIT_K_DESIGN.md 2.3) applied across LANES instead of
                // across workgroups: each lane folds a contiguous run of 2^q
                // K-tiles, and the cross-lane reduction folds adjacent pairs so
                // it reproduces the carry stack's association exactly.
                if (gemv_mode() == GemvMode::Forced) {
                    const uint32_t ktiles = (k + 31) / 32;
                    constexpr uint32_t kGemvWg = 64;
                    // Smallest power-of-two chunk that lets WG lanes cover K.
                    const uint32_t passes = (ktiles + kGemvWg - 1) / kGemvWg;
                    uint32_t levels = 1;
                    while ((1U << levels) < passes) {
                        ++levels;
                    }
                    ++levels;
                    for (const uint32_t bucket : {4U, 6U, 8U, 10U, 12U, 16U}) {
                        if (levels <= bucket) {
                            levels = bucket;
                            break;
                        }
                    }

                    vk::KernelConfig gcfg;
                    gcfg.workgroup_size = kGemvWg;
                    gcfg.shared_memory_bytes = (kGemvWg + 2 * kGemvWg * 32) * sizeof(float);
                    // M4-R2: LDS padding, discriminator only. Default 0.
                    static const uint32_t pad_floats = [] {
                        const char* v = std::getenv("VKML_GEMV_PAD_KB");
                        return v == nullptr ? 0U : static_cast<uint32_t>(std::atoi(v)) * 256U;
                    }();
                    gcfg.shared_memory_bytes += pad_floats * sizeof(float);
                    gcfg.spec_constants = {kGemvWg,
                                           levels,
                                           1,
                                           kGemvWg,
                                           pad_floats,
                                           spec_dtype(node->dtype),
                                           spec_dtype(node->dtype)};

                    const size_t gesz = dtype_size(node->dtype);
                    GemmPush gp{};
                    gp.a = address_of(a);
                    gp.b = address_of(b);
                    gp.d = address_of(*node);
                    gp.n_out = total;
                    gp.b1 = b1;
                    gp.m = m;
                    gp.n = n;
                    gp.k = k;
                    gp.op_a = to_gpu_operand(a.shape, gesz);
                    gp.op_b = to_gpu_operand(b.shape, gesz);

                    const auto& gpipe =
                        pipes.get("gemv", spv::gemv, spv::gemv_size, sizeof(GemmPush), gcfg);
                    if (debug_dispatch_enabled()) {
                        VKML_LOG_INFO("  gemv M={} N={} K={} ktiles={} passes={} levels={} "
                                      "groups={} wg={} lds={}B",
                                      m, n, k, ktiles, passes, levels, total, kGemvWg,
                                      gcfg.shared_memory_bytes);
                        if (gpipe.stats.available) {
                            const auto& st = gpipe.stats;
                            VKML_LOG_INFO("  gemv compiler: vgpr={} sgpr={} lds={}B "
                                          "waves={} scratch={}B instr={}",
                                          st.vgprs, st.sgprs, st.lds_bytes, st.max_waves,
                                          st.scratch_bytes, st.instructions);
                        }
                    }
                    rec.dispatch_groups(gpipe, &gp, sizeof(gp), total);
                    break;
                }

                // Tile edge. 16 gives a 256-invocation workgroup and 2 KiB of
                // shared memory -- comfortably inside this device's 64 KiB, and
                // an arithmetic intensity of TILE/4 = 4.0 FLOP/byte against the
                // naive kernel's 0.25.
                //
                // VKML_GEMM_NAIVE=1 selects the Stage 1 kernel instead, so the
                // two can be compared in one process rather than by rebuilding.
                // VKML_GEMM_KERNEL selects the implementation, so all three can
                // be compared in one process: naive | tiled | reg (default).
                static const int kernel_choice = [] {
                    const char* v = std::getenv("VKML_GEMM_KERNEL");
                    if (v == nullptr || v[0] == '\0') {
                        return 2;
                    }
                    if (std::string_view(v) == "naive") {
                        return 0;
                    }
                    if (std::string_view(v) == "tiled") {
                        return 1;
                    }
                    return 2;
                }();
                // Double buffering is opt-IN: it is a SEPARATE pipeline, so the
                // Stage 6 kernel stays byte-identical as the frozen control.
                // Stage 6.5 produced a spurious 1.4x by modifying the control
                // arm of its own A/B comparison; this avoids repeating that.
                static const bool want_db = [] {
                    const char* v = std::getenv("VKML_GEMM_DB");
                    return v != nullptr && v[0] != '\0' && v[0] != '0';
                }();

                const bool use_naive = kernel_choice == 0;
                const bool use_tiled = kernel_choice == 1;

                constexpr uint32_t kTile = 16;

                // Register-blocked geometry. 2x2 is the largest block for which
                // the per-accumulator pairwise carry stack fits in registers
                // without collapsing occupancy -- see the note in gemm_reg.comp.
                // Register-block geometry, selectable for the Stage 8 experiment.
                //
                // BM and BN are chosen so the workgroup stays at 256
                // invocations: (BM/RM) * (BN/RN) = 256. Holding the thread count
                // fixed is what makes the three variants comparable -- otherwise
                // a change in occupancy could not be attributed to register
                // pressure rather than to workgroup size.
                //
                // Each geometry compiles to a genuinely independent pipeline
                // (the values are specialisation constants), so the compiler
                // reports separate statistics for each and they benchmark
                // independently.
                struct BlockGeom {
                    uint32_t bm, bn, rm, rn;
                };

                static const BlockGeom forced_block = []() -> BlockGeom {
                    const char* v = std::getenv("VKML_GEMM_BLOCK");
                    const std::string_view sel = v != nullptr ? v : "";
                    if (sel == "4x2") {
                        return BlockGeom{64, 32, 4, 2};
                    }
                    if (sel == "4x4") {
                        return BlockGeom{64, 64, 4, 4};
                    }
                    // 2x4 exists only to TEST the register model (M3-R3), not
                    // as a performance candidate. It is the geometry that
                    // discriminates between two fits of C(RM,RN) which agree on
                    // every measured point and disagree by exactly 1 VGPR here:
                    // a structural model (av[RM] + bv[RN] + acc[RM*RN] + 11)
                    // predicts 25, a symmetric interpolation predicts 26.
                    // (32/2) * (64/4) = 256 invocations, matching every other
                    // geometry so the workgroup is not a second variable.
                    if (sel == "2x4") {
                        return BlockGeom{32, 64, 2, 4};
                    }
                    // 2x8 and 8x2 are the DISCRIMINATING pair for M3-R3's
                    // refined model C = 2*RM + 0.5*RN + RM*RN + 9. They have
                    // identical RM*RN and identical carry stacks, so any model
                    // symmetric in RM and RN must predict identical resources.
                    // The refined model predicts 97 VGPRs (clean) against 106
                    // (spilled) -- opposite sides of the cliff. Test geometry
                    // only; both are 256 invocations and ~20 KiB of LDS.
                    if (sel == "2x8") {
                        return BlockGeom{32, 128, 2, 8};
                    }
                    if (sel == "8x2") {
                        return BlockGeom{128, 32, 8, 2};
                    }
                    // 4x2 again, but at 512 invocations instead of 256. Exists
                    // to separate two readings of the scratch law that are
                    // numerically identical everywhere else: every other
                    // geometry has workgroup 256 AND subgroup 64, so
                    // "scratch = stack * workgroup" and "scratch = stack * 4
                    // bytes * 64 lanes" both give the same 256. Doubling the
                    // workgroup while the subgroup stays 64 forces them apart.
                    if (sel == "4x2w512") {
                        return BlockGeom{64, 64, 4, 2};
                    }
                    return BlockGeom{32, 32, 2, 2};  // Stage 6 default
                }();
                const bool block_forced = std::getenv("VKML_GEMM_BLOCK") != nullptr;

                // M3-01: shape-driven THREADBLOCK tile.
                //
                // Stage 8 established that the register block cannot grow: the
                // pairwise carry stack costs RM*RN*STACK_LEVELS registers, so
                // 4x4 spilled 24 KiB to scratch. BM and BN are the other axis --
                // they set how much output a WORKGROUP owns, not a thread, so
                // enlarging them raises arithmetic intensity
                //
                //     AI = 2*BM*BN*BK / ((BM*BK + BK*BN)*4)
                //
                // from 8.0 (32x32) to 16.0 (64x64) while leaving per-thread
                // register pressure exactly where it is. Stage 8 could not see
                // this because it deliberately pinned the workgroup at 256
                // invocations to isolate register pressure.
                //
                // Each dimension decides independently, which llama.cpp's ladder
                // cannot do: its `(m <= 32 || n <= 32)` test couples them, so a
                // tall-thin matrix gives up the large-M tile because N is small.
                // The 128 floor keeps at least 2x2 tiles in flight; a proper
                // compute-unit-aware rule needs shader_core_count, which arrives
                // with split-K (docs/M3_ROADMAP.md M3.4).
                //
                // MEASURED AND REJECTED (docs/M3-01-TILE-GEOMETRY.md).
                //
                // The arithmetic intensity gain is real -- 8.0 -> 16.0 FLOP/byte
                // -- and every compiler statistic came out exactly as predicted:
                // VGPR flat at 41, scratch 0, waves/SIMD still 16. It is still
                // SLOWER at every shape measured, monotonically in workgroup
                // size (256 -> 512 -> 1024 gives 1.00x -> 0.84x -> 0.77x at
                // 1024^3).
                //
                // Cause: waves/SIMD is not a sufficient occupancy metric. A
                // workgroup's waves all rendezvous at the same barrier, so they
                // are not independent work. Concurrent workgroups per CU -- the
                // number of INDEPENDENT barrier domains -- halves at each step
                // (8 -> 4 -> 2), and that loss exceeds the intensity gain.
                // Confirmed by double buffering, which removes one barrier per
                // k-tile and recovers 1.42x at 64x64 but only 1.05x at 32x32.
                //
                // Production libraries all grow the tile through PER-THREAD work
                // and hold the workgroup roughly constant: llama.cpp uses 128
                // invocations for both its 64x64 and 128x128 tiles
                // (ggml-vulkan.cpp:4030-4032); CLBlast's tuned gfx1010 entry
                // uses 256 threads for 64x64; rocBLAS navi21 uses 128 for
                // 128x64. That route is closed for vkML until the carry stack
                // shortens -- see docs/M3_ROADMAP.md.
                //
                // Kept selectable so the experiment reproduces, exactly as
                // VKML_GEMM_BLOCK preserves Stage 8. Default is the frozen
                // Stage 6 geometry.
                static const char* tile_force = std::getenv("VKML_GEMM_TILE");
                const std::string_view tile_sel = tile_force != nullptr ? tile_force : "";
                const uint32_t kBM = block_forced                           ? forced_block.bm
                                     : (tile_sel == "m" || tile_sel == "l") ? 64U
                                                                            : 32U;
                const uint32_t kBN = block_forced ? forced_block.bn : (tile_sel == "l") ? 64U : 32U;
                // BK=32, raised from 16 in Stage 5.75. Two reasons, both
                // measured: it halves the K-tile count (one fewer carry-stack
                // level, so fewer VGPRs), and it makes each block a 32-element
                // sequential sum -- exactly matching kPairwiseBlock in
                // src/backend/cpu/reduce.h, so the two backends fold K with the
                // same structure. It costs LDS, which is in surplus (8 KiB of
                // the device's 64 KiB).
                const uint32_t kBK = 32;
                const uint32_t kRM = forced_block.rm;
                const uint32_t kRN = forced_block.rn;
                const uint32_t kRegWg = (kBM / kRM) * (kBN / kRN);
                VKML_ASSERT(kRegWg <= impl_->caps.max_workgroup_invocations,
                            "gemm tile {}x{} at {}x{} needs {} invocations, device allows {}", kBM,
                            kBN, kRM, kRN, kRegWg, impl_->caps.max_workgroup_invocations);

                const uint32_t gemm_wg = use_naive ? wg : (use_tiled ? kTile * kTile : kRegWg);

                // -- split-K decision ---------------------------------------
                //
                // Only the register-blocked kernel participates: the naive and
                // tiled kernels are frozen comparison arms, and splitting them
                // would change what they are a control for.
                //
                // The chunk is a power-of-two number of K-tiles. That single
                // property is what makes the result bit-identical to the
                // unsplit kernel -- no fold inside a partition can cross a
                // boundary whose tile index has q low zero bits, so every
                // partial is exactly a subtree of the unsplit carry stack
                // (docs/SPLIT_K_DESIGN.md 2.3-2.4).
                const uint32_t total_ktiles = (k + kBK - 1) / kBK;
                const SplitKPlan sk = [&]() -> SplitKPlan {
                    if (use_naive || use_tiled || split_k_mode() == SplitKMode::Off) {
                        return {};
                    }
                    const uint32_t out_tiles =
                        ((m + kBM - 1) / kBM) * ((n + kBN - 1) / kBN) * b0 * b1;
                    // FORCED passes an explicit partition count and bypasses the
                    // profitability rule; AUTO passes 0 and lets it decide.
                    const uint32_t requested =
                        split_k_mode() == SplitKMode::Forced ? split_k_requested() : 0U;
                    return plan_split_k(out_tiles, total_ktiles, total,
                                        impl_->caps.shader_core_count, requested);
                }();
                const uint32_t split_chunk = sk.chunk;
                const uint32_t splits = sk.splits;
                const bool use_split_k = splits > 1;

                vk::KernelConfig cfg;
                cfg.workgroup_size = gemm_wg;
                // Operands and destination carry the same storage type here;
                // the split-K path below overrides the destination to f32.
                const uint32_t dt = spec_dtype(node->dtype);
                if (use_naive) {
                    cfg.shared_memory_bytes = 0;
                    cfg.spec_constants = {gemm_wg, dt, dt};
                } else if (use_tiled) {
                    cfg.shared_memory_bytes = 2 * kTile * kTile * sizeof(float);
                    cfg.spec_constants = {gemm_wg, kTile, dt, dt};
                } else {
                    // Stack depth from the actual K, bucketed so the pipeline
                    // cache does not gain an entry per distinct K.
                    //
                    // Under split-K this is the depth ONE PARTITION needs, so a
                    // partition covering `split_chunk` tiles gets a shorter
                    // stack and fewer accumulator registers. Sized from the
                    // chunk rather than from the last partition's possibly
                    // smaller tile count, so every partition shares one
                    // pipeline.
                    const uint32_t ktiles = use_split_k ? split_chunk : total_ktiles;
                    uint32_t levels = 1;
                    while ((1U << levels) < ktiles) {
                        ++levels;
                    }
                    ++levels;  // headroom for the final carry
                    for (const uint32_t bucket : {4U, 6U, 8U, 10U, 12U, 16U}) {
                        if (levels <= bucket) {
                            levels = bucket;
                            break;
                        }
                    }
                    // Vectorised loads are enabled only when the host can PROVE
                    // both operands are stride-1 in their innermost axis and the
                    // tile widths divide by 4. A transposed or broadcast operand
                    // fails this and silently takes the scalar path -- which is
                    // why the validation suite covers both.
                    //
                    // VKML_GEMM_NOVEC=1 forces the scalar path so the two can be
                    // compared in one process.
                    static const bool novec = [] {
                        const char* v = std::getenv("VKML_GEMM_NOVEC");
                        return v != nullptr && v[0] != '\0' && v[0] != '0';
                    }();
                    const bool can_vec4 =
                        !novec && (kBK % 4 == 0) && (kBN % 4 == 0) &&
                        a.shape.stride(3) == static_cast<int64_t>(dtype_size(a.dtype)) &&
                        b.shape.stride(3) == static_cast<int64_t>(dtype_size(b.dtype));

                    cfg.load_vector_width = can_vec4 ? 4 : 1;
                    // Double buffering doubles the tile storage.
                    cfg.shared_memory_bytes =
                        (want_db ? 2 : 1) * (kBM * kBK + kBK * kBN) * sizeof(float);
                    // Widened LDS reads require RN == 2 so the vec2 index is
                    // exact. VKML_GEMM_NOLDSVEC=1 forces the scalar path for
                    // comparison in one process.
                    static const bool no_ldsvec = [] {
                        const char* v = std::getenv("VKML_GEMM_NOLDSVEC");
                        return v != nullptr && v[0] != '\0' && v[0] != '0';
                    }();
                    const bool lds_vec = !no_ldsvec && kRN == 2;

                    // vec4 loading is f32-only: load4 reads through
                    // F32Vec4Buf, and a 16-bit vector would need its own
                    // buffer reference and alignment argument. Disabling it
                    // leaves the scalar fallback, which is already correct.
                    const bool vec4 = can_vec4 && node->dtype == DType::F32;

                    cfg.spec_constants = {
                        gemm_wg,           kBM, kBN, kBK, kRM, kRN, levels, vec4 ? 1U : 0U,
                        lds_vec ? 1U : 0U, dt,  dt};
                }

                // The operands' element size, which for a matmul equals the
                // output's -- vkml does not promote. Named for what it is
                // because the split-K workspace below deliberately does NOT use
                // it: partials are f32 whatever the operands are.
                const size_t esz = dtype_size(node->dtype);
                constexpr size_t kPartialEsz = sizeof(float);

                GemmPush push{};
                push.a = address_of(a);
                push.b = address_of(b);
                push.d = address_of(*node);
                push.n_out = total;
                push.b1 = b1;
                push.m = m;
                push.n = n;
                push.k = k;
                push.op_a = to_gpu_operand(a.shape, esz);
                push.op_b = to_gpu_operand(b.shape, esz);

                if (use_naive) {
                    if (debug_dispatch_enabled()) {
                        VKML_LOG_INFO("  gemm M={} N={} K={} batch={}x{} tile=none reg=1x1 "
                                      "shared=0B AI=0.25",
                                      m, n, k, b0, b1);
                        trace_dispatch(*node, "gemm_naive", cfg, sizeof(GemmPush),
                                       (total + gemm_wg - 1) / gemm_wg, impl_->caps.subgroup_size);
                    }
                    rec.dispatch(pipes.get("gemm_naive", spv::gemm_naive, spv::gemm_naive_size,
                                           sizeof(GemmPush), cfg),
                                 &push, sizeof(push), total);
                    break;
                }

                const uint32_t tm = use_tiled ? kTile : kBM;
                const uint32_t tn = use_tiled ? kTile : kBN;
                const uint32_t tk = use_tiled ? kTile : kBK;
                const uint32_t tiles_m = (m + tm - 1) / tm;
                const uint32_t tiles_n = (n + tn - 1) / tn;
                const uint32_t groups = tiles_m * tiles_n * b0 * b1;

                // AI = 2*BM*BN*BK / ((BM*BK + BK*BN) * 4 bytes)
                const double ai =
                    2.0 * tm * tn * tk / ((static_cast<double>(tm) * tk + tk * tn) * 4.0);

                if (debug_dispatch_enabled()) {
                    VKML_LOG_INFO(
                        "  gemm M={} N={} K={} batch={}x{} tile={}x{}x{} grid={}x{} ktiles={} "
                        "reg={}x{} threads={} shared={}B AI={:.2f} "
                        "owns_rows_per_thread={} owns_cols_per_thread={}",
                        m, n, k, b0, b1, tm, tn, tk, tiles_m, tiles_n, (k + tk - 1) / tk,
                        use_tiled ? 1 : kRM, use_tiled ? 1 : kRN, gemm_wg, cfg.shared_memory_bytes,
                        ai, use_tiled ? 1 : kRM, use_tiled ? 1 : kRN);
                    trace_dispatch(*node, use_tiled ? "gemm_tiled" : "gemm_reg", cfg,
                                   sizeof(GemmPush), groups, impl_->caps.subgroup_size);
                }

                const bool use_db = !use_tiled && !use_naive && want_db;
                const auto& gemm_pipe =
                    use_tiled ? pipes.get("gemm_tiled", spv::gemm_tiled, spv::gemm_tiled_size,
                                          sizeof(GemmPush), cfg)
                              : (use_db ? pipes.get("gemm_db", spv::gemm_db, spv::gemm_db_size,
                                                    sizeof(GemmPush), cfg)
                                        : pipes.get("gemm_reg", spv::gemm_reg, spv::gemm_reg_size,
                                                    sizeof(GemmPush), cfg));
                if (debug_dispatch_enabled() && gemm_pipe.stats.available) {
                    const auto& st = gemm_pipe.stats;
                    VKML_LOG_INFO("  compiler: vgpr={} sgpr={} spilled_vgpr={} spilled_sgpr={} "
                                  "scratch={}B lds={}B max_waves={}",
                                  st.vgprs, st.sgprs, st.spilled_vgprs, st.spilled_sgprs,
                                  st.scratch_bytes, st.lds_bytes, st.max_waves);
                    for (const auto& [nm, val] : st.raw) {
                        VKML_LOG_INFO("    stat {} = {}", nm, val);
                    }
                }

                if (use_tiled) {
                    rec.dispatch_groups(pipes.get("gemm_tiled", spv::gemm_tiled,
                                                  spv::gemm_tiled_size, sizeof(GemmPush), cfg),
                                        &push, sizeof(push), groups);
                } else if (use_db) {
                    rec.dispatch_groups(pipes.get("gemm_db", spv::gemm_db, spv::gemm_db_size,
                                                  sizeof(GemmPush), cfg),
                                        &push, sizeof(push), groups);
                } else if (!use_split_k) {
                    rec.dispatch_groups(pipes.get("gemm_reg", spv::gemm_reg, spv::gemm_reg_size,
                                                  sizeof(GemmPush), cfg),
                                        &push, sizeof(push), groups);
                } else {
                    // -- split-K -------------------------------------------
                    //
                    // The GEMM shader is UNCHANGED. A partition is expressed
                    // entirely by moving the operand base addresses forward
                    // along K, shortening `k`, and redirecting the output to a
                    // workspace slice. Nothing inside the kernel knows split-K
                    // exists, which is why its numerical behaviour cannot have
                    // drifted.
                    const uint64_t slice_elems = static_cast<uint64_t>(total);  // b0*b1*m*n
                    const uint64_t ws_bytes =
                        slice_elems * splits * static_cast<uint64_t>(kPartialEsz);
                    const uint64_t ws = impl_->splitk_workspace(ws_bytes);

                    // The partitions write f32 partial sums into the
                    // workspace, not the output, so their destination type
                    // differs from every other dispatch of this kernel.
                    vk::KernelConfig partial_cfg = cfg;
                    // OUT_DTYPE is the last constant this path builds. Asserted
                    // rather than assumed: appending another one would make
                    // back() overwrite the wrong slot, and the result would be
                    // a misread workspace rather than a failure.
                    VKML_ASSERT(
                        partial_cfg.spec_constants.size() == 11,
                        "gemm_reg spec constants changed shape; OUT_DTYPE is no longer last");
                    partial_cfg.spec_constants.back() = spec_dtype(DType::F32);

                    const auto& part_pipe = pipes.get("gemm_reg", spv::gemm_reg, spv::gemm_reg_size,
                                                      sizeof(GemmPush), partial_cfg);
                    for (uint32_t s = 0; s < splits; ++s) {
                        const uint32_t k_begin = s * split_chunk * kBK;
                        const uint32_t k_len = std::min(split_chunk * kBK, k - k_begin);

                        GemmPush sp = push;
                        // Strides are in ELEMENTS; addresses are in bytes.
                        sp.a = push.a + static_cast<uint64_t>(k_begin) * push.op_a.nb[3] * esz;
                        sp.b = push.b + static_cast<uint64_t>(k_begin) * push.op_b.nb[2] * esz;
                        sp.d = ws + static_cast<uint64_t>(s) * slice_elems * kPartialEsz;
                        sp.k = k_len;
                        // No barrier between partitions: they write disjoint
                        // workspace slices, so the driver is free to overlap
                        // them. That overlap is the entire point.
                        rec.dispatch_groups(part_pipe, &sp, sizeof(sp), groups);
                    }

                    // Partials must be complete before the fold reads them.
                    rec.barrier();

                    uint32_t split_levels = 1;
                    while ((1U << split_levels) < splits) {
                        ++split_levels;
                    }
                    ++split_levels;

                    vk::KernelConfig rcfg;
                    rcfg.workgroup_size = wg;
                    rcfg.spec_constants = {wg, split_levels, spec_dtype(node->dtype)};

                    SplitKReducePush rp{};
                    rp.src = ws;
                    rp.dst = address_of(*node);
                    rp.ne = total;
                    rp.splits = splits;

                    const auto& red_pipe =
                        pipes.get("gemm_split_k_reduce", spv::gemm_split_k_reduce,
                                  spv::gemm_split_k_reduce_size, sizeof(SplitKReducePush), rcfg);

                    if (debug_dispatch_enabled()) {
                        VKML_LOG_INFO("  split-k splits={} chunk={} ktiles/part={} "
                                      "stack_levels={} workspace={}KiB reduce_levels={}",
                                      splits, split_chunk, split_chunk, cfg.spec_constants[6],
                                      ws_bytes / 1024, split_levels);
                        if (red_pipe.stats.available) {
                            const auto& st = red_pipe.stats;
                            VKML_LOG_INFO("  reduce compiler: vgpr={} sgpr={} spilled_vgpr={} "
                                          "scratch={}B lds={}B max_waves={}",
                                          st.vgprs, st.sgprs, st.spilled_vgprs, st.scratch_bytes,
                                          st.lds_bytes, st.max_waves);
                        }
                    }

                    rec.dispatch(red_pipe, &rp, sizeof(rp), total);
                }
                break;
            }

            default:
                throw NotImplementedError(
                    std::format("vulkan backend: no kernel for op '{}'", op_name(node->op)));
        }

        // Between every pair of dispatches. Conservative: the executor gives no
        // aliasing information yet, so any node may read what the previous one
        // wrote. See the strategy note in vk_command.h.
        rec.barrier();
        traced.push_back(node);
    }

    const uint64_t ticket = rec.submit();
    // M1 is synchronous: wait here rather than returning a handle. Overlapping
    // host and device work is a performance change and needs the execution
    // graph to know what is safe to defer.
    rec.wait(ticket);

    if (debug_dispatch_enabled()) {
        const auto& profile = rec.profile();
        for (size_t i = 0; i < profile.size() && i < traced.size(); ++i) {
            VKML_LOG_INFO("  timing op={} gpu={:.4f}ms", op_name(traced[i]->op), profile[i].gpu_ms);
        }
    }

    if (const int64_t limit = debug_dump_limit(); limit > 0) {
        for (Node* node : traced) {
            if (node->shape.numel() > limit || node->dtype != DType::F32) {
                continue;
            }
            std::vector<float> host(static_cast<size_t>(node->shape.numel()));
            copy_to_host(host.data(), *node->storage, node->storage_offset,
                         host.size() * sizeof(float));
            std::string body;
            for (size_t i = 0; i < host.size(); ++i) {
                body += std::format("{}{:g}", i ? ", " : "", host[i]);
            }
            VKML_LOG_INFO("  dump op={} shape={} [{}]", op_name(node->op), node->shape.str(), body);
        }
    }
}

void VulkanBackend::copy_from_host(Storage& dst, int64_t dst_offset, const void* src,
                                   size_t nbytes) {
    if (nbytes == 0) {
        return;
    }
    const std::lock_guard<std::mutex> lock(impl_->map_mutex);
    const auto addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dst.data()));
    const auto it = impl_->live.find(addr);
    VKML_ASSERT(it != impl_->live.end(), "unknown device address in copy_from_host");
    impl_->staging.upload(src, it->second, static_cast<uint64_t>(dst_offset), nbytes);
}

void VulkanBackend::copy_to_host(void* dst, const Storage& src, int64_t src_offset, size_t nbytes) {
    if (nbytes == 0) {
        return;
    }
    const std::lock_guard<std::mutex> lock(impl_->map_mutex);
    const auto addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src.data()));
    const auto it = impl_->live.find(addr);
    VKML_ASSERT(it != impl_->live.end(), "unknown device address in copy_to_host");
    impl_->staging.download(dst, it->second, static_cast<uint64_t>(src_offset), nbytes);
}

std::vector<PipelineStats> VulkanBackend::pipeline_stats() const {
    std::vector<PipelineStats> out;
    for (const auto& [key, st] : impl_->pipelines.all_stats()) {
        PipelineStats p;
        p.name = key;
        p.available = st.available;
        p.vgprs = st.vgprs;
        p.sgprs = st.sgprs;
        p.spilled_vgprs = st.spilled_vgprs;
        p.spilled_sgprs = st.spilled_sgprs;
        p.scratch_bytes = st.scratch_bytes;
        p.lds_bytes = st.lds_bytes;
        p.waves_per_simd = st.max_waves;
        p.instructions = st.instructions;
        p.code_bytes = st.code_bytes;
        out.push_back(std::move(p));
    }
    return out;
}

void VulkanBackend::set_profiling(bool enabled) { impl_->recorder.set_profiling(enabled); }

std::vector<std::pair<std::string, double>> VulkanBackend::last_profile() const {
    std::vector<std::pair<std::string, double>> out;
    for (const vk::ProfileEntry& e : impl_->recorder.profile()) {
        out.emplace_back(e.label, e.gpu_ms);
    }
    return out;
}

void VulkanBackend::set_subgroup_override(uint32_t size) { impl_->subgroup_override = size; }

void VulkanBackend::synchronize() { impl_->recorder.wait_idle(); }

void VulkanBackend::trim() { impl_->allocator.trim(); }

VulkanStats VulkanBackend::stats() const {
    const vk::AllocatorStats a = impl_->allocator.stats();
    VulkanStats s;
    s.reserved_bytes = a.reserved_bytes;
    s.in_use_bytes = a.in_use_bytes;
    s.peak_in_use_bytes = a.peak_in_use_bytes;
    s.block_count = a.block_count;
    s.live_allocations = a.live_allocations;
    s.total_allocations = a.total_allocations;
    s.device_allocations = a.device_allocations;
    s.fragmentation = a.fragmentation();
    s.submissions = impl_->recorder.submitted_count();
    s.dispatches = impl_->recorder.dispatch_count();
    s.pipelines = impl_->pipelines.pipeline_count();
    return s;
}

bool vulkan_available() { return vk::enumerate_device_count() > 0; }

int vulkan_device_count() { return vk::enumerate_device_count(); }

std::vector<std::string> vulkan_device_names() { return vk::enumerate_device_names(); }

namespace {

std::mutex& backend_mutex() {
    static std::mutex m;
    return m;
}

/// Deliberately leaked, and the leak is load-bearing.
///
/// If these were owned by a static unique_ptr, they would be destroyed during
/// static destruction at process exit -- but the Vulkan loader and validation
/// layer tear down their own statics at the same stage, and the ordering
/// between translation units is unspecified. Destroying a VkDevice after the
/// layer has gone produces "The VkDevice dispatch handle was not found and
/// Validation will crash", which is exactly what happened before this change.
///
/// The OS reclaims the memory and the driver's resources at exit regardless, so
/// leaking here costs nothing real. Anything that genuinely needs a clean
/// teardown -- a test asserting the allocator has no live blocks, for instance
/// -- calls vulkan_shutdown() explicitly while the loader is still alive.
std::unordered_map<int, VulkanBackend*>& backend_registry() {
    static auto* registry = new std::unordered_map<int, VulkanBackend*>();
    return *registry;
}

}  // namespace

Backend& vulkan_backend(int index) {
    const std::lock_guard<std::mutex> lock(backend_mutex());
    auto& registry = backend_registry();

    if (const auto it = registry.find(index); it != registry.end()) {
        return *it->second;
    }

    const bool validation = env_flag("VKML_VULKAN_VALIDATION", true);
    auto* backend = new VulkanBackend(index, validation);
    registry.emplace(index, backend);
    register_backend(*backend);
    return *backend;
}

void vulkan_shutdown() {
    const std::lock_guard<std::mutex> lock(backend_mutex());
    auto& registry = backend_registry();
    for (auto& [index, backend] : registry) {
        backend->synchronize();
        delete backend;
    }
    registry.clear();
}

}  // namespace vkml
