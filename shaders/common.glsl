// Shared preamble for every vkml compute shader.
//
// DESCRIPTOR-LESS BINDING
// -----------------------
// Buffers are passed as 64-bit device addresses in push constants rather than
// through descriptor sets. The target GPU supports bufferDeviceAddress
// (measured, docs/ARCHITECTURE.md 1.1), and using it deletes descriptor pools,
// set layouts, per-dispatch vkUpdateDescriptorSets and the pool-growth logic
// that ggml-vulkan needs -- roughly 200-500 lines of the most error-prone code
// in a Vulkan backend, plus real per-dispatch CPU cost.
//
// scalar_block_layout means these structs lay out identically in GLSL and C++,
// so the push-constant block can be declared once per op and mirrored by a
// plain C++ struct with no std140 padding rules to get wrong.

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8  : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

// Rank limit, mirroring vkml::kMaxDims. Four keeps a 3-operand shape/stride
// block at 96 bytes, comfortably inside this GPU's 256-byte push constant
// budget; rank 8 would need 192 and force the metadata into a uniform buffer.
#define VKML_MAX_DIMS 4

layout(buffer_reference, scalar, buffer_reference_align = 4) buffer F32Buf { float v[]; };
// 4-wide access for tile loading. buffer_reference_align = 4 rather than 16:
// scalar_block_layout permits a vec4 at 4-byte alignment, so a tile row that
// does not start on a 16-byte boundary is still legal. The driver splits the
// access when it cannot prove alignment, which is the correct fallback -- but
// it also means a vec4 load is not automatically a single instruction.
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer F32Vec4Buf { vec4 v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer I32Buf { int   v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 8) buffer I64Buf { int64_t v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 2) buffer U16Buf { uint16_t v[]; };
// Native f16 access. The buffer is typed float16_t so a LOAD is a hardware
// widening, which is exact and needs no help. A STORE goes through
// f32_to_f16_bits below rather than the hardware narrowing, because that one
// has an implementation-defined rounding mode -- see the comment there.
layout(buffer_reference, scalar, buffer_reference_align = 2) buffer F16Buf { float16_t v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 1) buffer U8Buf  { uint8_t v[]; };

// ---------------------------------------------------------------------------
// Floating storage access
//
// Mirrors vkml::DType. Only the two floating types appear here: the integer
// types are storage and indices, not arithmetic (see the dtype contract in
// include/vkml/core/dtype.h), and their kernels read them through their own
// typed buffer references.
#define T_F32 0
#define T_F16 1

// F16 IS STORAGE, NOT ARITHMETIC. Both helpers convert at the memory boundary
// and everything between them is `float`, which is the fp32-accumulation half
// of docs/ARCHITECTURE.md 7.3 -- the same contract the CPU backend's `widen`
// implements, deliberately written to look the same.
//
// `dtype` is a specialisation constant at every call site, so the branch is
// folded away at pipeline creation and an f32 kernel is byte-identical to what
// it compiled to before this existed. It is a parameter rather than a shared
// constant because the shaders reached different constant_id numbers before
// this was added, and renumbering them would invalidate cached pipelines.
float load_f(uint64_t buf, uint idx, uint dtype) {
    if (dtype == T_F16) {
        return float(F16Buf(buf).v[idx]);
    }
    return F32Buf(buf).v[idx];
}

/// f32 -> f16 with round-to-nearest-even, done in the integer domain.
///
/// WHY NOT `float16_t(value)`. SPIR-V leaves OpFConvert's rounding mode
/// IMPLEMENTATION-DEFINED. RADV rounds to nearest even; AMD's Windows compiler
/// rounds toward zero, so the same program produced different f16 results on
/// the two, and the cross-backend oracle -- which compares bit for bit against
/// the CPU's fp32_to_fp16, an RTE routine -- failed on Windows (issue #3).
///
/// Determinism across drivers is a project invariant (docs/ARCHITECTURE.md), so
/// the fix cannot be a tolerance. It could have been the RoundingModeRTE
/// execution mode from VK_KHR_shader_float_controls, but that needs a device
/// capability, a fallback for devices without it, and trust that the driver
/// honours it -- three things that are not verifiable from here. Nothing below
/// is implementation-defined: integer shifts and comparisons only, no
/// floating-point operation whose rounding a driver could choose.
///
/// This is bit-for-bit the same function as vkml::fp32_to_fp16 in
/// src/core/dtype.cpp, which reaches the same result by a different route. The
/// two are checked against each other over the whole f32 exponent range.
uint f32_to_f16_bits(float value) {
    const uint w = floatBitsToUint(value);
    const uint sign = (w >> 16) & 0x8000u;
    const uint mag = w & 0x7fffffffu;

    // NaN must stay NaN. Truncating a payload can clear every mantissa bit and
    // silently turn it into an infinity.
    if (mag >= 0x7f800000u) {
        return sign | (mag > 0x7f800000u ? 0x7e00u : 0x7c00u);
    }

    // Normal f16, i.e. |value| >= 2^-14. Subtracting the bias difference
    // (127 - 15) << 23 rebiases the exponent and leaves the mantissa in place,
    // so one shift produces the exponent and mantissa fields together.
    if (mag >= 0x38800000u) {
        uint h = (mag - 0x38000000u) >> 13;
        const uint rem = mag & 0x1fffu;  // the 13 bits being discarded
        if (rem > 0x1000u || (rem == 0x1000u && (h & 1u) != 0u)) {
            ++h;  // carries mantissa -> exponent, and 65520 -> infinity, both wanted
        }
        return sign | h;
    }

    // Below 2^-32 nothing can round up even to the smallest subnormal, and the
    // shift below would exceed 31 and be undefined. Zero (either sign) lands
    // here too.
    if (mag < 0x2f800000u) {
        return sign;
    }

    // Subnormal f16: shift the significand, implicit bit restored, down to the
    // 2^-24 grid. `shift` is 14..31 over the range that reaches this point.
    const uint sig = (mag & 0x007fffffu) | 0x00800000u;
    const uint shift = 126u - (mag >> 23);
    const uint h = sig >> shift;
    const uint rem = sig & ((1u << shift) - 1u);
    const uint tie = 1u << (shift - 1u);
    if (rem > tie || (rem == tie && (h & 1u) != 0u)) {
        return sign | (h + 1u);  // may carry to the smallest normal, correctly
    }
    return sign | h;
}

void store_f(uint64_t buf, uint idx, float value, uint dtype) {
    if (dtype == T_F16) {
        F16Buf(buf).v[idx] = uint16BitsToFloat16(uint16_t(f32_to_f16_bits(value)));
        return;
    }
    F32Buf(buf).v[idx] = value;
}

/// Workgroup width, supplied by the host as specialisation constant 0.
///
/// Declared here rather than in each shader because global_index() below must
/// read gl_WorkGroupSize, and GLSL forbids that before the size is fixed -- with
/// the declaration in the including file it comes too late. Every shader used
/// the identical line, so this is also one definition instead of twenty-four.
/// The width itself is chosen by KernelConfig::workgroup_size on the host.
layout(local_size_x_id = 0) in;

/// This invocation's flat element index, across a dispatch grid of any shape.
///
/// WHY NOT gl_GlobalInvocationID.x. maxComputeWorkGroupCount[x] is guaranteed to
/// be only 65535, so a one-dimensional dispatch cannot cover more than
/// 65535 * workgroup_size elements -- 64 MiB of f32 at the usual width. The host
/// folds anything larger into y (Recorder::dispatch), and the flat index then has
/// to be reconstructed from both dimensions.
///
/// Identical to gl_GlobalInvocationID.x when y holds a single group, because
/// gl_GlobalInvocationID.y is then 0. The common case is unchanged, which is what
/// makes this safe to use in every kernel rather than only the large ones.
///
/// Local size in y is always 1, so gl_GlobalInvocationID.y is the y group index
/// and the x extent is gl_NumWorkGroups.x * gl_WorkGroupSize.x.
uint global_index() {
    return gl_GlobalInvocationID.y * (gl_NumWorkGroups.x * gl_WorkGroupSize.x)
         + gl_GlobalInvocationID.x;
}

/// This workgroup's flat index, across a dispatch grid of any shape.
///
/// The group-granularity counterpart of global_index(), for kernels where one
/// workgroup owns one output row rather than one invocation owning one element:
/// reductions, softmax, GEMV and the GEMM family.
///
/// Same ceiling and same remedy. A reduction dispatches one group per output
/// row, so a tensor with more than 65535 rows exceeds a one-dimensional grid --
/// 32 images of 3x64x64 pooled to 32x32 is already 98,304 rows. Identical to
/// gl_WorkGroupID.x when y holds a single group.
uint global_group_index() {
    return gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
}

/// Layout of one strided operand. Mirrors vkml::Shape.
///
/// Strides are in ELEMENTS here, not bytes as on the host side. The conversion
/// happens once, when the push constants are filled in, because a shader
/// indexes a typed buffer reference and would otherwise divide by the element
/// size on every access.
struct Operand {
    uvec4 ne;  // extents, unused axes = 1
    uvec4 nb;  // strides in elements
};

/// Flat index -> element offset, with extents and strides supplied separately.
///
/// Mirrors vkml::cpu::linear_to_offset exactly, so that a CPU/GPU disagreement
/// is a real bug rather than a difference in traversal order. Stride 0 handles
/// broadcasting for free.
///
/// Split from Operand because some kernels store the extents ONCE for several
/// operands that provably share them, and pass only the strides per operand --
/// which is what keeps their push-constant block inside the 128 bytes Vulkan
/// guarantees (docs/adr/0009 sec2). Those kernels call this directly.
uint offset_from(uint linear, uvec4 ne, uvec4 nb) {
    uint off = 0;
    [[unroll]] for (int i = VKML_MAX_DIMS - 1; i >= 0; --i) {
        uint extent = ne[i];
        uint idx = linear % extent;
        linear /= extent;
        off += idx * nb[i];
    }
    return off;
}

/// Flat index -> element offset for a possibly-strided operand.
uint operand_offset(uint linear, Operand op) {
    return offset_from(linear, op.ne, op.nb);
}

/// True when an operand can be walked with a flat index.
bool operand_is_contiguous(Operand op, uint total) {
    uint expected = 1;
    [[unroll]] for (int i = VKML_MAX_DIMS - 1; i >= 0; --i) {
        if (op.ne[i] != 1) {
            if (op.nb[i] != expected) { return false; }
            expected *= op.ne[i];
        }
    }
    return true;
}
