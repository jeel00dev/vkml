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
// Native f16 access. Using float16_t rather than manipulating bits means the
// hardware does the IEEE-754 binary16 conversion, including subnormals -- which
// is exactly what the CPU implementation goes to some length to preserve.
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

void store_f(uint64_t buf, uint idx, float value, uint dtype) {
    if (dtype == T_F16) {
        F16Buf(buf).v[idx] = float16_t(value);
        return;
    }
    F32Buf(buf).v[idx] = value;
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

/// Flat index -> element offset for a possibly-strided operand.
///
/// Mirrors vkml::cpu::linear_to_offset exactly, so that a CPU/GPU disagreement
/// is a real bug rather than a difference in traversal order. Stride 0 handles
/// broadcasting for free.
uint operand_offset(uint linear, Operand op) {
    uint off = 0;
    [[unroll]] for (int i = VKML_MAX_DIMS - 1; i >= 0; --i) {
        uint extent = op.ne[i];
        uint idx = linear % extent;
        linear /= extent;
        off += idx * op.nb[i];
    }
    return off;
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
