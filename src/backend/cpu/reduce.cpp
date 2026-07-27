#include "reduce.h"

#include "vkml/util/assert.h"

namespace vkml::cpu {

uint32_t all_axes_mask(int ndim) noexcept {
    return ndim <= 0 ? 0U : (1U << static_cast<uint32_t>(ndim)) - 1U;
}

ReducePlan make_reduce_plan(const Shape& in, uint32_t axes_mask) {
    std::vector<int64_t> kept_dims;
    std::vector<int64_t> kept_strides;
    std::vector<int64_t> red_dims;
    std::vector<int64_t> red_strides;

    for (int i = 0; i < in.ndim(); ++i) {
        const bool reduced = (axes_mask & (1U << static_cast<uint32_t>(i))) != 0;
        if (reduced) {
            red_dims.push_back(in.dim(i));
            red_strides.push_back(in.stride(i));
        } else {
            kept_dims.push_back(in.dim(i));
            kept_strides.push_back(in.stride(i));
        }
    }

    return ReducePlan{Shape::strided(kept_dims, kept_strides, in.itemsize()),
                      Shape::strided(red_dims, red_strides, in.itemsize())};
}

std::vector<int64_t> reduced_dims(const Shape& in, uint32_t axes_mask, bool keepdim) {
    std::vector<int64_t> out;
    for (int i = 0; i < in.ndim(); ++i) {
        const bool reduced = (axes_mask & (1U << static_cast<uint32_t>(i))) != 0;
        if (!reduced) {
            out.push_back(in.dim(i));
        } else if (keepdim) {
            out.push_back(1);
        }
    }
    return out;
}

}  // namespace vkml::cpu
