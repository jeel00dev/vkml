#include "doctest.h"
#include "test_support.h"

#include "vkml/core/device.h"
#include "vkml/core/dtype.h"
#include "vkml/core/storage.h"
#include "vkml/util/error.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

TEST_CASE("dtype sizes and predicates") {
    CHECK(vkml::dtype_size(vkml::DType::F32) == 4);
    CHECK(vkml::dtype_size(vkml::DType::F16) == 2);
    CHECK(vkml::dtype_size(vkml::DType::I64) == 8);
    CHECK(vkml::dtype_size(vkml::DType::Bool) == 1);

    CHECK(vkml::is_floating(vkml::DType::F16));
    CHECK_FALSE(vkml::is_floating(vkml::DType::I32));
    CHECK(vkml::is_integral(vkml::DType::I64));

    // Only floating tensors carry gradients, matching PyTorch.
    CHECK(vkml::is_differentiable(vkml::DType::F32));
    CHECK_FALSE(vkml::is_differentiable(vkml::DType::I64));
    CHECK_FALSE(vkml::is_differentiable(vkml::DType::Bool));
}

TEST_CASE("fp16 round-trips exactly representable values") {
    for (const float v : {0.0F, 1.0F, -1.0F, 0.5F, -0.5F, 2.0F, 1024.0F, -65504.0F, 65504.0F}) {
        const float back = vkml::fp16_to_fp32(vkml::fp32_to_fp16(v));
        CHECK(back == doctest::Approx(v));
    }
}

TEST_CASE("fp16 preserves subnormals rather than flushing to zero") {
    // The naive shift-and-mask conversion silently returns 0 here. That bug
    // would only surface as a tolerance failure on very small activations, so
    // it is worth an explicit test.
    const float smallest_subnormal = 5.9604645e-8F;  // 2^-24
    const uint16_t h = vkml::fp32_to_fp16(smallest_subnormal);
    CHECK(h != 0);
    CHECK(vkml::fp16_to_fp32(h) > 0.0F);

    const float mid_subnormal = 3.0e-8F;
    CHECK(vkml::fp16_to_fp32(vkml::fp32_to_fp16(mid_subnormal)) > 0.0F);
}

TEST_CASE("fp16 handles infinities and NaN") {
    const float inf = std::numeric_limits<float>::infinity();
    CHECK(std::isinf(vkml::fp16_to_fp32(vkml::fp32_to_fp16(inf))));
    CHECK(vkml::fp16_to_fp32(vkml::fp32_to_fp16(-inf)) < 0);
    CHECK(std::isinf(vkml::fp16_to_fp32(vkml::fp32_to_fp16(-inf))));

    const float nan = std::numeric_limits<float>::quiet_NaN();
    CHECK(std::isnan(vkml::fp16_to_fp32(vkml::fp32_to_fp16(nan))));

    // Overflow beyond fp16 range saturates to infinity, not to garbage.
    CHECK(std::isinf(vkml::fp16_to_fp32(vkml::fp32_to_fp16(1.0e30F))));
}

TEST_CASE("Half is storage-only and does not implicitly convert") {
    const vkml::Half h{1.5F};
    CHECK(h.to_float() == doctest::Approx(1.5F));
    CHECK(sizeof(vkml::Half) == 2);

    // Half must NOT be implicitly constructible from float: accumulating in 16
    // bits by accident is exactly the failure mode the explicit ctor prevents.
    CHECK_FALSE(std::is_convertible_v<float, vkml::Half>);
}

TEST_CASE("device formatting and parsing") {
    CHECK(vkml::Device::cpu().str() == "cpu");
    CHECK(vkml::Device::vulkan(0).str() == "vulkan:0");
    CHECK(vkml::Device::vulkan(2).str() == "vulkan:2");

    CHECK(vkml::Device::parse("cpu") == vkml::Device::cpu());
    CHECK(vkml::Device::parse("vulkan") == vkml::Device::vulkan(0));
    CHECK(vkml::Device::parse("vulkan:1") == vkml::Device::vulkan(1));

    SUBCASE("malformed device strings throw") {
        CHECK_THROWS_AS(discard(vkml::Device::parse("")), vkml::DeviceError);
        CHECK_THROWS_AS(discard(vkml::Device::parse("cuda")), vkml::DeviceError);
        CHECK_THROWS_AS(discard(vkml::Device::parse("vulkan:")), vkml::DeviceError);
        CHECK_THROWS_AS(discard(vkml::Device::parse("vulkan:x")), vkml::DeviceError);
        CHECK_THROWS_AS(discard(vkml::Device::parse("vulkan:1x")), vkml::DeviceError);
        CHECK_THROWS_AS(discard(vkml::Device::parse("cpu:1")), vkml::DeviceError);
    }
}

TEST_CASE("cpu storage allocates aligned memory and frees it") {
    const size_t before_blocks = vkml::storage_stats::live_blocks();
    const size_t before_bytes = vkml::storage_stats::live_bytes();

    {
        auto s = vkml::make_cpu_storage(1000);
        REQUIRE(s != nullptr);
        CHECK(s->nbytes() == 1000);
        CHECK(s->device() == vkml::Device::cpu());
        CHECK(reinterpret_cast<uintptr_t>(s->data()) % vkml::kCpuAlignment == 0);

        // The block must be fully writable even though the request was rounded
        // up internally for aligned_alloc.
        std::memset(s->data(), 0xAB, 1000);
        CHECK(static_cast<const uint8_t*>(s->data())[999] == 0xAB);

        CHECK(vkml::storage_stats::live_blocks() == before_blocks + 1);
        CHECK(vkml::storage_stats::live_bytes() == before_bytes + 1000);
    }

    CHECK(vkml::storage_stats::live_blocks() == before_blocks);
    CHECK(vkml::storage_stats::live_bytes() == before_bytes);
}

TEST_CASE("zero-size storage is legal") {
    auto s = vkml::make_cpu_storage(0);
    REQUIRE(s != nullptr);
    CHECK(s->nbytes() == 0);
    CHECK(s->data() == nullptr);
}

TEST_CASE("storage allocation is leak-free under churn") {
    const size_t before = vkml::storage_stats::live_bytes();
    for (int i = 0; i < 1000; ++i) {
        auto s = vkml::make_cpu_storage(static_cast<size_t>(i) + 1);
        CHECK(s->nbytes() == static_cast<size_t>(i) + 1);
    }
    CHECK(vkml::storage_stats::live_bytes() == before);
}
