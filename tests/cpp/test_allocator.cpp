#include "doctest.h"
#include "test_support.h"

#include "vkml/core/allocator.h"
#include "vkml/util/error.h"

#include <cstring>
#include <vector>

TEST_CASE("cpu allocator reports its identity") {
    auto& a = vkml::cpu_allocator();
    CHECK(a.name() == "cpu");
    CHECK(a.device() == vkml::Device::cpu());
}

TEST_CASE("cpu allocator returns aligned, writable, correctly sized memory") {
    auto& a = vkml::cpu_allocator();
    auto s = a.allocate(1000);
    REQUIRE(s != nullptr);
    CHECK(s->nbytes() == 1000);
    CHECK(s->device() == vkml::Device::cpu());
    CHECK(reinterpret_cast<uintptr_t>(s->data()) % vkml::kCpuAlignment == 0);

    std::memset(s->data(), 0x5A, 1000);
    CHECK(static_cast<const uint8_t*>(s->data())[999] == 0x5A);
}

TEST_CASE("allocator tracks live bytes and releases them") {
    auto& a = vkml::cpu_allocator();
    const size_t before = a.live_bytes();
    {
        auto s = a.allocate(4096);
        // Accounting is in rounded (actually reserved) bytes, not requested
        // bytes, so it reflects real footprint.
        CHECK(a.live_bytes() >= before + 4096);
    }
    CHECK(a.live_bytes() == before);
}

TEST_CASE("zero-size allocation is legal") {
    auto s = vkml::cpu_allocator().allocate(0);
    REQUIRE(s != nullptr);
    CHECK(s->nbytes() == 0);
    CHECK(s->data() == nullptr);
}

TEST_CASE("make_cpu_storage delegates to the cpu allocator") {
    auto& a = vkml::cpu_allocator();
    const size_t before = a.live_bytes();
    {
        auto s = vkml::make_cpu_storage(2048);
        CHECK(a.live_bytes() > before);
    }
    CHECK(a.live_bytes() == before);
}

TEST_CASE("allocator churn does not leak") {
    auto& a = vkml::cpu_allocator();
    const size_t before = a.live_bytes();
    for (int i = 1; i <= 500; ++i) {
        auto s = a.allocate(static_cast<size_t>(i) * 64);
        CHECK(s->nbytes() == static_cast<size_t>(i) * 64);
    }
    CHECK(a.live_bytes() == before);
}

TEST_CASE("Storage still accepts foreign memory with a custom deleter") {
    // This is the path a zero-copy NumPy / DLPack import takes: the memory is
    // someone else's, so an Allocator is the wrong owner and the deleter
    // closure is the right one. Keeping it public is deliberate (ADR 0002).
    std::vector<float> foreign(16, 1.5F);
    bool freed = false;

    {
        auto s = std::make_shared<vkml::Storage>(
            foreign.data(), foreign.size() * sizeof(float), vkml::Device::cpu(),
            [&freed](void*, size_t) { freed = true; });
        CHECK(s->data() == foreign.data());
        CHECK_FALSE(freed);
    }

    CHECK(freed);
    // The deleter did not actually free anything, so the vector is still valid.
    CHECK(foreign[0] == doctest::Approx(1.5F));
}
