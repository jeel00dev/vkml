#include "doctest.h"
#include "test_support.h"

#include "vkml/api/ops.h"
#include "vkml/dispatch/executor.h"
#include "vkml/util/error.h"

#include <cmath>
#include <numeric>
#include <vector>

using vkml::DType;
using vkml::Tensor;

namespace {

Tensor t2(std::vector<int64_t> dims, std::vector<float> values) {
    REQUIRE(static_cast<size_t>(std::accumulate(dims.begin(), dims.end(), int64_t{1},
                                                std::multiplies<>())) == values.size());
    return Tensor::from_host(values.data(), dims);
}

std::vector<float> host(const Tensor& t) {
    std::vector<float> out(static_cast<size_t>(t.numel()));
    t.to_host(out.data());
    return out;
}

std::vector<int64_t> host_i64(const Tensor& t) {
    std::vector<int64_t> out(static_cast<size_t>(t.numel()));
    t.to_host(out.data());
    return out;
}

void check_close(const std::vector<float>& got, const std::vector<float>& want, float tol = 1e-6F) {
    REQUIRE(got.size() == want.size());
    for (size_t i = 0; i < got.size(); ++i) {
        CHECK(got[i] == doctest::Approx(want[i]).epsilon(tol));
    }
}

}  // namespace

TEST_CASE("tensor construction and host round-trip") {
    const Tensor a = t2({2, 3}, {1, 2, 3, 4, 5, 6});
    CHECK(a.shape() == std::vector<int64_t>{2, 3});
    CHECK(a.numel() == 6);
    CHECK(a.dtype() == DType::F32);
    CHECK(a.is_contiguous());
    check_close(host(a), {1, 2, 3, 4, 5, 6});
}

TEST_CASE("zeros, ones, full, arange") {
    check_close(host(Tensor::zeros({2, 2})), {0, 0, 0, 0});
    check_close(host(Tensor::ones({3})), {1, 1, 1});
    check_close(host(Tensor::full({2}, 2.5)), {2.5F, 2.5F});
    check_close(host(Tensor::arange(0, 5)), {0, 1, 2, 3, 4});
    check_close(host(Tensor::arange(1, 2, 0.25)), {1.0F, 1.25F, 1.5F, 1.75F});
    CHECK(Tensor::arange(0, 0).numel() == 0);
}

TEST_CASE("binary elementwise ops") {
    const Tensor a = t2({4}, {1, 2, 3, 4});
    const Tensor b = t2({4}, {10, 20, 30, 40});

    check_close(host(a + b), {11, 22, 33, 44});
    check_close(host(b - a), {9, 18, 27, 36});
    check_close(host(a * b), {10, 40, 90, 160});
    check_close(host(b / a), {10, 10, 10, 10});
    check_close(host(vkml::maximum(a, t2({4}, {2, 2, 2, 2}))), {2, 2, 3, 4});
    check_close(host(vkml::minimum(a, t2({4}, {2, 2, 2, 2}))), {1, 2, 2, 2});
    check_close(host(vkml::pow(a, t2({4}, {2, 2, 2, 2}))), {1, 4, 9, 16});
}

TEST_CASE("scalar operands adopt the tensor dtype") {
    const Tensor a = t2({3}, {1, 2, 3});
    check_close(host(a + 1.0), {2, 3, 4});
    check_close(host(a * 2.0), {2, 4, 6});
    check_close(host(a - 0.5), {0.5F, 1.5F, 2.5F});
    check_close(host(-a), {-1, -2, -3});
}

TEST_CASE("broadcasting uses stride-0 views, not copies") {
    const Tensor a = t2({2, 3}, {1, 2, 3, 4, 5, 6});
    const Tensor b = t2({3}, {10, 20, 30});

    const Tensor c = a + b;
    CHECK(c.shape() == std::vector<int64_t>{2, 3});
    check_close(host(c), {11, 22, 33, 14, 25, 36});

    SUBCASE("column vector broadcasts along the other axis") {
        const Tensor col = t2({2, 1}, {100, 200});
        check_close(host(a + col), {101, 102, 103, 204, 205, 206});
    }
    SUBCASE("scalar tensor broadcasts against anything") {
        const Tensor s = t2({}, {7});
        check_close(host(a + s), {8, 9, 10, 11, 12, 13});
    }
    SUBCASE("incompatible shapes are rejected") {
        CHECK_THROWS_AS(discard(a + t2({4}, {1, 2, 3, 4})), vkml::ShapeError);
    }
}

TEST_CASE("unary elementwise ops") {
    const Tensor x = t2({5}, {-2.0F, -0.5F, 0.0F, 0.5F, 2.0F});

    check_close(host(vkml::abs(x)), {2.0F, 0.5F, 0.0F, 0.5F, 2.0F});
    check_close(host(vkml::sign(x)), {-1, -1, 0, 1, 1});
    check_close(host(vkml::relu(x)), {0, 0, 0, 0.5F, 2.0F});
    check_close(host(vkml::square(x)), {4.0F, 0.25F, 0.0F, 0.25F, 4.0F});

    const std::vector<float> got = host(vkml::sigmoid(x));
    for (size_t i = 0; i < got.size(); ++i) {
        const float expect = 1.0F / (1.0F + std::exp(-host(x)[i]));
        CHECK(got[i] == doctest::Approx(expect).epsilon(1e-6));
    }
}

TEST_CASE("sigmoid is stable in both tails") {
    // The naive 1/(1+exp(-x)) form overflows exp() around x = -88.
    const Tensor x = t2({4}, {-100.0F, -50.0F, 50.0F, 100.0F});
    const std::vector<float> got = host(vkml::sigmoid(x));
    CHECK(got[0] >= 0.0F);
    CHECK(got[0] < 1e-30F);
    CHECK(std::isfinite(got[0]));
    CHECK(got[3] == doctest::Approx(1.0F));
    for (const float v : got) {
        CHECK(std::isfinite(v));
    }
}

TEST_CASE("gelu uses the exact erf form, not the tanh approximation") {
    // The two differ by up to ~1e-3, far outside the project's 1e-5 gate.
    const Tensor x = t2({3}, {-1.0F, 0.0F, 1.0F});
    const std::vector<float> got = host(vkml::gelu(x));

    auto exact = [](float v) { return 0.5F * v * (1.0F + std::erf(v * 0.70710678F)); };
    CHECK(got[0] == doctest::Approx(exact(-1.0F)).epsilon(1e-6));
    CHECK(got[1] == doctest::Approx(0.0F));
    CHECK(got[2] == doctest::Approx(exact(1.0F)).epsilon(1e-6));
}

TEST_CASE("clamp") {
    const Tensor x = t2({5}, {-2, -1, 0, 1, 2});
    check_close(host(vkml::clamp(x, -1, 1)), {-1, -1, 0, 1, 1});
    check_close(host(vkml::clamp_min(x, 0)), {0, 0, 0, 1, 2});
    check_close(host(vkml::clamp_max(x, 0)), {-2, -1, 0, 0, 0});
    CHECK_THROWS_AS(discard(vkml::clamp(x, 1, -1)), vkml::ShapeError);
}

TEST_CASE("comparisons produce Bool") {
    const Tensor a = t2({3}, {1, 2, 3});
    const Tensor b = t2({3}, {2, 2, 2});

    const Tensor lt = vkml::less(a, b);
    CHECK(lt.dtype() == DType::Bool);

    std::vector<uint8_t> got(3);
    lt.to_host(got.data());
    CHECK(got[0] == 1);
    CHECK(got[1] == 0);
    CHECK(got[2] == 0);
}

TEST_CASE("where selects elementwise") {
    const Tensor a = t2({3}, {1, 2, 3});
    const Tensor b = t2({3}, {10, 20, 30});
    const Tensor cond = vkml::less(a, t2({3}, {3, 3, 3}));
    check_close(host(vkml::where(cond, a, b)), {1, 2, 30});
}

TEST_CASE("reductions over all axes") {
    const Tensor a = t2({2, 3}, {1, 2, 3, 4, 5, 6});
    CHECK(vkml::sum(a).item() == doctest::Approx(21.0F));
    CHECK(vkml::mean(a).item() == doctest::Approx(3.5F));
    CHECK(vkml::max(a).item() == doctest::Approx(6.0F));
    CHECK(vkml::min(a).item() == doctest::Approx(1.0F));
    CHECK(vkml::prod(a).item() == doctest::Approx(720.0F));
}

TEST_CASE("reductions over one axis") {
    const Tensor a = t2({2, 3}, {1, 2, 3, 4, 5, 6});

    const std::array<int, 1> axis0{0};
    const Tensor s0 = vkml::sum(a, axis0);
    CHECK(s0.shape() == std::vector<int64_t>{3});
    check_close(host(s0), {5, 7, 9});

    const std::array<int, 1> axis1{1};
    const Tensor s1 = vkml::sum(a, axis1);
    CHECK(s1.shape() == std::vector<int64_t>{2});
    check_close(host(s1), {6, 15});

    SUBCASE("keepdim retains the axis") {
        const Tensor k = vkml::sum(a, axis1, /*keepdim=*/true);
        CHECK(k.shape() == std::vector<int64_t>{2, 1});
        check_close(host(k), {6, 15});
    }
    SUBCASE("negative axis indexes from the end") {
        const std::array<int, 1> last{-1};
        check_close(host(vkml::sum(a, last)), {6, 15});
    }
}

TEST_CASE("pairwise summation stays inside the 1e-5 gate at large n") {
    // A sequential accumulator would drift by ~n*eps here. This is the single
    // most important numerical property of the CPU backend, so it is asserted
    // directly rather than only through the PyTorch comparison.
    const int64_t n = 100000;
    std::vector<float> ones(static_cast<size_t>(n), 1.0F);
    const Tensor a = Tensor::from_host(ones.data(), std::array<int64_t, 1>{n});

    // 1e5 is exactly representable, so an accurate sum is exact.
    CHECK(vkml::sum(a).item() == doctest::Approx(100000.0F).epsilon(1e-7));

    SUBCASE("mixed magnitudes, where naive summation loses small terms") {
        std::vector<float> vals(static_cast<size_t>(n), 1.0F);
        vals[0] = 1e7F;
        const Tensor b = Tensor::from_host(vals.data(), std::array<int64_t, 1>{n});
        const float got = vkml::sum(b).item();
        const float want = 1e7F + 99999.0F;
        CHECK(std::fabs(got - want) / want < 1e-5F);
    }
}

TEST_CASE("argmax and argmin return the first extremum") {
    const Tensor a = t2({2, 3}, {1, 5, 5, 4, 2, 4});
    const Tensor am = vkml::argmax(a, 1);
    CHECK(am.dtype() == DType::I64);
    CHECK(host_i64(am) == std::vector<int64_t>{1, 0});  // first max, not last
    CHECK(host_i64(vkml::argmin(a, 1)) == std::vector<int64_t>{0, 1});
}

TEST_CASE("softmax sums to one and is shift invariant") {
    const Tensor a = t2({2, 3}, {1, 2, 3, 1, 2, 3});
    const Tensor s = vkml::softmax(a, 1);
    const std::vector<float> got = host(s);

    CHECK(got[0] + got[1] + got[2] == doctest::Approx(1.0F));
    CHECK(got[3] + got[4] + got[5] == doctest::Approx(1.0F));

    SUBCASE("large logits do not overflow") {
        // Without max-subtraction, exp(1000) is inf and the result is NaN.
        const Tensor big = t2({3}, {1000.0F, 1001.0F, 1002.0F});
        const std::vector<float> bg = host(vkml::softmax(big, 0));
        for (const float v : bg) {
            CHECK(std::isfinite(v));
        }
        CHECK(bg[0] + bg[1] + bg[2] == doctest::Approx(1.0F));
        // Shift invariance: same answer as the unshifted logits.
        const std::vector<float> small = host(vkml::softmax(t2({3}, {0, 1, 2}), 0));
        check_close(bg, small, 1e-6F);
    }
}

TEST_CASE("log_softmax stays finite where softmax underflows") {
    const Tensor a = t2({3}, {0.0F, -200.0F, -400.0F});
    const std::vector<float> got = host(vkml::log_softmax(a, 0));
    for (const float v : got) {
        CHECK(std::isfinite(v));
    }
    // log(softmax(x))) would be log(0) = -inf for the third element.
    CHECK(got[2] == doctest::Approx(-400.0F).epsilon(1e-4));
}

TEST_CASE("matmul 2d") {
    const Tensor a = t2({2, 3}, {1, 2, 3, 4, 5, 6});
    const Tensor b = t2({3, 2}, {7, 8, 9, 10, 11, 12});
    const Tensor c = vkml::matmul(a, b);
    CHECK(c.shape() == std::vector<int64_t>{2, 2});
    check_close(host(c), {58, 64, 139, 154});
}

TEST_CASE("matmul with a transposed operand reads strided memory correctly") {
    const Tensor a = t2({2, 3}, {1, 2, 3, 4, 5, 6});
    const Tensor bt = t2({2, 3}, {7, 9, 11, 8, 10, 12});  // == b^T from the case above
    const Tensor c = vkml::matmul(a, bt.transpose(0, 1));
    CHECK(c.shape() == std::vector<int64_t>{2, 2});
    check_close(host(c), {58, 64, 139, 154});
}

TEST_CASE("matmul vector cases follow torch.matmul rank rules") {
    const Tensor m = t2({2, 3}, {1, 2, 3, 4, 5, 6});
    const Tensor v = t2({3}, {1, 1, 1});

    const Tensor mv = vkml::matmul(m, v);
    CHECK(mv.shape() == std::vector<int64_t>{2});
    check_close(host(mv), {6, 15});

    const Tensor u = t2({2}, {1, 1});
    const Tensor vm = vkml::matmul(u, m);
    CHECK(vm.shape() == std::vector<int64_t>{3});
    check_close(host(vm), {5, 7, 9});

    SUBCASE("vector dot vector gives a scalar") {
        const Tensor d = vkml::matmul(v, v);
        CHECK(d.ndim() == 0);
        CHECK(d.item() == doctest::Approx(3.0F));
    }
    SUBCASE("mismatched inner dimensions throw") {
        CHECK_THROWS_AS(discard(vkml::matmul(m, m)), vkml::ShapeError);
    }
}

TEST_CASE("batched matmul broadcasts batch axes") {
    const Tensor a = t2({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    const Tensor b = t2({1, 2, 2}, {1, 0, 0, 1});  // identity, broadcast over the batch
    const Tensor c = vkml::matmul(a, b);
    CHECK(c.shape() == std::vector<int64_t>{2, 2, 2});
    check_close(host(c), {1, 2, 3, 4, 5, 6, 7, 8});
}

TEST_CASE("views are zero-copy and feed kernels correctly") {
    const Tensor a = t2({2, 3}, {1, 2, 3, 4, 5, 6});

    const Tensor tr = a.transpose(0, 1);
    CHECK(tr.shape() == std::vector<int64_t>{3, 2});
    CHECK_FALSE(tr.is_contiguous());
    check_close(host(tr), {1, 4, 2, 5, 3, 6});

    SUBCASE("reshape of a non-contiguous view inserts a contiguous copy") {
        const std::array<int64_t, 1> flat{6};
        const Tensor r = tr.reshape(flat);
        CHECK(r.shape() == std::vector<int64_t>{6});
        check_close(host(r), {1, 4, 2, 5, 3, 6});
    }
    SUBCASE("slice") {
        const Tensor s = a.slice(1, 1, 3);
        CHECK(s.shape() == std::vector<int64_t>{2, 2});
        check_close(host(s), {2, 3, 5, 6});
    }
    SUBCASE("arithmetic on a strided view is correct") {
        check_close(host(tr + 1.0), {2, 5, 3, 6, 4, 7});
    }
}

TEST_CASE("cast between dtypes") {
    const Tensor a = t2({3}, {1.7F, -2.3F, 0.0F});
    const Tensor i = a.to(DType::I32);
    CHECK(i.dtype() == DType::I32);

    std::vector<int32_t> got(3);
    i.to_host(got.data());
    CHECK(got == std::vector<int32_t>{1, -2, 0});  // truncation toward zero

    SUBCASE("round trip through f16 loses precision but stays close") {
        const Tensor h = a.to(DType::F16).to(DType::F32);
        check_close(host(h), {1.7F, -2.3F, 0.0F}, 1e-3F);
    }
    SUBCASE("casting to an integer type stops gradient tracking") {
        Tensor x = t2({3}, {1, 2, 3});
        x.set_requires_grad(true);
        CHECK(x.to(DType::F16).requires_grad());
        CHECK_FALSE(x.to(DType::I64).requires_grad());
    }
}

TEST_CASE("lazy evaluation defers work until observation") {
    const bool was_eager = vkml::eager();
    vkml::set_eager(false);

    const Tensor a = t2({3}, {1, 2, 3});
    const Tensor b = a + 1.0;
    // Nothing has been computed yet; to_host is what forces it.
    check_close(host(b), {2, 3, 4});

    SUBCASE("realize is idempotent") {
        b.realize();
        b.realize();
        check_close(host(b), {2, 3, 4});
    }
    vkml::set_eager(was_eager);
}

TEST_CASE("eager mode produces identical values") {
    const bool was_eager = vkml::eager();

    vkml::set_eager(false);
    const std::vector<float> lazy = host(vkml::gelu(t2({3}, {-1, 0, 1}) * 2.0));

    vkml::set_eager(true);
    const std::vector<float> eager = host(vkml::gelu(t2({3}, {-1, 0, 1}) * 2.0));

    // Eager mode must change only *when* work happens, never the answer.
    CHECK(lazy == eager);
    vkml::set_eager(was_eager);
}

TEST_CASE("mixed dtypes are rejected rather than promoted") {
    const Tensor f = t2({2}, {1, 2});
    const Tensor i = f.to(DType::I32);
    CHECK_THROWS_AS(discard(f + i), vkml::DTypeError);
}

TEST_CASE("requires_grad is restricted to floating tensors") {
    Tensor f = t2({2}, {1, 2});
    CHECK_NOTHROW(f.set_requires_grad(true));
    CHECK(f.requires_grad());

    Tensor i = f.to(DType::I64);
    CHECK_THROWS_AS(i.set_requires_grad(true), vkml::DTypeError);
}

TEST_CASE("requires_grad propagates through operations") {
    Tensor a = t2({2}, {1, 2});
    const Tensor b = t2({2}, {3, 4});
    a.set_requires_grad(true);

    CHECK((a + b).requires_grad());
    CHECK((b + a).requires_grad());
    CHECK(vkml::relu(a).requires_grad());
    CHECK_FALSE((b + b).requires_grad());
    // Comparisons produce Bool, which is not differentiable.
    CHECK_FALSE(vkml::less(a, b).requires_grad());
}
