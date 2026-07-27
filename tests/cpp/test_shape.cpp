#include "doctest.h"
#include "test_support.h"

#include "vkml/core/shape.h"
#include "vkml/util/error.h"

#include <vector>

using vkml::Shape;

namespace {

Shape make(std::vector<int64_t> dims, size_t itemsize = 4) {
    return Shape::contiguous(dims, itemsize);
}

}  // namespace

TEST_CASE("contiguous strides are row-major") {
    // This is the load-bearing convention choice: dims[0] is the OUTERMOST
    // axis, so it gets the LARGEST stride. ggml is the other way round.
    const Shape s = make({2, 3, 4});
    CHECK(s.ndim() == 3);
    CHECK(s.stride(0) == 48);  // 3 * 4 * 4 bytes
    CHECK(s.stride(1) == 16);  // 4 * 4 bytes
    CHECK(s.stride(2) == 4);   // itemsize
    CHECK(s.numel() == 24);
    CHECK(s.nbytes() == 96);
    CHECK(s.is_contiguous());
}

TEST_CASE("rank-0 scalar has one element") {
    const Shape s;
    CHECK(s.ndim() == 0);
    CHECK(s.numel() == 1);
    CHECK(s.is_contiguous());
}

TEST_CASE("empty tensors are legal and contiguous") {
    const Shape s = make({0, 5});
    CHECK(s.numel() == 0);
    CHECK(s.nbytes() == 0);
    CHECK(s.is_contiguous());
}

TEST_CASE("rank above kMaxDims is rejected") {
    CHECK_THROWS_AS(discard(make({2, 2, 2, 2, 2})), vkml::ShapeError);
}

TEST_CASE("transpose makes a shape non-contiguous") {
    const Shape t = make({2, 3}).transposed(0, 1);
    CHECK(t.dim(0) == 3);
    CHECK(t.dim(1) == 2);
    CHECK(t.stride(0) == 4);
    CHECK(t.stride(1) == 12);
    CHECK_FALSE(t.is_contiguous());

    SUBCASE("negative axes are accepted") {
        const Shape u = make({2, 3}).transposed(-2, -1);
        CHECK(u == t);
    }
}

TEST_CASE("extents of 1 do not affect contiguity") {
    // A (1, N) tensor transposed to (N, 1) is still contiguous: the axis of
    // extent 1 cannot constrain the memory walk. numpy and torch agree.
    const Shape s = make({1, 5}).transposed(0, 1);
    CHECK(s.dim(0) == 5);
    CHECK(s.dim(1) == 1);
    CHECK(s.is_contiguous());
}

TEST_CASE("permute reorders axes and validates the permutation") {
    const Shape s = make({2, 3, 4});
    const std::vector<int> perm{2, 0, 1};
    const Shape p = s.permuted(perm);
    CHECK(p.dim(0) == 4);
    CHECK(p.dim(1) == 2);
    CHECK(p.dim(2) == 3);
    CHECK(p.stride(0) == 4);
    CHECK(p.stride(1) == 48);
    CHECK(p.stride(2) == 16);

    SUBCASE("repeated axis is rejected") {
        const std::vector<int> bad{0, 0, 1};
        CHECK_THROWS_AS(discard(s.permuted(bad)), vkml::ShapeError);
    }
    SUBCASE("wrong length is rejected") {
        const std::vector<int> bad{0, 1};
        CHECK_THROWS_AS(discard(s.permuted(bad)), vkml::ShapeError);
    }
}

TEST_CASE("squeeze and unsqueeze are inverses") {
    const Shape s = make({2, 3});
    const Shape u = s.unsqueezed(1);
    CHECK(u.ndim() == 3);
    CHECK(u.dim(0) == 2);
    CHECK(u.dim(1) == 1);
    CHECK(u.dim(2) == 3);
    CHECK(u.is_contiguous());
    CHECK(u.squeezed(1) == s);

    SUBCASE("unsqueeze accepts the end position") {
        const Shape e = s.unsqueezed(2);
        CHECK(e.ndim() == 3);
        CHECK(e.dim(2) == 1);
        CHECK(e.is_contiguous());
    }
    SUBCASE("unsqueeze accepts -1 as the end position") {
        CHECK(s.unsqueezed(-1) == s.unsqueezed(2));
    }
    SUBCASE("squeezing a non-unit axis is rejected") {
        CHECK_THROWS_AS(discard(s.squeezed(0)), vkml::ShapeError);
    }
}

TEST_CASE("broadcast_to uses stride 0 rather than copying") {
    const Shape s = make({3, 1});
    const std::vector<int64_t> target{2, 3, 4};
    const Shape b = s.broadcast_to(target);

    CHECK(b.ndim() == 3);
    CHECK(b.dim(0) == 2);
    CHECK(b.dim(1) == 3);
    CHECK(b.dim(2) == 4);
    CHECK(b.stride(0) == 0);  // brand new leading axis
    CHECK(b.stride(1) == 4);  // carried through from the source
    CHECK(b.stride(2) == 0);  // extent 1 stretched to 4
    CHECK(b.has_broadcast_stride());
    CHECK_FALSE(b.is_contiguous());

    SUBCASE("incompatible extents are rejected") {
        const std::vector<int64_t> bad{2, 5, 4};
        CHECK_THROWS_AS(discard(s.broadcast_to(bad)), vkml::ShapeError);
    }
    SUBCASE("broadcasting down in rank is rejected") {
        const std::vector<int64_t> smaller{3};
        CHECK_THROWS_AS(discard(s.broadcast_to(smaller)), vkml::ShapeError);
    }
}

TEST_CASE("broadcast_dims follows numpy right-alignment") {
    const std::vector<int64_t> a{3, 1, 5};
    const std::vector<int64_t> b{4, 5};
    const auto r = vkml::broadcast_dims(a, b);
    CHECK(r == std::vector<int64_t>{3, 4, 5});

    SUBCASE("scalar broadcasts against anything") {
        const std::vector<int64_t> scalar{};
        CHECK(vkml::broadcast_dims(a, scalar) == a);
    }
    SUBCASE("an extent of 1 stretches against any other extent") {
        // {3,1,5} vs {2,5} right-aligns to 5<->5, 1<->2, 3<->(missing) = {3,2,5}.
        const std::vector<int64_t> stretch{2, 5};
        CHECK(vkml::broadcast_dims(a, stretch) == std::vector<int64_t>{3, 2, 5});
    }
    SUBCASE("mismatched extents throw") {
        // Neither 5 nor 4 is 1, so the innermost axis cannot be reconciled.
        const std::vector<int64_t> bad{2, 4};
        CHECK_THROWS_AS(discard(vkml::broadcast_dims(a, bad)), vkml::ShapeError);
    }
}

TEST_CASE("reshape works on contiguous shapes and infers one extent") {
    const Shape s = make({2, 3, 4});

    const std::vector<int64_t> flat{24};
    REQUIRE(s.reshaped(flat).has_value());
    CHECK(s.reshaped(flat)->ndim() == 1);

    const std::vector<int64_t> inferred{6, -1};
    const auto r = s.reshaped(inferred);
    REQUIRE(r.has_value());
    CHECK(r->dim(0) == 6);
    CHECK(r->dim(1) == 4);

    SUBCASE("element count must be preserved") {
        const std::vector<int64_t> wrong{5, 5};
        CHECK_THROWS_AS(discard(s.reshaped(wrong)), vkml::ShapeError);
    }
    SUBCASE("two inferred extents are rejected") {
        const std::vector<int64_t> wrong{-1, -1};
        CHECK_THROWS_AS(discard(s.reshaped(wrong)), vkml::ShapeError);
    }
}

TEST_CASE("reshape of a non-contiguous view reports failure instead of guessing") {
    // The caller is expected to insert an explicit contiguous() node. Returning
    // nullopt rather than silently copying is what keeps that copy visible in
    // the graph.
    const Shape t = make({2, 3}).transposed(0, 1);
    const std::vector<int64_t> flat{6};
    CHECK_FALSE(t.reshaped(flat).has_value());
}

TEST_CASE("slice narrows one axis and reports a byte offset") {
    const Shape s = make({4, 5});
    const auto r = s.sliced(0, 1, 3);
    CHECK(r.shape.dim(0) == 2);
    CHECK(r.shape.dim(1) == 5);
    CHECK(r.shape.stride(0) == 20);
    CHECK(r.offset_bytes == 20);  // one row of 5 f32

    SUBCASE("step scales the stride") {
        const auto st = s.sliced(1, 0, 5, 2);
        CHECK(st.shape.dim(1) == 3);  // indices 0, 2, 4
        CHECK(st.shape.stride(1) == 8);
        CHECK(st.offset_bytes == 0);
    }
    SUBCASE("out-of-range bounds throw") {
        CHECK_THROWS_AS(discard(s.sliced(0, 0, 99)), vkml::IndexError);
        CHECK_THROWS_AS(discard(s.sliced(0, 3, 1)), vkml::IndexError);
    }
    SUBCASE("non-positive step throws") {
        CHECK_THROWS_AS(discard(s.sliced(0, 0, 4, 0)), vkml::ShapeError);
    }
}

TEST_CASE("offset_of computes byte offsets") {
    const Shape s = make({2, 3, 4});
    const std::vector<int64_t> idx{1, 2, 3};
    CHECK(s.offset_of(idx) == 48 + 32 + 12);

    SUBCASE("out-of-range index throws") {
        const std::vector<int64_t> bad{2, 0, 0};
        CHECK_THROWS_AS(discard(s.offset_of(bad)), vkml::IndexError);
    }
    SUBCASE("wrong rank throws") {
        const std::vector<int64_t> bad{1, 1};
        CHECK_THROWS_AS(discard(s.offset_of(bad)), vkml::IndexError);
    }
}

TEST_CASE("normalize_dim handles negatives and rejects out-of-range") {
    CHECK(vkml::normalize_dim(-1, 3) == 2);
    CHECK(vkml::normalize_dim(0, 3) == 0);
    CHECK_THROWS_AS(discard(vkml::normalize_dim(3, 3)), vkml::IndexError);
    CHECK_THROWS_AS(discard(vkml::normalize_dim(-4, 3)), vkml::IndexError);
    CHECK_THROWS_AS(discard(vkml::normalize_dim(0, 0)), vkml::IndexError);

    SUBCASE("extra widens the range for unsqueeze") {
        CHECK(vkml::normalize_dim(3, 3, 1) == 3);
        CHECK(vkml::normalize_dim(-1, 3, 1) == 3);
        CHECK(vkml::normalize_dim(0, 0, 1) == 0);
    }
}

TEST_CASE("itemsize is respected throughout") {
    const Shape s = make({2, 3}, 8);  // i64
    CHECK(s.stride(0) == 24);
    CHECK(s.stride(1) == 8);
    CHECK(s.nbytes() == 48);
    CHECK(s.is_contiguous());
}

TEST_CASE("same_dims ignores strides") {
    const Shape a = make({2, 3});
    const Shape b = make({2, 3}).transposed(0, 1).transposed(0, 1);
    CHECK(a.same_dims(b));
    CHECK_FALSE(a.same_dims(make({3, 2})));
}
