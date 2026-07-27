#include "doctest.h"
#include "test_support.h"

#include "vkml/graph/graph.h"

#include <vector>

using vkml::DType;
using vkml::Device;
using vkml::make_node;
using vkml::make_view;
using vkml::Node;
using vkml::NodePtr;
using vkml::OpKind;
using vkml::Shape;

namespace {

NodePtr leaf(std::vector<int64_t> dims = {2, 3}) {
    auto n = make_node(OpKind::Input, Shape::contiguous(dims, 4), DType::F32, Device::cpu());
    return n;
}

NodePtr realized_leaf(std::vector<int64_t> dims = {2, 3}) {
    auto n = leaf(dims);
    n->storage = vkml::make_cpu_storage(n->shape.nbytes());
    return n;
}

NodePtr binary(OpKind op, const NodePtr& a, const NodePtr& b) {
    auto n = make_node(op, a->shape, a->dtype, a->device);
    n->src[0] = a;
    n->src[1] = b;
    n->n_src = 2;
    return n;
}

NodePtr unary(OpKind op, const NodePtr& a) {
    auto n = make_node(op, a->shape, a->dtype, a->device);
    n->src[0] = a;
    n->n_src = 1;
    return n;
}

}  // namespace

TEST_CASE("op names cover every op") {
    for (int i = 0; i < vkml::kNumOps; ++i) {
        const auto name = vkml::op_name(static_cast<OpKind>(i));
        CHECK(!name.empty());
        CHECK(name != "<invalid>");
    }
}

TEST_CASE("op classification predicates") {
    CHECK(vkml::is_view_op(OpKind::Reshape));
    CHECK(vkml::is_view_op(OpKind::Broadcast));
    CHECK_FALSE(vkml::is_view_op(OpKind::Contiguous));
    CHECK_FALSE(vkml::is_view_op(OpKind::Add));

    CHECK(vkml::is_comparison_op(OpKind::Less));
    CHECK_FALSE(vkml::is_comparison_op(OpKind::Add));
}

TEST_CASE("OpParams round-trips typed structs without heap allocation") {
    vkml::OpParams p;
    const vkml::AdamParams in{.lr = 1e-3F,
                              .beta1 = 0.9F,
                              .beta2 = 0.999F,
                              .eps = 1e-8F,
                              .weight_decay = 0.01F,
                              .step = 42};
    p.set(in);
    const auto out = p.get<vkml::AdamParams>();
    CHECK(out.lr == doctest::Approx(1e-3F));
    CHECK(out.beta2 == doctest::Approx(0.999F));
    CHECK(out.step == 42);

    SUBCASE("a different op's params reuse the same inline buffer") {
        p.set(vkml::SliceParams{.axis = 1, .start = 2, .stop = 5, .step = 2});
        const auto s = p.get<vkml::SliceParams>();
        CHECK(s.axis == 1);
        CHECK(s.stop == 5);
    }
}

TEST_CASE("topological order puts sources before consumers") {
    auto a = leaf();
    auto b = leaf();
    auto c = binary(OpKind::Add, a, b);
    auto d = unary(OpKind::Relu, c);

    const auto order = vkml::topological_order(d);
    REQUIRE(order.size() == 4);
    CHECK(order.back() == d.get());

    // a and b must both precede c, which must precede d.
    auto pos = [&](const NodePtr& n) {
        for (size_t i = 0; i < order.size(); ++i) {
            if (order[i] == n.get()) {
                return static_cast<int>(i);
            }
        }
        return -1;
    };
    CHECK(pos(a) < pos(c));
    CHECK(pos(b) < pos(c));
    CHECK(pos(c) < pos(d));
}

TEST_CASE("a diamond visits the shared node exactly once") {
    auto x = leaf();
    auto l = unary(OpKind::Relu, x);
    auto r = unary(OpKind::Neg, x);
    auto out = binary(OpKind::Add, l, r);

    const auto order = vkml::topological_order(out);
    CHECK(order.size() == 4);  // x, l, r, out -- x appears once, not twice

    int seen_x = 0;
    for (const Node* n : order) {
        if (n == x.get()) {
            ++seen_x;
        }
    }
    CHECK(seen_x == 1);
}

TEST_CASE("realised nodes terminate the traversal") {
    // This is what makes repeated realize() cheap: an already-computed tensor
    // does not drag its construction history back into the schedule.
    auto a = leaf();
    auto b = unary(OpKind::Relu, a);
    b->storage = vkml::make_cpu_storage(b->shape.nbytes());

    auto c = unary(OpKind::Neg, b);

    const auto order = vkml::topological_order(c);
    REQUIRE(order.size() == 1);
    CHECK(order[0] == c.get());

    SUBCASE("a fully realised root schedules no work") {
        const auto none = vkml::topological_order(b);
        CHECK(none.empty());
    }
}

TEST_CASE("view_src collapses to the root but src[0] keeps the real chain") {
    auto base = realized_leaf({4, 5});

    const std::vector<int64_t> flat{20};
    auto v1 = make_view(OpKind::Reshape, base, *base->shape.reshaped(flat), 0);
    auto v2 = make_view(OpKind::Slice, v1, v1->shape.sliced(0, 4, 12).shape, 16);

    // STORAGE anchor: collapsed, so realisation resolves to one buffer plus one
    // offset without walking a list.
    CHECK(v2->view_src == base);
    CHECK(v1->view_src == base);
    CHECK(v2->view_offset == 16);

    // DEPENDENCY edge: the immediate base, NOT the root. Autograd reads
    // src[0]->shape to decide what shape a gradient must have, so collapsing
    // this too produces correctly-valued but wrongly-shaped gradients through
    // transposed views. Forward execution cannot detect that, which is why it
    // is pinned here explicitly.
    CHECK(v2->src[0] == v1);
    CHECK(v1->src[0] == base);
}

TEST_CASE("view nodes propagate requires_grad") {
    auto base = leaf();
    base->requires_grad = true;
    const std::vector<int64_t> flat{6};
    auto v = make_view(OpKind::Reshape, base, *base->shape.reshaped(flat), 0);
    CHECK(v->requires_grad);
}

TEST_CASE("use counts match the number of later consumers") {
    auto x = leaf();
    auto l = unary(OpKind::Relu, x);
    auto r = unary(OpKind::Neg, x);
    auto out = binary(OpKind::Add, l, r);

    const auto order = vkml::topological_order(out);
    const auto counts = vkml::compute_use_counts(order);
    REQUIRE(counts.size() == order.size());

    auto count_of = [&](const NodePtr& n) {
        for (size_t i = 0; i < order.size(); ++i) {
            if (order[i] == n.get()) {
                return counts[i];
            }
        }
        return -1;
    };
    CHECK(count_of(x) == 2);    // consumed by both l and r
    CHECK(count_of(l) == 1);
    CHECK(count_of(out) == 0);  // nothing consumes the root
}

TEST_CASE("deep chains do not overflow the stack") {
    // An unrolled RNN is exactly this shape. A recursive DFS would die here,
    // which is why topological_order is iterative.
    auto n = leaf({1});
    for (int i = 0; i < 200000; ++i) {
        n = unary(OpKind::Relu, n);
    }
    const auto order = vkml::topological_order(n);
    CHECK(order.size() == 200001);
}

TEST_CASE("to_dot emits every node and edge") {
    auto a = realized_leaf();
    auto b = leaf();
    auto c = binary(OpKind::Add, a, b);

    const std::array<NodePtr, 1> roots{c};
    const std::string dot = vkml::to_dot(roots);

    CHECK(dot.find("digraph vkml") != std::string::npos);
    CHECK(dot.find("add") != std::string::npos);
    CHECK(dot.find("input") != std::string::npos);
    // The realised leaf is shaded rather than omitted.
    CHECK(dot.find("style=filled") != std::string::npos);
}
