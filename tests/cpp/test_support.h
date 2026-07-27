#pragma once

#include <utility>

namespace vkml_test {

/// Consumes a [[nodiscard]] result inside a throw-expecting assertion.
///
/// doctest's CHECK_THROWS_AS discards whatever its expression evaluates to,
/// which trips -Wunused-result for the many [[nodiscard]] functions in the
/// public API. Silencing the warning at the call site is the right fix; the
/// attributes themselves are load-bearing, since ignoring the result of
/// something like `shape.permuted(...)` in real code is always a bug.
template <typename T>
void discard(T&& value) noexcept {
    (void)value;
}

}  // namespace vkml_test

using vkml_test::discard;
