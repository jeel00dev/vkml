#pragma once

#include <format>
#include <stdexcept>
#include <string>
#include <string_view>

// Error handling policy
// ---------------------
// vkml reports failures by throwing, never by aborting.
//
// This is a deliberate departure from ggml, whose GGML_ASSERT calls abort().
// Aborting is defensible for a CLI binary; it is not defensible for a library
// that is primarily driven from Python, where a failed assertion would take
// down the user's interpreter (and their notebook state) instead of raising a
// catchable exception. Every exception below maps onto a natural Python
// exception in the binding layer.
//
// The cost is that every internal invariant check is a potential throw, so
// code between an allocation and its owner must be exception-safe. RAII
// everywhere makes that mostly automatic.

namespace vkml {

/// Base of every exception vkml throws. Maps to Python RuntimeError.
class Error : public std::runtime_error {
public:
    explicit Error(const std::string& what) : std::runtime_error(what) {}
};

/// Shape, rank, or broadcasting incompatibility. Maps to Python ValueError.
class ShapeError : public Error {
public:
    explicit ShapeError(const std::string& what) : Error(what) {}
};

/// Wrong or unsupported dtype for an operation. Maps to Python TypeError.
class DTypeError : public Error {
public:
    explicit DTypeError(const std::string& what) : Error(what) {}
};

/// Device mismatch between operands, or an unavailable backend.
class DeviceError : public Error {
public:
    explicit DeviceError(const std::string& what) : Error(what) {}
};

/// Out-of-range index or slice. Maps to Python IndexError.
class IndexError : public Error {
public:
    explicit IndexError(const std::string& what) : Error(what) {}
};

/// A path that is planned but not yet built. Maps to Python NotImplementedError.
class NotImplementedError : public Error {
public:
    explicit NotImplementedError(const std::string& what) : Error(what) {}
};

/// A broken internal invariant: always a bug in vkml, never in user input.
class InternalError : public Error {
public:
    explicit InternalError(const std::string& what) : Error(what) {}
};

/// Allocation failure on some device.
class OutOfMemoryError : public Error {
public:
    explicit OutOfMemoryError(const std::string& what) : Error(what) {}
};

namespace detail {

/// Prefix an internal-error message with its source location.
[[nodiscard]] std::string format_internal(std::string_view file, int line, std::string_view func,
                                          std::string_view cond, std::string_view msg);

}  // namespace detail
}  // namespace vkml
