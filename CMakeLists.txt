cmake_minimum_required(VERSION 3.25)

project(vkml
    VERSION 0.1.0
    DESCRIPTION "Vulkan-native deep learning framework"
    LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
#
# VKML_VULKAN is OFF and stays OFF through M0. The architecture document is
# explicit that no Vulkan work begins until the CPU oracle and the validation
# harness are complete (docs/ARCHITECTURE.md §8, M0/M1).
# ---------------------------------------------------------------------------
option(VKML_BUILD_TESTS  "Build C++ unit tests"           ON)
option(VKML_BUILD_PYTHON "Build the Python extension"     ON)
option(VKML_BUILD_BENCH  "Build benchmarks"               ON)
option(VKML_VULKAN       "Build the Vulkan backend"        OFF)
option(VKML_WERROR       "Treat warnings as errors"       OFF)
option(VKML_SANITIZE     "Enable ASan+UBSan"              OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---------------------------------------------------------------------------
# Warning sets
#
# Two targets rather than one, because the numerics warnings are only wanted on
# library code:
#
#   vkml_warnings          general correctness warnings, applied everywhere.
#   vkml_numeric_warnings  float-precision warnings, applied to the library
#                          only. Silent float->double promotion changes both
#                          the cost and the rounding of a kernel, and the
#                          correctness gate is a 1e-5 tolerance against PyTorch
#                          (docs/ARCHITECTURE.md §7.3), so these matter a great
#                          deal in src/ -- and not at all in tests, where
#                          doctest::Approx takes a double and would trip
#                          -Wdouble-promotion on every float comparison.
#
# Splitting them is cleaner than suppressing per-target, since compile options
# inherited from an INTERFACE library are appended after the target's own and
# would win the argument anyway.
# ---------------------------------------------------------------------------
add_library(vkml_warnings INTERFACE)
add_library(vkml_numeric_warnings INTERFACE)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(vkml_warnings INTERFACE
        -Wall -Wextra -Wpedantic
        -Wshadow
        -Wnon-virtual-dtor
        -Wcast-align
        -Woverloaded-virtual
        -Wunused)
    target_compile_options(vkml_numeric_warnings INTERFACE
        -Wdouble-promotion
        -Wfloat-conversion)
    if(VKML_WERROR)
        target_compile_options(vkml_warnings INTERFACE -Werror)
    endif()
elseif(MSVC)
    target_compile_options(vkml_warnings INTERFACE /W4)
    if(VKML_WERROR)
        target_compile_options(vkml_warnings INTERFACE /WX)
    endif()
endif()

add_library(vkml_sanitizers INTERFACE)

if(VKML_SANITIZE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(vkml_sanitizers INTERFACE
            -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
        target_link_options(vkml_sanitizers INTERFACE -fsanitize=address,undefined)
    else()
        message(WARNING "VKML_SANITIZE is not supported for this compiler; ignoring")
    endif()
endif()

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
if(VKML_VULKAN)
    find_package(Vulkan REQUIRED)
    add_subdirectory(shaders)
endif()

add_subdirectory(src)

if(VKML_BUILD_PYTHON)
    add_subdirectory(bindings)
endif()

if(VKML_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()

if(VKML_BUILD_BENCH)
    add_subdirectory(bench/cpp)
endif()

message(STATUS "")
message(STATUS "vkml ${PROJECT_VERSION}")
message(STATUS "  build type ....... ${CMAKE_BUILD_TYPE}")
message(STATUS "  compiler ......... ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  tests ............ ${VKML_BUILD_TESTS}")
message(STATUS "  python ........... ${VKML_BUILD_PYTHON}")
message(STATUS "  benchmarks ....... ${VKML_BUILD_BENCH}")
message(STATUS "  vulkan backend ... ${VKML_VULKAN}")
message(STATUS "  sanitizers ....... ${VKML_SANITIZE}")
message(STATUS "")
