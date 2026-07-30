cmake_minimum_required(VERSION 3.25)

project(vkml
    VERSION 0.1.0
    DESCRIPTION "Vulkan-native deep learning framework"
    LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
#
# VKML_VULKAN defaults OFF so that a plain `cmake ..` builds and tests the CPU
# backend without needing the Vulkan SDK -- that backend is the correctness
# oracle and is useful on its own. It is not a statement about the project's
# focus: the Python wheel turns it ON (pyproject.toml), and turning it on is what
# gets you the library's reason for existing. The PRESETS do not -- they set only
# the build type and warning flags -- so a CMake build wanting Vulkan passes
# -DVKML_VULKAN=ON explicitly, as README.md's build commands do.
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

# Whether sanitizers are actually IN the build, as opposed to merely requested.
# The summary reports this rather than VKML_SANITIZE, which said "ON" for an MSVC
# build that had just warned it was ignoring the request -- a build that looks
# instrumented and is not is worse than one that plainly is not (issue #17).
set(VKML_SANITIZE_EFFECTIVE OFF)

if(VKML_SANITIZE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(vkml_sanitizers INTERFACE
            -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
        target_link_options(vkml_sanitizers INTERFACE -fsanitize=address,undefined)
        set(VKML_SANITIZE_EFFECTIVE ON)
    else()
        message(WARNING "VKML_SANITIZE is not supported for this compiler; ignoring")
    endif()
endif()

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
if(VKML_VULKAN)
    # Not REQUIRED: FindVulkan's own failure names the two cache variables it
    # could not fill and stops, which tells a user nothing they can act on. The
    # remedy is knowable here, so it is stated (issue #5).
    #
    # VULKAN_SDK is how FindVulkan locates a Windows SDK, and the LunarG
    # installer does not always set it -- reported from a machine with the SDK
    # fully installed and the variable absent at both user and machine scope, so
    # "open a new terminal" was not the answer the README assumed.
    find_package(Vulkan)
    if(NOT Vulkan_FOUND)
        set(_vkml_sdk_hint "")
        if(WIN32)
            file(GLOB _vkml_sdk_candidates "C:/VulkanSDK/*")
            if(_vkml_sdk_candidates)
                list(GET _vkml_sdk_candidates -1 _vkml_newest_sdk)
                set(_vkml_sdk_hint
                    "\n  An SDK appears to be installed at ${_vkml_newest_sdk}."
                    "\n  Set VULKAN_SDK to it, then configure again:"
                    "\n      $env:VULKAN_SDK = \"${_vkml_newest_sdk}\"")
            endif()
        endif()
        message(FATAL_ERROR
            "The Vulkan SDK was not found, and VKML_VULKAN=ON needs it.\n"
            "  Install it from https://vulkan.lunarg.com/sdk/home and make sure\n"
            "  VULKAN_SDK points at the SDK root (currently: '$ENV{VULKAN_SDK}')."
            ${_vkml_sdk_hint}
            "\n"
            "\n  To build without the GPU backend instead, configure with\n"
            "      -DVKML_VULKAN=OFF\n"
            "  which gives the CPU backend only.")
    endif()
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

# A multi-config generator leaves CMAKE_BUILD_TYPE empty on purpose -- the
# configuration is chosen by --config at build time -- so printing it bare gave
# a blank line on every Visual Studio build. Reporting the EFFECTIVE state
# rather than the requested one is the same reason the sanitizer line below is
# computed rather than echoed (issue #17).
if(CMAKE_CONFIGURATION_TYPES)
    message(STATUS "  build type ....... ${CMAKE_CONFIGURATION_TYPES} (chosen by --config)")
else()
    message(STATUS "  build type ....... ${CMAKE_BUILD_TYPE}")
endif()
message(STATUS "  compiler ......... ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  tests ............ ${VKML_BUILD_TESTS}")
message(STATUS "  python ........... ${VKML_BUILD_PYTHON}")
message(STATUS "  benchmarks ....... ${VKML_BUILD_BENCH}")
message(STATUS "  vulkan backend ... ${VKML_VULKAN}")
message(STATUS "  sanitizers ....... ${VKML_SANITIZE_EFFECTIVE}")
message(STATUS "")
