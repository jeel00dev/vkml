# nanobind was chosen over pybind11 (open question 1 in docs/ARCHITECTURE.md
# §10). Verified working against Python 3.14.6 / nanobind 2.13.0 on this
# machine. The deciding factors were binary size and compile time -- nanobind
# produces roughly 4x smaller extensions and compiles noticeably faster, which
# matters because the binding TU is rebuilt on every core header change.
#
# The choice is contained: it touches this file and bindings/*.cpp only.

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_VARIABLE NANOBIND_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NANOBIND_LOOKUP_RESULT
    ERROR_QUIET)

if(NOT NANOBIND_LOOKUP_RESULT EQUAL 0)
    message(FATAL_ERROR
        "nanobind not found for ${Python_EXECUTABLE}.\n"
        "Install it with:  ${Python_EXECUTABLE} -m pip install nanobind\n"
        "Or configure with -DVKML_BUILD_PYTHON=OFF to skip the Python extension.")
endif()

list(APPEND CMAKE_PREFIX_PATH "${NANOBIND_CMAKE_DIR}")
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_vkml_core
    STABLE_ABI
    NB_STATIC
    module.cpp)

target_link_libraries(_vkml_core PRIVATE vkml)

# Drop the built extension straight into the Python package so that an in-tree
# `import vkml` works without an install step. This keeps the edit-build-test
# loop short, which matters a lot while the validation suite is being written.
set_target_properties(_vkml_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/vkml)

# ...and install it beside the package for a wheel build. Both are needed and
# they are not the same thing: the line above serves the in-tree loop, while
# scikit-build-core assembles a wheel from what CMake INSTALLS. Without this the
# wheel would contain the Python files and no extension, and fail at import.
install(TARGETS _vkml_core LIBRARY DESTINATION vkml)
