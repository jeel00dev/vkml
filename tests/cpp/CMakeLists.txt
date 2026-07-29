# doctest is vendored as a single header in third_party/ rather than pulled in
# with FetchContent. Two reasons: the build then needs no network, and CMake 4
# rejects the `cmake_minimum_required` floor declared by several releases of
# doctest's own build files.

add_library(doctest INTERFACE)
target_include_directories(doctest SYSTEM INTERFACE ${CMAKE_SOURCE_DIR}/third_party/doctest)

# Without this, doctest forward-declares std::ostream inside namespace std --
# something its own source comments admit "is not permitted by the C++
# Standard". It disables the hack automatically for libc++, but not for MSVC,
# whose STL rejects the declaration outright (C2027: use of undefined type
# 'std::basic_ostream'). That is how the Windows job failed on its second run,
# once the real library defects it found first had been fixed.
#
# Set unconditionally rather than only for MSVC. There is no reason for the
# platforms to differ here, and one path compiled everywhere is worth more than
# a branch only a single CI job ever exercises. The cost is <ostream> and
# <istream> in each test translation unit.
target_compile_definitions(doctest INTERFACE DOCTEST_CONFIG_USE_STD_HEADERS)

set(VKML_TEST_SOURCES
    main.cpp
    test_util.cpp
    test_core.cpp
    test_allocator.cpp
    test_shape.cpp
    test_graph.cpp
    test_ops.cpp
    test_aliasing.cpp
)

if(VKML_VULKAN)
    list(APPEND VKML_TEST_SOURCES test_vulkan_device.cpp)
endif()

add_executable(vkml_tests ${VKML_TEST_SOURCES})

target_link_libraries(vkml_tests PRIVATE vkml doctest vkml_warnings)

if(VKML_VULKAN)
    # The Vulkan test reaches into the backend's internal header directly:
    # vk_device.h is intentionally not a public header, and exposing it just to
    # test it would defeat the point of keeping Vulkan types out of the API.
    target_include_directories(vkml_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/src/backend/vulkan
        ${CMAKE_BINARY_DIR}/generated)
    target_link_libraries(vkml_tests PRIVATE Vulkan::Vulkan)
endif()


add_test(NAME vkml_tests COMMAND vkml_tests)

# Surface doctest's own subcase names to ctest so a failure names the case.
set_tests_properties(vkml_tests PROPERTIES
    FAIL_REGULAR_EXPRESSION "\\[doctest\\] assertion.*FAILED")
