# The whole core is a single target.
#
# Per-layer object libraries would enforce the layering rule at link time, but
# they add real friction (every layer needs its own target, include set and
# install rule) for a project with ten layers and one consumer. The rule is
# enforced instead by scripts/check_layering.py in CI, which also produces a far
# better error message than a linker does.

set(VKML_SOURCES
    util/log.cpp
    util/error.cpp
    util/coverage.cpp

    core/dtype.cpp
    core/device.cpp
    core/allocator.cpp
    core/shape.cpp
    core/storage.cpp

    graph/op.cpp
    graph/graph.cpp
    graph/grad_mode.cpp

    backend/api/backend.cpp
    backend/cpu/cpu_backend.cpp
    backend/cpu/reduce.cpp
    backend/cpu/kernels_movement.cpp
    backend/cpu/kernels_elementwise.cpp
    backend/cpu/kernels_reduce.cpp
    backend/cpu/kernels_matmul.cpp

    dispatch/executor.cpp

    api/tensor.cpp
    api/ops.cpp

    autograd/autograd.cpp
)

add_library(vkml ${VKML_SOURCES})
add_library(vkml::vkml ALIAS vkml)

target_include_directories(vkml
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

target_link_libraries(vkml
    PRIVATE vkml_warnings vkml_numeric_warnings
    PUBLIC  vkml_sanitizers)

set_target_properties(vkml PROPERTIES
    VERSION   ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})

if(VKML_VULKAN)
    target_sources(vkml PRIVATE
        backend/vulkan/vk_device.cpp
        backend/vulkan/vk_allocator.cpp
        backend/vulkan/vk_pipeline.cpp
        backend/vulkan/vk_command.cpp
        backend/vulkan/vulkan_backend.cpp
    )
    target_link_libraries(vkml PRIVATE Vulkan::Vulkan vkml_spirv)
    target_compile_definitions(vkml PUBLIC VKML_HAS_VULKAN=1)
endif()
