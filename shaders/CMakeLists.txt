# Compiles every .comp to SPIR-V at build time and embeds it as a C++ header.
#
# Deliberately simpler than llama.cpp's vulkan-shaders-gen, which is a 1200-line
# C++ program producing thousands of blobs. Most of that machinery exists to
# cover quantisation types vkml does not have (docs/ARCHITECTURE.md 9 lists
# quantisation as a non-goal). Where variants are needed here they will use
# specialisation constants -- one SPIR-V module, many pipelines -- rather than
# recompiling per variant.

# Either shader compiler will do, and requiring one specifically was a real
# barrier: glslc comes from shaderc, which is a SEPARATE package from the Vulkan
# SDK on most distributions, while glslangValidator ships with glslang and is
# what find_package(Vulkan) locates. A GitHub runner with the Vulkan SDK
# installed has the second and not the first, so the build failed there while
# the README claimed the SDK was enough.
#
# glslc is preferred only because -O optimises harder than glslangValidator's
# -Os; both emit valid SPIR-V and the suite passes on either.
find_program(VKML_GLSLC glslc)
find_program(VKML_GLSLANG glslangValidator)

# Fall back to whatever find_package(Vulkan) already located.
#
# find_program searches PATH, and the LunarG Windows installer does not put
# C:\VulkanSDK\<ver>\Bin on it. So a configure could print
# "found components: glslc glslangValidator" and then fail two lines later with
# "No GLSL compiler found" -- from the same run (issue #5). find_package had the
# absolute paths all along and nothing in the repo referenced them.
#
# PATH still wins, so nothing changes on a machine where this already worked;
# this only adds a route where there was none. Set as normal variables rather
# than into the cache, so a later reconfigure re-derives them from whichever SDK
# find_package finds then.
if(NOT VKML_GLSLC AND Vulkan_GLSLC_EXECUTABLE)
    set(VKML_GLSLC "${Vulkan_GLSLC_EXECUTABLE}")
    message(STATUS "  glslc not on PATH; using the one find_package(Vulkan) found")
endif()
if(NOT VKML_GLSLANG AND Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
    set(VKML_GLSLANG "${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}")
    message(STATUS "  glslangValidator not on PATH; using the one find_package(Vulkan) found")
endif()

if(VKML_GLSLC)
    message(STATUS "  shader compiler .. ${VKML_GLSLC}")
elseif(VKML_GLSLANG)
    message(STATUS "  shader compiler .. ${VKML_GLSLANG} (glslc not found)")

    # glslangValidator accepts -Os only when it was built against SPIRV-Tools.
    # Distribution packages generally are; a minimal source build is not, and
    # then EVERY shader fails with "-Os not available; optimizer not linked" --
    # which is how the manylinux wheel build failed, since that image ships no
    # GLSL compiler and one has to be built from source.
    #
    # Probed by compiling rather than by reading --help, because the help text
    # lists the flag either way. Unoptimised SPIR-V is valid and correct; the
    # driver optimises regardless, so losing -Os costs module size, not
    # behaviour.
    set(_opt_probe ${CMAKE_CURRENT_BINARY_DIR}/opt_probe.comp)
    file(WRITE ${_opt_probe} "#version 450\nvoid main() {}\n")
    execute_process(
        COMMAND ${VKML_GLSLANG} --target-env vulkan1.3 -Os
                -o ${CMAKE_CURRENT_BINARY_DIR}/opt_probe.spv ${_opt_probe}
        RESULT_VARIABLE _opt_result OUTPUT_QUIET ERROR_QUIET)
    if(_opt_result EQUAL 0)
        set(VKML_GLSLANG_OPT -Os)
        message(STATUS "  spirv optimiser .. available (-Os)")
    else()
        set(VKML_GLSLANG_OPT "")
        message(STATUS "  spirv optimiser .. not linked in; compiling unoptimised")
    endif()
else()
    message(FATAL_ERROR
        "No GLSL compiler found. vkml needs one of:\n"
        "  glslc            (Debian/Ubuntu: glslc, Arch: shaderc, or the Vulkan SDK)\n"
        "  glslangValidator (Debian/Ubuntu: glslang-tools, Arch: glslang)")
endif()

set(VKML_SHADERS
    fill.comp
    scaled_add.comp
    unary.comp
    binary.comp
    where.comp
    tri.comp
    cat.comp
    index_select.comp
    scatter_add.comp
    im2col.comp
    col2im.comp
    max_pool2d.comp
    rand.comp
    slice_backward.comp
    cast.comp
    reduce.comp
    softmax.comp
    gemm_naive.comp
    gemm_tiled.comp
    gemm_reg.comp
    gemm_db.comp
    gemm_split_k_reduce.comp
    gemv.comp
    probe_private_array.comp
    probe_mlp.comp
)

set(SPV_HEADER_DIR ${CMAKE_BINARY_DIR}/generated/vkml/spv)
file(MAKE_DIRECTORY ${SPV_HEADER_DIR})

set(GENERATED_HEADERS "")

foreach(SHADER ${VKML_SHADERS})
    get_filename_component(NAME_WE ${SHADER} NAME_WE)
    set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER})
    set(SPV ${CMAKE_CURRENT_BINARY_DIR}/${NAME_WE}.spv)
    set(HDR ${SPV_HEADER_DIR}/${NAME_WE}.h)

    # Both compilers emit a depfile, so an edit to common.glsl rebuilds every
    # shader that includes it; without one, a change to the shared preamble is
    # silently ignored until a clean build.
    #
    # The flag SPELLINGS differ and are not interchangeable: glslangValidator
    # wants the include path attached to the flag (-Idir, not -I dir) and takes
    # --target-env as two words.
    if(VKML_GLSLC)
        set(VKML_COMPILE_SHADER ${VKML_GLSLC}
            --target-env=vulkan1.3 -O
            -I ${CMAKE_CURRENT_SOURCE_DIR}
            -MD -MF ${SPV}.d
            -o ${SPV} ${SRC})
    else()
        set(VKML_COMPILE_SHADER ${VKML_GLSLANG}
            --target-env vulkan1.3 ${VKML_GLSLANG_OPT}
            -I${CMAKE_CURRENT_SOURCE_DIR}
            --depfile ${SPV}.d
            -o ${SPV} ${SRC})
    endif()

    add_custom_command(
        OUTPUT ${SPV}
        COMMAND ${VKML_COMPILE_SHADER}
        DEPENDS ${SRC}
        DEPFILE ${SPV}.d
        COMMENT "compile ${SHADER}"
        VERBATIM)

    add_custom_command(
        OUTPUT ${HDR}
        COMMAND ${CMAKE_COMMAND}
                -DSPV=${SPV} -DOUT=${HDR} -DSYMBOL=${NAME_WE}
                -P ${CMAKE_SOURCE_DIR}/cmake/EmbedSpirv.cmake
        DEPENDS ${SPV} ${CMAKE_SOURCE_DIR}/cmake/EmbedSpirv.cmake
        COMMENT "embed ${NAME_WE}.spv"
        VERBATIM)

    list(APPEND GENERATED_HEADERS ${HDR})
endforeach()

add_custom_target(vkml_shaders DEPENDS ${GENERATED_HEADERS})

add_library(vkml_spirv INTERFACE)
add_dependencies(vkml_spirv vkml_shaders)
target_include_directories(vkml_spirv INTERFACE ${CMAKE_BINARY_DIR}/generated)
