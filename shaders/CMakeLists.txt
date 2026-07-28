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

if(VKML_GLSLC)
    message(STATUS "  shader compiler .. ${VKML_GLSLC}")
elseif(VKML_GLSLANG)
    message(STATUS "  shader compiler .. ${VKML_GLSLANG} (glslc not found)")
else()
    message(FATAL_ERROR
        "No GLSL compiler found. vkml needs one of:\n"
        "  glslc            (Debian/Ubuntu: glslc, Arch: shaderc, or the Vulkan SDK)\n"
        "  glslangValidator (Debian/Ubuntu: glslang-tools, Arch: glslang)")
endif()

set(VKML_SHADERS
    fill.comp
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
            --target-env vulkan1.3 -Os
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
