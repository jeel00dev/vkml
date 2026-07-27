# Compiles every .comp to SPIR-V at build time and embeds it as a C++ header.
#
# Deliberately simpler than llama.cpp's vulkan-shaders-gen, which is a 1200-line
# C++ program producing thousands of blobs. Most of that machinery exists to
# cover quantisation types vkml does not have (docs/ARCHITECTURE.md 9 lists
# quantisation as a non-goal). Where variants are needed here they will use
# specialisation constants -- one SPIR-V module, many pipelines -- rather than
# recompiling per variant.

find_program(VKML_GLSLC glslc REQUIRED)

set(VKML_SHADERS
    fill.comp
    unary.comp
    binary.comp
    where.comp
    tri.comp
    cat.comp
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

    # -MD emits a depfile so edits to common.glsl trigger a rebuild of every
    # shader that includes it; without it, a change to the shared preamble is
    # silently ignored until a clean build.
    add_custom_command(
        OUTPUT ${SPV}
        COMMAND ${VKML_GLSLC}
                --target-env=vulkan1.3
                -O
                -I ${CMAKE_CURRENT_SOURCE_DIR}
                -MD -MF ${SPV}.d
                -o ${SPV} ${SRC}
        DEPENDS ${SRC}
        DEPFILE ${SPV}.d
        COMMENT "glslc ${SHADER}"
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
