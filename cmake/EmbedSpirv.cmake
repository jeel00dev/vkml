# Turns a compiled SPIR-V module into a C++ header holding a uint32_t array.
#
# Shaders are embedded rather than loaded from disk at runtime for two reasons:
# the library then has no data-file dependency to install or locate, and a
# shader can never be out of sync with the binary that dispatches it.
#
# Invoked in script mode: cmake -DSPV=... -DOUT=... -DSYMBOL=... -P EmbedSpirv.cmake

file(READ "${SPV}" HEX_CONTENT HEX)
string(LENGTH "${HEX_CONTENT}" HEX_LEN)
math(EXPR BYTE_COUNT "${HEX_LEN} / 2")

if(NOT BYTE_COUNT GREATER 0)
    message(FATAL_ERROR "EmbedSpirv: '${SPV}' is empty")
endif()

math(EXPR REMAINDER "${BYTE_COUNT} % 4")
if(NOT REMAINDER EQUAL 0)
    message(FATAL_ERROR "EmbedSpirv: '${SPV}' is ${BYTE_COUNT} bytes, not a multiple of 4")
endif()

math(EXPR WORD_COUNT "${BYTE_COUNT} / 4")

set(BODY "")
set(COL 0)
math(EXPR LAST "${WORD_COUNT} - 1")

foreach(i RANGE ${LAST})
    math(EXPR OFFSET "${i} * 8")
    string(SUBSTRING "${HEX_CONTENT}" ${OFFSET} 8 WORD_HEX)

    # SPIR-V is little-endian on disk; reverse the byte pairs so the literal
    # matches the word the driver expects.
    string(SUBSTRING "${WORD_HEX}" 0 2 B0)
    string(SUBSTRING "${WORD_HEX}" 2 2 B1)
    string(SUBSTRING "${WORD_HEX}" 4 2 B2)
    string(SUBSTRING "${WORD_HEX}" 6 2 B3)

    string(APPEND BODY "0x${B3}${B2}${B1}${B0},")
    math(EXPR COL "${COL} + 1")
    if(COL EQUAL 8)
        string(APPEND BODY "\n    ")
        set(COL 0)
    else()
        string(APPEND BODY " ")
    endif()
endforeach()

file(WRITE "${OUT}"
"// Generated from ${SPV}. Do not edit.
#pragma once

#include <cstdint>

namespace vkml::spv {

inline constexpr uint32_t ${SYMBOL}[] = {
    ${BODY}
};

inline constexpr size_t ${SYMBOL}_size = sizeof(${SYMBOL});

}  // namespace vkml::spv
")
