set(VKML_BENCH_SOURCES
    bench_main.cpp
)

add_executable(vkml_bench ${VKML_BENCH_SOURCES})
target_link_libraries(vkml_bench PRIVATE vkml vkml_warnings)
