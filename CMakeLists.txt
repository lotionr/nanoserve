cmake_minimum_required(VERSION 3.24)
project(nanoserve VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(NANOSERVE_BUILD_TESTS "Build nanoserve tests" ON)

add_library(nanoserve_warnings INTERFACE)
target_compile_options(nanoserve_warnings INTERFACE
    -Wall -Wextra -Wpedantic -Werror
    -Wshadow -Wconversion -Wsign-conversion
)

# --- core library -----------------------------------------------------------
set(NANOSERVE_CORE_SOURCES
    src/core/json.cpp
    src/core/mmap_file.cpp
    src/core/ops.cpp
    src/core/safetensors.cpp
    src/core/threadpool.cpp
    src/model/chat_template.cpp
    src/model/config.cpp
    src/model/qwen2.cpp
    src/model/sampler.cpp
    src/model/tokenizer.cpp
    src/model/unicode.cpp
)

add_library(nanoserve_core STATIC ${NANOSERVE_CORE_SOURCES})
target_include_directories(nanoserve_core PUBLIC src)
target_link_libraries(nanoserve_core PRIVATE nanoserve_warnings)

# Same sources with the SIMD kernels compiled out (F026). Lets this arm64
# machine build and test the scalar fallback that x86 CI will take, instead
# of assuming it still works.
add_library(nanoserve_core_scalar STATIC ${NANOSERVE_CORE_SOURCES})
target_include_directories(nanoserve_core_scalar PUBLIC src)
target_compile_definitions(nanoserve_core_scalar PUBLIC NANO_FORCE_SCALAR=1)
target_link_libraries(nanoserve_core_scalar PRIVATE nanoserve_warnings)

# --- CLI --------------------------------------------------------------------
add_executable(nanoserve src/server/main.cpp)
target_link_libraries(nanoserve PRIVATE nanoserve_core nanoserve_warnings)

# --- microbenchmarks (not tests; see bench/bench_ops.cpp header) -------------
add_executable(bench_ops bench/bench_ops.cpp)
target_link_libraries(bench_ops PRIVATE nanoserve_core nanoserve_warnings)

add_executable(bench_ops_scalar bench/bench_ops.cpp)
target_link_libraries(bench_ops_scalar PRIVATE nanoserve_core_scalar nanoserve_warnings)

# --- tests ------------------------------------------------------------------
# Plain-executable tests (llama.cpp style): exit 0 = pass, 77 = skipped
# (e.g. model weights not downloaded), anything else = failure.
if(NANOSERVE_BUILD_TESTS)
    enable_testing()
    function(nanoserve_test name)
        add_executable(${name} tests/${name}.cpp)
        target_link_libraries(${name} PRIVATE nanoserve_core nanoserve_warnings)
        add_test(NAME ${name} COMMAND ${name})
        set_tests_properties(${name} PROPERTIES
            SKIP_RETURN_CODE 77
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        )
    endfunction()

    nanoserve_test(test_json)
    nanoserve_test(test_ops)
    nanoserve_test(test_safetensors)
    nanoserve_test(test_config)
    nanoserve_test(test_tokenizer)
    nanoserve_test(test_forward)
    nanoserve_test(test_generate)
    nanoserve_test(test_kvcache)
    nanoserve_test(test_sampling)
    nanoserve_test(test_streaming)
    nanoserve_test(test_alloc)

    # The scalar fallback runs the same op goldens (F026): if SIMD and scalar
    # ever disagree beyond tolerance, one of these two tests fails.
    add_executable(test_ops_scalar tests/test_ops.cpp)
    target_link_libraries(test_ops_scalar PRIVATE nanoserve_core_scalar
                                                  nanoserve_warnings)
    add_test(NAME test_ops_scalar COMMAND test_ops_scalar)
    set_tests_properties(test_ops_scalar PROPERTIES
        SKIP_RETURN_CODE 77
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endif()
