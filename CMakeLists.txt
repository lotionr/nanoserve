cmake_minimum_required(VERSION 3.24)
project(nanoserve VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(NANOSERVE_BUILD_TESTS "Build nanoserve tests" ON)
option(NANOSERVE_BUILD_PYTHON "Build the pybind11 Python module (F028)" OFF)

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
    src/core/quant.cpp
    src/core/safetensors.cpp
    src/core/threadpool.cpp
    src/model/batch.cpp
    src/model/chat_template.cpp
    src/model/config.cpp
    src/model/paged_kv.cpp
    src/model/qwen2.cpp
    src/model/sampler.cpp
    src/model/tokenizer.cpp
    src/model/unicode.cpp
)

# Metal backend (F033): Objective-C++ on Apple (the MSL kernels are embedded
# in metal.mm and compiled at runtime — no .metallib build step), a stub that
# reports "unavailable" everywhere else. Both nanoserve_core variants get it:
# the GPU path is orthogonal to the CPU SIMD-vs-scalar split.
if(APPLE)
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 20)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
    list(APPEND NANOSERVE_CORE_SOURCES src/core/metal.mm)
    set_source_files_properties(src/core/metal.mm PROPERTIES
        COMPILE_OPTIONS "-fobjc-arc")
else()
    list(APPEND NANOSERVE_CORE_SOURCES src/core/metal_stub.cpp)
endif()

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

if(APPLE)
    # PUBLIC: every executable that links a core library needs the
    # frameworks metal.mm calls into.
    target_link_libraries(nanoserve_core PUBLIC
        "-framework Metal" "-framework Foundation")
    target_link_libraries(nanoserve_core_scalar PUBLIC
        "-framework Metal" "-framework Foundation")
endif()

# --- CLI + HTTP server ------------------------------------------------------
add_executable(nanoserve
    src/server/main.cpp
    src/server/http.cpp
    src/server/serve.cpp
)
target_link_libraries(nanoserve PRIVATE nanoserve_core nanoserve_warnings)

# --- Python bindings (F028) ---------------------------------------------------
# Built only by pip/scikit-build-core (pyproject.toml turns this ON); the
# plain CMake build stays Python-free. The bindings file compiles WITHOUT
# nanoserve_warnings: pybind11's headers are not clean under
# -Wconversion/-Wsign-conversion, and -Werror would make their warnings
# fatal. The core library itself is still built with the full set.
if(NANOSERVE_BUILD_PYTHON)
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_nanoserve src/python/bindings.cpp)
    target_link_libraries(_nanoserve PRIVATE nanoserve_core)
    install(TARGETS _nanoserve LIBRARY DESTINATION nanoserve)
endif()

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
    nanoserve_test(test_paged_alloc)  # model-free: runs everywhere, incl. CI
    nanoserve_test(test_batch)   # F036: batched == solo, bit for bit; skips w/o model
    nanoserve_test(test_sampling)
    nanoserve_test(test_streaming)
    nanoserve_test(test_alloc)
    nanoserve_test(test_quant)
    nanoserve_test(test_metal)   # skips (77) without a Metal device

    # test_server spawns the real `nanoserve` binary (path passed as argv[1])
    # and speaks HTTP to it over a socket; skips without model weights.
    add_executable(test_server tests/test_server.cpp)
    target_link_libraries(test_server PRIVATE nanoserve_core nanoserve_warnings)
    add_test(NAME test_server COMMAND test_server $<TARGET_FILE:nanoserve>)
    set_tests_properties(test_server PROPERTIES
        SKIP_RETURN_CODE 77
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

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
