cmake_minimum_required(VERSION 3.14)
project(alpha_expansion)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

# Core maxflow library
add_library(graph_cuts STATIC
    src/bk_maxflow_impl/graph.cpp
    src/bk_maxflow_impl/maxflow.cpp
    src/bk_maxflow_impl/instances.cpp
)
target_include_directories(graph_cuts PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Testing setup
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(alpha_expansion_tests tests/tests.cpp tests/test_alpha_expansion.cpp)
target_link_libraries(alpha_expansion_tests
  graph_cuts
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(alpha_expansion_tests)

# Google OR-Tools dependency
option(USE_OR_TOOLS "Enable Google OR-Tools dependency" ON)

if(USE_OR_TOOLS)
  find_package(ortools CONFIG REQUIRED)

  target_link_libraries(alpha_expansion_tests
    ortools::ortools
  )
  target_compile_definitions(alpha_expansion_tests PRIVATE USE_OR_TOOLS)
endif()

# Python bindings with pybind11
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(alpha_expansion_py src/bindings/bindings.cpp)
target_link_libraries(alpha_expansion_py PRIVATE graph_cuts)

if(USE_OR_TOOLS)
  target_link_libraries(alpha_expansion_py PRIVATE ortools::ortools)
  target_compile_definitions(alpha_expansion_py PRIVATE USE_OR_TOOLS)
endif()
