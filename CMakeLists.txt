cmake_minimum_required(VERSION 3.14)
project(alpha_expansion)

set(CMAKE_CXX_STANDARD 17)

# Core maxflow library
add_library(graph_cuts STATIC
    graph.cpp
    maxflow.cpp
    instances.cpp
)
target_include_directories(graph_cuts PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

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

add_executable(alpha_expansion_tests tests.cpp)
target_link_libraries(alpha_expansion_tests
  graph_cuts
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(alpha_expansion_tests)

# Google OR-Tools dependency
find_package(ortools CONFIG REQUIRED)

target_link_libraries(alpha_expansion_tests
  ortools::ortools
)
