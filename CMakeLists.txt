cmake_minimum_required(VERSION 3.14)
project(alpha_expansion VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

# Core maxflow library
add_library(graph_cuts STATIC
    src/bk_maxflow_impl/graph.cpp
    src/bk_maxflow_impl/maxflow.cpp
    src/bk_maxflow_impl/instances.cpp
)
target_include_directories(graph_cuts PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

if(NOT SKBUILD)
  install(TARGETS graph_cuts EXPORT alpha_expansionTargets
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

  install(DIRECTORY src/
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
      FILES_MATCHING
      PATTERN "*.h"
      PATTERN "*.hpp"
      PATTERN "bindings" EXCLUDE)

  install(EXPORT alpha_expansionTargets
      FILE alpha_expansionTargets.cmake
      NAMESPACE alpha_expansion::
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/alpha_expansion)

  configure_package_config_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/cmake/alpha_expansionConfig.cmake.in
      ${CMAKE_CURRENT_BINARY_DIR}/alpha_expansionConfig.cmake
      INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/alpha_expansion)

  write_basic_package_version_file(
      ${CMAKE_CURRENT_BINARY_DIR}/alpha_expansionConfigVersion.cmake
      VERSION ${PROJECT_VERSION}
      COMPATIBILITY SameMajorVersion)

  install(FILES
      ${CMAKE_CURRENT_BINARY_DIR}/alpha_expansionConfig.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/alpha_expansionConfigVersion.cmake
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/alpha_expansion)
endif()

include(FetchContent)

# Google OR-Tools dependency
option(USE_OR_TOOLS "Enable Google OR-Tools dependency" ON)
if(USE_OR_TOOLS)
  find_package(ortools CONFIG REQUIRED)
endif()

# Testing setup (skipped when scikit-build-core drives the build)
if(NOT SKBUILD)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  )
  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()

  add_executable(alpha_expansion_tests tests/tests.cpp tests/test_alpha_expansion.cpp tests/test_energy_model.cpp)
  target_link_libraries(alpha_expansion_tests
    graph_cuts
    GTest::gtest_main
  )

  add_test(NAME alpha_expansion_tests COMMAND alpha_expansion_tests)

  if(USE_OR_TOOLS)
    target_link_libraries(alpha_expansion_tests ortools::ortools)
    target_compile_definitions(alpha_expansion_tests PRIVATE USE_OR_TOOLS)
  endif()
endif()

# Python bindings with pybind11
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
set(PYBIND11_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(alpha_expansion_py src/bindings/bindings.cpp)
target_link_libraries(alpha_expansion_py PRIVATE graph_cuts)

if(USE_OR_TOOLS)
  target_link_libraries(alpha_expansion_py PRIVATE ortools::ortools)
  target_compile_definitions(alpha_expansion_py PRIVATE USE_OR_TOOLS)
endif()

if(SKBUILD)
  install(TARGETS alpha_expansion_py DESTINATION ".")
endif()
