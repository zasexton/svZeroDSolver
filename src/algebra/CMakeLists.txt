# SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
# SPDX-License-Identifier: BSD-3-Clause

# Build the algebra object files.

set(lib svzero_algebra_library)

set(CXXSRCS Integrator.cpp SparseSystem.cpp State.cpp )

set(HDRS Integrator.h SparseSystem.h State.h )

add_library(${lib} OBJECT ${CXXSRCS} )

target_include_directories(${lib} PUBLIC
  ${CMAKE_SOURCE_DIR}/src/algebra
  ${CMAKE_SOURCE_DIR}/src/model
  ${CMAKE_SOURCE_DIR}/src/solve
)

#
# Configure linear solver backend.
# Currently the default backend is Eigen's SparseLU, but this
# interface allows using alternative backends (e.g., Pardiso).
#
set(SVZERODSOLVER_LINEAR_SOLVER "SparseLU"
    CACHE STRING "Linear solver backend for svZeroDSolver (e.g., SparseLU)")
set_property(CACHE SVZERODSOLVER_LINEAR_SOLVER PROPERTY STRINGS
  "SparseLU"
  "PardisoLU"
  "PardisoLDLT"
)
if(SVZERODSOLVER_LINEAR_SOLVER STREQUAL "SparseLU")
  target_compile_definitions(${lib} PUBLIC SVZERODSOLVER_LINEAR_SOLVER_SPARSELU)
elseif(SVZERODSOLVER_LINEAR_SOLVER STREQUAL "PardisoLU")
  target_compile_definitions(${lib} PUBLIC SVZERODSOLVER_LINEAR_SOLVER_PARDISO_LU)
elseif(SVZERODSOLVER_LINEAR_SOLVER STREQUAL "PardisoLDLT")
  target_compile_definitions(${lib} PUBLIC SVZERODSOLVER_LINEAR_SOLVER_PARDISO_LDLT)
else()
  message(FATAL_ERROR
    "Unknown SVZERODSOLVER_LINEAR_SOLVER='${SVZERODSOLVER_LINEAR_SOLVER}'. "
    "Supported values are: SparseLU, PardisoLU, PardisoLDLT.")
endif()

# If a Pardiso-based backend is selected, we need Intel MKL with
# Pardiso support available to Eigen (mkl_pardiso.h and the MKL libs).
if(SVZERODSOLVER_LINEAR_SOLVER STREQUAL "PardisoLU"
   OR SVZERODSOLVER_LINEAR_SOLVER STREQUAL "PardisoLDLT")
  find_package(MKL CONFIG QUIET)

  if(NOT MKL_FOUND OR NOT TARGET MKL::MKL)
    message(FATAL_ERROR
      "SVZERODSOLVER_LINEAR_SOLVER is set to '${SVZERODSOLVER_LINEAR_SOLVER}', "
      "which uses Eigen's Pardiso backend. This requires Intel MKL with Pardiso "
      "support and a CMake package that provides the MKL::MKL target.\n"
      "Either install/configure Intel MKL so that find_package(MKL CONFIG) succeeds, "
      "or select SVZERODSOLVER_LINEAR_SOLVER=SparseLU.")
  endif()

  # Link MKL so that <Eigen/PardisoSupport> can find mkl_pardiso.h and
  # the final binaries are linked to the Pardiso implementation.
  target_link_libraries(${lib} PUBLIC MKL::MKL)
endif()

target_link_libraries( ${lib} Eigen3::Eigen )
