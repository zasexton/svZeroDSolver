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
  "ConjugateGradient"
  "LeastSquaresConjugateGradient"
  "BiCGSTAB"
)
if(SVZERODSOLVER_LINEAR_SOLVER STREQUAL "SparseLU")
  target_compile_definitions(${lib} PUBLIC SVZERODSOLVER_LINEAR_SOLVER_SPARSELU)
elseif(SVZERODSOLVER_LINEAR_SOLVER STREQUAL "PardisoLU")
  target_compile_definitions(${lib} PUBLIC SVZERODSOLVER_LINEAR_SOLVER_PARDISO_LU)
elseif(SVZERODSOLVER_LINEAR_SOLVER STREQUAL "PardisoLDLT")
  target_compile_definitions(${lib} PUBLIC SVZERODSOLVER_LINEAR_SOLVER_PARDISO_LDLT)
elseif(SVZERODSOLVER_LINEAR_SOLVER STREQUAL "ConjugateGradient")
  target_compile_definitions(${lib} PUBLIC SVZERODSOLVER_LINEAR_SOLVER_CONJUGATE_GRADIENT)
elseif(SVZERODSOLVER_LINEAR_SOLVER STREQUAL "LeastSquaresConjugateGradient")
  target_compile_definitions(${lib} PUBLIC SVZERODSOLVER_LINEAR_SOLVER_LEAST_SQUARES_CONJUGATE_GRADIENT)
elseif(SVZERODSOLVER_LINEAR_SOLVER STREQUAL "BiCGSTAB")
  target_compile_definitions(${lib} PUBLIC SVZERODSOLVER_LINEAR_SOLVER_BICGSTAB)
else()
  message(FATAL_ERROR
    "Unknown SVZERODSOLVER_LINEAR_SOLVER='${SVZERODSOLVER_LINEAR_SOLVER}'. "
    "Supported values are: SparseLU, PardisoLU, PardisoLDLT, "
    "ConjugateGradient, LeastSquaresConjugateGradient, BiCGSTAB.")
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

# If an iterative backend is selected, expose preconditioner configuration.
if(SVZERODSOLVER_LINEAR_SOLVER STREQUAL "ConjugateGradient"
   OR SVZERODSOLVER_LINEAR_SOLVER STREQUAL "LeastSquaresConjugateGradient"
   OR SVZERODSOLVER_LINEAR_SOLVER STREQUAL "BiCGSTAB")
  set(SVZERODSOLVER_ITERATIVE_PRECONDITIONER "Diagonal"
      CACHE STRING
      "Preconditioner for iterative linear solvers (Identity, Diagonal)")
  set_property(CACHE SVZERODSOLVER_ITERATIVE_PRECONDITIONER PROPERTY STRINGS
    "Identity"
    "Diagonal"
  )

  if(SVZERODSOLVER_ITERATIVE_PRECONDITIONER STREQUAL "Identity")
    target_compile_definitions(${lib} PUBLIC SVZERODSOLVER_PRECONDITIONER_IDENTITY)
  elseif(SVZERODSOLVER_ITERATIVE_PRECONDITIONER STREQUAL "Diagonal")
    target_compile_definitions(${lib} PUBLIC SVZERODSOLVER_PRECONDITIONER_DIAGONAL)
  else()
    message(FATAL_ERROR
      "Unknown SVZERODSOLVER_ITERATIVE_PRECONDITIONER='${SVZERODSOLVER_ITERATIVE_PRECONDITIONER}'. "
      "Supported values are: Identity, Diagonal.")
  endif()

  # Iterative solver control parameters (tolerance and max iterations).
  # These can be tuned for specific hydraulic circuits to reduce fallbacks.
  set(SVZERODSOLVER_ITERATIVE_SOLVER_TOLERANCE "1e-6"
      CACHE STRING
      "Relative tolerance for iterative linear solvers (e.g., 1e-6)")
  set(SVZERODSOLVER_ITERATIVE_SOLVER_MAX_ITERS "0"
      CACHE STRING
      "Maximum iterations for iterative linear solvers (0 => use system size)")

  target_compile_definitions(${lib} PUBLIC
    SVZERODSOLVER_ITERATIVE_SOLVER_TOLERANCE=${SVZERODSOLVER_ITERATIVE_SOLVER_TOLERANCE}
    SVZERODSOLVER_ITERATIVE_SOLVER_MAX_ITERS=${SVZERODSOLVER_ITERATIVE_SOLVER_MAX_ITERS}
  )
endif()

target_link_libraries( ${lib} Eigen3::Eigen )
