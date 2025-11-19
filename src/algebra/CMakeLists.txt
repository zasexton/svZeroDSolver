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
  "PetscGMRES"
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
elseif(SVZERODSOLVER_LINEAR_SOLVER STREQUAL "PetscGMRES")
  target_compile_definitions(${lib} PUBLIC SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
else()
  message(FATAL_ERROR
    "Unknown SVZERODSOLVER_LINEAR_SOLVER='${SVZERODSOLVER_LINEAR_SOLVER}'. "
    "Supported values are: SparseLU, PardisoLU, PardisoLDLT, "
    "ConjugateGradient, LeastSquaresConjugateGradient, BiCGSTAB, PetscGMRES.")
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

# If a PETSc backend is selected, configure PETSc and expose its preconditioners.
if(SVZERODSOLVER_LINEAR_SOLVER STREQUAL "PetscGMRES")
  # Configure PETSc preconditioner via the same cache variable used for other
  # iterative solvers so that the interface remains consistent.
  set(SVZERODSOLVER_ITERATIVE_PRECONDITIONER "jacobi"
      CACHE STRING
      "PETSc preconditioner for PetscGMRES (none, jacobi, bjacobi, ilu, gamg, hypre_BoomerAMG)")
  set_property(CACHE SVZERODSOLVER_ITERATIVE_PRECONDITIONER PROPERTY STRINGS
    "none"
    "jacobi"
    "bjacobi"
    "ilu"
    "gamg"
    "hypre_BoomerAMG"
  )

  # Expose the chosen PETSc preconditioner type as a string macro to the code.
  target_compile_definitions(${lib} PUBLIC
    SVZERODSOLVER_PETSC_PRECONDITIONER=\"${SVZERODSOLVER_ITERATIVE_PRECONDITIONER}\"
  )

  # Iterative solver control parameters (tolerance and max iterations) are
  # shared with other iterative backends.
  set(SVZERODSOLVER_ITERATIVE_SOLVER_TOLERANCE "1e-6"
      CACHE STRING
      "Relative tolerance for iterative linear solvers (e.g., 1e-6)")
  set(SVZERODSOLVER_ITERATIVE_SOLVER_MAX_ITERS "0"
      CACHE STRING
      "Maximum iterations for iterative linear solvers (0 => size-based heuristic)")

  target_compile_definitions(${lib} PUBLIC
    SVZERODSOLVER_ITERATIVE_SOLVER_TOLERANCE=${SVZERODSOLVER_ITERATIVE_SOLVER_TOLERANCE}
    SVZERODSOLVER_ITERATIVE_SOLVER_MAX_ITERS=${SVZERODSOLVER_ITERATIVE_SOLVER_MAX_ITERS}
  )

  # Try to find an existing PETSc installation first.
  find_package(PETSc QUIET)

  if(NOT PETSC_FOUND)
    # If PETSc is not found, automatically download and build a local PETSc
    # using ExternalProject. This avoids requiring a system-wide installation.
    include(ExternalProject)

    # Prefer a Python interpreter that still provides the deprecated 'xdrlib'
    # module used by PETSc's configure scripts. Fall back to 'python3' if no
    # older interpreter is available.
    find_program(SVZERODSOLVER_PETSC_PYTHON
      NAMES python3.11 python3.10 python3.9 python3.8 python3
    )
    if(NOT SVZERODSOLVER_PETSC_PYTHON)
      message(FATAL_ERROR
        "Could not find a Python interpreter for building PETSc. "
        "Please install a Python 3.x interpreter (e.g., python3.10) and "
        "reconfigure, or install PETSc separately so that find_package(PETSc) succeeds.")
    endif()

    set(SVZERODSOLVER_PETSC_PREFIX
        ${CMAKE_BINARY_DIR}/petsc_ep
        CACHE PATH "Prefix for the internally built PETSc" FORCE)
    set(SVZERODSOLVER_PETSC_INSTALL_DIR
        ${SVZERODSOLVER_PETSC_PREFIX}/install
        CACHE PATH "Install directory for the internally built PETSc" FORCE)

    ExternalProject_Add(petsc_ep
      PREFIX          ${SVZERODSOLVER_PETSC_PREFIX}
      GIT_REPOSITORY  https://gitlab.com/petsc/petsc.git
      GIT_TAG         v3.20.0
      GIT_SHALLOW     TRUE
      GIT_PROGRESS    TRUE
      CONFIGURE_COMMAND
        ${CMAKE_COMMAND} -E chdir <SOURCE_DIR>
        ${SVZERODSOLVER_PETSC_PYTHON} ./configure
          --prefix=${SVZERODSOLVER_PETSC_INSTALL_DIR}
          --with-debugging=0
          --with-shared-libraries=1
          --with-mpi=0
          --download-f2cblaslapack=1
      BUILD_COMMAND
        ${CMAKE_COMMAND} -E chdir <SOURCE_DIR>
        ${CMAKE_MAKE_PROGRAM} PETSC_DIR=<SOURCE_DIR> PETSC_ARCH=arch-linux-c-opt all
      INSTALL_COMMAND
        ${CMAKE_COMMAND} -E chdir <SOURCE_DIR>
        ${CMAKE_MAKE_PROGRAM} PETSC_DIR=<SOURCE_DIR> PETSC_ARCH=arch-linux-c-opt install
      UPDATE_DISCONNECTED TRUE
    )

    # Point the algebra library at the locally built PETSc.
    set(PETSC_INCLUDE_DIRS ${SVZERODSOLVER_PETSC_INSTALL_DIR}/include)
    set(PETSC_LIBRARIES    ${SVZERODSOLVER_PETSC_INSTALL_DIR}/lib/libpetsc${CMAKE_SHARED_LIBRARY_SUFFIX})

    # Ensure PETSc is built before the algebra library.
    add_dependencies(${lib} petsc_ep)
  endif()

  target_include_directories(${lib} PUBLIC ${PETSC_INCLUDE_DIRS})
  target_link_libraries(${lib} PUBLIC ${PETSC_LIBRARIES})
  target_compile_definitions(${lib} PUBLIC SVZERODSOLVER_HAVE_PETSC)

  # Propagate PETSc availability and link information to parent scope so that
  # final executables/modules can link against PETSc as well. Note that link
  # libraries on OBJECT libraries are not used when linking, so consumers must
  # link ${SVZERODSOLVER_PETSC_LIBRARIES} explicitly.
  set(SVZERODSOLVER_HAVE_PETSC TRUE PARENT_SCOPE)
  set(SVZERODSOLVER_PETSC_LIBRARIES "${PETSC_LIBRARIES}" PARENT_SCOPE)
endif()

target_link_libraries(${lib} PUBLIC Eigen3::Eigen)
