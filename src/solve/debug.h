// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the
// University of California, and others. SPDX-License-Identifier: BSD-3-Clause
/**
 * @file Debug.h
 * @brief DEBUG_MSG source file
 */
#ifndef SVZERODSOLVER_HELPERS_DEBUG_HPP_
#define SVZERODSOLVER_HELPERS_DEBUG_HPP_

#include <iostream>
#include <cstdlib>

#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
#include <petscsys.h>
#include <mpi.h>
#endif

/**
 * @brief DEBUG_MSG Macro to print debug messages for debug build
 */
#ifndef NDEBUG

namespace svzero {
namespace detail {
inline bool debug_should_print() {
#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
  PetscBool petsc_init = PETSC_FALSE;
  if (PetscInitialized(&petsc_init) == 0 && petsc_init) {
    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    return rank == 0;
  }
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank == 0;
  }
  if (const char* env_rank = std::getenv("OMPI_COMM_WORLD_RANK")) {
    return std::atoi(env_rank) == 0;
  }
  if (const char* env_rank = std::getenv("PMI_RANK")) {
    return std::atoi(env_rank) == 0;
  }
  if (const char* env_rank = std::getenv("MPI_RANK")) {
    return std::atoi(env_rank) == 0;
  }
#endif

  // Even without PETSc/MPI headers, best effort based on common MPI env vars.
  if (const char* env_rank = std::getenv("OMPI_COMM_WORLD_RANK")) {
    return std::atoi(env_rank) == 0;
  }
  if (const char* env_rank = std::getenv("PMI_RANK")) {
    return std::atoi(env_rank) == 0;
  }
  if (const char* env_rank = std::getenv("MPI_RANK")) {
    return std::atoi(env_rank) == 0;
  }

  return true;
}
}  // namespace detail
}  // namespace svzero

#define DEBUG_MSG(str)                                   \
  do {                                                   \
    if (svzero::detail::debug_should_print()) {          \
      std::cout << "[DEBUG MESSAGE] " << str << std::endl; \
    }                                                    \
  } while (false)
#else
#define DEBUG_MSG(str) \
  do {                 \
  } while (false)
#endif

#endif  // SVZERODSOLVER_HELPERS_DEBUG_HPP_
