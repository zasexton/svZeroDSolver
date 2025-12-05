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
#if __has_include(<mpi.h>)
#include <mpi.h>
#endif
#endif

/**
 * @brief DEBUG_MSG Macro to print debug messages for debug build
 */
#ifndef NDEBUG

namespace svzero {
namespace detail {
inline bool debug_should_print() {
#if defined(SVZERODSOLVER_DEBUG_ALL_RANKS)
  // Compile-time option: when SVZERODSOLVER_DEBUG_ALL_RANKS is defined via
  // CMake, emit DEBUG_MSG output from every MPI rank.
  return true;
#endif

  // Optional override: when SVZERO_DEBUG_ALL_RANKS is set to a nonzero
  // value, emit DEBUG_MSG output from every MPI rank instead of only the
  // root rank. This is useful for diagnosing PETSc/MPI issues on
  // non-root ranks.
  if (const char* all = std::getenv("SVZERO_DEBUG_ALL_RANKS")) {
    if (std::atoi(all) != 0) {
      return true;
    }
  }

#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
  PetscBool petsc_init = PETSC_FALSE;
  if (PetscInitialized(&petsc_init) == 0 && petsc_init) {
    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    return rank == 0;
  }
#endif

  // Best effort based on common MPI/launcher env vars (works before MPI_Init).
  if (const char* env_rank = std::getenv("OMPI_COMM_WORLD_RANK")) {
    return std::atoi(env_rank) == 0;
  }
  if (const char* env_rank = std::getenv("PMI_RANK")) {
    return std::atoi(env_rank) == 0;
  }
  if (const char* env_rank = std::getenv("MPI_RANK")) {
    return std::atoi(env_rank) == 0;
  }
  if (const char* env_rank = std::getenv("PMIX_RANK")) {
    return std::atoi(env_rank) == 0;
  }
  if (const char* env_rank = std::getenv("SLURM_PROCID")) {
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
