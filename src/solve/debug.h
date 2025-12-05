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

// Rank-aware debug macro: always prints with rank information.
// Use this for tracking issues that occur on non-root ranks.
#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
#define DEBUG_MSG_RANK(str)                                                 \
  do {                                                                      \
    int _dbg_rank = -1;                                                     \
    int _dbg_mpi_init = 0;                                                  \
    MPI_Initialized(&_dbg_mpi_init);                                        \
    if (_dbg_mpi_init) {                                                    \
      MPI_Comm_rank(PETSC_COMM_WORLD, &_dbg_rank);                          \
    }                                                                       \
    std::cout << "[DEBUG RANK " << _dbg_rank << "] " << str << std::endl;   \
    std::cout.flush();                                                      \
  } while (false)

// Check a value for NaN/Inf and report with context if found.
#define DEBUG_CHECK_VALUE(val, context)                                     \
  do {                                                                      \
    if (!std::isfinite(val)) {                                              \
      int _dbg_rank = -1;                                                   \
      int _dbg_mpi_init = 0;                                                \
      MPI_Initialized(&_dbg_mpi_init);                                      \
      if (_dbg_mpi_init) {                                                  \
        MPI_Comm_rank(PETSC_COMM_WORLD, &_dbg_rank);                        \
      }                                                                     \
      std::cerr << "[FP ERROR RANK " << _dbg_rank << "] Non-finite value: " \
                << (val) << " at " << (context) << std::endl;               \
      std::cerr.flush();                                                    \
    }                                                                       \
  } while (false)

// Check an array/vector for NaN/Inf and report with context if found.
#define DEBUG_CHECK_ARRAY(arr, size, context)                               \
  do {                                                                      \
    for (int _i = 0; _i < (size); ++_i) {                                   \
      if (!std::isfinite((arr)[_i])) {                                      \
        int _dbg_rank = -1;                                                 \
        int _dbg_mpi_init = 0;                                              \
        MPI_Initialized(&_dbg_mpi_init);                                    \
        if (_dbg_mpi_init) {                                                \
          MPI_Comm_rank(PETSC_COMM_WORLD, &_dbg_rank);                      \
        }                                                                   \
        std::cerr << "[FP ERROR RANK " << _dbg_rank << "] Non-finite value at index " \
                  << _i << ": " << (arr)[_i] << " in " << (context) << std::endl; \
        std::cerr.flush();                                                  \
        break;                                                              \
      }                                                                     \
    }                                                                       \
  } while (false)
#else
#define DEBUG_MSG_RANK(str) DEBUG_MSG(str)
#define DEBUG_CHECK_VALUE(val, context) do {} while (false)
#define DEBUG_CHECK_ARRAY(arr, size, context) do {} while (false)
#endif

#else
#define DEBUG_MSG(str) \
  do {                 \
  } while (false)
#define DEBUG_MSG_RANK(str) do {} while (false)
#define DEBUG_CHECK_VALUE(val, context) do {} while (false)
#define DEBUG_CHECK_ARRAY(arr, size, context) do {} while (false)
#endif

#endif  // SVZERODSOLVER_HELPERS_DEBUG_HPP_
