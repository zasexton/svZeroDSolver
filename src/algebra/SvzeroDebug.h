// SPDX-FileCopyrightText: Copyright (c) Stanford University,
// The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause
/**
 * @file SvzeroDebug.h
 * @brief Shared debug instrumentation for tracking where floating-point
 *        exceptions occur in large PETSc GMRES runs.
 *
 * This header is only active when building with the PETSc GMRES backend.
 */

#ifndef SVZERODSOLVER_ALGEBRA_SVZERODEBUG_HPP_
#define SVZERODSOLVER_ALGEBRA_SVZERODEBUG_HPP_

#include <cstddef>

#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)

// Global index of the block currently being processed. This is updated on
// the root rank before calling each block's update_* method.
extern volatile std::size_t svzero_current_block_index;

// Approximate simulation time associated with the current operation. This is
// updated by the Integrator before major phases of the time-stepping loop.
extern volatile double svzero_current_time;

// Global phase indicator for where in the assembly / solve the code is
// currently operating when a floating-point exception occurs.
enum : int {
  SVZERO_PHASE_NONE = 0,
  SVZERO_PHASE_RESERVE_CONSTANT = 1,
  SVZERO_PHASE_RESERVE_TIME = 2,
  SVZERO_PHASE_RESERVE_SOLUTION = 3,
  SVZERO_PHASE_STEP_TIME = 4,
  SVZERO_PHASE_STEP_SOLUTION = 5,
  // Phases within a single nonlinear time step:
  // - residual assembly (RHS)
  // - Jacobian assembly
  // - linear solve
  SVZERO_PHASE_STEP_RESIDUAL = 6,
  SVZERO_PHASE_STEP_JACOBIAN = 7,
  SVZERO_PHASE_STEP_SOLVE = 8,
  // MPI communication phases (for debugging non-root rank issues)
  SVZERO_PHASE_MPI_BCAST_RESIDUAL = 9,
  SVZERO_PHASE_MPI_BCAST_STATE_Y = 10,
  SVZERO_PHASE_MPI_BCAST_STATE_YDOT = 11,
  SVZERO_PHASE_MPI_SCATTER_NNZ = 12,
  SVZERO_PHASE_MPI_VEC_ASSEMBLY = 13,
  SVZERO_PHASE_MPI_MAT_ASSEMBLY = 14,
  SVZERO_PHASE_MPI_VEC_SCATTER = 15,
  SVZERO_PHASE_PETSC_KSP_SOLVE = 16,
  SVZERO_PHASE_PETSC_INIT = 17
};

// Helper function to convert phase to string for debugging output.
inline const char* svzero_phase_to_string(int phase) {
  switch (phase) {
    case SVZERO_PHASE_NONE: return "none";
    case SVZERO_PHASE_RESERVE_CONSTANT: return "reserve_update_constant";
    case SVZERO_PHASE_RESERVE_TIME: return "reserve_update_time";
    case SVZERO_PHASE_RESERVE_SOLUTION: return "reserve_update_solution";
    case SVZERO_PHASE_STEP_TIME: return "step_update_time";
    case SVZERO_PHASE_STEP_SOLUTION: return "step_update_solution";
    case SVZERO_PHASE_STEP_RESIDUAL: return "step_residual_assembly";
    case SVZERO_PHASE_STEP_JACOBIAN: return "step_jacobian_assembly";
    case SVZERO_PHASE_STEP_SOLVE: return "step_linear_solve";
    case SVZERO_PHASE_MPI_BCAST_RESIDUAL: return "mpi_bcast_residual";
    case SVZERO_PHASE_MPI_BCAST_STATE_Y: return "mpi_bcast_state_y";
    case SVZERO_PHASE_MPI_BCAST_STATE_YDOT: return "mpi_bcast_state_ydot";
    case SVZERO_PHASE_MPI_SCATTER_NNZ: return "mpi_scatter_nnz";
    case SVZERO_PHASE_MPI_VEC_ASSEMBLY: return "mpi_vec_assembly";
    case SVZERO_PHASE_MPI_MAT_ASSEMBLY: return "mpi_mat_assembly";
    case SVZERO_PHASE_MPI_VEC_SCATTER: return "mpi_vec_scatter";
    case SVZERO_PHASE_PETSC_KSP_SOLVE: return "petsc_ksp_solve";
    case SVZERO_PHASE_PETSC_INIT: return "petsc_init";
    default: return "unknown";
  }
}

extern volatile int svzero_current_phase;

#endif  // SVZERODSOLVER_HAVE_PETSC && SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES

#endif  // SVZERODSOLVER_ALGEBRA_SVZERODEBUG_HPP_
