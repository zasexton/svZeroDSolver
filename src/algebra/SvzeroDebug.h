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
  SVZERO_PHASE_STEP_SOLVE = 8
};

extern volatile int svzero_current_phase;

#endif  // SVZERODSOLVER_HAVE_PETSC && SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES

#endif  // SVZERODSOLVER_ALGEBRA_SVZERODEBUG_HPP_
