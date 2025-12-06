// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the
// University of California, and others. SPDX-License-Identifier: BSD-3-Clause

#include "Integrator.h"

#include <cmath>
#include <stdexcept>

#include "debug.h"

#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
#include <petscsys.h>
#if __has_include(<mpi.h>)
#include <mpi.h>
#endif
#include "SvzeroDebug.h"
namespace {
inline bool integrator_is_root_rank() {
#ifdef MPI_VERSION
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) {
    return true;
  }
  int rank = 0;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  return rank == 0;
#else
  return true;
#endif
}
}  // namespace
#endif  // SVZERODSOLVER_HAVE_PETSC && SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES

Integrator::Integrator(Model* model, double time_step_size, double rho,
                       double atol, int max_iter)
    : Integrator(model,
                 model ? model->dofhandler.size() : 0,
                 time_step_size,
                 rho,
                 atol,
                 max_iter) {}

Integrator::Integrator(Model* model, int system_size, double time_step_size,
                       double rho, double atol, int max_iter)
    : system(SparseSystem(system_size)), model(model) {
#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
  {
    int rank = 0;
    int mpi_init = 0;
    MPI_Initialized(&mpi_init);
    if (mpi_init) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    std::cerr << "[RANK " << rank << "] Integrator::Integrator - ENTER, system_size="
              << system_size << ", model=" << (model ? "valid" : "null") << std::endl;
    std::cerr.flush();
  }
#endif
  DEBUG_MSG("Integrator::Integrator - begin, system_size=" << system_size);
  const double denom_rho = 1.0 + rho;
  if (!std::isfinite(denom_rho) || std::fabs(denom_rho) <= 1e-12) {
    throw std::runtime_error(
        "Integrator::Integrator - invalid rho (1+rho too close to zero)");
  }
  alpha_m = 0.5 * (3.0 - rho) / denom_rho;
  alpha_f = 1.0 / denom_rho;
  gamma = 0.5 + alpha_m - alpha_f;
  if (!std::isfinite(gamma) || std::fabs(gamma) <= 1e-12) {
    throw std::runtime_error(
        "Integrator::Integrator - invalid generalized-alpha gamma (zero or non-finite)");
  }
  ydot_init_coeff = 1.0 - 1.0 / gamma;

  y_coeff = gamma * time_step_size;
  y_coeff_jacobian = alpha_f * y_coeff;

  size = system_size;
  this->time_step_size = time_step_size;
  this->atol = atol;
  this->max_iter = max_iter;

  y_af = Eigen::Matrix<double, Eigen::Dynamic, 1>(size);
  ydot_am = Eigen::Matrix<double, Eigen::Dynamic, 1>(size);

  // Make some memory reservations
  DEBUG_MSG("Integrator::Integrator - calling SparseSystem::reserve");
  system.reserve(model);
  DEBUG_MSG("Integrator::Integrator - end");
}

// Must declare default constructord and dedtructor
// because of Eigen.
Integrator::Integrator() {}
Integrator::~Integrator() {}

void Integrator::clean() {
  // Cannot be in destructor because dynamically allocated pointers will be lost
  // when objects are assigned from temporary objects.
  system.clean();
}

void Integrator::update_params(double time_step_size) {
  this->time_step_size = time_step_size;
  y_coeff = gamma * time_step_size;
  y_coeff_jacobian = alpha_f * y_coeff;
#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
  const bool is_root = integrator_is_root_rank();
  if (!is_root || model == nullptr) {
    return;
  }
#endif
  if (model) {
    // Rebuild constant/time-dependent contributions at t=0.
#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
    svzero_current_phase = SVZERO_PHASE_RESERVE_CONSTANT;
    svzero_current_block_index = static_cast<std::size_t>(-1);
#endif
    model->update_constant(system);
#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
    svzero_current_phase = SVZERO_PHASE_RESERVE_TIME;
    svzero_current_block_index = static_cast<std::size_t>(-1);
#endif
    model->update_time(system, 0.0);
  }
}

State Integrator::step(const State& old_state, double time) {
  // Predictor: Constant y, consistent ydot
  State new_state = State::Zero(size);
#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
  const bool is_root = integrator_is_root_rank();
  // Track the current physical time for debug reporting on all ranks.
  svzero_current_time = time;
  int rank = 0;
#if defined(MPI_VERSION)
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized) {
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  }
#endif
  DEBUG_MSG("Integrator::step - rank=" << rank
                                       << ", is_root=" << (is_root ? "true" : "false")
                                       << ", time=" << time);
#else
  const bool is_root = true;
#endif

  if (is_root) {
    DEBUG_MSG("Integrator::step - begin, time=" << time);
    new_state.ydot += old_state.ydot * ydot_init_coeff;
    new_state.y += old_state.y;
  }

  // Determine new time (evaluate terms at generalized mid-point)
  double new_time = time + alpha_f * time_step_size;

  // Evaluate time-dependent element contributions in system
  if (is_root) {
    DEBUG_MSG("Integrator::step - calling Model::update_time at t=" << new_time);
#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
    svzero_current_phase = SVZERO_PHASE_STEP_TIME;
    svzero_current_block_index = static_cast<std::size_t>(-1);
#endif
    model->update_time(system, new_time);
  }

  // Count total number of step calls
  n_iter++;

  // Non-linear Newton-Raphson iterations
  for (size_t i = 0; i < max_iter; i++) {
    // Initiator: Evaluate the iterates at the intermediate time levels
    ydot_am.setZero();
    y_af.setZero();
    if (is_root) {
      ydot_am += old_state.ydot +
                 (new_state.ydot - old_state.ydot) * alpha_m;
      y_af += old_state.y + (new_state.y - old_state.y) * alpha_f;

      // Update solution-dependent element contributions
      DEBUG_MSG("Integrator::step - calling Model::update_solution, iter=" << i);
#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
      svzero_current_phase = SVZERO_PHASE_STEP_SOLUTION;
      svzero_current_block_index = static_cast<std::size_t>(-1);
#endif
      model->update_solution(system, y_af, ydot_am);
    }

    // Evaluate residual (collective across all ranks when using PETSc).
    DEBUG_MSG("Integrator::step - calling SparseSystem::update_residual");
    system.update_residual(y_af, ydot_am);

    // Check termination criterium (based on residual on the root rank).
    double max_residual = 0.0;
    if (is_root) {
      max_residual = system.residual.cwiseAbs().maxCoeff();
      DEBUG_CHECK_VALUE(max_residual, "max_residual before Bcast");
    }
#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES) && defined(MPI_VERSION)
    DEBUG_MSG_RANK("Integrator::step - before MPI_Bcast(max_residual), iter=" << i
                   << ", rank=" << rank << ", max_residual=" << max_residual);
    svzero_current_phase = SVZERO_PHASE_MPI_BCAST_RESIDUAL;
    MPI_Bcast(&max_residual, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    svzero_current_phase = SVZERO_PHASE_NONE;
    DEBUG_MSG_RANK("Integrator::step - after MPI_Bcast(max_residual), iter=" << i
                   << ", rank=" << rank << ", max_residual=" << max_residual);
    DEBUG_CHECK_VALUE(max_residual, "max_residual after Bcast");
#endif
    if (max_residual < atol) {
      DEBUG_MSG("Integrator::step - residual below atol, iter=" << i
                << ", max_residual=" << max_residual);
      break;
    }

    // Abort if maximum number of non-linear iterations is reached
    else if (i == max_iter - 1) {
      throw std::runtime_error(
          "Maximum number of non-linear iterations reached at time " +
          std::to_string(time));
    }

    // Evaluate Jacobian
    DEBUG_MSG("Integrator::step - calling SparseSystem::update_jacobian");
    system.update_jacobian(alpha_m, y_coeff_jacobian);

    // Solve system for increment in ydot
    DEBUG_MSG("Integrator::step - calling SparseSystem::solve");
    system.solve();

    // Perform post-solve actions on blocks
    if (is_root) {
      model->post_solve(new_state.y);

      // Update the solution on the root rank.
      new_state.ydot += system.dydot;
      new_state.y += system.dydot * y_coeff;
      // Count total number of nonlinear iterations
      n_nonlin_iter++;
    }
  }

#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES) && defined(MPI_VERSION)
  // Broadcast the updated state from the root rank so that all ranks see a
  // consistent state, while only the root performed the updates.

  // Check state for NaN/Inf before broadcast on root
  if (is_root) {
    for (Eigen::Index i = 0; i < new_state.y.size(); ++i) {
      if (!std::isfinite(new_state.y[i])) {
        std::cerr << "[FP ERROR RANK 0] Non-finite new_state.y[" << i << "]="
                  << new_state.y[i] << " before broadcast, time=" << time << std::endl;
        break;
      }
    }
    for (Eigen::Index i = 0; i < new_state.ydot.size(); ++i) {
      if (!std::isfinite(new_state.ydot[i])) {
        std::cerr << "[FP ERROR RANK 0] Non-finite new_state.ydot[" << i << "]="
                  << new_state.ydot[i] << " before broadcast, time=" << time << std::endl;
        break;
      }
    }
  }

  DEBUG_MSG_RANK("Integrator::step - before MPI_Bcast(state.y), rank=" << rank
                 << ", y.size=" << new_state.y.size() << ", time=" << time);
  svzero_current_phase = SVZERO_PHASE_MPI_BCAST_STATE_Y;
  MPI_Bcast(new_state.y.data(),
            static_cast<int>(new_state.y.size()),
            MPI_DOUBLE,
            0,
            PETSC_COMM_WORLD);
  svzero_current_phase = SVZERO_PHASE_NONE;
  DEBUG_MSG_RANK("Integrator::step - after MPI_Bcast(state.y), rank=" << rank);

  // Check for NaN/Inf after receiving broadcast on non-root ranks
  {
    bool has_nonfinite_y = false;
    Eigen::Index bad_idx = -1;
    for (Eigen::Index i = 0; i < new_state.y.size(); ++i) {
      if (!std::isfinite(new_state.y[i])) {
        has_nonfinite_y = true;
        bad_idx = i;
        break;
      }
    }
    if (has_nonfinite_y) {
      std::cerr << "[FP ERROR RANK " << rank << "] Non-finite new_state.y[" << bad_idx
                << "]=" << new_state.y[bad_idx] << " AFTER broadcast, time=" << time
                << std::endl;
    }
  }

  DEBUG_MSG_RANK("Integrator::step - before MPI_Bcast(state.ydot), rank=" << rank
                 << ", ydot.size=" << new_state.ydot.size());
  svzero_current_phase = SVZERO_PHASE_MPI_BCAST_STATE_YDOT;
  MPI_Bcast(new_state.ydot.data(),
            static_cast<int>(new_state.ydot.size()),
            MPI_DOUBLE,
            0,
            PETSC_COMM_WORLD);
  svzero_current_phase = SVZERO_PHASE_NONE;
  DEBUG_MSG_RANK("Integrator::step - after MPI_Bcast(state.ydot), rank=" << rank);

  // Check for NaN/Inf after receiving broadcast on non-root ranks
  {
    bool has_nonfinite_ydot = false;
    Eigen::Index bad_idx = -1;
    for (Eigen::Index i = 0; i < new_state.ydot.size(); ++i) {
      if (!std::isfinite(new_state.ydot[i])) {
        has_nonfinite_ydot = true;
        bad_idx = i;
        break;
      }
    }
    if (has_nonfinite_ydot) {
      std::cerr << "[FP ERROR RANK " << rank << "] Non-finite new_state.ydot[" << bad_idx
                << "]=" << new_state.ydot[bad_idx] << " AFTER broadcast, time=" << time
                << std::endl;
    }
  }

  DEBUG_MSG_RANK("Integrator::step - EXIT, rank=" << rank << ", time=" << time);
#endif

  return new_state;
}

double Integrator::avg_nonlin_iter() {
  if (n_iter == 0) {
    return 0.0;
  }
  return static_cast<double>(n_nonlin_iter) / static_cast<double>(n_iter);
}
