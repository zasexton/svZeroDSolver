// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the
// University of California, and others. SPDX-License-Identifier: BSD-3-Clause

#include "SparseSystem.h"

#include <algorithm>
#include <sstream>

#include "Model.h"

SparseSystem::SparseSystem() {
  solver_backend = create_linear_solver_backend(linear_solver_settings);
  linear_solve_stats.backend_name = solver_backend->name();
  linear_solve_stats.iterative = solver_backend->is_iterative();
}

SparseSystem::SparseSystem(int n) : SparseSystem(n, LinearSolverSettings()) {}

SparseSystem::SparseSystem(int n,
                           const LinearSolverSettings& linear_solver_settings) {
  F = Eigen::SparseMatrix<double>(n, n);
  E = Eigen::SparseMatrix<double>(n, n);
  dC_dy = Eigen::SparseMatrix<double>(n, n);
  dC_dydot = Eigen::SparseMatrix<double>(n, n);
  C = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(n);

  jacobian = Eigen::SparseMatrix<double>(n, n);
  residual = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(n);
  dydot = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(n);

  this->linear_solver_settings = linear_solver_settings;
  solver_backend = create_linear_solver_backend(this->linear_solver_settings);
  linear_solve_stats.backend_name = solver_backend->name();
  linear_solve_stats.iterative = solver_backend->is_iterative();
}

SparseSystem::~SparseSystem() {}

void SparseSystem::clean() {
  // Cannot be in destructor because dynamically allocated pointers will be lost
  // when objects are assigned from temporary objects.
  // delete solver;
}

void SparseSystem::reserve(Model* model) {
  solver_backend->configure_model(model);
  has_solution_dependent_terms = model->has_solution_dependent_terms();
  constant_jacobian = model->has_constant_jacobian();
  jacobian_current = false;
  factorization_current = false;

  auto num_triplets = model->get_num_triplets();
  if (!model->supports_fast_system_initialization()) {
    F.reserve(num_triplets.F);
    E.reserve(num_triplets.E);
  }
  if (has_solution_dependent_terms) {
    dC_dy.reserve(num_triplets.D);
    dC_dydot.reserve(num_triplets.D);
  }

  model->initialize_system(*this, 0.0);

  if (has_solution_dependent_terms) {
    Eigen::Matrix<double, Eigen::Dynamic, 1> dummy_y =
        Eigen::Matrix<double, Eigen::Dynamic, 1>::Ones(residual.size());

    Eigen::Matrix<double, Eigen::Dynamic, 1> dummy_dy =
        Eigen::Matrix<double, Eigen::Dynamic, 1>::Ones(residual.size());

    model->update_solution(*this, dummy_y, dummy_dy);
  }

  F.makeCompressed();
  E.makeCompressed();
  if (has_solution_dependent_terms) {
    dC_dy.makeCompressed();
    dC_dydot.makeCompressed();
  }
  jacobian.reserve(num_triplets.F + num_triplets.E);  // Just an estimate
  update_jacobian(1.0, 1.0);  // Update it once to have sparsity pattern
  jacobian.makeCompressed();
  solver_backend->analyze_pattern(jacobian);  // Let solver analyze pattern
  jacobian_current = false;
  factorization_current = false;
}

void SparseSystem::update_residual(
    Eigen::Matrix<double, Eigen::Dynamic, 1>& y,
    Eigen::Matrix<double, Eigen::Dynamic, 1>& ydot) {
  residual.setZero();
  residual -= C;
  residual.noalias() -= E * ydot;
  residual.noalias() -= F * y;
}

void SparseSystem::update_jacobian(double time_coeff_ydot,
                                   double time_coeff_y) {
  if (constant_jacobian && jacobian_current) {
    return;
  }

  jacobian.setZero();
  jacobian += (E + dC_dydot) * time_coeff_ydot;
  jacobian += (F + dC_dy) * time_coeff_y;
  jacobian_current = true;
  factorization_current = false;
}

void SparseSystem::solve() {
  jacobian.makeCompressed();
  if (!factorization_current) {
    solver_backend->factorize(jacobian);
    linear_solve_stats.factorization_calls += 1;
    factorization_current = true;
    if (solver_backend->info() != Eigen::Success) {
      throw std::runtime_error(
          "Linear solver " + linear_solve_stats.backend_name +
          " failed during factorization or preconditioner setup. Check your "
          "model (connections, boundary conditions, parameters).");
    }
  }

  dydot = solver_backend->solve(residual);

  linear_solve_stats.solve_calls += 1;
  const auto step_info = solver_backend->last_step_info();
  linear_solve_stats.last_iterations = step_info.iterations;
  linear_solve_stats.total_iterations += step_info.iterations;
  linear_solve_stats.last_error = step_info.error;
  linear_solve_stats.max_error =
      std::max(linear_solve_stats.max_error, step_info.error);

  if (solver_backend->info() != Eigen::Success) {
    std::ostringstream message;
    message << "Linear solver " << linear_solve_stats.backend_name
            << " failed";
    if (linear_solve_stats.iterative) {
      message << " to converge after " << step_info.iterations
              << " iterations with estimated error " << step_info.error;
    }
    message << ". Check your model (connections, boundary conditions, "
               "parameters)";
    if (linear_solve_stats.iterative) {
      message << ", or adjust linear_solver_tolerance and "
                 "linear_solver_max_iterations";
    }
    message << ".";
    throw std::runtime_error(message.str());
  }
}

const LinearSolveStats& SparseSystem::get_linear_solve_stats() const {
  return linear_solve_stats;
}

void SparseSystem::invalidate_linearization_cache() {
  jacobian_current = false;
  factorization_current = false;
}
