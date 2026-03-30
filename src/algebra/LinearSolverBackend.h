// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the
// University of California, and others. SPDX-License-Identifier: BSD-3-Clause
/**
 * @file LinearSolverBackend.h
 * @brief Linear solver backend abstraction.
 */
#ifndef SVZERODSOLVER_ALGEBRA_LINEARSOLVERBACKEND_HPP_
#define SVZERODSOLVER_ALGEBRA_LINEARSOLVERBACKEND_HPP_

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include <memory>

#include "LinearSolverSettings.h"

class Model;

struct LinearSolveStepInfo {
  int iterations{0};
  double error{0.0};
};

class LinearSolverBackend {
 public:
  using SparseMatrix = Eigen::SparseMatrix<double>;
  using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;

  virtual ~LinearSolverBackend() = default;

  virtual void analyze_pattern(const SparseMatrix& matrix) = 0;
  virtual void factorize(const SparseMatrix& matrix) = 0;
  virtual Vector solve(const Vector& rhs) = 0;
  virtual Eigen::ComputationInfo info() const = 0;
  virtual const char* name() const = 0;
  virtual bool is_iterative() const = 0;
  virtual LinearSolveStepInfo last_step_info() const = 0;
  virtual void configure_model(const Model* model) {}
};

std::shared_ptr<LinearSolverBackend> create_linear_solver_backend(
    const LinearSolverSettings& settings);

#endif  // SVZERODSOLVER_ALGEBRA_LINEARSOLVERBACKEND_HPP_
