// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the
// University of California, and others. SPDX-License-Identifier: BSD-3-Clause

#include "SparseSystem.h"

#include <stdexcept>
#include <sstream>

#include "Model.h"
#include "debug.h"

// Default values if not provided via compile definitions from CMake.
#ifndef SVZERODSOLVER_ITERATIVE_SOLVER_TOLERANCE
#define SVZERODSOLVER_ITERATIVE_SOLVER_TOLERANCE 1e-6
#endif

#ifndef SVZERODSOLVER_ITERATIVE_SOLVER_MAX_ITERS
#define SVZERODSOLVER_ITERATIVE_SOLVER_MAX_ITERS 0
#endif

void SparseLULinearSolver::analyze_pattern(
    const Eigen::SparseMatrix<double>& A) {
  solver_.analyzePattern(A);
}

void SparseLULinearSolver::factorize(
    const Eigen::SparseMatrix<double>& A) {
  solver_.factorize(A);
  if (solver_.info() != Eigen::Success) {
    throw std::runtime_error(
        "System is singular. Check your model (connections, boundary "
        "conditions, parameters).");
  }
}

void SparseLULinearSolver::solve(
    const Eigen::Matrix<double, Eigen::Dynamic, 1>& b,
    Eigen::Matrix<double, Eigen::Dynamic, 1>& x) {
  x.setZero();
  x += solver_.solve(b);
  if (solver_.info() != Eigen::Success) {
    throw std::runtime_error("Linear solve failed.");
  }
}

#if defined(SVZERODSOLVER_LINEAR_SOLVER_CONJUGATE_GRADIENT)
void ConjugateGradientLinearSolver::analyze_pattern(
    const Eigen::SparseMatrix<double>&) {
  // ConjugateGradient does not require a separate pattern analysis step.
}

void ConjugateGradientLinearSolver::factorize(
    const Eigen::SparseMatrix<double>& A) {
  int max_iters = SVZERODSOLVER_ITERATIVE_SOLVER_MAX_ITERS;
  if (max_iters <= 0) {
    const int n = static_cast<int>(A.rows());
    if (n <= 5000) {
      max_iters = std::max(50, n);
    } else if (n <= 100000) {
      max_iters = 5000;
    } else {
      max_iters = 10000;
    }
  }
  solver_.setMaxIterations(max_iters);
  solver_.setTolerance(SVZERODSOLVER_ITERATIVE_SOLVER_TOLERANCE);

  solver_.compute(A);
  if (solver_.info() != Eigen::Success) {
    throw std::runtime_error(
        "System is singular or ill-conditioned. Check your model "
        "(connections, boundary conditions, parameters).");
  }
}

void ConjugateGradientLinearSolver::solve(
    const Eigen::Matrix<double, Eigen::Dynamic, 1>& b,
    Eigen::Matrix<double, Eigen::Dynamic, 1>& x) {
  x.setZero();
  x += solver_.solve(b);
  if (solver_.info() != Eigen::Success) {
    std::ostringstream oss;
    oss << "Iterative linear solve (ConjugateGradient) failed. "
        << "info=" << static_cast<int>(solver_.info())
        << ", iterations=" << solver_.iterations()
        << ", error=" << solver_.error();
    throw std::runtime_error(oss.str());
  }
}
#endif  // SVZERODSOLVER_LINEAR_SOLVER_CONJUGATE_GRADIENT

#if defined(SVZERODSOLVER_LINEAR_SOLVER_LEAST_SQUARES_CONJUGATE_GRADIENT)
void LeastSquaresConjugateGradientLinearSolver::analyze_pattern(
    const Eigen::SparseMatrix<double>&) {
  // LeastSquaresConjugateGradient does not require a separate pattern analysis step.
}

void LeastSquaresConjugateGradientLinearSolver::factorize(
    const Eigen::SparseMatrix<double>& A) {
  int max_iters = SVZERODSOLVER_ITERATIVE_SOLVER_MAX_ITERS;
  if (max_iters <= 0) {
    const int n = static_cast<int>(A.rows());
    if (n <= 5000) {
      max_iters = std::max(50, n);
    } else if (n <= 100000) {
      max_iters = 5000;
    } else {
      max_iters = 10000;
    }
  }
  solver_.setMaxIterations(max_iters);
  solver_.setTolerance(SVZERODSOLVER_ITERATIVE_SOLVER_TOLERANCE);

  solver_.compute(A);
  if (solver_.info() != Eigen::Success) {
    throw std::runtime_error(
        "System is singular or ill-conditioned. Check your model "
        "(connections, boundary conditions, parameters).");
  }
}

void LeastSquaresConjugateGradientLinearSolver::solve(
    const Eigen::Matrix<double, Eigen::Dynamic, 1>& b,
    Eigen::Matrix<double, Eigen::Dynamic, 1>& x) {
  x.setZero();
  x += solver_.solve(b);
  if (solver_.info() != Eigen::Success) {
    std::ostringstream oss;
    oss << "Iterative linear solve (LeastSquaresConjugateGradient) failed. "
        << "info=" << static_cast<int>(solver_.info())
        << ", iterations=" << solver_.iterations()
        << ", error=" << solver_.error();
    throw std::runtime_error(oss.str());
  }
}
#endif  // SVZERODSOLVER_LINEAR_SOLVER_LEAST_SQUARES_CONJUGATE_GRADIENT

#if defined(SVZERODSOLVER_LINEAR_SOLVER_BICGSTAB)
void BiCGSTABLinearSolver::analyze_pattern(
    const Eigen::SparseMatrix<double>&) {
  // BiCGSTAB does not require a separate pattern analysis step.
}

void BiCGSTABLinearSolver::factorize(
    const Eigen::SparseMatrix<double>& A) {
  int max_iters = SVZERODSOLVER_ITERATIVE_SOLVER_MAX_ITERS;
  if (max_iters <= 0) {
    const int n = static_cast<int>(A.rows());
    if (n <= 5000) {
      max_iters = std::max(50, n);
    } else if (n <= 100000) {
      max_iters = 5000;
    } else {
      max_iters = 10000;
    }
  }
  solver_.setMaxIterations(max_iters);
  solver_.setTolerance(SVZERODSOLVER_ITERATIVE_SOLVER_TOLERANCE);

  solver_.compute(A);
  if (solver_.info() != Eigen::Success) {
    throw std::runtime_error(
        "System is singular or ill-conditioned. Check your model "
        "(connections, boundary conditions, parameters).");
  }
}

void BiCGSTABLinearSolver::solve(
    const Eigen::Matrix<double, Eigen::Dynamic, 1>& b,
    Eigen::Matrix<double, Eigen::Dynamic, 1>& x) {
  x.setZero();
  x += solver_.solve(b);
  if (solver_.info() != Eigen::Success) {
    std::ostringstream oss;
    oss << "Iterative linear solve (BiCGSTAB) failed. "
        << "info=" << static_cast<int>(solver_.info())
        << ", iterations=" << solver_.iterations()
        << ", error=" << solver_.error();
    throw std::runtime_error(oss.str());
  }
}
#endif  // SVZERODSOLVER_LINEAR_SOLVER_BICGSTAB

#if defined(SVZERODSOLVER_LINEAR_SOLVER_PARDISO_LU)
void PardisoLULinearSolver::analyze_pattern(
    const Eigen::SparseMatrix<double>& A) {
  solver_.analyzePattern(A);
}

void PardisoLULinearSolver::factorize(
    const Eigen::SparseMatrix<double>& A) {
  solver_.factorize(A);
  if (solver_.info() != Eigen::Success) {
    throw std::runtime_error(
        "System is singular. Check your model (connections, boundary "
        "conditions, parameters).");
  }
}

void PardisoLULinearSolver::solve(
    const Eigen::Matrix<double, Eigen::Dynamic, 1>& b,
    Eigen::Matrix<double, Eigen::Dynamic, 1>& x) {
  x.setZero();
  x += solver_.solve(b);
  if (solver_.info() != Eigen::Success) {
    throw std::runtime_error("Linear solve failed.");
  }
}
#endif  // SVZERODSOLVER_LINEAR_SOLVER_PARDISO_LU

#if defined(SVZERODSOLVER_LINEAR_SOLVER_PARDISO_LDLT)
void PardisoLDLTLinearSolver::analyze_pattern(
    const Eigen::SparseMatrix<double>& A) {
  solver_.analyzePattern(A);
}

void PardisoLDLTLinearSolver::factorize(
    const Eigen::SparseMatrix<double>& A) {
  solver_.factorize(A);
  if (solver_.info() != Eigen::Success) {
    throw std::runtime_error(
        "System is singular. Check your model (connections, boundary "
        "conditions, parameters).");
  }
}

void PardisoLDLTLinearSolver::solve(
    const Eigen::Matrix<double, Eigen::Dynamic, 1>& b,
    Eigen::Matrix<double, Eigen::Dynamic, 1>& x) {
  x.setZero();
  x += solver_.solve(b);
  if (solver_.info() != Eigen::Success) {
    throw std::runtime_error("Linear solve failed.");
  }
}
#endif  // SVZERODSOLVER_LINEAR_SOLVER_PARDISO_LDLT

namespace {

std::shared_ptr<LinearSolver> make_default_linear_solver() {
#if defined(SVZERODSOLVER_LINEAR_SOLVER_SPARSELU)
  return std::make_shared<SparseLULinearSolver>();
#elif defined(SVZERODSOLVER_LINEAR_SOLVER_CONJUGATE_GRADIENT)
  return std::make_shared<ConjugateGradientLinearSolver>();
#elif defined(SVZERODSOLVER_LINEAR_SOLVER_LEAST_SQUARES_CONJUGATE_GRADIENT)
  return std::make_shared<LeastSquaresConjugateGradientLinearSolver>();
#elif defined(SVZERODSOLVER_LINEAR_SOLVER_BICGSTAB)
  return std::make_shared<BiCGSTABLinearSolver>();
#elif defined(SVZERODSOLVER_LINEAR_SOLVER_PARDISO_LU)
  return std::make_shared<PardisoLULinearSolver>();
#elif defined(SVZERODSOLVER_LINEAR_SOLVER_PARDISO_LDLT)
  return std::make_shared<PardisoLDLTLinearSolver>();
#else
  // Fallback to SparseLU if no other backend is configured.
  return std::make_shared<SparseLULinearSolver>();
#endif
}

}  // namespace

SparseSystem::SparseSystem() : solver(make_default_linear_solver()) {}

SparseSystem::SparseSystem(int n) : solver(make_default_linear_solver()) {
  F = Eigen::SparseMatrix<double>(n, n);
  E = Eigen::SparseMatrix<double>(n, n);
  dC_dy = Eigen::SparseMatrix<double>(n, n);
  dC_dydot = Eigen::SparseMatrix<double>(n, n);
  C = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(n);

  jacobian = Eigen::SparseMatrix<double>(n, n);
  residual = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(n);
  dydot = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(n);
}

SparseSystem::~SparseSystem() {}

void SparseSystem::clean() {
  // Cannot be in destructor because dynamically allocated pointers will be lost
  // when objects are assigned from temporary objects.
  // delete solver;
}

void SparseSystem::reserve(Model* model) {
  auto num_triplets = model->get_num_triplets();
  F.reserve(num_triplets.F);
  E.reserve(num_triplets.E);
  dC_dy.reserve(num_triplets.D);
  dC_dydot.reserve(num_triplets.D);

  model->update_constant(*this);
  model->update_time(*this, 0.0);

  Eigen::Matrix<double, Eigen::Dynamic, 1> dummy_y =
      Eigen::Matrix<double, Eigen::Dynamic, 1>::Ones(residual.size());

  Eigen::Matrix<double, Eigen::Dynamic, 1> dummy_dy =
      Eigen::Matrix<double, Eigen::Dynamic, 1>::Ones(residual.size());

  model->update_solution(*this, dummy_y, dummy_dy);

  F.makeCompressed();
  E.makeCompressed();
  dC_dy.makeCompressed();
  dC_dydot.makeCompressed();
  jacobian.reserve(num_triplets.F + num_triplets.E);  // Just an estimate
  update_jacobian(1.0, 1.0);  // Update it once to have sparsity pattern
  jacobian.makeCompressed();
  solver->analyze_pattern(jacobian);  // Let solver analyze pattern
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
  jacobian.setZero();
  jacobian += (E + dC_dydot) * time_coeff_ydot;
  jacobian += (F + dC_dy) * time_coeff_y;
}

void SparseSystem::solve() {
  try {
    solver->factorize(jacobian);
    solver->solve(residual, dydot);
  } catch (const std::runtime_error& e) {
#if defined(SVZERODSOLVER_LINEAR_SOLVER_CONJUGATE_GRADIENT) || \
    defined(SVZERODSOLVER_LINEAR_SOLVER_LEAST_SQUARES_CONJUGATE_GRADIENT) || \
    defined(SVZERODSOLVER_LINEAR_SOLVER_BICGSTAB)
    // Fallback to a direct SparseLU solve if an iterative backend fails.
    DEBUG_MSG("Iterative linear solver failed ("
              << e.what()
              << "), falling back to SparseLU.");
    SparseLULinearSolver direct_solver;
    direct_solver.analyze_pattern(jacobian);
    direct_solver.factorize(jacobian);
    direct_solver.solve(residual, dydot);
#else
    throw;
#endif
  }
}
