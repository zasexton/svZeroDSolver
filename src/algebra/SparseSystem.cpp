// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the
// University of California, and others. SPDX-License-Identifier: BSD-3-Clause

#include "SparseSystem.h"

#include <stdexcept>
#include <sstream>
#include <cstring>

#include "Model.h"
#include "debug.h"

// Default values if not provided via compile definitions from CMake.
#ifndef SVZERODSOLVER_ITERATIVE_SOLVER_TOLERANCE
#define SVZERODSOLVER_ITERATIVE_SOLVER_TOLERANCE 1e-6
#endif

#ifndef SVZERODSOLVER_ITERATIVE_SOLVER_MAX_ITERS
#define SVZERODSOLVER_ITERATIVE_SOLVER_MAX_ITERS 0
#endif

#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
namespace {

// Ensure PETSc is initialized once per process.
void ensure_petsc_initialized() {
  static bool initialized = false;
  if (!initialized) {
    PetscErrorCode ierr = PetscInitialize(nullptr, nullptr, nullptr, nullptr);
    if (ierr) {
      throw std::runtime_error("Failed to initialize PETSc");
    }
    initialized = true;
  }
}

// Compute a size-aware maximum number of iterations based on the system size.
int petsc_max_iterations(PetscInt n) {
  int max_iters = SVZERODSOLVER_ITERATIVE_SOLVER_MAX_ITERS;
  if (max_iters <= 0) {
    if (n <= 5000) {
      max_iters = std::max<PetscInt>(50, n);
    } else if (n <= 100000) {
      max_iters = 5000;
    } else {
      max_iters = 10000;
    }
  }
  return max_iters;
}

}  // namespace
#endif  // SVZERODSOLVER_HAVE_PETSC && SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES

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

#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
PetscGMRESLinearSolver::PetscGMRESLinearSolver() {
  ensure_petsc_initialized();
}

PetscGMRESLinearSolver::~PetscGMRESLinearSolver() {
  if (x_ != nullptr) {
    VecDestroy(&x_);
  }
  if (b_ != nullptr) {
    VecDestroy(&b_);
  }
  if (ksp_ != nullptr) {
    KSPDestroy(&ksp_);
  }
  if (A_ != nullptr) {
    MatDestroy(&A_);
  }
}

void PetscGMRESLinearSolver::analyze_pattern(
    const Eigen::SparseMatrix<double>& A) {
  n_ = static_cast<PetscInt>(A.rows());

  if (A_ != nullptr) {
    MatDestroy(&A_);
    A_ = nullptr;
  }
  if (x_ != nullptr) {
    VecDestroy(&x_);
    x_ = nullptr;
  }
  if (b_ != nullptr) {
    VecDestroy(&b_);
    b_ = nullptr;
  }
  if (ksp_ != nullptr) {
    KSPDestroy(&ksp_);
    ksp_ = nullptr;
  }

  PetscErrorCode ierr;

  // For now, create a sequential AIJ matrix. If the application is run under
  // MPI, PETSc can still manage parallelism internally based on PETSC_COMM_WORLD.
  ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD, n_, n_, 50, nullptr, &A_);
  if (ierr) {
    throw std::runtime_error("Failed to create PETSc matrix");
  }

  ierr = VecCreateSeq(PETSC_COMM_WORLD, n_, &x_);
  if (ierr) {
    throw std::runtime_error("Failed to create PETSc solution vector");
  }

  ierr = VecCreateSeq(PETSC_COMM_WORLD, n_, &b_);
  if (ierr) {
    throw std::runtime_error("Failed to create PETSc RHS vector");
  }

  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp_);
  if (ierr) {
    throw std::runtime_error("Failed to create PETSc KSP");
  }

  ierr = KSPSetType(ksp_, KSPGMRES);
  if (ierr) {
    throw std::runtime_error("Failed to set KSP type to GMRES");
  }

  PC pc;
  ierr = KSPGetPC(ksp_, &pc);
  if (ierr) {
    throw std::runtime_error("Failed to get PETSc PC");
  }

#ifdef SVZERODSOLVER_PETSC_PRECONDITIONER
  const char* pc_opt = SVZERODSOLVER_PETSC_PRECONDITIONER;
  if (std::strcmp(pc_opt, "none") == 0) {
    ierr = PCSetType(pc, PCNONE);
  } else if (std::strcmp(pc_opt, "jacobi") == 0) {
    ierr = PCSetType(pc, PCJACOBI);
  } else if (std::strcmp(pc_opt, "bjacobi") == 0) {
    ierr = PCSetType(pc, PCBJACOBI);
  } else if (std::strcmp(pc_opt, "ilu") == 0) {
    ierr = PCSetType(pc, PCILU);
  } else if (std::strcmp(pc_opt, "gamg") == 0) {
    ierr = PCSetType(pc, PCGAMG);
  } else if (std::strcmp(pc_opt, "hypre_BoomerAMG") == 0) {
    ierr = PCSetType(pc, PCHYPRE);
    if (!ierr) {
      ierr = PCHYPRESetType(pc, "boomeramg");
    }
  } else {
    // Default to Jacobi if an unknown option is provided.
    ierr = PCSetType(pc, PCJACOBI);
  }
  if (ierr) {
    throw std::runtime_error("Failed to set PETSc preconditioner type");
  }
#endif
}

void PetscGMRESLinearSolver::factorize(
    const Eigen::SparseMatrix<double>& A) {
  if (A_ == nullptr || ksp_ == nullptr) {
    throw std::runtime_error("PETSc GMRES solver not initialized");
  }

  // Assemble the PETSc matrix from the Eigen sparse matrix.
  PetscErrorCode ierr = MatZeroEntries(A_);
  if (ierr) {
    throw std::runtime_error("Failed to zero PETSc matrix");
  }

  for (int k = 0; k < A.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
      const PetscInt row = static_cast<PetscInt>(it.row());
      const PetscInt col = static_cast<PetscInt>(it.col());
      const PetscScalar val = static_cast<PetscScalar>(it.value());
      ierr = MatSetValue(A_, row, col, val, INSERT_VALUES);
      if (ierr) {
        throw std::runtime_error("Failed to set PETSc matrix entry");
      }
    }
  }

  ierr = MatAssemblyBegin(A_, MAT_FINAL_ASSEMBLY);
  if (ierr) {
    throw std::runtime_error("Failed to begin PETSc matrix assembly");
  }
  ierr = MatAssemblyEnd(A_, MAT_FINAL_ASSEMBLY);
  if (ierr) {
    throw std::runtime_error("Failed to end PETSc matrix assembly");
  }

  ierr = KSPSetOperators(ksp_, A_, A_);
  if (ierr) {
    throw std::runtime_error("Failed to set PETSc KSP operators");
  }

  const int max_iters = petsc_max_iterations(n_);
  ierr = KSPSetTolerances(ksp_, SVZERODSOLVER_ITERATIVE_SOLVER_TOLERANCE,
                          PETSC_DEFAULT, PETSC_DEFAULT, max_iters);
  if (ierr) {
    throw std::runtime_error("Failed to set PETSc KSP tolerances");
  }

  // Allow command-line options to further tune the solver if desired.
  ierr = KSPSetFromOptions(ksp_);
  if (ierr) {
    throw std::runtime_error("Failed to apply PETSc KSP options");
  }
}

void PetscGMRESLinearSolver::solve(
    const Eigen::Matrix<double, Eigen::Dynamic, 1>& b,
    Eigen::Matrix<double, Eigen::Dynamic, 1>& x) {
  if (ksp_ == nullptr || x_ == nullptr || b_ == nullptr) {
    throw std::runtime_error("PETSc GMRES solver not initialized");
  }

  PetscErrorCode ierr = VecSet(b_, 0.0);
  if (ierr) {
    throw std::runtime_error("Failed to zero PETSc RHS vector");
  }

  for (PetscInt i = 0; i < n_; ++i) {
    ierr = VecSetValue(b_, i, static_cast<PetscScalar>(b[i]), INSERT_VALUES);
    if (ierr) {
      throw std::runtime_error("Failed to set PETSc RHS entry");
    }
  }

  ierr = VecAssemblyBegin(b_);
  if (ierr) {
    throw std::runtime_error("Failed to begin PETSc RHS assembly");
  }
  ierr = VecAssemblyEnd(b_);
  if (ierr) {
    throw std::runtime_error("Failed to end PETSc RHS assembly");
  }

  ierr = VecSet(x_, 0.0);
  if (ierr) {
    throw std::runtime_error("Failed to zero PETSc solution vector");
  }

  ierr = KSPSolve(ksp_, b_, x_);
  if (ierr) {
    throw std::runtime_error("PETSc KSPSolve failed");
  }

  KSPConvergedReason reason;
  ierr = KSPGetConvergedReason(ksp_, &reason);
  if (ierr) {
    throw std::runtime_error("Failed to get PETSc KSP convergence reason");
  }

  if (reason < 0) {
    PetscInt its = 0;
    PetscReal rnorm = 0.0;
    KSPGetIterationNumber(ksp_, &its);
    KSPGetResidualNorm(ksp_, &rnorm);

    std::ostringstream oss;
    oss << "PETSc GMRES solve failed. reason=" << static_cast<int>(reason)
        << ", iterations=" << static_cast<int>(its)
        << ", residual=" << static_cast<double>(rnorm);
    throw std::runtime_error(oss.str());
  }

  const PetscScalar* px = nullptr;
  ierr = VecGetArrayRead(x_, &px);
  if (ierr) {
    throw std::runtime_error("Failed to access PETSc solution vector");
  }

  x.setZero();
  for (PetscInt i = 0; i < n_; ++i) {
    x[i] = static_cast<double>(px[i]);
  }

  ierr = VecRestoreArrayRead(x_, &px);
  if (ierr) {
    throw std::runtime_error("Failed to restore PETSc solution vector");
  }
}
#endif  // SVZERODSOLVER_HAVE_PETSC && SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES

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
#elif defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
  return std::make_shared<PetscGMRESLinearSolver>();
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
