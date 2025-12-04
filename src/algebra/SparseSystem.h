// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the
// University of California, and others. SPDX-License-Identifier: BSD-3-Clause
/**
 * @file SparseSystem.h
 * @brief SparseSystem source file
 */
#ifndef SVZERODSOLVER_ALGREBRA_SPARSESYSTEM_HPP_
#define SVZERODSOLVER_ALGREBRA_SPARSESYSTEM_HPP_

#include <Eigen/SparseLU>
#include <Eigen/Sparse>

#if defined(SVZERODSOLVER_LINEAR_SOLVER_PARDISO_LU) || \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PARDISO_LDLT)
#include <Eigen/PardisoSupport>
#endif

#include <Eigen/IterativeLinearSolvers>

#if defined(SVZERODSOLVER_LINEAR_SOLVER_CONJUGATE_GRADIENT) ||         \
    defined(SVZERODSOLVER_LINEAR_SOLVER_LEAST_SQUARES_CONJUGATE_GRADIENT) || \
    defined(SVZERODSOLVER_LINEAR_SOLVER_BICGSTAB)
#if defined(SVZERODSOLVER_PRECONDITIONER_IDENTITY)
using SvZeroIterativePreconditioner = Eigen::IdentityPreconditioner;
#elif defined(SVZERODSOLVER_PRECONDITIONER_DIAGONAL)
using SvZeroIterativePreconditioner =
    Eigen::DiagonalPreconditioner<double>;
#else
using SvZeroIterativePreconditioner =
    Eigen::DiagonalPreconditioner<double>;
#endif
#endif

#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
#include <petscksp.h>
#endif

#include <iostream>
#include <memory>
#include <vector>

// Forward declaration of Model
class Model;

/// Backend used by SparseSystem for its linear algebra representation.
enum class LinearBackend {
  Eigen,
  PETSc
};

/**
 * @brief Abstract base class for linear solvers
 *
 * This provides a small interface that allows plugging in different
 * sparse linear solver backends (e.g. SparseLU, Pardiso, etc.)
 * without changing the rest of the algebra or model code.
 */
class LinearSolver {
 public:
  virtual ~LinearSolver() = default;

  virtual void analyze_pattern(const Eigen::SparseMatrix<double>& A) = 0;

  virtual void factorize(const Eigen::SparseMatrix<double>& A) = 0;

  virtual void solve(const Eigen::Matrix<double, Eigen::Dynamic, 1>& b,
                     Eigen::Matrix<double, Eigen::Dynamic, 1>& x) = 0;
};

/**
 * @brief Eigen::SparseLU-based implementation of LinearSolver.
 *
 * This is the default solver backend and matches the behavior that
 * SparseSystem used previously.
 */
class SparseLULinearSolver : public LinearSolver {
 public:
  SparseLULinearSolver() = default;
  ~SparseLULinearSolver() override = default;

  void analyze_pattern(const Eigen::SparseMatrix<double>& A) override;

  void factorize(const Eigen::SparseMatrix<double>& A) override;

  void solve(const Eigen::Matrix<double, Eigen::Dynamic, 1>& b,
             Eigen::Matrix<double, Eigen::Dynamic, 1>& x) override;

 private:
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver_;
};

#if defined(SVZERODSOLVER_LINEAR_SOLVER_CONJUGATE_GRADIENT)
/**
 * @brief Eigen::ConjugateGradient-based implementation of LinearSolver.
 */
class ConjugateGradientLinearSolver : public LinearSolver {
 public:
  ConjugateGradientLinearSolver() = default;
  ~ConjugateGradientLinearSolver() override = default;

  void analyze_pattern(const Eigen::SparseMatrix<double>& A) override;

  void factorize(const Eigen::SparseMatrix<double>& A) override;

  void solve(const Eigen::Matrix<double, Eigen::Dynamic, 1>& b,
             Eigen::Matrix<double, Eigen::Dynamic, 1>& x) override;

 private:
  Eigen::ConjugateGradient<Eigen::SparseMatrix<double>,
                           Eigen::Lower | Eigen::Upper,
                           SvZeroIterativePreconditioner>
      solver_;
};
#endif  // SVZERODSOLVER_LINEAR_SOLVER_CONJUGATE_GRADIENT

#if defined(SVZERODSOLVER_LINEAR_SOLVER_LEAST_SQUARES_CONJUGATE_GRADIENT)
/**
 * @brief Eigen::LeastSquaresConjugateGradient-based implementation of LinearSolver.
 */
class LeastSquaresConjugateGradientLinearSolver : public LinearSolver {
 public:
  LeastSquaresConjugateGradientLinearSolver() = default;
  ~LeastSquaresConjugateGradientLinearSolver() override = default;

  void analyze_pattern(const Eigen::SparseMatrix<double>& A) override;

  void factorize(const Eigen::SparseMatrix<double>& A) override;

  void solve(const Eigen::Matrix<double, Eigen::Dynamic, 1>& b,
             Eigen::Matrix<double, Eigen::Dynamic, 1>& x) override;

 private:
  Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>,
                                       SvZeroIterativePreconditioner>
      solver_;
};
#endif  // SVZERODSOLVER_LINEAR_SOLVER_LEAST_SQUARES_CONJUGATE_GRADIENT

#if defined(SVZERODSOLVER_LINEAR_SOLVER_BICGSTAB)
/**
 * @brief Eigen::BiCGSTAB-based implementation of LinearSolver.
 */
class BiCGSTABLinearSolver : public LinearSolver {
 public:
  BiCGSTABLinearSolver() = default;
  ~BiCGSTABLinearSolver() override = default;

  void analyze_pattern(const Eigen::SparseMatrix<double>& A) override;

  void factorize(const Eigen::SparseMatrix<double>& A) override;

  void solve(const Eigen::Matrix<double, Eigen::Dynamic, 1>& b,
             Eigen::Matrix<double, Eigen::Dynamic, 1>& x) override;

 private:
  Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, SvZeroIterativePreconditioner>
      solver_;
};
#endif  // SVZERODSOLVER_LINEAR_SOLVER_BICGSTAB

#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
/**
 * @brief PETSc GMRES-based implementation of LinearSolver.
 *
 * This backend uses PETSc's KSP with GMRES and a configurable PETSc
 * preconditioner (e.g., Jacobi, ASM, GAMG, hypre BoomerAMG).
 */
class PetscGMRESLinearSolver : public LinearSolver {
 public:
  PetscGMRESLinearSolver();
  ~PetscGMRESLinearSolver() override;

  void analyze_pattern(const Eigen::SparseMatrix<double>& A) override;

  void factorize(const Eigen::SparseMatrix<double>& A) override;

  void solve(const Eigen::Matrix<double, Eigen::Dynamic, 1>& b,
             Eigen::Matrix<double, Eigen::Dynamic, 1>& x) override;

  // Accessors used by PETSc-centric assembly paths. These return aliases to
  // the underlying PETSc objects managed by this solver.
  Mat get_matrix() const { return A_; }
  Vec get_rhs() const { return b_; }
  Vec get_solution() const { return x_; }

 private:
  Mat A_ = nullptr;
  KSP ksp_ = nullptr;
  Vec x_ = nullptr;
  Vec b_ = nullptr;
  Vec x_seq_ = nullptr;
  VecScatter scatter_to_root_ = nullptr;
  PetscInt n_ = 0;
  PetscInt rstart_ = 0;
  PetscInt rend_ = 0;
};
#endif  // SVZERODSOLVER_HAVE_PETSC && SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES

#if defined(SVZERODSOLVER_LINEAR_SOLVER_PARDISO_LU)
/**
 * @brief Eigen::PardisoLU-based implementation of LinearSolver.
 *
 * This backend requires Eigen to be configured with MKL/PARDISO support.
 */
class PardisoLULinearSolver : public LinearSolver {
 public:
  PardisoLULinearSolver() = default;
  ~PardisoLULinearSolver() override = default;

  void analyze_pattern(const Eigen::SparseMatrix<double>& A) override;

  void factorize(const Eigen::SparseMatrix<double>& A) override;

  void solve(const Eigen::Matrix<double, Eigen::Dynamic, 1>& b,
             Eigen::Matrix<double, Eigen::Dynamic, 1>& x) override;

 private:
  Eigen::PardisoLU<Eigen::SparseMatrix<double>> solver_;
};
#endif  // SVZERODSOLVER_LINEAR_SOLVER_PARDISO_LU

#if defined(SVZERODSOLVER_LINEAR_SOLVER_PARDISO_LDLT)
/**
 * @brief Eigen::PardisoLDLT-based implementation of LinearSolver.
 *
 * This backend requires Eigen to be configured with MKL/PARDISO support
 * and is suitable for (numerically) symmetric positive definite systems.
 */
class PardisoLDLTLinearSolver : public LinearSolver {
 public:
  PardisoLDLTLinearSolver() = default;
  ~PardisoLDLTLinearSolver() override = default;

  void analyze_pattern(const Eigen::SparseMatrix<double>& A) override;

  void factorize(const Eigen::SparseMatrix<double>& A) override;

  void solve(const Eigen::Matrix<double, Eigen::Dynamic, 1>& b,
             Eigen::Matrix<double, Eigen::Dynamic, 1>& x) override;

 private:
  Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> solver_;
};
#endif  // SVZERODSOLVER_LINEAR_SOLVER_PARDISO_LDLT

/**
 * @brief Sparse system
 *
 * This class contains all attributes and methods to create, modify, and
 * solve sparse systems.
 *
 * Mathematical details related to setting up the governing system of
 * equations are available on the <a
 * href="https://simvascular.github.io/documentation/rom_simulation.html#0d-solver-theory">SimVascular
 * documentation</a>.
 *
 */
class SparseSystem {
 public:
  /**
   * @brief Construct a new Sparse System object
   *
   */
  SparseSystem();

  /**
   * @brief Construct a new Sparse System object
   *
   * @param n Size of the system
   */
  SparseSystem(int n);

  /**
   * @brief Destroy the Sparse System object
   *
   */
  ~SparseSystem();

  Eigen::SparseMatrix<double> F;               ///< System matrix F
  Eigen::SparseMatrix<double> E;               ///< System matrix E
  Eigen::SparseMatrix<double> dC_dy;           ///< System matrix dC/dy
  Eigen::SparseMatrix<double> dC_dydot;        ///< System matrix dC/dydot
  Eigen::Matrix<double, Eigen::Dynamic, 1> C;  ///< System vector C

  Eigen::SparseMatrix<double> jacobian;  ///< Jacobian of the system
  Eigen::Matrix<double, Eigen::Dynamic, 1>
      residual;  ///< Residual of the system
  Eigen::Matrix<double, Eigen::Dynamic, 1>
      dydot;  ///< Solution increment of the system

  /**
   * @brief Reserve memory in system matrices based on number of triplets
   *
   * @param model The model to reserve space for in the system
   */
  void reserve(Model* model);

  /**
   * @brief Update the residual of the system
   *
   * @param y Vector of current solution quantities
   * @param ydot Derivate of y
   */
  void update_residual(Eigen::Matrix<double, Eigen::Dynamic, 1>& y,
                       Eigen::Matrix<double, Eigen::Dynamic, 1>& ydot);

  /**
   * @brief Update the jacobian of the system
   *
   * @param time_coeff_ydot Coefficent ydot-dependent part of jacobian
   * @param time_coeff_y Coefficent ydot-dependent part of jacobian
   */
  void update_jacobian(double time_coeff_ydot, double time_coeff_y);

  /**
   * @brief Solve the system
   */
  void solve();

  /**
   * @brief Delete dynamically allocated memory (class member
   * LinearSolver solver)
   */
  void clean();

  // Assembly helpers: use these instead of touching F/E/dC_* directly so that
  // we can switch between direct coeffRef-based assembly and triplet-based
  // assembly transparently.
  void add_F(int row, int col, double value);
  void add_E(int row, int col, double value);
  void add_dC_dy(int row, int col, double value);
  void add_dC_dydot(int row, int col, double value);

 private:
  /// Linear solver backend
  std::shared_ptr<LinearSolver> solver;

  /// Representation backend (Eigen or PETSc) used by this sparse system.
  LinearBackend backend_ = LinearBackend::Eigen;

  // When true, assembly into F/E/dC_* is performed via triplet lists instead
  // of direct coeffRef writes. This is enabled during the initial reserve()
  // phase for large PETSc-based runs to avoid expensive sparse matrix
  // reallocations when inserting many entries.
  bool use_triplets_ = false;
  std::vector<Eigen::Triplet<double>> F_triplets_;
  std::vector<Eigen::Triplet<double>> E_triplets_;
  std::vector<Eigen::Triplet<double>> dC_dy_triplets_;
  std::vector<Eigen::Triplet<double>> dC_dydot_triplets_;
};

#endif  // SVZERODSOLVER_ALGREBRA_SPARSESYSTEM_HPP_
