// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the
// University of California, and others. SPDX-License-Identifier: BSD-3-Clause
/**
 * @file LinearSolverSettings.h
 * @brief Linear solver backend configuration.
 */
#ifndef SVZERODSOLVER_ALGEBRA_LINEARSOLVERSETTINGS_HPP_
#define SVZERODSOLVER_ALGEBRA_LINEARSOLVERSETTINGS_HPP_

enum class LinearSolverBackendType {
  sparse_lu,
  bicgstab_ilut,
  gmres_ilut,
  gmres_diagonal,
  tree_linear
};

struct LinearSolverSettings {
  LinearSolverBackendType backend{LinearSolverBackendType::sparse_lu};
  double tolerance{1e-8};
  int max_iterations{0};
  double ilut_drop_tolerance{1e-4};
  int ilut_fill_factor{10};
  int gmres_restart{50};
};

inline const char* linear_solver_backend_name(
    LinearSolverBackendType backend) {
  switch (backend) {
    case LinearSolverBackendType::sparse_lu:
      return "SparseLU";
    case LinearSolverBackendType::bicgstab_ilut:
      return "BiCGSTAB_ILUT";
    case LinearSolverBackendType::gmres_ilut:
      return "GMRES_ILUT";
    case LinearSolverBackendType::gmres_diagonal:
      return "GMRES_DIAGONAL";
    case LinearSolverBackendType::tree_linear:
      return "TreeLinear";
  }
  return "Unknown";
}

#endif  // SVZERODSOLVER_ALGEBRA_LINEARSOLVERSETTINGS_HPP_
