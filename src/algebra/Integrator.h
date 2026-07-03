// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the
// University of California, and others. SPDX-License-Identifier: BSD-3-Clause
/**
 * @file Integrator.h
 * @brief Integrator source file
 */
#ifndef SVZERODSOLVER_ALGEBRA_INTEGRATOR_HPP_
#define SVZERODSOLVER_ALGEBRA_INTEGRATOR_HPP_

#include <Eigen/Dense>

#include <string>

#include "Model.h"
#include "State.h"

/**
 * @brief Options controlling nonlinear Newton iteration.
 */
struct IntegratorOptions {
  bool newton_line_search{false};  ///< Use damped Newton backtracking
  double line_search_reduction{0.5};  ///< Step reduction factor in (0, 1)
  double line_search_min_step{1.0e-4};  ///< Smallest damping factor to try
  int line_search_max_iterations{12};  ///< Maximum backtracking trials
  double line_search_sufficient_decrease{
      1.0e-4};  ///< Armijo-like sufficient decrease coefficient
  bool line_search_fallback_to_full_step{
      true};  ///< Use full Newton step if no damping step improves residual
  bool use_newton_scaling{
      false};  ///< Use scaled residual norms and scaled linear solves
  double pressure_scale{1.0e4};  ///< Typical pressure magnitude for scaling
  double flow_scale{1.0e2};      ///< Typical flow magnitude for scaling
  double volume_scale{1.0};      ///< Typical volume magnitude for scaling
  double variable_scale{1.0};    ///< Fallback variable magnitude for scaling
  double residual_scale_floor{1.0};  ///< Minimum equation residual scale
};

/**
 * @brief Generalized-alpha integrator
 *
 * This class handles the time integration scheme for solving 0D blood
 * flow system using the generalized-\f$\alpha\f$ method \cite JANSEN2000305.
 *
 * Mathematical details are available on the <a
 * href="https://simvascular.github.io/documentation/rom_simulation.html#0d-solver-theory">SimVascular
 * documentation</a>.
 */

class Integrator {
 private:
  double alpha_m{0.0};
  double alpha_f{0.0};
  double gamma{0.0};
  double time_step_size{0.0};
  double ydot_init_coeff{0.0};
  double y_coeff{0.0};
  double y_coeff_jacobian{0.0};
  double atol{0.0};
  int max_iter{0};
  int size{0};
  int n_iter{0};
  int n_nonlin_iter{0};
  Eigen::Matrix<double, Eigen::Dynamic, 1> y_af;
  Eigen::Matrix<double, Eigen::Dynamic, 1> ydot_am;
  Eigen::Matrix<double, Eigen::Dynamic, 1> y_scale;
  Eigen::Matrix<double, Eigen::Dynamic, 1> ydot_scale;
  Eigen::Matrix<double, Eigen::Dynamic, 1> residual_scale;
  SparseSystem system;
  Model* model{nullptr};
  IntegratorOptions options;
  int last_step_n_nonlin_iter{0};
  int last_step_line_search_trials{0};
  double last_step_min_line_search_step{1.0};
  double last_step_residual_norm{0.0};

  /**
   * @brief Configure variable and derivative scales.
   */
  void setup_scaling();

  /**
   * @brief Update derivative scales after time-step size changes.
   */
  void update_derivative_scaling();

  /**
   * @brief Get variable scale from variable name and user options.
   */
  double get_variable_scale(const std::string& variable_name) const;

  /**
   * @brief Update generalized-alpha intermediate state vectors.
   */
  void update_intermediate_state(const State& old_state,
                                 const State& new_state);

  /**
   * @brief Evaluate residual norm for a candidate Newton state.
   */
  double evaluate_residual(const State& old_state, const State& new_state);

  /**
   * @brief Update row scales used for residual norm and scaled linear solve.
   */
  void update_residual_scaling();

  /**
   * @brief Return current residual norm using configured scaling.
   */
  double residual_norm() const;

  /**
   * @brief Apply Newton update, optionally using damped line search.
   */
  void apply_newton_update(State& new_state, const State& old_state,
                           double current_residual_norm, double time);

 public:
  /**
   * @brief Construct a new Integrator object
   *
   * @param model The model to simulate
   * @param time_step_size Time step size for generalized-alpha step
   * @param rho Spectral radius for generalized-alpha step
   * @param atol Absolut tolerance for non-linear iteration termination
   * @param max_iter Maximum number of non-linear iterations
   * @param options Nonlinear solver options
   */
  Integrator(Model* model, double time_step_size, double rho, double atol,
             int max_iter, const IntegratorOptions& options = {});

  /**
   * @brief Construct a new Integrator object
   *
   */
  Integrator();

  /**
   * @brief Destroy the Integrator object
   *
   */
  ~Integrator();

  /**
   * @brief Delete dynamically allocated memory (in class member
   * SparseSystem<double> system).
   */
  void clean();

  /**
   * @brief Update integrator parameter and system matrices with model parameter
   * updates.
   *
   * @param time_step_size Time step size for 0D model
   */
  void update_params(double time_step_size);

  /**
   * @brief Perform a time step
   *
   * @param state Current state
   * @param time Current time
   * @return New state
   */
  State step(const State& state, double time);

  /**
   * @brief Get average number of nonlinear iterations in all step calls
   *
   * @return Average number of nonlinear iterations in all step calls
   *
   */
  double avg_nonlin_iter();

  /**
   * @brief Return number of Newton updates used in the last step.
   */
  int get_last_step_n_nonlin_iter() const;

  /**
   * @brief Return number of line-search trial residual evaluations.
   */
  int get_last_step_line_search_trials() const;

  /**
   * @brief Return smallest accepted line-search step in the last step.
   */
  double get_last_step_min_line_search_step() const;

  /**
   * @brief Return final residual norm observed in the last step.
   */
  double get_last_step_residual_norm() const;
};

#endif  // SVZERODSOLVER_ALGEBRA_INTEGRATOR_HPP_
