// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the
// University of California, and others. SPDX-License-Identifier: BSD-3-Clause

#include "Integrator.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <stdexcept>

Integrator::Integrator(Model* model, double time_step_size, double rho,
                       double atol, int max_iter,
                       const IntegratorOptions& options) {
  this->model = model;
  alpha_m = 0.5 * (3.0 - rho) / (1.0 + rho);
  alpha_f = 1.0 / (1.0 + rho);
  gamma = 0.5 + alpha_m - alpha_f;
  ydot_init_coeff = 1.0 - 1.0 / gamma;

  y_coeff = gamma * time_step_size;
  y_coeff_jacobian = alpha_f * y_coeff;

  size = model->dofhandler.size();
  system = SparseSystem(size);
  this->time_step_size = time_step_size;
  this->atol = atol;
  this->max_iter = max_iter;
  this->options = options;
  if (this->options.newton_line_search) {
    if (!(0.0 < this->options.line_search_reduction &&
          this->options.line_search_reduction < 1.0)) {
      throw std::runtime_error(
          "Newton line search reduction must be between 0 and 1.");
    }
    if (!(0.0 < this->options.line_search_min_step &&
          this->options.line_search_min_step <= 1.0)) {
      throw std::runtime_error(
          "Newton line search minimum step must be in the interval (0, 1].");
    }
    if (this->options.line_search_max_iterations <= 0) {
      throw std::runtime_error(
          "Newton line search maximum iterations must be positive.");
    }
    if (!(0.0 <= this->options.line_search_sufficient_decrease &&
          this->options.line_search_sufficient_decrease < 1.0)) {
      throw std::runtime_error(
          "Newton line search sufficient decrease must be in the interval "
          "[0, 1).");
    }
  }
  if (this->options.use_newton_scaling) {
    if (this->options.pressure_scale <= 0.0 ||
        this->options.flow_scale <= 0.0 ||
        this->options.volume_scale <= 0.0 ||
        this->options.variable_scale <= 0.0 ||
        this->options.residual_scale_floor <= 0.0) {
      throw std::runtime_error(
          "Newton scaling factors and residual scale floor must be positive.");
    }
  }

  y_af = Eigen::Matrix<double, Eigen::Dynamic, 1>(size);
  ydot_am = Eigen::Matrix<double, Eigen::Dynamic, 1>(size);
  y_scale = Eigen::Matrix<double, Eigen::Dynamic, 1>::Ones(size);
  ydot_scale = Eigen::Matrix<double, Eigen::Dynamic, 1>::Ones(size);
  residual_scale = Eigen::Matrix<double, Eigen::Dynamic, 1>::Ones(size);
  setup_scaling();

  // Make some memory reservations
  system.reserve(model);
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
  update_derivative_scaling();
  model->update_constant(system);
  model->update_time(system, 0.0);
}

void Integrator::setup_scaling() {
  if (!options.use_newton_scaling) {
    y_scale.setOnes();
    ydot_scale.setOnes();
    residual_scale.setOnes();
    return;
  }

  for (size_t i = 0; i < model->dofhandler.variables.size(); i++) {
    y_scale[i] = get_variable_scale(model->dofhandler.variables[i]);
  }
  update_derivative_scaling();
}

void Integrator::update_derivative_scaling() {
  if (!options.use_newton_scaling) {
    ydot_scale.setOnes();
    return;
  }

  const double derivative_time_scale =
      std::max(std::abs(time_step_size), 1.0e-15);
  ydot_scale = y_scale / derivative_time_scale;
}

double Integrator::get_variable_scale(
    const std::string& variable_name) const {
  std::string name = variable_name;
  std::transform(name.begin(), name.end(), name.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (name.find("pressure") != std::string::npos ||
      name.rfind("p_", 0) == 0 || name.rfind("p:", 0) == 0) {
    return options.pressure_scale;
  }
  if (name.find("flow") != std::string::npos ||
      name.rfind("q_", 0) == 0 || name.rfind("q:", 0) == 0) {
    return options.flow_scale;
  }
  if (name.find("volume") != std::string::npos ||
      name.rfind("v_", 0) == 0 || name.rfind("v:", 0) == 0) {
    return options.volume_scale;
  }
  return options.variable_scale;
}

void Integrator::update_intermediate_state(const State& old_state,
                                           const State& new_state) {
  ydot_am.setZero();
  y_af.setZero();
  ydot_am += old_state.ydot + (new_state.ydot - old_state.ydot) * alpha_m;
  y_af += old_state.y + (new_state.y - old_state.y) * alpha_f;
}

double Integrator::evaluate_residual(const State& old_state,
                                     const State& new_state) {
  update_intermediate_state(old_state, new_state);
  model->update_solution(system, y_af, ydot_am);
  system.update_residual(y_af, ydot_am);
  update_residual_scaling();
  return residual_norm();
}

void Integrator::update_residual_scaling() {
  if (!options.use_newton_scaling) {
    residual_scale.setOnes();
    return;
  }

  residual_scale.setConstant(options.residual_scale_floor);
  residual_scale.array() += system.C.cwiseAbs().array();

  for (int col = 0; col < system.F.outerSize(); col++) {
    const double column_scale = std::max(std::abs(y_af[col]), y_scale[col]);
    for (Eigen::SparseMatrix<double>::InnerIterator it(system.F, col); it;
         ++it) {
      residual_scale[it.row()] += std::abs(it.value()) * column_scale;
    }
  }

  for (int col = 0; col < system.E.outerSize(); col++) {
    const double column_scale =
        std::max(std::abs(ydot_am[col]), ydot_scale[col]);
    for (Eigen::SparseMatrix<double>::InnerIterator it(system.E, col); it;
         ++it) {
      residual_scale[it.row()] += std::abs(it.value()) * column_scale;
    }
  }
}

double Integrator::residual_norm() const {
  if (!options.use_newton_scaling) {
    return system.residual.cwiseAbs().maxCoeff();
  }
  return (system.residual.cwiseAbs().array() / residual_scale.array())
      .maxCoeff();
}

void Integrator::apply_newton_update(State& new_state,
                                     const State& old_state,
                                     double current_residual_norm,
                                     double time) {
  // Preserve existing post-solve timing for the undamped full Newton step.
  model->post_solve(new_state.y);

  if (!options.newton_line_search) {
    new_state.ydot += system.dydot;
    new_state.y += system.dydot * y_coeff;
    return;
  }

  const State base_state = new_state;
  const double reduction = options.line_search_reduction;
  const double min_step = options.line_search_min_step;
  const int max_line_search_iter = options.line_search_max_iterations;
  const double sufficient_decrease =
      options.line_search_sufficient_decrease;

  double step = 1.0;
  double best_residual_norm = std::numeric_limits<double>::infinity();
  double best_step = 1.0;
  State best_state = base_state;
  bool has_best_state = false;

  for (int i = 0; i < max_line_search_iter && step >= min_step; i++) {
    State trial_state = base_state;
    trial_state.ydot += step * system.dydot;
    trial_state.y += step * system.dydot * y_coeff;

    const double trial_residual_norm =
        evaluate_residual(old_state, trial_state);
    last_step_line_search_trials++;

    if (std::isfinite(trial_residual_norm) &&
        trial_residual_norm < best_residual_norm) {
      best_residual_norm = trial_residual_norm;
      best_step = step;
      best_state = trial_state;
      has_best_state = true;
    }

    if (std::isfinite(trial_residual_norm) &&
        trial_residual_norm <=
            (1.0 - sufficient_decrease * step) * current_residual_norm) {
      new_state = trial_state;
      last_step_min_line_search_step =
          std::min(last_step_min_line_search_step, step);
      return;
    }

    step *= reduction;
  }

  // A strict Armijo decrease can be too conservative for nonsmooth terms such
  // as |Q|Q near Q=0. If we found a residual-decreasing step, accept it.
  if (has_best_state && best_residual_norm < current_residual_norm) {
    new_state = best_state;
    last_step_min_line_search_step =
        std::min(last_step_min_line_search_step, best_step);
    return;
  }

  if (!options.line_search_fallback_to_full_step) {
    throw std::runtime_error("Newton line search failed to reduce residual at "
                             "time " +
                             std::to_string(time));
  }

  // Preserve legacy nonmonotone Newton behavior when no damping step improves
  // the residual.
  new_state = base_state;
  new_state.ydot += system.dydot;
  new_state.y += system.dydot * y_coeff;
}

State Integrator::step(const State& old_state, double time) {
  last_step_n_nonlin_iter = 0;
  last_step_line_search_trials = 0;
  last_step_min_line_search_step = 1.0;
  last_step_residual_norm = std::numeric_limits<double>::infinity();

  // Predictor: Constant y, consistent ydot
  State new_state = State::Zero(size);
  new_state.ydot += old_state.ydot * ydot_init_coeff;
  new_state.y += old_state.y;

  // Determine new time (evaluate terms at generalized mid-point)
  double new_time = time + alpha_f * time_step_size;

  // Evaluate time-dependent element contributions in system
  model->update_time(system, new_time);

  // Count total number of step calls
  n_iter++;

  // Non-linear Newton-Raphson iterations
  for (size_t i = 0; i < max_iter; i++) {
    double residual_norm = evaluate_residual(old_state, new_state);
    last_step_residual_norm = residual_norm;

    // Check termination criterium
    if (residual_norm < atol) {
      break;
    }

    // Abort if maximum number of non-linear iterations is reached
    else if (i == max_iter - 1) {
      throw std::runtime_error(
          "Maximum number of non-linear iterations reached at time " +
          std::to_string(time));
    }

    // Evaluate Jacobian
    system.update_jacobian(alpha_m, y_coeff_jacobian);

    // Solve system for increment in ydot
    if (options.use_newton_scaling) {
      system.solve_scaled(residual_scale, ydot_scale);
    } else {
      system.solve();
    }

    apply_newton_update(new_state, old_state, residual_norm, time);

    // Count total number of nonlinear iterations
    n_nonlin_iter++;
    last_step_n_nonlin_iter++;
  }

  return new_state;
}

double Integrator::avg_nonlin_iter() {
  return (double)n_nonlin_iter / (double)n_iter;
}

int Integrator::get_last_step_n_nonlin_iter() const {
  return last_step_n_nonlin_iter;
}

int Integrator::get_last_step_line_search_trials() const {
  return last_step_line_search_trials;
}

double Integrator::get_last_step_min_line_search_step() const {
  return last_step_min_line_search_step;
}

double Integrator::get_last_step_residual_norm() const {
  return last_step_residual_norm;
}
