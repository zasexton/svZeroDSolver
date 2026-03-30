// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the
// University of California, and others. SPDX-License-Identifier: BSD-3-Clause
/**
 * @file ResultHistory.h
 * @brief Lightweight storage for saved solver output states.
 */
#ifndef SVZERODSOLVER_SOLVE_RESULTHISTORY_HPP_
#define SVZERODSOLVER_SOLVE_RESULTHISTORY_HPP_

#include <algorithm>
#include <stdexcept>

#include <Eigen/Core>

#include "State.h"

/**
 * @brief Stores output snapshots without retaining full transient State objects.
 *
 * The time-marching solver only needs the current State. Saved output points
 * only require the solution vector and, when requested for output, its time
 * derivative.
 */
class ResultHistory {
 public:
  void clear() {
    y.resize(0, 0);
    ydot.resize(0, 0);
    count = 0;
    store_derivatives = false;
  }

  void reserve(int state_size, int num_states, bool save_derivatives) {
    if (state_size < 0 || num_states < 0) {
      throw std::runtime_error("ResultHistory reserve received negative sizes.");
    }

    y.resize(state_size, num_states);
    if (save_derivatives) {
      ydot.resize(state_size, num_states);
    } else {
      ydot.resize(0, 0);
    }

    count = 0;
    store_derivatives = save_derivatives;
  }

  void append(const State& state) {
    ensure_capacity(state.y.size(), count + 1);
    y.col(count) = state.y;
    if (store_derivatives) {
      ydot.col(count) = state.ydot;
    }
    count += 1;
  }

  int size() const { return count; }

  bool has_derivatives() const { return store_derivatives; }

  double value(int state_index, int dof_index) const {
    return y(dof_index, state_index);
  }

  double derivative(int state_index, int dof_index) const {
    if (!store_derivatives) {
      throw std::runtime_error(
          "Derivative history was not stored for this simulation.");
    }
    return ydot(dof_index, state_index);
  }

  Eigen::VectorXd values_for_dof(int dof_index) const {
    return y.row(dof_index).head(count).transpose();
  }

  double mean_value_for_dof(int dof_index) const {
    if (count == 0) {
      return 0.0;
    }
    return y.row(dof_index).head(count).mean();
  }

 private:
  void ensure_capacity(int state_size, int required_count) {
    if (y.rows() == 0) {
      reserve(state_size, required_count, store_derivatives);
      return;
    }

    if (state_size != y.rows()) {
      throw std::runtime_error("ResultHistory state size changed unexpectedly.");
    }

    if (required_count <= y.cols()) {
      return;
    }

    const int current_capacity = static_cast<int>(y.cols());
    int new_capacity =
        std::max(required_count, std::max(1, current_capacity * 2));
    y.conservativeResize(Eigen::NoChange, new_capacity);
    if (store_derivatives) {
      ydot.conservativeResize(Eigen::NoChange, new_capacity);
    }
  }

  Eigen::MatrixXd y;
  Eigen::MatrixXd ydot;
  int count{0};
  bool store_derivatives{false};
};

#endif  // SVZERODSOLVER_SOLVE_RESULTHISTORY_HPP_
