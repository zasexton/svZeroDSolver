// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the
// University of California, and others. SPDX-License-Identifier: BSD-3-Clause

#include "Solver.h"

#include "csv_writer.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace {
IntegratorOptions make_integrator_options(
    const SimulationParameters& simparams) {
  IntegratorOptions options;
  options.newton_line_search = simparams.sim_newton_line_search;
  options.line_search_reduction =
      simparams.sim_newton_line_search_reduction;
  options.line_search_min_step =
      simparams.sim_newton_line_search_min_step;
  options.line_search_max_iterations =
      simparams.sim_newton_line_search_max_iterations;
  options.line_search_sufficient_decrease =
      simparams.sim_newton_line_search_sufficient_decrease;
  options.line_search_fallback_to_full_step =
      simparams.sim_newton_line_search_fallback_to_full_step;
  options.use_newton_scaling = simparams.sim_newton_use_scaling;
  options.pressure_scale = simparams.sim_newton_pressure_scale;
  options.flow_scale = simparams.sim_newton_flow_scale;
  options.volume_scale = simparams.sim_newton_volume_scale;
  options.variable_scale = simparams.sim_newton_variable_scale;
  options.residual_scale_floor = simparams.sim_newton_residual_scale_floor;
  return options;
}
}  // namespace

Solver::Solver(const nlohmann::json& config) {
  validate_input(config);
  DEBUG_MSG("Read simulation parameters");
  simparams = load_simulation_params(config);
  DEBUG_MSG("Load model");
  this->model = std::shared_ptr<Model>(new Model());
  load_simulation_model(config, *this->model.get());

  // If period isn't specified anywhere, set to 1
  if (simparams.sim_cardiac_period < 0 &&
      this->model->cardiac_cycle_period < 0) {
    this->model->cardiac_cycle_period = 1;
  } else if (this->model->cardiac_cycle_period >= 0) {
    // Check for inconsistent period definition
    if (simparams.sim_cardiac_period >= 0 &&
        (this->model->cardiac_cycle_period != simparams.sim_cardiac_period)) {
      throw std::runtime_error(
          "Inconsistent cardiac cycle period defined in parameters");
    }
    // If period is only defined in parameters, set value in model
  } else {
    this->model->cardiac_cycle_period = simparams.sim_cardiac_period;
  }
  DEBUG_MSG("Load initial condition");
  initial_state = load_initial_condition(config, *this->model.get());

  DEBUG_MSG("Cardiac cycle period " << this->model->cardiac_cycle_period);

  if (!simparams.sim_coupled && simparams.use_cycle_to_cycle_error &&
      this->model->get_has_windkessel_bc()) {
    simparams.sim_num_cycles =
        int(ceil(-1 * this->model->get_largest_windkessel_time_constant() /
                 this->model->cardiac_cycle_period *
                 log(simparams.sim_cycle_to_cycle_error)));  // equation 21 of
                                                             // Pfaller 2021
    simparams.sim_num_time_steps =
        (simparams.sim_pts_per_cycle - 1) * simparams.sim_num_cycles + 1;
  }

  // Calculate time step size
  if (!simparams.sim_coupled) {
    simparams.sim_time_step_size = this->model->cardiac_cycle_period /
                                   (double(simparams.sim_pts_per_cycle) - 1.0);
  } else {
    simparams.sim_time_step_size = simparams.sim_external_step_size /
                                   (double(simparams.sim_num_time_steps) - 1.0);
  }

  sanity_checks();
}

void Solver::setup_initial() {
  state = initial_state;

  // Create steady initial condition
  if (simparams.sim_steady_initial) {
    DEBUG_MSG("Calculate steady initial condition");
    double time_step_size_steady = this->model->cardiac_cycle_period / 10.0;
    this->model->to_steady();

    Integrator integrator_steady(this->model.get(), time_step_size_steady,
                                 simparams.sim_rho_infty, simparams.sim_abs_tol,
                                 simparams.sim_nliter,
                                 make_integrator_options(simparams));

    adaptive_time_step_size = time_step_size_steady;
    for (int i = 0; i < 31; i++) {
      const double start_time = time_step_size_steady * double(i);
      const double target_time = time_step_size_steady * double(i + 1);
      state = advance_state_to_time(integrator_steady, state, start_time,
                                    target_time, time_step_size_steady);
    }

    this->model->to_unsteady();
  }

  // Use the initial condition (steady or user-provided) to set up parameters
  // which depend on the initial condition
  this->model->setup_initial_state_dependent_parameters(state);
  DEBUG_MSG("Setup initial");
}

void Solver::setup_integrator() {
  // Set-up integrator
  DEBUG_MSG("Setup time integration");
  integrator = Integrator(this->model.get(), simparams.sim_time_step_size,
                          simparams.sim_rho_infty, simparams.sim_abs_tol,
                          simparams.sim_nliter,
                          make_integrator_options(simparams));

  // Initialize loop
  states = std::vector<State>();
  times = std::vector<double>();

  if (simparams.output_all_cycles) {
    int num_states =
        simparams.sim_num_time_steps / simparams.output_interval + 1;
    states.reserve(num_states);
    times.reserve(num_states);

  } else {
    int num_states =
        simparams.sim_pts_per_cycle / simparams.output_interval + 1;
    states.reserve(num_states);
    times.reserve(num_states);
  }
  time = 0.0;
  adaptive_time_step_size = simparams.sim_time_step_size;
}

State Solver::advance_state_to_time(Integrator& step_integrator,
                                    const State& start_state,
                                    double start_time, double target_time,
                                    double nominal_time_step) {
  const double total_time_step = target_time - start_time;
  if (total_time_step <= 0.0) {
    return start_state;
  }

  if (!simparams.sim_adaptive_time_step) {
    if (std::abs(total_time_step - nominal_time_step) >
        1.0e-12 * std::max(1.0, std::abs(nominal_time_step))) {
      step_integrator.update_params(total_time_step);
    }
    return step_integrator.step(start_state, start_time);
  }

  State current_state = start_state;
  double current_time = start_time;
  double next_time_step =
      adaptive_time_step_size > 0.0 ? adaptive_time_step_size
                                    : nominal_time_step;
  const double min_time_step =
      std::abs(nominal_time_step) * simparams.sim_adaptive_time_step_min_factor;
  const double time_tolerance =
      1.0e-12 * std::max(1.0, std::abs(target_time));

  while (current_time < target_time - time_tolerance) {
    const double remaining_time = target_time - current_time;
    double trial_time_step = std::min(next_time_step, remaining_time);
    trial_time_step = std::max(trial_time_step,
                               std::min(min_time_step, remaining_time));

    State trial_state;
    bool accepted = false;
    std::string last_error;

    for (int retry = 0; retry <= simparams.sim_adaptive_time_step_max_retries;
         retry++) {
      try {
        step_integrator.update_params(trial_time_step);
        trial_state = step_integrator.step(current_state, current_time);
        accepted = true;
        break;
      } catch (const std::exception& error) {
        last_error = error.what();
        const bool retry_limit_reached =
            retry == simparams.sim_adaptive_time_step_max_retries;
        const bool min_step_reached =
            trial_time_step <= min_time_step * (1.0 + 1.0e-12);

        if (retry_limit_reached || min_step_reached) {
          std::ostringstream message;
          message << "Adaptive time step failed between time " << current_time
                  << " and " << target_time << " with dt "
                  << trial_time_step << ". Last error: " << last_error;
          throw std::runtime_error(message.str());
        }

        trial_time_step *= simparams.sim_adaptive_time_step_reduction;
        trial_time_step = std::max(trial_time_step, min_time_step);
        trial_time_step = std::min(trial_time_step, remaining_time);
      }
    }

    if (!accepted) {
      throw std::runtime_error("Adaptive time step failed without an error.");
    }

    current_state = trial_state;
    current_time += trial_time_step;

    const bool easy_step =
        step_integrator.get_last_step_n_nonlin_iter() <=
            simparams.sim_adaptive_time_step_growth_threshold &&
        step_integrator.get_last_step_min_line_search_step() >= 0.99;

    if (easy_step) {
      next_time_step = std::min(nominal_time_step,
                                trial_time_step *
                                    simparams.sim_adaptive_time_step_growth);
    } else {
      next_time_step = std::min(nominal_time_step, trial_time_step);
    }
    adaptive_time_step_size = next_time_step;
  }

  return current_state;
}

void Solver::run_integration() {
  // Run integrator
  DEBUG_MSG("Run time integration");
  int interval_counter = 0;
  int start_last_cycle =
      simparams.sim_num_time_steps - simparams.sim_pts_per_cycle;

  if (simparams.output_all_cycles || (0 >= start_last_cycle)) {
    times.push_back(time);
    states.push_back(std::move(state));
    DEBUG_MSG("Added initial state and time");
  }

  int num_time_pts_in_two_cycles;
  std::vector<State> states_last_two_cycles;
  int last_two_cycles_time_pt_counter = 0;

  if (simparams.use_cycle_to_cycle_error) {
    num_time_pts_in_two_cycles = 2 * (simparams.sim_pts_per_cycle - 1) + 1;
    states_last_two_cycles =
        std::vector<State>(num_time_pts_in_two_cycles, state);
    DEBUG_MSG("Initialized cycle to cycle error tracking with "
              << num_time_pts_in_two_cycles << " points");
  }

  for (int i = 1; i < simparams.sim_num_time_steps; i++) {
    if (simparams.use_cycle_to_cycle_error) {
      if (i == simparams.sim_num_time_steps - num_time_pts_in_two_cycles + 1) {
        states_last_two_cycles[last_two_cycles_time_pt_counter] = state;
        last_two_cycles_time_pt_counter += 1;
      }
    }

    const double target_time = simparams.sim_time_step_size * double(i);
    state = advance_state_to_time(integrator, state, time, target_time,
                                  simparams.sim_time_step_size);
    time = target_time;

    if (simparams.use_cycle_to_cycle_error &&
        last_two_cycles_time_pt_counter > 0) {
      states_last_two_cycles[last_two_cycles_time_pt_counter] = state;
      last_two_cycles_time_pt_counter += 1;
    }

    interval_counter += 1;

    if ((interval_counter == simparams.output_interval) ||
        (!simparams.output_all_cycles && (i == start_last_cycle))) {
      if (simparams.output_all_cycles || (i >= start_last_cycle)) {
        times.push_back(time);
        states.push_back(std::move(state));
      }
      interval_counter = 0;
    }
  }

  if (simparams.use_cycle_to_cycle_error) {
    std::vector<std::pair<int, int>> vessel_caps_dof_indices =
        get_vessel_caps_dof_indices();

    if (!(this->model->get_has_windkessel_bc())) {
      assert(last_two_cycles_time_pt_counter == num_time_pts_in_two_cycles);
      double converged = check_vessel_cap_convergence(states_last_two_cycles,
                                                      vessel_caps_dof_indices);
      int extra_num_cycles = 0;

      while (!converged) {
        std::rotate(
            states_last_two_cycles.begin(),
            states_last_two_cycles.begin() + simparams.sim_pts_per_cycle - 1,
            states_last_two_cycles.end());

        last_two_cycles_time_pt_counter = simparams.sim_pts_per_cycle;
        for (size_t i = 1; i < simparams.sim_pts_per_cycle; i++) {
          const double target_time = time + simparams.sim_time_step_size;
          state = advance_state_to_time(integrator, state, time, target_time,
                                        simparams.sim_time_step_size);
          time = target_time;

          states_last_two_cycles[last_two_cycles_time_pt_counter] = state;
          last_two_cycles_time_pt_counter += 1;
          interval_counter += 1;

          if ((interval_counter == simparams.output_interval) ||
              (!simparams.output_all_cycles && (i == start_last_cycle))) {
            if (simparams.output_all_cycles || (i >= start_last_cycle)) {
              times.push_back(time);
              states.push_back(std::move(state));
            }
            interval_counter = 0;
          }
        }
        extra_num_cycles++;

        converged = check_vessel_cap_convergence(states_last_two_cycles,
                                                 vessel_caps_dof_indices);

        assert(last_two_cycles_time_pt_counter == num_time_pts_in_two_cycles);
      }
      std::cout << "Ran simulation for " << extra_num_cycles
                << " more cycles to converge flow and pressures at caps"
                << std::endl;
    } else {
      for (const std::pair<int, int>& dof_indices : vessel_caps_dof_indices) {
        std::pair<double, double> cycle_to_cycle_errors_in_flow_and_pressure =
            get_cycle_to_cycle_errors_in_flow_and_pressure(
                states_last_two_cycles, dof_indices);

        double cycle_to_cycle_error_flow =
            cycle_to_cycle_errors_in_flow_and_pressure.first;
        double cycle_to_cycle_error_pressure =
            cycle_to_cycle_errors_in_flow_and_pressure.second;
        std::cout << "Percent error between last two simulated cardiac cycles "
                     "for dof index "
                  << dof_indices.first
                  << " (mean flow)    : " << cycle_to_cycle_error_flow * 100.0
                  << std::endl;
        std::cout << "Percent error between last two simulated cardiac cycles "
                     "for dof index "
                  << dof_indices.second << " (mean pressure): "
                  << cycle_to_cycle_error_pressure * 100.0 << std::endl;
      }
    }
  }

  DEBUG_MSG("Avg. number of nonlinear iterations per time step: "
            << integrator.avg_nonlin_iter());

  // Make times start from 0
  if (!simparams.output_all_cycles) {
    double start_time = times[0];
    for (auto& time : times) {
      time -= start_time;
    }
  }
  DEBUG_MSG("Ran time integration");
}

void Solver::run() {
  setup_initial();
  setup_integrator();
  run_integration();
}

std::vector<std::pair<int, int>> Solver::get_vessel_caps_dof_indices() {
  std::vector<std::pair<int, int>> vessel_caps_dof_indices;

  for (size_t i = 0; i < this->model->get_num_blocks(); i++) {
    auto block = this->model->get_block(i);

    if (block->block_class == BlockClass::vessel) {
      if ((block->vessel_type == VesselType::inlet) ||
          (block->vessel_type == VesselType::both)) {
        int inflow_dof = block->inlet_nodes[0]->flow_dof;
        int inpres_dof = block->inlet_nodes[0]->pres_dof;
        std::pair<int, int> dofs{inflow_dof, inpres_dof};
        vessel_caps_dof_indices.push_back(dofs);
      } else if ((block->vessel_type == VesselType::outlet) ||
                 (block->vessel_type == VesselType::both)) {
        int outflow_dof = block->outlet_nodes[0]->flow_dof;
        int outpres_dof = block->outlet_nodes[0]->pres_dof;
        std::pair<int, int> dofs{outflow_dof, outpres_dof};
        vessel_caps_dof_indices.push_back(dofs);
      }
    }
  }

  return vessel_caps_dof_indices;
}

bool Solver::check_vessel_cap_convergence(
    const std::vector<State>& states_last_two_cycles,
    const std::vector<std::pair<int, int>>& vessel_caps_dof_indices) {
  double converged = true;
  for (const std::pair<int, int>& dof_indices : vessel_caps_dof_indices) {
    std::pair<double, double> cycle_to_cycle_errors_in_flow_and_pressure =
        get_cycle_to_cycle_errors_in_flow_and_pressure(states_last_two_cycles,
                                                       dof_indices);
    double cycle_to_cycle_error_flow =
        cycle_to_cycle_errors_in_flow_and_pressure.first;
    double cycle_to_cycle_error_pressure =
        cycle_to_cycle_errors_in_flow_and_pressure.second;

    if (cycle_to_cycle_error_flow > simparams.sim_cycle_to_cycle_error ||
        cycle_to_cycle_error_pressure > simparams.sim_cycle_to_cycle_error) {
      converged = false;
      break;
    }
  }

  return converged;
}

std::pair<double, double>
Solver::get_cycle_to_cycle_errors_in_flow_and_pressure(
    const std::vector<State>& states_last_two_cycles,
    const std::pair<int, int>& dof_indices) {
  double mean_flow_second_to_last_cycle = 0.0;
  double mean_pressure_second_to_last_cycle = 0.0;
  double mean_flow_last_cycle = 0.0;
  double mean_pressure_last_cycle = 0.0;

  for (size_t i = 0; i < simparams.sim_pts_per_cycle; i++) {
    mean_flow_second_to_last_cycle +=
        states_last_two_cycles[i].y[dof_indices.first];
    mean_pressure_second_to_last_cycle +=
        states_last_two_cycles[i].y[dof_indices.second];
    mean_flow_last_cycle +=
        states_last_two_cycles[simparams.sim_pts_per_cycle - 1 + i]
            .y[dof_indices.first];
    mean_pressure_last_cycle +=
        states_last_two_cycles[simparams.sim_pts_per_cycle - 1 + i]
            .y[dof_indices.second];
  }
  mean_flow_second_to_last_cycle /= simparams.sim_pts_per_cycle;
  mean_pressure_second_to_last_cycle /= simparams.sim_pts_per_cycle;
  mean_flow_last_cycle /= simparams.sim_pts_per_cycle;
  mean_pressure_last_cycle /= simparams.sim_pts_per_cycle;

  double cycle_to_cycle_error_flow =
      abs((mean_flow_last_cycle - mean_flow_second_to_last_cycle) /
          mean_flow_second_to_last_cycle);
  double cycle_to_cycle_error_pressure =
      abs((mean_pressure_last_cycle - mean_pressure_second_to_last_cycle) /
          mean_pressure_second_to_last_cycle);

  std::pair<double, double> cycle_to_cycle_errors_in_flow_and_pressure{
      cycle_to_cycle_error_flow, cycle_to_cycle_error_pressure};

  return cycle_to_cycle_errors_in_flow_and_pressure;
}

std::vector<double> Solver::get_times() const { return times; }

std::string Solver::get_full_result() const {
  std::string output;

  if (simparams.output_variable_based) {
    output = to_variable_csv(times, states, *this->model.get(),
                             simparams.output_mean_only,
                             simparams.output_derivative);

  } else {
    output =
        to_vessel_csv(times, states, *this->model.get(),
                      simparams.output_mean_only, simparams.output_derivative);
  }

  return output;
}

Eigen::VectorXd Solver::get_single_result(const std::string& dof_name) const {
  int dof_index = this->model->dofhandler.get_variable_index(dof_name);
  int num_states = states.size();
  Eigen::VectorXd result = Eigen::VectorXd::Zero(num_states);

  for (size_t i = 0; i < num_states; i++) {
    result[i] = states[i].y[dof_index];
  }

  return result;
}

double Solver::get_single_result_avg(const std::string& dof_name) const {
  int dof_index = this->model->dofhandler.get_variable_index(dof_name);
  int num_states = states.size();
  Eigen::VectorXd result = Eigen::VectorXd::Zero(num_states);

  for (size_t i = 0; i < num_states; i++) {
    result[i] = states[i].y[dof_index];
  }

  return result.mean();
}

void Solver::update_block_params(const std::string& block_name,
                                 const std::vector<double>& new_params) {
  auto block = this->model->get_block(block_name);

  if (new_params.size() != block->global_param_ids.size()) {
    throw std::runtime_error(
        "New parameter vector (given size = " +
        std::to_string(new_params.size()) +
        ") does not match number of parameters of block " + block_name +
        " (required size = " + std::to_string(block->global_param_ids.size()) +
        ")");
  }

  for (size_t i = 0; i < new_params.size(); i++) {
    this->model->get_parameter(block->global_param_ids[i])
        ->update(new_params[i]);
    // parameter_values vector needs to be seperately updated for constant
    // parameters. This does not need to be done for time-dependent parameters
    // because it is handled in Model::update_time
    this->model->update_parameter_value(block->global_param_ids[i],
                                        new_params[i]);
  }
}

std::vector<double> Solver::read_block_params(const std::string& block_name) {
  auto block = this->model->get_block(block_name);
  std::vector<double> params(block->global_param_ids.size());
  for (size_t i = 0; i < block->global_param_ids.size(); i++) {
    params[i] = this->model->get_parameter_value(block->global_param_ids[i]);
  }
  return params;
}

void Solver::sanity_checks() {
  // Check that steady initial is not used with ClosedLoopHeartAndPulmonary
  if ((simparams.sim_steady_initial == true) &&
      (this->model->has_block("CLH"))) {
    std::runtime_error(
        "ERROR: Steady initial condition is not compatible with "
        "ClosedLoopHeartAndPulmonary block.");
  }

  if (simparams.sim_adaptive_time_step) {
    if (!(0.0 < simparams.sim_adaptive_time_step_min_factor &&
          simparams.sim_adaptive_time_step_min_factor <= 1.0)) {
      throw std::runtime_error(
          "adaptive_time_step_min_factor must be in the interval (0, 1].");
    }
    if (!(0.0 < simparams.sim_adaptive_time_step_reduction &&
          simparams.sim_adaptive_time_step_reduction < 1.0)) {
      throw std::runtime_error(
          "adaptive_time_step_reduction must be between 0 and 1.");
    }
    if (simparams.sim_adaptive_time_step_growth < 1.0) {
      throw std::runtime_error(
          "adaptive_time_step_growth must be greater than or equal to 1.");
    }
    if (simparams.sim_adaptive_time_step_max_retries < 0) {
      throw std::runtime_error(
          "adaptive_time_step_max_retries must be nonnegative.");
    }
    if (simparams.sim_adaptive_time_step_growth_threshold < 0) {
      throw std::runtime_error(
          "adaptive_time_step_growth_threshold must be nonnegative.");
    }
  }
}

void Solver::write_result_to_csv(const std::string& filename) const {
  DEBUG_MSG("Write output");
  std::ofstream ofs(filename);
  ofs << get_full_result();
  ofs.close();
}
