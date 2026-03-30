// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the
// University of California, and others. SPDX-License-Identifier: BSD-3-Clause

#include "Solver.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "csv_writer.h"

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
  auto started = std::chrono::steady_clock::now();
  state = initial_state;

  // Create steady initial condition
  if (simparams.sim_steady_initial) {
    DEBUG_MSG("Calculate steady initial condition");
    double time_step_size_steady = this->model->cardiac_cycle_period / 10.0;
    this->model->to_steady();

    Integrator integrator_steady(this->model.get(), time_step_size_steady,
                                 simparams.sim_rho_infty, simparams.sim_abs_tol,
                                 simparams.sim_nliter,
                                 simparams.linear_solver);

    for (int i = 0; i < 31; i++) {
      state = integrator_steady.step(state, time_step_size_steady * double(i));
    }

    this->model->to_unsteady();
  }

  // Use the initial condition (steady or user-provided) to set up parameters
  // which depend on the initial condition
  this->model->setup_initial_state_dependent_parameters(state);
  performance_summary.setup_initial_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  DEBUG_MSG("Setup initial");
}

void Solver::setup_integrator() {
  auto started = std::chrono::steady_clock::now();
  // Set-up integrator
  DEBUG_MSG("Setup time integration");
  integrator = Integrator(this->model.get(), simparams.sim_time_step_size,
                          simparams.sim_rho_infty, simparams.sim_abs_tol,
                          simparams.sim_nliter, simparams.linear_solver);

  // Initialize loop
  results.clear();
  times.clear();

  int num_states = 0;
  if (simparams.output_all_cycles) {
    num_states = simparams.sim_num_time_steps / simparams.output_interval + 1;
  } else {
    num_states = simparams.sim_pts_per_cycle / simparams.output_interval + 1;
  }
  results.reserve(this->model->dofhandler.size(), num_states,
                  simparams.output_derivative);
  times.reserve(num_states);
  time = 0.0;
  performance_summary.setup_integrator_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
}

void Solver::run_integration() {
  auto started = std::chrono::steady_clock::now();
  // Run integrator
  DEBUG_MSG("Run time integration");
  int interval_counter = 0;
  int start_last_cycle =
      simparams.sim_num_time_steps - simparams.sim_pts_per_cycle;

  if (simparams.output_all_cycles || (0 >= start_last_cycle)) {
    store_output_state(state);
    DEBUG_MSG("Added initial state and time");
  }

  int num_time_pts_in_two_cycles;
  std::vector<Eigen::VectorXd> states_last_two_cycles;
  int last_two_cycles_time_pt_counter = 0;

  if (simparams.use_cycle_to_cycle_error) {
    num_time_pts_in_two_cycles = 2 * (simparams.sim_pts_per_cycle - 1) + 1;
    states_last_two_cycles =
        std::vector<Eigen::VectorXd>(num_time_pts_in_two_cycles, state.y);
    DEBUG_MSG("Initialized cycle to cycle error tracking with "
              << num_time_pts_in_two_cycles << " points");
  }

  for (int i = 1; i < simparams.sim_num_time_steps; i++) {
    if (simparams.use_cycle_to_cycle_error) {
      if (i == simparams.sim_num_time_steps - num_time_pts_in_two_cycles + 1) {
        states_last_two_cycles[last_two_cycles_time_pt_counter] = state.y;
        last_two_cycles_time_pt_counter += 1;
      }
    }

    state = integrator.step(state, time);

    if (simparams.use_cycle_to_cycle_error &&
        last_two_cycles_time_pt_counter > 0) {
      states_last_two_cycles[last_two_cycles_time_pt_counter] = state.y;
      last_two_cycles_time_pt_counter += 1;
    }

    interval_counter += 1;
    time = simparams.sim_time_step_size * double(i);

    if ((interval_counter == simparams.output_interval) ||
        (!simparams.output_all_cycles && (i == start_last_cycle))) {
      if (simparams.output_all_cycles || (i >= start_last_cycle)) {
        store_output_state(state);
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
          state = integrator.step(state, time);

          states_last_two_cycles[last_two_cycles_time_pt_counter] = state.y;
          last_two_cycles_time_pt_counter += 1;
          interval_counter += 1;
          time = simparams.sim_time_step_size * double(i);

          if ((interval_counter == simparams.output_interval) ||
              (!simparams.output_all_cycles && (i == start_last_cycle))) {
            if (simparams.output_all_cycles || (i >= start_last_cycle)) {
              store_output_state(state);
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
  performance_summary.run_integration_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  performance_summary.stored_output_states = results.size();
  performance_summary.stored_output_derivatives = results.has_derivatives();

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
  if (simparams.report_performance) {
    std::cout << get_performance_report();
  }
}

Solver::PerformanceSummary Solver::get_performance_summary() const {
  auto summary = performance_summary;
  summary.stored_output_states = results.size();
  summary.stored_output_derivatives = results.has_derivatives();
  return summary;
}

std::string Solver::get_performance_report() const {
  std::ostringstream out;
  const auto summary = get_performance_summary();
  const auto& integrator_stats = integrator.get_performance_stats();
  const auto& linear_stats = integrator.get_linear_solve_stats();

  out << std::fixed << std::setprecision(6);
  out << "[svzerodsolver] Performance summary\n";
  out << "  setup_initial_seconds: " << summary.setup_initial_seconds << "\n";
  out << "  setup_integrator_seconds: " << summary.setup_integrator_seconds
      << "\n";
  out << "  run_integration_seconds: " << summary.run_integration_seconds
      << "\n";
  out << "  stored_output_states: " << summary.stored_output_states << "\n";
  out << "  stored_output_derivatives: "
      << (summary.stored_output_derivatives ? "true" : "false") << "\n";
  out << "  avg_nonlinear_iterations: " << integrator.avg_nonlin_iter()
      << "\n";
  out << "  linear_solver_backend: " << linear_stats.backend_name << "\n";
  out << "  linear_solver_solve_calls: " << linear_stats.solve_calls << "\n";
  out << "  linear_solver_factorization_calls: "
      << linear_stats.factorization_calls << "\n";
  out << "  linear_solver_avg_iterations: "
      << (linear_stats.solve_calls > 0
              ? static_cast<double>(linear_stats.total_iterations) /
                    static_cast<double>(linear_stats.solve_calls)
              : 0.0)
      << "\n";
  out << "  linear_solver_last_iterations: " << linear_stats.last_iterations
      << "\n";
  out << "  linear_solver_last_error: " << linear_stats.last_error << "\n";
  out << "  linear_solver_max_error: " << linear_stats.max_error << "\n";
  out << "  model_update_time_seconds: "
      << integrator_stats.update_time_seconds << "\n";
  out << "  model_update_solution_seconds: "
      << integrator_stats.update_solution_seconds << "\n";
  out << "  residual_seconds: " << integrator_stats.residual_seconds << "\n";
  out << "  jacobian_seconds: " << integrator_stats.jacobian_seconds << "\n";
  out << "  linear_solve_seconds: " << integrator_stats.linear_solve_seconds
      << "\n";
  out << "  post_solve_seconds: " << integrator_stats.post_solve_seconds
      << "\n";
  return out.str();
}

void Solver::store_output_state(const State& current_state) {
  times.push_back(time);
  results.append(current_state);
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
    const std::vector<Eigen::VectorXd>& states_last_two_cycles,
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
    const std::vector<Eigen::VectorXd>& states_last_two_cycles,
    const std::pair<int, int>& dof_indices) {
  double mean_flow_second_to_last_cycle = 0.0;
  double mean_pressure_second_to_last_cycle = 0.0;
  double mean_flow_last_cycle = 0.0;
  double mean_pressure_last_cycle = 0.0;

  for (size_t i = 0; i < simparams.sim_pts_per_cycle; i++) {
    mean_flow_second_to_last_cycle += states_last_two_cycles[i][dof_indices.first];
    mean_pressure_second_to_last_cycle +=
        states_last_two_cycles[i][dof_indices.second];
    mean_flow_last_cycle +=
        states_last_two_cycles[simparams.sim_pts_per_cycle - 1 + i]
                             [dof_indices.first];
    mean_pressure_last_cycle +=
        states_last_two_cycles[simparams.sim_pts_per_cycle - 1 + i]
                             [dof_indices.second];
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
    output = to_variable_csv(times, results, *this->model.get(),
                             simparams.output_mean_only,
                             simparams.output_derivative);

  } else {
    output = to_vessel_csv(times, results, *this->model.get(),
                           simparams.output_mean_only,
                           simparams.output_derivative);
  }

  return output;
}

Eigen::VectorXd Solver::get_single_result(const std::string& dof_name) const {
  int dof_index = this->model->dofhandler.get_variable_index(dof_name);
  return results.values_for_dof(dof_index);
}

double Solver::get_single_result_avg(const std::string& dof_name) const {
  int dof_index = this->model->dofhandler.get_variable_index(dof_name);
  return results.mean_value_for_dof(dof_index);
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
}

void Solver::write_result_to_csv(const std::string& filename) const {
  DEBUG_MSG("Write output");
  std::ofstream ofs(filename);
  if (simparams.output_variable_based) {
    write_variable_csv(ofs, times, results, *this->model.get(),
                       simparams.output_mean_only,
                       simparams.output_derivative);
  } else {
    write_vessel_csv(ofs, times, results, *this->model.get(),
                     simparams.output_mean_only,
                     simparams.output_derivative);
  }
  ofs.close();
}
