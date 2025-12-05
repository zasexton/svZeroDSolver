// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the
// University of California, and others. SPDX-License-Identifier: BSD-3-Clause

#include "Solver.h"

#include "csv_writer.h"

#if defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
#if __has_include(<mpi.h>)
#include <mpi.h>
#endif
#if __has_include(<petscsys.h>)
#include <petscsys.h>
#endif
#endif

Solver::Solver(const nlohmann::json& config)
    : Solver(config, []() {
#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES) && defined(MPI_VERSION)
        // Determine if this rank is root (rank 0) for MPI runs.
        // Use MPI_COMM_WORLD because PETSc isn't initialized yet.
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        if (mpi_initialized) {
          int rank = 0;
          MPI_Comm_rank(MPI_COMM_WORLD, &rank);
          return rank == 0;
        }
#endif
        return true;  // Non-MPI or MPI not initialized: assume root
      }()) {}

Solver::Solver(const nlohmann::json& config, bool is_root)
    : is_root_(is_root) {
#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES) && defined(MPI_VERSION)
  {
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    int rank = -1;
    if (mpi_initialized) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    DEBUG_MSG("Solver::Solver - begin, is_root=" << (is_root_ ? "true" : "false")
              << ", mpi_initialized=" << mpi_initialized
              << ", mpi_rank=" << rank);
  }
#else
  DEBUG_MSG("Solver::Solver - begin, is_root=" << (is_root_ ? "true" : "false"));
#endif

  if (is_root_) {
    validate_input(config);
    DEBUG_MSG("Read simulation parameters");
    simparams = load_simulation_params(config);
    DEBUG_MSG("Load model");
    this->model = std::shared_ptr<Model>(new Model());
    load_simulation_model(config, *this->model.get());
    DEBUG_MSG("Model loaded: num_blocks=" << this->model->get_num_blocks()
                                          << ", dofs=" << this->model->dofhandler.size());

    // If period isn't specified anywhere, set to 1
    if (simparams.sim_cardiac_period < 0 &&
        this->model->cardiac_cycle_period < 0) {
      this->model->cardiac_cycle_period = 1;
    } else if (this->model->cardiac_cycle_period >= 0) {
      // Check for inconsistent period definition
      if (simparams.sim_cardiac_period >= 0 &&
          (this->model->cardiac_cycle_period !=
           simparams.sim_cardiac_period)) {
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
      simparams.sim_time_step_size =
          this->model->cardiac_cycle_period /
          (double(simparams.sim_pts_per_cycle) - 1.0);
    } else {
      simparams.sim_time_step_size =
          simparams.sim_external_step_size /
          (double(simparams.sim_num_time_steps) - 1.0);
    }

    sanity_checks();
    DEBUG_MSG("Simulation parameters: pts_per_cycle="
              << simparams.sim_pts_per_cycle
              << ", num_time_steps=" << simparams.sim_num_time_steps
              << ", steady_initial=" << (simparams.sim_steady_initial ? "true" : "false")
              << ", use_cycle_to_cycle_error="
              << (simparams.use_cycle_to_cycle_error ? "true" : "false"));
  }

#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES) && defined(MPI_VERSION)
  // Broadcast simulation parameters to all ranks for PETSc GMRES runs.
  // NOTE: Use MPI_COMM_WORLD here, NOT PETSC_COMM_WORLD, because PETSc
  // hasn't been initialized yet. PETSC_COMM_WORLD is only valid after
  // PetscInitialize() is called (which happens later in the Integrator).
  MPI_Bcast(&simparams, sizeof(SimulationParameters), MPI_BYTE, 0,
            MPI_COMM_WORLD);

  // Broadcast initial state so all ranks have the correct system size.
  // This is critical: non-root ranks need to know the system size to
  // participate in MPI collective operations during Integrator setup.
  int system_size = 0;
  if (is_root_) {
    system_size = static_cast<int>(initial_state.y.size());
  }
  MPI_Bcast(&system_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  DEBUG_MSG("Solver::Solver - broadcast system_size=" << system_size);

  if (!is_root_) {
    initial_state = State(system_size);
  }
  if (system_size > 0) {
    MPI_Bcast(initial_state.y.data(), system_size, MPI_DOUBLE, 0,
              MPI_COMM_WORLD);
    MPI_Bcast(initial_state.ydot.data(), system_size, MPI_DOUBLE, 0,
              MPI_COMM_WORLD);
  }

  if (!is_root_) {
    DEBUG_MSG("Solver::Solver - non-root received simparams: pts_per_cycle="
              << simparams.sim_pts_per_cycle
              << ", num_time_steps=" << simparams.sim_num_time_steps
              << ", initial_state.y.size=" << initial_state.y.size());
  }
#endif

  DEBUG_MSG("Solver::Solver - end");
}

Solver::Solver(const SimulationParameters& simparams_in,
               std::shared_ptr<Model> model_in,
               const State& initial_state_in,
               bool is_root)
    : is_root_(is_root), model(std::move(model_in)) {
  simparams = simparams_in;
  initial_state = initial_state_in;

  // ALWAYS print this to help diagnose issues - no preprocessor guards
  std::cerr << "[Solver::Solver(streaming)] ENTER is_root=" << (is_root_ ? "true" : "false")
            << ", initial_state.y.size=" << initial_state.y.size() << std::endl;

#if defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
  int rank = 0;
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  }
  std::cerr << "[RANK " << rank << "] Solver(streaming) MPI path active, mpi_initialized="
            << mpi_initialized << std::endl;

  if (mpi_initialized) {
    // Broadcast simulation parameters to all ranks for PETSc GMRES runs.
    std::cerr << "[RANK " << rank << "] Before Bcast simparams" << std::endl;
    int mpi_err = MPI_Bcast(&simparams, sizeof(SimulationParameters), MPI_BYTE, 0,
                            MPI_COMM_WORLD);
    std::cerr << "[RANK " << rank << "] After Bcast simparams, err=" << mpi_err
              << ", time_step_size=" << simparams.sim_time_step_size << std::endl;

    // Broadcast initial state so all ranks see the same starting point.
    int system_size = 0;
    if (is_root_) {
      system_size = static_cast<int>(initial_state.y.size());
    }
    std::cerr << "[RANK " << rank << "] Before Bcast system_size, local=" << system_size << std::endl;
    mpi_err = MPI_Bcast(&system_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::cerr << "[RANK " << rank << "] After Bcast system_size=" << system_size
              << ", err=" << mpi_err << std::endl;

    if (system_size == 0) {
      std::cerr << "[RANK " << rank << "] FATAL: system_size=0 after broadcast!" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (!is_root_) {
      initial_state = State(system_size);
      std::cerr << "[RANK " << rank << "] Created State with size=" << initial_state.y.size() << std::endl;
    }

    std::cerr << "[RANK " << rank << "] Before Bcast y/ydot data" << std::endl;
    mpi_err = MPI_Bcast(initial_state.y.data(), system_size, MPI_DOUBLE, 0,
                        MPI_COMM_WORLD);
    mpi_err = MPI_Bcast(initial_state.ydot.data(), system_size, MPI_DOUBLE, 0,
                        MPI_COMM_WORLD);
    std::cerr << "[RANK " << rank << "] After Bcast y/ydot, initial_state.y.size="
              << initial_state.y.size() << std::endl;
  }
#endif

  if (is_root_ && this->model) {
    DEBUG_MSG("Solver::Solver(streaming) - root, dofs="
              << this->model->dofhandler.size());
  }

  std::cerr << "[Solver::Solver(streaming)] EXIT initial_state.y.size=" << initial_state.y.size() << std::endl;
}

void Solver::setup_initial() {
#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES) && defined(MPI_VERSION)
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  DEBUG_MSG("Solver::setup_initial - ENTER, rank=" << rank
            << ", is_root=" << (is_root_ ? "true" : "false")
            << ", initial_state.y.size=" << initial_state.y.size()
            << ", steady_initial=" << (simparams.sim_steady_initial ? "true" : "false"));
#else
  DEBUG_MSG("Solver::setup_initial - begin, steady_initial="
            << (simparams.sim_steady_initial ? "true" : "false")
            << ", is_root=" << (is_root_ ? "true" : "false"));
#endif
  state = initial_state;
#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES) && defined(MPI_VERSION)
  DEBUG_MSG("Solver::setup_initial - rank=" << rank
            << ", state.y.size after copy=" << state.y.size());
#endif

  // Create steady initial condition.
  // IMPORTANT: When using PETSc GMRES, ALL ranks must participate in Integrator
  // creation and step() calls because they contain MPI collective operations.
  if (simparams.sim_steady_initial) {
#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES) && defined(MPI_VERSION)
    // For PETSc GMRES, all ranks must participate in collective operations.
    // Root rank does actual computation; non-root ranks participate in MPI collectives.
    DEBUG_MSG("Calculate steady initial condition (PETSc parallel mode)");

    // Broadcast time_step_size_steady from root to all ranks
    double time_step_size_steady = 0.0;
    if (is_root_) {
      time_step_size_steady = this->model->cardiac_cycle_period / 10.0;
      this->model->to_steady();
    }
    // NOTE: Use MPI_COMM_WORLD here because PETSc hasn't been initialized yet.
    // PETSc initialization happens when the Integrator is created below.
    MPI_Bcast(&time_step_size_steady, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int system_size = state.y.size();
    // All ranks create Integrator - this triggers MPI collective operations
    // in analyze_pattern() that ALL ranks must participate in.
    Integrator integrator_steady(is_root_ ? this->model.get() : nullptr,
                                 system_size, time_step_size_steady,
                                 simparams.sim_rho_infty,
                                 simparams.sim_abs_tol, simparams.sim_nliter);

    // All ranks must participate in step() calls (MPI collectives inside)
    for (int i = 0; i < 31; i++) {
      state = integrator_steady.step(state, time_step_size_steady * double(i));
    }

    if (is_root_) {
      this->model->to_unsteady();
    }
#else
    // Non-PETSc path: only root rank needs to compute
    if (is_root_) {
      DEBUG_MSG("Calculate steady initial condition");
      double time_step_size_steady = this->model->cardiac_cycle_period / 10.0;
      this->model->to_steady();

      Integrator integrator_steady(this->model.get(), time_step_size_steady,
                                   simparams.sim_rho_infty,
                                   simparams.sim_abs_tol, simparams.sim_nliter);

      for (int i = 0; i < 31; i++) {
        state =
            integrator_steady.step(state, time_step_size_steady * double(i));
      }

      this->model->to_unsteady();
    }
#endif
  }

  // Use the initial condition (steady or user-provided) to set up parameters
  // which depend on the initial condition (root rank only).
  if (is_root_ && this->model) {
    DEBUG_MSG("Solver::setup_initial - calling setup_initial_state_dependent_parameters");
    this->model->setup_initial_state_dependent_parameters(state);
  }
  DEBUG_MSG("Solver::setup_initial - end");
}

void Solver::setup_integrator() {
  // Set-up integrator
  int system_size = state.y.size();
#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES) && defined(MPI_VERSION)
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  DEBUG_MSG("Solver::setup_integrator - ENTER, rank=" << rank
            << ", is_root=" << (is_root_ ? "true" : "false")
            << ", state.y.size=" << system_size
            << ", time_step_size=" << simparams.sim_time_step_size);
#else
  DEBUG_MSG("Solver::setup_integrator - begin");
#endif

#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
  integrator = Integrator(is_root_ ? this->model.get() : nullptr,
                          system_size, simparams.sim_time_step_size,
                          simparams.sim_rho_infty, simparams.sim_abs_tol,
                          simparams.sim_nliter);
#else
  integrator = Integrator(this->model.get(), simparams.sim_time_step_size,
                          simparams.sim_rho_infty, simparams.sim_abs_tol,
                          simparams.sim_nliter);
#endif
  DEBUG_MSG("Solver::setup_integrator - system_size=" << system_size
                                                      << ", time_step_size="
                                                      << simparams.sim_time_step_size);

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
}

void Solver::run_integration() {
  // Run integrator
  DEBUG_MSG("Solver::run_integration - begin, num_time_steps="
            << simparams.sim_num_time_steps
            << ", pts_per_cycle=" << simparams.sim_pts_per_cycle);
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

    state = integrator.step(state, time);

    if (simparams.use_cycle_to_cycle_error &&
        last_two_cycles_time_pt_counter > 0) {
      states_last_two_cycles[last_two_cycles_time_pt_counter] = state;
      last_two_cycles_time_pt_counter += 1;
    }

    interval_counter += 1;
    time = simparams.sim_time_step_size * double(i);

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
          state = integrator.step(state, time);

          states_last_two_cycles[last_two_cycles_time_pt_counter] = state;
          last_two_cycles_time_pt_counter += 1;
          interval_counter += 1;
          time = simparams.sim_time_step_size * double(i);

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
}

void Solver::write_result_to_csv(const std::string& filename) const {
  DEBUG_MSG("Write output");
  std::ofstream ofs(filename);
  ofs << get_full_result();
  ofs.close();
}
