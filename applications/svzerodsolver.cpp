// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the
// University of California, and others. SPDX-License-Identifier: BSD-3-Clause
/**
 * @file svzerodsolver.cpp
 * @brief Main routine of svZeroDSolver
 */
#include <fstream>

#include "Solver.h"
#if __has_include(<mpi.h>)
#include <mpi.h>
#endif
#include "StreamingConfigLoader.h"

/**
 *
 * @brief svZeroDSolver main routine
 *
 * This is the main routine of the svZeroDSolver. It exectutes the following
 * steps:
 *
 * 1. Read the input file
 * 2. Create the 0D model
 * 3. (Optional) Solve for steady initial condition
 * 4. Run simulation
 * 5. Write output to file
 *
 * @param argc Number of command line arguments
 * @param argv Command line arguments
 * @return Return code
 */
int main(int argc, char* argv[]) {
  DEBUG_MSG("Starting svZeroDSolver");

#if defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES) && defined(MPI_VERSION)
  // Ensure MPI is initialized so that PETSc can use it and ranks can
  // coordinate before PETSc is first touched.
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) {
    DEBUG_MSG("svzerodsolver - calling MPI_Init");
    MPI_Init(&argc, &argv);
  }
#endif

  // Get input and output file name
  if (argc < 2 || argc > 3) {
    throw std::runtime_error(
        "Usage: svzerodsolver path/to/config.json "
        "[optional:path/to/output.csv]");
  }

  std::string input_file_name = argv[1];
  std::string output_file_path;
  std::string output_file_name;

  if (argc == 3) {
    output_file_name = argv[2];

  } else {
    // If output file is not provided, default is <path to .json>+"output.csv"
    std::size_t end_of_path = input_file_name.rfind("/");

    if (end_of_path == std::string::npos) {
      end_of_path = input_file_name.rfind("\\");  // For Windows paths (?)

      // If <path to .json> is still not found, use current directory
      if (end_of_path == std::string::npos) {
        output_file_path = ".";
      }
    } else {
      output_file_path = input_file_name.substr(0, end_of_path);
    }

    output_file_name = output_file_path + "/output.csv";
  }

  bool is_root = true;
#if defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES) && defined(MPI_VERSION)
  // Determine if this rank is the root rank (rank 0 in MPI_COMM_WORLD).
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  is_root = (world_rank == 0);
  DEBUG_MSG("svzerodsolver - MPI rank " << world_rank
                                        << " (is_root=" << (is_root ? "true" : "false") << ")");
#endif
  std::ifstream input_file(input_file_name);

  if (!input_file.is_open()) {
    std::cerr << "[svzerodsolver] Error: The input file '" << input_file_name
              << "' cannot be opened." << std::endl;
    return 1;
  }

  SimulationParameters simparams;
  std::shared_ptr<Model> model;
  State initial_state;
  if (is_root) {
    DEBUG_MSG("svzerodsolver - root streaming JSON config");
    model = std::make_shared<Model>();
    try {
      load_config_streaming(input_file, simparams, *model, initial_state);
    } catch (const std::exception& e) {
      std::cout << "[svzerodsolver] Error: Streaming parse of input file '"
                << input_file_name << "' has failed." << std::endl;
      std::cout << "[svzerodsolver] Details: " << e.what() << std::endl;
      return 1;
    }
  }
  auto solver = Solver(simparams, model, initial_state, is_root);

#ifdef SVZERODSOLVER_LINEAR_SOLVER_NAME
  DEBUG_MSG("Using linear solver: " << SVZERODSOLVER_LINEAR_SOLVER_NAME);
#else
  DEBUG_MSG("Using linear solver: <unknown>");
#endif

  solver.run();

  auto detect_mpi_rank = []() -> std::pair<bool, int> {
    int rank = 0;
    bool active = false;
#if defined(MPI_VERSION)
    int mpi_init = 0;
    MPI_Initialized(&mpi_init);
    if (mpi_init) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      active = true;
    }
#endif
    if (!active) {
      if (const char* env_rank = std::getenv("OMPI_COMM_WORLD_RANK")) {
        rank = std::atoi(env_rank);
        active = true;
      } else if (const char* env_rank = std::getenv("PMI_RANK")) {
        rank = std::atoi(env_rank);
        active = true;
      } else if (const char* env_rank = std::getenv("MPI_RANK")) {
        rank = std::atoi(env_rank);
        active = true;
      }
    }
    return {active, rank};
  };

  auto [mpi_active, out_rank] = detect_mpi_rank();
  if (!mpi_active || out_rank == 0) {
    std::cout << "[svzerodsolver] Output will be written to '"
              << output_file_name << "'." << std::endl;
    solver.write_result_to_csv(output_file_name);
  }

  return 0;
}
