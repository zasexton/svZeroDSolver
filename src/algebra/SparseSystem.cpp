// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the
// University of California, and others. SPDX-License-Identifier: BSD-3-Clause

#include "SparseSystem.h"

// When using PETSc with mpiuni (no real MPI), we must avoid using any real MPI
// calls. Include petscconf.h early to get PETSC_HAVE_MPIUNI before any code
// that might check MPI_VERSION.
#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
#if __has_include(<petscconf.h>)
#include <petscconf.h>
#endif
// Define SVZERO_HAVE_REAL_MPI only if PETSc was built with real MPI
#if !defined(PETSC_HAVE_MPIUNI)
#define SVZERO_HAVE_REAL_MPI 1
#endif
#endif

#include <cstdlib>
#include <cstring>
#include <csignal>
#include <cstddef>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>

#if __has_include(<execinfo.h>)
#define SVZERO_HAVE_EXECINFO 1
#include <execinfo.h>
#else
#define SVZERO_HAVE_EXECINFO 0
#endif

// Check for GNU extension to disable floating-point exceptions.
// The fedisableexcept() function is a GNU extension available on Linux
// that allows us to disable FP exception traps at the system level.
#if defined(__linux__)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <fenv.h>
// Check if fedisableexcept is available (it's declared when _GNU_SOURCE is set)
#if defined(__GLIBC__) || defined(FE_ALL_EXCEPT)
#define SVZERO_HAVE_FEDISABLEEXCEPT 1
#endif
#else
// On non-Linux systems, include standard fenv for feclearexcept
#include <cfenv>
#define SVZERO_HAVE_FEDISABLEEXCEPT 0
#endif

#include "SvzeroDebug.h"
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
static PetscLogStage stage_analyze = 0;
static PetscLogStage stage_factorize = 0;
static PetscLogStage stage_solve = 0;

// Global debug state used to report where a floating-point exception occurred.
volatile std::size_t svzero_current_block_index =
    static_cast<std::size_t>(-1);
volatile int svzero_current_phase = SVZERO_PHASE_NONE;
volatile double svzero_current_time = 0.0;

namespace {

#if SVZERO_HAVE_EXECINFO
// Simple SIGFPE handler for debug builds that prints a backtrace to stderr.
// This is useful on batch HPC systems where interactive debuggers are not
// available and only log output can be inspected.
void svzero_sigfpe_handler(int sig) {
  int rank = -1;
#if defined(SVZERO_HAVE_REAL_MPI)
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized) {
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  }
#endif

  // Report the current phase and last block index being processed.
  // Using fprintf/backtrace here is not strictly async-signal-safe, but is
  // acceptable for debugging on batch systems where the process is about to abort.
  const char* phase_str = svzero_phase_to_string(svzero_current_phase);

  std::fprintf(stderr,
               "====================================================================\n"
               "svZeroDSolver SIGFPE (Floating Point Exception)\n"
               "  Rank:        %d\n"
               "  Time:        %g\n"
               "  Phase:       %s (code=%d)\n"
               "  Block Index: %zu\n"
               "====================================================================\n",
               rank,
               svzero_current_time,
               phase_str,
               svzero_current_phase,
               svzero_current_block_index);
  std::fflush(stderr);

  void* frames[64];
  int n = backtrace(frames, 64);
  backtrace_symbols_fd(frames, n, 2);  // 2 = stderr
  std::fflush(stderr);
  std::signal(sig, SIG_DFL);
  raise(sig);
}
#endif

// Return true on the rank that owns the global Eigen system (rank 0 in
// PETSC_COMM_WORLD). If MPI is not yet initialized, treat the caller as root.
// When using mpiuni (no real MPI), always return true since there's only one rank.
bool is_root_rank() {
#if defined(SVZERO_HAVE_REAL_MPI)
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  int rank = 0;
  if (mpi_initialized) {
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  }
  return rank == 0;
#else
  return true;  // mpiuni mode: always root
#endif
}

// Ensure PETSc is initialized once per process.
void ensure_petsc_initialized() {
  static bool initialized = false;
  static bool registered_finalize = false;
  static bool log_stages_registered = false;

  // Early stderr output to trace where crashes occur
  {
    int rank = -1;
#if defined(SVZERO_HAVE_REAL_MPI)
    int mpi_init = 0;
    MPI_Initialized(&mpi_init);
    if (mpi_init) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
#endif
    std::fprintf(stderr, "[RANK %d] ensure_petsc_initialized - ENTER, already_initialized=%s\n",
                 rank, initialized ? "true" : "false");
    std::fflush(stderr);
  }

  auto finalize_petsc = []() {
    // Guard against double-finalize.
    static bool finalized_once = false;
    if (!finalized_once && initialized) {
      PetscFinalize();
      finalized_once = true;
    }
  };

  if (!initialized) {
    // Trace before any PETSc operations
    {
      int rank = -1;
#if defined(SVZERO_HAVE_REAL_MPI)
      int mpi_init = 0;
      MPI_Initialized(&mpi_init);
      if (mpi_init) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      }
#endif
      std::fprintf(stderr, "[RANK %d] ensure_petsc_initialized - about to call PetscInitialize\n", rank);
      std::fflush(stderr);
    }

    svzero_current_phase = SVZERO_PHASE_PETSC_INIT;

    // Disable system-level floating-point exceptions BEFORE PetscInitialize.
    // Some HPC systems (particularly with Intel compilers or certain SLURM
    // configurations) may have FP traps enabled at the OS level. Disabling
    // them here ensures that PETSc initialization and subsequent operations
    // don't trigger SIGFPE on non-root ranks during collective operations.
#if SVZERO_HAVE_FEDISABLEEXCEPT
    // Clear any pending FP exceptions first
    feclearexcept(FE_ALL_EXCEPT);
    // Disable all FP exception traps (GNU extension)
    fedisableexcept(FE_ALL_EXCEPT);
    DEBUG_MSG("ensure_petsc_initialized - disabled system FP exceptions");
#endif

    DEBUG_MSG("ensure_petsc_initialized - calling PetscInitialize");
    PetscErrorCode ierr = PetscInitialize(nullptr, nullptr, nullptr, nullptr);
    if (ierr) {
      throw std::runtime_error("Failed to initialize PETSc");
    }
    initialized = true;

    // Trace after PetscInitialize succeeds
    {
      int rank = 0;
#if defined(SVZERO_HAVE_REAL_MPI)
      MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
#endif
      std::fprintf(stderr, "[RANK %d] ensure_petsc_initialized - PetscInitialize complete\n", rank);
      std::fflush(stderr);
    }

    // Get rank info for debug output
    int rank = 0, comm_size = 1;
#if defined(SVZERO_HAVE_REAL_MPI)
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &comm_size);
#endif
    DEBUG_MSG_RANK("ensure_petsc_initialized - PetscInitialize complete, rank="
                   << rank << "/" << comm_size);

    // PETSc may enable floating-point traps (SIGFPE) based on the initial
    // environment and the -fp_trap option. This is useful for catching
    // numerical issues inside PETSc itself, but in large MPI runs it can
    // cause floating-point exceptions to be raised on non-root ranks during
    // internal setup, before our application code has entered a well-defined
    // phase. To make diagnostics more robust, we explicitly disable PETSc's
    // FP traps here so that invalid operations produce NaNs/Infs instead of
    // signals; our own residual/Jacobian checks will detect and report them.
    ierr = PetscSetFPTrap(PETSC_FP_TRAP_OFF);
    if (ierr) {
      throw std::runtime_error("Failed to disable PETSc FP traps");
    }

    // After PetscSetFPTrap, also ensure system-level traps are disabled again
    // in case PetscInitialize or PetscSetFPTrap re-enabled them.
#if SVZERO_HAVE_FEDISABLEEXCEPT
    feclearexcept(FE_ALL_EXCEPT);
    fedisableexcept(FE_ALL_EXCEPT);
#endif

    // In debug builds, install the PETSc traceback error handler so that
    // a full stack trace is printed whenever a PETSc error occurs.
    ierr = PetscPushErrorHandler(PetscTraceBackErrorHandler, nullptr);
    if (ierr) {
      throw std::runtime_error(
          "Failed to install PETSc traceback error handler");
    }

#ifndef NDEBUG
    // NOTE: We intentionally do NOT install a SIGFPE handler here.
    // Some HPC environments (Intel MKL, PETSc internals, etc.) may trigger
    // FP exceptions during initialization that are harmless and expected.
    // The main() function has already installed a permissive SIGFPE handler
    // that ignores these exceptions and allows execution to continue.
    // If we need to debug FP issues during solve(), we can enable a more
    // strict handler there.
    //
    // Previously this installed svzero_sigfpe_handler which would terminate
    // on any SIGFPE, but this caused crashes during PETSc collective operations
    // on non-root MPI ranks.

    // Start the default logging handler so that -log_view can safely
    // generate a summary at PetscFinalize in newer PETSc versions.
    ierr = PetscLogDefaultBegin();
    if (ierr) {
      throw std::runtime_error("Failed to start PETSc default logging");
    }

    // In debug builds, always enable KSP monitors and views to aid diagnosis
    // of convergence and performance issues.
    PetscOptionsSetValue(nullptr, "-ksp_monitor", nullptr);
    PetscOptionsSetValue(nullptr, "-ksp_view", nullptr);
    // Enable PETSc informational output as well so that setup/factorization
    // progress is reported in real time.
    PetscOptionsSetValue(nullptr, "-info", nullptr);
    // Always generate a PETSc log summary in debug builds so that timings for
    // the registered log stages and operations are available.
    PetscOptionsSetValue(nullptr, "-log_view", nullptr);
    // Enable PETSc's malloc debugging in Debug builds so that heap
    // corruption or misuse of PETSc objects is detected as early as
    // possible (at some performance cost).
    PetscOptionsSetValue(nullptr, "-malloc_debug", nullptr);
#endif

    if (!log_stages_registered) {
      PetscLogStageRegister("svzero_analyze_pattern", &stage_analyze);
      PetscLogStageRegister("svzero_factorize", &stage_factorize);
      PetscLogStageRegister("svzero_solve", &stage_solve);
      log_stages_registered = true;
      DEBUG_MSG_RANK("ensure_petsc_initialized - log stages registered, rank=" << rank);
    }

    // Ensure PETSc is finalized on clean exit.
    if (!registered_finalize) {
      std::atexit(finalize_petsc);
      registered_finalize = true;
    }

    svzero_current_phase = SVZERO_PHASE_NONE;
    std::fprintf(stderr, "[RANK %d] ensure_petsc_initialized - complete\n", rank);
    std::fflush(stderr);
    DEBUG_MSG_RANK("ensure_petsc_initialized - complete, rank=" << rank);
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
  DEBUG_MSG("PetscGMRESLinearSolver::PetscGMRESLinearSolver - created");
}

PetscGMRESLinearSolver::~PetscGMRESLinearSolver() {
  DEBUG_MSG("PetscGMRESLinearSolver::~PetscGMRESLinearSolver - destroying");
  if (scatter_to_root_ != nullptr) {
    VecScatterDestroy(&scatter_to_root_);
  }
  if (x_seq_ != nullptr) {
    VecDestroy(&x_seq_);
  }
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
  PetscLogStagePush(stage_analyze);
  const bool root = is_root_rank();

  int rank = 0, comm_size_dbg = 0;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &comm_size_dbg);
  std::fprintf(stderr, "[RANK %d] PetscGMRESLinearSolver::analyze_pattern - ENTER\n", rank);
  std::fflush(stderr);
  DEBUG_MSG_RANK("PetscGMRESLinearSolver::analyze_pattern - ENTER, rank=" << rank
                 << "/" << comm_size_dbg << ", is_root=" << (root ? "yes" : "no"));

  // Determine the global system size on the root rank and broadcast it to all
  // ranks so that PETSc can create distributed objects consistently.
  PetscInt n_global = 0;
  if (root) {
    DEBUG_MSG("PetscGMRESLinearSolver::analyze_pattern - computing per-row nnz from Eigen pattern");
    n_global = static_cast<PetscInt>(A.rows());
  }
  DEBUG_MSG_RANK("analyze_pattern - before MPI_Bcast(n_global), rank=" << rank
                 << ", local n_global=" << n_global);
  svzero_current_phase = SVZERO_PHASE_MPI_SCATTER_NNZ;
  MPI_Bcast(&n_global, 1, MPIU_INT, 0, PETSC_COMM_WORLD);
  svzero_current_phase = SVZERO_PHASE_NONE;
  n_ = n_global;
  DEBUG_MSG_RANK("analyze_pattern - after MPI_Bcast(n_global), rank=" << rank
                 << ", n_=" << n_);

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
  if (x_seq_ != nullptr) {
    VecDestroy(&x_seq_);
    x_seq_ = nullptr;
  }
  if (scatter_to_root_ != nullptr) {
    VecScatterDestroy(&scatter_to_root_);
    scatter_to_root_ = nullptr;
  }

  PetscErrorCode ierr;

  // Determine communicator size and rank.
  int comm_size = 1;
  int comm_rank = 0;
  MPI_Comm_size(PETSC_COMM_WORLD, &comm_size);
  MPI_Comm_rank(PETSC_COMM_WORLD, &comm_rank);

  // Compute a simple contiguous row/column partition [rstart_, rend_) per rank.
  // This partition is also used for the PETSc MATMPIAIJ diagonal/off-diagonal
  // split.
  const PetscInt base = (comm_size > 0) ? (n_ / comm_size) : 0;
  const PetscInt extra = (comm_size > 0) ? (n_ % comm_size) : 0;

  const PetscInt n_local =
      base + ((comm_rank < static_cast<int>(extra)) ? 1 : 0);
  if (comm_rank < static_cast<int>(extra)) {
    rstart_ = comm_rank * (base + 1);
  } else {
    rstart_ = extra * (base + 1) +
              (comm_rank - static_cast<int>(extra)) * base;
  }
  rend_ = rstart_ + n_local;

  DEBUG_MSG("PetscGMRESLinearSolver::analyze_pattern - ownership range ["
            << rstart_ << ", " << rend_ << ")");

  // Report communicator size and matrix type once (rank 0) for debugging.
  static bool reported_parallel_config = false;
  if (!reported_parallel_config) {
    if (comm_rank == 0) {
      DEBUG_MSG("PETSc GMRES using communicator size " << comm_size
                 << ", matrix type mpiaij");
    }
    reported_parallel_config = true;
  }

  // Compute exact per-row diagonal and off-diagonal nonzero counts from the
  // Eigen sparsity pattern on the root rank. This provides accurate
  // MatMPIAIJ preallocation and avoids PETSc reallocations during
  // MatSetValues calls.
  std::vector<PetscInt> diag_nnz_local;
  std::vector<PetscInt> off_nnz_local;
  diag_nnz_local.resize(static_cast<std::size_t>(n_local));
  off_nnz_local.resize(static_cast<std::size_t>(n_local));

  std::vector<PetscInt> diag_nnz_global;
  std::vector<PetscInt> off_nnz_global;

  if (root) {
    diag_nnz_global.assign(static_cast<std::size_t>(n_), 0);
    off_nnz_global.assign(static_cast<std::size_t>(n_), 0);

    // Helper to compute the owning rank for a global index using the same
    // [rstart,rend) partition defined above.
    const auto owner = [base, extra](PetscInt idx) -> PetscInt {
      if (base == 0) {
        // All rows, if any, live in the first 'extra' ranks; idx will always
        // satisfy idx < (base + 1) * extra in this case.
        return (extra > 0) ? (idx / (base + 1)) : 0;
      }
      const PetscInt threshold = (base + 1) * extra;
      if (idx < threshold) {
        return idx / (base + 1);
      }
      return extra + (idx - threshold) / base;
    };

    for (int k = 0; k < A.outerSize(); ++k) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
        const PetscInt row = static_cast<PetscInt>(it.row());
        const PetscInt col = static_cast<PetscInt>(it.col());
        const PetscInt row_owner = owner(row);
        const PetscInt col_owner = owner(col);
        if (row_owner == col_owner) {
          ++diag_nnz_global[static_cast<std::size_t>(row)];
        } else {
          ++off_nnz_global[static_cast<std::size_t>(row)];
        }
      }
    }
  }

  // Build Scatterv metadata describing the row partition so that the global
  // per-row nnz counts can be distributed to each rank's local rows.
  std::vector<int> sendcounts(static_cast<std::size_t>(comm_size));
  std::vector<int> displs(static_cast<std::size_t>(comm_size));
  for (int p = 0; p < comm_size; ++p) {
    const PetscInt n_local_p =
        base + ((p < static_cast<int>(extra)) ? 1 : 0);
    PetscInt rstart_p = 0;
    if (p < static_cast<int>(extra)) {
      rstart_p = p * (base + 1);
    } else {
      rstart_p = extra * (base + 1) +
                 (p - static_cast<int>(extra)) * base;
    }
    sendcounts[static_cast<std::size_t>(p)] =
        static_cast<int>(n_local_p);
    displs[static_cast<std::size_t>(p)] =
        static_cast<int>(rstart_p);
  }

  // Distribute the exact per-row nnz to each rank.
  DEBUG_MSG_RANK("analyze_pattern - before MPI_Scatterv(diag_nnz), rank=" << comm_rank
                 << ", n_local=" << n_local);
  svzero_current_phase = SVZERO_PHASE_MPI_SCATTER_NNZ;
  MPI_Scatterv(root ? diag_nnz_global.data() : nullptr,
               sendcounts.data(), displs.data(), MPIU_INT,
               diag_nnz_local.data(),
               static_cast<int>(n_local),
               MPIU_INT, 0, PETSC_COMM_WORLD);
  DEBUG_MSG_RANK("analyze_pattern - after MPI_Scatterv(diag_nnz), rank=" << comm_rank);

  DEBUG_MSG_RANK("analyze_pattern - before MPI_Scatterv(off_nnz), rank=" << comm_rank);
  MPI_Scatterv(root ? off_nnz_global.data() : nullptr,
               sendcounts.data(), displs.data(), MPIU_INT,
               off_nnz_local.data(),
               static_cast<int>(n_local),
               MPIU_INT, 0, PETSC_COMM_WORLD);
  svzero_current_phase = SVZERO_PHASE_NONE;
  DEBUG_MSG_RANK("analyze_pattern - after MPI_Scatterv(off_nnz), rank=" << comm_rank);

  // Create a distributed AIJ matrix and matching distributed vectors using the
  // exact per-row diagonal/off-diagonal nonzero counts for preallocation.
  DEBUG_MSG("PetscGMRESLinearSolver::analyze_pattern - creating distributed PETSc matrix and vectors");
  ierr = MatCreateAIJ(PETSC_COMM_WORLD,
                      n_local, n_local, n_, n_,
                      0, diag_nnz_local.data(),
                      0, off_nnz_local.data(),
                      &A_);
  if (ierr) {
    throw std::runtime_error("Failed to create PETSc matrix");
  }

  // Ensure strict "new nonzero" errors are disabled on the operator by
  // default, even if the user enabled them globally (e.g., via PETSC_OPTIONS).
  // Algebraic multigrid preconditioners like GAMG/HYPRE duplicate A_ and add
  // fill during setup; if these strict options propagate to those internal
  // matrices, PETSc will abort with "Inserting a new nonzero ..." errors.
  ierr = MatSetOption(A_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
  if (ierr) {
    throw std::runtime_error(
        "Failed to disable PETSc MAT_NEW_NONZERO_ALLOCATION_ERR");
  }
  ierr = MatSetOption(A_, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE);
  if (ierr) {
    throw std::runtime_error(
        "Failed to disable PETSc MAT_NEW_NONZERO_LOCATION_ERR");
  }

#ifndef NDEBUG
  bool strict_prealloc_requested = false;
  // In debug builds, PETSc can be instructed to error on any attempt to insert
  // a new structural nonzero that was not preallocated. This is useful for
  // catching sparsity-pattern bugs, but it breaks some parallel preconditioners
  // (e.g., GAMG) that legitimately add fill to internal matrices. Keep this
  // OFF by default and allow users to enable it explicitly.
  PetscBool strict_prealloc = PETSC_FALSE;
  PetscBool strict_set = PETSC_FALSE;
  PetscOptionsGetBool(nullptr, nullptr, "-svzero_strict_petsc_prealloc",
                      &strict_prealloc, &strict_set);
  strict_prealloc_requested = (strict_set && strict_prealloc);
#endif

  // Create matching distributed solution and RHS vectors.
  ierr = VecCreateMPI(PETSC_COMM_WORLD, n_local, n_, &x_);
  if (ierr) {
    throw std::runtime_error("Failed to create PETSc solution vector");
  }

  ierr = VecCreateMPI(PETSC_COMM_WORLD, n_local, n_, &b_);
  if (ierr) {
    throw std::runtime_error("Failed to create PETSc RHS vector");
  }

  std::fprintf(stderr, "[RANK %d] analyze_pattern - creating KSP\n", comm_rank);
  std::fflush(stderr);
  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp_);
  if (ierr) {
    throw std::runtime_error("Failed to create PETSc KSP");
  }

  std::fprintf(stderr, "[RANK %d] analyze_pattern - setting KSP type to GMRES\n", comm_rank);
  std::fflush(stderr);
  ierr = KSPSetType(ksp_, KSPGMRES);
  if (ierr) {
    throw std::runtime_error("Failed to set KSP type to GMRES");
  }

  PC pc;
  ierr = KSPGetPC(ksp_, &pc);
  if (ierr) {
    throw std::runtime_error("Failed to get PETSc PC");
  }

	  std::fprintf(stderr, "[RANK %d] analyze_pattern - setting up preconditioner\n", comm_rank);
	  std::fflush(stderr);

#ifdef SVZERODSOLVER_PETSC_PRECONDITIONER
	  const char* pc_opt = SVZERODSOLVER_PETSC_PRECONDITIONER;
	#ifndef NDEBUG
	  const bool pc_adds_fill = (std::strcmp(pc_opt, "gamg") == 0 ||
	                             std::strcmp(pc_opt, "hypre_BoomerAMG") == 0);
	#endif
	  use_bjacobi_with_ilu_ = false;
	  if (std::strcmp(pc_opt, "none") == 0) {
	    ierr = PCSetType(pc, PCNONE);
	  } else if (std::strcmp(pc_opt, "jacobi") == 0) {
	    ierr = PCSetType(pc, PCJACOBI);
  } else if (std::strcmp(pc_opt, "bjacobi") == 0) {
    ierr = PCSetType(pc, PCBJACOBI);
  } else if (std::strcmp(pc_opt, "ilu") == 0) {
    if (comm_size > 1) {
      // ILU is not supported on parallel AIJ matrices (mpiaij). Use block Jacobi
      // with ILU on each local block instead.
      use_bjacobi_with_ilu_ = true;
      ierr = PCSetType(pc, PCBJACOBI);
    } else {
      ierr = PCSetType(pc, PCILU);
    }
  } else if (std::strcmp(pc_opt, "gamg") == 0) {
    ierr = PCSetType(pc, PCGAMG);
  } else if (std::strcmp(pc_opt, "hypre_BoomerAMG") == 0) {
#if defined(PETSC_HAVE_HYPRE)
    ierr = PCSetType(pc, PCHYPRE);
    if (!ierr) {
      ierr = PCHYPRESetType(pc, "boomeramg");
    }
#else
    throw std::runtime_error(
        "PETSc was built without HYPRE support, but "
        "SVZERODSOLVER_ITERATIVE_PRECONDITIONER=hypre_BoomerAMG was requested. "
        "Reconfigure PETSc with HYPRE enabled (e.g., configure with "
        "--with-mpi=1 --download-hypre) or select a different preconditioner.");
#endif
  } else {
    // Default to Jacobi if an unknown option is provided.
    ierr = PCSetType(pc, PCJACOBI);
  }
	  if (ierr) {
	    throw std::runtime_error("Failed to set PETSc preconditioner type");
	  }

#ifndef NDEBUG
	  if (strict_prealloc_requested) {
	    if (pc_adds_fill) {
	      if (comm_rank == 0) {
	        std::fprintf(
	            stderr,
	            "[RANK 0] analyze_pattern - strict PETSc preallocation requested, "
	            "but selected preconditioner adds fill (GAMG/HYPRE). "
	            "Disabling strict preallocation to avoid PETSc errors.\n");
	        std::fflush(stderr);
	      }
	    } else {
	      ierr = MatSetOption(A_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE);
	      if (ierr) {
	        throw std::runtime_error(
	            "Failed to enable PETSc MAT_NEW_NONZERO_ALLOCATION_ERR");
	      }
	    }
	  }
#endif
#endif

	  // Create a sequential vector and a scatter to gather the distributed
	  // solution onto the root rank only. VecScatterCreateToZero will allocate
	  // x_seq_ only on rank 0.
  std::fprintf(stderr, "[RANK %d] analyze_pattern - creating VecScatter to root\n", comm_rank);
  std::fflush(stderr);
  ierr = VecScatterCreateToZero(x_, &scatter_to_root_, &x_seq_);
  if (ierr) {
    throw std::runtime_error("Failed to create PETSc VecScatter to root");
  }
  std::fprintf(stderr, "[RANK %d] analyze_pattern - complete\n", comm_rank);
  std::fflush(stderr);
  DEBUG_MSG("PetscGMRESLinearSolver::analyze_pattern - VecScatter to root created");
  PetscLogStagePop();
}

void PetscGMRESLinearSolver::factorize(
    const Eigen::SparseMatrix<double>& A) {
  int comm_rank = 0;
  MPI_Comm_rank(PETSC_COMM_WORLD, &comm_rank);
  std::fprintf(stderr, "[RANK %d] PetscGMRESLinearSolver::factorize - ENTER\n", comm_rank);
  std::fflush(stderr);
  DEBUG_MSG("PetscGMRESLinearSolver::factorize - begin");
  PetscLogStagePush(stage_factorize);
  if (A_ == nullptr || ksp_ == nullptr) {
    throw std::runtime_error("PETSc GMRES solver not initialized");
  }

  // The PETSc matrix has already been assembled by the SparseSystem. Here we
  // simply (re)attach it to the KSP object and configure tolerances and any
  // options. The Eigen matrix argument is ignored for the PETSc backend.
  (void)A;

  std::fprintf(stderr, "[RANK %d] factorize - KSPSetOperators\n", comm_rank);
  std::fflush(stderr);
  PetscErrorCode ierr = KSPSetOperators(ksp_, A_, A_);
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
  // NOTE: This is where command-line options like -pc_type hypre are applied.
  // If hypre BoomerAMG is requested, this can trigger the AMG setup which
  // may crash on large distributed systems if hypre isn't properly configured.
  std::fprintf(stderr, "[RANK %d] factorize - KSPSetFromOptions (applies -pc_type, etc.)\n", comm_rank);
  std::fflush(stderr);
  ierr = KSPSetFromOptions(ksp_);
  if (ierr) {
    throw std::runtime_error("Failed to apply PETSc KSP options");
  }

  // Some AMG-style preconditioners (e.g., GAMG, hypre BoomerAMG) will
  // duplicate the operator matrix and add fill during setup. If strict
  // "new nonzero" errors were enabled globally, they can propagate to those
  // internal matrices and crash the solve. Make sure these options are off
  // whenever such a preconditioner is selected via runtime options.
  {
    PC pc = nullptr;
    ierr = KSPGetPC(ksp_, &pc);
    if (ierr) {
      throw std::runtime_error("Failed to get PETSc PC after options");
    }
    PetscBool is_gamg = PETSC_FALSE;
    PetscBool is_hypre = PETSC_FALSE;
    ierr = PetscObjectTypeCompare((PetscObject)pc, PCGAMG, &is_gamg);
    if (ierr) {
      throw std::runtime_error("Failed to query PETSc PC type (GAMG)");
    }
    ierr = PetscObjectTypeCompare((PetscObject)pc, PCHYPRE, &is_hypre);
    if (ierr) {
      throw std::runtime_error("Failed to query PETSc PC type (HYPRE)");
    }
    if (is_gamg || is_hypre) {
      ierr = MatSetOption(A_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
      if (ierr) {
        throw std::runtime_error(
            "Failed to disable PETSc MAT_NEW_NONZERO_ALLOCATION_ERR for AMG PC");
      }
      ierr = MatSetOption(A_, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE);
      if (ierr) {
        throw std::runtime_error(
            "Failed to disable PETSc MAT_NEW_NONZERO_LOCATION_ERR for AMG PC");
      }
      if (comm_rank == 0) {
        std::fprintf(
            stderr,
            "[RANK 0] factorize - AMG preconditioner selected; strict PETSc "
            "new-nonzero errors disabled.\n");
        std::fflush(stderr);
      }
    }
  }

  // If ILU was requested in a parallel run, we mapped to PCBJACOBI in
  // analyze_pattern. Configure each sub-PC to use ILU(0) by default.
  if (use_bjacobi_with_ilu_) {
    PC pc;
    ierr = KSPGetPC(ksp_, &pc);
    if (ierr) {
      throw std::runtime_error("Failed to get PETSc PC for ILU setup");
    }
    PetscBool is_bjacobi = PETSC_FALSE;
    ierr = PetscObjectTypeCompare((PetscObject)pc, PCBJACOBI, &is_bjacobi);
    if (ierr) {
      throw std::runtime_error("Failed to query PETSc PC type");
    }
    if (is_bjacobi) {
      ierr = KSPSetUp(ksp_);
      if (ierr) {
        throw std::runtime_error("Failed to set up PETSc KSP for ILU sub-PCs");
      }
      PetscInt n_local_blocks = 0;
      KSP* subksps = nullptr;
      ierr = PCBJacobiGetSubKSP(pc, &n_local_blocks, nullptr, &subksps);
      if (ierr) {
        throw std::runtime_error("Failed to get PETSc BJACOBI sub-KSPs");
      }
      for (PetscInt i = 0; i < n_local_blocks; ++i) {
        PC subpc;
        ierr = KSPSetType(subksps[i], KSPPREONLY);
        if (!ierr) {
          ierr = KSPGetPC(subksps[i], &subpc);
        }
        if (!ierr) {
          ierr = PCSetType(subpc, PCILU);
        }
        if (ierr) {
          throw std::runtime_error("Failed to configure BJACOBI ILU sub-preconditioner");
        }
      }
    }
  }
  std::fprintf(stderr, "[RANK %d] factorize - complete\n", comm_rank);
  std::fflush(stderr);
  DEBUG_MSG("PetscGMRESLinearSolver::factorize - end");
  PetscLogStagePop();
}

void PetscGMRESLinearSolver::solve(
    const Eigen::Matrix<double, Eigen::Dynamic, 1>& b,
    Eigen::Matrix<double, Eigen::Dynamic, 1>& x) {
  int rank = 0;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  std::fprintf(stderr, "[RANK %d] PetscGMRESLinearSolver::solve - ENTER\n", rank);
  std::fflush(stderr);
  DEBUG_MSG_RANK("PetscGMRESLinearSolver::solve - ENTER, rank=" << rank);

  PetscLogStagePush(stage_solve);
  if (ksp_ == nullptr || x_ == nullptr || b_ == nullptr) {
    throw std::runtime_error("PETSc GMRES solver not initialized");
  }

  // The RHS has already been assembled by the SparseSystem into b_. The Eigen
  // vector argument is ignored for the PETSc backend.
  (void)b;

  // Check for NaN/Inf in the RHS vector before solving (on local portion).
  {
    const PetscScalar* b_arr = nullptr;
    PetscInt b_local_size = 0;
    VecGetLocalSize(b_, &b_local_size);
    VecGetArrayRead(b_, &b_arr);
    bool has_nonfinite = false;
    PetscInt bad_idx = -1;
    PetscScalar bad_val = 0.0;
    for (PetscInt i = 0; i < b_local_size; ++i) {
      if (!std::isfinite(static_cast<double>(b_arr[i]))) {
        has_nonfinite = true;
        bad_idx = i;
        bad_val = b_arr[i];
        break;
      }
    }
    VecRestoreArrayRead(b_, &b_arr);
    std::fprintf(stderr, "[RANK %d] solve - RHS check: local_size=%d, has_nonfinite=%s\n",
                 rank, static_cast<int>(b_local_size), has_nonfinite ? "YES" : "no");
    std::fflush(stderr);
    if (has_nonfinite) {
      std::ostringstream oss;
      oss << "[RANK " << rank << "] Non-finite value in RHS vector b_ at local index "
          << bad_idx << " (global ~" << (rstart_ + bad_idx) << "): " << bad_val
          << " before KSPSolve. time=" << svzero_current_time;
      std::cerr << oss.str() << std::endl;
      std::cerr.flush();
    }
    DEBUG_MSG_RANK("solve - RHS check complete, rank=" << rank
                   << ", local_size=" << b_local_size
                   << ", has_nonfinite=" << (has_nonfinite ? "YES" : "no"));
  }

  std::fprintf(stderr, "[RANK %d] solve - zeroing solution vector\n", rank);
  std::fflush(stderr);
  PetscErrorCode ierr = VecSet(x_, 0.0);
  if (ierr) {
    throw std::runtime_error("Failed to zero PETSc solution vector");
  }

  std::fprintf(stderr, "[RANK %d] solve - about to call KSPSolve\n", rank);
  std::fflush(stderr);
  DEBUG_MSG_RANK("solve - before KSPSolve, rank=" << rank);
  svzero_current_phase = SVZERO_PHASE_PETSC_KSP_SOLVE;
  ierr = KSPSolve(ksp_, b_, x_);
  svzero_current_phase = SVZERO_PHASE_NONE;
  std::fprintf(stderr, "[RANK %d] solve - KSPSolve returned, ierr=%d\n", rank, static_cast<int>(ierr));
  std::fflush(stderr);
  DEBUG_MSG_RANK("solve - after KSPSolve, rank=" << rank << ", ierr=" << ierr);

  if (ierr) {
    std::ostringstream oss;
    oss << "[RANK " << rank << "] PETSc KSPSolve failed with ierr=" << ierr;
    throw std::runtime_error(oss.str());
  }

  KSPConvergedReason reason;
  ierr = KSPGetConvergedReason(ksp_, &reason);
  if (ierr) {
    throw std::runtime_error("Failed to get PETSc KSP convergence reason");
  }

  PetscInt its = 0;
  PetscReal rnorm = 0.0;
  KSPGetIterationNumber(ksp_, &its);
  KSPGetResidualNorm(ksp_, &rnorm);
  DEBUG_MSG_RANK("solve - KSP result: rank=" << rank
                 << ", reason=" << static_cast<int>(reason)
                 << ", iterations=" << its
                 << ", residual_norm=" << rnorm);

  if (reason < 0) {
    std::ostringstream oss;
    oss << "PETSc GMRES solve failed. reason=" << static_cast<int>(reason)
        << ", iterations=" << static_cast<int>(its)
        << ", residual=" << static_cast<double>(rnorm);
    throw std::runtime_error(oss.str());
  }

  // Check for NaN/Inf in solution vector after solve (on local portion).
  {
    const PetscScalar* x_arr = nullptr;
    PetscInt x_local_size = 0;
    VecGetLocalSize(x_, &x_local_size);
    VecGetArrayRead(x_, &x_arr);
    bool has_nonfinite = false;
    PetscInt bad_idx = -1;
    PetscScalar bad_val = 0.0;
    for (PetscInt i = 0; i < x_local_size; ++i) {
      if (!std::isfinite(static_cast<double>(x_arr[i]))) {
        has_nonfinite = true;
        bad_idx = i;
        bad_val = x_arr[i];
        break;
      }
    }
    VecRestoreArrayRead(x_, &x_arr);
    if (has_nonfinite) {
      std::ostringstream oss;
      oss << "[RANK " << rank << "] Non-finite value in solution x_ at local index "
          << bad_idx << " (global ~" << (rstart_ + bad_idx) << "): " << bad_val
          << " after KSPSolve. time=" << svzero_current_time;
      std::cerr << oss.str() << std::endl;
      std::cerr.flush();
    }
    DEBUG_MSG_RANK("solve - solution check complete, rank=" << rank
                   << ", has_nonfinite=" << (has_nonfinite ? "YES" : "no"));
  }

  // Scatter the distributed PETSc solution onto a sequential vector on the
  // root rank only, then copy into the Eigen vector x on that rank.
  //
  // NOTE: VecScatterCreateToZero only allocates x_seq_ on the root rank.
  // On non-root ranks x_seq_ may legitimately be nullptr, but the scatter
  // operations are still collective and must be called on all ranks. We
  // therefore only require scatter_to_root_ to be non-null here.
  if (scatter_to_root_ == nullptr) {
    throw std::runtime_error("PETSc GMRES scatter to root not initialized");
  }

  DEBUG_MSG_RANK("solve - before VecScatter, rank=" << rank);
  svzero_current_phase = SVZERO_PHASE_MPI_VEC_SCATTER;
  ierr = VecScatterBegin(scatter_to_root_, x_, x_seq_, INSERT_VALUES,
                         SCATTER_FORWARD);
  if (ierr) {
    throw std::runtime_error("Failed to begin PETSc VecScatter to root");
  }
  ierr = VecScatterEnd(scatter_to_root_, x_, x_seq_, INSERT_VALUES,
                       SCATTER_FORWARD);
  svzero_current_phase = SVZERO_PHASE_NONE;
  DEBUG_MSG_RANK("solve - after VecScatter, rank=" << rank);

  if (ierr) {
    throw std::runtime_error("Failed to end PETSc VecScatter to root");
  }

  if (is_root_rank()) {
    const PetscScalar* px = nullptr;
    ierr = VecGetArrayRead(x_seq_, &px);
    if (ierr) {
      throw std::runtime_error("Failed to access PETSc sequential solution");
    }

    x.resize(static_cast<Eigen::Index>(n_));
    bool has_nonfinite_seq = false;
    for (PetscInt i = 0; i < n_; ++i) {
      x[static_cast<Eigen::Index>(i)] = static_cast<double>(px[i]);
      if (!std::isfinite(static_cast<double>(px[i]))) {
        if (!has_nonfinite_seq) {
          std::cerr << "[RANK 0] Non-finite in sequential solution at index " << i
                    << ": " << px[i] << std::endl;
          has_nonfinite_seq = true;
        }
      }
    }

    ierr = VecRestoreArrayRead(x_seq_, &px);
    if (ierr) {
      throw std::runtime_error("Failed to restore PETSc sequential solution");
    }
    DEBUG_MSG_RANK("solve - copied solution to Eigen, has_nonfinite="
                   << (has_nonfinite_seq ? "YES" : "no"));
  }
  DEBUG_MSG_RANK("PetscGMRESLinearSolver::solve - EXIT, rank=" << rank);
  PetscLogStagePop();
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

SparseSystem::SparseSystem()
    : solver(make_default_linear_solver())
#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
    , backend_(LinearBackend::PETSc)
#else
    , backend_(LinearBackend::Eigen)
#endif
{}

SparseSystem::SparseSystem(int n)
    : solver(make_default_linear_solver())
#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
    , backend_(LinearBackend::PETSc)
#else
    , backend_(LinearBackend::Eigen)
#endif
{
#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
  {
    int rank = 0;
    int mpi_init = 0;
    MPI_Initialized(&mpi_init);
    if (mpi_init) {
      MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    }
    std::cerr << "[RANK " << rank << "] SparseSystem::SparseSystem(n=" << n << ") - ENTER" << std::endl;
    std::cerr.flush();
  }
  const bool is_root = is_root_rank();
#else
  const bool is_root = true;
#endif

  if (is_root) {
    F = Eigen::SparseMatrix<double>(n, n);
    E = Eigen::SparseMatrix<double>(n, n);
    dC_dy = Eigen::SparseMatrix<double>(n, n);
    dC_dydot = Eigen::SparseMatrix<double>(n, n);
    C = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(n);

    if (backend_ == LinearBackend::Eigen) {
      jacobian = Eigen::SparseMatrix<double>(n, n);
      residual = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(n);
      dydot = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(n);
    } else {
      jacobian.resize(0, 0);
      // residual and dydot will be sized on demand during reserve() for the
      // PETSc backend (on the root rank only).
      residual.resize(0);
      dydot.resize(0);
    }
  } else {
    F.resize(0, 0);
    E.resize(0, 0);
    dC_dy.resize(0, 0);
    dC_dydot.resize(0, 0);
    C.resize(0);
    jacobian.resize(0, 0);
    residual.resize(0);
    dydot.resize(0);
  }
}

SparseSystem::~SparseSystem() {}

void SparseSystem::clean() {
  // Cannot be in destructor because dynamically allocated pointers will be lost
  // when objects are assigned from temporary objects.
  // delete solver;
}

void SparseSystem::add_F(int row, int col, double value) {
  const Eigen::Index r = static_cast<Eigen::Index>(row);
  const Eigen::Index c = static_cast<Eigen::Index>(col);
  if (use_triplets_) {
    F_triplets_.emplace_back(r, c, value);
  } else {
    F.coeffRef(r, c) = value;
  }
}

void SparseSystem::add_E(int row, int col, double value) {
  const Eigen::Index r = static_cast<Eigen::Index>(row);
  const Eigen::Index c = static_cast<Eigen::Index>(col);
  if (use_triplets_) {
    E_triplets_.emplace_back(r, c, value);
  } else {
    E.coeffRef(r, c) = value;
  }
}

void SparseSystem::add_dC_dy(int row, int col, double value) {
  const Eigen::Index r = static_cast<Eigen::Index>(row);
  const Eigen::Index c = static_cast<Eigen::Index>(col);
  if (use_triplets_) {
    dC_dy_triplets_.emplace_back(r, c, value);
  } else {
    dC_dy.coeffRef(r, c) = value;
  }
}

void SparseSystem::add_dC_dydot(int row, int col, double value) {
  const Eigen::Index r = static_cast<Eigen::Index>(row);
  const Eigen::Index c = static_cast<Eigen::Index>(col);
  if (use_triplets_) {
    dC_dydot_triplets_.emplace_back(r, c, value);
  } else {
    dC_dydot.coeffRef(r, c) = value;
  }
}

void SparseSystem::reserve(Model* model) {
#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
  const bool is_root = is_root_rank();
#else
  const bool is_root = true;
#endif

  if (is_root) {
    DEBUG_MSG("SparseSystem::reserve - begin on root");
    auto num_triplets = model->get_num_triplets();
    DEBUG_MSG("SparseSystem::reserve - triplets F=" << num_triplets.F
              << ", E=" << num_triplets.E
              << ", D=" << num_triplets.D);

    // During the initial reserve/build phase we always assemble via triplet
    // lists, regardless of the backend. This avoids costly incremental sparse
    // insertions for large systems and produces an identical sparsity pattern
    // for both Eigen- and PETSc-based solves.
    use_triplets_ = true;
    F_triplets_.clear();
    E_triplets_.clear();
    dC_dy_triplets_.clear();
    dC_dydot_triplets_.clear();
    F_triplets_.reserve(static_cast<std::size_t>(num_triplets.F));
    E_triplets_.reserve(static_cast<std::size_t>(num_triplets.E));
    dC_dy_triplets_.reserve(static_cast<std::size_t>(num_triplets.D));
    dC_dydot_triplets_.reserve(static_cast<std::size_t>(num_triplets.D));

    DEBUG_MSG("SparseSystem::reserve - calling Model::update_constant");
#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
    svzero_current_phase = SVZERO_PHASE_RESERVE_CONSTANT;
    svzero_current_block_index = static_cast<std::size_t>(-1);
#endif
    model->update_constant(*this);
    DEBUG_MSG("SparseSystem::reserve - calling Model::update_time");
#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
    svzero_current_phase = SVZERO_PHASE_RESERVE_TIME;
    svzero_current_block_index = static_cast<std::size_t>(-1);
#endif
    model->update_time(*this, 0.0);

    const Eigen::Index n =
        static_cast<Eigen::Index>(C.size());
    Eigen::Matrix<double, Eigen::Dynamic, 1> dummy_y =
        Eigen::Matrix<double, Eigen::Dynamic, 1>::Ones(n);
    Eigen::Matrix<double, Eigen::Dynamic, 1> dummy_dy =
        Eigen::Matrix<double, Eigen::Dynamic, 1>::Ones(n);

    DEBUG_MSG("SparseSystem::reserve - calling Model::update_solution");
#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
    svzero_current_phase = SVZERO_PHASE_RESERVE_SOLUTION;
    svzero_current_block_index = static_cast<std::size_t>(-1);
#endif
    model->update_solution(*this, dummy_y, dummy_dy);

    DEBUG_MSG("SparseSystem::reserve - compressing system matrices");
    if (use_triplets_) {
      // Build Eigen sparse matrices from the assembled triplet lists.
      const Eigen::Index nrows =
          static_cast<Eigen::Index>(C.size());
      const Eigen::Index ncols = nrows;

      F = Eigen::SparseMatrix<double>(nrows, ncols);
      E = Eigen::SparseMatrix<double>(nrows, ncols);
      dC_dy = Eigen::SparseMatrix<double>(nrows, ncols);
      dC_dydot = Eigen::SparseMatrix<double>(nrows, ncols);

      F.reserve(num_triplets.F);
      E.reserve(num_triplets.E);
      dC_dy.reserve(num_triplets.D);
      dC_dydot.reserve(num_triplets.D);

      if (!F_triplets_.empty()) {
        F.setFromTriplets(F_triplets_.begin(), F_triplets_.end());
      }
      if (!E_triplets_.empty()) {
        E.setFromTriplets(E_triplets_.begin(), E_triplets_.end());
      }
      if (!dC_dy_triplets_.empty()) {
        dC_dy.setFromTriplets(dC_dy_triplets_.begin(), dC_dy_triplets_.end());
      }
      if (!dC_dydot_triplets_.empty()) {
        dC_dydot.setFromTriplets(dC_dydot_triplets_.begin(),
                                 dC_dydot_triplets_.end());
      }

      F_triplets_.clear();
      E_triplets_.clear();
      dC_dy_triplets_.clear();
      dC_dydot_triplets_.clear();
      use_triplets_ = false;
    }

    F.makeCompressed();
    E.makeCompressed();
    dC_dy.makeCompressed();
    dC_dydot.makeCompressed();
    DEBUG_MSG("SparseSystem::reserve - nonzeros F=" << F.nonZeros()
              << ", E=" << E.nonZeros()
              << ", dC_dy=" << dC_dy.nonZeros()
              << ", dC_dydot=" << dC_dydot.nonZeros());

    if (backend_ == LinearBackend::Eigen) {
      DEBUG_MSG("SparseSystem::reserve - building Eigen jacobian pattern");
      jacobian.reserve(num_triplets.F + num_triplets.E);  // Just an estimate
      update_jacobian(1.0, 1.0);  // Update it once to have sparsity pattern
      jacobian.makeCompressed();
      // Let the solver analyze the sparsity pattern and set up any internal
      // data structures on the root rank. Non-root ranks will call
      // analyze_pattern with an empty matrix below.
      solver->analyze_pattern(jacobian);
    } else {
      DEBUG_MSG("SparseSystem::reserve - PETSc backend: initializing residual/dydot on root");
      // For PETSc we keep small Eigen vectors on the root for nonlinear
      // convergence checks and solution updates, but we do not store a full
      // Eigen Jacobian.
      residual =
          Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(C.size());
      dydot =
          Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(C.size());

      // For the PETSc backend, we still build a one-time Eigen Jacobian-style
      // sparsity pattern using the existing system matrices, but we do not
      // keep it around for repeated use.
      DEBUG_MSG("SparseSystem::reserve - building temporary Eigen jac_pattern");
      Eigen::SparseMatrix<double> jac_pattern(F.rows(), F.cols());
      jac_pattern.reserve(num_triplets.F + num_triplets.E);
      jac_pattern = (E + dC_dydot) + (F + dC_dy);
      jac_pattern.makeCompressed();
      DEBUG_MSG("SparseSystem::reserve - calling solver->analyze_pattern (PETSc)");
      solver->analyze_pattern(jac_pattern);
    }
    DEBUG_MSG("SparseSystem::reserve - end on root");
#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
    svzero_current_phase = SVZERO_PHASE_NONE;
    svzero_current_block_index = static_cast<std::size_t>(-1);
#endif
  }

  if (!is_root) {
    // Non-root ranks still need the solver to see a matrix so that it can
    // perform any necessary collective setup (e.g., PETSc matrix/vector
    // creation). For Eigen backends, provide an empty matrix with the
    // correct dimensions; for PETSc we can simply pass an empty matrix.
    Eigen::SparseMatrix<double> empty;
    if (backend_ == LinearBackend::Eigen) {
      empty.resize(jacobian.rows(), jacobian.cols());
    }
    DEBUG_MSG("SparseSystem::reserve - non-root calling solver->analyze_pattern with empty pattern");
    solver->analyze_pattern(empty);
  }
}

void SparseSystem::update_residual(
    Eigen::Matrix<double, Eigen::Dynamic, 1>& y,
    Eigen::Matrix<double, Eigen::Dynamic, 1>& ydot) {
#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
  // When using the PETSc GMRES backend, track that we are in the residual
  // assembly phase of a nonlinear step. This helps pinpoint floating-point
  // exceptions that occur during residual construction.
  svzero_current_phase = SVZERO_PHASE_STEP_RESIDUAL;
  svzero_current_block_index = static_cast<std::size_t>(-1);

  // Emit a per-rank debug message so that we can see how non-root ranks
  // participate in PETSc residual assembly when debugging MPI/PETSc issues.
  int rank = 0;
#if defined(SVZERO_HAVE_REAL_MPI)
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized) {
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  }
#endif
  DEBUG_MSG_RANK("SparseSystem::update_residual - ENTER, rank=" << rank
                 << ", backend=" << (backend_ == LinearBackend::PETSc ? "PETSc" : "Eigen")
                 << ", time=" << svzero_current_time);

  // Check input vectors for NaN/Inf on root rank
  if (is_root_rank()) {
    bool y_has_nonfinite = false;
    bool ydot_has_nonfinite = false;
    Eigen::Index bad_y_idx = -1, bad_ydot_idx = -1;
    double bad_y_val = 0.0, bad_ydot_val = 0.0;

    for (Eigen::Index i = 0; i < y.size(); ++i) {
      if (!std::isfinite(y[i])) {
        y_has_nonfinite = true;
        bad_y_idx = i;
        bad_y_val = y[i];
        break;
      }
    }
    for (Eigen::Index i = 0; i < ydot.size(); ++i) {
      if (!std::isfinite(ydot[i])) {
        ydot_has_nonfinite = true;
        bad_ydot_idx = i;
        bad_ydot_val = ydot[i];
        break;
      }
    }

    if (y_has_nonfinite) {
      std::cerr << "[FP ERROR RANK " << rank << "] Non-finite y[" << bad_y_idx
                << "]=" << bad_y_val << " in update_residual input, time="
                << svzero_current_time << std::endl;
    }
    if (ydot_has_nonfinite) {
      std::cerr << "[FP ERROR RANK " << rank << "] Non-finite ydot[" << bad_ydot_idx
                << "]=" << bad_ydot_val << " in update_residual input, time="
                << svzero_current_time << std::endl;
    }
    DEBUG_MSG_RANK("update_residual - input check: y_size=" << y.size()
                   << ", y_has_nonfinite=" << (y_has_nonfinite ? "YES" : "no")
                   << ", ydot_has_nonfinite=" << (ydot_has_nonfinite ? "YES" : "no"));
  }

  if (backend_ == LinearBackend::PETSc) {
    auto petsc_solver =
        std::dynamic_pointer_cast<PetscGMRESLinearSolver>(solver);
    if (petsc_solver) {
      Vec b = petsc_solver->get_rhs();
      if (b == nullptr) {
        throw std::runtime_error(
            "PETSc GMRES RHS vector not initialized in update_residual");
      }

      PetscErrorCode ierr = VecSet(b, 0.0);
      if (ierr) {
        throw std::runtime_error("Failed to zero PETSc RHS vector");
      }

      if (is_root_rank()) {
        const PetscInt n = static_cast<PetscInt>(C.size());

        // Maintain an Eigen residual on the root rank for nonlinear
        // convergence checks.
        if (residual.size() != C.size()) {
          residual.setZero(C.size());
        } else {
          residual.setZero();
        }

        // Contribution from the constant term: residual -= C.
        for (PetscInt i = 0; i < n; ++i) {
          const double val = -C[static_cast<Eigen::Index>(i)];
          residual[static_cast<Eigen::Index>(i)] += val;
          ierr = VecSetValue(b, i, static_cast<PetscScalar>(val),
                             ADD_VALUES);
          if (ierr) {
            throw std::runtime_error("Failed to set PETSc RHS entry (C)");
          }
        }

        // Contribution from -E * ydot.
        for (int k = 0; k < E.outerSize(); ++k) {
          for (Eigen::SparseMatrix<double>::InnerIterator it(E, k); it; ++it) {
            const PetscInt row = static_cast<PetscInt>(it.row());
            const PetscInt col = static_cast<PetscInt>(it.col());
            const double val =
                -it.value() * ydot[static_cast<Eigen::Index>(col)];
            residual[static_cast<Eigen::Index>(row)] += val;
            ierr = VecSetValue(b, row, static_cast<PetscScalar>(val),
                               ADD_VALUES);
            if (ierr) {
              throw std::runtime_error(
                  "Failed to set PETSc RHS entry (E * ydot)");
            }
          }
        }

        // Contribution from -F * y.
        for (int k = 0; k < F.outerSize(); ++k) {
          for (Eigen::SparseMatrix<double>::InnerIterator it(F, k); it; ++it) {
            const PetscInt row = static_cast<PetscInt>(it.row());
            const PetscInt col = static_cast<PetscInt>(it.col());
            const double val =
                -it.value() * y[static_cast<Eigen::Index>(col)];
            residual[static_cast<Eigen::Index>(row)] += val;
            ierr = VecSetValue(b, row, static_cast<PetscScalar>(val),
                               ADD_VALUES);
            if (ierr) {
              throw std::runtime_error(
                  "Failed to set PETSc RHS entry (F * y)");
            }
          }
        }
      }

      DEBUG_MSG_RANK("update_residual - before VecAssembly, rank=" << rank);
      svzero_current_phase = SVZERO_PHASE_MPI_VEC_ASSEMBLY;
      ierr = VecAssemblyBegin(b);
      if (ierr) {
        throw std::runtime_error("Failed to begin PETSc RHS assembly");
      }
      ierr = VecAssemblyEnd(b);
      svzero_current_phase = SVZERO_PHASE_STEP_RESIDUAL;
      DEBUG_MSG_RANK("update_residual - after VecAssembly, rank=" << rank);

      if (ierr) {
        throw std::runtime_error("Failed to end PETSc RHS assembly");
      }

      // Check local portion of PETSc vector for NaN/Inf on all ranks
      {
        const PetscScalar* b_arr = nullptr;
        PetscInt b_local_size = 0;
        VecGetLocalSize(b, &b_local_size);
        VecGetArrayRead(b, &b_arr);
        bool has_nonfinite = false;
        PetscInt bad_idx = -1;
        PetscScalar bad_val = 0.0;
        for (PetscInt i = 0; i < b_local_size; ++i) {
          if (!std::isfinite(static_cast<double>(b_arr[i]))) {
            has_nonfinite = true;
            bad_idx = i;
            bad_val = b_arr[i];
            break;
          }
        }
        VecRestoreArrayRead(b, &b_arr);
        if (has_nonfinite) {
          std::cerr << "[FP ERROR RANK " << rank << "] Non-finite in PETSc RHS at local index "
                    << bad_idx << ": " << bad_val << " after VecAssembly, time="
                    << svzero_current_time << std::endl;
          std::cerr.flush();
        }
        DEBUG_MSG_RANK("update_residual - PETSc vec check: rank=" << rank
                       << ", local_size=" << b_local_size
                       << ", has_nonfinite=" << (has_nonfinite ? "YES" : "no"));
      }

      // After assembling the RHS, check for non-finite entries in the
      // residual vector on the root rank. If any are present, throw an
      // informative error so that the offending time step can be located.
      if (is_root_rank()) {
        for (Eigen::Index i = 0; i < residual.size(); ++i) {
          const double v = residual[i];
          if (!std::isfinite(v)) {
            std::ostringstream oss;
            oss << "Non-finite entry in Eigen residual at index " << i
                << " value=" << v
                << " during PETSc residual assembly, time=" << svzero_current_time;
            std::cerr << "[FP ERROR RANK 0] " << oss.str() << std::endl;
            throw std::runtime_error(oss.str());
          }
        }
      }
      DEBUG_MSG_RANK("update_residual - EXIT (PETSc path), rank=" << rank);
      svzero_current_phase = SVZERO_PHASE_NONE;
      svzero_current_block_index = static_cast<std::size_t>(-1);
      return;
    }
  }
#endif

#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
  if (!is_root_rank()) {
    return;
  }
#endif
  residual.setZero();
  residual -= C;
  residual.noalias() -= E * ydot;
  residual.noalias() -= F * y;
}

void SparseSystem::update_jacobian(double time_coeff_ydot,
                                   double time_coeff_y) {
#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
  // Track that we are assembling the Jacobian for a nonlinear step.
  svzero_current_phase = SVZERO_PHASE_STEP_JACOBIAN;
  svzero_current_block_index = static_cast<std::size_t>(-1);

  // Emit a per-rank debug message so that we can see how non-root ranks
  // participate in PETSc Jacobian assembly when debugging MPI/PETSc issues.
  int rank = 0;
#if defined(SVZERO_HAVE_REAL_MPI)
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized) {
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  }
#endif
  DEBUG_MSG_RANK("SparseSystem::update_jacobian - ENTER, rank=" << rank
                 << ", backend=" << (backend_ == LinearBackend::PETSc ? "PETSc" : "Eigen")
                 << ", time_coeff_ydot=" << time_coeff_ydot
                 << ", time_coeff_y=" << time_coeff_y
                 << ", time=" << svzero_current_time);

  // Check coefficients for NaN/Inf
  if (!std::isfinite(time_coeff_ydot) || !std::isfinite(time_coeff_y)) {
    std::cerr << "[FP ERROR RANK " << rank << "] Non-finite Jacobian coefficients: "
              << "time_coeff_ydot=" << time_coeff_ydot
              << ", time_coeff_y=" << time_coeff_y
              << ", time=" << svzero_current_time << std::endl;
  }

  if (backend_ == LinearBackend::PETSc) {
    auto petsc_solver =
        std::dynamic_pointer_cast<PetscGMRESLinearSolver>(solver);
    if (petsc_solver) {
      Mat A = petsc_solver->get_matrix();
      if (A == nullptr) {
        throw std::runtime_error(
            "PETSc GMRES matrix not initialized in update_jacobian");
      }

      PetscErrorCode ierr = MatZeroEntries(A);
      if (ierr) {
        throw std::runtime_error("Failed to zero PETSc matrix");
      }

      if (is_root_rank()) {
        // J = (E + dC_dydot) * time_coeff_ydot
        //   + (F + dC_dy) * time_coeff_y

        // Contribution from E.
        for (int k = 0; k < E.outerSize(); ++k) {
          for (Eigen::SparseMatrix<double>::InnerIterator it(E, k); it; ++it) {
            const PetscInt row = static_cast<PetscInt>(it.row());
            const PetscInt col = static_cast<PetscInt>(it.col());
            const PetscScalar val =
                static_cast<PetscScalar>(time_coeff_ydot * it.value());
            ierr = MatSetValue(A, row, col, val, ADD_VALUES);
            if (ierr) {
              throw std::runtime_error(
                  "Failed to set PETSc matrix entry (E)");
            }
          }
        }

        // Contribution from dC_dydot.
        for (int k = 0; k < dC_dydot.outerSize(); ++k) {
          for (Eigen::SparseMatrix<double>::InnerIterator it(dC_dydot, k); it;
               ++it) {
            const PetscInt row = static_cast<PetscInt>(it.row());
            const PetscInt col = static_cast<PetscInt>(it.col());
            const PetscScalar val =
                static_cast<PetscScalar>(time_coeff_ydot * it.value());
            ierr = MatSetValue(A, row, col, val, ADD_VALUES);
            if (ierr) {
              throw std::runtime_error(
                  "Failed to set PETSc matrix entry (dC_dydot)");
            }
          }
        }

        // Contribution from F.
        for (int k = 0; k < F.outerSize(); ++k) {
          for (Eigen::SparseMatrix<double>::InnerIterator it(F, k); it; ++it) {
            const PetscInt row = static_cast<PetscInt>(it.row());
            const PetscInt col = static_cast<PetscInt>(it.col());
            const PetscScalar val =
                static_cast<PetscScalar>(time_coeff_y * it.value());
            ierr = MatSetValue(A, row, col, val, ADD_VALUES);
            if (ierr) {
              throw std::runtime_error(
                  "Failed to set PETSc matrix entry (F)");
            }
          }
        }

        // Contribution from dC_dy.
        for (int k = 0; k < dC_dy.outerSize(); ++k) {
          for (Eigen::SparseMatrix<double>::InnerIterator it(dC_dy, k); it;
               ++it) {
            const PetscInt row = static_cast<PetscInt>(it.row());
            const PetscInt col = static_cast<PetscInt>(it.col());
            const PetscScalar val =
                static_cast<PetscScalar>(time_coeff_y * it.value());
            ierr = MatSetValue(A, row, col, val, ADD_VALUES);
            if (ierr) {
              throw std::runtime_error(
                  "Failed to set PETSc matrix entry (dC_dy)");
            }
          }
        }
      }

      DEBUG_MSG_RANK("update_jacobian - before MatAssembly, rank=" << rank);
      svzero_current_phase = SVZERO_PHASE_MPI_MAT_ASSEMBLY;
      ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
      if (ierr) {
        throw std::runtime_error("Failed to begin PETSc matrix assembly");
      }
      ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
      svzero_current_phase = SVZERO_PHASE_STEP_JACOBIAN;
      DEBUG_MSG_RANK("update_jacobian - after MatAssembly, rank=" << rank);

      if (ierr) {
        throw std::runtime_error("Failed to end PETSc matrix assembly");
      }

      // Check the matrix norm for sanity (detect NaN propagation in matrix)
      {
        PetscReal mat_norm = 0.0;
        MatNorm(A, NORM_FROBENIUS, &mat_norm);
        if (!std::isfinite(static_cast<double>(mat_norm))) {
          std::cerr << "[FP ERROR RANK " << rank << "] Non-finite matrix norm: "
                    << mat_norm << " after MatAssembly, time=" << svzero_current_time
                    << std::endl;
          std::cerr.flush();
        }
        DEBUG_MSG_RANK("update_jacobian - matrix Frobenius norm=" << mat_norm
                       << ", rank=" << rank);
      }

      DEBUG_MSG_RANK("update_jacobian - EXIT (PETSc path), rank=" << rank);
      svzero_current_phase = SVZERO_PHASE_NONE;
      svzero_current_block_index = static_cast<std::size_t>(-1);
      return;
    }
  }
#endif

#if defined(SVZERODSOLVER_HAVE_PETSC) && \
    defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
  if (!is_root_rank()) {
    return;
  }
#endif
  jacobian.setZero();
  jacobian += (E + dC_dydot) * time_coeff_ydot;
  jacobian += (F + dC_dy) * time_coeff_y;
}

void SparseSystem::solve() {
#if defined(SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES)
  // Mark that we are in the linear solve phase of a nonlinear step. This
  // catches floating-point exceptions that occur inside PETSc's KSPSolve or
  // factorization routines.
  svzero_current_phase = SVZERO_PHASE_STEP_SOLVE;
  svzero_current_block_index = static_cast<std::size_t>(-1);

  // Emit a per-rank debug message so that we can see how non-root ranks
  // participate in the PETSc linear solve when debugging MPI/PETSc issues.
  int rank = 0;
#if defined(SVZERO_HAVE_REAL_MPI)
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized) {
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  }
#endif
  std::fprintf(stderr, "[RANK %d] SparseSystem::solve - ENTER, backend=%s\n",
               rank, backend_ == LinearBackend::PETSc ? "PETSc" : "Eigen");
  std::fflush(stderr);
  DEBUG_MSG("SparseSystem::solve - rank=" << rank
                                          << ", backend="
                                          << (backend_ == LinearBackend::PETSc
                                                  ? "PETSc"
                                                  : "Eigen"));
  try {
    std::fprintf(stderr, "[RANK %d] SparseSystem::solve - calling factorize\n", rank);
    std::fflush(stderr);
    solver->factorize(jacobian);
    std::fprintf(stderr, "[RANK %d] SparseSystem::solve - calling solve\n", rank);
    std::fflush(stderr);
    solver->solve(residual, dydot);
    std::fprintf(stderr, "[RANK %d] SparseSystem::solve - solve complete\n", rank);
    std::fflush(stderr);
  } catch (const std::runtime_error& e) {
    std::fprintf(stderr, "[RANK %d] SparseSystem::solve - EXCEPTION: %s\n", rank, e.what());
    std::fflush(stderr);
    svzero_current_phase = SVZERO_PHASE_NONE;
    svzero_current_block_index = static_cast<std::size_t>(-1);
    throw;
  }
  svzero_current_phase = SVZERO_PHASE_NONE;
  svzero_current_block_index = static_cast<std::size_t>(-1);
#else
  // Non-PETSc path
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
#endif  // SVZERODSOLVER_LINEAR_SOLVER_PETSC_GMRES
}
