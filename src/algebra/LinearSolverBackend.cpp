// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the
// University of California, and others. SPDX-License-Identifier: BSD-3-Clause

#include "LinearSolverBackend.h"

#include <unsupported/Eigen/IterativeSolvers>

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "Model.h"

namespace {

using SparseMatrix = LinearSolverBackend::SparseMatrix;
using Vector = LinearSolverBackend::Vector;

constexpr double kAbsolutePivotTolerance = 1e-18;

double pivot_tolerance(double scale) {
  return std::max(kAbsolutePivotTolerance,
                  std::numeric_limits<double>::epsilon() * 64.0 *
                      std::max(1.0, scale));
}

double require_nonzero(double value, double scale, const std::string& context) {
  if (std::abs(value) <= pivot_tolerance(scale)) {
    throw std::runtime_error(context +
                             " is numerically singular for TreeLinear.");
  }
  return value;
}

struct ScaledLinearSystem {
  SparseMatrix matrix;
  Vector row_scale;
  Vector col_scale;
};

ScaledLinearSystem scale_linear_system(const SparseMatrix& matrix) {
  ScaledLinearSystem scaled;
  scaled.row_scale = Vector::Ones(matrix.rows());
  scaled.col_scale = Vector::Ones(matrix.cols());

  Vector row_max = Vector::Zero(matrix.rows());
  for (int k = 0; k < matrix.outerSize(); ++k) {
    for (SparseMatrix::InnerIterator it(matrix, k); it; ++it) {
      row_max[it.row()] = std::max(row_max[it.row()], std::abs(it.value()));
    }
  }
  for (int i = 0; i < row_max.size(); ++i) {
    if (row_max[i] > 0.0) {
      scaled.row_scale[i] = 1.0 / row_max[i];
    }
  }

  Vector col_max = Vector::Zero(matrix.cols());
  for (int k = 0; k < matrix.outerSize(); ++k) {
    for (SparseMatrix::InnerIterator it(matrix, k); it; ++it) {
      const double scaled_value = scaled.row_scale[it.row()] * it.value();
      col_max[it.col()] = std::max(col_max[it.col()], std::abs(scaled_value));
    }
  }
  for (int i = 0; i < col_max.size(); ++i) {
    if (col_max[i] > 0.0) {
      scaled.col_scale[i] = 1.0 / col_max[i];
    }
  }

  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(matrix.nonZeros());
  for (int k = 0; k < matrix.outerSize(); ++k) {
    for (SparseMatrix::InnerIterator it(matrix, k); it; ++it) {
      triplets.emplace_back(
          it.row(), it.col(),
          scaled.row_scale[it.row()] * it.value() * scaled.col_scale[it.col()]);
    }
  }
  scaled.matrix = SparseMatrix(matrix.rows(), matrix.cols());
  scaled.matrix.setFromTriplets(triplets.begin(), triplets.end());
  return scaled;
}

template <typename SolverType>
Vector solve_scaled_system(SolverType& solver, const Vector& rhs,
                           const Vector& row_scale, const Vector& col_scale,
                           Vector& last_solution_scaled,
                           bool& has_last_solution,
                           LinearSolveStepInfo& last_step_info) {
  const Vector rhs_scaled = row_scale.cwiseProduct(rhs);

  Vector solution_scaled;
  if (has_last_solution && last_solution_scaled.size() == rhs.size()) {
    solution_scaled = solver.solveWithGuess(rhs_scaled, last_solution_scaled);
    if (solver.info() != Eigen::Success) {
      solution_scaled = solver.solve(rhs_scaled);
    }
  } else {
    solution_scaled = solver.solve(rhs_scaled);
  }

  last_step_info.iterations = static_cast<int>(solver.iterations());
  last_step_info.error = solver.error();
  last_solution_scaled = solution_scaled;
  has_last_solution = true;
  return col_scale.cwiseProduct(solution_scaled);
}

class SparseLULinearSolverBackend : public LinearSolverBackend {
 public:
  void analyze_pattern(const SparseMatrix& matrix) override {
    solver_.analyzePattern(matrix);
  }

  void factorize(const SparseMatrix& matrix) override { solver_.factorize(matrix); }

  Vector solve(const Vector& rhs) override {
    last_step_info_ = {};
    return solver_.solve(rhs);
  }

  Eigen::ComputationInfo info() const override { return solver_.info(); }

  const char* name() const override { return "SparseLU"; }

  bool is_iterative() const override { return false; }

  LinearSolveStepInfo last_step_info() const override { return last_step_info_; }

 private:
  Eigen::SparseLU<SparseMatrix> solver_;
  LinearSolveStepInfo last_step_info_;
};

class BiCGSTABILUTLinearSolverBackend : public LinearSolverBackend {
 public:
  explicit BiCGSTABILUTLinearSolverBackend(
      const LinearSolverSettings& settings)
      : settings_(settings) {
    solver_.setTolerance(settings_.tolerance);
    if (settings_.max_iterations > 0) {
      solver_.setMaxIterations(settings_.max_iterations);
    }
    solver_.preconditioner().setDroptol(settings_.ilut_drop_tolerance);
    solver_.preconditioner().setFillfactor(settings_.ilut_fill_factor);
  }

  void analyze_pattern(const SparseMatrix& matrix) override {
    solver_.analyzePattern(matrix);
  }

  void factorize(const SparseMatrix& matrix) override {
    scaled_system_ = scale_linear_system(matrix);
    solver_.factorize(scaled_system_.matrix);
    has_last_solution_ = false;
  }

  Vector solve(const Vector& rhs) override {
    return solve_scaled_system(solver_, rhs, scaled_system_.row_scale,
                               scaled_system_.col_scale, last_solution_scaled_,
                               has_last_solution_, last_step_info_);
  }

  Eigen::ComputationInfo info() const override { return solver_.info(); }

  const char* name() const override { return "BiCGSTAB_ILUT"; }

  bool is_iterative() const override { return true; }

  LinearSolveStepInfo last_step_info() const override { return last_step_info_; }

 private:
  using SolverType =
      Eigen::BiCGSTAB<SparseMatrix, Eigen::IncompleteLUT<double>>;

  LinearSolverSettings settings_;
  SolverType solver_;
  ScaledLinearSystem scaled_system_;
  Vector last_solution_scaled_;
  bool has_last_solution_{false};
  LinearSolveStepInfo last_step_info_;
};

class GMRESILUTLinearSolverBackend : public LinearSolverBackend {
 public:
  explicit GMRESILUTLinearSolverBackend(const LinearSolverSettings& settings)
      : settings_(settings) {
    solver_.setTolerance(settings_.tolerance);
    if (settings_.max_iterations > 0) {
      solver_.setMaxIterations(settings_.max_iterations);
    }
    solver_.set_restart(settings_.gmres_restart);
    solver_.preconditioner().setDroptol(settings_.ilut_drop_tolerance);
    solver_.preconditioner().setFillfactor(settings_.ilut_fill_factor);
  }

  void analyze_pattern(const SparseMatrix& matrix) override {
    solver_.analyzePattern(matrix);
  }

  void factorize(const SparseMatrix& matrix) override {
    scaled_system_ = scale_linear_system(matrix);
    solver_.factorize(scaled_system_.matrix);
    has_last_solution_ = false;
  }

  Vector solve(const Vector& rhs) override {
    return solve_scaled_system(solver_, rhs, scaled_system_.row_scale,
                               scaled_system_.col_scale, last_solution_scaled_,
                               has_last_solution_, last_step_info_);
  }

  Eigen::ComputationInfo info() const override { return solver_.info(); }

  const char* name() const override { return "GMRES_ILUT"; }

  bool is_iterative() const override { return true; }

  LinearSolveStepInfo last_step_info() const override { return last_step_info_; }

 private:
  using SolverType = Eigen::GMRES<SparseMatrix, Eigen::IncompleteLUT<double>>;

  LinearSolverSettings settings_;
  SolverType solver_;
  ScaledLinearSystem scaled_system_;
  Vector last_solution_scaled_;
  bool has_last_solution_{false};
  LinearSolveStepInfo last_step_info_;
};

class GMRESDiagonalLinearSolverBackend : public LinearSolverBackend {
 public:
  explicit GMRESDiagonalLinearSolverBackend(
      const LinearSolverSettings& settings)
      : settings_(settings) {
    solver_.setTolerance(settings_.tolerance);
    if (settings_.max_iterations > 0) {
      solver_.setMaxIterations(settings_.max_iterations);
    }
    solver_.set_restart(settings_.gmres_restart);
  }

  void analyze_pattern(const SparseMatrix& matrix) override {
    solver_.analyzePattern(matrix);
  }

  void factorize(const SparseMatrix& matrix) override {
    scaled_system_ = scale_linear_system(matrix);
    solver_.factorize(scaled_system_.matrix);
    has_last_solution_ = false;
  }

  Vector solve(const Vector& rhs) override {
    return solve_scaled_system(solver_, rhs, scaled_system_.row_scale,
                               scaled_system_.col_scale, last_solution_scaled_,
                               has_last_solution_, last_step_info_);
  }

  Eigen::ComputationInfo info() const override { return solver_.info(); }

  const char* name() const override { return "GMRES_DIAGONAL"; }

  bool is_iterative() const override { return true; }

  LinearSolveStepInfo last_step_info() const override { return last_step_info_; }

 private:
  using SolverType =
      Eigen::GMRES<SparseMatrix, Eigen::DiagonalPreconditioner<double>>;

  LinearSolverSettings settings_;
  SolverType solver_;
  ScaledLinearSystem scaled_system_;
  Vector last_solution_scaled_;
  bool has_last_solution_{false};
  LinearSolveStepInfo last_step_info_;
};

struct CondensedRelation {
  double a{0.0};
  double b{0.0};
  double c{0.0};
};

enum class TreeDownstreamType { terminal, junction };

struct TreeTerminalInfo {
  int block_id{-1};
  int row{-1};
  int p_dof{-1};
  int q_dof{-1};
  double coeff_p{0.0};
  double coeff_q{0.0};
};

struct TreeJunctionChildInfo {
  int pressure_row{-1};
  int child_p_dof{-1};
  int child_q_dof{-1};
  double pressure_coeff_parent{0.0};
  double pressure_coeff_child{0.0};
  double mass_coeff_child_q{0.0};
  int child_vessel{-1};
};

struct TreeJunctionInfo {
  int block_id{-1};
  int mass_row{-1};
  int parent_p_dof{-1};
  int parent_q_dof{-1};
  double mass_coeff_parent_q{0.0};
  std::vector<TreeJunctionChildInfo> children;
};

struct TreeVesselInfo {
  int block_id{-1};
  int row0{-1};
  int row1{-1};
  int p_in_dof{-1};
  int q_in_dof{-1};
  int p_out_dof{-1};
  int q_out_dof{-1};
  double row0_p_in{0.0};
  double row0_q_in{0.0};
  double row0_p_out{0.0};
  double row0_q_out{0.0};
  double row1_p_in{0.0};
  double row1_q_in{0.0};
  double row1_q_out{0.0};
  TreeDownstreamType downstream_type{TreeDownstreamType::terminal};
  int downstream_index{-1};
};

struct TreeRootInfo {
  int block_id{-1};
  int row{-1};
  int p_dof{-1};
  int q_dof{-1};
  double coeff_q{0.0};
  int root_vessel{-1};
};

class TreeLinearSolverBackend : public LinearSolverBackend {
 public:
  void configure_model(const Model* model) override {
    if (model == nullptr) {
      throw std::runtime_error("TreeLinear requires a valid model.");
    }

    model_ = model;
    root_ = {};
    vessels_.clear();
    junctions_.clear();
    terminals_.clear();
    vessel_index_by_block_id_.clear();
    junction_index_by_block_id_.clear();
    terminal_index_by_block_id_.clear();

    const Block* root_block = nullptr;
    int flow_block_count = 0;

    const int num_blocks = model_->get_num_blocks(false);
    for (int i = 0; i < num_blocks; ++i) {
      const Block* block = model_->get_block(i);
      switch (block->block_type) {
        case BlockType::blood_vessel:
        case BlockType::junction:
        case BlockType::resistance_bc:
          break;
        case BlockType::flow_bc:
          root_block = block;
          flow_block_count += 1;
          break;
        default:
          throw std::runtime_error(
              "TreeLinear supports only FLOW, RESISTANCE, NORMAL_JUNCTION, "
              "and BloodVessel blocks.");
      }
    }

    if (flow_block_count != 1 || root_block == nullptr) {
      throw std::runtime_error(
          "TreeLinear requires exactly one FLOW boundary condition.");
    }
    if (root_block->global_eqn_ids.size() != 1 ||
        root_block->outlet_nodes.size() != 1 || !root_block->inlet_nodes.empty()) {
      throw std::runtime_error(
          "TreeLinear requires the FLOW boundary condition to be the unique "
          "root block.");
    }

    const Node* root_node = root_block->outlet_nodes[0];
    if (root_node->outlet_eles.size() != 1 ||
        root_node->outlet_eles[0]->block_type != BlockType::blood_vessel) {
      throw std::runtime_error(
          "TreeLinear requires the FLOW boundary condition to connect to a "
          "single BloodVessel.");
    }

    root_.block_id = root_block->id;
    root_.row = root_block->global_eqn_ids[0];
    root_.p_dof = root_node->pres_dof;
    root_.q_dof = root_node->flow_dof;
    root_.root_vessel = build_vessel(root_node->outlet_eles[0]);

    int vessel_count = 0;
    int junction_count = 0;
    int terminal_count = 0;
    for (int i = 0; i < num_blocks; ++i) {
      const Block* block = model_->get_block(i);
      if (block->block_type == BlockType::blood_vessel) {
        vessel_count += 1;
      } else if (block->block_type == BlockType::junction) {
        junction_count += 1;
      } else if (block->block_type == BlockType::resistance_bc) {
        terminal_count += 1;
      }
    }

    if (static_cast<int>(vessels_.size()) != vessel_count ||
        static_cast<int>(junctions_.size()) != junction_count ||
        static_cast<int>(terminals_.size()) != terminal_count) {
      throw std::runtime_error(
          "TreeLinear requires a connected rooted tree with resistance "
          "terminals.");
    }

    vessel_relations_.resize(vessels_.size());
    vessel_relation_valid_.resize(vessels_.size());
    coefficients_ready_ = false;
    info_ = Eigen::Success;
  }

  void analyze_pattern(const SparseMatrix& matrix) override {
    if (matrix.rows() != matrix.cols()) {
      info_ = Eigen::InvalidInput;
      throw std::runtime_error("TreeLinear requires a square system matrix.");
    }
  }

  void factorize(const SparseMatrix& matrix) override {
    if (model_ == nullptr) {
      info_ = Eigen::InvalidInput;
      throw std::runtime_error("TreeLinear was not configured with a model.");
    }

    try {
      info_ = Eigen::Success;
      system_size_ = matrix.cols();
      root_.coeff_q = require_nonzero(
          matrix.coeff(root_.row, root_.q_dof), 1.0,
          "FLOW boundary-condition coefficient");

      for (auto& terminal : terminals_) {
        terminal.coeff_p = matrix.coeff(terminal.row, terminal.p_dof);
        terminal.coeff_q = matrix.coeff(terminal.row, terminal.q_dof);
      }

      for (auto& vessel : vessels_) {
        vessel.row0_p_in = matrix.coeff(vessel.row0, vessel.p_in_dof);
        vessel.row0_q_in = matrix.coeff(vessel.row0, vessel.q_in_dof);
        vessel.row0_p_out = require_nonzero(
            matrix.coeff(vessel.row0, vessel.p_out_dof), 1.0,
            "BloodVessel outlet-pressure coefficient");
        vessel.row0_q_out = matrix.coeff(vessel.row0, vessel.q_out_dof);
        vessel.row1_p_in = matrix.coeff(vessel.row1, vessel.p_in_dof);
        vessel.row1_q_in = matrix.coeff(vessel.row1, vessel.q_in_dof);
        vessel.row1_q_out = require_nonzero(
            matrix.coeff(vessel.row1, vessel.q_out_dof), 1.0,
            "BloodVessel outlet-flow coefficient");
      }

      for (auto& junction : junctions_) {
        junction.mass_coeff_parent_q = require_nonzero(
            matrix.coeff(junction.mass_row, junction.parent_q_dof), 1.0,
            "Junction inlet-flow coefficient");
        for (auto& child : junction.children) {
          child.pressure_coeff_parent = matrix.coeff(
              child.pressure_row, junction.parent_p_dof);
          child.pressure_coeff_child = require_nonzero(
              matrix.coeff(child.pressure_row, child.child_p_dof), 1.0,
              "Junction outlet-pressure coefficient");
          child.mass_coeff_child_q = matrix.coeff(
              junction.mass_row, child.child_q_dof);
        }
      }

      coefficients_ready_ = true;
      std::fill(vessel_relation_valid_.begin(), vessel_relation_valid_.end(),
                false);
    } catch (...) {
      info_ = Eigen::NumericalIssue;
      throw;
    }
  }

  Vector solve(const Vector& rhs) override {
    if (!coefficients_ready_) {
      info_ = Eigen::InvalidInput;
      throw std::runtime_error(
          "TreeLinear solve requested before factorization.");
    }

    try {
      info_ = Eigen::Success;
      last_step_info_ = {};
      std::fill(vessel_relation_valid_.begin(), vessel_relation_valid_.end(),
                false);

      Vector solution = Vector::Zero(system_size_);
      const CondensedRelation root_relation =
          compute_vessel_relation(root_.root_vessel, rhs);
      const double q_root =
          rhs[root_.row] / require_nonzero(root_.coeff_q, 1.0, "FLOW coefficient");
      const double root_scale = std::abs(root_relation.a) +
                                std::abs(root_relation.b) +
                                std::abs(root_relation.c) + 1.0;
      const double p_root = (root_relation.c - root_relation.b * q_root) /
                            require_nonzero(root_relation.a, root_scale,
                                            "Root condensed pressure relation");

      solve_vessel_down(root_.root_vessel, p_root, q_root, rhs, solution);
      return solution;
    } catch (...) {
      info_ = Eigen::NumericalIssue;
      throw;
    }
  }

  Eigen::ComputationInfo info() const override { return info_; }

  const char* name() const override { return "TreeLinear"; }

  bool is_iterative() const override { return false; }

  LinearSolveStepInfo last_step_info() const override { return last_step_info_; }

 private:
  int build_terminal(const Block* block) {
    const auto found = terminal_index_by_block_id_.find(block->id);
    if (found != terminal_index_by_block_id_.end()) {
      return found->second;
    }

    if (block->block_type != BlockType::resistance_bc ||
        block->global_eqn_ids.size() != 1 || block->inlet_nodes.size() != 1 ||
        !block->outlet_nodes.empty()) {
      throw std::runtime_error(
          "TreeLinear supports only one-port RESISTANCE terminal blocks.");
    }

    const Node* node = block->inlet_nodes[0];
    TreeTerminalInfo terminal;
    terminal.block_id = block->id;
    terminal.row = block->global_eqn_ids[0];
    terminal.p_dof = node->pres_dof;
    terminal.q_dof = node->flow_dof;

    const int index = static_cast<int>(terminals_.size());
    terminals_.push_back(terminal);
    terminal_index_by_block_id_.insert({block->id, index});
    return index;
  }

  int build_junction(const Block* block) {
    const auto found = junction_index_by_block_id_.find(block->id);
    if (found != junction_index_by_block_id_.end()) {
      return found->second;
    }

    if (block->block_type != BlockType::junction || block->inlet_nodes.size() != 1 ||
        block->outlet_nodes.empty() ||
        block->global_eqn_ids.size() != block->outlet_nodes.size() + 1) {
      throw std::runtime_error(
          "TreeLinear supports only one-inlet NORMAL_JUNCTION tree nodes.");
    }

    const int index = static_cast<int>(junctions_.size());
    junction_index_by_block_id_.insert({block->id, index});
    junctions_.push_back({});
    junctions_[index].block_id = block->id;
    junctions_[index].mass_row = block->global_eqn_ids.back();
    junctions_[index].parent_p_dof = block->inlet_nodes[0]->pres_dof;
    junctions_[index].parent_q_dof = block->inlet_nodes[0]->flow_dof;
    junctions_[index].children.reserve(block->outlet_nodes.size());

    for (size_t i = 0; i < block->outlet_nodes.size(); ++i) {
      const Node* child_node = block->outlet_nodes[i];
      if (child_node->outlet_eles.size() != 1 ||
          child_node->outlet_eles[0]->block_type != BlockType::blood_vessel) {
        throw std::runtime_error(
            "TreeLinear requires each junction outlet to connect to a single "
            "BloodVessel.");
      }

      TreeJunctionChildInfo child;
      child.pressure_row = block->global_eqn_ids[i];
      child.child_p_dof = child_node->pres_dof;
      child.child_q_dof = child_node->flow_dof;
      child.child_vessel = build_vessel(child_node->outlet_eles[0]);
      junctions_[index].children.push_back(child);
    }

    return index;
  }

  int build_vessel(const Block* block) {
    const auto found = vessel_index_by_block_id_.find(block->id);
    if (found != vessel_index_by_block_id_.end()) {
      return found->second;
    }

    if (block->block_type != BlockType::blood_vessel ||
        block->global_eqn_ids.size() != 2 || block->inlet_nodes.size() != 1 ||
        block->outlet_nodes.size() != 1) {
      throw std::runtime_error(
          "TreeLinear supports only two-equation BloodVessel blocks.");
    }

    const int index = static_cast<int>(vessels_.size());
    vessel_index_by_block_id_.insert({block->id, index});
    vessels_.push_back({});
    vessels_[index].block_id = block->id;
    vessels_[index].row0 = block->global_eqn_ids[0];
    vessels_[index].row1 = block->global_eqn_ids[1];
    vessels_[index].p_in_dof = block->inlet_nodes[0]->pres_dof;
    vessels_[index].q_in_dof = block->inlet_nodes[0]->flow_dof;
    vessels_[index].p_out_dof = block->outlet_nodes[0]->pres_dof;
    vessels_[index].q_out_dof = block->outlet_nodes[0]->flow_dof;

    const Node* outlet_node = block->outlet_nodes[0];
    if (outlet_node->outlet_eles.size() != 1) {
      throw std::runtime_error(
          "TreeLinear requires each vessel outlet to connect to a single "
          "downstream block.");
    }
    const Block* downstream = outlet_node->outlet_eles[0];
    if (downstream->block_type == BlockType::resistance_bc) {
      vessels_[index].downstream_type = TreeDownstreamType::terminal;
      vessels_[index].downstream_index = build_terminal(downstream);
    } else if (downstream->block_type == BlockType::junction) {
      vessels_[index].downstream_type = TreeDownstreamType::junction;
      vessels_[index].downstream_index = build_junction(downstream);
    } else {
      throw std::runtime_error(
          "TreeLinear supports only RESISTANCE or NORMAL_JUNCTION blocks "
          "downstream of BloodVessel blocks.");
    }

    return index;
  }

  CondensedRelation terminal_relation(int terminal_index,
                                      const Vector& rhs) const {
    const TreeTerminalInfo& terminal = terminals_[terminal_index];
    return {terminal.coeff_p, terminal.coeff_q, rhs[terminal.row]};
  }

  CondensedRelation compute_junction_relation(int junction_index,
                                              const Vector& rhs) {
    const TreeJunctionInfo& junction = junctions_[junction_index];
    CondensedRelation relation;
    relation.b = junction.mass_coeff_parent_q;
    relation.c = rhs[junction.mass_row];

    for (const TreeJunctionChildInfo& child : junction.children) {
      const CondensedRelation child_relation =
          compute_vessel_relation(child.child_vessel, rhs);
      const double child_scale =
          std::abs(child_relation.a) + std::abs(child_relation.b) +
          std::abs(child_relation.c) + 1.0;
      const double relation_b = require_nonzero(
          child_relation.b, child_scale,
          "Child condensed flow relation at a junction");
      const double pressure_slope =
          -child.pressure_coeff_parent / child.pressure_coeff_child;
      const double pressure_offset =
          rhs[child.pressure_row] / child.pressure_coeff_child;
      relation.a += child.mass_coeff_child_q *
                    (-(child_relation.a / relation_b) * pressure_slope);
      relation.c -= child.mass_coeff_child_q *
                    ((child_relation.c -
                      child_relation.a * pressure_offset) /
                     relation_b);
    }

    return relation;
  }

  CondensedRelation compute_vessel_relation(int vessel_index,
                                            const Vector& rhs) {
    if (vessel_relation_valid_[vessel_index]) {
      return vessel_relations_[vessel_index];
    }

    const TreeVesselInfo& vessel = vessels_[vessel_index];
    CondensedRelation downstream_relation;
    if (vessel.downstream_type == TreeDownstreamType::terminal) {
      downstream_relation = terminal_relation(vessel.downstream_index, rhs);
    } else {
      downstream_relation = compute_junction_relation(vessel.downstream_index, rhs);
    }

    const double q_out_scale = std::abs(vessel.row1_p_in) +
                               std::abs(vessel.row1_q_in) +
                               std::abs(vessel.row1_q_out) + 1.0;
    const double q_out_denom = require_nonzero(
        vessel.row1_q_out, q_out_scale,
        "BloodVessel outlet-flow coefficient");
    const double q_out_p = -vessel.row1_p_in / q_out_denom;
    const double q_out_q = -vessel.row1_q_in / q_out_denom;
    const double q_out_c = rhs[vessel.row1] / q_out_denom;

    const double p_out_scale = std::abs(vessel.row0_p_in) +
                               std::abs(vessel.row0_q_in) +
                               std::abs(vessel.row0_p_out) +
                               std::abs(vessel.row0_q_out) + 1.0;
    const double p_out_denom = require_nonzero(
        vessel.row0_p_out, p_out_scale,
        "BloodVessel outlet-pressure coefficient");
    const double p_out_p =
        (-vessel.row0_p_in - vessel.row0_q_out * q_out_p) / p_out_denom;
    const double p_out_q =
        (-vessel.row0_q_in - vessel.row0_q_out * q_out_q) / p_out_denom;
    const double p_out_c =
        (rhs[vessel.row0] - vessel.row0_q_out * q_out_c) / p_out_denom;

    CondensedRelation relation;
    relation.a = downstream_relation.a * p_out_p +
                 downstream_relation.b * q_out_p;
    relation.b = downstream_relation.a * p_out_q +
                 downstream_relation.b * q_out_q;
    relation.c = downstream_relation.c - downstream_relation.a * p_out_c -
                 downstream_relation.b * q_out_c;

    vessel_relations_[vessel_index] = relation;
    vessel_relation_valid_[vessel_index] = true;
    return relation;
  }

  void solve_junction_down(int junction_index, double parent_pressure,
                           double parent_flow, const Vector& rhs,
                           Vector& solution) {
    const TreeJunctionInfo& junction = junctions_[junction_index];
    double mass_balance = junction.mass_coeff_parent_q * parent_flow;

    for (const TreeJunctionChildInfo& child : junction.children) {
      const double child_pressure =
          (rhs[child.pressure_row] -
           child.pressure_coeff_parent * parent_pressure) /
          child.pressure_coeff_child;
      const CondensedRelation& child_relation =
          vessel_relations_[child.child_vessel];
      const double child_scale =
          std::abs(child_relation.a) + std::abs(child_relation.b) +
          std::abs(child_relation.c) + 1.0;
      const double child_flow =
          (child_relation.c - child_relation.a * child_pressure) /
          require_nonzero(child_relation.b, child_scale,
                          "Child condensed flow relation during backsolve");

      mass_balance += child.mass_coeff_child_q * child_flow;
      solve_vessel_down(child.child_vessel, child_pressure, child_flow, rhs,
                        solution);
    }

    const double mass_scale =
        std::abs(rhs[junction.mass_row]) + std::abs(mass_balance) + 1.0;
    if (std::abs(mass_balance - rhs[junction.mass_row]) >
        1e3 * pivot_tolerance(mass_scale)) {
      throw std::runtime_error(
          "TreeLinear junction backsolve violated mass conservation.");
    }
  }

  void solve_vessel_down(int vessel_index, double inlet_pressure,
                         double inlet_flow, const Vector& rhs,
                         Vector& solution) {
    const TreeVesselInfo& vessel = vessels_[vessel_index];
    solution[vessel.p_in_dof] = inlet_pressure;
    solution[vessel.q_in_dof] = inlet_flow;

    const double q_out =
        (rhs[vessel.row1] - vessel.row1_p_in * inlet_pressure -
         vessel.row1_q_in * inlet_flow) /
        vessel.row1_q_out;
    const double p_out =
        (rhs[vessel.row0] - vessel.row0_p_in * inlet_pressure -
         vessel.row0_q_in * inlet_flow - vessel.row0_q_out * q_out) /
        vessel.row0_p_out;

    solution[vessel.p_out_dof] = p_out;
    solution[vessel.q_out_dof] = q_out;

    if (vessel.downstream_type == TreeDownstreamType::junction) {
      solve_junction_down(vessel.downstream_index, p_out, q_out, rhs, solution);
    }
  }

  const Model* model_{nullptr};
  Eigen::ComputationInfo info_{Eigen::InvalidInput};
  LinearSolveStepInfo last_step_info_;
  TreeRootInfo root_;
  int system_size_{0};
  bool coefficients_ready_{false};

  std::vector<TreeVesselInfo> vessels_;
  std::vector<TreeJunctionInfo> junctions_;
  std::vector<TreeTerminalInfo> terminals_;
  std::unordered_map<int, int> vessel_index_by_block_id_;
  std::unordered_map<int, int> junction_index_by_block_id_;
  std::unordered_map<int, int> terminal_index_by_block_id_;
  std::vector<CondensedRelation> vessel_relations_;
  std::vector<bool> vessel_relation_valid_;
};

}  // namespace

std::shared_ptr<LinearSolverBackend> create_linear_solver_backend(
    const LinearSolverSettings& settings) {
  switch (settings.backend) {
    case LinearSolverBackendType::sparse_lu:
      return std::make_shared<SparseLULinearSolverBackend>();
    case LinearSolverBackendType::bicgstab_ilut:
      return std::make_shared<BiCGSTABILUTLinearSolverBackend>(settings);
    case LinearSolverBackendType::gmres_ilut:
      return std::make_shared<GMRESILUTLinearSolverBackend>(settings);
    case LinearSolverBackendType::gmres_diagonal:
      return std::make_shared<GMRESDiagonalLinearSolverBackend>(settings);
    case LinearSolverBackendType::tree_linear:
      return std::make_shared<TreeLinearSolverBackend>();
  }

  throw std::runtime_error("Unsupported linear solver backend");
}
