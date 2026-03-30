// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the
// University of California, and others. SPDX-License-Identifier: BSD-3-Clause
#include "Model.h"

template <typename block_type>
BlockFactoryFunc block_factory() {
  return [](int count, Model* model) -> Block* {
    return new block_type(count, model);
  };
}

Model::Model() {
  // Add all implemented blocks to factory
  block_factory_map = {
      {"BloodVessel", block_factory<BloodVessel>()},
      {"ChamberSphere", block_factory<ChamberSphere>()},
      {"BloodVesselJunction", block_factory<BloodVesselJunction>()},
      {"ClosedLoopCoronaryLeft", block_factory<ClosedLoopCoronaryLeftBC>()},
      {"ClosedLoopCoronaryRight", block_factory<ClosedLoopCoronaryRightBC>()},
      {"ClosedLoopHeartAndPulmonary",
       block_factory<ClosedLoopHeartPulmonary>()},
      {"ClosedLoopRCR", block_factory<ClosedLoopRCRBC>()},
      {"CORONARY", block_factory<OpenLoopCoronaryBC>()},
      {"FLOW", block_factory<FlowReferenceBC>()},
      {"NORMAL_JUNCTION", block_factory<Junction>()},
      {"PRESSURE", block_factory<PressureReferenceBC>()},
      {"RCR", block_factory<WindkesselBC>()},
      {"RESISTANCE", block_factory<ResistanceBC>()},
      {"resistive_junction", block_factory<ResistiveJunction>()},
      {"ValveTanh", block_factory<ValveTanh>()},
      {"ChamberElastanceInductor", block_factory<ChamberElastanceInductor>()}};
}

Model::~Model() {}

Block* Model::create_block(const std::string& block_type) {
  // Get block from factory
  auto it = block_factory_map.find(block_type);
  if (it == block_factory_map.end()) {
    throw std::runtime_error("Invalid block type " + block_type);
  }
  Block* block = it->second(block_count, this);
  return block;
}

int Model::add_block(Block* block, const std::string_view& name,
                     const std::vector<int>& block_param_ids, bool internal) {
  // Set global parameter IDs
  block->setup_params_(block_param_ids);

  auto name_string = static_cast<std::string>(name);

  if (internal) {
    hidden_blocks.push_back(std::shared_ptr<Block>(block));
  } else {
    blocks.push_back(std::shared_ptr<Block>(block));
  }

  block_types.push_back(block->block_type);
  block_index_map.insert({name_string, block_count});
  block_names.push_back(name_string);

  return block_count++;
}

int Model::add_block(const std::string& block_name,
                     const std::vector<int>& block_param_ids,
                     const std::string_view& name, bool internal) {
  // Generate block from factory
  auto block = this->create_block(block_name);

  // Add block to model
  return this->add_block(block, name, block_param_ids, internal);
}

bool Model::has_block(const std::string& name) const {
  if (block_index_map.find(name) == block_index_map.end()) {
    return false;
  } else {
    return true;
  }
}

Block* Model::get_block(const std::string_view& name) const {
  auto name_string = static_cast<std::string>(name);

  if (!has_block(name_string)) {
    throw std::runtime_error("No block defined with name " + name_string);
  }

  return blocks[block_index_map.at(name_string)].get();
}

Block* Model::get_block(int block_id) const {
  if (block_id >= blocks.size()) {
    return hidden_blocks[block_id - blocks.size()].get();
  }

  return blocks[block_id].get();
}

BlockType Model::get_block_type(const std::string_view& name) const {
  auto name_string = static_cast<std::string>(name);

  if (block_index_map.find(name_string) == block_index_map.end()) {
    throw std::runtime_error("Could not find block with name " + name_string);
  }

  return block_types[block_index_map.at(name_string)];
}

std::string Model::get_block_name(int block_id) const {
  return block_names[block_id];
}

int Model::add_node(const std::vector<Block*>& inlet_eles,
                    const std::vector<Block*>& outlet_eles,
                    const std::string_view& name) {
  // DEBUG_MSG("Adding node " << name);
  auto node = std::shared_ptr<Node>(
      new Node(node_count, inlet_eles, outlet_eles, this));
  nodes.push_back(node);
  node_names.push_back(static_cast<std::string>(name));

  return node_count++;
}

std::string Model::get_node_name(int node_id) const {
  return node_names[node_id];
}

int Model::add_parameter(double value) {
  parameters.push_back(Parameter(parameter_count, value));
  parameter_values.push_back(parameters.back().get(0.0));
  return parameter_count++;
}

int Model::add_parameter(const std::vector<double>& times,
                         const std::vector<double>& values, bool periodic) {
  auto param = Parameter(parameter_count, times, values, periodic);
  if (periodic && (param.is_constant == false)) {
    if ((this->cardiac_cycle_period > 0.0) &&
        (param.cycle_period != this->cardiac_cycle_period)) {
      throw std::runtime_error(
          "Inconsistent cardiac cycle period defined in parameters");
    }
    this->cardiac_cycle_period = param.cycle_period;
  }
  parameter_values.push_back(param.get(0.0));
  if (!param.is_constant) {
    time_varying_parameter_ids.push_back(param.id);
  }
  parameters.push_back(std::move(param));
  return parameter_count++;
}

Parameter* Model::get_parameter(int param_id) { return &parameters[param_id]; }

double Model::get_parameter_value(int param_id) const {
  return parameter_values[param_id];
}

void Model::update_parameter_value(int param_id, double param_value) {
  parameter_values[param_id] = param_value;
}

void Model::finalize() {
  DEBUG_MSG("Setup degrees-of-freedom of nodes");
  for (auto& node : nodes) {
    node->setup_dofs(dofhandler);
  }
  DEBUG_MSG("Setup degrees-of-freedom of blocks");
  for (auto& block : blocks) {
    block->setup_dofs(dofhandler);
  }
  DEBUG_MSG("Setup model-dependent parameters");
  for (auto& block : blocks) {
    block->setup_model_dependent_params();
  }
  refresh_runtime_structure();
}

int Model::get_num_blocks(bool internal) const {
  int num_blocks = blocks.size();

  if (internal) {
    num_blocks += hidden_blocks.size();
  }

  return num_blocks;
}

void Model::update_constant(SparseSystem& system) {
  for (const auto& block : blocks) {
    block->update_constant(system, parameter_values);
  }
}

void Model::initialize_system(SparseSystem& system, double time) {
  this->time = time;

  for (const int param_id : time_varying_parameter_ids) {
    parameter_values[param_id] = parameters[param_id].get(time);
  }

  if (!fast_system_initialization_supported) {
    update_constant(system);
    initialize_time(system, time);
    return;
  }

  system.F.setZero();
  system.E.setZero();
  system.C.setZero();

  std::vector<Eigen::Triplet<double>> f_triplets;
  std::vector<Eigen::Triplet<double>> e_triplets;
  f_triplets.reserve(triplets_cache.F + time_initializer_blocks.size());
  e_triplets.reserve(triplets_cache.E);

  for (const auto& block_ptr : blocks) {
    append_fast_system_entries(*block_ptr, f_triplets, e_triplets, system.C);
  }

  system.F.setFromTriplets(f_triplets.begin(), f_triplets.end());
  system.E.setFromTriplets(e_triplets.begin(), e_triplets.end());
}

void Model::initialize_time(SparseSystem& system, double time) {
  this->time = time;

  for (const int param_id : time_varying_parameter_ids) {
    parameter_values[param_id] = parameters[param_id].get(time);
  }

  for (Block* block : time_initializer_blocks) {
    block->update_time(system, parameter_values);
  }
}

void Model::update_time(SparseSystem& system, double time) {
  this->time = time;

  for (const int param_id : time_varying_parameter_ids) {
    parameter_values[param_id] = parameters[param_id].get(time);
  }

  for (Block* block : active_time_update_blocks) {
    block->update_time(system, parameter_values);
  }
}

void Model::update_solution(SparseSystem& system,
                            Eigen::Matrix<double, Eigen::Dynamic, 1>& y,
                            Eigen::Matrix<double, Eigen::Dynamic, 1>& dy) {
  for (Block* block : active_solution_update_blocks) {
    block->update_solution(system, parameter_values, y, dy);
  }
}

void Model::post_solve(Eigen::Matrix<double, Eigen::Dynamic, 1>& y) {
  for (Block* block : post_solve_blocks) {
    block->post_solve(y);
  }
}

void Model::to_steady() {
  for (auto& param : parameters) {
    param.to_steady();
  }

  // Special handling for time-varying capacitance
  for (size_t i = 0; i < get_num_blocks(true); i++) {
    get_block(i)->steady = true;
    if ((block_types[i] == BlockType::windkessel_bc) ||
        (block_types[i] == BlockType::closed_loop_rcr_bc)) {
      int param_id_capacitance = blocks[i]->global_param_ids[1];
      double value = parameters[param_id_capacitance].get(0.0);
      param_value_cache.insert({param_id_capacitance, value});
      parameters[param_id_capacitance].update(0.0);
    }
  }
  refresh_runtime_structure();
}

void Model::to_unsteady() {
  for (auto& param : parameters) {
    param.to_unsteady();
  }
  for (auto& [param_id_capacitance, value] : param_value_cache) {
    // DEBUG_MSG("Setting Windkessel capacitance back to " << value);
    parameters[param_id_capacitance].update(value);
  }
  for (size_t i = 0; i < get_num_blocks(true); i++) {
    get_block(i)->steady = false;
  }
  refresh_runtime_structure();
}

TripletsContributions Model::get_num_triplets() const { return triplets_cache; }

bool Model::has_solution_dependent_terms() const {
  return !active_solution_update_blocks.empty();
}

bool Model::has_constant_jacobian() const {
  return !has_time_dependent_jacobian_terms &&
         !has_solution_dependent_jacobian_terms;
}

bool Model::supports_fast_system_initialization() const {
  return fast_system_initialization_supported;
}

void Model::setup_initial_state_dependent_parameters(
    const State& initial_state) {
  DEBUG_MSG("Setup initial state dependent parameters");
  for (Block* block : initial_state_setup_blocks) {
    block->setup_initial_state_dependent_params(initial_state,
                                                parameter_values);
  }
}

void Model::update_has_windkessel_bc(bool has_windkessel) {
  has_windkessel_bc = has_windkessel;
}

void Model::update_largest_windkessel_time_constant(double time_constant) {
  largest_windkessel_time_constant = time_constant;
}

bool Model::get_has_windkessel_bc() { return has_windkessel_bc; }

double Model::get_largest_windkessel_time_constant() {
  return largest_windkessel_time_constant;
}

bool Model::parameter_is_time_varying(int param_id) const {
  return !parameters[param_id].is_constant;
}

bool Model::parameter_is_constant_zero(int param_id) const {
  const Parameter& parameter = parameters[param_id];
  return parameter.is_constant && parameter.value == 0.0;
}

bool Model::block_has_time_initializer(const Block& block) const {
  switch (block.block_type) {
    case BlockType::flow_bc:
    case BlockType::pressure_bc:
    case BlockType::resistance_bc:
    case BlockType::windkessel_bc:
    case BlockType::open_loop_coronary_bc:
    case BlockType::chamber_elastance_inductor:
    case BlockType::chamber_sphere:
    case BlockType::closed_loop_heart_pulmonary:
      return true;
    default:
      return false;
  }
}

bool Model::block_requires_time_update(const Block& block) const {
  switch (block.block_type) {
    case BlockType::flow_bc:
    case BlockType::pressure_bc:
      return parameter_is_time_varying(block.global_param_ids[0]);
    case BlockType::resistance_bc:
      return parameter_is_time_varying(block.global_param_ids[0]) ||
             parameter_is_time_varying(block.global_param_ids[1]);
    case BlockType::windkessel_bc:
      return parameter_is_time_varying(block.global_param_ids[0]) ||
             parameter_is_time_varying(block.global_param_ids[1]) ||
             parameter_is_time_varying(block.global_param_ids[2]) ||
             parameter_is_time_varying(block.global_param_ids[3]);
    case BlockType::open_loop_coronary_bc:
      return parameter_is_time_varying(block.global_param_ids[5]);
    case BlockType::chamber_elastance_inductor:
    case BlockType::chamber_sphere:
    case BlockType::closed_loop_heart_pulmonary:
      return true;
    default:
      return false;
  }
}

bool Model::block_time_update_affects_jacobian(const Block& block) const {
  switch (block.block_type) {
    case BlockType::resistance_bc:
      return parameter_is_time_varying(block.global_param_ids[0]);
    case BlockType::windkessel_bc:
      return parameter_is_time_varying(block.global_param_ids[0]) ||
             parameter_is_time_varying(block.global_param_ids[1]) ||
             parameter_is_time_varying(block.global_param_ids[2]);
    case BlockType::chamber_elastance_inductor:
    case BlockType::chamber_sphere:
    case BlockType::closed_loop_heart_pulmonary:
      return true;
    default:
      return false;
  }
}

bool Model::block_requires_solution_update(const Block& block) const {
  switch (block.block_type) {
    case BlockType::blood_vessel:
      return !parameter_is_constant_zero(block.global_param_ids[3]);
    case BlockType::blood_vessel_junction: {
      const size_t num_outlets = block.outlet_nodes.size();
      if (num_outlets == 0 || block.global_param_ids.size() <= 2 * num_outlets) {
        return false;
      }
      for (size_t i = 2 * num_outlets; i < block.global_param_ids.size(); ++i) {
        if (!parameter_is_constant_zero(block.global_param_ids[i])) {
          return true;
        }
      }
      return false;
    }
    case BlockType::closed_loop_coronary_left_bc:
    case BlockType::closed_loop_coronary_right_bc:
    case BlockType::closed_loop_heart_pulmonary:
    case BlockType::valve_tanh:
    case BlockType::chamber_sphere:
      return true;
    default:
      return false;
  }
}

bool Model::block_solution_update_affects_jacobian(const Block& block) const {
  switch (block.block_type) {
    case BlockType::blood_vessel:
    case BlockType::blood_vessel_junction:
    case BlockType::closed_loop_heart_pulmonary:
    case BlockType::valve_tanh:
    case BlockType::chamber_sphere:
      return block_requires_solution_update(block);
    default:
      return false;
  }
}

bool Model::block_has_initial_state_setup(const Block& block) const {
  return block.block_type == BlockType::open_loop_coronary_bc;
}

bool Model::block_has_post_solve(const Block& block) const {
  return block.block_type == BlockType::closed_loop_heart_pulmonary;
}

bool Model::block_supports_fast_system_initialization(const Block& block) const {
  switch (block.block_type) {
    case BlockType::blood_vessel:
    case BlockType::junction:
    case BlockType::flow_bc:
    case BlockType::pressure_bc:
    case BlockType::resistance_bc:
      return true;
    default:
      return false;
  }
}

void Model::append_fast_system_entries(
    const Block& block, std::vector<Eigen::Triplet<double>>& f_triplets,
    std::vector<Eigen::Triplet<double>>& e_triplets,
    Eigen::Matrix<double, Eigen::Dynamic, 1>& c_vector) const {
  switch (block.block_type) {
    case BlockType::blood_vessel: {
      const double capacitance =
          parameter_values[block.global_param_ids[BloodVessel::CAPACITANCE]];
      const double inductance =
          parameter_values[block.global_param_ids[BloodVessel::INDUCTANCE]];
      const double resistance =
          parameter_values[block.global_param_ids[BloodVessel::RESISTANCE]];

      e_triplets.emplace_back(block.global_eqn_ids[0], block.global_var_ids[3],
                              -inductance);
      e_triplets.emplace_back(block.global_eqn_ids[1], block.global_var_ids[0],
                              -capacitance);
      e_triplets.emplace_back(block.global_eqn_ids[1], block.global_var_ids[1],
                              capacitance * resistance);

      f_triplets.emplace_back(block.global_eqn_ids[0], block.global_var_ids[0],
                              1.0);
      f_triplets.emplace_back(block.global_eqn_ids[0], block.global_var_ids[1],
                              -resistance);
      f_triplets.emplace_back(block.global_eqn_ids[0], block.global_var_ids[2],
                              -1.0);
      f_triplets.emplace_back(block.global_eqn_ids[1], block.global_var_ids[1],
                              1.0);
      f_triplets.emplace_back(block.global_eqn_ids[1], block.global_var_ids[3],
                              -1.0);
      return;
    }
    case BlockType::junction: {
      const size_t num_inlets = block.inlet_nodes.size();
      const size_t num_outlets = block.outlet_nodes.size();
      const size_t num_connections = num_inlets + num_outlets;

      for (size_t i = 0; i + 1 < num_connections; ++i) {
        f_triplets.emplace_back(block.global_eqn_ids[i], block.global_var_ids[0],
                                1.0);
        f_triplets.emplace_back(block.global_eqn_ids[i],
                                block.global_var_ids[2 * i + 2], -1.0);
      }

      const int mass_eqn_id = block.global_eqn_ids[num_connections - 1];
      for (size_t i = 1; i < num_inlets * 2; i += 2) {
        f_triplets.emplace_back(mass_eqn_id, block.global_var_ids[i], 1.0);
      }
      for (size_t i = (num_inlets * 2) + 1; i < num_connections * 2; i += 2) {
        f_triplets.emplace_back(mass_eqn_id, block.global_var_ids[i], -1.0);
      }
      return;
    }
    case BlockType::flow_bc:
      f_triplets.emplace_back(block.global_eqn_ids[0], block.global_var_ids[1],
                              1.0);
      c_vector(block.global_eqn_ids[0]) = -parameter_values[block.global_param_ids[0]];
      return;
    case BlockType::pressure_bc:
      f_triplets.emplace_back(block.global_eqn_ids[0], block.global_var_ids[0],
                              1.0);
      c_vector(block.global_eqn_ids[0]) = -parameter_values[block.global_param_ids[0]];
      return;
    case BlockType::resistance_bc:
      f_triplets.emplace_back(block.global_eqn_ids[0], block.global_var_ids[0],
                              1.0);
      f_triplets.emplace_back(block.global_eqn_ids[0], block.global_var_ids[1],
                              -parameter_values[block.global_param_ids[0]]);
      c_vector(block.global_eqn_ids[0]) =
          -parameter_values[block.global_param_ids[1]];
      return;
    default:
      throw std::runtime_error(
          "Fast system initialization encountered unsupported block type");
  }
}

void Model::refresh_runtime_structure() {
  triplets_cache = {};
  time_initializer_blocks.clear();
  active_time_update_blocks.clear();
  active_solution_update_blocks.clear();
  initial_state_setup_blocks.clear();
  post_solve_blocks.clear();
  has_time_dependent_jacobian_terms = false;
  has_solution_dependent_jacobian_terms = false;
  fast_system_initialization_supported = true;

  for (const auto& block_ptr : blocks) {
    Block& block = *block_ptr;
    triplets_cache += block.get_num_triplets();

    if (block_has_time_initializer(block)) {
      time_initializer_blocks.push_back(&block);
    }
    if (block_requires_time_update(block)) {
      active_time_update_blocks.push_back(&block);
    }
    if (block_requires_solution_update(block)) {
      active_solution_update_blocks.push_back(&block);
      fast_system_initialization_supported = false;
    }
    if (block_has_initial_state_setup(block)) {
      initial_state_setup_blocks.push_back(&block);
    }
    if (block_has_post_solve(block)) {
      post_solve_blocks.push_back(&block);
    }

    has_time_dependent_jacobian_terms =
        has_time_dependent_jacobian_terms ||
        block_time_update_affects_jacobian(block);
    has_solution_dependent_jacobian_terms =
        has_solution_dependent_jacobian_terms ||
        block_solution_update_affects_jacobian(block);
    fast_system_initialization_supported =
        fast_system_initialization_supported &&
        block_supports_fast_system_initialization(block);
  }
}
