// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the
// University of California, and others. SPDX-License-Identifier: BSD-3-Clause

#include "StreamingConfigLoader.h"

#include <algorithm>
#include <stdexcept>

#include "SimulationParameters.h"
#include "debug.h"

using Json = nlohmann::json;

namespace {

enum class Section {
  None,
  SimulationParameters,
  Vessels,
  BoundaryConditions,
  ExternalCoupling,
  Junctions,
  ClosedLoopBlocks,
  Valves,
  Chambers,
  InitialCondition,
  InitialConditionD
};

class StreamingConfigLoaderSax : public Json::json_sax_t {
 public:
  StreamingConfigLoaderSax(SimulationParameters& simparams,
                           Model& model,
                           State& initial_state)
      : simparams_(simparams), model_(model), initial_state_(initial_state) {}

  // json_sax interface
  bool null() override { return handle_value(Json(nullptr)); }

  bool boolean(bool v) override { return handle_value(Json(v)); }

  bool number_integer(number_integer_t v) override {
    return handle_value(Json(v));
  }

  bool number_unsigned(number_unsigned_t v) override {
    return handle_value(Json(v));
  }

  bool number_float(number_float_t v, const string_t&) override {
    return handle_value(Json(v));
  }

  bool string(string_t& v) override { return handle_value(Json(v)); }

  bool start_object(std::size_t) override {
    ++depth_;
    // simulation_parameters, initial_condition and initial_condition_d are
    // small objects that we keep as DOM subtrees for reuse with existing
    // loaders.
    if (!building_dom_ && depth_ == 2) {
      if (section_ == Section::SimulationParameters) {
        sim_params_json_ = Json::object();
        begin_dom(sim_params_json_, depth_);
      } else if (section_ == Section::InitialCondition) {
        initial_condition_json_ = Json::object();
        begin_dom(initial_condition_json_, depth_);
      } else if (section_ == Section::InitialConditionD) {
        initial_condition_d_json_ = Json::object();
        begin_dom(initial_condition_d_json_, depth_);
      }
    }
    if (should_start_dom_object()) {
      // Root of a DOM subtree has already been created; only create children.
      if (!dom_stack_.empty()) {
        Json* parent = dom_stack_.back();
        if (parent->is_object()) {
          (*parent)[dom_pending_key_] = Json::object();
          dom_stack_.push_back(&(*parent)[dom_pending_key_]);
          dom_pending_key_.clear();
        } else if (parent->is_array()) {
          parent->push_back(Json::object());
          dom_stack_.push_back(&parent->back());
        }
      }
    }
    // Block object begins immediately after entering the array element.
    if (array_section_ != Section::None && depth_ == array_depth_ + 1 &&
        !building_block_) {
      start_block_dom();
    }
    return true;
  }

  bool key(string_t& v) override {
    current_key_ = v;
    if (building_dom_) {
      dom_pending_key_ = v;
    }
    // Track top-level sections.
    if (depth_ == 1) {
      on_top_level_key(v);
    }
    return true;
  }

  bool end_object() override {
    // Handle end of DOM subtree for blocks or small subtrees.
    if (building_block_ && depth_ == block_start_depth_) {
      finish_block_dom();
    }
    if (building_dom_ && dom_root_ && depth_ == dom_start_depth_) {
      end_dom();
    }
    if (building_dom_ && dom_root_ && !dom_stack_.empty()) {
      if (dom_stack_.back() != dom_root_) {
        dom_stack_.pop_back();
      }
    }

    --depth_;
    return true;
  }

  bool start_array(std::size_t) override {
    ++depth_;
    if (should_start_dom_array()) {
      if (!dom_stack_.empty()) {
        Json* parent = dom_stack_.back();
        if (parent->is_object()) {
          (*parent)[dom_pending_key_] = Json::array();
          dom_stack_.push_back(&(*parent)[dom_pending_key_]);
          dom_pending_key_.clear();
        } else if (parent->is_array()) {
          parent->push_back(Json::array());
          dom_stack_.push_back(&parent->back());
        }
      }
    }
    // Entering an array of blocks for a model subsection.
    if (section_ == Section::Vessels ||
        section_ == Section::BoundaryConditions ||
        section_ == Section::ExternalCoupling ||
        section_ == Section::Junctions ||
        section_ == Section::ClosedLoopBlocks ||
        section_ == Section::Valves ||
        section_ == Section::Chambers) {
      if (!in_array_ && depth_ == 2) {
        in_array_ = true;
        array_section_ = section_;
        array_depth_ = depth_;
      }
    }
    return true;
  }

  bool end_array() override {
    if (in_array_ && depth_ == array_depth_) {
      in_array_ = false;
      array_section_ = Section::None;
      array_depth_ = -1;
    }
    if (building_dom_ && dom_root_ && !dom_stack_.empty()) {
      if (dom_stack_.back() != dom_root_) {
        dom_stack_.pop_back();
      }
    }
    --depth_;
    return true;
  }

  bool binary(binary_t&) override { return false; }

  bool parse_error(std::size_t,
                   const std::string&,
                   const nlohmann::detail::exception& ex) override {
    last_error_ = ex.what();
    return false;
  }

  void finalize() {
    if (!saw_sim_params_) {
      throw std::runtime_error("Define simulation_parameters");
    }
    if (!saw_boundary_conditions_) {
      throw std::runtime_error("Define at least one boundary condition");
    }
    // Build a minimal config for simulation parameters.
    Json sim_config;
    sim_config["simulation_parameters"] = sim_params_json_;
    simparams_ = load_simulation_params(sim_config);

    // Mirror the time-stepping logic from Solver/Interface: ensure the model
    // has a valid cardiac_cycle_period and derive the time step size and,
    // if needed, the total number of time steps.
    if (model_.cardiac_cycle_period < 0.0) {
      // If it has not been read from config or Parameter yet, set a default
      // value of 1.0 s.
      model_.cardiac_cycle_period = 1.0;
    }

    if (!simparams_.sim_coupled && simparams_.use_cycle_to_cycle_error &&
        model_.get_has_windkessel_bc()) {
      const double tau = model_.get_largest_windkessel_time_constant();
      const double T = model_.cardiac_cycle_period;
      if (T <= 0.0) {
        throw std::runtime_error(
            "Invalid cardiac_cycle_period when computing cycle-to-cycle "
            "error-based number of cycles");
      }
      simparams_.sim_num_cycles =
          int(std::ceil(-1.0 * tau / T *
                        std::log(simparams_.sim_cycle_to_cycle_error)));
      simparams_.sim_num_time_steps =
          (simparams_.sim_pts_per_cycle - 1) * simparams_.sim_num_cycles + 1;
    }

    if (!simparams_.sim_coupled) {
      const double denom =
          static_cast<double>(simparams_.sim_pts_per_cycle) - 1.0;
      if (denom <= 0.0) {
        throw std::runtime_error(
            "Invalid pts_per_cycle when computing time step size");
      }
      simparams_.sim_time_step_size = model_.cardiac_cycle_period / denom;
    } else {
      const double denom =
          static_cast<double>(simparams_.sim_num_time_steps) - 1.0;
      if (denom <= 0.0) {
        throw std::runtime_error(
            "Invalid num_time_steps when computing external time step size");
      }
      simparams_.sim_time_step_size =
          simparams_.sim_external_step_size / denom;
    }

    // Create junction blocks and their connections now that all vessels and
    // vessel_id_map_ have been processed. This avoids depending on the
    // ordering of top-level JSON sections.
    for (const auto& junction_config : junctions_) {
      const std::string j_type = junction_config.at("junction_type");
      const std::string junction_name = junction_config.at("junction_name");

      if (!junction_config.contains("junction_values")) {
        generate_block(model_, Json::object(), j_type, junction_name);
      } else {
        generate_block(model_, junction_config.at("junction_values"), j_type,
                       junction_name);
      }

      if (junction_config.contains("inlet_vessels") &&
          junction_config.contains("outlet_vessels")) {
        for (int vessel_id : junction_config.at("inlet_vessels")) {
          auto it = vessel_id_map_.find(vessel_id);
          if (it == vessel_id_map_.end()) {
            throw std::runtime_error(
                "StreamingConfigLoader: junction '" + junction_name +
                "' references unknown inlet vessel_id " +
                std::to_string(vessel_id) +
                " (vessel not defined in 'vessels' section)");
          }
          connections_.push_back({it->second, junction_name});
        }
        for (int vessel_id : junction_config.at("outlet_vessels")) {
          auto it = vessel_id_map_.find(vessel_id);
          if (it == vessel_id_map_.end()) {
            throw std::runtime_error(
                "StreamingConfigLoader: junction '" + junction_name +
                "' references unknown outlet vessel_id " +
                std::to_string(vessel_id) +
                " (vessel not defined in 'vessels' section)");
          }
          connections_.push_back({junction_name, it->second});
        }
      } else if (junction_config.contains("inlet_blocks") &&
                 junction_config.contains("outlet_blocks")) {
        for (const auto& block_name : junction_config.at("inlet_blocks")) {
          connections_.push_back({block_name, junction_name});
        }
        for (const auto& block_name : junction_config.at("outlet_blocks")) {
          connections_.push_back({junction_name, block_name});
        }
      }
    }

    // Finalize connections and model.
    for (const auto& conn : connections_) {
      auto ele1 = model_.get_block(std::get<0>(conn));
      auto ele2 = model_.get_block(std::get<1>(conn));
      model_.add_node({ele1}, {ele2},
                      ele1->get_name() + ":" + ele2->get_name());
    }
    model_.finalize();

    // Build a minimal config for initial conditions and call the existing
    // loader.
    Json ic_config = Json::object();
    if (!initial_condition_json_.is_null()) {
      ic_config["initial_condition"] = initial_condition_json_;
    }
    if (!initial_condition_d_json_.is_null()) {
      ic_config["initial_condition_d"] = initial_condition_d_json_;
    }
    initial_state_ = load_initial_condition(ic_config, model_);
  }

  const std::string& last_error() const { return last_error_; }

 private:
  SimulationParameters& simparams_;
  Model& model_;
  State& initial_state_;

  int depth_ = 0;

  Section section_ = Section::None;
  bool saw_sim_params_ = false;
  bool saw_boundary_conditions_ = false;

  // Array-of-blocks tracking.
  bool in_array_ = false;
  Section array_section_ = Section::None;
  int array_depth_ = -1;

  // DOM builder for small subtrees and per-block DOMs.
  bool building_dom_ = false;
  Json* dom_root_ = nullptr;
  std::vector<Json*> dom_stack_;
  std::string dom_pending_key_;
  int dom_start_depth_ = -1;

  // Per-block DOM state.
  bool building_block_ = false;
  int block_start_depth_ = -1;
  Json current_block_;

  // Top-level small DOMs.
  Json sim_params_json_;
  Json initial_condition_json_;
  Json initial_condition_d_json_;

  // Model-assembly state.
  std::vector<std::tuple<std::string, std::string>> connections_;
  std::map<int, std::string> vessel_id_map_;
  std::map<std::string, std::string> bc_type_map_;
  std::vector<std::string> closed_loop_bcs_;
  bool heartpulmonary_present_ = false;

  // Simple counters for debug logging.
  std::size_t vessels_count_ = 0;
  std::size_t bcs_count_ = 0;
  std::size_t external_coupling_count_ = 0;
  std::size_t junctions_count_ = 0;
  std::size_t closed_loop_count_ = 0;
  std::size_t valves_count_ = 0;
  std::size_t chambers_count_ = 0;

  // Junction configs stored for processing in finalize(), after all vessels
  // have been read and vessel_id_map_ is complete.
  std::vector<Json> junctions_;

  std::string current_key_;
  std::string last_error_;

  // Helpers ------------------------------------------------------------------

  void on_top_level_key(const std::string& key) {
    if (key == "simulation_parameters") {
      section_ = Section::SimulationParameters;
      saw_sim_params_ = true;
      DEBUG_MSG("StreamingConfigLoader: entering section simulation_parameters");
    } else if (key == "vessels") {
      section_ = Section::Vessels;
      DEBUG_MSG("StreamingConfigLoader: entering section vessels");
    } else if (key == "boundary_conditions") {
      section_ = Section::BoundaryConditions;
      saw_boundary_conditions_ = true;
      DEBUG_MSG("StreamingConfigLoader: entering section boundary_conditions");
    } else if (key == "external_solver_coupling_blocks") {
      section_ = Section::ExternalCoupling;
      DEBUG_MSG("StreamingConfigLoader: entering section external_solver_coupling_blocks");
    } else if (key == "junctions") {
      section_ = Section::Junctions;
      DEBUG_MSG("StreamingConfigLoader: entering section junctions");
    } else if (key == "closed_loop_blocks") {
      section_ = Section::ClosedLoopBlocks;
      DEBUG_MSG("StreamingConfigLoader: entering section closed_loop_blocks");
    } else if (key == "valves") {
      section_ = Section::Valves;
      DEBUG_MSG("StreamingConfigLoader: entering section valves");
    } else if (key == "chambers") {
      section_ = Section::Chambers;
      DEBUG_MSG("StreamingConfigLoader: entering section chambers");
    } else if (key == "initial_condition") {
      section_ = Section::InitialCondition;
      DEBUG_MSG("StreamingConfigLoader: entering section initial_condition");
    } else if (key == "initial_condition_d") {
      section_ = Section::InitialConditionD;
      DEBUG_MSG("StreamingConfigLoader: entering section initial_condition_d");
    } else {
      section_ = Section::None;
    }
  }

  bool should_start_dom_object() const {
    if (!building_dom_ || !dom_root_) {
      return false;
    }
    // Only create new child objects when we are below the root depth of the
    // DOM subtree. At the root depth the object has already been created by
    // begin_dom().
    return !dom_stack_.empty() && depth_ > dom_start_depth_;
  }

  bool should_start_dom_array() const {
    if (!building_dom_ || !dom_root_) {
      return false;
    }
    // Only create new child arrays when we are below the root depth of the
    // DOM subtree.
    return !dom_stack_.empty() && depth_ > dom_start_depth_;
  }

  void begin_dom(Json& root, int start_depth) {
    building_dom_ = true;
    dom_root_ = &root;
    dom_stack_.clear();
    dom_stack_.push_back(&root);
    dom_pending_key_.clear();
    dom_start_depth_ = start_depth;
  }

  void end_dom() {
    building_dom_ = false;
    dom_root_ = nullptr;
    dom_stack_.clear();
    dom_pending_key_.clear();
    dom_start_depth_ = -1;
  }

  void start_block_dom() {
    building_block_ = true;
    current_block_ = Json::object();
    block_start_depth_ = depth_;
    begin_dom(current_block_, depth_);
  }

  void finish_block_dom() {
    // Dispatch based on which array we are in.
    switch (array_section_) {
      case Section::Vessels:
        handle_vessel(current_block_);
        ++vessels_count_;
        if ((vessels_count_ % 1000) == 0) {
          DEBUG_MSG("StreamingConfigLoader: processed " << vessels_count_
                                                        << " vessels");
        }
        break;
      case Section::BoundaryConditions:
        handle_boundary_condition(current_block_);
        ++bcs_count_;
        if ((bcs_count_ % 1000) == 0) {
          DEBUG_MSG("StreamingConfigLoader: processed " << bcs_count_
                                                        << " boundary conditions");
        }
        break;
      case Section::ExternalCoupling:
        handle_external_coupling(current_block_);
        ++external_coupling_count_;
        break;
      case Section::Junctions:
        handle_junction(current_block_);
        ++junctions_count_;
        break;
      case Section::ClosedLoopBlocks:
        handle_closed_loop_block(current_block_);
        ++closed_loop_count_;
        break;
      case Section::Valves:
        handle_valve(current_block_);
        ++valves_count_;
        break;
      case Section::Chambers:
        handle_chamber(current_block_);
        ++chambers_count_;
        break;
      default:
        break;
    }
    end_dom();
    building_block_ = false;
    block_start_depth_ = -1;
    current_block_ = Json();  // release memory
  }

  bool handle_value(const Json& value) {
    // Build small DOMs and per-block DOMs.
    if (building_dom_ && !dom_stack_.empty()) {
      Json* parent = dom_stack_.back();
      if (parent->is_object()) {
        if (dom_pending_key_.empty()) {
          // Should not happen in well-formed JSON with objects.
        } else {
          (*parent)[dom_pending_key_] = value;
          dom_pending_key_.clear();
        }
      } else if (parent->is_array()) {
        parent->push_back(value);
      }
      return true;
    }

    // For simulation_parameters and initial_condition sections, start root DOM
    // on first value if not already started (e.g., simple scalar cases).
    if (section_ == Section::SimulationParameters) {
      // simulation_parameters is always an object; handled via DOM builder.
      return true;
    }
    if (section_ == Section::InitialCondition ||
        section_ == Section::InitialConditionD) {
      return true;
    }

    return true;
  }

  // Per-section handlers -----------------------------------------------------

  void handle_vessel(const Json& vessel_config) {
    const auto& vessel_values = vessel_config.at("zero_d_element_values");
    const std::string vessel_name = vessel_config.at("vessel_name");
    int vessel_id = vessel_config.at("vessel_id");
    vessel_id_map_.insert({vessel_id, vessel_name});

    generate_block(model_, vessel_values,
                   vessel_config.at("zero_d_element_type"), vessel_name);

    if (vessel_config.contains("boundary_conditions")) {
      const auto& vessel_bc = vessel_config.at("boundary_conditions");
      if (vessel_bc.contains("inlet")) {
        connections_.push_back({vessel_bc.at("inlet"), vessel_name});
        if (vessel_bc.contains("outlet")) {
          model_.get_block(vessel_name)->update_vessel_type(VesselType::both);
        } else {
          model_.get_block(vessel_name)->update_vessel_type(VesselType::inlet);
        }
      }
      if (vessel_bc.contains("outlet")) {
        connections_.push_back({vessel_name, vessel_bc.at("outlet")});
        model_.get_block(vessel_name)->update_vessel_type(VesselType::outlet);
      }
    }
  }

  void handle_boundary_condition(const Json& bc_config) {
    std::string bc_type = bc_config.at("bc_type");
    std::string bc_name = bc_config.at("bc_name");
    const auto& bc_values = bc_config.at("bc_values");

    bc_type_map_.insert({bc_name, bc_type});

    int block_id = generate_block(model_, bc_values, bc_type, bc_name);

    Block* block = model_.get_block(block_id);

    if (block->block_type == BlockType::windkessel_bc) {
      model_.update_has_windkessel_bc(true);
      double Rd = bc_values.at("Rd");
      double C = bc_values.at("C");
      double time_constant = Rd * C;
      model_.update_largest_windkessel_time_constant(
          std::max(model_.get_largest_windkessel_time_constant(),
                   time_constant));
    }

    if (block->block_type == BlockType::closed_loop_rcr_bc) {
      if (bc_values.at("closed_loop_outlet") == true) {
        closed_loop_bcs_.push_back(bc_name);
      }
    } else if (block->block_class == BlockClass::closed_loop) {
      closed_loop_bcs_.push_back(bc_name);
    }
  }

  void handle_external_coupling(const Json& coupling_config) {
    std::string coupling_type = coupling_config.at("type");
    std::string coupling_name = coupling_config.at("name");
    std::string coupling_loc = coupling_config.at("location");
    bool periodic = coupling_config.value("periodic", true);
    const auto& coupling_values = coupling_config.at("values");
    const bool internal = false;

    generate_block(model_, coupling_values, coupling_type, coupling_name,
                   internal, periodic);

    std::string connected_block = coupling_config.at("connected_block");
    std::string connected_type;
    int found_block = 0;

    if (connected_block == "ClosedLoopHeartAndPulmonary") {
      connected_type = "ClosedLoopHeartAndPulmonary";
      found_block = 1;
    } else {
      auto it = bc_type_map_.find(connected_block);
      if (it != bc_type_map_.end()) {
        connected_type = it->second;
        found_block = 1;
      }
      if (found_block == 0) {
        for (const auto& vessel : vessel_id_map_) {
          if (connected_block == vessel.second) {
            connected_type = "BloodVessel";
            found_block = 1;
            break;
          }
        }
      }
      if (found_block == 0) {
        throw std::runtime_error(
            "Error! Could not connected type for block: " + connected_block);
      }
    }

    if (coupling_loc == "inlet") {
      std::vector<std::string> possible_types = {
          "RESISTANCE", "RCR", "ClosedLoopRCR", "SimplifiedRCR", "CORONARY",
          "ClosedLoopCoronaryLeft", "ClosedLoopCoronaryRight", "BloodVessel"};
      if (std::find(possible_types.begin(), possible_types.end(),
                    connected_type) == possible_types.end()) {
        throw std::runtime_error(
            "Error: The specified connection type for inlet "
            "external_coupling_block is invalid.");
      }
      connections_.push_back({coupling_name, connected_block});
    } else if (coupling_loc == "outlet") {
      std::vector<std::string> possible_types = {
          "ClosedLoopRCR", "ClosedLoopHeartAndPulmonary", "BloodVessel"};
      if (std::find(possible_types.begin(), possible_types.end(),
                    connected_type) == possible_types.end()) {
        throw std::runtime_error(
            "Error: The specified connection type for outlet "
            "external_coupling_block is invalid.");
      }
      if (connected_type == "ClosedLoopRCR" ||
          connected_type == "BloodVessel") {
        connections_.push_back({connected_block, coupling_name});
      }
    }
  }

  void handle_junction(const Json& junction_config) {
    junctions_.push_back(junction_config);
    ++junctions_count_;
  }

  void handle_closed_loop_block(const Json& closed_loop_config) {
    std::string closed_loop_type = closed_loop_config.at("closed_loop_type");
    if (closed_loop_type == "ClosedLoopHeartAndPulmonary") {
      if (heartpulmonary_present_) {
        throw std::runtime_error(
            "Error. Only one ClosedLoopHeartAndPulmonary can be included.");
      }
      heartpulmonary_present_ = true;
      std::string heartpulmonary_name = "CLH";
      double cycle_period = closed_loop_config.at("cardiac_cycle_period");
      if (model_.cardiac_cycle_period > 0.0 &&
          cycle_period != model_.cardiac_cycle_period) {
        throw std::runtime_error(
            "Inconsistent cardiac cycle period defined in "
            "ClosedLoopHeartAndPulmonary.");
      } else {
        model_.cardiac_cycle_period = cycle_period;
      }
      const auto& heart_params = closed_loop_config.at("parameters");

      generate_block(model_, heart_params, closed_loop_type,
                     heartpulmonary_name);

      std::string heart_inlet_junction_name = "J_heart_inlet";
      connections_.push_back({heart_inlet_junction_name, heartpulmonary_name});
      generate_block(model_, Json::object(), "NORMAL_JUNCTION",
                     heart_inlet_junction_name);

      for (const auto& heart_inlet_elem : closed_loop_bcs_) {
        connections_.push_back({heart_inlet_elem, heart_inlet_junction_name});
      }

      std::string heart_outlet_junction_name = "J_heart_outlet";
      connections_.push_back(
          {heartpulmonary_name, heart_outlet_junction_name});
      generate_block(model_, Json::object(), "NORMAL_JUNCTION",
                     heart_outlet_junction_name);
      for (const auto& outlet_block :
           closed_loop_config.at("outlet_blocks")) {
        connections_.push_back({heart_outlet_junction_name, outlet_block});
      }
    }
  }

  void handle_valve(const Json& valve_config) {
    std::string valve_type = valve_config.at("type");
    std::string valve_name = valve_config.at("name");
    const auto& params = valve_config.at("params");

    generate_block(model_, params, valve_type, valve_name);
    connections_.push_back({params.at("upstream_block"), valve_name});
    connections_.push_back({valve_name, params.at("downstream_block")});
  }

  void handle_chamber(const Json& chamber_config) {
    std::string chamber_type = chamber_config.at("type");
    std::string chamber_name = chamber_config.at("name");
    const auto& values = chamber_config.at("values");

    generate_block(model_, values, chamber_type, chamber_name);
  }
};

}  // namespace

void load_config_streaming(std::istream& is,
                           SimulationParameters& simparams,
                           Model& model,
                           State& initial_state) {
  DEBUG_MSG("Streaming JSON configuration");
  StreamingConfigLoaderSax handler(simparams, model, initial_state);
  const bool ok =
      Json::sax_parse(is, &handler, nlohmann::json::input_format_t::json);
  if (!ok) {
    throw std::runtime_error("Streaming JSON parse failed: " +
                             handler.last_error());
  }
  handler.finalize();
  DEBUG_MSG("Finished streaming JSON configuration");
}
