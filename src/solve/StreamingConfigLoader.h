// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the
// University of California, and others. SPDX-License-Identifier: BSD-3-Clause
/**
 * @file StreamingConfigLoader.h
 * @brief Streaming JSON loader for large 0D solver configurations.
 */

#ifndef SVZERODSOLVER_SOLVE_STREAMINGCONFIGLOADER_HPP_
#define SVZERODSOLVER_SOLVE_STREAMINGCONFIGLOADER_HPP_

#include <istream>

#include <nlohmann/json.hpp>

#include "Model.h"
#include "SimulationParameters.h"
#include "State.h"

/**
 * @brief Load simulation parameters, model, and initial state from a JSON
 *        configuration using a streaming (SAX) parser.
 *
 * This avoids constructing a single monolithic nlohmann::json DOM for the
 * entire configuration file and instead builds the Model incrementally.
 *
 * @param is Input stream providing the JSON configuration.
 * @param simparams Simulation parameters to fill.
 * @param model Model to build.
 * @param initial_state Initial state to construct.
 */
void load_config_streaming(std::istream& is,
                           SimulationParameters& simparams,
                           Model& model,
                           State& initial_state);

#endif  // SVZERODSOLVER_SOLVE_STREAMINGCONFIGLOADER_HPP_

