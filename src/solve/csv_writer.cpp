// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the
// University of California, and others. SPDX-License-Identifier: BSD-3-Clause
#include "csv_writer.h"

#include <iomanip>
#include <sstream>

void write_vessel_csv(std::ostream& out, const std::vector<double>& times,
                      const ResultHistory& states, const Model& model,
                      bool mean, bool derivative) {
  out << std::scientific << std::setprecision(16);

  if (derivative) {
    out << "name,time,flow_in,flow_out,pressure_in,pressure_out,d_flow_in,d_"
           "flow_out,d_pressure_in,d_pressure_out\n";
  } else {
    out << "name,time,flow_in,flow_out,pressure_in,pressure_out\n";
  }

  int num_steps = states.size();

  for (size_t i = 0; i < model.get_num_blocks(); i++) {
    auto block = model.get_block(i);

    if (dynamic_cast<const BloodVessel*>(block) == nullptr &&
        dynamic_cast<const ChamberSphere*>(block) == nullptr) {
      continue;
    }

    const std::string name = block->get_name();
    const int inflow_dof = block->inlet_nodes[0]->flow_dof;
    const int outflow_dof = block->outlet_nodes[0]->flow_dof;
    const int inpres_dof = block->inlet_nodes[0]->pres_dof;
    const int outpres_dof = block->outlet_nodes[0]->pres_dof;

    if (derivative) {
      if (mean) {
        double inflow_mean = 0.0;
        double outflow_mean = 0.0;
        double inpres_mean = 0.0;
        double outpres_mean = 0.0;
        double d_inflow_mean = 0.0;
        double d_outflow_mean = 0.0;
        double d_inpres_mean = 0.0;
        double d_outpres_mean = 0.0;

        for (int step = 0; step < num_steps; step++) {
          inflow_mean += states.value(step, inflow_dof);
          outflow_mean += states.value(step, outflow_dof);
          inpres_mean += states.value(step, inpres_dof);
          outpres_mean += states.value(step, outpres_dof);
          d_inflow_mean += states.derivative(step, inflow_dof);
          d_outflow_mean += states.derivative(step, outflow_dof);
          d_inpres_mean += states.derivative(step, inpres_dof);
          d_outpres_mean += states.derivative(step, outpres_dof);
        }

        inflow_mean /= num_steps;
        outflow_mean /= num_steps;
        inpres_mean /= num_steps;
        outpres_mean /= num_steps;
        d_inflow_mean /= num_steps;
        d_outflow_mean /= num_steps;
        d_inpres_mean /= num_steps;
        d_outpres_mean /= num_steps;

        out << name << ",," << inflow_mean << "," << outflow_mean << ","
            << inpres_mean << "," << outpres_mean << "," << d_inflow_mean
            << "," << d_outflow_mean << "," << d_inpres_mean << ","
            << d_outpres_mean << "\n";
      } else {
        for (int step = 0; step < num_steps; step++) {
          out << name << "," << times[step] << "," << states.value(step, inflow_dof)
              << "," << states.value(step, outflow_dof) << ","
              << states.value(step, inpres_dof) << ","
              << states.value(step, outpres_dof) << ","
              << states.derivative(step, inflow_dof) << ","
              << states.derivative(step, outflow_dof) << ","
              << states.derivative(step, inpres_dof) << ","
              << states.derivative(step, outpres_dof) << "\n";
        }
      }
    } else {
      if (mean) {
        double inflow_mean = 0.0;
        double outflow_mean = 0.0;
        double inpres_mean = 0.0;
        double outpres_mean = 0.0;

        for (int step = 0; step < num_steps; step++) {
          inflow_mean += states.value(step, inflow_dof);
          outflow_mean += states.value(step, outflow_dof);
          inpres_mean += states.value(step, inpres_dof);
          outpres_mean += states.value(step, outpres_dof);
        }

        inflow_mean /= num_steps;
        outflow_mean /= num_steps;
        inpres_mean /= num_steps;
        outpres_mean /= num_steps;

        out << name << ",," << inflow_mean << "," << outflow_mean << ","
            << inpres_mean << "," << outpres_mean << "\n";
      } else {
        for (int step = 0; step < num_steps; step++) {
          out << name << "," << times[step] << "," << states.value(step, inflow_dof)
              << "," << states.value(step, outflow_dof) << ","
              << states.value(step, inpres_dof) << ","
              << states.value(step, outpres_dof) << "\n";
        }
      }
    }
  }
}

void write_variable_csv(std::ostream& out, const std::vector<double>& times,
                        const ResultHistory& states, const Model& model,
                        bool mean, bool derivative) {
  out << std::scientific << std::setprecision(16);

  const int num_steps = states.size();

  if (derivative) {
    out << "name,time,y,ydot\n";
    if (mean) {
      for (size_t i = 0; i < model.dofhandler.size(); i++) {
        const std::string name = model.dofhandler.variables[i];
        double mean_y = 0.0;
        double mean_ydot = 0.0;

        for (int step = 0; step < num_steps; step++) {
          mean_y += states.value(step, i);
          mean_ydot += states.derivative(step, i);
        }

        mean_y /= num_steps;
        mean_ydot /= num_steps;
        out << name << ",," << mean_y << "," << mean_ydot << "\n";
      }
    } else {
      for (size_t i = 0; i < model.dofhandler.size(); i++) {
        const std::string name = model.dofhandler.variables[i];
        for (int step = 0; step < num_steps; step++) {
          out << name << "," << times[step] << "," << states.value(step, i)
              << "," << states.derivative(step, i) << "\n";
        }
      }
    }
  } else {
    out << "name,time,y\n";
    if (mean) {
      for (size_t i = 0; i < model.dofhandler.size(); i++) {
        const std::string name = model.dofhandler.variables[i];
        double mean_y = 0.0;

        for (int step = 0; step < num_steps; step++) {
          mean_y += states.value(step, i);
        }

        mean_y /= num_steps;
        out << name << ",," << mean_y << "\n";
      }
    } else {
      for (size_t i = 0; i < model.dofhandler.size(); i++) {
        const std::string name = model.dofhandler.variables[i];
        for (int step = 0; step < num_steps; step++) {
          out << name << "," << times[step] << "," << states.value(step, i)
              << "\n";
        }
      }
    }
  }
}

std::string to_vessel_csv(const std::vector<double>& times,
                          const ResultHistory& states, const Model& model,
                          bool mean, bool derivative) {
  std::stringstream out;
  write_vessel_csv(out, times, states, model, mean, derivative);
  return out.str();
}

std::string to_variable_csv(const std::vector<double>& times,
                            const ResultHistory& states, const Model& model,
                            bool mean, bool derivative) {
  std::stringstream out;
  write_variable_csv(out, times, states, model, mean, derivative);
  return out.str();
}
