// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the
// University of California, and others. SPDX-License-Identifier: BSD-3-Clause
#include "Parameter.h"

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <sstream>

Parameter::Parameter(int id, double value) {
  this->id = id;
  update(value);
}

Parameter::Parameter(int id, const std::vector<double>& times,
                     const std::vector<double>& values, bool periodic) {
  this->id = id;
  this->is_periodic = periodic;
  update(times, values);
}

void Parameter::update(double update_value) {
  is_constant = true;
  is_periodic = true;
  value = update_value;
}

void Parameter::update(const std::vector<double>& update_times,
                       const std::vector<double>& update_values) {
  this->size = update_values.size();

  if (size == 1) {
    value = update_values[0];
    is_constant = true;
  } else {
    times = update_times;
    values = update_values;
    if (times.size() != values.size()) {
      std::ostringstream oss;
      oss << "Parameter " << id
          << " has mismatched times/values vector sizes: times="
          << times.size() << ", values=" << values.size();
      throw std::runtime_error(oss.str());
    }
    // Enforce strictly increasing time grid to avoid zero denominators.
    for (std::size_t i = 1; i < times.size(); ++i) {
      if (!(times[i] > times[i - 1])) {
        std::ostringstream oss;
        oss << "Parameter " << id
            << " has non-increasing time grid at index " << i
            << " (t[i-1]=" << times[i - 1]
            << ", t[i]=" << times[i] << ")";
        throw std::runtime_error(oss.str());
      }
    }
    cycle_period = update_times.back() - update_times[0];
    if (!std::isfinite(cycle_period) || cycle_period <= 0.0) {
      std::ostringstream oss;
      oss << "Parameter " << id
          << " has non-positive or non-finite cycle_period=" << cycle_period;
      throw std::runtime_error(oss.str());
    }
    is_constant = false;
  }
}

double Parameter::get(double time) {
  // Return the constant value if parameter is constant
  if (is_constant) {
    return value;
  }

  // Determine the time within this->times (necessary to extrapolate)
  double rtime;

  if (is_periodic == true) {
    rtime = fmod(time, cycle_period);
  } else {
    // this->times is not periodic when running with external solver
    rtime = time;
  }

  // Determine the lower and upper element for interpolation
  auto i = lower_bound(times.begin(), times.end(), rtime);
  int k = i - times.begin();

  if (i == times.end()) {
    --i;
  } else if (*i == rtime) {
    return values[k];
  }
  int m = k ? k - 1 : 1;

  // Perform linear interpolation
  // TODO: Implement periodic cubic spline
  const double denom = times[k] - times[m];
  if (!std::isfinite(denom) || std::fabs(denom) <= 1e-12) {
    std::ostringstream oss;
    oss << "Parameter " << id
        << " has invalid time grid for interpolation: "
        << "times[k]=" << times[k]
        << ", times[m]=" << times[m]
        << ", denom=" << denom;
    throw std::runtime_error(oss.str());
  }
  const double slope = (values[k] - values[m]) / denom;
  return values[m] + slope * (rtime - times[m]);
}

void Parameter::to_steady() {
  if (is_constant) {
    return;
  }

  value = std::accumulate(values.begin(), values.end(), 0.0) / double(size);
  is_constant = true;
  steady_converted = true;
}

void Parameter::to_unsteady() {
  if (steady_converted) {
    is_constant = false;
    steady_converted = false;
  }
}
