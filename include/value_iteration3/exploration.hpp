// SPDX-FileCopyrightText: 2026 nop and CIT autonomous robot laboratory
// SPDX-License-Identifier: BSD-3-Clause

#ifndef VALUE_ITERATION3_EXPLORATION_HPP_
#define VALUE_ITERATION3_EXPLORATION_HPP_

#include "value_iteration3/global_planner.hpp"

#include <cstdint>
#include <vector>

namespace value_iteration3 {

struct ExplorationGrid {
  int width = 0;
  int height = 0;
  double resolution = 0;
  double origin_x = 0;
  double origin_y = 0;
  std::vector<std::int8_t> costmap;
  std::vector<std::int8_t> information;
  std::vector<std::int8_t> occupancy;
};

struct Observation {
  double x;
  double y;
  double yaw;
};

struct ExplorationOptions {
  double sensor_range = 3.0;
  double sensor_fov_deg = 180.0;
  double candidate_spacing = 0.3;
  double observation_radius = 0.2;
  double min_gain = 0.02;       // Square meters, weighted by information / 100.
  double gain_reference = 1.0;  // Fixed square-meter scale, never map-relative.
  double information_cost = 30.0;
  double revisit_radius = 0.75;
  double revisit_angle_deg = 60.0;
  double switch_penalty = 3.0;
  int free_threshold = 49;
};

// Raw occupancy controls visibility; inflated costmap controls robot motion.
bool exploration_grid_valid(const ExplorationGrid &grid);
double observation_gain(const ExplorationGrid &grid, const Observation &pose,
                        const ExplorationOptions &options);
std::vector<GlobalPlanner::Terminal> observation_terminals(
    const ExplorationGrid &grid, double robot_x, double robot_y, int headings,
    const ExplorationOptions &options, const std::vector<Observation> &recent,
    const Observation *preferred = nullptr);

// Conservative supercover traversal of every touched grid cell.
bool clear_segment(const std::vector<std::int8_t> &cells, int width, int height,
                   double x0, double y0, double x1, double y1, int free_threshold);

}  // namespace value_iteration3

#endif
