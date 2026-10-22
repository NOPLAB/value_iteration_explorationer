// SPDX-FileCopyrightText: 2026 nop and CIT autonomous robot laboratory
// SPDX-License-Identifier: BSD-3-Clause

#ifndef VALUE_ITERATION3_MAP_LAYERS_HPP_
#define VALUE_ITERATION3_MAP_LAYERS_HPP_

#include <cstdint>
#include <vector>

namespace value_iteration3 {

struct MapGeometry {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  double resolution = 0;
  double origin_x = 0;
  double origin_y = 0;
  double origin_z = 0;
  double qx = 0;
  double qy = 0;
  double qz = 0;
  double qw = 1;
};

struct MapOptions {
  int free_threshold = 49;
  double robot_radius = 0.15;
  double safety_margin = 0.05;
  double inflation_radius = 0.5;
  int inflation_cost = 30;
  int unknown_value = 100;
  int ambiguous_value = 100;
};

struct MapLayers {
  std::vector<std::int8_t> costmap;
  std::vector<std::int8_t> information_map;
};

// Throws std::invalid_argument for invalid geometry, options, or occupancy.
// Geometry is not resampled. Costs are cost_add values: -1 blocked, 0..100 extra.
MapLayers process_map_layers(const MapGeometry &geometry,
                            const std::vector<std::int8_t> &occupancy,
                            const MapOptions &options = MapOptions{});

}  // namespace value_iteration3

#endif  // VALUE_ITERATION3_MAP_LAYERS_HPP_
