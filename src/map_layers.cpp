// SPDX-FileCopyrightText: 2026 nop and CIT autonomous robot laboratory
// SPDX-License-Identifier: BSD-3-Clause

#include "value_iteration_explorationer/map_layers.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace value_iteration_explorationer {

MapLayers process_map_layers(const MapGeometry &g,
                            const std::vector<std::int8_t> &occupancy,
                            const MapOptions &o) {
  const auto score = [](int v) { return v >= 0 && v <= 100; };
  const auto radius = [](double v) { return std::isfinite(v) && v >= 0; };
  if (!g.width || !g.height || !std::isfinite(g.resolution) || g.resolution <= 0 ||
      !std::isfinite(g.origin_x) || !std::isfinite(g.origin_y) ||
      g.origin_z != 0 || g.qx != 0 || g.qy != 0 || g.qz != 0 || g.qw != 1 ||
      !std::isfinite(g.width * g.resolution) ||
      !std::isfinite(g.height * g.resolution) ||
      !std::isfinite(g.origin_x + g.width * g.resolution) ||
      !std::isfinite(g.origin_y + g.height * g.resolution) ||
      g.width > std::numeric_limits<std::size_t>::max() / g.height ||
      occupancy.size() != static_cast<std::size_t>(g.width) * g.height) {
    throw std::invalid_argument("invalid axis-aligned planar map geometry or data size");
  }
  if (!score(o.free_threshold) || !score(o.inflation_cost) ||
      !score(o.unknown_value) || !score(o.ambiguous_value) ||
      !radius(o.robot_radius) || !radius(o.safety_margin) ||
      !radius(o.inflation_radius) || !std::isfinite(o.robot_radius + o.safety_margin)) {
    throw std::invalid_argument("invalid map options");
  }
  MapLayers layers;
  layers.costmap.resize(occupancy.size(), 0);
  layers.information_map.resize(occupancy.size());
  for (std::size_t i = 0; i < occupancy.size(); ++i) {
    const int raw = occupancy[i];
    if (raw < -1 || raw > 100) {
      throw std::invalid_argument("occupancy must be in [-1, 100]");
    }
    const double p = raw / 100.0;
    layers.information_map[i] = static_cast<std::int8_t>(
        raw == -1 ? o.unknown_value : std::lround(o.ambiguous_value * 4 * p * (1 - p)));
  }
  const double footprint = o.robot_radius + o.safety_margin;
  const double reach = std::max(footprint, o.inflation_radius);
  // Include the half-cell extent of each blocking square. Cap before converting
  // to integers so very large valid radii cannot overflow a loop bound.
  const auto span = [reach, &g](std::uint32_t dimension) {
    return static_cast<std::int64_t>(std::min(
        static_cast<double>(dimension), std::ceil(reach / g.resolution + 0.5)));
  };
  const auto sx = span(g.width);
  const auto sy = span(g.height);
  for (std::int64_t y = 0; y < g.height; ++y) {
    for (std::int64_t x = 0; x < g.width; ++x) {
      const auto i = static_cast<std::size_t>(y) * g.width + x;
      if (occupancy[i] == -1) {
        layers.costmap[i] = -1;
        continue;
      }
      double clearance = std::min({x + 0.5, y + 0.5, g.width - x - 0.5,
                                   g.height - y - 0.5}) * g.resolution;
      for (auto by = std::max<std::int64_t>(0, y - sy);
           by <= std::min<std::int64_t>(g.height - 1, y + sy); ++by) {
        for (auto bx = std::max<std::int64_t>(0, x - sx);
             bx <= std::min<std::int64_t>(g.width - 1, x + sx); ++bx) {
          const int raw = occupancy[static_cast<std::size_t>(by) * g.width + bx];
          if (raw > o.free_threshold || (raw == -1 && o.inflate_around_unknown)) {
            const double dx = std::max(0.0, std::abs(static_cast<double>(x - bx)) - 0.5);
            const double dy = std::max(0.0, std::abs(static_cast<double>(y - by)) - 0.5);
            clearance = std::min(clearance, std::hypot(dx, dy) * g.resolution);
          }
        }
      }
      if (clearance <= footprint) {
        layers.costmap[i] = -1;
      } else if (clearance < o.inflation_radius && o.inflation_radius > footprint) {
        layers.costmap[i] = static_cast<std::int8_t>(std::lround(
            o.inflation_cost * (o.inflation_radius - clearance) /
            (o.inflation_radius - footprint)));
      }
    }
  }
  return layers;
}

}  // namespace value_iteration_explorationer
