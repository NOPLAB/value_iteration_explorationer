// SPDX-FileCopyrightText: 2026 nop and CIT autonomous robot laboratory
// SPDX-License-Identifier: BSD-3-Clause

#include "value_iteration3/map_layers.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const std::string &text) {
  if (!condition) {
    std::cerr << "FAIL " << text << "\n";
    ++failures;
  }
}

template <typename F>
void rejects(F run, const std::string &text) {
  try {
    run();
    expect(false, text);
  } catch (const std::invalid_argument &) {
  }
}

}  // namespace

int main() {
  using value_iteration3::MapGeometry;
  using value_iteration3::MapOptions;
  using value_iteration3::process_map_layers;
  MapGeometry geometry{9, 9, 1.0, 12.5, -3.0};
  MapOptions options;
  options.free_threshold = 25;
  options.robot_radius = 0;
  options.safety_margin = 0;
  options.inflation_radius = 0;
  std::vector<std::int8_t> raw(81, 0);
  raw[20] = -1;
  raw[21] = 25;
  raw[22] = 26;
  raw[23] = 50;
  raw[24] = 100;
  auto layers = process_map_layers(geometry, raw, options);
  expect(layers.costmap[20] == -1, "unknown blocked");
  expect(layers.costmap[21] == 0, "threshold inclusive");
  expect(layers.costmap[22] == -1 && layers.costmap[23] == -1 &&
         layers.costmap[24] == -1, "uncertain and occupied blocked");
  expect(layers.information_map[20] == 100, "unknown score");
  expect(layers.information_map[21] == 75, "fixed probability score");
  expect(layers.information_map[23] == 100, "maximum ambiguity");
  expect(layers.information_map[24] == 0 && layers.information_map[40] == 0,
         "certain occupied and free scores");
  options.unknown_value = 70;
  options.ambiguous_value = 80;
  layers = process_map_layers(geometry, raw, options);
  expect(layers.information_map[20] == 70 && layers.information_map[23] == 80 &&
         layers.information_map[21] == 60, "configured fixed scaling");
  raw.assign(81, 0);
  raw[40] = 100;
  options.robot_radius = 0.6;
  options.inflation_radius = 2.5;
  layers = process_map_layers(geometry, raw, options);
  expect(layers.costmap[40] == -1 && layers.costmap[41] == -1,
         "circular footprint intersects obstacle square");
  expect(layers.costmap[50] >= 0, "diagonal beyond circular footprint");
  expect(layers.costmap[42] > 0 && layers.costmap[42] <= options.inflation_cost,
         "soft clearance cost beyond footprint");
  expect(layers.costmap[0] == -1 && layers.costmap[8] == -1 &&
         layers.costmap[72] == -1 && layers.costmap[80] == -1,
         "map exterior blocks footprint on every edge");
  options.robot_radius = 0.1;
  options.safety_margin = 0.5;
  expect(process_map_layers(geometry, raw, options).costmap == layers.costmap,
         "safety margin contributes to circular footprint");
  raw.assign(81, 0);
  options = MapOptions{};
  options.robot_radius = 0;
  options.safety_margin = 0;
  options.inflation_radius = 0;
  auto translated = geometry;
  translated.origin_x = -400;
  translated.origin_y = 200;
  expect(process_map_layers(translated, raw, options).costmap ==
         process_map_layers(geometry, raw, options).costmap, "translation preserves raster");
  rejects([&] { process_map_layers(geometry, {}, options); }, "data size");
  for (int value : {-128, -2, 101, 127}) {
    raw[40] = static_cast<std::int8_t>(value);
    rejects([&] { process_map_layers(geometry, raw, options); }, "invalid occupancy");
  }
  raw[40] = 0;
  for (double value : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) {
    auto invalid = geometry;
    invalid.resolution = value;
    rejects([&] { process_map_layers(invalid, raw, options); }, "invalid resolution");
  }
  for (int field = 0; field < 7; ++field) {
    auto invalid = geometry;
    switch (field) {
      case 0: invalid.width = 0; break;
      case 1: invalid.origin_x = std::numeric_limits<double>::infinity(); break;
      case 2: invalid.origin_y = std::numeric_limits<double>::quiet_NaN(); break;
      case 3: invalid.origin_z = 0.1; break;
      case 4: invalid.qx = 0.1; break;
      case 5: invalid.qz = std::sin(0.2); break;
      case 6: invalid.qw = 0; break;
    }
    rejects([&] { process_map_layers(invalid, raw, options); }, "invalid origin/dimensions");
  }
  options.robot_radius = -1;
  rejects([&] { process_map_layers(geometry, raw, options); }, "negative footprint");
  options = MapOptions{};
  options.unknown_value = 101;
  rejects([&] { process_map_layers(geometry, raw, options); }, "invalid score");
  raw.assign(81, 35);
  raw[40] = 50;
  raw[60] = -1;
  const auto startup = process_map_layers(geometry, raw);
  expect(startup.costmap[20] >= 0, "default accepts initial Cartographer free space at 35");
  expect(startup.costmap[40] == -1 && startup.costmap[60] == -1,
         "default still blocks ambiguous occupancy and unknown");
  expect(startup.information_map[20] > 0,
         "initial free space retains its observation interest");
  std::cout << "map layers: " << failures << " failures\n";
  return failures == 0 ? 0 : 1;
}
