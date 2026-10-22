// SPDX-FileCopyrightText: 2026 nop and CIT autonomous robot laboratory
// SPDX-License-Identifier: BSD-3-Clause

#include "value_iteration3/exploration.hpp"
#include "value_iteration3/local_planner.hpp"

#include <atomic>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {
int failures = 0;
void expect(bool condition, const std::string &name) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL " << name << "\n";
  }
}

value_iteration3::ExplorationGrid make_grid() {
  value_iteration3::ExplorationGrid grid;
  grid.width = 24;
  grid.height = 16;
  grid.resolution = 0.1;
  grid.origin_x = 3;
  grid.origin_y = -2;
  grid.costmap.assign(384, 0);
  grid.occupancy.assign(384, 0);
  grid.information.assign(384, 0);
  return grid;
}
}  // namespace

int main() {
  using namespace value_iteration3;
  auto grid = make_grid();
  ExplorationOptions options;
  options.sensor_range = 1;
  options.sensor_fov_deg = 90;
  options.candidate_spacing = 0.2;
  options.min_gain = 0.001;
  const Observation pose{3.45, -1.25, 0};
  grid.information[7 * 24 + 10] = 100;
  grid.occupancy[7 * 24 + 10] = 50;
  grid.costmap[7 * 24 + 10] = -1;
  const double ambiguity = observation_gain(grid, pose, options);
  expect(ambiguity > 0, "ambiguous wall itself is observable");
  expect(observation_gain(grid, {pose.x, pose.y, 3.141592653589793}, options) == 0,
         "sensor heading affects information gain");
  grid.occupancy[7 * 24 + 10] = -1;
  expect(observation_gain(grid, pose, options) == ambiguity, "unknown also has information");
  grid.occupancy[7 * 24 + 7] = 100;
  expect(observation_gain(grid, pose, options) == 0, "wall occludes information behind it");
  grid.occupancy[7 * 24 + 7] = 0;
  auto terminals = observation_terminals(grid, pose.x, pose.y, 12, options, {});
  expect(!terminals.empty(), "safe observation terminals generated");
  bool terminals_safe = true;
  for (const auto &terminal : terminals) {
    terminals_safe = terminals_safe && grid.costmap[terminal.y * 24 + terminal.x] >= 0;
  }
  expect(terminals_safe, "information targets are not robot destinations");
  options.revisit_radius = 100;
  options.revisit_angle_deg = 180;
  expect(observation_terminals(grid, pose.x, pose.y, 12, options, {pose}).empty(),
         "recent observation suppresses repeated candidates");
  options.revisit_radius = 0.75;
  for (int y = 0; y < grid.height; ++y) {
    grid.costmap[y * 24 + 12] = -1;
  }
  terminals = observation_terminals(grid, pose.x, pose.y, 12, options, {});
  bool reachable_only = true;
  for (const auto &terminal : terminals) {
    reachable_only = reachable_only && terminal.x < 12;
  }
  expect(reachable_only, "disconnected region has no observation terminals");
  auto invalid = grid;
  invalid.information.clear();
  expect(!exploration_grid_valid(invalid), "reject mismatched grids");
  expect(observation_terminals(invalid, pose.x, pose.y, 12, options, {}).empty(),
         "malformed layers produce no policy");
  expect(observation_terminals(grid, -100, -100, 12, options, {}).empty(),
         "robot outside map produces no policy");
  std::vector<std::int8_t> diagonal(9, 0);
  diagonal[1] = -1;
  expect(!clear_segment(diagonal, 3, 3, 0.5, 0.5, 2.5, 2.5, 100),
         "supercover rejects diagonal corner cutting");

  Planner planner(1);
  expect(planner.set_actions({{"forward", 0.3, 0}, {"back", -0.3, 0},
                             {"left", 0, 90}, {"right", 0, -90}}), "actions");
  planner.set_swept_collision_check(true);
  auto load = [&] {
    return planner.load_map(24, 16, 0.1, 3, -2, 0, 0, 0, 1, grid.costmap,
                            12, 0, 0, 0.4, 30, MapType::CostAdd);
  };
  expect(load(), "load exploration costmap");
  GlobalPlanner global(planner);
  std::atomic<std::uint32_t> epoch{1};
  std::vector<GlobalPlanner::Terminal> goals;
  for (int t = 0; t < 12; ++t) {
    goals.push_back({19, 7, t, 8});
    goals.push_back({4, 7, t, 3});
  }
  expect(global.prepare_terminals(goals), "install multiple nonzero terminals");
  expect(global.solve(epoch, 1).converged, "multiple-terminal value iteration converges");
  auto field = planner.view();
  expect(field && field->cost[field->index(4, 7, 0)] == 3 * Scale::prob_base,
         "information terminal cost is preserved");
  expect(field && field->cost[field->index(19, 7, 0)] == 8 * Scale::prob_base,
         "second information terminal cost is preserved");
  global.prepare_goal(4.95, -1.25, 0);
  expect(global.solve(epoch, 1).converged, "wall-separated solve converges");
  field = planner.view();
  expect(field && field->cost[field->index(4, 7, 0)] == Scale::max_cost,
         "long actions cannot jump across a one-cell wall");
  for (int y = 0; y < grid.height; ++y) {
    grid.costmap[y * 24 + 12] = 0;
  }
  expect(load(), "reload changed map with cached transitions");
  global.prepare_goal(4.95, -1.25, 0);
  expect(global.solve(epoch, 1).converged, "opened map converges");
  field = planner.view();
  expect(field && field->cost[field->index(4, 7, 0)] < Scale::max_cost,
         "newly free cells become reachable after map update");
  expect(!global.prepare_terminals({{12, 7, 12, 0}, {-1, 0, 0, 0},
                                    {1, 1, 0, std::numeric_limits<double>::quiet_NaN()}}),
         "invalid terminal list is rejected");
  expect(!planner.view(), "removing terminals invalidates stale policy");
  std::cout << "exploration_test " << (failures == 0 ? "ok" : "failed") << "\n";
  return failures == 0 ? 0 : 1;
}
