// SPDX-FileCopyrightText: 2026 nop and CIT autonomous robot laboratory
// SPDX-License-Identifier: BSD-3-Clause

#include "value_iteration_explorationer/exploration.hpp"
#include "value_iteration_explorationer/local_planner.hpp"

#include <atomic>
#include <algorithm>
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

value_iteration_explorationer::ExplorationGrid make_grid() {
  value_iteration_explorationer::ExplorationGrid grid;
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

std::vector<std::size_t> measured_circle(
    const value_iteration_explorationer::ExplorationGrid &grid,
    const value_iteration_explorationer::Observation &pose,
    const value_iteration_explorationer::ExplorationOptions &options) {
  return value_iteration_explorationer::scan_observation_cells(
      grid, pose, std::vector<float>(720, static_cast<float>(options.sensor_range)),
      -3.141592653589793, 3.141592653589793 / 360, .01, options.sensor_range + .01, options);
}
}  // namespace

int main() {
  using namespace value_iteration_explorationer;
  const Observation origin{0, 0, 0};
  expect(scan_motion_clear({{.16, 0, 0}}, origin, -.2, .2),
         "retreat may strictly increase existing wall clearance");
  expect(!scan_motion_clear({{.16, 0, 0}}, origin, .3, .2),
         "recovery cannot move deeper into an existing clearance violation");
  expect(!scan_motion_clear({{.16, 0, 0}}, origin, 0, .2),
         "near-wall recovery does not authorize in-place rotation");
  expect(!scan_motion_clear({{.16, 0, 0}, {-.35, 0, 0}}, origin, -.2, .2),
         "retreat must not enter a different obstacle's clearance");
  expect(!scan_motion_clear({{.16, 0, 0}, {-.16, 0, 0}}, origin, -.2, .2),
         "recovery must increase clearance from every close obstacle");
  expect(!scan_motion_clear({{.3, 0, 0}}, origin, .6, .2),
         "clear endpoints cannot hide an obstacle crossed by the sweep");
  expect(scan_motion_clear({{.16, 0, 0}, {.16, .02, 0}}, origin, -.2, .2),
         "retreat clears a sampled wall without weakening its clearance");
  {
    Planner footprint(1);
    std::vector<std::int8_t> free_map(1600, 0);
    expect(footprint.load_map(40, 40, .05, 0, 0, 0, 0, 0, 1, free_map,
                              12, 0, 0, .1, 10, MapType::CostAdd),
           "load scan-footprint regression");
    LocalPlanner local_footprint(footprint, .1);
    expect(!local_footprint.apply_scan({.3f}, 0, 1, .05f, 5, .5, .5, 0),
           "legacy two-cell hit inflation stays outside this local window");
    local_footprint.set_obstacle_radius(.3);
    expect(local_footprint.apply_scan({.3f}, 0, 1, .05f, 5, .5, .5, 0),
           "configured metric footprint expands scan hits into the protected window");
  }
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
  {
    auto region_grid = grid;
    ObservedRegions observed(.05, 100);
    const auto cached = observation_candidates(region_grid, pose.x, pose.y, 12, options, {});
    const auto before = consume_observation_terminals(cached, region_grid, options, observed);
    expect(!before.empty(), "unobserved region supplies overlapping candidate views");
    expect(std::any_of(before.begin(), before.end(), [](const auto &terminal) {
      return terminal.x == 19 && terminal.y == 7;
    }), "the same interest region supports a pose far outside the old pose-consumption radius");
    const auto short_return = scan_observation_cells(
        region_grid, pose, {.2f}, 0, 1, .01, 2, options);
    observed.observe(region_grid, short_return);
    expect(observed.remaining(region_grid, 7 * 24 + 10) == 1,
           "measured return does not consume unknown interest behind it");
    auto occluded = region_grid;
    occluded.occupancy[7 * 24 + 7] = 100;
    observed.observe(occluded, scan_observation_cells(
        occluded, pose, {.9f}, 0, 1, .01, 2, options));
    expect(observed.remaining(region_grid, 7 * 24 + 10) == 1,
           "actual ray cannot consume information through a mapped wall");
    occluded.occupancy[7 * 24 + 7] = -1;
    observed.observe(occluded, scan_observation_cells(
        occluded, pose, {.9f}, 0, 1, .01, 2, options));
    expect(observed.remaining(region_grid, 7 * 24 + 10) == 1,
           "first unknown cell stops actual evidence too");
    const auto measured = scan_observation_cells(
        region_grid, pose, {.8f}, 0, 1, .01, 2, options);
    expect(observed.observe(region_grid, measured), "valid return records visible map region");
    expect(!observed.observe(region_grid, measured), "duplicate scans do not accumulate evidence");
    expect(consume_observation_terminals(cached, region_grid, options, observed).empty(),
           "same region is consumed for all poses, including delayed cached candidates");
    region_grid.information[8 * 24 + 10] = 100;
    const auto changed = observation_candidates(region_grid, pose.x, pose.y, 12, options, {});
    const auto new_view = consume_observation_terminals(changed, region_grid, options, observed);
    expect(std::any_of(new_view.begin(), new_view.end(), [](const auto &terminal) {
      return terminal.x == 4 && terminal.y == 7 && terminal.heading == 0;
    }), "same robot pose remains eligible when it sees a new information region");
    const auto raw_new_view = observation_terminals(region_grid, pose.x, pose.y, 12, options, {});
    const auto find_cost = [](const auto &list) {
      const auto found = std::find_if(list.begin(), list.end(), [](const auto &terminal) {
        return terminal.x == 4 && terminal.y == 7 && terminal.heading == 0;
      });
      return found == list.end() ? -1.0 : found->cost;
    };
    expect(find_cost(new_view) > find_cost(raw_new_view),
           "partly overlapping view pays for remaining gain, not its already seen region");
    auto newer = region_grid;
    newer.occupancy[7 * 24 + 10] = 35;
    observed.reconcile(newer);
    expect(observed.remaining(newer, 7 * 24 + 10) == 1,
           "changed occupancy invalidates older consumed evidence");
    newer = region_grid;
    observed.reconcile(newer);
    expect(observed.remaining(newer, 7 * 24 + 10) == 1,
           "returning to an older occupancy value does not revive older evidence");
    observed.observe(region_grid, measured);
    newer.information[7 * 24 + 10] = 50;
    observed.reconcile(newer);
    expect(observed.remaining(newer, 7 * 24 + 10) == .5,
           "changed interest has new evidence identity, even without an occupancy-class change");

    // A 360-degree real LDS consumes rear evidence even when prediction uses 90 degrees.
    auto rear = make_grid();
    rear.information[7 * 24 + 1] = 100;
    ObservedRegions rear_seen;
    rear_seen.observe(rear, scan_observation_cells(
        rear, pose, {.5f}, 3.141592653589793, 1, .01, 2, options));
    expect(rear_seen.remaining(rear, 7 * 24 + 1) == 0,
           "actual LDS bearing is not clipped by the hypothetical planning FOV");
    expect(scan_observation_cells(rear, pose, {
        std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(), -.1f},
        0, 1, .01, 2, options).empty(), "invalid/no-return ranges do not invent observed regions");
    expect(scan_observation_cells(rear, pose, {0}, 0, 1, 0, 2, options).empty(),
           "zero-length returns do not invent observed area even when range_min is zero");

    auto grown = region_grid;
    grown.width += 2;
    grown.height += 2;
    grown.origin_x -= .1;
    grown.origin_y -= .1;
    grown.costmap.assign(grown.width * grown.height, 0);
    grown.occupancy.assign(grown.width * grown.height, 0);
    grown.information.assign(grown.width * grown.height, 0);
    for (int y = 0; y < region_grid.height; ++y) {
      for (int x = 0; x < region_grid.width; ++x) {
        const auto old_cell = y * region_grid.width + x;
        const auto new_cell = (y + 1) * grown.width + x + 1;
        grown.occupancy[new_cell] = region_grid.occupancy[old_cell];
        grown.information[new_cell] = region_grid.information[old_cell];
      }
    }
    observed.observe(region_grid, measured);
    observed.reconcile(grown);
    expect(observed.remaining(grown, 8 * grown.width + 11) == 0,
           "map growth and shifted origin preserve the same observed world region");
    grown.origin_x += .025;
    observed.reconcile(grown);
    expect(observed.remaining(grown, 8 * grown.width + 11) == 0,
           "subcell origin shifts retain evidence only for the same world region and signature");
    grown.origin_x += .05;
    observed.reconcile(grown);
    expect(observed.remaining(grown, 8 * grown.width + 11) == 1,
           "origin shift into a different world region cannot inherit consumed evidence by index");
    ObservedRegions bounded(.05, 1);
    bounded.observe(region_grid, {7 * 24 + 10, 8 * 24 + 10});
    expect(bounded.size() == 1 && bounded.remaining(region_grid, 7 * 24 + 10) == 1,
           "bounded FIFO evidence eviction permits conservative re-exploration");
    auto limited = options;
    limited.max_visibility_cells = 1;
    const auto budgeted = observation_candidates(region_grid, pose.x, pose.y, 12, limited, {});
    std::size_t supports = 0;
    for (const auto &view : budgeted.views) {
      supports += view.visible.size();
    }
    expect(budgeted.truncated && supports <= 1, "geometry footprint cache obeys its explicit budget");
  }
  options.revisit_radius = 100;
  options.revisit_angle_deg = 180;
  const auto revisited = observation_terminals(grid, pose.x, pose.y, 12, options, {pose});
  expect(revisited.size() == terminals.size(), "failed attempts retain every safe observation terminal");
  bool finite_penalties = !revisited.empty();
  for (std::size_t i = 0; i < revisited.size() && i < terminals.size(); ++i) {
    finite_penalties = finite_penalties && revisited[i].x == terminals[i].x &&
        revisited[i].y == terminals[i].y && revisited[i].heading == terminals[i].heading &&
        revisited[i].cost > terminals[i].cost &&
        revisited[i].cost <= terminals[i].cost + options.revisit_penalty;
  }
  expect(finite_penalties, "failed-target costs remain positive and bounded by the configured maximum");
  const auto repeated = observation_terminals(grid, pose.x, pose.y, 12, options, {pose, pose});
  bool same_penalties = repeated.size() == revisited.size();
  for (std::size_t i = 0; i < repeated.size() && i < revisited.size(); ++i) {
    same_penalties = same_penalties && repeated[i].cost == revisited[i].cost;
  }
  expect(same_penalties, "duplicate failed targets do not accumulate penalties");
  auto no_information = grid;
  std::fill(no_information.information.begin(), no_information.information.end(), 0);
  expect(observation_terminals(no_information, pose.x, pose.y, 12, options, {pose}).empty(),
         "revisit preference does not invent observation gain");
  auto blocked_start = grid;
  blocked_start.costmap[7 * grid.width + 4] = -1;
  expect(observation_terminals(blocked_start, pose.x, pose.y, 12, options, {}).empty(),
         "blocked start is never cleared to force a policy");
  expect(!exploration_progressed(pose, pose, 0.1), "stationary pose is not a new observation");
  expect(!exploration_progressed(pose, {pose.x + 0.001, pose.y, pose.yaw}, 0.1),
         "small pose noise is not progress");
  expect(exploration_progressed(pose, {pose.x + 0.2, pose.y, pose.yaw}, 0.1),
         "translation records observation progress");
  expect(exploration_progressed(pose, {pose.x, pose.y, pose.yaw + 0.3}, 0.1),
         "rotation records observation progress");
  expect(!exploration_progressed(pose, {pose.x, pose.y, pose.yaw + 2 * 3.141592653589793}, 0.1),
         "wrapped yaw does not create phantom progress");
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

  ExplorationGoal destination;
  destination.pose = {4.95, -1.25, 0};
  destination.radius = 0.25;
  destination.angle_deg = 30;
  destination.cost_per_meter = 4;
  const auto biased = observation_terminals(
      grid, pose.x, pose.y, 12, options, {}, nullptr, &destination);
  expect(biased.size() == terminals.size(), "goal bias preserves safe observation candidates");
  bool biased_costs = !biased.empty();
  for (std::size_t i = 0; i < biased.size() && i < terminals.size(); ++i) {
    const auto &terminal = terminals[i];
    const double x = grid.origin_x + (terminal.x + 0.5) * grid.resolution;
    const double y = grid.origin_y + (terminal.y + 0.5) * grid.resolution;
    biased_costs = biased_costs && biased[i].x == terminal.x && biased[i].y == terminal.y &&
        biased[i].heading == terminal.heading &&
        std::fabs(biased[i].cost - terminal.cost -
                  std::hypot(x - destination.pose.x, y - destination.pose.y) * 4) < 1e-9;
  }
  expect(biased_costs, "observation cost includes remaining goal distance in seconds");
  auto destinations = goal_terminals(grid, 12, destination);
  expect(!destinations.empty(), "known goal generates terminals without information");
  bool safe_goals = true;
  for (const auto &terminal : destinations) {
    safe_goals = safe_goals && terminal.cost == 0 &&
        grid.costmap[terminal.y * grid.width + terminal.x] >= 0 &&
        (terminal.heading == 0 || terminal.heading == 11);
  }
  expect(safe_goals, "goal terminals preserve free space and wraparound yaw tolerance");
  auto any_heading_goal = destination;
  any_heading_goal.angle_deg = 180;
  expect(goal_terminals(grid, 12, any_heading_goal).size() == destinations.size() * 6,
         "180-degree tolerance accepts every heading including the antipodal bin");
  auto unknown_goal_grid = grid;
  std::fill(unknown_goal_grid.costmap.begin(), unknown_goal_grid.costmap.end(), -1);
  std::fill(unknown_goal_grid.occupancy.begin(), unknown_goal_grid.occupancy.end(), -1);
  expect(goal_terminals(unknown_goal_grid, 12, destination).empty(),
         "unknown goal never becomes traversable");
  auto outside_goal = destination;
  outside_goal.pose.x = 100;
  expect(goal_terminals(grid, 12, outside_goal).empty(),
         "out-of-map goal creates no unsafe terminal");
  expect(!observation_terminals(grid, pose.x, pose.y, 12, options, {},
                               nullptr, &outside_goal).empty(),
         "outside-map goal retains safe exploration candidates");
  expect(exploration_goal_reached({4.95, -1.25, -0.1}, destination),
         "goal arrival accepts wrapped heading");
  expect(!exploration_goal_reached({4.95, -1.25, 1.0}, destination),
         "position alone does not complete an oriented goal");
  expect(!exploration_goal_reached(pose, destination), "distant goal is not reached");

  const auto known_grid = make_grid();
  ExplorationGoal default_goal;
  default_goal.pose = destination.pose;
  Planner goal_planner(1);
  goal_planner.set_actions({{"forward", 0.3, 0}, {"back", -0.2, 0},
                            {"right", 0, -20}, {"rightfw", 0.2, -20},
                            {"left", 0, 20}, {"leftfw", 0.2, 20}});
  goal_planner.set_swept_collision_check(true);
  expect(goal_planner.load_map(24, 16, 0.1, 3, -2, 0, 0, 0, 1, known_grid.costmap,
                              36, 0, 0, 0.3, 10, MapType::CostAdd),
         "load known map without information");
  GlobalPlanner goal_global(goal_planner);
  std::atomic<std::uint32_t> goal_epoch{1};
  expect(goal_global.prepare_terminals(goal_terminals(known_grid, 36, default_goal)),
         "default goal creates safe terminal region");
  expect(goal_global.solve(goal_epoch, 1).converged, "default goal field converges");
  expect(LocalPlanner(goal_planner, 1).command(pose.x, pose.y, pose.yaw).have_action,
         "default margins propagate goal value across a 10 cm map without information");
  auto observed_grid = known_grid;
  observed_grid.information.assign(observed_grid.costmap.size(), 80);
  auto all_visited = options;
  all_visited.revisit_radius = 100;
  all_visited.revisit_angle_deg = 180;
  const auto revisit_targets = observation_terminals(
      observed_grid, pose.x, pose.y, 36, all_visited, {pose});
  expect(goal_global.prepare_terminals(revisit_targets, true),
          "failed attempts do not remove unobserved-region terminals");
  expect(goal_global.solve(goal_epoch, 1).converged, "failed-attempt field converges");
  expect(LocalPlanner(goal_planner, 1).command(pose.x, pose.y, pose.yaw, true).have_action,
         "failed-attempt penalty still permits ordinary value-based motion");

  {
    auto room = make_grid();
    room.width = 32;
    room.height = 24;
    room.origin_x = room.origin_y = 0;
    room.costmap.assign(32 * 24, 0);
    room.occupancy.assign(32 * 24, 35);
    room.information.assign(32 * 24, 90);
    for (int y = 0; y < 24; ++y) {
      room.costmap[y * 32 + 25] = -1;
      room.occupancy[y * 32 + 25] = 100;
    }
    ExplorationOptions trail_options;
    trail_options.sensor_range = 1;
    trail_options.gain_reference = 0.05;
    trail_options.min_gain = 0.001;
    trail_options.candidate_spacing = 0.2;
    Planner trail_planner(1);
    trail_planner.set_actions({{"forward", .3, 0}, {"back", -.2, 0},
                               {"right", 0, -20}, {"rightfw", .2, -20},
                               {"left", 0, 20}, {"leftfw", .2, 20}});
    trail_planner.set_swept_collision_check(true);
    expect(trail_planner.load_map(32, 24, .1, 0, 0, 0, 0, 0, 1, room.costmap,
                                  36, 0, 0, .2, 30, MapType::CostAdd),
           "load wall-side exploration regression");
    GlobalPlanner trail_global(trail_planner);
    LocalPlanner trail_local(trail_planner, 1);
    std::atomic<std::uint32_t> trail_epoch{1};
    Observation current{1.85, 1.25, 0};
    ObservedRegions trail;
    for (const Observation visit : std::vector<Observation>{
             {1.65, 1.25, 0}, {1.75, 1.25, 0}, current}) {
      trail.observe(room, measured_circle(room, visit, trail_options));
    }
    double previous_linear = 0;
    int reversals = 0;
    bool actionable = true, turned = false, collision_free = true;
    const auto initial_regions = trail.size();
    // Geometry generation is deliberately held back throughout these control steps.
    const auto cached_candidates = observation_candidates(
        room, current.x, current.y, 36, trail_options, {});
    for (int step = 0; step < 20; ++step) {
      auto available = consume_observation_terminals(cached_candidates, room, trail_options, trail);
      trail_global.prepare_terminals(available, true);
      trail_global.solve(trail_epoch, 1);
      const auto command = trail_local.command(current.x, current.y, current.yaw);
      actionable = actionable && command.have_action;
      if (!command.have_action) {
        break;
      }
      if (std::abs(command.angular_z) < .01 && previous_linear * command.linear_x < 0) {
        ++reversals;
      }
      turned = turned || std::abs(command.angular_z) > .01;
      previous_linear = std::abs(command.angular_z) < .01 ? command.linear_x : 0;
      collision_free = collision_free && clear_segment(
          room.costmap, room.width, room.height, current.x / .1, current.y / .1,
          (current.x + command.linear_x * std::cos(current.yaw)) / .1,
          (current.y + command.linear_x * std::sin(current.yaw)) / .1, 100);
      current.x += command.linear_x * std::cos(current.yaw);
      current.y += command.linear_x * std::sin(current.yaw);
      current.yaw += command.angular_z;
      trail.observe(room, measured_circle(room, current, trail_options));
    }
    expect(actionable, "wall-side exploration continues without a manual nudge");
    expect(turned && trail.size() > initial_regions && collision_free,
           "region-based wall approach collects new evidence and respects map obstacles");
    expect(std::abs(current.y - 1.25) > .3,
           "wall-side exploration leaves its original forward-back track");
    std::cout << "region_walk steps=20 initial_regions=" << initial_regions
              << " final_regions=" << trail.size() << " reversals=" << reversals
              << " final_pose=" << current.x << "," << current.y << "," << current.yaw << "\n";
  }

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
  // This coarse 90-degree motion set needs the same 0.4 m margin as load().
  auto solver_goal = destination;
  solver_goal.radius = 0.4;
  expect(global.prepare_terminals(goal_terminals(grid, 12, solver_goal), true),
         "install goal-only exploration terminals");
  expect(global.solve(epoch, 1).converged, "goal-only exploration field converges");
  LocalPlanner goal_local(planner, 1);
  expect(goal_local.command(pose.x, pose.y, pose.yaw).have_action,
         "no-information map still drives toward a supplied goal");
  expect(goal_local.command(destination.pose.x, destination.pose.y, 0).arrived,
         "goal terminal is reachable with the original solver");
  const auto accepted_goal_field = planner.view();
  expect(!global.prepare_terminals({{12, 7, 12, 0}, {-1, 0, 0, 0},
                                    {1, 1, 0, std::numeric_limits<double>::quiet_NaN()}}),
         "invalid terminal list is rejected");
  expect(!planner.view(), "removing terminals invalidates stale policy");
  expect(goal_local.command(*accepted_goal_field, pose.x, pose.y, pose.yaw).have_action,
         "accepted policy remains readable while replacement field is unavailable");

  Planner cold(1);
  cold.set_actions(planner.actions());
  cold.set_swept_collision_check(true);
  GlobalPlanner cold_global(cold);
  int width = 24, height = 16;
  double origin_x = 3, origin_y = -2;
  auto refresh = [&](Planner &target, bool warm) {
    return target.load_map(width, height, 0.1, origin_x, origin_y, 0, 0, 0, 1,
                           grid.costmap, 12, 0, 0, 0.4, 30, MapType::CostAdd, warm);
  };
  auto compare_cold = [&](const std::string &name) {
    expect(refresh(cold, false), name + " cold map");
    cold_global.prepare_terminals(goals);
    expect(cold_global.solve(epoch, 1).converged, name + " cold solve");
    const auto warm_view = planner.view();
    const auto cold_view = cold.view();
    expect(warm_view && cold_view && warm_view->cost == cold_view->cost &&
           warm_view->final_state == cold_view->final_state, name + " matches cold field");
    bool same_policy = true;
    for (std::size_t i = 0; i < warm_view->action.size(); ++i) {
      if (!warm_view->final_state[i]) {
        same_policy = same_policy && warm_view->action[i] == cold_view->action[i];
      }
    }
    expect(same_policy, name + " matches cold nonterminal policy");
  };
  expect(refresh(planner, true), "initial warm map");
  expect(global.prepare_terminals(goals, true), "initial warm terminals");
  expect(global.solve(epoch, 1).converged, "initial warm solve");
  LocalPlanner continuous(planner, 1);
  const auto stop_at_terminal = continuous.command(4.95, -1.25, 0);
  const auto pass_terminal = continuous.command(4.95, -1.25, 0, true);
  expect(stop_at_terminal.arrived && !stop_at_terminal.have_action,
         "legacy commands still stop at terminals");
  expect(pass_terminal.arrived && pass_terminal.have_action,
         "continuous commands retain a value-based action at terminals");
  auto previous = planner.view();
  expect(refresh(planner, true), "unchanged warm map");
  expect(global.prepare_terminals(goals, true), "unchanged warm terminals");
  expect(planner.view() == previous, "snapshot retained during warm preparation");
  const auto retained = global.solve(epoch, 1);
  expect(retained.converged && retained.updates == 0, "unchanged refresh retains all values");
  compare_cold("unchanged refresh");

  for (auto &goal : goals) {
    if (goal.x == 4) {
      goal.cost = 20;
    }
  }
  global.prepare_terminals(goals, true);
  expect(global.solve(epoch, 1).converged, "increased terminal solve");
  compare_cold("increased terminal");
  goals.erase(goals.begin() + 1);  // Remove one heading, leaving the others intact.
  global.prepare_terminals(goals, true);
  expect(global.solve(epoch, 1).converged, "removed terminal solve");
  compare_cold("removed terminal");

  grid.costmap[7 * width + 10] = 40;
  refresh(planner, true);
  global.prepare_terminals(goals, true);
  expect(global.solve(epoch, 1).converged, "increased map cost solve");
  compare_cold("increased map cost");
  for (int y = 0; y < height; ++y) {
    grid.costmap[y * width + 12] = -1;
  }
  goals.clear();
  for (int t = 0; t < 12; ++t) {
    goals.push_back({19, 7, t, 8});
  }
  refresh(planner, true);
  global.prepare_terminals(goals, true);
  expect(global.solve(epoch, 1).converged, "disconnected former terminal region converges");
  compare_cold("disconnected former terminal region");
  expect(planner.view()->cost[planner.view()->index(4, 7, 0)] == Scale::max_cost,
         "removed terminals do not leave finite orphan cycles");

  previous = planner.view();
  std::vector<std::int8_t> grown(26 * 18, -1);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      grown[(y + 1) * 26 + x + 1] = grid.costmap[y * width + x];
    }
  }
  grid.costmap = std::move(grown);
  width = 26;
  height = 18;
  origin_x -= 0.1;
  origin_y -= 0.1;
  for (auto &goal : goals) {
    ++goal.x;
    ++goal.y;
  }
  refresh(planner, true);
  global.prepare_terminals(goals, true);
  expect(planner.view() == previous, "map growth retains old geometry snapshot until publication");
  const auto shifted = global.solve(epoch, 1);
  expect(shifted.converged && shifted.updates == 0,
         "integer origin shift retains world-aligned field values");
  compare_cold("map growth and shifted origin");
  origin_x += 0.025;
  refresh(planner, true);
  global.prepare_terminals(goals, true);
  expect(global.solve(epoch, 1).converged, "fractional origin shift rebuilds working field");
  compare_cold("fractional origin shift");
  previous = planner.view();
  expect(!global.prepare_terminals({}, true), "empty warm terminals reported");
  expect(planner.view() == previous, "empty warm preparation preserves published snapshot");
  expect(global.solve(epoch, 1).converged, "empty warm field converges without count-up");
  bool unreachable = true;
  for (const auto value : planner.view()->cost) {
    unreachable = unreachable && value == Scale::max_cost;
  }
  expect(unreachable, "empty terminal publication removes all old finite values");

  Planner zero_cost(1);
  zero_cost.set_actions({{"left", 0, 90}, {"right", 0, -90}});
  expect(zero_cost.load_map(1, 1, 0.1, 0, 0, 0, 0, 0, 1, {0},
                           4, 0, 0, 0, 0, MapType::CostReplace, true),
         "zero-cost cycle map");
  GlobalPlanner zero_global(zero_cost);
  zero_global.prepare_terminals({{0, 0, 0, 900000000}}, true);
  expect(zero_global.solve(epoch, 1).converged, "large finite terminal converges");
  expect(zero_cost.view()->cost[1] == 900000000ull * Scale::prob_base,
         "weighted large finite values do not overflow");
  zero_global.prepare_terminals({}, true);
  expect(zero_global.solve(epoch, 1).converged, "zero-cost orphan cycle converges");
  for (const auto value : zero_cost.view()->cost) {
    expect(value == Scale::max_cost, "zero-cost tied policies invalidate with removed terminal");
  }
  std::cout << "exploration_test " << (failures == 0 ? "ok" : "failed") << "\n";
  return failures == 0 ? 0 : 1;
}
