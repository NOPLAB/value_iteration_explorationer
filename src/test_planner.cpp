// SPDX-FileCopyrightText: 2026 nop and CIT autonomous robot laboratory
// SPDX-License-Identifier: BSD-3-Clause

#include "value_iteration3/actions.hpp"
#include "value_iteration3/global_planner.hpp"
#include "value_iteration3/local_planner.hpp"
#include "value_iteration3/planner.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace value_iteration3 {
namespace {

constexpr int kWidth = 32;
constexpr int kHeight = 24;
constexpr int kPocketX = 20;
constexpr int kPocketY = 12;

int failures = 0;

void expect(bool condition, const std::string &text) {
  if (!condition) {
    std::cerr << "FAIL " << text << "\n";
    ++failures;
  }
}

std::vector<std::int8_t> make_map() {
  std::vector<std::int8_t> occupancy(kWidth * kHeight, 0);
  for (int y = kPocketY - 8; y <= kPocketY + 8; ++y) {
    for (int x = kPocketX - 8; x <= kPocketX + 8; ++x) {
      if (x == kPocketX && y == kPocketY) {
        continue;
      }
      occupancy[static_cast<std::size_t>(x) + static_cast<std::size_t>(y) * kWidth] = 100;
    }
  }
  return occupancy;
}

bool load(Planner &planner, int threads) {
  planner.set_thread_num(threads);
  return planner.load_map(kWidth, kHeight, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, make_map(), 60, 0.2,
                          30.0, 0.3, 15);
}

}  // namespace

int planner_self_test() {
  const std::string list_path = "action_list_test.yaml";
  {
    std::ofstream out(list_path);
    out << "/**:\n  ros__parameters:\n    online: true\n\naction_list:\n"
           "  - name: right\n    onestep_forward_m: 0.0\n    onestep_rotation_deg: -20.0\n"
           "  - name: left\n    onestep_forward_m: 0.0\n    onestep_rotation_deg: 20.0\n"
           "  - name: forward\n    onestep_forward_m: 0.3\n    onestep_rotation_deg: 0.0\n";
  }
  const ActionList parsed = read_action_list(list_path);
  expect(parsed.error.empty(), parsed.error.empty() ? "action list" : parsed.error);
  expect(parsed.actions.size() == 3, "action count");
  if (parsed.actions.size() == 3) {
    expect(parsed.actions[0].name == "right" && parsed.actions[0].rotate_deg == -20.0, "right");
    expect(parsed.actions[1].name == "left" && parsed.actions[1].forward_m == 0.0, "left");
    expect(parsed.actions[2].forward_m == 0.3, "forward");
  }
  Planner custom(1);
  expect(custom.set_actions(parsed.actions), "install action list");
  expect(!custom.set_actions({}), "reject an empty action list");
  std::remove(list_path.c_str());

  // Cost grids preserve all eight bits and already include obstacle inflation.
  Planner costs(1);
  const std::vector<std::int8_t> raw{0, 1, 100, 127, -128, -56, -2, -1};
  expect(costs.load_map(8, 1, 0.1, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, raw,
                        12, 10.0, 999.0, 0.3, 30, MapType::CostReplace), "load raw costs");
  for (std::size_t i = 0; i < raw.size(); ++i) {
    const auto value = static_cast<std::uint8_t>(raw[i]);
    expect(costs.free_[i] == (value != 255), "raw cost traversability");
    expect(costs.static_pen_[i] == (value == 255 ? 0 : value * Scale::prob_base),
           "raw cost without safety inflation or base cost");
  }
  Planner added(1);
  expect(added.load_map(8, 1, 0.1, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, raw,
                        12, 10.0, 999.0, 0.3, 30, MapType::CostAdd), "load added costs");
  for (std::size_t i = 0; i < raw.size(); ++i) {
    const auto value = static_cast<std::uint8_t>(raw[i]);
    expect(added.free_[i] == (value != 255), "added cost traversability");
    expect(added.static_pen_[i] == (value == 255 ? 0 : (value + 1) * Scale::prob_base),
           "added cost is the base cost plus the raw cost");
  }
  Planner occupancy(1);
  expect(occupancy.load_map(8, 1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, raw,
                            12, 0.0, 30.0, 0.3, 30), "default occupancy mode");
  for (std::size_t i = 0; i < raw.size(); ++i) {
    expect(occupancy.free_[i] == (raw[i] == 0), "occupancy only accepts zero");
  }
  expect(!costs.load_map(8, 1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, {1},
                         12, 0.0, 0.0, 0.3, 30, MapType::CostReplace), "reject short cost grid");

  Planner one(1);
  Planner two(2);
  expect(load(one, 1), "load 1 thread");
  expect(load(two, 2), "load 2 threads");
  if (failures > 0) {
    return 1;
  }

  std::atomic<std::uint32_t> epoch{1};
  GlobalPlanner global_one(one);
  GlobalPlanner global_two(two);
  global_one.prepare_goal(0.10, 0.10, 0);
  global_two.prepare_goal(0.10, 0.10, 0);
  const SolveResult first = global_one.solve(epoch, 1);
  const SolveResult second = global_two.solve(epoch, 1);
  expect(first.converged, "1 thread converged");
  expect(second.converged, "2 threads converged");
  Planner weighted(1);
  auto weighted_map = make_map();
  for (auto &cell : weighted_map) {
    cell = cell == 0 ? 4 : -1;
  }
  expect(weighted.load_map(kWidth, kHeight, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                           weighted_map, 60, 0.2, 30.0, 0.3, 15, MapType::CostReplace),
         "load weighted map");
  GlobalPlanner weighted_global(weighted);
  weighted_global.prepare_goal(0.10, 0.10, 0);
  expect(weighted_global.solve(epoch, 1).converged, "weighted map converged");
  expect(weighted.audit().empty(), "weighted field audit");
  const auto weighted_view = weighted.view();
  expect(weighted_view != nullptr, "weighted view");
  if (weighted_view) {
    expect(weighted_view->cost[weighted_view->index(kWidth - 1, kHeight - 1, 0)] <
               Scale::max_cost, "weighted far cell reached");
    expect(weighted_view->cost[weighted_view->index(kPocketX, kPocketY, 0)] ==
               Scale::max_cost, "cost grid sealed cell stays blocked");
    for (std::size_t xy = 0; xy < weighted.static_pen_.size(); ++xy) {
      if (weighted.free_[xy]) {
        expect(weighted.pen_[weighted.pad_col(static_cast<int>(xy % kWidth),
                                             static_cast<int>(xy / kWidth))] ==
                   4 * Scale::prob_base, "solver uses raw static cost");
      }
    }
  }
  expect(one.audit().empty(), "1 thread " + one.audit());
  expect(two.audit().empty(), "2 threads " + two.audit());

  const auto left = one.view();
  const auto right = two.view();
  expect(left != nullptr && right != nullptr, "views");
  if (left && right) {
    expect(left->cost == right->cost, "thread counts agree on cost");
    expect(left->action == right->action, "thread counts agree on action");
    const std::size_t corner = left->index(kWidth - 1, kHeight - 1, 0);
    expect(left->cost[corner] < Scale::max_cost, "far corner is reached");
    bool pocket_blocked = true;
    for (int it = 0; it < left->nt; ++it) {
      if (left->cost[left->index(kPocketX, kPocketY, it)] != Scale::max_cost) {
        pocket_blocked = false;
      }
    }
    expect(pocket_blocked, "sealed cell stays blocked");
  }

  const std::vector<std::uint64_t> before = left ? left->cost : std::vector<std::uint64_t>{};
  LocalPlanner local(one, 1.0);
  const LocalPlanner::Command corner =
      local.command((kWidth - 1 + 0.5) * 0.05, (kHeight - 1 + 0.5) * 0.05, 0.0);
  expect(corner.have_action, "local command at the far corner");
  const bool painted = local.apply_scan({0.35f}, 0.0f, 0.0f, 0.0f, 10.0f, 0.4, 0.1, 0.0);
  expect(painted, "scan changes a penalty");
  const SolveResult repaired = local.update(epoch, 1);
  expect(repaired.converged, "penalty update converged");
  expect(one.audit().empty(), "after scan " + one.audit());
  const auto after = one.view();
  if (after && !before.empty()) {
    bool raised = false;
    bool decreased = false;
    for (std::size_t i = 0; i < before.size(); ++i) {
      if (after->cost[i] > before[i]) {
        raised = true;
      }
      if (after->cost[i] < before[i]) {
        decreased = true;
      }
    }
    expect(raised, "a penalty raises some cost");
    expect(!decreased, "a penalty does not lower cost");
  }

  epoch.store(2);
  const SolveResult cancelled = global_one.solve(epoch, 1);
  expect(cancelled.cancelled, "a new request cancels the running solve");

  if (failures == 0) {
    std::cout << "planner_test ok rounds " << first.rounds << " " << second.rounds << "\n";
  }
  return failures == 0 ? 0 : 1;
}

}  // namespace value_iteration3

int main() { return value_iteration3::planner_self_test(); }
