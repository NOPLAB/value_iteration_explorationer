// SPDX-FileCopyrightText: 2026 nop and CIT autonomous robot laboratory
// SPDX-License-Identifier: BSD-3-Clause

#ifndef VALUE_ITERATION_EXPLORATIONER_EXPLORATION_HPP_
#define VALUE_ITERATION_EXPLORATIONER_EXPLORATION_HPP_

#include "value_iteration_explorationer/global_planner.hpp"

#include <cstdint>
#include <deque>
#include <map>
#include <vector>

namespace value_iteration_explorationer {

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

struct ExplorationGoal {
  Observation pose{};
  double radius = 0.3;
  double angle_deg = 10.0;
  double cost_per_meter = 1.0 / 0.3;
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
  double revisit_penalty = 30.0;
  double switch_penalty = 3.0;
  int free_threshold = 49;
  double information_region_size = 0.05;
  std::size_t max_observed_regions = 100000;
  std::size_t max_visibility_cells = 1000000;
};

// One world-aligned region stores consumed interest, not a visit to a robot pose.
// FIFO eviction bounds memory; evicted evidence may be explored again.
class ObservedRegions {
 public:
  ObservedRegions(double region_size = 0.05, std::size_t capacity = 100000);
  void clear();
  void reconcile(const ExplorationGrid &grid);
  bool observe(const ExplorationGrid &grid, const std::vector<std::size_t> &cells);
  double remaining(const ExplorationGrid &grid, std::size_t cell) const;
  std::size_t size() const { return regions_.size(); }

 private:
  using Key = std::pair<std::int64_t, std::int64_t>;
  struct Evidence { int occupancy; int information; };
  Key key(const ExplorationGrid &grid, std::size_t cell) const;
  double region_size_;
  std::size_t capacity_;
  std::map<Key, Evidence> regions_;
  std::deque<Key> order_;
};

struct VisibleInformationCell {
  std::size_t cell;
  double weight;  // Area and distance weight, before interest/100.
  std::uint64_t headings;
};

struct ObservationView {
  int x;
  int y;
  std::vector<VisibleInformationCell> visible;
  std::vector<double> base_cost;  // Goal, switching and failed-attempt costs only.
};

struct ObservationCandidates {
  std::vector<ObservationView> views;
  bool truncated = false;
};

ObservationCandidates observation_candidates(
    const ExplorationGrid &grid, double robot_x, double robot_y, int headings,
    const ExplorationOptions &options, const std::vector<Observation> &failed,
    const Observation *preferred = nullptr, const ExplorationGoal *goal = nullptr);
// Re-evaluate cached footprints against current evidence without casting rays.
std::vector<GlobalPlanner::Terminal> consume_observation_terminals(
    const ObservationCandidates &candidates, const ExplorationGrid &grid,
    const ExplorationOptions &options, const ObservedRegions &observed);
// Only measured, valid scan rays are evidence. Unknown/occupied cells stop rays.
std::vector<std::size_t> scan_observation_cells(
    const ExplorationGrid &grid, const Observation &sensor,
    const std::vector<float> &ranges, double angle_min, double angle_increment,
    double range_min, double range_max, const ExplorationOptions &options);

// Raw occupancy controls visibility; inflated costmap controls robot motion.
bool exploration_grid_valid(const ExplorationGrid &grid);
double observation_gain(const ExplorationGrid &grid, const Observation &pose,
                        const ExplorationOptions &options);
std::vector<GlobalPlanner::Terminal> observation_terminals(
    const ExplorationGrid &grid, double robot_x, double robot_y, int headings,
    const ExplorationOptions &options, const std::vector<Observation> &failed,
    const Observation *preferred = nullptr, const ExplorationGoal *goal = nullptr);
std::vector<GlobalPlanner::Terminal> goal_terminals(
    const ExplorationGrid &grid, int headings, const ExplorationGoal &goal);
bool exploration_goal_reached(const Observation &pose, const ExplorationGoal &goal);
bool exploration_progressed(const Observation &from, const Observation &to, double distance);
// Permit retreat from an existing clearance violation, never deeper penetration.
bool scan_motion_clear(const std::vector<Observation> &obstacles, const Observation &pose,
                       double forward, double clearance);
// Conservative supercover traversal of every touched grid cell.
bool clear_segment(const std::vector<std::int8_t> &cells, int width, int height,
                   double x0, double y0, double x1, double y1, int free_threshold);

}  // namespace value_iteration_explorationer

#endif
