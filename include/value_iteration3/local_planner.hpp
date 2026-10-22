// SPDX-FileCopyrightText: 2026 nop and CIT autonomous robot laboratory
// SPDX-License-Identifier: BSD-3-Clause

#ifndef VALUE_ITERATION3_LOCAL_PLANNER_HPP_
#define VALUE_ITERATION3_LOCAL_PLANNER_HPP_

#include "value_iteration3/planner.hpp"

#include <atomic>
#include <cstdint>
#include <vector>

namespace value_iteration3 {

inline int heading_index(double yaw_rad, int t_resolution_deg) {
  constexpr double kPi = 3.14159265358979323846;
  const int degrees = static_cast<int>(180.0 * yaw_rad / kPi);
  int wrapped = (degrees + 360 * 100) % 360;
  if (wrapped < 0) {
    wrapped += 360;
  }
  if (t_resolution_deg <= 0) {
    return 0;
  }
  return wrapped / t_resolution_deg;
}

// Tracks the robot window, folds laser hits into the value function, and
// turns the resulting action into a velocity command.
class LocalPlanner {
 public:
  struct Command {
    double linear_x = 0;
    double angular_z = 0;
    bool arrived = false;
    bool have_action = false;
  };

  LocalPlanner(Planner &planner, double range_m);

  void set_range(double range_m);
  bool apply_scan(const std::vector<float> &ranges, float angle_min, float angle_increment,
                  float range_min, float range_max, double x, double y, double yaw);
  SolveResult update(const std::atomic<std::uint32_t> &epoch, std::uint32_t ticket);
  Command command(double x, double y, double yaw) const;

 private:
  void set_window(double x, double y);
  bool in_window(int ix, int iy) const;
  bool write_local(int ix, int iy, std::uint64_t neu);

  Planner &planner_;
  double range_m_ = 1.0;
  int win_x0_ = 0;
  int win_x1_ = -1;
  int win_y0_ = 0;
  int win_y1_ = -1;
};

}  // namespace value_iteration3

#endif
