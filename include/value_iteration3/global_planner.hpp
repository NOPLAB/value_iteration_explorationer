// SPDX-FileCopyrightText: 2026 nop and CIT autonomous robot laboratory
// SPDX-License-Identifier: BSD-3-Clause

#ifndef VALUE_ITERATION3_GLOBAL_PLANNER_HPP_
#define VALUE_ITERATION3_GLOBAL_PLANNER_HPP_

#include "value_iteration3/planner.hpp"

#include <atomic>
#include <cstdint>

namespace value_iteration3 {

// Puts a goal on the shared field and sweeps it to convergence.
class GlobalPlanner {
 public:
  struct Terminal {
    int x;
    int y;
    int heading;
    double cost;
  };

  explicit GlobalPlanner(Planner &planner);

  void prepare_goal(double x, double y, int yaw_deg);
  // Warm updates keep the published snapshot and unaffected working values.
  bool prepare_terminals(const std::vector<Terminal> &terminals, bool warm_start = false);
  SolveResult solve(const std::atomic<std::uint32_t> &epoch, std::uint32_t ticket);

 private:
  bool heading_is_goal(int ix, int iy, int it) const;
  void warn_unreached() const;

  Planner &planner_;
};

}  // namespace value_iteration3

#endif
