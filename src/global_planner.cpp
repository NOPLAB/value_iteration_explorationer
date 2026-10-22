// SPDX-FileCopyrightText: 2026 nop and CIT autonomous robot laboratory
// SPDX-License-Identifier: BSD-3-Clause

#include "value_iteration3/global_planner.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>

namespace value_iteration3 {
namespace {
constexpr std::uint64_t kUnreached = ~std::uint64_t{0};
}

GlobalPlanner::GlobalPlanner(Planner &planner) : planner_(planner) {}

bool GlobalPlanner::heading_is_goal(int ix, int iy, int it) const {
  const Planner &field = planner_;
  const std::size_t xy = static_cast<std::size_t>(ix) + static_cast<std::size_t>(iy) * field.nx_;
  if (!field.free_[xy]) {
    return false;
  }
  const double x0 = ix * field.resolution_ + field.origin_x_;
  const double y0 = iy * field.resolution_ + field.origin_y_;
  const double r0 = (x0 - field.goal_x_) * (x0 - field.goal_x_) + (y0 - field.goal_y_) * (y0 - field.goal_y_);
  const double x1 = x0 + field.resolution_;
  const double y1 = y0 + field.resolution_;
  const double r1 = (x1 - field.goal_x_) * (x1 - field.goal_x_) + (y1 - field.goal_y_) * (y1 - field.goal_y_);
  const double radius2 = field.goal_margin_radius_ * field.goal_margin_radius_;
  if (!(r0 < radius2 && r1 < radius2)) {
    return false;
  }
  const int t0 = it * field.t_resolution_;
  const int t1 = (it + 1) * field.t_resolution_;
  const int yaw_wrap = field.goal_yaw_ > 180 ? field.goal_yaw_ - 360 : field.goal_yaw_ + 360;
  const bool direct =
      field.goal_yaw_ - field.goal_margin_theta_ <= t0 && t1 <= field.goal_yaw_ + field.goal_margin_theta_;
  const bool wrapped = yaw_wrap - field.goal_margin_theta_ <= t0 && t1 <= yaw_wrap + field.goal_margin_theta_;
  return direct || wrapped;
}

void GlobalPlanner::prepare_goal(double x, double y, int yaw_deg) {
  Planner &field = planner_;
  if (!field.map_ready_) {
    return;
  }
  while (yaw_deg < 0) {
    yaw_deg += 360;
  }
  while (yaw_deg >= 360) {
    yaw_deg -= 360;
  }
  field.goal_x_ = x;
  field.goal_y_ = y;
  field.goal_yaw_ = yaw_deg;
  std::fill(field.local_.begin(), field.local_.end(), 0);
  std::fill(field.dirty_.begin(), field.dirty_.end(), 0);
  {
    std::lock_guard<std::mutex> lock(field.view_mu_);
    field.view_.reset();
  }
  field.field_ready_ = false;
  field.goal_set_ = true;

  field.for_rows([this, &field](int y0, int y1) {
    for (int iy = y0; iy < y1; ++iy) {
      for (int ix = 0; ix < field.nx_; ++ix) {
        const std::size_t xy = static_cast<std::size_t>(ix) + static_cast<std::size_t>(iy) * field.nx_;
        const std::uint64_t penalty = field.free_[xy] ? field.static_pen_[xy] : 0;
        const std::int64_t column = field.pad_col(ix, iy);
        for (int it = 0; it < field.nt_; ++it) {
          const bool final_state = heading_is_goal(ix, iy, it);
          const std::size_t original = field.index_of(ix, iy, it);
          const std::size_t pad = static_cast<std::size_t>(column + it);
          field.final_[original] = final_state ? 1 : 0;
          field.pen_[pad] = penalty;
          field.eval_ok_[pad] = field.free_[xy] && !final_state;
          field.cp_[pad].store(field.free_[xy] && final_state ? penalty : kUnreached,
                               std::memory_order_relaxed);
        }
      }
    }
  });

  std::size_t goals = 0;
  for (std::uint8_t flag : field.final_) {
    goals += flag ? 1 : 0;
  }
  std::ostringstream text;
  text << "goal (" << field.goal_x_ << ", " << field.goal_y_ << ", " << field.goal_yaw_ << " deg) cells "
       << goals;
  field.log(text.str());
}

bool GlobalPlanner::prepare_terminals(const std::vector<Terminal> &terminals) {
  Planner &field = planner_;
  if (!field.map_ready_) {
    return false;
  }
  std::fill(field.local_.begin(), field.local_.end(), 0);
  std::fill(field.dirty_.begin(), field.dirty_.end(), 0);
  std::fill(field.final_.begin(), field.final_.end(), 0);
  {
    std::lock_guard<std::mutex> lock(field.view_mu_);
    field.view_.reset();
  }
  field.field_ready_ = false;
  field.goal_set_ = false;
  field.fill_cp(kUnreached);
  field.for_rows([&field](int y0, int y1) {
    for (int iy = y0; iy < y1; ++iy) {
      for (int ix = 0; ix < field.nx_; ++ix) {
        const auto xy = static_cast<std::size_t>(iy) * field.nx_ + ix;
        const auto column = field.pad_col(ix, iy);
        for (int it = 0; it < field.nt_; ++it) {
          const auto pad = static_cast<std::size_t>(column + it);
          field.pen_[pad] = field.free_[xy] ? field.static_pen_[xy] : 0;
          field.eval_ok_[pad] = field.free_[xy];
        }
      }
    }
  });
  for (const Terminal &terminal : terminals) {
    if (terminal.x < 0 || terminal.x >= field.nx_ || terminal.y < 0 ||
        terminal.y >= field.ny_ || terminal.heading < 0 || terminal.heading >= field.nt_ ||
        !std::isfinite(terminal.cost) || terminal.cost < 0 || terminal.cost >= 1e9) {
      continue;
    }
    const auto xy = static_cast<std::size_t>(terminal.y) * field.nx_ + terminal.x;
    if (!field.free_[xy]) {
      continue;
    }
    const auto original = field.index_of(terminal.x, terminal.y, terminal.heading);
    const auto pad = static_cast<std::size_t>(
        field.pad_col(terminal.x, terminal.y) + terminal.heading);
    const auto cost = static_cast<std::uint64_t>(terminal.cost * Scale::prob_base) +
                      field.pen_[pad];
    field.final_[original] = 1;
    field.eval_ok_[pad] = 0;
    field.cp_[pad].store(std::min(cost, field.cp_[pad].load(std::memory_order_relaxed)),
                         std::memory_order_relaxed);
    field.goal_set_ = true;
  }
  return field.goal_set_;
}

SolveResult GlobalPlanner::solve(const std::atomic<std::uint32_t> &epoch, std::uint32_t ticket) {
  Planner &field = planner_;
  if (!field.goal_set_) {
    return {};
  }
  field.zero_masks();
  field.prev_cells_.clear();
  std::vector<std::pair<int, int>> seeds;
  for (int iy = 0; iy < field.ny_; ++iy) {
    for (int ix = 0; ix < field.nx_; ++ix) {
      for (int it = 0; it < field.nt_; ++it) {
        if (field.final_[field.index_of(ix, iy, it)]) {
          seeds.emplace_back(ix, iy);
          break;
        }
      }
    }
  }
  field.candidates_ = field.dilate(seeds);
  const SolveResult result = field.run(true, epoch, ticket);
  if (result.converged) {
    warn_unreached();
  }
  return result;
}

void GlobalPlanner::warn_unreached() const {
  const auto field_view = planner_.view();
  if (!field_view) {
    return;
  }
  const Planner &field = planner_;
  std::size_t free_cells = 0;
  std::size_t bare = 0;
  for (int iy = 0; iy < field.ny_; ++iy) {
    for (int ix = 0; ix < field.nx_; ++ix) {
      const std::size_t xy = static_cast<std::size_t>(ix) + static_cast<std::size_t>(iy) * field.nx_;
      if (!field.free_[xy]) {
        continue;
      }
      ++free_cells;
      bool finite = false;
      for (int it = 0; it < field.nt_; ++it) {
        if (field_view->cost[field.index_of(ix, iy, it)] < Scale::max_cost) {
          finite = true;
          break;
        }
      }
      if (!finite) {
        ++bare;
      }
    }
  }
  if (free_cells > 0 && bare * 2 > free_cells) {
    field.log("most free cells have no finite cost; raise goal_margin_radius");
  }
}

}  // namespace value_iteration3
