// SPDX-FileCopyrightText: 2026 nop and CIT autonomous robot laboratory
// SPDX-License-Identifier: BSD-3-Clause

#include "value_iteration3/local_planner.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace value_iteration3 {
namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr std::uint64_t kUnreached = ~std::uint64_t{0};
constexpr std::uint64_t kHitPenalty = 2048ull << Scale::prob_base_bit;

std::uint64_t heading_mask(int nt) {
  return nt >= 64 ? kUnreached : ((std::uint64_t{1} << nt) - 1);
}
}  // namespace

LocalPlanner::LocalPlanner(Planner &planner, double range_m)
    : planner_(planner), range_m_(range_m > 0.0 ? range_m : 1.0) {}

void LocalPlanner::set_range(double range_m) {
  if (range_m > 0.0) {
    range_m_ = range_m;
  }
}

bool LocalPlanner::in_window(int ix, int iy) const {
  return ix >= win_x0_ && ix <= win_x1_ && iy >= win_y0_ && iy <= win_y1_;
}

void LocalPlanner::set_window(double x, double y) {
  const Planner &field = planner_;
  if (!field.map_ready_) {
    return;
  }
  const int reach = static_cast<int>(range_m_ / field.resolution_);
  const int ix = static_cast<int>(std::floor((x - field.origin_x_) / field.resolution_));
  const int iy = static_cast<int>(std::floor((y - field.origin_y_) / field.resolution_));
  win_x0_ = std::max(0, ix - reach);
  win_y0_ = std::max(0, iy - reach);
  win_x1_ = std::min(field.nx_ - 1, ix + reach);
  win_y1_ = std::min(field.ny_ - 1, iy + reach);
}

bool LocalPlanner::write_local(int ix, int iy, std::uint64_t neu) {
  Planner &field = planner_;
  const std::size_t xy = static_cast<std::size_t>(ix) + static_cast<std::size_t>(iy) * field.nx_;
  if (field.local_[xy] == neu) {
    return false;
  }
  field.local_[xy] = neu;
  const std::uint64_t penalty = field.free_[xy] ? field.static_pen_[xy] + neu : 0;
  bool changed = false;
  const std::int64_t column = field.pad_col(ix, iy);
  for (int it = 0; it < field.nt_; ++it) {
    const std::size_t pad = static_cast<std::size_t>(column + it);
    const std::uint64_t previous = field.pen_[pad];
    if (previous == penalty) {
      continue;
    }
    field.pen_[pad] = penalty;
    const std::uint64_t value = field.cp_[pad].load(std::memory_order_relaxed);
    if (value != kUnreached) {
      field.cp_[pad].store(value - previous + penalty, std::memory_order_relaxed);
    }
    changed = true;
  }
  if (changed) {
    field.dirty_[xy] = 1;
  }
  return changed;
}

bool LocalPlanner::apply_scan(const std::vector<float> &ranges, float angle_min, float angle_increment,
                              float range_min, float range_max, double x, double y, double yaw) {
  set_window(x, y);
  const Planner &field = planner_;
  if (!field.map_ready_ || win_x1_ < win_x0_) {
    return false;
  }
  bool any = false;
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    const float range = ranges[i];
    if (!std::isfinite(range) || range <= 0.0f || range < range_min) {
      continue;
    }
    if (range_max > 0.0f && range > range_max) {
      continue;
    }
    const double angle = yaw + static_cast<double>(angle_increment) * static_cast<double>(i) +
                         static_cast<double>(angle_min);
    const double hit_x = x + static_cast<double>(range) * std::cos(angle);
    const double hit_y = y + static_cast<double>(range) * std::sin(angle);
    const int hit_ix = static_cast<int>(std::floor((hit_x - field.origin_x_) / field.resolution_));
    const int hit_iy = static_cast<int>(std::floor((hit_y - field.origin_y_) / field.resolution_));

    for (double fraction = 0.1; fraction <= 0.9; fraction += 0.1) {
      const double sx = x + static_cast<double>(range) * std::cos(angle) * fraction;
      const double sy = y + static_cast<double>(range) * std::sin(angle) * fraction;
      const int ix = static_cast<int>(std::floor((sx - field.origin_x_) / field.resolution_));
      const int iy = static_cast<int>(std::floor((sy - field.origin_y_) / field.resolution_));
      if (!in_window(ix, iy)) {
        continue;
      }
      const std::size_t xy = static_cast<std::size_t>(ix) + static_cast<std::size_t>(iy) * field.nx_;
      any = write_local(ix, iy, field.local_[xy] / 2) || any;
    }

    for (int dx = -2; dx <= 2; ++dx) {
      for (int dy = -2; dy <= 2; ++dy) {
        const int ix = hit_ix + dx;
        const int iy = hit_iy + dy;
        if (!in_window(ix, iy)) {
          continue;
        }
        any = write_local(ix, iy, kHitPenalty) || any;
      }
    }
  }
  return any;
}

SolveResult LocalPlanner::update(const std::atomic<std::uint32_t> &epoch, std::uint32_t ticket) {
  Planner &field = planner_;
  if (!field.field_ready_) {
    return {};
  }
  std::vector<std::pair<int, int>> seeds;
  const std::size_t cells = static_cast<std::size_t>(field.nx_) * field.ny_;
  for (std::size_t xy = 0; xy < cells; ++xy) {
    if (!field.dirty_[xy]) {
      continue;
    }
    field.dirty_[xy] = 0;
    seeds.emplace_back(static_cast<int>(xy % static_cast<std::size_t>(field.nx_)),
                       static_cast<int>(xy / static_cast<std::size_t>(field.nx_)));
  }
  if (seeds.empty()) {
    return SolveResult{true, false, 0, 0};
  }
  field.zero_masks();
  const std::uint64_t full = heading_mask(field.nt_);
  for (const auto &cell : seeds) {
    field.mask_[field.mask_index(cell.first, cell.second)].store(full, std::memory_order_relaxed);
  }
  field.prev_cells_ = seeds;
  field.candidates_ = field.dilate(seeds);
  return field.run(false, epoch, ticket);
}

LocalPlanner::Command LocalPlanner::command(double x, double y, double yaw) const {
  Command command;
  const auto field = planner_.view();
  if (!field || field->resolution <= 0.0) {
    return command;
  }
  const int ix = static_cast<int>(std::floor((x - field->origin_x) / field->resolution));
  const int iy = static_cast<int>(std::floor((y - field->origin_y) / field->resolution));
  const int it = heading_index(yaw, field->t_resolution_deg);
  if (ix < 0 || iy < 0 || ix >= field->nx || iy >= field->ny || it < 0 || it >= field->nt) {
    return command;
  }
  const std::size_t cell = field->index(ix, iy, it);
  if (field->final_state[cell]) {
    command.arrived = true;
    return command;
  }
  const int action = field->action[cell];
  if (action < 0 || static_cast<std::size_t>(action) >= planner_.actions().size()) {
    return command;
  }
  const Action &chosen = planner_.actions()[static_cast<std::size_t>(action)];
  command.have_action = true;
  command.linear_x = chosen.forward_m;
  command.angular_z = chosen.rotate_deg * kPi / 180.0;
  return command;
}

}  // namespace value_iteration3
