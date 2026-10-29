// SPDX-FileCopyrightText: 2026 nop and CIT autonomous robot laboratory
// SPDX-License-Identifier: BSD-3-Clause

#include "value_iteration_explorationer/exploration.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <tuple>

namespace value_iteration_explorationer {
namespace {
constexpr double kPi = 3.14159265358979323846;

double angle_distance(double a, double b) {
  return std::fabs(std::remainder(a - b, 2 * kPi));
}

int occupancy_class(int value, int threshold) {
  return value < 0 ? -1 : value <= threshold ? 0 : 1;
}

bool visible(const ExplorationGrid &grid, int x0, int y0, int x1, int y1,
             int threshold) {
  // Exclude the target cell: a wall itself can be observed, but not through it.
  const double dx = x1 - x0;
  const double dy = y1 - y0;
  const double length = std::max(std::fabs(dx), std::fabs(dy));
  if (length == 0) {
    return true;
  }
  const double fraction = std::max(0.0, 1.0 - 0.501 / length);
  return clear_segment(grid.occupancy, grid.width, grid.height, x0 + 0.5, y0 + 0.5,
                       x0 + 0.5 + fraction * dx, y0 + 0.5 + fraction * dy, threshold);
}
}  // namespace

ObservedRegions::ObservedRegions(double region_size, std::size_t capacity)
    : region_size_(region_size), capacity_(capacity) {}

void ObservedRegions::clear() {
  regions_.clear();
  order_.clear();
}

void ObservedRegions::reconcile(const ExplorationGrid &grid) {
  for (auto &region : regions_) {
    const double wx = (region.first.first + 0.5) * region_size_;
    const double wy = (region.first.second + 0.5) * region_size_;
    const int x = static_cast<int>(std::floor((wx - grid.origin_x) / grid.resolution));
    const int y = static_cast<int>(std::floor((wy - grid.origin_y) / grid.resolution));
    if (x < 0 || y < 0 || x >= grid.width || y >= grid.height) {
      continue;  // A temporarily cropped region retains its world identity.
    }
    const auto cell = static_cast<std::size_t>(y) * grid.width + x;
    if (region.second.occupancy != grid.occupancy[cell] ||
        region.second.information != grid.information[cell]) {
      // Do not let a later return to an old value resurrect older evidence.
      region.second = {-2, -1};
    }
  }
}

ObservedRegions::Key ObservedRegions::key(const ExplorationGrid &grid, std::size_t cell) const {
  return {static_cast<std::int64_t>(std::floor(
              (grid.origin_x + (cell % grid.width + 0.5) * grid.resolution) / region_size_ + 1e-8)),
          static_cast<std::int64_t>(std::floor(
              (grid.origin_y + (cell / grid.width + 0.5) * grid.resolution) / region_size_ + 1e-8))};
}

bool ObservedRegions::observe(const ExplorationGrid &grid,
                             const std::vector<std::size_t> &cells) {
  bool changed = false;
  for (const auto cell : cells) {
    if (cell >= grid.information.size() || grid.information[cell] == 0 || capacity_ == 0) {
      continue;
    }
    const auto region = key(grid, cell);
    const Evidence evidence{grid.occupancy[cell], grid.information[cell]};
    auto existing = regions_.find(region);
    if (existing != regions_.end()) {
      if (existing->second.occupancy == evidence.occupancy &&
          existing->second.information == evidence.information) {
        continue;
      }
      existing->second = evidence;
    } else {
      if (regions_.size() == capacity_) {
        regions_.erase(order_.front());
        order_.pop_front();
      }
      regions_.emplace(region, evidence);
      order_.push_back(region);
    }
    changed = true;
  }
  return changed;
}

double ObservedRegions::remaining(const ExplorationGrid &grid, std::size_t cell) const {
  const auto found = regions_.find(key(grid, cell));
  const int information = grid.information[cell];
  if (found == regions_.end() || found->second.occupancy != grid.occupancy[cell] ||
      found->second.information != information) {
    return information / 100.0;
  }
  return 0;
}

std::vector<std::size_t> scan_observation_cells(
    const ExplorationGrid &grid, const Observation &sensor, const std::vector<float> &ranges,
    double angle_min, double angle_increment, double range_min, double range_max,
    const ExplorationOptions &options) {
  std::vector<std::size_t> cells;
  if (!exploration_grid_valid(grid) || !std::isfinite(sensor.x) ||
      !std::isfinite(sensor.y) || !std::isfinite(sensor.yaw) ||
      !std::isfinite(angle_min) || !std::isfinite(angle_increment) ||
      !std::isfinite(range_min) || !std::isfinite(range_max) ||
      range_min < 0 || range_max < range_min || options.sensor_range <= 0) {
    return cells;
  }
  const double x0 = (sensor.x - grid.origin_x) / grid.resolution;
  const double y0 = (sensor.y - grid.origin_y) / grid.resolution;
  const int start_x = static_cast<int>(std::floor(x0));
  const int start_y = static_cast<int>(std::floor(y0));
  if (start_x < 0 || start_y < 0 || start_x >= grid.width || start_y >= grid.height) {
    return cells;
  }
  std::vector<std::uint8_t> seen(grid.information.size(), 0);
  auto visit = [&](int x, int y) {
    if (x < 0 || y < 0 || x >= grid.width || y >= grid.height) {
      return false;
    }
    const auto cell = static_cast<std::size_t>(y) * grid.width + x;
    if (!seen[cell]) {
      seen[cell] = 1;
      cells.push_back(cell);
    }
    return occupancy_class(grid.occupancy[cell], options.free_threshold) == 0;
  };
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    const double range = ranges[i];
    const double relative = angle_min + i * angle_increment;
    // Actual LDS coverage is independent of the hypothetical planning FOV.
    if (!std::isfinite(range) || range <= 0 || range < range_min || range > range_max) {
      continue;
    }
    const double distance = std::min(range, options.sensor_range) / grid.resolution;
    const double dx = distance * std::cos(sensor.yaw + relative);
    const double dy = distance * std::sin(sensor.yaw + relative);
    const int end_x = static_cast<int>(std::floor(x0 + dx));
    const int end_y = static_cast<int>(std::floor(y0 + dy));
    int x = start_x, y = start_y;
    const int sx = dx > 0 ? 1 : -1, sy = dy > 0 ? 1 : -1;
    const double inf = std::numeric_limits<double>::infinity();
    const double tx = dx == 0 ? inf : std::fabs(1.0 / dx);
    const double ty = dy == 0 ? inf : std::fabs(1.0 / dy);
    double next_x = dx == 0 ? inf : (sx > 0 ? x + 1 - x0 : x0 - x) * tx;
    double next_y = dy == 0 ? inf : (sy > 0 ? y + 1 - y0 : y0 - y) * ty;
    while (visit(x, y) && (x != end_x || y != end_y)) {
      if (next_x < next_y) {
        x += sx;
        next_x += tx;
      } else if (next_y < next_x) {
        y += sy;
        next_y += ty;
      } else {
        // At a corner both side cells must be clear before continuing.
        const bool side_x = visit(x + sx, y);
        const bool side_y = visit(x, y + sy);
        if (!side_x || !side_y) {
          break;
        }
        x += sx;
        y += sy;
        next_x += tx;
        next_y += ty;
      }
    }
  }
  std::sort(cells.begin(), cells.end());
  cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
  return cells;
}

bool exploration_grid_valid(const ExplorationGrid &grid) {
  if (grid.width <= 0 || grid.height <= 0 ||
      !std::isfinite(grid.resolution) || grid.resolution < 0.0001 ||
      !std::isfinite(grid.origin_x) || !std::isfinite(grid.origin_y) ||
      grid.width > std::numeric_limits<int>::max() / grid.height) {
    return false;
  }
  const auto size = static_cast<std::size_t>(grid.width) * grid.height;
  if (grid.costmap.size() != size || grid.information.size() != size ||
      grid.occupancy.size() != size) {
    return false;
  }
  for (std::size_t i = 0; i < size; ++i) {
    if (grid.costmap[i] < -1 || grid.costmap[i] > 100 ||
        grid.information[i] < 0 || grid.information[i] > 100 ||
        grid.occupancy[i] < -1 || grid.occupancy[i] > 100) {
      return false;
    }
  }
  return true;
}

bool clear_segment(const std::vector<std::int8_t> &cells, int width, int height,
                   double x0, double y0, double x1, double y1, int threshold) {
  auto free_cell = [&](int x, int y) {
    return x >= 0 && y >= 0 && x < width && y < height &&
           cells[static_cast<std::size_t>(y) * width + x] >= 0 &&
           cells[static_cast<std::size_t>(y) * width + x] <= threshold;
  };
  int x = static_cast<int>(std::floor(x0));
  int y = static_cast<int>(std::floor(y0));
  const int end_x = static_cast<int>(std::floor(x1));
  const int end_y = static_cast<int>(std::floor(y1));
  const double dx = x1 - x0;
  const double dy = y1 - y0;
  const int sx = dx > 0 ? 1 : -1;
  const int sy = dy > 0 ? 1 : -1;
  const double inf = std::numeric_limits<double>::infinity();
  const double tx = dx == 0 ? inf : std::fabs(1.0 / dx);
  const double ty = dy == 0 ? inf : std::fabs(1.0 / dy);
  double next_x = dx == 0 ? inf : (sx > 0 ? x + 1 - x0 : x0 - x) * tx;
  double next_y = dy == 0 ? inf : (sy > 0 ? y + 1 - y0 : y0 - y) * ty;
  for (;;) {
    if (!free_cell(x, y)) {
      return false;
    }
    if (x == end_x && y == end_y) {
      return true;
    }
    if (next_x < next_y) {
      x += sx;
      next_x += tx;
    } else if (next_y < next_x) {
      y += sy;
      next_y += ty;
    } else {
      if (!free_cell(x + sx, y) || !free_cell(x, y + sy)) {
        return false;
      }
      x += sx;
      y += sy;
      next_x += tx;
      next_y += ty;
    }
  }
}

double observation_gain(const ExplorationGrid &grid, const Observation &pose,
                        const ExplorationOptions &options) {
  const int px = static_cast<int>(std::floor((pose.x - grid.origin_x) / grid.resolution));
  const int py = static_cast<int>(std::floor((pose.y - grid.origin_y) / grid.resolution));
  if (px < 0 || py < 0 || px >= grid.width || py >= grid.height) {
    return 0;
  }
  const int radius = static_cast<int>(std::ceil(options.sensor_range / grid.resolution));
  double gain = 0;
  for (int y = std::max(0, py - radius); y <= std::min(grid.height - 1, py + radius); ++y) {
    for (int x = std::max(0, px - radius); x <= std::min(grid.width - 1, px + radius); ++x) {
      const auto index = static_cast<std::size_t>(y) * grid.width + x;
      if (grid.information[index] == 0) {
        continue;
      }
      const double dx = (x + 0.5) * grid.resolution + grid.origin_x - pose.x;
      const double dy = (y + 0.5) * grid.resolution + grid.origin_y - pose.y;
      const double distance = std::hypot(dx, dy);
      if (distance > options.sensor_range ||
          angle_distance(std::atan2(dy, dx), pose.yaw) >
              options.sensor_fov_deg * kPi / 360.0 ||
          !visible(grid, px, py, x, y, options.free_threshold)) {
        continue;
      }
      gain += grid.information[index] / 100.0 *
              (1.0 - 0.5 * distance / options.sensor_range);
    }
  }
  return gain * grid.resolution * grid.resolution;
}

ObservationCandidates observation_candidates(
    const ExplorationGrid &grid, double robot_x, double robot_y, int headings,
    const ExplorationOptions &options, const std::vector<Observation> &failed,
    const Observation *preferred, const ExplorationGoal *goal) {
  ObservationCandidates candidates;
  if (!exploration_grid_valid(grid) || headings <= 0 || headings > 64 ||
      360 % headings != 0 || options.sensor_range <= 0 || options.gain_reference <= 0) {
    return candidates;
  }
  const int px = static_cast<int>(std::floor((robot_x - grid.origin_x) / grid.resolution));
  const int py = static_cast<int>(std::floor((robot_y - grid.origin_y) / grid.resolution));
  if (px < 0 || py < 0 || px >= grid.width || py >= grid.height) {
    return candidates;
  }
  const auto start = static_cast<std::size_t>(py) * grid.width + px;
  if (grid.costmap[start] < 0) {
    return candidates;
  }
  std::vector<std::uint8_t> reachable(grid.costmap.size(), 0);
  std::queue<std::size_t> pending;
  reachable[start] = 1;
  pending.push(start);
  constexpr int dx[] = {-1, 1, 0, 0};
  constexpr int dy[] = {0, 0, -1, 1};
  while (!pending.empty()) {
    const auto index = pending.front();
    pending.pop();
    const int x = static_cast<int>(index % grid.width);
    const int y = static_cast<int>(index / grid.width);
    for (int d = 0; d < 4; ++d) {
      const int nx = x + dx[d];
      const int ny = y + dy[d];
      if (nx < 0 || ny < 0 || nx >= grid.width || ny >= grid.height) {
        continue;
      }
      const auto next = static_cast<std::size_t>(ny) * grid.width + nx;
      if (!reachable[next] && grid.costmap[next] >= 0) {
        reachable[next] = 1;
        pending.push(next);
      }
    }
  }
  const int spacing = std::max(1, static_cast<int>(std::ceil(
      options.candidate_spacing / grid.resolution)));
  const int margin = static_cast<int>(std::ceil(
      options.observation_radius / grid.resolution));
  std::vector<std::uint8_t> eligible(grid.costmap.size(), 0);
  // Expand the spatial support first, then evaluate the actual terminal cells.
  // A margin never borrows another pose's view through an intervening wall.
  for (int y = 0; y < grid.height; ++y) {
    for (int x = 0; x < grid.width; ++x) {
      if (!reachable[static_cast<std::size_t>(y) * grid.width + x] ||
          ((x % spacing != 0 || y % spacing != 0) && !(x == px && y == py))) {
        continue;
      }
      for (int gy = std::max(0, y - margin); gy <= std::min(grid.height - 1, y + margin); ++gy) {
        for (int gx = std::max(0, x - margin); gx <= std::min(grid.width - 1, x + margin); ++gx) {
          const auto cell = static_cast<std::size_t>(gy) * grid.width + gx;
          if (reachable[cell] && std::hypot(gx - x, gy - y) * grid.resolution <=
                                    options.observation_radius) {
            eligible[cell] = 1;
          }
        }
      }
    }
  }
  const int angle_step = 360 / headings;
  const double revisit_angle = options.revisit_angle_deg * kPi / 180.0;
  std::size_t support_count = 0;
  std::vector<std::size_t> locations;
  for (std::size_t cell = 0; cell < eligible.size(); ++cell) {
    if (eligible[cell]) {
      locations.push_back(cell);
    }
  }
  std::sort(locations.begin(), locations.end(), [&](std::size_t a, std::size_t b) {
    const double ax = static_cast<double>(a % grid.width) - px;
    const double ay = static_cast<double>(a / grid.width) - py;
    const double bx = static_cast<double>(b % grid.width) - px;
    const double by = static_cast<double>(b / grid.width) - py;
    return std::make_pair(ax * ax + ay * ay, a) < std::make_pair(bx * bx + by * by, b);
  });
  for (const auto location : locations) {
      const int x = static_cast<int>(location % grid.width);
      const int y = static_cast<int>(location / grid.width);
      // Visibility is independent of heading. Cast each target ray only once.
      std::vector<double> gains(static_cast<std::size_t>(headings), 0);
      ObservationView view{x, y, {}, std::vector<double>(
          static_cast<std::size_t>(headings), std::numeric_limits<double>::infinity())};
      const int radius = static_cast<int>(std::ceil(options.sensor_range / grid.resolution));
      for (int cy = std::max(0, y - radius);
           cy <= std::min(grid.height - 1, y + radius); ++cy) {
        for (int cx = std::max(0, x - radius);
             cx <= std::min(grid.width - 1, x + radius); ++cx) {
          const auto index = static_cast<std::size_t>(cy) * grid.width + cx;
          const double distance = std::hypot(cx - x, cy - y) * grid.resolution;
          if (grid.information[index] == 0 || distance > options.sensor_range ||
              !visible(grid, x, y, cx, cy, options.free_threshold)) {
            continue;
          }
          const double angle = std::atan2(cy - y, cx - x);
          const double weight = (1.0 - 0.5 * distance / options.sensor_range) *
                                grid.resolution * grid.resolution;
          const double gain = grid.information[index] / 100.0 * weight;
          std::uint64_t mask = 0;
          for (int heading = 0; heading < headings; ++heading) {
            const double yaw = (heading + 0.5) * angle_step * kPi / 180.0;
            if (angle_distance(angle, yaw) <= options.sensor_fov_deg * kPi / 360.0) {
              gains[static_cast<std::size_t>(heading)] += gain;
              mask |= std::uint64_t{1} << heading;
            }
          }
          if (mask) {
            view.visible.push_back({index, weight, mask});
          }
        }
      }
      for (int heading = 0; heading < headings; ++heading) {
        const Observation pose{grid.origin_x + (x + 0.5) * grid.resolution,
                               grid.origin_y + (y + 0.5) * grid.resolution,
                               (heading + 0.5) * angle_step * kPi / 180.0};
        const double gain = gains[static_cast<std::size_t>(heading)];
        if (gain < options.min_gain) {
          continue;
        }
        double cost = 0;
        if (preferred &&
            (std::hypot(pose.x - preferred->x, pose.y - preferred->y) > options.revisit_radius ||
             angle_distance(pose.yaw, preferred->yaw) > options.revisit_angle_deg * kPi / 180.0)) {
          cost += options.switch_penalty;
        }
        double revisit_cost = 0;
        for (const auto &visit : failed) {
          const double distance = std::hypot(pose.x - visit.x, pose.y - visit.y);
          if (distance > options.revisit_radius) {
            continue;
          }
          const double angle = angle_distance(pose.yaw, visit.yaw);
          if (angle <= revisit_angle) {
            // Failed targets are not observations. Keep their retry penalty
            // separate from information-region consumption.
            revisit_cost = std::max(revisit_cost, options.revisit_penalty *
                (1.0 - distance / options.revisit_radius) * (1.0 - angle / revisit_angle));
          }
        }
        const double remaining = goal ? std::hypot(
            pose.x - goal->pose.x, pose.y - goal->pose.y) * goal->cost_per_meter : 0;
        const double terminal_cost = cost + remaining + revisit_cost;
        view.base_cost[static_cast<std::size_t>(heading)] =
            goal ? std::min(1e8, terminal_cost) : terminal_cost;
      }
      if (std::any_of(view.base_cost.begin(), view.base_cost.end(),
                      [](double cost) { return std::isfinite(cost); })) {
        if (view.visible.size() > options.max_visibility_cells - support_count) {
          candidates.truncated = true;
          return candidates;
        }
        support_count += view.visible.size();
        candidates.views.push_back(std::move(view));
      }
  }
  return candidates;
}

std::vector<GlobalPlanner::Terminal> observation_terminals(
    const ExplorationGrid &grid, double robot_x, double robot_y, int headings,
    const ExplorationOptions &options, const std::vector<Observation> &failed,
    const Observation *preferred, const ExplorationGoal *goal) {
  return consume_observation_terminals(
      observation_candidates(grid, robot_x, robot_y, headings, options, failed, preferred, goal),
      grid, options, ObservedRegions(options.information_region_size, options.max_observed_regions));
}

std::vector<GlobalPlanner::Terminal> consume_observation_terminals(
    const ObservationCandidates &candidates, const ExplorationGrid &grid,
    const ExplorationOptions &options, const ObservedRegions &observed) {
  std::vector<GlobalPlanner::Terminal> terminals;
  // A map cell can support many poses/headings. Resolve its region identity at
  // most once per repair, then only sum cached weights (never recast its ray).
  std::vector<double> remaining(grid.information.size(), -1);
  for (const auto &view : candidates.views) {
    std::vector<double> gains(view.base_cost.size(), 0);
    for (const auto &support : view.visible) {
      auto &interest = remaining[support.cell];
      if (interest < 0) {
        interest = observed.remaining(grid, support.cell);
      }
      if (interest == 0) {
        continue;
      }
      const double gain = interest * support.weight;
      for (std::size_t heading = 0; heading < gains.size(); ++heading) {
        if (support.headings & (std::uint64_t{1} << heading)) {
          gains[heading] += gain;
        }
      }
    }
    for (std::size_t heading = 0; heading < gains.size(); ++heading) {
      if (gains[heading] >= options.min_gain && std::isfinite(view.base_cost[heading])) {
        const double cost = view.base_cost[heading] + options.information_cost *
            (1.0 - std::min(1.0, gains[heading] / options.gain_reference));
        terminals.push_back({view.x, view.y, static_cast<int>(heading), std::min(1e8, cost)});
      }
    }
  }
  std::sort(terminals.begin(), terminals.end(), [](const auto &a, const auto &b) {
    return std::tie(a.x, a.y, a.heading) < std::tie(b.x, b.y, b.heading);
  });
  return terminals;
}

bool exploration_goal_reached(const Observation &pose, const ExplorationGoal &goal) {
  return std::hypot(pose.x - goal.pose.x, pose.y - goal.pose.y) <= goal.radius &&
         angle_distance(pose.yaw, goal.pose.yaw) <= goal.angle_deg * kPi / 180.0;
}

bool exploration_progressed(const Observation &from, const Observation &to, double distance) {
  return std::hypot(to.x - from.x, to.y - from.y) >= distance ||
         angle_distance(to.yaw, from.yaw) >= 0.2;
}

bool scan_motion_clear(const std::vector<Observation> &obstacles, const Observation &pose,
                       double forward, double clearance) {
  const double dx = forward * std::cos(pose.yaw);
  const double dy = forward * std::sin(pose.yaw);
  const double length2 = dx * dx + dy * dy;
  for (const auto &hit : obstacles) {
    const double hx = hit.x - pose.x;
    const double hy = hit.y - pose.y;
    if (std::hypot(hx, hy) <= clearance) {
      // A negative dot product makes squared distance strictly increase for
      // every point on the translation, not merely at its endpoint.
      if (hx * dx + hy * dy >= 0) {
        return false;
      }
      continue;
    }
    const double fraction = length2 == 0 ? 0 : std::clamp(
        (hx * dx + hy * dy) / length2, 0.0, 1.0);
    if (std::hypot(hx - fraction * dx, hy - fraction * dy) <= clearance) {
      return false;
    }
  }
  return true;
}

std::vector<GlobalPlanner::Terminal> goal_terminals(
    const ExplorationGrid &grid, int headings, const ExplorationGoal &goal) {
  std::vector<GlobalPlanner::Terminal> terminals;
  if (!exploration_grid_valid(grid) || headings <= 0 || headings > 64 ||
      360 % headings != 0) {
    return terminals;
  }
  const double angle_step = 2 * kPi / headings;
  const double tolerance = goal.angle_deg * kPi / 180.0;
  for (int y = 0; y < grid.height; ++y) {
    for (int x = 0; x < grid.width; ++x) {
      if (grid.costmap[static_cast<std::size_t>(y) * grid.width + x] < 0) {
        continue;
      }
      const double x0 = grid.origin_x + x * grid.resolution;
      const double y0 = grid.origin_y + y * grid.resolution;
      bool inside = true;
      for (int corner = 0; corner < 4; ++corner) {
        inside = inside && std::hypot(
            x0 + (corner % 2) * grid.resolution - goal.pose.x,
            y0 + (corner / 2) * grid.resolution - goal.pose.y) <= goal.radius;
      }
      if (!inside) {
        continue;
      }
      for (int t = 0; t < headings; ++t) {
        const double center = (t + 0.5) * angle_step;
        if (goal.angle_deg == 180 ||
            angle_distance(center, goal.pose.yaw) + angle_step / 2 <= tolerance + 1e-12) {
          terminals.push_back({x, y, t, 0});
        }
      }
    }
  }
  return terminals;
}

}  // namespace value_iteration_explorationer
