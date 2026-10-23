// SPDX-FileCopyrightText: 2026 nop and CIT autonomous robot laboratory
// SPDX-License-Identifier: BSD-3-Clause

#include "value_iteration3/exploration.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <tuple>

namespace value_iteration3 {
namespace {
constexpr double kPi = 3.14159265358979323846;

double angle_distance(double a, double b) {
  return std::fabs(std::remainder(a - b, 2 * kPi));
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

std::vector<GlobalPlanner::Terminal> observation_terminals(
    const ExplorationGrid &grid, double robot_x, double robot_y, int headings,
    const ExplorationOptions &options, const std::vector<Observation> &recent,
    const Observation *preferred, const ExplorationGoal *goal) {
  std::vector<GlobalPlanner::Terminal> terminals;
  if (!exploration_grid_valid(grid) || headings <= 0 || headings > 64 ||
      options.sensor_range <= 0 || options.gain_reference <= 0) {
    return terminals;
  }
  const int px = static_cast<int>(std::floor((robot_x - grid.origin_x) / grid.resolution));
  const int py = static_cast<int>(std::floor((robot_y - grid.origin_y) / grid.resolution));
  if (px < 0 || py < 0 || px >= grid.width || py >= grid.height) {
    return terminals;
  }
  const auto start = static_cast<std::size_t>(py) * grid.width + px;
  if (grid.costmap[start] < 0) {
    return terminals;
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
  const int angle_step = 360 / headings;
  for (int y = 0; y < grid.height; ++y) {
    for (int x = 0; x < grid.width; ++x) {
      if (!reachable[static_cast<std::size_t>(y) * grid.width + x] ||
          ((x % spacing != 0 || y % spacing != 0) && !(x == px && y == py))) {
        continue;
      }
      // Visibility is independent of heading. Cast each target ray only once.
      std::vector<double> gains(static_cast<std::size_t>(headings), 0);
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
          const double gain = grid.information[index] / 100.0 *
                              (1.0 - 0.5 * distance / options.sensor_range) *
                              grid.resolution * grid.resolution;
          for (int heading = 0; heading < headings; ++heading) {
            const double yaw = (heading + 0.5) * angle_step * kPi / 180.0;
            if (angle_distance(angle, yaw) <= options.sensor_fov_deg * kPi / 360.0) {
              gains[static_cast<std::size_t>(heading)] += gain;
            }
          }
        }
      }
      for (int heading = 0; heading < headings; ++heading) {
        const Observation pose{grid.origin_x + (x + 0.5) * grid.resolution,
                               grid.origin_y + (y + 0.5) * grid.resolution,
                               (heading + 0.5) * angle_step * kPi / 180.0};
        bool revisited = false;
        for (const Observation &visit : recent) {
          if (std::hypot(pose.x - visit.x, pose.y - visit.y) <= options.revisit_radius &&
              angle_distance(pose.yaw, visit.yaw) <= options.revisit_angle_deg * kPi / 180.0) {
            revisited = true;
            break;
          }
        }
        if (revisited) {
          continue;
        }
        const double gain = gains[static_cast<std::size_t>(heading)];
        if (gain < options.min_gain) {
          continue;
        }
        double cost = options.information_cost *
                      (1.0 - std::min(1.0, gain / options.gain_reference));
        if (preferred &&
            (std::hypot(pose.x - preferred->x, pose.y - preferred->y) > options.revisit_radius ||
             angle_distance(pose.yaw, preferred->yaw) > options.revisit_angle_deg * kPi / 180.0)) {
          cost += options.switch_penalty;
        }
        // As with the original goal margin, a terminal must have spatial extent:
        // all stochastic successor buckets must be able to enter the region.
        const int margin = static_cast<int>(std::ceil(
            options.observation_radius / grid.resolution));
        for (int gy = std::max(0, y - margin);
             gy <= std::min(grid.height - 1, y + margin); ++gy) {
          for (int gx = std::max(0, x - margin);
               gx <= std::min(grid.width - 1, x + margin); ++gx) {
            if (!reachable[static_cast<std::size_t>(gy) * grid.width + gx] ||
                std::hypot(gx - x, gy - y) * grid.resolution >
                    options.observation_radius) {
              continue;
            }
            const Observation actual{grid.origin_x + (gx + 0.5) * grid.resolution,
                                     grid.origin_y + (gy + 0.5) * grid.resolution,
                                     pose.yaw};
            bool suppressed = false;
            for (const auto &visit : recent) {
              if (std::hypot(actual.x - visit.x, actual.y - visit.y) <= options.revisit_radius &&
                  angle_distance(actual.yaw, visit.yaw) <=
                      options.revisit_angle_deg * kPi / 180.0) {
                suppressed = true;
                break;
              }
            }
            if (!suppressed) {
              const double remaining = goal ? std::hypot(
                  actual.x - goal->pose.x, actual.y - goal->pose.y) *
                      goal->cost_per_meter : 0;
              terminals.push_back({gx, gy, heading,
                                   goal ? std::min(1e8, cost + remaining) : cost});
            }
          }
        }
      }
    }
  }
  std::sort(terminals.begin(), terminals.end(), [](const auto &a, const auto &b) {
    return std::tie(a.x, a.y, a.heading, a.cost) < std::tie(b.x, b.y, b.heading, b.cost);
  });
  terminals.erase(std::unique(terminals.begin(), terminals.end(), [](const auto &a, const auto &b) {
    return a.x == b.x && a.y == b.y && a.heading == b.heading;
  }), terminals.end());
  return terminals;
}

bool exploration_goal_reached(const Observation &pose, const ExplorationGoal &goal) {
  return std::hypot(pose.x - goal.pose.x, pose.y - goal.pose.y) <= goal.radius &&
         angle_distance(pose.yaw, goal.pose.yaw) <= goal.angle_deg * kPi / 180.0;
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

}  // namespace value_iteration3
