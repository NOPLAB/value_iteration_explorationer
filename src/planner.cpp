// SPDX-FileCopyrightText: 2026 nop and CIT autonomous robot laboratory
// SPDX-License-Identifier: BSD-3-Clause

#include "value_iteration_explorationer/planner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <utility>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace value_iteration_explorationer {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr std::uint64_t kUnreached = ~std::uint64_t{0};
constexpr std::uint32_t kMaxRounds = 100000000u;
constexpr int kSampleBit = 6;

int countr_zero(std::uint64_t bits) {
#if defined(_MSC_VER)
  unsigned long index = 0;
  _BitScanForward64(&index, bits);
  return static_cast<int>(index);
#else
  return __builtin_ctzll(bits);
#endif
}

std::uint64_t heading_mask(int nt) {
  return nt >= 64 ? kUnreached : ((std::uint64_t{1} << nt) - 1);
}

// Expand a heading bitmask by k steps on a circle of length nt.
std::uint64_t dilate_headings(std::uint64_t mask, int k, int nt) {
  const std::uint64_t full = heading_mask(nt);
  std::uint64_t acc = mask & full;
  std::uint64_t left = acc;
  std::uint64_t right = acc;
  for (int i = 0; i < k; ++i) {
    left = ((left << 1) | (left >> (nt - 1))) & full;
    right = ((right >> 1) | (right << (nt - 1))) & full;
    acc |= left | right;
  }
  return acc;
}

void cell_delta(double x, double y, double t, double resolution, double t_resolution,
                int &ix, int &iy, int &it) {
  ix = static_cast<int>(std::floor(std::fabs(x) / resolution));
  if (x < 0.0) {
    ix = -ix - 1;
  }
  iy = static_cast<int>(std::floor(std::fabs(y) / resolution));
  if (y < 0.0) {
    iy = -iy - 1;
  }
  it = static_cast<int>(std::floor(t / t_resolution));
}

void no_noise(double forward, double rotate_deg, double from_x, double from_y, double from_t,
              double &to_x, double &to_y, double &to_t) {
  const double ang = from_t / 180.0 * kPi;
  to_x = from_x + forward * std::cos(ang);
  to_y = from_y + forward * std::sin(ang);
  to_t = from_t + rotate_deg;
  while (to_t < 0.0) {
    to_t += 360.0;
  }
}

struct DeltaKey {
  int dix;
  int diy;
  int dit;
  bool operator==(const DeltaKey &other) const {
    return dix == other.dix && diy == other.diy && dit == other.dit;
  }
};

struct DeltaHash {
  std::size_t operator()(const DeltaKey &key) const {
    const auto h = static_cast<std::uint32_t>(key.dix) * 73856093u ^
                   static_cast<std::uint32_t>(key.diy) * 19349663u ^
                   static_cast<std::uint32_t>(key.dit) * 83492791u;
    return h;
  }
};

struct RawDelta {
  int dix;
  int diy;
  int dit;
  std::uint32_t prob;
};

class PhaseBarrier {
 public:
  explicit PhaseBarrier(int parties) : parties_(parties) {}

  void arrive() {
    std::unique_lock<std::mutex> lock(mu_);
    const int generation = generation_;
    if (++waiting_ == parties_) {
      waiting_ = 0;
      ++generation_;
      cv_.notify_all();
    } else {
      cv_.wait(lock, [&] { return generation_ != generation; });
    }
  }

 private:
  int parties_;
  int waiting_ = 0;
  int generation_ = 0;
  std::mutex mu_;
  std::condition_variable cv_;
};

}  // namespace

Planner::Planner(int thread_num) {
  set_thread_num(thread_num);
  actions_.push_back({"forward", 0.3, 0.0});
  actions_.push_back({"back", -0.2, 0.0});
  actions_.push_back({"right", 0.0, -20.0});
  actions_.push_back({"rightfw", 0.2, -20.0});
  actions_.push_back({"left", 0.0, 20.0});
  actions_.push_back({"leftfw", 0.2, 20.0});
}

bool Planner::set_actions(std::vector<Action> actions) {
  if (actions.empty() || actions.size() > 127 || map_ready_) {
    return false;
  }
  actions_ = std::move(actions);
  return true;
}

void Planner::set_thread_num(int thread_num) {
  if (thread_num <= 0) {
    thread_num = static_cast<int>(std::thread::hardware_concurrency());
  }
  if (thread_num < 1) {
    thread_num = 1;
  }
  thread_num_ = thread_num;
}

void Planner::set_logger(std::function<void(const std::string &)> logger) {
  logger_ = std::move(logger);
}

void Planner::log(const std::string &text) const {
  if (logger_) {
    logger_(text);
  }
}

std::int64_t Planner::pad_col(int ix, int iy) const {
  return static_cast<std::int64_t>(ix + mx_) * nt_ +
         static_cast<std::int64_t>(iy + my_) * row_stride_;
}

std::size_t Planner::index_of(int ix, int iy, int it) const {
  return static_cast<std::size_t>(it) + static_cast<std::size_t>(ix) * static_cast<std::size_t>(nt_) +
         static_cast<std::size_t>(iy) * static_cast<std::size_t>(nt_) * static_cast<std::size_t>(nx_);
}

std::size_t Planner::mask_index(int ix, int iy) const {
  return static_cast<std::size_t>(ix + mx_) +
         static_cast<std::size_t>(iy + my_) * static_cast<std::size_t>(mask_w_);
}

bool Planner::cancelled(const std::atomic<std::uint32_t> &epoch, std::uint32_t ticket) const {
  return epoch.load(std::memory_order_relaxed) != ticket;
}

void Planner::fill_cp(std::uint64_t value) {
  const std::size_t n = n_pad_;
  int parties = std::min(thread_num_, static_cast<int>(n > 0 ? n : 1));
  if (parties < 1) {
    parties = 1;
  }
  std::vector<std::thread> threads;
  threads.reserve(static_cast<std::size_t>(parties));
  const std::size_t chunk = (n + static_cast<std::size_t>(parties) - 1) / parties;
  for (int i = 0; i < parties; ++i) {
    const std::size_t begin = std::min(n, chunk * static_cast<std::size_t>(i));
    const std::size_t end = std::min(n, begin + chunk);
    threads.emplace_back([this, begin, end, value] {
      for (std::size_t k = begin; k < end; ++k) {
        cp_[k].store(value, std::memory_order_relaxed);
      }
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

void Planner::zero_masks() {
  const std::size_t n = mask_n_;
  int parties = std::min(thread_num_, static_cast<int>(n > 0 ? n : 1));
  if (parties < 1) {
    parties = 1;
  }
  std::vector<std::thread> threads;
  threads.reserve(static_cast<std::size_t>(parties));
  const std::size_t chunk = (n + static_cast<std::size_t>(parties) - 1) / parties;
  for (int i = 0; i < parties; ++i) {
    const std::size_t begin = std::min(n, chunk * static_cast<std::size_t>(i));
    const std::size_t end = std::min(n, begin + chunk);
    threads.emplace_back([this, begin, end] {
      for (std::size_t k = begin; k < end; ++k) {
        mask_[k].store(0, std::memory_order_relaxed);
      }
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

void Planner::build_transitions() {
  const int sample_n = 1 << kSampleBit;
  const double xy_step = resolution_ / sample_n;
  const double t_step = static_cast<double>(t_resolution_) / sample_n;
  const int action_n = static_cast<int>(actions_.size());

  std::vector<std::vector<std::vector<RawDelta>>> raw(
      actions_.size(), std::vector<std::vector<RawDelta>>(static_cast<std::size_t>(nt_)));
  std::vector<int> mx(static_cast<std::size_t>(nt_), 0);
  std::vector<int> my(static_cast<std::size_t>(nt_), 0);
  std::vector<int> mt(static_cast<std::size_t>(nt_), 0);

  int parties = std::min(thread_num_, nt_);
  std::vector<std::thread> threads;
  threads.reserve(static_cast<std::size_t>(parties));
  for (int worker = 0; worker < parties; ++worker) {
    threads.emplace_back([&, worker] {
      for (int it = worker; it < nt_; it += parties) {
        int local_mx = 0;
        int local_my = 0;
        int local_mt = 0;
        const double theta_origin = it * static_cast<double>(t_resolution_);
        for (int ai = 0; ai < action_n; ++ai) {
          std::unordered_map<DeltaKey, std::uint32_t, DeltaHash> hist;
          hist.reserve(64);
          for (double oy = 0.5 * xy_step; oy < resolution_; oy += xy_step) {
            for (double ox = 0.5 * xy_step; ox < resolution_; ox += xy_step) {
              for (double ot = 0.5 * t_step; ot < t_resolution_; ot += t_step) {
                double dx = 0;
                double dy = 0;
                double dt = 0;
                no_noise(actions_[ai].forward_m, actions_[ai].rotate_deg, ox, oy,
                         ot + theta_origin, dx, dy, dt);
                int dix = 0;
                int diy = 0;
                int dit = 0;
                cell_delta(dx, dy, dt, resolution_, t_resolution_, dix, diy, dit);
                hist[DeltaKey{dix, diy, dit}] += 1;
                local_mx = std::max(local_mx, std::abs(dix));
                local_my = std::max(local_my, std::abs(diy));
                int raw = dit - it;
                raw %= nt_;
                if (raw < 0) {
                  raw += nt_;
                }
                const int circ = std::min(raw, nt_ - raw);
                local_mt = std::max(local_mt, circ);
              }
            }
          }
          auto &dst = raw[ai][it];
          dst.reserve(hist.size());
          for (const auto &item : hist) {
            dst.push_back(RawDelta{item.first.dix, item.first.diy, item.first.dit, item.second});
          }
        }
        mx[it] = local_mx;
        my[it] = local_my;
        mt[it] = local_mt;
      }
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }

  int reach_x = 1;
  int reach_y = 1;
  int reach_t = 0;
  for (int it = 0; it < nt_; ++it) {
    reach_x = std::max(reach_x, mx[it]);
    reach_y = std::max(reach_y, my[it]);
    reach_t = std::max(reach_t, mt[it]);
  }
  mx_ = reach_x;
  my_ = reach_y;
  mt_ = reach_t;
  row_stride_ = static_cast<std::int64_t>(nt_) * (nx_ + 2 * mx_);

  transitions_.assign(actions_.size(),
                      std::vector<std::vector<Bucket>>(static_cast<std::size_t>(nt_)));
  for (int ai = 0; ai < action_n; ++ai) {
    for (int it = 0; it < nt_; ++it) {
      auto &dst = transitions_[ai][it];
      dst.reserve(raw[ai][it].size());
      for (const RawDelta &delta : raw[ai][it]) {
        int nit = delta.dit % nt_;
        if (nit < 0) {
          nit += nt_;
        }
        Bucket bucket;
        bucket.offset = static_cast<std::int64_t>(delta.dix) * nt_ +
                        static_cast<std::int64_t>(delta.diy) * row_stride_ + nit;
        bucket.prob = delta.prob;
        dst.push_back(bucket);
      }
    }
  }
}

void Planner::allocate_field() {
  const std::size_t cells = static_cast<std::size_t>(nx_) * ny_;
  const std::size_t states = cells * static_cast<std::size_t>(nt_);
  local_.assign(cells, 0);
  dirty_.assign(cells, 0);
  final_.assign(states, 0);
  n_pad_ = static_cast<std::size_t>(row_stride_ * (ny_ + 2 * my_));
  cp_.reset(new std::atomic<std::uint64_t>[n_pad_]);
  fill_cp(kUnreached);
  pen_.assign(n_pad_, 0);
  eval_ok_.assign(n_pad_, 0);
  mask_w_ = nx_ + 2 * mx_;
  mask_n_ = static_cast<std::size_t>(mask_w_) * static_cast<std::size_t>(ny_ + 2 * my_);
  mask_.reset(new std::atomic<std::uint64_t>[mask_n_]);
  zero_masks();
}

void Planner::invalidate_dependents(const std::vector<std::size_t> &seeds) {
  if (seeds.empty()) {
    return;
  }
  // A retained value must still have its old policy available. Invalidate the
  // reverse closure BEFORE changing costs/terminals, while that policy is known.
  // Unlike repeated upward Bellman backups, this also handles orphaned cycles.
  std::vector<std::vector<std::size_t>> predecessors(n_pad_);
  for (int y = 0; y < ny_; ++y) {
    for (int x = 0; x < nx_; ++x) {
      const auto column = pad_col(x, y);
      for (int t = 0; t < nt_; ++t) {
        const auto pad = static_cast<std::size_t>(column + t);
        if (!eval_ok_[pad] || cp_[pad].load() == kUnreached) {
          continue;
        }
        const auto value = cp_[pad].load(std::memory_order_relaxed);
        const auto bound = value - pen_[pad];
        for (std::size_t a = 0; a < actions_.size(); ++a) {
          const auto cost = action_cost(static_cast<int>(a), t, column);
          // Include ties, not just the published action: a zero-cost cycle
          // can tie the route that originally connected it to a terminal.
          if (cost < Scale::max_cost && cost <= bound) {
            for (const auto &bucket : transitions_[a][t]) {
              predecessors[static_cast<std::size_t>(column + bucket.offset)].push_back(pad);
            }
          }
        }
      }
    }
  }
  std::vector<std::uint8_t> invalid(n_pad_, 0);
  std::vector<std::size_t> queue = seeds;
  for (const auto pad : seeds) {
    invalid[pad] = 1;
  }
  for (std::size_t i = 0; i < queue.size(); ++i) {
    const auto pad = queue[i];
    cp_[pad].store(kUnreached, std::memory_order_relaxed);
    for (const auto predecessor : predecessors[pad]) {
      if (!invalid[predecessor]) {
        invalid[predecessor] = 1;
        queue.push_back(predecessor);
      }
    }
  }
}

bool Planner::load_map(int width, int height, double resolution, double origin_x, double origin_y,
                       double qx, double qy, double qz, double qw,
                       const std::vector<std::int8_t> &occupancy, int theta_cells,
                       double safety_radius, double safety_penalty, double goal_margin_radius,
                       int goal_margin_theta_deg, MapType map_type, bool warm_start) {
  if (width <= 0 || height <= 0 || resolution < 0.0001 || theta_cells <= 0 || theta_cells > 64) {
    return false;
  }
  if (static_cast<int>(occupancy.size()) != width * height) {
    return false;
  }
  if (360 / theta_cells <= 0) {
    return false;
  }

  std::vector<std::uint8_t> new_free(static_cast<std::size_t>(width) * height, 0);
  std::vector<std::uint64_t> new_pen(new_free.size(), 0);
  const int margin = static_cast<int>(std::ceil(safety_radius / resolution));
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const auto xy = static_cast<std::size_t>(y) * width + x;
      if (map_type != MapType::Occupancy) {
        const auto cost = static_cast<std::uint8_t>(occupancy[xy]);
        new_free[xy] = cost != 255;
        const auto base = map_type == MapType::CostAdd ? Scale::prob_base : 0;
        new_pen[xy] = new_free[xy] ? base + cost * Scale::prob_base : 0;
      } else if (occupancy[xy] == 0) {
        new_free[xy] = 1;
        new_pen[xy] = Scale::prob_base;
        for (int ix = x - margin; ix <= x + margin; ++ix) {
          for (int iy = y - margin; iy <= y + margin; ++iy) {
            const long pos = static_cast<long>(iy) * width + ix;
            if (pos >= 0 && pos < static_cast<long>(occupancy.size()) && occupancy[pos] != 0) {
              new_pen[xy] = static_cast<std::uint64_t>(safety_penalty * Scale::prob_base) +
                            Scale::prob_base;
            }
          }
        }
      }
    }
  }
  const double dx = map_ready_ ? (origin_x_ - origin_x) / resolution : 0;
  const double dy = map_ready_ ? (origin_y_ - origin_y) / resolution : 0;
  const bool aligned = warm_start && map_ready_ && resolution == resolution_ &&
      theta_cells == nt_ && qx == origin_qx_ && qy == origin_qy_ &&
      qz == origin_qz_ && qw == origin_qw_ && qx == 0 && qy == 0 && qz == 0 &&
      std::fabs(dx - std::round(dx)) < 1e-6 && std::fabs(dy - std::round(dy)) < 1e-6 &&
      std::fabs(dx) < 1e9 && std::fabs(dy) < 1e9;
  std::vector<std::uint64_t> retained;
  std::vector<std::uint8_t> retained_final;
  if (aligned) {
    const int sx = static_cast<int>(std::round(dx));
    const int sy = static_cast<int>(std::round(dy));
    std::vector<std::size_t> seeds;
    for (int y = 0; y < ny_; ++y) {
      for (int x = 0; x < nx_; ++x) {
        const int xx = x + sx, yy = y + sy;
        bool invalid = xx < 0 || yy < 0 || xx >= width || yy >= height;
        if (!invalid) {
          const auto target = static_cast<std::size_t>(yy) * width + xx;
          invalid = !new_free[target] || new_pen[target] > static_pen_[y * nx_ + x];
        }
        // A new obstacle can invalidate an action's swept path without being
        // one of its stochastic endpoints. Conservatively seed its footprint.
        if (!invalid && swept_collision_check_) {
          for (int oy = std::max(0, y - my_); oy <= std::min(ny_ - 1, y + my_); ++oy) {
            for (int ox = std::max(0, x - mx_); ox <= std::min(nx_ - 1, x + mx_); ++ox) {
              const int tx = ox + sx, ty = oy + sy;
              if (free_[oy * nx_ + ox] &&
                  (tx < 0 || ty < 0 || tx >= width || ty >= height ||
                   !new_free[static_cast<std::size_t>(ty) * width + tx])) {
                invalid = true;
              }
            }
          }
        }
        if (invalid && free_[static_cast<std::size_t>(y) * nx_ + x]) {
          for (int t = 0; t < nt_; ++t) {
            seeds.push_back(static_cast<std::size_t>(pad_col(x, y) + t));
          }
        }
      }
    }
    invalidate_dependents(seeds);
    retained.assign(new_free.size() * nt_, kUnreached);
    retained_final.assign(retained.size(), 0);
    for (int y = 0; y < ny_; ++y) {
      for (int x = 0; x < nx_; ++x) {
        const int xx = x + sx, yy = y + sy;
        if (xx < 0 || yy < 0 || xx >= width || yy >= height) {
          continue;
        }
        const auto target = static_cast<std::size_t>(yy) * width + xx;
        if (!new_free[target]) {
          continue;
        }
        for (int t = 0; t < nt_; ++t) {
          const auto pad = static_cast<std::size_t>(pad_col(x, y) + t);
          const auto value = cp_[pad].load(std::memory_order_relaxed);
          retained[target * nt_ + t] =
              value == kUnreached ? kUnreached : value - pen_[pad] + new_pen[target];
          retained_final[target * nt_ + t] = final_[index_of(x, y, t)];
        }
      }
    }
  }
  const bool reuse_transitions =
      map_ready_ && width == nx_ && height == ny_ && resolution == resolution_ && theta_cells == nt_;
  nx_ = width;
  ny_ = height;
  nt_ = theta_cells;
  t_resolution_ = 360 / nt_;
  resolution_ = resolution;
  origin_x_ = origin_x;
  origin_y_ = origin_y;
  origin_qx_ = qx;
  origin_qy_ = qy;
  origin_qz_ = qz;
  origin_qw_ = qw;
  goal_margin_radius_ = goal_margin_radius;
  goal_margin_theta_ = goal_margin_theta_deg;
  map_ready_ = false;
  goal_set_ = false;
  field_ready_ = false;
  warm_start_ = warm_start;

  free_ = std::move(new_free);
  static_pen_ = std::move(new_pen);

  if (!reuse_transitions) {
    build_transitions();
  }
  allocate_field();
  if (aligned) {
    final_ = std::move(retained_final);
    for (int y = 0; y < ny_; ++y) {
      for (int x = 0; x < nx_; ++x) {
        const auto xy = static_cast<std::size_t>(y) * nx_ + x;
        for (int t = 0; t < nt_; ++t) {
          const auto pad = static_cast<std::size_t>(pad_col(x, y) + t);
          pen_[pad] = static_pen_[xy];
          eval_ok_[pad] = free_[xy] && !final_[index_of(x, y, t)];
          cp_[pad].store(retained[index_of(x, y, t)], std::memory_order_relaxed);
        }
      }
    }
  }
  map_ready_ = true;
  std::ostringstream text;
  text << "map " << nx_ << "x" << ny_ << "x" << nt_ << " threads " << thread_num_;
  log(text.str());
  return true;
}

std::vector<std::pair<int, int>> Planner::dilate(
    const std::vector<std::pair<int, int>> &cells) const {
  std::vector<std::uint8_t> seen(static_cast<std::size_t>(nx_) * ny_, 0);
  std::vector<std::pair<int, int>> out;
  out.reserve(cells.size());
  for (const auto &cell : cells) {
    const int x0 = std::max(0, cell.first - mx_);
    const int x1 = std::min(nx_ - 1, cell.first + mx_);
    const int y0 = std::max(0, cell.second - my_);
    const int y1 = std::min(ny_ - 1, cell.second + my_);
    for (int iy = y0; iy <= y1; ++iy) {
      for (int ix = x0; ix <= x1; ++ix) {
        const std::size_t xy = static_cast<std::size_t>(ix) + static_cast<std::size_t>(iy) * nx_;
        if (seen[xy]) {
          continue;
        }
        seen[xy] = 1;
        out.emplace_back(ix, iy);
      }
    }
  }
  return out;
}

std::uint64_t Planner::action_cost(int action, int theta, std::int64_t column) const {
  if (swept_collision_check_) {
    // Check all translations represented by this heading bin and source cell.
    const int iy = static_cast<int>(column / row_stride_) - my_;
    const int ix = static_cast<int>((column % row_stride_) / nt_) - mx_;
    const double distance = actions_[static_cast<std::size_t>(action)].forward_m / resolution_;
    const int steps = std::max(1, static_cast<int>(std::ceil(std::fabs(distance) * 2)));
    const double a0 = theta * t_resolution_ * kPi / 180.0;
    const double a1 = (theta + 1) * t_resolution_ * kPi / 180.0;
    double min_dx = std::min(distance * std::cos(a0), distance * std::cos(a1));
    double max_dx = std::max(distance * std::cos(a0), distance * std::cos(a1));
    double min_dy = std::min(distance * std::sin(a0), distance * std::sin(a1));
    double max_dy = std::max(distance * std::sin(a0), distance * std::sin(a1));
    for (int cardinal = 0; cardinal <= 4; ++cardinal) {
      const double angle = cardinal * kPi / 2;
      if (angle >= a0 && angle <= a1) {
        min_dx = std::min(min_dx, distance * std::cos(angle));
        max_dx = std::max(max_dx, distance * std::cos(angle));
        min_dy = std::min(min_dy, distance * std::sin(angle));
        max_dy = std::max(max_dy, distance * std::sin(angle));
      }
    }
    for (int step = 0; step <= steps; ++step) {
      const double fraction = static_cast<double>(step) / steps;
      const int x0 = ix + static_cast<int>(std::floor(fraction * min_dx));
      const int x1 = ix + static_cast<int>(std::ceil(fraction * max_dx));
      const int y0 = iy + static_cast<int>(std::floor(fraction * min_dy));
      const int y1 = iy + static_cast<int>(std::ceil(fraction * max_dy));
      for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
          if (x < 0 || y < 0 || x >= nx_ || y >= ny_ ||
              !free_[static_cast<std::size_t>(y) * nx_ + x]) {
            return Scale::max_cost;
          }
        }
      }
    }
  }
  std::uint64_t cost = 0;
  std::uint64_t remainder = 0;
  for (const Bucket &bucket : transitions_[action][theta]) {
    const std::uint64_t value =
        cp_[static_cast<std::size_t>(column + bucket.offset)].load(std::memory_order_relaxed);
    if (value >= Scale::max_cost) {
      return Scale::max_cost;
    }
    cost += (value >> Scale::prob_base_bit) * bucket.prob;
    remainder += (value & (Scale::prob_base - 1)) * bucket.prob;
  }
  return cost + (remainder >> Scale::prob_base_bit);
}

std::uint64_t Planner::compute_phase(int worker, bool all_headings) {
  const std::size_t n = candidates_.size();
  auto &mine = changed_[static_cast<std::size_t>(worker)];
  mine.clear();
  std::uint64_t my_updates = 0;
  const std::uint64_t full = heading_mask(nt_);
  constexpr std::size_t kBlock = 16;
  const int action_n = static_cast<int>(actions_.size());

  while (true) {
    const std::size_t begin = cursor_.fetch_add(kBlock, std::memory_order_relaxed);
    if (begin >= n) {
      break;
    }
    const std::size_t end = std::min(begin + kBlock, n);
    for (std::size_t j = begin; j < end; ++j) {
      const int ix = candidates_[j].first;
      const int iy = candidates_[j].second;
      std::uint64_t eval_mask = full;
      if (!all_headings) {
        std::uint64_t gathered = 0;
        for (int dy = -my_; dy <= my_; ++dy) {
          const std::size_t row = mask_index(ix - mx_, iy + dy);
          for (int k = 0; k <= 2 * mx_; ++k) {
            gathered |= mask_[row + static_cast<std::size_t>(k)].load(std::memory_order_relaxed);
          }
        }
        if (gathered == 0) {
          continue;
        }
        eval_mask = dilate_headings(gathered, mt_, nt_);
      }

      const std::int64_t column = pad_col(ix, iy);
      std::uint64_t changed_mask = 0;
      std::uint64_t bits = eval_mask;
      while (bits != 0) {
        const int it = countr_zero(bits);
        bits &= bits - 1;
        const std::size_t pad = static_cast<std::size_t>(column + it);
        if (!eval_ok_[pad]) {
          continue;
        }
        const std::uint64_t stored = cp_[pad].load(std::memory_order_relaxed);
        const std::uint64_t before =
            stored == kUnreached ? Scale::max_cost : stored - pen_[pad];
        std::uint64_t best = Scale::max_cost;
        for (int action = 0; action < action_n; ++action) {
          const std::uint64_t cost = action_cost(action, it, column);
          if (cost < best) {
            best = cost;
          }
        }
        // Penalties injected while driving can raise a value, so any move propagates.
        if (best != before) {
          cp_[pad].store(best + pen_[pad], std::memory_order_relaxed);
          ++my_updates;
          changed_mask |= std::uint64_t{1} << it;
        }
      }
      if (changed_mask != 0) {
        mine.emplace_back(ix, iy, changed_mask);
      }
    }
  }
  return my_updates;
}

void Planner::publish() {
  auto view = std::make_shared<View>();
  view->nx = nx_;
  view->ny = ny_;
  view->nt = nt_;
  view->t_resolution_deg = t_resolution_;
  view->resolution = resolution_;
  view->origin_x = origin_x_;
  view->origin_y = origin_y_;
  view->origin_qx = origin_qx_;
  view->origin_qy = origin_qy_;
  view->origin_qz = origin_qz_;
  view->origin_qw = origin_qw_;
  const std::size_t states =
      static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_) * static_cast<std::size_t>(nt_);
  view->cost.assign(states, Scale::max_cost);
  view->action.assign(states, -1);
  view->final_state = final_;

  for_rows([this, &view](int y0, int y1) {
    const int action_n = static_cast<int>(actions_.size());
    for (int iy = y0; iy < y1; ++iy) {
      for (int ix = 0; ix < nx_; ++ix) {
        const std::int64_t column = pad_col(ix, iy);
        for (int it = 0; it < nt_; ++it) {
          const std::size_t pad = static_cast<std::size_t>(column + it);
          const std::size_t orig = index_of(ix, iy, it);
          const std::uint64_t stored = cp_[pad].load(std::memory_order_relaxed);
          if (stored == kUnreached) {
            continue;
          }
          view->cost[orig] = stored - pen_[pad];
          if (!eval_ok_[pad] && !(warm_start_ && final_[orig])) {
            continue;
          }
          std::uint64_t best = Scale::max_cost;
          int chosen = -1;
          for (int action = 0; action < action_n; ++action) {
            const std::uint64_t cost = action_cost(action, it, column);
            if (cost < best) {
              best = cost;
              chosen = action;
            }
          }
          if (chosen >= 0 && best < Scale::max_cost) {
            view->action[orig] = static_cast<std::int8_t>(chosen);
          }
        }
      }
    }
  });

  std::lock_guard<std::mutex> lock(view_mu_);
  view_ = std::move(view);
}

SolveResult Planner::run(bool all_headings_first, const std::atomic<std::uint32_t> &epoch,
                         std::uint32_t ticket) {
  changed_.assign(static_cast<std::size_t>(thread_num_), {});
  cursor_.store(0, std::memory_order_relaxed);
  updates_.store(0, std::memory_order_relaxed);

  PhaseBarrier barrier(thread_num_);
  std::atomic<bool> done{false};
  std::vector<std::thread> workers;
  workers.reserve(static_cast<std::size_t>(std::max(0, thread_num_ - 1)));
  for (int worker = 1; worker < thread_num_; ++worker) {
    workers.emplace_back([&, worker] {
      int round = 0;
      for (;;) {
        const std::uint64_t ups = compute_phase(worker, all_headings_first && round == 0);
        if (ups > 0) {
          updates_.fetch_add(ups, std::memory_order_relaxed);
        }
        ++round;
        barrier.arrive();
        barrier.arrive();
        if (done.load(std::memory_order_relaxed)) {
          break;
        }
      }
    });
  }

  SolveResult result;
  const auto started = std::chrono::steady_clock::now();
  auto next_log = started + std::chrono::seconds(2);
  auto next_publish = started + std::chrono::milliseconds(100);
  std::uint32_t iter = 0;
  for (;;) {
    const std::uint64_t ups = compute_phase(0, all_headings_first && iter == 0);
    if (ups > 0) {
      updates_.fetch_add(ups, std::memory_order_relaxed);
    }
    barrier.arrive();

    ++iter;
    for (const auto &cell : prev_cells_) {
      mask_[mask_index(cell.first, cell.second)].store(0, std::memory_order_relaxed);
    }
    prev_cells_.clear();

    std::vector<std::pair<int, int>> touched;
    for (int worker = 0; worker < thread_num_; ++worker) {
      for (const auto &item : changed_[static_cast<std::size_t>(worker)]) {
        const int ix = std::get<0>(item);
        const int iy = std::get<1>(item);
        const std::uint64_t bits = std::get<2>(item);
        mask_[mask_index(ix, iy)].store(bits, std::memory_order_relaxed);
        touched.emplace_back(ix, iy);
      }
    }
    prev_cells_ = touched;
    const bool any = !touched.empty();
    const auto now = std::chrono::steady_clock::now();
    if (now >= next_log) {
      std::ostringstream text;
      text << "round " << iter << " updates " << updates_.load(std::memory_order_relaxed);
      log(text.str());
      next_log = now + std::chrono::seconds(2);
    }

    const bool stop = cancelled(epoch, ticket) || !any || iter >= kMaxRounds;
    if (warm_start_ && !stop && now >= next_publish) {
      // Workers are parked at the barrier: publish a coherent immutable copy.
      publish();
      next_publish = now + std::chrono::milliseconds(100);
    }
    if (!stop) {
      candidates_ = dilate(touched);
      if (iter % 2 == 1) {
        std::reverse(candidates_.begin(), candidates_.end());
      }
      cursor_.store(0, std::memory_order_relaxed);
    } else {
      result.rounds = iter;
      result.updates = updates_.load(std::memory_order_relaxed);
      result.cancelled = cancelled(epoch, ticket);
      result.converged = !result.cancelled && !any;
      done.store(true, std::memory_order_relaxed);
    }
    barrier.arrive();
    if (done.load(std::memory_order_relaxed)) {
      break;
    }
  }

  for (auto &worker : workers) {
    worker.join();
  }

  if (!result.cancelled) {
    publish();
    field_ready_ = true;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  std::ostringstream text;
  text << (result.cancelled ? "cancelled" : result.converged ? "converged" : "stopped")
       << " rounds " << result.rounds << " updates " << result.updates << " in " << seconds << " s";
  log(text.str());
  return result;
}

std::shared_ptr<const View> Planner::view() const {
  std::lock_guard<std::mutex> lock(view_mu_);
  return view_;
}

std::string Planner::audit() const {
  if (!view_ || transitions_.empty() || transitions_[0].empty()) {
    return "no field";
  }
  for (std::size_t action = 0; action < transitions_.size(); ++action) {
    for (int theta = 0; theta < nt_; ++theta) {
      std::uint64_t sum = 0;
      for (const Bucket &bucket : transitions_[action][static_cast<std::size_t>(theta)]) {
        sum += bucket.prob;
      }
      if (sum != Scale::prob_base) {
        return "transition weight sum " + std::to_string(sum);
      }
    }
  }
  const int action_n = static_cast<int>(actions_.size());
  for (int iy = 0; iy < ny_; ++iy) {
    for (int ix = 0; ix < nx_; ++ix) {
      const std::int64_t column = pad_col(ix, iy);
      for (int it = 0; it < nt_; ++it) {
        const std::size_t pad = static_cast<std::size_t>(column + it);
        const std::size_t cell = index_of(ix, iy, it);
        if (!eval_ok_[pad]) {
          if (final_[cell]) {
            if (view_->cost[cell] != 0) {
              return "goal cost";
            }
          } else if (view_->cost[cell] != Scale::max_cost) {
            return "blocked cost";
          }
          continue;
        }
        std::uint64_t best = Scale::max_cost;
        int chosen = -1;
        for (int action = 0; action < action_n; ++action) {
          const std::uint64_t cost = action_cost(action, it, column);
          if (cost < best) {
            best = cost;
            chosen = action;
          }
        }
        if (view_->cost[cell] != best) {
          return "backup";
        }
        const std::int8_t expected =
            chosen >= 0 && best < Scale::max_cost ? static_cast<std::int8_t>(chosen) : -1;
        if (view_->action[cell] != expected) {
          return "action";
        }
      }
    }
  }
  return {};
}

}  // namespace value_iteration_explorationer
