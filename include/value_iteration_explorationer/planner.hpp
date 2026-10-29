// SPDX-FileCopyrightText: 2026 nop and CIT autonomous robot laboratory
// SPDX-License-Identifier: BSD-3-Clause

#ifndef VALUE_ITERATION_EXPLORATIONER_PLANNER_HPP_
#define VALUE_ITERATION_EXPLORATIONER_PLANNER_HPP_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace value_iteration_explorationer {

// CostAdd: 1 s base cost plus the raw cell cost. CostReplace: the raw cell cost alone.
enum class MapType { Occupancy, CostAdd, CostReplace };

int planner_self_test();
class GlobalPlanner;
class LocalPlanner;

struct Scale {
  static constexpr unsigned prob_base_bit = 18;
  static constexpr std::uint64_t prob_base = 1ull << prob_base_bit;
  static constexpr std::uint64_t max_cost = 1000000000ull * prob_base;
};

struct Action {
  std::string name;
  double forward_m;
  double rotate_deg;
};

// Immutable field snapshot. Safe to read while the worker continues sweeping.
struct View {
  int nx = 0;
  int ny = 0;
  int nt = 0;
  int t_resolution_deg = 1;
  double resolution = 0;
  double origin_x = 0;
  double origin_y = 0;
  double origin_qx = 0;
  double origin_qy = 0;
  double origin_qz = 0;
  double origin_qw = 1;
  std::vector<std::uint64_t> cost;
  std::vector<std::int8_t> action;
  std::vector<std::uint8_t> final_state;

  std::size_t index(int ix, int iy, int it) const {
    return static_cast<std::size_t>(it) +
           static_cast<std::size_t>(ix) * static_cast<std::size_t>(nt) +
           static_cast<std::size_t>(iy) * static_cast<std::size_t>(nt) *
               static_cast<std::size_t>(nx);
  }
};

struct SolveResult {
  bool converged = false;
  bool cancelled = false;
  std::uint32_t rounds = 0;
  std::uint64_t updates = 0;
};

// Map, actions, and the value field. One sweep updates that field.
// view() is safe from other threads. Sweeps are called from one worker.
class Planner {
 public:
  explicit Planner(int thread_num = 0);
  void set_thread_num(int thread_num);
  void set_logger(std::function<void(const std::string &)> logger);
  // Replaces the motion set. Call it before load_map. At most 127 actions.
  bool set_actions(std::vector<Action> actions);
  // Exploration additionally checks the swept translation, not just its successors.
  void set_swept_collision_check(bool enabled) { swept_collision_check_ = enabled; }

  // Warm mode retains the last view and world-aligned values. Follow with
  // prepare_terminals(..., true) and solve(); mutating calls share one worker.
  // Different resolution/headings or noninteger origin shifts start cold.
  bool load_map(int width, int height, double resolution, double origin_x,
                double origin_y, double qx, double qy, double qz, double qw,
                const std::vector<std::int8_t> &occupancy, int theta_cells,
                double safety_radius, double safety_penalty,
                double goal_margin_radius, int goal_margin_theta_deg,
                MapType map_type = MapType::Occupancy, bool warm_start = false);

  std::shared_ptr<const View> view() const;
  const std::vector<Action> &actions() const { return actions_; }

 private:
  friend int planner_self_test();
  friend class GlobalPlanner;
  friend class LocalPlanner;

  struct Bucket {
    std::int64_t offset;
    std::uint64_t prob;
  };

  void log(const std::string &text) const;
  void build_transitions();
  void allocate_field();
  void fill_cp(std::uint64_t value);
  void zero_masks();
  // Discard affected values and their old policy predecessors before an increase.
  void invalidate_dependents(const std::vector<std::size_t> &seeds);
  std::vector<std::pair<int, int>> dilate(
      const std::vector<std::pair<int, int>> &cells) const;
  std::uint64_t action_cost(int action, int theta, std::int64_t col) const;
  std::string audit() const;
  std::uint64_t compute_phase(int worker, bool all_headings);
  SolveResult run(bool all_headings_first, const std::atomic<std::uint32_t> &epoch,
                  std::uint32_t ticket);
  void publish();
  std::int64_t pad_col(int ix, int iy) const;
  std::size_t index_of(int ix, int iy, int it) const;
  std::size_t mask_index(int ix, int iy) const;
  bool cancelled(const std::atomic<std::uint32_t> &epoch, std::uint32_t ticket) const;

  template <class F>
  void for_rows(F fn) const;

  int thread_num_;
  std::function<void(const std::string &)> logger_;

  std::vector<Action> actions_;
  bool swept_collision_check_ = false;
  // transitions_[action][theta] = successor buckets in the padded field.
  std::vector<std::vector<std::vector<Bucket>>> transitions_;

  int nx_ = 0;
  int ny_ = 0;
  int nt_ = 0;
  int t_resolution_ = 1;
  int mx_ = 1;
  int my_ = 1;
  int mt_ = 0;
  std::int64_t row_stride_ = 0;
  std::size_t n_pad_ = 0;
  int mask_w_ = 0;
  std::size_t mask_n_ = 0;
  double resolution_ = 0;
  double origin_x_ = 0;
  double origin_y_ = 0;
  double origin_qx_ = 0;
  double origin_qy_ = 0;
  double origin_qz_ = 0;
  double origin_qw_ = 1;
  double goal_margin_radius_ = 0.2;
  int goal_margin_theta_ = 10;
  double goal_x_ = 0;
  double goal_y_ = 0;
  int goal_yaw_ = 0;
  bool map_ready_ = false;
  bool goal_set_ = false;
  bool field_ready_ = false;
  bool warm_start_ = false;

  std::vector<std::uint8_t> free_;
  std::vector<std::uint64_t> static_pen_;
  std::vector<std::uint64_t> local_;
  std::vector<std::uint8_t> dirty_;
  std::vector<std::uint8_t> final_;

  std::unique_ptr<std::atomic<std::uint64_t>[]> cp_;
  std::vector<std::uint64_t> pen_;
  std::vector<std::uint8_t> eval_ok_;
  std::unique_ptr<std::atomic<std::uint64_t>[]> mask_;

  std::vector<std::pair<int, int>> candidates_;
  std::vector<std::vector<std::tuple<int, int, std::uint64_t>>> changed_;
  std::vector<std::pair<int, int>> prev_cells_;
  std::atomic<std::size_t> cursor_{0};
  std::atomic<std::uint64_t> updates_{0};

  mutable std::mutex view_mu_;
  std::shared_ptr<const View> view_;
};

template <class F>
void Planner::for_rows(F fn) const {
  int parties = thread_num_;
  if (parties > ny_) {
    parties = ny_;
  }
  if (parties <= 1) {
    fn(0, ny_);
    return;
  }
  std::vector<std::thread> threads;
  threads.reserve(static_cast<std::size_t>(parties));
  const int base = ny_ / parties;
  const int extra = ny_ % parties;
  int y = 0;
  for (int i = 0; i < parties; ++i) {
    const int height = base + (i < extra ? 1 : 0);
    const int y0 = y;
    const int y1 = y + height;
    y = y1;
    threads.emplace_back([fn, y0, y1] { fn(y0, y1); });
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

}  // namespace value_iteration_explorationer

#endif
