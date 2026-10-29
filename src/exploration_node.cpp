// SPDX-FileCopyrightText: 2026 nop and CIT autonomous robot laboratory
// SPDX-License-Identifier: BSD-3-Clause

#include "value_iteration_explorationer/actions.hpp"
#include "value_iteration_explorationer/exploration.hpp"
#include "value_iteration_explorationer/local_planner.hpp"
#include "value_iteration_explorationer/msg/exploration_map.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "std_msgs/msg/string.hpp"
#include "tf2/utils.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/create_timer_ros.h"
#include "tf2_ros/message_filter.h"
#include "tf2_ros/transform_listener.h"

namespace value_iteration_explorationer {
namespace {
constexpr double kPi = 3.14159265358979323846;
using Clock = std::chrono::steady_clock;
using Layers = value_iteration_explorationer::msg::ExplorationMap;

ExplorationGrid grid_from(const Layers &layers) {
  ExplorationGrid grid;
  grid.width = static_cast<int>(layers.costmap.info.width);
  grid.height = static_cast<int>(layers.costmap.info.height);
  grid.resolution = layers.costmap.info.resolution;
  grid.origin_x = layers.costmap.info.origin.position.x;
  grid.origin_y = layers.costmap.info.origin.position.y;
  grid.costmap = layers.costmap.data;
  grid.information = layers.information_map.data;
  grid.occupancy = layers.occupancy.data;
  return grid;
}

geometry_msgs::msg::Pose pose_msg(const Observation &pose) {
  geometry_msgs::msg::Pose message;
  message.position.x = pose.x;
  message.position.y = pose.y;
  message.orientation.z = std::sin(pose.yaw / 2);
  message.orientation.w = std::cos(pose.yaw / 2);
  return message;
}
}  // namespace

class ExplorationNode : public rclcpp::Node {
 public:
  ExplorationNode() : Node("exploration_planner_node"), planner_(1), global_(planner_),
                      local_(planner_, 1.0), tf_buffer_(get_clock()),
                      tf_listener_(std::make_unique<tf2_ros::TransformListener>(tf_buffer_)) {
    theta_cells_ = declare_parameter<int>("theta_cell_num", 60);
    planner_.set_thread_num(declare_parameter<int>("global_thread_num", 2));
    options_.sensor_range = declare_parameter<double>("sensor_range", 3.0);
    options_.sensor_fov_deg = declare_parameter<double>("sensor_fov_deg", 180.0);
    options_.candidate_spacing = declare_parameter<double>("candidate_spacing", 0.3);
    options_.observation_radius = declare_parameter<double>("observation_radius", 0.2);
    options_.min_gain = declare_parameter<double>("min_gain", 0.02);
    options_.gain_reference = declare_parameter<double>("gain_reference", 1.0);
    options_.information_cost = declare_parameter<double>("information_cost", 30.0);
    options_.revisit_radius = declare_parameter<double>("revisit_radius", 0.75);
    options_.revisit_angle_deg = declare_parameter<double>("revisit_angle_deg", 60.0);
    options_.revisit_penalty = declare_parameter<double>("revisit_penalty", 30.0);
    options_.switch_penalty = declare_parameter<double>("switch_penalty", 3.0);
    options_.free_threshold = declare_parameter<int>("free_threshold", 49);
    options_.information_region_size = declare_parameter<double>("information_region_size", 0.05);
    const int region_limit = declare_parameter<int>("max_observed_regions", 100000);
    const int visibility_limit = declare_parameter<int>("max_visibility_cells", 1000000);
    if (region_limit <= 0 || visibility_limit <= 0) {
      throw std::invalid_argument("information region limits must be positive");
    }
    options_.max_observed_regions = static_cast<std::size_t>(region_limit);
    options_.max_visibility_cells = static_cast<std::size_t>(visibility_limit);
    goal_options_.radius = declare_parameter<double>("goal_margin_radius", 0.3);
    goal_options_.angle_deg = declare_parameter<double>("goal_margin_theta", 10.0);
    goal_options_.cost_per_meter = declare_parameter<double>("goal_cost_per_meter", 1.0 / 0.3);
    replan_interval_ = declare_parameter<double>("replan_interval", 1.0);
    revisit_cooldown_ = declare_parameter<double>("revisit_cooldown", 120.0);
    progress_timeout_ = declare_parameter<double>("progress_timeout", 15.0);
    progress_distance_ = declare_parameter<double>("progress_distance", 0.1);
    input_timeout_ = declare_parameter<double>("input_timeout", 2.0);
    map_timeout_ = declare_parameter<double>("map_timeout", 10.0);
    robot_radius_ = declare_parameter<double>("robot_radius", 0.15) +
                    declare_parameter<double>("safety_margin", 0.05);
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    const auto path = declare_parameter<std::string>(
        "config_file", ament_index_cpp::get_package_share_directory(
            "value_iteration_explorationer") + "/config/params.yaml");
    if (theta_cells_ <= 0 || theta_cells_ > 64 || 360 % theta_cells_ != 0 ||
        !valid_parameters()) {
      throw std::invalid_argument("invalid exploration parameters");
    }
    observed_regions_ = ObservedRegions(options_.information_region_size,
                                        options_.max_observed_regions);
    local_.set_obstacle_radius(robot_radius_);
    const auto actions = read_action_list(path);
    if (!actions.error.empty() || !planner_.set_actions(actions.actions)) {
      throw std::invalid_argument("invalid action list: " + actions.error);
    }
    planner_.set_swept_collision_check(true);
    planner_.set_logger([this](const std::string &text) {
      RCLCPP_INFO(get_logger(), "%s", text.c_str());
    });
    const auto map_qos = rclcpp::QoS(1).reliable().transient_local();
    pub_action_ = create_publisher<geometry_msgs::msg::Twist>(
        declare_parameter<std::string>("action_topic", "/exploration/action"), 1);
    pub_value_ = create_publisher<nav_msgs::msg::OccupancyGrid>("/value_function", map_qos);
    pub_candidates_ = create_publisher<geometry_msgs::msg::PoseArray>(
        "/exploration/candidates", map_qos);
    pub_path_ = create_publisher<nav_msgs::msg::Path>("/exploration/path", map_qos);
    pub_status_ = create_publisher<std_msgs::msg::String>("/exploration/status", map_qos);
    pub_diagnostics_ = create_publisher<std_msgs::msg::String>(
        "/exploration/diagnostics", map_qos);
    tf_buffer_.setCreateTimerInterface(std::make_shared<tf2_ros::CreateTimerROS>(
        get_node_base_interface(), get_node_timers_interface()));
    scan_filter_ = std::make_shared<tf2_ros::MessageFilter<sensor_msgs::msg::LaserScan>>(
        tf_buffer_, base_frame_, 100, get_node_logging_interface(),
        get_node_clock_interface(), tf2::durationFromSec(input_timeout_));
    scan_filter_->registerCallback(
        [this](sensor_msgs::msg::LaserScan::ConstSharedPtr scan) { on_scan(*scan); });
    sub_layers_ = create_subscription<Layers>(
        "/exploration/map_layers", map_qos,
        [this](Layers::ConstSharedPtr layers) { on_layers(std::move(layers)); });
    sub_scan_ = create_subscription<sensor_msgs::msg::LaserScan>(
        "/scan", rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::LaserScan::ConstSharedPtr scan) {
          std::string frame;
          {
            std::lock_guard<std::mutex> lock(mu_);
            if (!latest_) {
              return;
            }
            frame = latest_->header.frame_id;
          }
          scan_filter_->add(scan);
          if (!tf_buffer_.canTransform(frame, scan->header.frame_id,
                                      rclcpp::Time(scan->header.stamp),
                                      rclcpp::Duration::from_seconds(0))) {
            std::lock_guard<std::mutex> lock(mu_);
            waiting_transform_ = true;
            status("waiting_transform");
          }
        });
    sub_goal_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/goal_pose", 10,
        [this](geometry_msgs::msg::PoseStamped::ConstSharedPtr goal) { on_goal(*goal); });
    timer_ = create_wall_timer(std::chrono::milliseconds(100), [this] { on_decision(); });
    status("waiting_input");
    worker_ = std::thread([this] { worker_main(); });
    try {
      geometry_worker_ = std::thread([this] { geometry_main(); });
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(mu_);
        stop_ = true;
      }
      cv_.notify_one();
      worker_.join();
      throw;
    }
  }

  ~ExplorationNode() override {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
      epoch_.fetch_add(1);
    }
    cv_.notify_all();
    geometry_cv_.notify_all();
    // Join the dedicated TF thread before its filter callbacks can outlive us.
    tf_listener_.reset();
    scan_filter_->clear();
    worker_.join();
    geometry_worker_.join();
  }

 private:
  bool valid_parameters() const {
    const double positive[] = {
        options_.sensor_range, options_.sensor_fov_deg, options_.candidate_spacing,
        options_.observation_radius, options_.information_region_size,
        options_.min_gain, options_.gain_reference, options_.revisit_radius,
        options_.revisit_angle_deg, replan_interval_, revisit_cooldown_, progress_timeout_,
        progress_distance_, input_timeout_, map_timeout_, goal_options_.radius,
        goal_options_.angle_deg, goal_options_.cost_per_meter};
    for (double value : positive) {
      if (!std::isfinite(value) || value <= 0) {
        return false;
      }
    }
    return std::isfinite(robot_radius_) && robot_radius_ >= 0 &&
           std::isfinite(options_.information_cost) && options_.information_cost >= 0 &&
           options_.information_cost < 1e8 &&
           std::isfinite(options_.switch_penalty) && options_.switch_penalty >= 0 &&
           options_.switch_penalty < 1e8 &&
           std::isfinite(options_.revisit_penalty) && options_.revisit_penalty >= 0 &&
           options_.revisit_penalty < 1e8 &&
           goal_options_.angle_deg <= 180 && goal_options_.cost_per_meter < 1e8 &&
           options_.sensor_fov_deg <= 360 && options_.revisit_angle_deg <= 180 &&
           options_.free_threshold >= 0 && options_.free_threshold < 100;
  }

  void status(const std::string &text) {
    std_msgs::msg::String message;
    message.data = text;
    pub_status_->publish(message);
  }

  void on_goal(const geometry_msgs::msg::PoseStamped &message) {
    const auto &p = message.pose.position;
    const auto &q = message.pose.orientation;
    const double norm = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
    std::lock_guard<std::mutex> lock(mu_);
    if (message.header.frame_id.empty() || !std::isfinite(p.x) || !std::isfinite(p.y) ||
        !std::isfinite(p.z) || std::fabs(p.z) > 1e-6 ||
        !std::isfinite(norm) || std::fabs(norm - 1.0) > 1e-3 ||
        std::fabs(q.x) > 1e-6 || std::fabs(q.y) > 1e-6 ||
        (latest_ && message.header.frame_id != latest_->header.frame_id)) {
      RCLCPP_WARN(get_logger(), "rejected goal: expected a planar pose in the map frame");
      return;
    }
    goal_ = goal_options_;
    goal_->pose = {p.x, p.y, tf2::getYaw(q)};
    goal_frame_ = message.header.frame_id;
    goal_reached_ = false;
    ready_ = false;
    requested_ = true;
    candidates_.reset();
    terminals_pending_ = false;
    ++generation_;
    preferred_.reset();
    progress_pose_.reset();
    epoch_.fetch_add(1);
    pub_action_->publish(geometry_msgs::msg::Twist());
    geometry_cv_.notify_one();
  }

  void on_layers(Layers::ConstSharedPtr layers) {
    const auto &q = layers->costmap.info.origin.orientation;
    const bool valid = !layers->header.frame_id.empty() &&
        layers->header == layers->costmap.header &&
        layers->header == layers->information_map.header &&
        layers->header == layers->occupancy.header &&
        layers->costmap.info == layers->information_map.info &&
        layers->costmap.info == layers->occupancy.info &&
        options_.information_region_size <= layers->costmap.info.resolution + 1e-8 &&
        std::fabs(q.x) < 1e-9 && std::fabs(q.y) < 1e-9 && std::fabs(q.z) < 1e-9 &&
        std::fabs(std::fabs(q.w) - 1.0) < 1e-9 &&
        std::isfinite(layers->costmap.info.origin.position.z) &&
        std::fabs(layers->costmap.info.origin.position.z) < 1e-9 &&
        exploration_grid_valid(grid_from(*layers));
    std::string previous_frame;
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (latest_) {
        previous_frame = latest_->header.frame_id;
      }
    }
    if (!valid || previous_frame != layers->header.frame_id) {
      // Filter callbacks can run on the TF listener; never clear it while
      // holding the planner mutex that those callbacks acquire.
      scan_filter_->clear();
      if (valid) {
        scan_filter_->setTargetFrame(layers->header.frame_id);
      }
    }
    std::lock_guard<std::mutex> lock(mu_);
    if (!valid) {
      latest_.reset();
      ready_ = false;
      candidates_.reset();
      terminals_pending_ = false;
      ++generation_;
      epoch_.fetch_add(1);
      status("invalid_map");
      pub_action_->publish(geometry_msgs::msg::Twist());
      return;
    }
    if (!latest_ || latest_->header.frame_id != layers->header.frame_id ||
        latest_->costmap.info.resolution != layers->costmap.info.resolution) {
      // Resizing and world-aligned origin changes retain the warm field. A
      // different frame or resolution invalidates outstanding geometry.
      candidates_.reset();
      terminals_pending_ = false;
      ready_ = false;
      have_pose_ = false;
      observed_regions_.clear();
      failed_.clear();
      progress_pose_.reset();
      preferred_.reset();
      latest_scan_.reset();
      waiting_transform_ = false;
      scan_pending_ = false;
      obstacles_.clear();
      scan_received_ = {};
      ++generation_;
      epoch_.fetch_add(1);
      requested_ = true;
    }
    // Cached rays are valid only for the map geometry/occluders they traversed.
    // Keep the accepted immutable policy available, but cancel stale replacements.
    if (latest_ && (latest_->costmap.info != layers->costmap.info ||
                    latest_->costmap.data != layers->costmap.data ||
                    latest_->information_map.data != layers->information_map.data ||
                    latest_->occupancy.data != layers->occupancy.data)) {
      candidates_.reset();
      terminals_pending_ = false;
      ++generation_;
      epoch_.fetch_add(1);
      requested_ = true;
    }
    observed_regions_.reconcile(grid_from(*layers));
    latest_ = std::move(layers);
    map_received_ = Clock::now();
  }

  void on_scan(const sensor_msgs::msg::LaserScan &scan) {
    if (scan.ranges.empty() || !std::isfinite(scan.angle_min) ||
        !std::isfinite(scan.angle_increment)) {
      return;
    }
    std::string frame;
    Layers::ConstSharedPtr layers;
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (!latest_) {
        return;
      }
      frame = latest_->header.frame_id;
      layers = latest_;
    }
    try {
      // MessageFilter invokes this only after the acquisition-time TF exists.
      const auto transform = tf_buffer_.lookupTransform(frame, scan.header.frame_id,
          rclcpp::Time(scan.header.stamp), rclcpp::Duration::from_seconds(0));
      const double yaw = tf2::getYaw(transform.transform.rotation);
      const double x = transform.transform.translation.x;
      const double y = transform.transform.translation.y;
      std::vector<Observation> obstacles;
      for (std::size_t i = 0; i < scan.ranges.size(); ++i) {
        const double range = scan.ranges[i];
        if (!std::isfinite(range) || range < scan.range_min || range > scan.range_max) {
          continue;
        }
        const double angle = yaw + scan.angle_min + i * scan.angle_increment;
        obstacles.push_back({x + range * std::cos(angle), y + range * std::sin(angle), 0});
      }
      const auto grid = grid_from(*layers);
      const auto observed = scan_observation_cells(
          grid, {x, y, yaw}, scan.ranges, scan.angle_min, scan.angle_increment,
          scan.range_min, scan.range_max, options_);
      const auto age = (now() - rclcpp::Time(scan.header.stamp)).seconds();
      std::lock_guard<std::mutex> lock(mu_);
      if (!latest_ || latest_->header.frame_id != frame ||
          age < -input_timeout_ || age > input_timeout_ ||
          (latest_scan_ && rclcpp::Time(scan.header.stamp) <=
                           rclcpp::Time(latest_scan_->message.header.stamp))) {
        return;
      }
      obstacles_ = std::move(obstacles);
      scan_received_ = Clock::now();
      latest_scan_ = Scan{scan, x, y, yaw, frame};
      // Evidence belongs to the map sampled outside the mutex. Never apply an
      // old scan footprint to a newer map that may contain another occluder.
      if (latest_ == layers && rclcpp::Time(scan.header.stamp) >= rclcpp::Time(layers->header.stamp) &&
          observed_regions_.observe(grid, observed)) {
        request_consumption();
      }
      waiting_transform_ = false;
      scan_pending_ = true;
      cv_.notify_one();
    } catch (const tf2::TransformException &) {
      // Without a transform for this scan's acquisition time, do not use it.
    }
  }

  bool safe_action(const Layers &layers, const Observation &pose,
                   const LocalPlanner::Command &action) const {
    const auto &map = layers.costmap;
    const double x0 = (pose.x - map.info.origin.position.x) / map.info.resolution;
    const double y0 = (pose.y - map.info.origin.position.y) / map.info.resolution;
    const double x1 = x0 + action.linear_x * std::cos(pose.yaw) / map.info.resolution;
    const double y1 = y0 + action.linear_x * std::sin(pose.yaw) / map.info.resolution;
    if (!clear_segment(map.data, static_cast<int>(map.info.width),
                       static_cast<int>(map.info.height), x0, y0, x1, y1, 100)) {
      return false;
    }
    return scan_motion_clear(obstacles_, pose, action.linear_x, robot_radius_);
  }

  void on_decision() {
    Layers::ConstSharedPtr layers;
    {
      std::lock_guard<std::mutex> lock(mu_);
      layers = latest_;
    }
    geometry_msgs::msg::Twist output;
    if (!layers) {
      pub_action_->publish(output);
      return;
    }
    Observation pose{};
    bool pose_valid = false;
    try {
      const auto transform = tf_buffer_.lookupTransform(
          layers->header.frame_id, base_frame_, tf2::TimePointZero);
      const auto stamp = rclcpp::Time(transform.header.stamp);
      const double age = (now() - stamp).seconds();
      pose_valid = stamp.nanoseconds() == 0 || (age >= -input_timeout_ && age <= input_timeout_);
      pose = {transform.transform.translation.x, transform.transform.translation.y,
              tf2::getYaw(transform.transform.rotation)};
      pose_valid = pose_valid && std::isfinite(pose.x) && std::isfinite(pose.y) &&
                   std::isfinite(pose.yaw);
    } catch (const tf2::TransformException &) {
      pose_valid = false;
    }
    const auto current = Clock::now();
    std::lock_guard<std::mutex> lock(mu_);
    if (latest_ != layers || !pose_valid ||
        std::chrono::duration<double>(current - map_received_).count() >
                           map_timeout_ ||
        std::chrono::duration<double>(current - scan_received_).count() > input_timeout_) {
      status(waiting_transform_ ? "waiting_transform" :
             !latest_scan_ ? "waiting_scan" : "waiting_input");
      pub_action_->publish(output);
      return;
    }
    pose_ = pose;
    have_pose_ = true;
    if (goal_ && goal_frame_ != layers->header.frame_id) {
      status("waiting_goal_frame");
      pub_action_->publish(output);
      return;
    }
    if (goal_ && (goal_reached_ || exploration_goal_reached(pose, *goal_))) {
      if (!goal_reached_) {
        goal_reached_ = true;
        ready_ = false;
        requested_ = false;
        epoch_.fetch_add(1);
      }
      status("goal_reached");
      pub_action_->publish(output);
      return;
    }
    const auto expire = [&](auto &history) {
      const auto size = history.size();
      history.erase(std::remove_if(history.begin(), history.end(), [&](const Visit &visit) {
        return std::chrono::duration<double>(current - visit.time).count() >= revisit_cooldown_;
      }), history.end());
      return size != history.size();
    };
    expire(failed_);
    if (!geometry_planning_ &&
        std::chrono::duration<double>(current - last_geometry_).count() >= replan_interval_) {
      requested_ = true;
    }
    if (requested_) {
      geometry_cv_.notify_one();
    }
    if (ready_ && solved_ && accepted_view_ &&
        solved_->header.frame_id == layers->header.frame_id &&
        std::chrono::duration<double>(current - scan_received_).count() <= input_timeout_) {
      const auto chosen = local_.command(*accepted_view_, pose.x, pose.y, pose.yaw);
      if (chosen.arrived) {
        decision_reason_ = "observation";
        // Absorbing states have no outgoing Bellman policy. Consume this real
        // observation and wait for the ordinary warm field to be repaired.
        ready_ = false;
        request_consumption();
        preferred_.reset();
        progress_pose_.reset();
        status("observing");
        pub_action_->publish(output);
        return;
      }
      if (chosen.have_action && safe_action(*layers, pose, chosen)) {
        decision_reason_ = "exploring";
        output.linear.x = chosen.linear_x;
        output.angular.z = chosen.angular_z;
        status("exploring");
      } else {
        decision_reason_ = chosen.have_action ? "safety_check" : "no_policy";
        status("blocked");
      }
      if (!progress_pose_ || exploration_progressed(*progress_pose_, pose, progress_distance_)) {
        progress_pose_ = pose;
        progress_started_ = current;
      } else if (std::chrono::duration<double>(current - progress_started_).count() >=
                 progress_timeout_) {
        if (failed_.size() == 256) {
          failed_.erase(failed_.begin());
        }
        failed_.push_back({preferred_.value_or(pose), current});
        preferred_.reset();
        progress_pose_.reset();
        requested_ = true;
        geometry_cv_.notify_one();
        output = geometry_msgs::msg::Twist();
        status("stalled");
      }
    }
    pub_action_->publish(output);
  }

  // Called with mu_ held. Geometry is deliberately not cancelled: its raw
  // result will be filtered against the newest history by the planner worker.
  void request_consumption() {
    terminals_pending_ = true;
    epoch_.fetch_add(1);
    cv_.notify_one();
  }

  std::optional<Observation> publish_field(const Layers &layers, const Observation &pose) {
    const auto field = planner_.view();
    if (!field) {
      return std::nullopt;
    }
    auto value = layers.costmap;
    const int heading = heading_index(pose.yaw, field->t_resolution_deg);
    for (int y = 0; y < field->ny; ++y) {
      for (int x = 0; x < field->nx; ++x) {
        const double cost = static_cast<double>(field->cost[field->index(x, y, heading)]) /
                            Scale::prob_base;
        value.data[static_cast<std::size_t>(y) * field->nx + x] =
            static_cast<std::int8_t>(std::min(100.0, cost));
      }
    }
    pub_value_->publish(value);
    nav_msgs::msg::Path path;
    path.header = layers.header;
    std::vector<std::uint8_t> visited(field->cost.size(), 0);
    Observation step = pose;
    std::optional<Observation> target;
    const auto limit = std::min<std::size_t>(visited.size(), 10000);
    for (std::size_t n = 0; n < limit; ++n) {
      const int x = static_cast<int>(std::floor((step.x - field->origin_x) / field->resolution));
      const int y = static_cast<int>(std::floor((step.y - field->origin_y) / field->resolution));
      const int t = heading_index(step.yaw, field->t_resolution_deg);
      if (x < 0 || y < 0 || x >= field->nx || y >= field->ny || t < 0 || t >= field->nt) {
        break;
      }
      const auto index = field->index(x, y, t);
      if (visited[index]) {
        break;
      }
      visited[index] = 1;
      geometry_msgs::msg::PoseStamped point;
      point.header = path.header;
      point.pose = pose_msg(step);
      path.poses.push_back(point);
      if (field->final_state[index]) {
        target = step;
        break;
      }
      const int action = field->action[index];
      if (action < 0) {
        break;
      }
      const auto &motion = planner_.actions()[static_cast<std::size_t>(action)];
      step.x += motion.forward_m * std::cos(step.yaw);
      step.y += motion.forward_m * std::sin(step.yaw);
      step.yaw = std::remainder(step.yaw + motion.rotate_deg * kPi / 180.0, 2 * kPi);
    }
    pub_path_->publish(path);
    return target;
  }

  void geometry_main() {
    for (;;) {
      Layers::ConstSharedPtr layers;
      Observation pose{};
      std::vector<Observation> failed;
      std::optional<Observation> preferred;
      std::optional<ExplorationGoal> goal;
      std::uint64_t generation;
      {
        std::unique_lock<std::mutex> lock(mu_);
        geometry_cv_.wait(lock, [&] {
          return stop_ || (requested_ && have_pose_ && latest_ && !goal_reached_ &&
                           (!goal_ || goal_frame_ == latest_->header.frame_id));
        });
        if (stop_) {
          return;
        }
        layers = latest_;
        pose = pose_;
        preferred = preferred_;
        goal = goal_;
        generation = generation_;
        for (const Visit &visit : failed_) {
          failed.push_back(visit.pose);
        }
        requested_ = false;
        geometry_planning_ = true;
      }
      auto candidates = std::make_shared<Candidates>();
      candidates->layers = layers;
      candidates->grid = grid_from(*layers);
      candidates->raw = observation_candidates(
          candidates->grid, pose.x, pose.y, theta_cells_, options_, failed,
          preferred ? &*preferred : nullptr, goal ? &*goal : nullptr);
      {
        std::lock_guard<std::mutex> lock(mu_);
        geometry_planning_ = false;
        if (stop_) {
          return;
        }
        if (generation != generation_ || !latest_ || goal_reached_) {
          continue;
        }
        // A changed raster cancels this generation. Header-only refreshes may
        // reuse identical geometry. Install only after applying current evidence.
        candidates_ = std::move(candidates);
        last_geometry_ = Clock::now();
        request_consumption();
      }
    }
  }

  void worker_main() {
    for (;;) {
      Layers::ConstSharedPtr layers;
      Observation pose{};
      ObservedRegions observed;
      std::shared_ptr<const Candidates> candidates;
      std::optional<ExplorationGoal> goal;
      std::optional<Scan> scan;
      bool scan_only = false;
      std::size_t observations = 0, failures = 0;
      std::uint32_t ticket = 0;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] {
          return stop_ || (have_pose_ && latest_ && !goal_reached_ &&
                           (!goal_ || goal_frame_ == latest_->header.frame_id) &&
                           ((terminals_pending_ && candidates_) ||
                            (scan_pending_ && ready_)));
        });
        if (stop_) {
          return;
        }
        pose = pose_;
        goal = goal_;
        ticket = epoch_.load();
        scan_only = !terminals_pending_;
        candidates = candidates_;
        layers = scan_only ? solved_ : candidates->layers;
        scan = latest_scan_;
        scan_pending_ = false;
        observed = observed_regions_;
        observations = observed_regions_.size();
        failures = failed_.size();
        terminals_pending_ = false;
        if (!scan_only) {
          status("planning");
        }
      }
      const auto apply_scan = [&] {
        if (!scan || scan->frame != layers->header.frame_id) {
          return false;
        }
        const auto &message = scan->message;
        return local_.apply_scan(message.ranges, message.angle_min, message.angle_increment,
                                 message.range_min, message.range_max, scan->x, scan->y, scan->yaw);
      };
      if (scan_only) {
        bool ready = true;
        if (apply_scan()) {
          ready = local_.update(epoch_, ticket).converged;
        }
        {
          std::lock_guard<std::mutex> lock(mu_);
          if (ticket == epoch_.load()) {
            ready_ = ready;
            if (ready) {
              accepted_view_ = planner_.view();
            }
          }
        }
        continue;
      }
      const auto &grid = candidates->grid;
      const int cell_x = static_cast<int>(std::floor((pose.x - grid.origin_x) / grid.resolution));
      const int cell_y = static_cast<int>(std::floor((pose.y - grid.origin_y) / grid.resolution));
      const bool inside = cell_x >= 0 && cell_y >= 0 &&
                          cell_x < grid.width && cell_y < grid.height;
      const auto cell = inside ? static_cast<std::size_t>(cell_y) * grid.width + cell_x : 0;
      const auto filtering_started = Clock::now();
      auto terminals = consume_observation_terminals(candidates->raw, grid, options_, observed);
      const auto filtering_finished = Clock::now();
      geometry_msgs::msg::PoseArray candidate_poses;
      candidate_poses.header = layers->header;
      for (const auto &terminal : terminals) {
        candidate_poses.poses.push_back(pose_msg({
            grid.origin_x + (terminal.x + 0.5) * grid.resolution,
            grid.origin_y + (terminal.y + 0.5) * grid.resolution,
            (terminal.heading + 0.5) * (360 / theta_cells_) * kPi / 180.0}));
      }
      if (goal) {
        const auto destinations = goal_terminals(grid, theta_cells_, *goal);
        terminals.insert(terminals.end(), destinations.begin(), destinations.end());
      }
      // Consumption-only updates keep both the current field and scan costs.
      // Geometry updates load the exact map used to produce their terminals.
      const auto preparation_started = Clock::now();
      const bool loaded = installed_layers_ == layers || planner_.load_map(
          grid.width, grid.height, grid.resolution, grid.origin_x, grid.origin_y,
          0, 0, 0, 1, grid.costmap, theta_cells_, 0, 0, 0.2, 10, MapType::CostAdd, true);
      if (loaded) {
        installed_layers_ = layers;
      }
      bool ready = false;
      std::optional<Observation> target;
      auto solve_started = Clock::now();
      if (loaded) {
        // Even an empty terminal set must invalidate dependencies and finish
        // its warm solve; otherwise the published view retains old supports.
        global_.prepare_terminals(terminals, true);
        apply_scan();
        solve_started = Clock::now();
        ready = global_.solve(epoch_, ticket).converged;
      }
      const auto solve_finished = Clock::now();
      {
        std::lock_guard<std::mutex> lock(mu_);
        if (ticket != epoch_.load()) {
          continue;
        }
        const auto publishing_started = Clock::now();
        pub_candidates_->publish(candidate_poses);
        if (ready) {
          target = publish_field(*layers, pose);
          const auto command = local_.command(pose.x, pose.y, pose.yaw);
          ready = command.have_action || command.arrived;
        } else {
          nav_msgs::msg::Path empty;
          empty.header = layers->header;
          pub_path_->publish(empty);
          auto value = layers->costmap;
          std::fill(value.data.begin(), value.data.end(), 100);
          pub_value_->publish(value);
        }
        solved_ = layers;
        ready_ = ready;
        if (ready) {
          accepted_view_ = planner_.view();
        }
        preferred_ = target;
        const std::string reason = !inside ? "robot_outside_map" :
            grid.costmap[cell] < 0 ? "robot_blocked" :
            !loaded ? "invalid_map" :
            terminals.empty() ? "no_candidates" : ready ? "ready" : "unreachable";
        status(reason);
        std::ostringstream text;
        text << "{\"plan_status\":\"" << reason << "\",\"plan_sequence\":" << ++plan_sequence_
             << ",\"map_revision\":" << layers->revision
             << ",\"accepted_scan_stamp_ns\":"
             << (scan ? rclcpp::Time(scan->message.header.stamp).nanoseconds() : 0)
             << ",\"robot_cell_x\":" << cell_x << ",\"robot_cell_y\":" << cell_y
             << ",\"raw_occupancy\":";
        if (inside) {
          text << static_cast<int>(grid.occupancy[cell]);
        } else {
          text << "null";
        }
        text << ",\"movement_cost\":";
        if (inside) {
          text << static_cast<int>(grid.costmap[cell]);
        } else {
          text << "null";
        }
        text << ",\"terminal_count\":" << terminals.size()
             << ",\"observation_history_size\":" << observations
             << ",\"observed_region_count\":" << observations
             << ",\"visibility_truncated\":" << (candidates->raw.truncated ? "true" : "false")
             << ",\"failure_history_size\":" << failures
             << ",\"decision_reason\":\"" << decision_reason_ << "\""
             << ",\"terminal_filter_ms\":"
             << std::chrono::duration<double, std::milli>(
                 filtering_finished - filtering_started).count()
             << ",\"candidate_poses_ms\":"
             << std::chrono::duration<double, std::milli>(
                 preparation_started - filtering_finished).count()
             << ",\"prepare_ms\":"
             << std::chrono::duration<double, std::milli>(
                 solve_started - preparation_started).count()
             << ",\"solve_ms\":"
             << std::chrono::duration<double, std::milli>(
                 solve_finished - solve_started).count()
             << ",\"publish_ms\":"
             << std::chrono::duration<double, std::milli>(
                 Clock::now() - publishing_started).count() << "}";
        std_msgs::msg::String diagnostics;
        diagnostics.data = text.str();
        pub_diagnostics_->publish(diagnostics);
      }
    }
  }

  struct Visit {
    Observation pose;
    Clock::time_point time;
  };
  struct Scan {
    sensor_msgs::msg::LaserScan message;
    double x;
    double y;
    double yaw;
    std::string frame;
  };
  struct Candidates {
    Layers::ConstSharedPtr layers;
    ExplorationGrid grid;
    ObservationCandidates raw;
  };
  Planner planner_;
  GlobalPlanner global_;
  LocalPlanner local_;
  ExplorationOptions options_;
  ExplorationGoal goal_options_;
  tf2_ros::Buffer tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  std::shared_ptr<tf2_ros::MessageFilter<sensor_msgs::msg::LaserScan>> scan_filter_;
  int theta_cells_ = 60;
  double replan_interval_ = 1, revisit_cooldown_ = 120;
  double progress_timeout_ = 15, progress_distance_ = 0.1;
  double input_timeout_ = 2, map_timeout_ = 10, robot_radius_ = 0.2;
  std::string base_frame_;
  std::string decision_reason_ = "waiting_input";
  std::mutex mu_;
  std::condition_variable cv_;
  std::condition_variable geometry_cv_;
  std::thread worker_, geometry_worker_;
  std::atomic<std::uint32_t> epoch_{0};
  bool stop_ = false, requested_ = false, ready_ = false;
  bool geometry_planning_ = false, terminals_pending_ = false;
  bool scan_pending_ = false;
  bool waiting_transform_ = false;
  bool have_pose_ = false;
  bool goal_reached_ = false;
  std::optional<ExplorationGoal> goal_;
  std::string goal_frame_;
  Layers::ConstSharedPtr latest_, solved_;
  std::shared_ptr<const View> accepted_view_;
  std::shared_ptr<const Candidates> candidates_;
  // Only worker_main reads or writes the installed map identity.
  Layers::ConstSharedPtr installed_layers_;
  std::uint64_t generation_ = 0;
  Observation pose_{};
  std::optional<Observation> preferred_, progress_pose_;
  std::vector<Observation> obstacles_;
  std::optional<Scan> latest_scan_;
  std::uint64_t plan_sequence_ = 0;
  ObservedRegions observed_regions_;
  std::vector<Visit> failed_;
  Clock::time_point last_geometry_{}, map_received_{}, scan_received_{};
  Clock::time_point progress_started_{};
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_action_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr pub_value_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pub_candidates_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_status_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_diagnostics_;
  rclcpp::Subscription<Layers>::SharedPtr sub_layers_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr sub_scan_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_goal_;
  rclcpp::TimerBase::SharedPtr timer_;
};
}  // namespace value_iteration_explorationer

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<value_iteration_explorationer::ExplorationNode>());
  rclcpp::shutdown();
  return 0;
}
