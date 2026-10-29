// SPDX-FileCopyrightText: 2026 nop and CIT autonomous robot laboratory
// SPDX-License-Identifier: BSD-3-Clause

#include "value_iteration_explorationer/actions.hpp"
#include "value_iteration_explorationer/global_planner.hpp"
#include "value_iteration_explorationer/local_planner.hpp"
#include "value_iteration_explorationer/planner.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/srv/get_map.hpp"
#include "rclcpp/exceptions.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "tf2/exceptions.h"
#include "tf2/time.h"
#include "tf2/utils.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

namespace value_iteration_explorationer {
namespace {

constexpr double kPi = 3.14159265358979323846;

struct Scan {
  std::vector<float> ranges;
  float angle_min;
  float angle_increment;
  float range_min;
  float range_max;
  double x;
  double y;
  double yaw;
};

}  // namespace

class ViNode : public rclcpp::Node {
 public:
  ViNode() : rclcpp::Node("vi_node"), planner_(0), global_(planner_) {
    online_ = declare_parameter<bool>("online", false);
    const auto map_type = declare_parameter<std::string>("map_type", "occupancy");
    if (map_type == "cost_add" || map_type == "cost_replace") {
      map_type_ = map_type == "cost_add" ? MapType::CostAdd : MapType::CostReplace;
      map_service_ = "/cost_map";
    } else if (map_type != "occupancy") {
      throw std::invalid_argument("map_type must be occupancy, cost_add, or cost_replace");
    }
    theta_cells_ = declare_parameter<int>("theta_cell_num", 60);
    safety_radius_ = declare_parameter<double>("safety_radius", 0.2);
    safety_penalty_ = declare_parameter<double>("safety_radius_penalty", 30.0);
    goal_margin_radius_ = declare_parameter<double>("goal_margin_radius", 0.2);
    goal_margin_theta_ = declare_parameter<int>("goal_margin_theta", 10);
    cost_threshold_ = declare_parameter<int>("cost_drawing_threshold", 60);
    local_range_ = declare_parameter<double>("local_xy_range", 1.0);
    declare_parameter<std::string>("config_file", "");
    const int threads = declare_parameter<int>("global_thread_num", 1);
    planner_.set_thread_num(threads);
    local_ = std::make_unique<LocalPlanner>(planner_, local_range_);
    planner_.set_logger([this](const std::string &text) {
      RCLCPP_INFO(get_logger(), "%s", text.c_str());
    });

    pub_value_ = create_publisher<nav_msgs::msg::OccupancyGrid>("/value_function", 2);
    value_timer_ = create_wall_timer(std::chrono::milliseconds(1500), [this] { publish_value(); });

    if (!online_) {
      return;
    }
    pub_cmd_ = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 1);
    sub_scan_ = create_subscription<sensor_msgs::msg::LaserScan>(
        "/scan", rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::LaserScan::ConstSharedPtr msg) { on_scan(msg); });
    sub_goal_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/goal_pose", 10,
        [this](geometry_msgs::msg::PoseStamped::ConstSharedPtr msg) { on_goal(msg); });
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    decision_timer_ =
        create_wall_timer(std::chrono::milliseconds(100), [this] { on_decision(); });
  }

  ~ViNode() override {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
      epoch_.fetch_add(1, std::memory_order_relaxed);
    }
    cv_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  void init() {
    load_actions();
    while (rclcpp::ok()) {
      try {
        auto client = create_client<nav_msgs::srv::GetMap>(map_service_);
        if (!client->wait_for_service(std::chrono::seconds(1))) {
          if (!rclcpp::ok()) {
            return;
          }
          RCLCPP_INFO(get_logger(), "waiting for %s", map_service_.c_str());
          continue;
        }
        auto request = std::make_shared<nav_msgs::srv::GetMap::Request>();
        auto future = client->async_send_request(request);
        if (rclcpp::spin_until_future_complete(get_node_base_interface(), future) !=
            rclcpp::FutureReturnCode::SUCCESS) {
          if (!rclcpp::ok()) {
            return;
          }
          RCLCPP_ERROR(get_logger(), "map service call failed");
          continue;
        }
        const auto map = future.get()->map;
        std::vector<std::int8_t> occupancy(map.data.begin(), map.data.end());
        const auto &rotation = map.info.origin.orientation;
        if (planner_.load_map(static_cast<int>(map.info.width), static_cast<int>(map.info.height),
                              map.info.resolution, map.info.origin.position.x,
                              map.info.origin.position.y, rotation.x, rotation.y, rotation.z,
                              rotation.w, occupancy, theta_cells_, safety_radius_, safety_penalty_,
                              goal_margin_radius_, goal_margin_theta_, map_type_)) {
          break;
        }
        RCLCPP_ERROR(get_logger(), "rejected occupancy grid");
        std::this_thread::sleep_for(std::chrono::seconds(1));
      } catch (const rclcpp::exceptions::RCLError &) {
        if (!rclcpp::ok()) {
          return;
        }
        throw;
      }
    }
    if (!online_ || !rclcpp::ok()) {
      return;
    }
    worker_ = std::thread([this] { worker_main(); });
  }

 private:
  void load_actions() {
    std::string path = get_parameter("config_file").as_string();
    if (path.empty()) {
      try {
        path = ament_index_cpp::get_package_share_directory("value_iteration_explorationer") +
               "/config/params.yaml";
      } catch (const std::exception &ex) {
        RCLCPP_WARN(get_logger(), "action list: %s", ex.what());
        return;
      }
    }
    const ActionList list = read_action_list(path);
    if (!list.error.empty()) {
      RCLCPP_ERROR(get_logger(), "%s", list.error.c_str());
      return;
    }
    if (!planner_.set_actions(list.actions)) {
      RCLCPP_ERROR(get_logger(), "rejected %zu actions", list.actions.size());
      return;
    }
    std::string names;
    for (const Action &action : planner_.actions()) {
      if (!names.empty()) {
        names += ", ";
      }
      names += action.name;
    }
    RCLCPP_INFO(get_logger(), "actions (%zu): %s", planner_.actions().size(), names.c_str());
  }

  void on_goal(const geometry_msgs::msg::PoseStamped::ConstSharedPtr &msg) {
    const auto &q = msg->pose.orientation;
    const double yaw = std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
    const int degrees = static_cast<int>(yaw * 180.0 / kPi);
    {
      std::lock_guard<std::mutex> lock(mu_);
      goal_x_ = msg->pose.position.x;
      goal_y_ = msg->pose.position.y;
      goal_yaw_ = degrees;
      goal_pending_ = true;
      goal_seen_ = true;
      epoch_.fetch_add(1, std::memory_order_relaxed);
      idling_ = false;
    }
    cv_.notify_all();
    RCLCPP_INFO(get_logger(), "goal %.3f %.3f %d deg", msg->pose.position.x, msg->pose.position.y,
                degrees);
    if (pub_cmd_) {
      pub_cmd_->publish(geometry_msgs::msg::Twist());
    }
  }

  void on_scan(const sensor_msgs::msg::LaserScan::ConstSharedPtr &msg) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (!goal_seen_ || !have_pose_) {
        return;
      }
      if (scans_.size() >= 50) {
        scans_.erase(scans_.begin());
      }
      Scan scan;
      scan.ranges.assign(msg->ranges.begin(), msg->ranges.end());
      scan.angle_min = msg->angle_min;
      scan.angle_increment = msg->angle_increment;
      scan.range_min = msg->range_min;
      scan.range_max = msg->range_max;
      scan.x = x_;
      scan.y = y_;
      scan.yaw = yaw_;
      scans_.push_back(std::move(scan));
    }
    cv_.notify_all();
  }

  void on_decision() {
    if (tf_buffer_) {
      try {
        const auto transform =
            tf_buffer_->lookupTransform("map", "base_link", tf2::TimePointZero);
        std::lock_guard<std::mutex> lock(mu_);
        x_ = transform.transform.translation.x;
        y_ = transform.transform.translation.y;
        yaw_ = tf2::getYaw(transform.transform.rotation);
        have_pose_ = true;
      } catch (const tf2::TransformException &ex) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "%s", ex.what());
      }
    }

    double x = 0;
    double y = 0;
    double yaw = 0;
    bool have_pose = false;
    bool idling = false;
    {
      std::lock_guard<std::mutex> lock(mu_);
      x = x_;
      y = y_;
      yaw = yaw_;
      have_pose = have_pose_;
      idling = idling_;
    }

    geometry_msgs::msg::Twist command;
    if (!idling && have_pose && local_) {
      const LocalPlanner::Command chosen = local_->command(x, y, yaw);
      if (chosen.arrived) {
        std::lock_guard<std::mutex> lock(mu_);
        idling_ = true;
      } else if (chosen.have_action) {
        command.linear.x = chosen.linear_x;
        command.angular.z = chosen.angular_z;
      }
    }
    if (pub_cmd_) {
      pub_cmd_->publish(command);
    }
  }

  void publish_value() {
    const auto field = planner_.view();
    if (!field || field->nx <= 0 || field->ny <= 0) {
      return;
    }
    double yaw = 0;
    {
      std::lock_guard<std::mutex> lock(mu_);
      yaw = yaw_;
    }
    int it = heading_index(yaw, field->t_resolution_deg);
    if (it < 0 || it >= field->nt) {
      it = 0;
    }
    const double threshold = cost_threshold_ > 0 ? static_cast<double>(cost_threshold_) : 1.0;

    nav_msgs::msg::OccupancyGrid grid;
    grid.header.stamp = now();
    grid.header.frame_id = "map";
    grid.info.resolution = static_cast<float>(field->resolution);
    grid.info.width = static_cast<std::uint32_t>(field->nx);
    grid.info.height = static_cast<std::uint32_t>(field->ny);
    grid.info.origin.position.x = field->origin_x;
    grid.info.origin.position.y = field->origin_y;
    grid.info.origin.orientation.x = field->origin_qx;
    grid.info.origin.orientation.y = field->origin_qy;
    grid.info.origin.orientation.z = field->origin_qz;
    grid.info.origin.orientation.w = field->origin_qw;
    grid.data.reserve(static_cast<std::size_t>(field->nx) * field->ny);
    for (int y = 0; y < field->ny; ++y) {
      for (int x = 0; x < field->nx; ++x) {
        const double seconds =
            static_cast<double>(field->cost[field->index(x, y, it)]) / Scale::prob_base;
        std::int8_t value = 100;
        if (seconds < threshold) {
          value = static_cast<std::int8_t>(std::floor(seconds / threshold * 100.0));
        }
        grid.data.push_back(value);
      }
    }
    pub_value_->publish(grid);
  }

  void worker_main() {
    bool ready = false;
    for (;;) {
      std::vector<Scan> scans;
      bool have_goal = false;
      double goal_x = 0;
      double goal_y = 0;
      int goal_yaw = 0;
      std::uint32_t ticket = 0;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] {
          return stop_ || goal_pending_ || (ready && !scans_.empty());
        });
        if (stop_) {
          return;
        }
        if (goal_pending_) {
          have_goal = true;
          goal_x = goal_x_;
          goal_y = goal_y_;
          goal_yaw = goal_yaw_;
          ticket = epoch_.load(std::memory_order_relaxed);
          goal_pending_ = false;
          ready = false;
        } else {
          scans.swap(scans_);
          ticket = epoch_.load(std::memory_order_relaxed);
        }
      }

      if (have_goal) {
        global_.prepare_goal(goal_x, goal_y, goal_yaw);
        const SolveResult result = global_.solve(epoch_, ticket);
        ready = !result.cancelled;
        continue;
      }

      bool changed = false;
      for (const Scan &scan : scans) {
        changed = local_->apply_scan(scan.ranges, scan.angle_min, scan.angle_increment,
                                     scan.range_min, scan.range_max, scan.x, scan.y, scan.yaw) ||
                  changed;
      }
      if (changed) {
        const SolveResult result = local_->update(epoch_, ticket);
        if (result.cancelled) {
          ready = false;
        }
      }
    }
  }

  Planner planner_;
  GlobalPlanner global_;
  std::unique_ptr<LocalPlanner> local_;
  bool online_ = false;
  MapType map_type_ = MapType::Occupancy;
  std::string map_service_ = "/map_server/map";
  int theta_cells_ = 60;
  double safety_radius_ = 0.2;
  double safety_penalty_ = 30.0;
  double goal_margin_radius_ = 0.2;
  int goal_margin_theta_ = 10;
  int cost_threshold_ = 60;
  double local_range_ = 1.0;

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_cmd_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr pub_value_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr sub_scan_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_goal_;
  rclcpp::TimerBase::SharedPtr decision_timer_;
  rclcpp::TimerBase::SharedPtr value_timer_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  std::mutex mu_;
  std::condition_variable cv_;
  std::thread worker_;
  bool stop_ = false;
  bool goal_pending_ = false;
  bool goal_seen_ = false;
  bool have_pose_ = false;
  bool idling_ = true;
  double goal_x_ = 0;
  double goal_y_ = 0;
  int goal_yaw_ = 0;
  double x_ = 0;
  double y_ = 0;
  double yaw_ = 0;
  std::atomic<std::uint32_t> epoch_{0};
  std::vector<Scan> scans_;
};

}  // namespace value_iteration_explorationer

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<value_iteration_explorationer::ViNode>();
  try {
    node->init();
    if (rclcpp::ok()) {
      rclcpp::spin(node);
    }
  } catch (const rclcpp::exceptions::RCLError &) {
    if (rclcpp::ok()) {
      node.reset();
      rclcpp::shutdown();
      throw;
    }
  }
  node.reset();
  rclcpp::shutdown();
  return 0;
}
