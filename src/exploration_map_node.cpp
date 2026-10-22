// SPDX-FileCopyrightText: 2026 nop and CIT autonomous robot laboratory
// SPDX-License-Identifier: BSD-3-Clause

#include "value_iteration3/map_layers.hpp"

#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <utility>

#include "nav_msgs/msg/occupancy_grid.hpp"
#include "rclcpp/rclcpp.hpp"
#include "value_iteration_explorationer/msg/exploration_map.hpp"

namespace value_iteration3 {

class ExplorationMapNode : public rclcpp::Node {
 public:
  ExplorationMapNode() : rclcpp::Node("exploration_map_node") {
    options_.free_threshold = declare_parameter<int>("free_threshold", 49);
    options_.robot_radius = declare_parameter<double>("robot_radius", 0.15);
    options_.safety_margin = declare_parameter<double>("safety_margin", 0.05);
    options_.inflation_radius = declare_parameter<double>("inflation_radius", 0.5);
    options_.inflation_cost = declare_parameter<int>("inflation_cost", 30);
    options_.unknown_value = declare_parameter<int>("unknown_value", 100);
    options_.ambiguous_value = declare_parameter<int>("ambiguous_value", 100);
    const auto topic = declare_parameter<std::string>("map_topic", "/map");
    const auto output_qos = rclcpp::QoS(1).reliable().transient_local();
    costmap_ = create_publisher<Grid>("/exploration/costmap", output_qos);
    information_ = create_publisher<Grid>("/exploration/information_map", output_qos);
    paired_ = create_publisher<Bundle>("/exploration/map_layers", output_qos);
    input_ = create_subscription<Grid>(
        topic, rclcpp::QoS(1).reliable().durability_volatile(),
        [this](Grid::ConstSharedPtr map) { on_map(*map); });
  }

 private:
  using Grid = nav_msgs::msg::OccupancyGrid;
  using Bundle = value_iteration_explorationer::msg::ExplorationMap;

  void on_map(const Grid &map) {
    try {
      const auto &p = map.info.origin.position;
      const auto &q = map.info.origin.orientation;
      const MapGeometry geometry{map.info.width, map.info.height, map.info.resolution,
                                 p.x, p.y, p.z, q.x, q.y, q.z, q.w};
      auto layers = process_map_layers(geometry, map.data, options_);
      Bundle bundle;
      bundle.header = map.header;
      bundle.revision = ++revision_;
      bundle.occupancy = map;
      bundle.costmap = map;
      bundle.information_map = map;
      bundle.costmap.data = std::move(layers.costmap);
      bundle.information_map.data = std::move(layers.information_map);
      costmap_->publish(bundle.costmap);
      information_->publish(bundle.information_map);
      paired_->publish(bundle);
    } catch (const std::exception &error) {
      RCLCPP_ERROR(get_logger(), "Map processing failed: %s", error.what());
    }
  }

  MapOptions options_;
  std::uint64_t revision_ = 0;
  rclcpp::Publisher<Grid>::SharedPtr costmap_;
  rclcpp::Publisher<Grid>::SharedPtr information_;
  rclcpp::Publisher<Bundle>::SharedPtr paired_;
  rclcpp::Subscription<Grid>::SharedPtr input_;
};

}  // namespace value_iteration3

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<value_iteration3::ExplorationMapNode>());
  rclcpp::shutdown();
  return 0;
}
