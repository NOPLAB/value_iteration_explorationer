// SPDX-FileCopyrightText: 2026 nop and CIT autonomous robot laboratory
// SPDX-License-Identifier: BSD-3-Clause

#ifndef VALUE_ITERATION3_ACTIONS_HPP_
#define VALUE_ITERATION3_ACTIONS_HPP_

#include "value_iteration3/planner.hpp"

#include <string>
#include <vector>

namespace value_iteration3 {

struct ActionList {
  std::vector<Action> actions;
  std::string error;
};

// Reads `action_list` from a params file. The list may sit beside
// `ros__parameters`; rclcpp cannot store a list of maps itself.
//
// action_list:
//   - name: forward
//     onestep_forward_m: 0.3
//     onestep_rotation_deg: 0.0
ActionList read_action_list(const std::string &path);

}  // namespace value_iteration3

#endif
