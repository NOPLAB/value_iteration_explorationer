# value_iteration_explorationer

Exploration fork of [NOPLAB/value_iteration3_cpp](https://github.com/NOPLAB/value_iteration3_cpp).
Header paths and the `value_iteration3` C++ namespace are retained from upstream.
The action parser, local planner, original solver tests, and goal-driven node
remain; exploration uses the same solver and `cost_add`.

## Exploration

```text
Cartographer /map
  -> exploration_map_node
       -> /exploration/costmap          movement cost
       -> /exploration/information_map  observation interest
       -> /exploration/map_layers      atomic pair + raw occupancy + revision
  -> exploration_node (+ map -> base_link TF and /scan)
       -> /exploration/action          discrete action, NOT a velocity
       -> /exploration/candidates      safe observation poses
       -> /exploration/path            nominal policy rollout for inspection
       -> /exploration/status
       -> /value_function
```

`/goal_pose` is optional; without a goal the node explores for information alone.
The costmap uses the original `cost_add` contract:
`-1` is blocked; `0..100` is extra cost in seconds, added to the solver's 1 second
base cost. **100 is traversable, not a lethal-cost sentinel.** The map producer
blocks unknown cells and cells above `free_threshold`, inflates them by
`robot_radius + safety_margin`, and adds soft clearance costs. The robot is
modeled as a circle; map exterior also constrains its footprint.
The default `free_threshold` is 49. Initial online Cartographer maps can have
free cells around 35 rather than 0, so a threshold of 25 can block free space.
Values 50 and above remain blocked.

The information map is separate: `0..100` is observation interest, not movement
cost. Unknown cells receive `unknown_value`. Known cells receive
`ambiguous_value * 4*p*(1-p)`, where `p = occupancy/100`. This is a heuristic for
occupancy ambiguity, not Cartographer localization confidence or observation
count. It uses a fixed scale, not normalization against each map's maximum.

The planner evaluates reachable free observation poses, counts weighted visible
information within `sensor_range` and `sensor_fov_deg`, and uses
`information_cost * (1 - min(gain/gain_reference, 1))` as a terminal cost.
Gain is in square meters weighted by information/100 and distance. Raw occupancy
stops visibility rays; the inflated costmap controls robot motion. Only the
first unknown/ambiguous boundary is visible, not the unknown space behind it.
Observation terminals have `observation_radius` spatial extent, as the original
goal margins do, to accommodate stochastic successor cells. This is a local
approximation to observation gain over that region.

### Exploration with a goal

Publish a `geometry_msgs/PoseStamped` on `/goal_pose` to give exploration a
destination. The pose must be planar, have a unit quaternion, and use the map
layers' frame (normally `map`); other frames are rejected, not transformed.
A goal can arrive before the first map. Planning waits if that map's frame does
not match the stored goal.

Safe free cells within `goal_margin_radius` (0.3 m) and
`goal_margin_theta` (10 degrees) of the goal form zero-cost terminals.
Each observation terminal also includes its straight-line distance to the goal
times `goal_cost_per_meter` (default 3.3333333333333335 seconds per meter, matching
the shipped 0.3 m forward step's 1 second base cost).
The value field compares movement cost plus these terminal costs, so information
and progress toward the goal compete in the same solve. This distance is a
heuristic: it does not account for walls, turns, or future map discoveries.
Terminal costs are capped at 100 million seconds.

Goals in unknown, blocked, disconnected, or out-of-map space do not make that
space traversable. The node can continue through reachable observation poses
while the map develops, but cannot guarantee discovering a route.
Goal terminal regions must contain whole grid cells and heading bins; coarse
maps or heading bins may require larger goal margins.

Reaching the goal's position and yaw tolerances publishes zero actions and
`goal_reached` on `/exploration/status`. The node stays stopped until another
valid goal arrives. Reaching an observation terminal still continues exploration.
A new goal replaces the old one, cancels its pending solve, and stops action
output until the replacement field is ready. Invalid goals leave the current
goal unchanged.

Recent observation poses are suppressed within `revisit_radius` and
`revisit_angle_deg` for `revisit_cooldown` seconds. Observations are recorded
while moving; arrival does not start a timed observation stop or wait for a map.
Exploration also selects a value-based action at an observation terminal, so
reaching one only updates observation history and does not itself emit a stop.
Lack of translation or rotation for `progress_timeout` suppresses the current
target and requests another plan. `switch_penalty` favors the previous target
when nominal rollout identifies one. History is bounded to 256 observations.
`no_candidates` does not claim the entire environment is mapped; cooldowns,
visibility limits, or disconnected space can leave unexplored regions.

Map updates are coalesced, not queued. The value field is refreshed every
`replan_interval`, including when no new map or arrival event occurs. Actions
continue from the latest immutable field snapshot while the worker updates it;
each action is checked against the newest costmap and scan. Working values are
retained across aligned map updates, including integer-cell origin shifts and
map growth. Added obstacles, increased costs and removed observation terminals
invalidate their old dependent values before propagation. A changed resolution
or non-aligned origin requires rebuilding the working raster; its old snapshot
remains readable until the replacement is published. Long updates publish an
intermediate snapshot between sweeps every 100 ms. Transition
tables are reused when raster geometry is unchanged. Swept translations are
checked in exploration mode so an action cannot jump over blocked cells.
Fresh scans update the original local penalties and independently veto unsafe
actions. Missing/stale TF, scans, or map stop action output. Inflation and scan
checks use the same circular footprint parameters.
Scan-time TF lookup waits up to 0.1 seconds for the SLAM transform to arrive,
instead of discarding fresh scans whose transform follows by a few milliseconds.

Two map topics are published for inspection, but the planner consumes only
`ExplorationMap` on `/exploration/map_layers` so it cannot mix revisions.
The bundle includes raw occupancy for occlusion checks. All grids share the
same header, resolution, origin and dimensions. Only planar, unrotated maps
are accepted. Input `/map` uses reliable/volatile QoS (waits for the next update);
output layers use reliable/transient-local QoS.

### Build and launch

Run from a ROS 2 workspace root, with this package under `src/` (keep the
workspace's `build/`, `install/`, and `log/` outside the package source).

```bash
colcon build --packages-select value_iteration_explorationer \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
ros2 launch value_iteration_explorationer exploration.launch.py
```

Exploration parameters are in `config/exploration.yaml`. The original action
list remains in `config/params.yaml` without changes.

**`/exploration/action` preserves the original discrete-action convention:**
`linear.x` is a forward displacement in meters and `angular.z` is a rotation in
radians, repeated at 10 Hz. Connect an action executor that understands this
contract; do not send it directly to a standard velocity controller. The
`action_topic` launch argument can select an existing compatible executor input.
`/exploration/path` is a bounded nominal visualization of a stochastic policy,
not a collision-checked trajectory for a path-following controller.

For Cartographer and exploration together:

```bash
ros2 launch value_iteration_explorationer cartographer_exploration.launch.py \
  configuration_directory:=/absolute/path/to/your/cartographer/config \
  configuration_basename:=your_robot.lua \
  scan_topic:=/scan use_sim_time:=false
```

The source example is `example/cartographer_exploration.launch.py`. It requires
`cartographer_ros` and an existing robot-specific Lua configuration, and starts
both Cartographer and its occupancy-grid node. Sensors, robot TF and any odometry
required by that Lua configuration must already be running. Optional arguments
include `map_resolution`, `map_publish_period_sec`, `params_file`, and
`action_topic`. It does not invent a robot sensor configuration.
This example also starts RViz with `example/exploration.rviz`: map, robot, scan,
and policy rollout are visible; the two derived maps, value field and candidates
can be toggled in Displays. Set `use_rviz:=false` for headless use, or pass
`rviz_config:=/absolute/path/to/your.rviz`. The regular exploration launch does
not start RViz.
For simulation, `execute_actions:=true use_sim_time:=true` starts the optional
`example_action_executor`. It uses `/odom` feedback to execute each translation
then rotation, limits velocities, and stops on zero/stale actions, stale odometry
or scan, or insufficient laser clearance. It continues accepting actions after
each step; there is no measurement-success auto-stop. `odom_topic` and
`cmd_vel_topic` are configurable. This executor is simulator-only and is off by
default; a real robot still needs its own compatible controller.

### Tests and retained code

Run `colcon test --packages-select value_iteration_explorationer
--event-handlers console_direct+ --return-code-on-test-failure`, then
`colcon test-result --verbose`. Tests cover the original solver and map-service
mode, two-layer processing, observation visibility, weighted terminals, swept
collisions, map updates, and actual ROS exploration topics.

The sections below describe the retained **goal-driven `vi_node` mode**.
Exploration uses the separate launch above.

A ROS 2 node that drives a robot to a goal by value iteration. It reads an occupancy grid, computes the cost of reaching the goal from every free cell and heading, and publishes the discrete action at the robot's current pose.

Author: nop (noplab90@gmail.com). License: BSD-3-Clause. Copyright: nop and CIT autonomous robot laboratory. See [License](#license).

## What you need running

Before the node can drive:

- A map server that serves `nav_msgs/GetMap` on `/map_server/map` (the default). Cell value `0` is free. Any other value is occupied. For a raw cost grid, set `map_type: cost_add` or `map_type: cost_replace` and serve it on `/cost_map`.
- A transform from `map` to `base_link`.
- A `sensor_msgs/LaserScan` on `/scan`.
- A base that accepts `geometry_msgs/Twist` on `/cmd_vel`.

The command is not a smooth velocity. While an action is selected, the node republishes that action at 10 Hz: `linear.x` is the action's forward step in meters, and `angular.z` is its rotation in radians. At the goal, and whenever the current cell has no action, it publishes zeros.

## Build and run

Place this directory in a workspace `src` folder, then:

```bash
colcon build --packages-select value_iteration_explorationer --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
ros2 launch value_iteration_explorationer vi.launch.py
```

Send a goal in the `map` frame on `/goal_pose` (`geometry_msgs/PoseStamped`). The node solves from that pose, then starts publishing `/cmd_vel`. Solving can take seconds on a large map. The robot holds still until a policy exists.

`/value_function` (`nav_msgs/OccupancyGrid`) shows the cost for the robot's current heading. The value is `0` at the goal and `100` at `cost_drawing_threshold` seconds or above. It is published about every 1.5 s.

## Topics and the map service

| Name | Type | Role |
|---|---|---|
| `/map_server/map` | `nav_msgs/GetMap` | Occupancy grid, read once at startup with `map_type: occupancy` |
| `/cost_map` | `nav_msgs/GetMap` | Raw cost grid, read once at startup with `map_type: cost_add` or `cost_replace` |
| `/goal_pose` | `geometry_msgs/PoseStamped` | Goal position and yaw in `map` |
| `/scan` | `sensor_msgs/LaserScan` | Local obstacle update. Sensor data QoS |
| `/cmd_vel` | `geometry_msgs/Twist` | Selected action, 10 Hz |
| `/value_function` | `nav_msgs/OccupancyGrid` | Cost of the current heading |
| `map` → `base_link` | TF | Robot pose |

Set `map_type` to `cost_add` or `cost_replace` in `config/params.yaml`, or run `ros2 run value_iteration_explorationer vi_node --ros-args -p map_type:=cost_replace -p online:=true`. Each data byte is interpreted as unsigned: 0–254 is a cost in seconds and 255 (signed -1) is blocked. For example, signed -2 represents cost 254. The two modes differ in how that value is used:

| `map_type` | Immediate cost of a free cell |
|---|---|
| `cost_add` | The 1 second base cost plus the cell value. A cell of `0` costs 1 second, as a free occupancy cell does |
| `cost_replace` | The cell value alone. It replaces the 1 second base cost, so a cell of `0` costs nothing |

Cost grids already contain their desired inflation, so `safety_radius` and `safety_radius_penalty` apply only to occupancy mode. Invalid map types are rejected, including the earlier ambiguous `cost`; use `cost_replace` for the behavior of `value_iteration`'s `cost`.

Set `online` to `false` to load the map and publish nothing else. The launch file leaves it `true`.

## How a goal is stored

A cell is a goal only when both of its checked corners lie inside `goal_margin_radius` of the goal position, and its heading bin lies inside `goal_margin_theta` degrees of the goal yaw. Goal cells have cost 0. Every other free cell starts unreachable.

Headings are `theta_cell_num` equal bins around the circle. With 60 bins, each bin is 6 degrees. The value must be from 1 to 64.

Every free cell already costs 1 second to enter. Cells within `safety_radius` of an occupied cell cost an extra `safety_radius_penalty` seconds.

If the log says that most free cells have no finite cost, the goal region is too small for the action steps. Increase `goal_margin_radius` (0.3 m is a useful next try on a 5 cm map).

## Laser update

`/scan` changes costs inside a square of radius `local_xy_range` around the robot (default 1 m).

Along each return, from 10% to 90% of the range, the local penalty is halved. The cells within two cells of the hit are then given a local penalty of 2048 seconds. Returns that are non-finite or outside the scan's range are ignored. The field is then updated from those cells until the change settles. A new goal clears local penalties.

## Parameters

Numeric defaults below are the values in `config/params.yaml`, which the launch file loads. If you run `vi_node` with no parameter file, `online` is `false` and `global_thread_num` is `1`.

| Parameter | Default | Meaning |
|---|---|---|
| `map_type` | `occupancy` | `occupancy` selects `/map_server/map`; `cost_add` and `cost_replace` select `/cost_map` |
| `online` | `true` | Subscribe to the goal and the scan, and publish `/cmd_vel` |
| `global_thread_num` | `2` | Threads used while solving. `0` uses all logical cores |
| `theta_cell_num` | `60` | Number of heading bins |
| `safety_radius` | `0.2` | Meters around an occupied cell that receive the extra penalty |
| `safety_radius_penalty` | `30.0` | Extra seconds to enter a cell near an obstacle |
| `goal_margin_radius` | `0.2` | Meters. Both checked corners of a goal cell must lie inside this radius |
| `goal_margin_theta` | `10` | Degrees of heading that still count as the goal |
| `cost_drawing_threshold` | `60` | Seconds at which `/value_function` saturates at 100 |
| `local_xy_range` | `1.0` | Meters. Half-width of the square updated from the laser |
| `config_file` | set by the launch file | Path of the YAML file that contains `action_list` |

## Actions

Actions live in `config/params.yaml` under `action_list`, next to `ros__parameters`. ROS 2 cannot store a list of maps as a parameter, so the node reads this list from the file. The number of entries is the number of actions. Add or remove entries to change the set. Each entry needs all three fields:

```yaml
action_list:
  - name: forward
    onestep_forward_m: 0.3
    onestep_rotation_deg: 0.0
  - name: back
    onestep_forward_m: -0.2
    onestep_rotation_deg: 0.0
  - name: right
    onestep_forward_m: 0.0
    onestep_rotation_deg: -20.0
  - name: rightfw
    onestep_forward_m: 0.2
    onestep_rotation_deg: -20.0
  - name: left
    onestep_forward_m: 0.0
    onestep_rotation_deg: 20.0
  - name: leftfw
    onestep_forward_m: 0.2
    onestep_rotation_deg: 20.0
```

`onestep_forward_m` is the forward step in meters. Negative is backward. `onestep_rotation_deg` is the rotation in degrees. Negative is clockwise when viewed from above. At most 127 actions.

The shipped list is six motions: 0.3 m forward, 0.2 m back, a turn of 20 degrees either way, and the same turn combined with 0.2 m forward.

## ROS 2 tests and lint

CI builds and tests on ROS 2 Humble / Ubuntu 22.04. From a sourced ROS workspace:

```bash
rosdep install --from-paths src --ignore-src --rosdistro humble -y
colcon build --packages-select value_iteration_explorationer --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
colcon test --packages-select value_iteration_explorationer --event-handlers console_direct+ --return-code-on-test-failure
colcon test-result --verbose
```

Tests use `ament_cmake_test` and `ament_cmake_pytest`. The solver checks unsigned cost decoding, blocked cells, and unchanged occupancy behavior. ROS integration tests launch the installed node with real `nav_msgs/GetMap` services, check both service routes and map metadata, and verify that raw 254 is traversable while 255 is blocked. Lint uses ament cppcheck (C++ static analysis), flake8 (Python), lint_cmake, and xmllint. CI uploads their result files.

## Solver test

This checks the solver without ROS. It needs a C++17 compiler and pthreads.

```bash
c++ -std=c++17 -O2 -pthread -I include \
  src/planner.cpp src/actions.cpp src/global_planner.cpp src/local_planner.cpp \
  src/test_planner.cpp -o /tmp/planner_test
/tmp/planner_test
```

`planner_test ok` means the action list parsed, two thread counts agreed, a far cell reached the goal, a walled-in cell stayed blocked, and a laser hit raised costs without lowering any.

## Differences from value_iteration and value_iteration2

`value_iteration` is the ROS 1 planner. `value_iteration2` is its ROS 2 port. This package keeps the same kind of goal, scan, and velocity command, and changes when the field is solved and which inputs are accepted.

Shared with both of them:

- An occupancy grid, a pose in `map`, and a laser scan.
- A goal cell must lie inside `goal_margin_radius` and `goal_margin_theta`. The node defaults are 0.2 m and 10 degrees. Many `value_iteration` launch files override those to 0.3 m and 15 degrees.
- The same six motions are shipped: 0.3 m forward, 0.2 m back, ±20 degrees, and ±20 degrees with 0.2 m forward.
- Free cells cost 1 second to enter. Cells within `safety_radius` (0.2 m) cost an extra `safety_radius_penalty` (30 seconds).
- A laser hit inside 1 m adds a local penalty of 2048 seconds, and cells along the beam have that penalty halved.
- `/cmd_vel` repeats the chosen step. It is not a smooth velocity.

Different in this package:

| | `value_iteration` | `value_iteration2` | this package |
|---|---|---|---|
| ROS | ROS 1 | ROS 2 | ROS 2 |
| Goal input | Action `value_iteration/ViAction` | `/goal_pose` | `/goal_pose` |
| Map service | `static_map`, or `cost_map` when `map_type` is `cost` | `/map_server/map`, or `/cost_map` for a cost grid | `/map_server/map`, or `/cost_map` with `map_type: cost_add` or `cost_replace` |
| Value output | `policy` and `value` services (`grid_map_msgs/GetGridMap`) | `/value_function` occupancy grid, every 1.5 s | `/value_function` occupancy grid, every 1.5 s |
| Actions | `action_list` parameter. The count can change | Six actions fixed in the source. The `action_list` in its YAML is commented out | `action_list` in `config/params.yaml`. The count can change, up to 127 |
| Threads | `thread_num`, default 4 | `global_thread_num`, default 1 in code and 2 in its YAML. `local_thread_num` is in the YAML and is never read | `global_thread_num`, default 1 without a file and 2 in `params.yaml`. `0` uses every logical core. There is no `local_thread_num` |
| When solving starts | The field is swept continuously in the background, including before a goal | Same continuous sweep. Threads start toward the origin before any goal | Solving starts when `/goal_pose` arrives, and runs until that goal's field stops changing |
| Laser update | A separate thread keeps sweeping only the 1 m window, even when no new scan arrives. `local_xy_range` is fixed at 1 m | Same one local thread and the same fixed 1 m window | A scan that changes a penalty starts an update from those cells. The update may leave the window, and it stops when that change settles. `local_xy_range` is a parameter |
| Scan samples | Every range in the message | Every range in the message | Drops non-finite ranges and ranges outside the scan's reported limits. Subscription uses sensor-data QoS |
| No action at the current cell | `/cmd_vel` is published only while `online` is true and an action exists | A zero command is published while idling. A cell with no action publishes nothing, so the previous command can remain | A zero command is published, so the robot waits |
| YAML keys `gthread_num`, `gsafety_*`, `ggoal_*`, `gmap_type` | Not used | Present in `params.yaml` and not read by the node | Not used. The names this node reads are listed under [Parameters](#parameters) |

The cost-grid conversion matches `value_iteration2` (`ValueIterator::setMapWithCostGrid` and the raw-cost `State` constructor). Its upstream ROS 2 `/cost_map` service branch is commented out; this package implements that service path with ROS 2 clients. `cost_replace` keeps that conversion; `cost_add` adds the 1 second base cost to each free cell. Neither cost mode applies occupancy safety inflation.

## License

BSD-3-Clause. Copyright (c) 2026, nop and CIT autonomous robot laboratory.

This package refers to two earlier BSD-3-Clause programs by Ryuichi Ueda:

- `value_iteration`, Copyright (c) 2021, CIT autonomous robot laboratory
- `value_iteration2`, Copyright (c) 2024, Ryuichi Ueda and CIT autonomous robot laboratory
