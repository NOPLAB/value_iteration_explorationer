# SPDX-FileCopyrightText: 2026 nop and CIT autonomous robot laboratory
# SPDX-License-Identifier: BSD-3-Clause

std_msgs/Header header
uint64 revision
nav_msgs/OccupancyGrid costmap
nav_msgs/OccupancyGrid information_map
# Raw occupancy is required to distinguish occluders during visibility scoring.
nav_msgs/OccupancyGrid occupancy
