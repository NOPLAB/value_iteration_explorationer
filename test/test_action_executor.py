# SPDX-FileCopyrightText: 2026 nop and CIT autonomous robot laboratory
# SPDX-License-Identifier: BSD-3-Clause

import math

import pytest

from discrete_action_executor import Pose2, step_command


@pytest.mark.parametrize("forward, expected", [(0.3, 0.12), (-0.3, -0.12)])
def test_translation_has_direction_and_velocity_limit(forward, expected):
    command = step_command(Pose2(0, 0, 0), Pose2(0, 0, 0), forward, 0, False)
    assert command.linear == pytest.approx(expected)
    assert command.angular == 0
    assert not command.complete and not command.turning


@pytest.mark.parametrize("pose, turning, linear, angular, complete", [
    (Pose2(0, 0, 0), False, 0.12, 0.0, False),
    (Pose2(0.3, 0, 0), False, 0.0, 0.5, False),
    (Pose2(0.3, 0, math.radians(20)), True, 0.0, 0.0, True),
])
def test_combined_step_translates_before_rotating(pose, turning, linear, angular, complete):
    command = step_command(Pose2(0, 0, 0), pose, 0.3, math.radians(20), turning)
    assert command.linear == pytest.approx(linear)
    assert command.angular == pytest.approx(angular)
    assert command.complete == complete


def test_rotation_completion_wraps_at_pi():
    command = step_command(
        Pose2(0, 0, math.radians(179)), Pose2(0, 0, math.radians(-179)),
        0, math.radians(2), True)
    assert command.complete
    assert command.linear == command.angular == 0
