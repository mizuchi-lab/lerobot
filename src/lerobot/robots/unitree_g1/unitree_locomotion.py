#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unitree's built-in locomotion as the G1's "controller" (`--robot.controller=UnitreeLocomotion`).

No policy runs on this side: Unitree's loco service on the robot keeps the legs, and the relay must run with
`--locomotion unitree`. The client's controller thread still ticks at 50 Hz, republishing the arm targets (the
relay forwards them to `rt/arm_sdk`) together with the walk command from the teleop (`nav`), which the relay
passes to Unitree's controller. Exiting, Ctrl-C or a lost connection hands the upper body back to Unitree and
stops walking; the robot stays up.
"""

from __future__ import annotations

from .g1_utils import nav_command


class UnitreeLocomotion:
    """Pass-through controller: commands no joints, keeps the latest walk command for the relay."""

    external = True  # Unitree's controller on the robot does the locomotion
    control_dt = 0.02
    kp = None  # the arms use the robot config gains
    kd = None

    def __init__(self) -> None:
        self.velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def run_step(self, action: dict, lowstate) -> dict:
        self.velocity = nav_command(action)
        return {}

    def reset(self) -> None:
        self.velocity = (0.0, 0.0, 0.0)
