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

"""Relay side of `run_g1_server.py --locomotion unitree`: Unitree's own locomotion keeps the legs, the client
gets the upper body.

Unitree's loco service stays in charge of the robot (the relay never releases it and never publishes
`rt/lowcmd`). The client's lowcmds go to `rt/arm_sdk`: only the waist and arm slots, blended with Unitree's
upper-body command by the weight in slot 29 (0 = Unitree, 1 = client). The client's walk command (`nav` in its
lowcmd messages) goes to `LocoClient.SetVelocity` with a short duration, so the robot stops by itself if the relay
goes silent.

When the client stops (exit, Ctrl-C, crash: the lowcmd watchdog trips) the weight ramps back to 0 while the arms
hold the last command, and the walk command is set to zero: the robot keeps standing (or walking in place) on
Unitree's controller. The remote e-stop drops the weight to 0 at once (Unitree's own remote handling stays
active in this mode).

Pure logic, no Unitree SDK import: the relay injects the publisher and the loco call.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

# rt/arm_sdk: waist (12-14) and arm (15-28) slots are commanded; slot 29's q is the blend weight.
ARM_SDK_SLOTS = range(12, 29)
ARM_SDK_WEIGHT_SLOT = 29
NUM_LOWCMD_MOTORS = 35
# Unitree loco FSM ids of Regular mode (remote R1+X; walk, walk with 3dof waist): the robot balances, takes velocity
# commands and accepts rt/arm_sdk. Running mode (R2+A, FSM 801) does not support arm_sdk (Unitree xr_teleoperate).
WALK_FSMS = (500, 501)
DEFAULT_MAX_VELOCITY = (0.6, 0.4, 0.8)  # |vx| m/s, |vy| m/s, |yaw rate| rad/s
ARM_RAMP_IN_S = 1.0
ARM_RAMP_OUT_S = 2.0

_IDLE_MOTOR = {"mode": 0, "q": 0.0, "dq": 0.0, "kp": 0.0, "kd": 0.0, "tau": 0.0}


def check_walking(fsm_id: int | None) -> str | None:
    """None if Unitree's controller is in a walking FSM (clients may command the upper body), else why not."""
    if fsm_id in WALK_FSMS:
        return None
    if fsm_id is None:
        return (
            "Unitree's loco service does not answer (a previous low-level relay released it?). Run "
            "restore_ai_mode.py, then stand the robot up and switch it to walking with the Unitree remote."
        )
    return (
        f"the robot is in loco FSM {fsm_id}, not Regular (walking) mode {WALK_FSMS}. Stand it up and press R1+X on "
        "the Unitree remote (rope attached; Running mode R2+A does not support arm_sdk)."
    )


class WalkMonitor:
    """Polls Unitree's loco FSM (`get_fsm()`, None = no answer) every `period_s`. Client commands are only
    forwarded while `walking` (Regular mode); losing it (remote L2+B damping, a fall, another FSM) calls
    `on_lost(fsm)` once, which hands the upper body back."""

    def __init__(
        self,
        get_fsm: Callable[[], int | None],
        on_lost: Callable[[int | None], None] | None = None,
        period_s: float = 0.5,
    ) -> None:
        self._get_fsm = get_fsm
        self._on_lost = on_lost
        self.period_s = period_s
        self.fsm: int | None = None
        self.walking = False

    def poll(self) -> None:
        try:
            fsm = self._get_fsm()
        except Exception:  # noqa: BLE001
            fsm = None
        was = self.walking
        self.fsm = fsm
        self.walking = fsm in WALK_FSMS
        if was != self.walking:
            print(
                f"unitree loco: FSM {fsm}: "
                + (
                    "Regular mode, clients may command the upper body"
                    if self.walking
                    else "not Regular mode, client commands ignored (stand up + R1+X to resume)"
                ),
                flush=True,
            )
            if was and self._on_lost is not None:
                self._on_lost(fsm)

    def state(self) -> str:
        """The relay's idle state for the lowstate: "unitree" (Regular mode) or "unitree_not_walking"."""
        return "unitree" if self.walking else "unitree_not_walking"

    def run(self, shutdown_event: threading.Event) -> None:
        while not shutdown_event.is_set():
            start = time.monotonic()
            self.poll()
            time.sleep(max(0.0, self.period_s - (time.monotonic() - start)))


class ArmSdkForwarder:
    """Turns client lowcmds into `rt/arm_sdk` commands and owns the blend weight.

    `tick(now, forwarding)` runs from the relay's command loop: while the client forwards, the weight ramps
    to 1 over `ramp_in_s`; otherwise back to 0 over `ramp_out_s`, holding the client's last upper-body command.
    Nothing is published once the weight is back at 0. `publish` gets the bridge's JSON lowcmd form
    (`{"mode_machine", "motor_cmd"}`).
    """

    def __init__(
        self,
        publish: Callable[[dict[str, Any]], None],
        mode_machine: int = 0,
        ramp_in_s: float = ARM_RAMP_IN_S,
        ramp_out_s: float = ARM_RAMP_OUT_S,
        period_s: float = 0.02,
    ) -> None:
        if ramp_in_s <= 0 or ramp_out_s <= 0:
            raise ValueError("ramp times must be > 0")
        self._publish = publish
        self.mode_machine = mode_machine
        self.ramp_in_s = ramp_in_s
        self.ramp_out_s = ramp_out_s
        self.period_s = period_s
        self.weight = 0.0
        self._upper: list[dict[str, Any]] | None = None  # client command for ARM_SDK_SLOTS
        self._last_t: float | None = None
        self._last_pub_t = float("-inf")
        self._published_nonzero = False
        self._estop = threading.Event()
        self._lock = threading.Lock()

    def on_client_cmd(self, data: dict[str, Any]) -> None:
        """Keep the waist/arm part of a forwarded client lowcmd (bridge JSON form)."""
        motors = data.get("motor_cmd", [])
        if len(motors) < ARM_SDK_SLOTS.stop:
            return
        with self._lock:
            self._upper = [dict(motors[i]) for i in ARM_SDK_SLOTS]

    def estop(self) -> None:
        """Hand the upper body back to Unitree at once (next tick)."""
        self._estop.set()

    def command(self, weight: float) -> dict[str, Any]:
        """The arm_sdk command for `weight`: legs untouched (zero), waist/arms from the client."""
        motors = [dict(_IDLE_MOTOR) for _ in range(NUM_LOWCMD_MOTORS)]
        with self._lock:
            upper = self._upper
        if upper is not None:
            for i, cmd in zip(ARM_SDK_SLOTS, upper, strict=True):
                motors[i] = dict(cmd)
        motors[ARM_SDK_WEIGHT_SLOT]["q"] = float(weight)
        return {"mode_machine": self.mode_machine, "motor_cmd": motors}

    def tick(self, now: float, forwarding: bool) -> None:
        dt = 0.0 if self._last_t is None else min(max(now - self._last_t, 0.0), 0.1)
        self._last_t = now
        if self._estop.is_set():
            self._estop.clear()
            if self.weight > 0 or self._published_nonzero:
                self.weight = 0.0
                self._publish(self.command(0.0))
                self._published_nonzero = False
                self._last_pub_t = now
            return
        if forwarding and self._upper is not None:
            self.weight = min(1.0, self.weight + dt / self.ramp_in_s)
        else:
            self.weight = max(0.0, self.weight - dt / self.ramp_out_s)
        if self.weight <= 0.0 and not self._published_nonzero:
            return  # Unitree has the upper body: publish nothing
        if now - self._last_pub_t >= self.period_s or self.weight <= 0.0:
            self._publish(self.command(self.weight))
            self._last_pub_t = now
            self._published_nonzero = self.weight > 0.0

    def release(self, duration_s: float = 1.0, sleep: Callable[[float], None] = time.sleep) -> None:
        """Ramp the weight to 0 over `duration_s` (blocking), e.g. when the relay shuts down."""
        if self.weight <= 0.0 and not self._published_nonzero:
            return
        start_w = self.weight
        steps = max(1, int(duration_s / self.period_s))
        for k in range(1, steps + 1):
            self.weight = start_w * (1.0 - k / steps)
            self._publish(self.command(self.weight))
            sleep(self.period_s)
        self._published_nonzero = False


class NavForwarder:
    """Sends the client's walk command to Unitree's loco service (`set_velocity(vx, vy, wz, duration)`).

    While the client forwards and its last `nav` is younger than `stale_s`, a nonzero command is re-sent every
    `period_s` with `duration_s` (so a silent relay stops the robot by itself); the first zero after motion is
    sent once. Each axis is clipped to `max_velocity`.
    """

    def __init__(
        self,
        set_velocity: Callable[[float, float, float, float], Any],
        max_velocity: Sequence[float] = DEFAULT_MAX_VELOCITY,
        period_s: float = 0.1,
        stale_s: float = 0.3,
        duration_s: float = 0.5,
    ) -> None:
        if len(max_velocity) != 3 or any(v < 0 for v in max_velocity):
            raise ValueError(f"max_velocity must be three values >= 0, got {max_velocity!r}")
        self._set_velocity = set_velocity
        self.max_velocity = tuple(float(v) for v in max_velocity)
        self.period_s = period_s
        self.stale_s = stale_s
        self.duration_s = duration_s
        self._nav: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._nav_t: float | None = None
        self._moving = False
        self._lock = threading.Lock()

    def update(self, nav: Sequence[float] | None, now: float) -> None:
        if nav is None or len(nav) != 3:
            return
        clipped = tuple(
            min(max(float(v), -limit), limit) for v, limit in zip(nav, self.max_velocity, strict=True)
        )
        with self._lock:
            self._nav = clipped
            self._nav_t = now

    def step(self, now: float, forwarding: bool) -> None:
        with self._lock:
            nav, nav_t = self._nav, self._nav_t
        fresh = forwarding and nav_t is not None and now - nav_t <= self.stale_s
        target = nav if fresh else (0.0, 0.0, 0.0)
        if any(v != 0.0 for v in target):
            self._call(target)
            self._moving = True
        elif self._moving:
            self._call(target)
            self._moving = False

    def stop(self) -> None:
        self._call((0.0, 0.0, 0.0))
        self._moving = False

    def run(self, shutdown_event: threading.Event, forwarding: Callable[[], bool]) -> None:
        while not shutdown_event.is_set():
            start = time.monotonic()
            self.step(start, forwarding())
            time.sleep(max(0.0, self.period_s - (time.monotonic() - start)))

    def _call(self, v: tuple[float, float, float]) -> None:
        try:
            self._set_velocity(v[0], v[1], v[2], self.duration_s)
        except Exception as e:  # noqa: BLE001
            print(f"unitree loco: SetVelocity{v} failed: {e}", flush=True)
