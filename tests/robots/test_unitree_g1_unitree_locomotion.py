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

"""Unitree's built-in locomotion: relay `--locomotion unitree` (arm_sdk + walk command) and the client's
`UnitreeLocomotion` controller."""

import contextlib
import json
import threading
import time
from unittest.mock import MagicMock

import pytest

from lerobot.robots.unitree_g1.safety import LowCmdWatchdog
from lerobot.robots.unitree_g1.unitree_loco_relay import (
    ARM_SDK_SLOTS,
    ARM_SDK_WEIGHT_SLOT,
    ArmSdkForwarder,
    NavForwarder,
    check_walking,
)
from lerobot.robots.unitree_g1.unitree_locomotion import UnitreeLocomotion
from lerobot.utils.import_utils import _unitree_sdk_available

sdk = pytest.mark.skipif(not _unitree_sdk_available, reason="Unitree SDK not available")


def _client_cmd(q=0.5, kp=40.0):
    return {"motor_cmd": [{"mode": 1, "q": q, "dq": 0.0, "kp": kp, "kd": 1.0, "tau": 0.0}] * 29}


def _weight(cmd):
    return cmd["motor_cmd"][ARM_SDK_WEIGHT_SLOT]["q"]


def test_arm_sdk_ramps_in_forwards_only_the_upper_body_and_ramps_out():
    sent = []
    arms = ArmSdkForwarder(sent.append, ramp_in_s=0.1, ramp_out_s=0.2, period_s=0.0)
    arms.tick(0.0, forwarding=True)
    assert sent == []  # no client command yet: Unitree keeps everything
    arms.on_client_cmd(_client_cmd())
    t = 0.0
    for _ in range(8):
        t += 0.02
        arms.tick(t, forwarding=True)
    assert _weight(sent[-1]) == pytest.approx(1.0)
    weights = [_weight(c) for c in sent]
    assert weights == sorted(weights) and weights[0] < 0.5  # ramped, not stepped
    cmd = sent[-1]["motor_cmd"]
    assert all(cmd[i]["kp"] == 0.0 and cmd[i]["q"] == 0.0 for i in range(12))  # legs untouched
    assert all(cmd[i]["kp"] == 40.0 and cmd[i]["q"] == 0.5 for i in ARM_SDK_SLOTS)
    for _ in range(15):  # client gone: back to Unitree over ramp_out_s, holding the last arm command
        t += 0.02
        arms.tick(t, forwarding=False)
    assert _weight(sent[-1]) == 0.0
    assert all(sent[-2]["motor_cmd"][i]["q"] == 0.5 for i in ARM_SDK_SLOTS)
    n = len(sent)
    arms.tick(t + 0.02, forwarding=False)
    assert len(sent) == n  # silent once handed back


def test_arm_sdk_estop_hands_back_at_once():
    sent = []
    arms = ArmSdkForwarder(sent.append, ramp_in_s=0.02, period_s=0.0)
    arms.on_client_cmd(_client_cmd())
    arms.tick(0.0, True)
    arms.tick(0.05, True)
    assert _weight(sent[-1]) == 1.0
    arms.estop()
    arms.tick(0.06, False)
    assert _weight(sent[-1]) == 0.0
    assert arms.weight == 0.0


def test_arm_sdk_release_ramps_to_zero():
    sent = []
    arms = ArmSdkForwarder(sent.append, ramp_in_s=0.02, period_s=0.01)
    arms.on_client_cmd(_client_cmd())
    arms.tick(0.0, True)
    arms.tick(0.05, True)
    arms.release(duration_s=0.05, sleep=lambda _: None)
    assert _weight(sent[-1]) == 0.0
    assert [_weight(c) for c in sent[-5:]] == sorted((_weight(c) for c in sent[-5:]), reverse=True)


def test_nav_forwarder_clips_resends_and_stops():
    calls = []
    nav = NavForwarder(lambda *a: calls.append(a), max_velocity=(0.5, 0.3, 0.8), stale_s=0.3, duration_s=0.5)
    nav.step(0.0, True)
    assert calls == []  # never moved: nothing to send
    nav.update([2.0, -1.0, 0.1], 0.0)
    nav.step(0.1, True)
    nav.step(0.2, True)
    assert calls == [(0.5, -0.3, 0.1, 0.5)] * 2
    nav.step(0.5, True)  # command older than stale_s: stop once
    nav.step(0.6, True)
    assert calls[-1] == (0.0, 0.0, 0.0, 0.5) and len(calls) == 3
    nav.update([0.2, 0.0, 0.0], 0.7)
    nav.step(0.7, False)  # client not forwarding: nothing to stop (was already stopped)
    assert len(calls) == 3
    nav.update([0.2, 0.0, 0.0], 0.8)
    nav.step(0.8, True)
    nav.step(0.85, False)  # watchdog tripped while walking: stop
    assert calls[-1] == (0.0, 0.0, 0.0, 0.5)


def test_check_walking_needs_a_walking_fsm():
    assert check_walking(500) is None and check_walking(501) is None
    assert "not Regular" in check_walking(1)
    assert "not Regular" in check_walking(801)  # Running mode: no arm_sdk
    assert "does not answer" in check_walking(None)


def test_unitree_locomotion_passes_the_walk_command_through():
    ctrl = UnitreeLocomotion()
    assert ctrl.run_step({"kNavVx.cmd": 0.3, "kNavVy.cmd": 0.0, "kNavYawRate.cmd": -0.2}, None) == {}
    assert ctrl.velocity == (0.3, 0.0, -0.2)
    ctrl.reset()
    assert ctrl.velocity == (0.0, 0.0, 0.0)


def test_unitree_locomotion_config_is_real_only():
    from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config

    with pytest.raises(ValueError, match="real-robot only"):
        UnitreeG1Config(controller="UnitreeLocomotion", is_simulation=True)
    UnitreeG1Config(controller="UnitreeLocomotion", is_simulation=False)


@sdk
def test_relay_arm_sdk_loop_forwards_then_hands_back_when_the_client_stops():
    import zmq

    from lerobot.robots.unitree_g1 import run_g1_server

    ctx = zmq.Context()
    pull = ctx.socket(zmq.PULL)
    port = pull.bind_to_random_port("tcp://127.0.0.1")
    push = ctx.socket(zmq.PUSH)
    push.connect(f"tcp://127.0.0.1:{port}")
    sent, vel = [], []
    arms = ArmSdkForwarder(sent.append, ramp_in_s=0.05, ramp_out_s=0.1)
    nav = NavForwarder(lambda *a: vel.append(a))
    watchdog = LowCmdWatchdog(0.1)
    stop = threading.Event()
    thread = threading.Thread(
        target=run_g1_server.arm_sdk_forward_loop,
        args=(pull, arms, nav, watchdog),
        kwargs={"shutdown_event": stop},
        daemon=True,
    )
    thread.start()
    try:
        for _ in range(10):
            msg = {"topic": "rt/lowcmd", "data": _client_cmd(), "session": "s1", "nav": [0.3, 0.0, 0.0]}
            push.send(json.dumps(msg).encode())
            nav.step(time.monotonic(), watchdog.forwarding)
            time.sleep(0.02)
        assert arms.weight == pytest.approx(1.0)
        assert vel and vel[-1][:3] == (0.3, 0.0, 0.0)
        time.sleep(0.4)  # client silent: watchdog trips, weight ramps back to 0
        nav.step(time.monotonic(), watchdog.forwarding)
        assert arms.weight == 0.0 and _weight(sent[-1]) == 0.0
        assert vel[-1][:3] == (0.0, 0.0, 0.0)
    finally:
        stop.set()
        thread.join(timeout=2.0)
        push.close(linger=0)
        pull.close(linger=0)
        ctx.term()


@contextlib.contextmanager
def _unitree_mode_robot(relay_state="unitree"):
    from tests.robots.test_unitree_g1 import _mocked_unitree_g1

    with _mocked_unitree_g1() as (robot, mocks):
        robot.controller = UnitreeLocomotion()
        robot.config.is_simulation = False
        robot._ChannelFactoryInitialize = MagicMock()
        robot._ChannelPublisher = MagicMock(return_value=mocks["publisher_mock"])
        robot._ChannelSubscriber = MagicMock(return_value=mocks["subscriber_mock"])
        mocks["lowstate_msg"].relay_state = relay_state
        written = []

        def capture(msg, handover=False, nav=None):
            written.append(
                {
                    "kp": [float(msg.motor_cmd[i].kp) for i in range(29)],
                    "nav": None if nav is None else list(nav),
                }
            )

        mocks["publisher_mock"].Write.side_effect = capture
        yield robot, mocks, written


@sdk
def test_client_unitree_mode_sends_arms_and_walk_and_never_damps():
    with _unitree_mode_robot() as (robot, mocks, written):
        robot.connect(calibrate=False)
        assert not robot._locomotion_gated
        robot.send_action({"kLeftElbow.q": 0.2, "kNavVx.cmd": 0.4, "kNavVy.cmd": 0.0, "kNavYawRate.cmd": 0.1})
        time.sleep(0.1)
        assert written and written[-1]["nav"] == [0.4, 0.0, 0.1]
        robot.disconnect()
        n = len(written)
        time.sleep(0.05)
        assert len(written) == n  # silent after exit
        assert all(any(kp > 0 for kp in w["kp"]) for w in written)  # never a damping command


@sdk
def test_client_unitree_mode_stale_state_goes_silent_without_damping():
    with _unitree_mode_robot() as (robot, mocks, written):
        robot.config.lowstate_timeout_s = 0.05
        robot.connect(calibrate=False)
        time.sleep(0.05)
        mocks["subscriber_mock"].Read.return_value = None
        time.sleep(0.2)
        assert robot._fault is not None
        n = len(written)
        time.sleep(0.1)
        assert len(written) == n
        assert all(any(kp > 0 for kp in w["kp"]) for w in written)


@sdk
def test_client_unitree_mode_refuses_a_low_level_relay():
    with _unitree_mode_robot(relay_state="damping") as (robot, _, written):
        with pytest.raises(RuntimeError, match="--locomotion unitree"):
            robot.connect(calibrate=False)
        assert written == []


def test_walk_monitor_gates_and_hands_back_once_when_regular_mode_is_lost():
    from lerobot.robots.unitree_g1.unitree_loco_relay import WalkMonitor

    fsm = [500]
    lost = []
    walk = WalkMonitor(lambda: fsm[0], on_lost=lost.append)
    walk.poll()
    assert walk.walking and walk.state() == "unitree" and lost == []
    fsm[0] = 1  # remote L2+B: damping
    walk.poll()
    walk.poll()
    assert not walk.walking and walk.state() == "unitree_not_walking" and lost == [1]
    fsm[0] = None  # loco service silent
    walk.poll()
    assert lost == [1]
    fsm[0] = 501
    walk.poll()
    assert walk.walking


@sdk
def test_relay_arm_sdk_loop_ignores_clients_outside_regular_mode():
    import zmq

    from lerobot.robots.unitree_g1 import run_g1_server

    ctx = zmq.Context()
    pull = ctx.socket(zmq.PULL)
    port = pull.bind_to_random_port("tcp://127.0.0.1")
    push = ctx.socket(zmq.PUSH)
    push.connect(f"tcp://127.0.0.1:{port}")
    sent = []
    arms = ArmSdkForwarder(sent.append, ramp_in_s=0.05)
    watchdog = LowCmdWatchdog(0.1)
    stop = threading.Event()
    thread = threading.Thread(
        target=run_g1_server.arm_sdk_forward_loop,
        args=(pull, arms, NavForwarder(lambda *a: None), watchdog),
        kwargs={"shutdown_event": stop, "accept": lambda: False},
        daemon=True,
    )
    thread.start()
    try:
        for _ in range(5):
            push.send(json.dumps({"topic": "rt/lowcmd", "data": _client_cmd(), "session": "s1"}).encode())
            time.sleep(0.02)
        assert not watchdog.forwarding and sent == [] and arms.weight == 0.0
    finally:
        stop.set()
        thread.join(timeout=2.0)
        push.close(linger=0)
        pull.close(linger=0)
        ctx.term()


@sdk
def test_client_unitree_mode_refuses_a_relay_outside_regular_mode():
    with _unitree_mode_robot(relay_state="unitree_not_walking") as (robot, _, written):
        with pytest.raises(RuntimeError, match="R1\\+X"):
            robot.connect(calibrate=False)
        assert written == []
