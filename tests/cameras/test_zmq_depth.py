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

"""Depth over the ZMQ image stream: lossless uint16 round trip, color/depth pairing."""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("zmq")

from lerobot.cameras.zmq import ZMQCamera, ZMQCameraConfig  # noqa: E402
from lerobot.cameras.zmq.depth_codec import decode_depth, encode_depth  # noqa: E402
from lerobot.cameras.zmq.image_server import encode_image  # noqa: E402


def _camera(use_depth: bool, message: dict) -> ZMQCamera:
    cam = ZMQCamera(
        ZMQCameraConfig(server_address="127.0.0.1", camera_name="head_camera", use_depth=use_depth)
    )
    cam._connected = True
    cam.context = MagicMock()
    cam.socket = MagicMock()
    cam.socket.recv_string.return_value = json.dumps(message)
    return cam


def _message(depth=None):
    color = np.zeros((48, 64, 3), dtype=np.uint8)
    msg = {"timestamps": {"head_camera": 1.0}, "images": {"head_camera": encode_image(color)}}
    if depth is not None:
        msg["depths"] = {"head_camera": encode_depth(depth)}
    return msg


def test_depth_round_trip_is_lossless():
    rng = np.random.default_rng(0)
    depth = rng.integers(0, 65535, size=(48, 64, 1), dtype=np.uint16)
    depth[0, 0, 0] = 0  # invalid pixels stay 0
    frame, decoded = _camera(True, _message(depth))._read_rgbd_from_hardware()
    assert frame.shape == (48, 64, 3)
    assert decoded.dtype == np.uint16 and decoded.shape == (48, 64, 1)
    np.testing.assert_array_equal(decoded, depth)


def test_depth_ignored_without_use_depth_and_absent_from_old_servers():
    depth = np.full((48, 64), 1234, dtype=np.uint16)
    assert _camera(False, _message(depth))._read_rgbd_from_hardware()[1] is None
    assert _camera(True, _message())._read_rgbd_from_hardware()[1] is None


def test_read_latest_depth_needs_use_depth():
    cam = _camera(False, _message())
    with pytest.raises(RuntimeError, match="use_depth=False"):
        cam.read_latest_depth()


def test_zlib_codec_is_lossless_and_old_png_strings_still_decode():
    import base64

    import cv2

    depth = np.random.default_rng(1).integers(0, 65535, size=(48, 64), dtype=np.uint16)
    encoded = encode_depth(depth[..., None])
    assert encoded.startswith("zlib:48x64:")
    np.testing.assert_array_equal(decode_depth(encoded), depth)
    png = base64.b64encode(cv2.imencode(".png", depth)[1]).decode()
    np.testing.assert_array_equal(decode_depth(png), depth)
