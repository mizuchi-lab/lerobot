#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Lossless depth-map strings for the ZMQ image stream (server `image_server.py`, client `camera_zmq.py`)."""

import base64
import zlib

import cv2
import numpy as np


def encode_depth(depth: np.ndarray) -> str:
    """Encode a uint16 depth map (mm, (H, W) or (H, W, 1)) losslessly as "zlib:<H>x<W>:<base64 of zlib(level 1) of
    the little-endian uint16 bytes>". On the Jetson this takes 6 ms for a 640x480 D455 frame, half of a level-1
    16-bit PNG (12.4 ms) at the same size (~127 KiB); the encoding delays each color frame, so it matters."""
    depth = np.ascontiguousarray(depth.reshape(depth.shape[0], depth.shape[1]), dtype="<u2")
    payload = base64.b64encode(zlib.compress(depth.tobytes(), 1)).decode("ascii")
    return f"zlib:{depth.shape[0]}x{depth.shape[1]}:{payload}"


def decode_depth(encoded: str) -> np.ndarray:
    """Inverse of `encode_depth` (also reads the earlier base64 16-bit PNG form): uint16 (H, W) in mm."""
    if encoded.startswith("zlib:"):
        _, shape, payload = encoded.split(":", 2)
        height, width = (int(v) for v in shape.split("x"))
        raw = zlib.decompress(base64.b64decode(payload))
        return np.frombuffer(raw, dtype="<u2").reshape(height, width).astype(np.uint16)
    depth = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError("not a 16-bit depth PNG")
    return depth
