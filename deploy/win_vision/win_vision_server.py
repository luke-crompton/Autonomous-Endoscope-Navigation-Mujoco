"""
win_vision_server.py -- the Windows half of the deployment loop.

    scope camera (OpenCV index 1)  ->  DA3-SMALL + fine-tuned head  ->
    Da3Depth.to_obs (resize 54x96, per-frame-max normalise to [0,1])  ->
    TCP  ->  depth_bridge (WSL)  ->  /scope/depth

Windows runs NO ROS (see the deployment-architecture memory). This is a plain
socket client. It reuses `perception/realtime/da3_runtime.py` unchanged -- the
one shared inference path -- so what ships here is exactly what the benchmark
timed.

NO fisheye undistortion. The real lens is ~122 deg horizontal where the policy
expects 100 deg; that gap is a knowingly-accepted transfer risk
(docs/CURRENT_PLAN.md section 8), not corrected here.

Run with the Python 3.11 interpreter (DA3 is not installed in 3.14):
    py -3.11 deploy/win_vision/win_vision_server.py --host <WSL_IP> --port 5599

If WSL networkingMode=mirrored, --host 127.0.0.1 works. Otherwise use the WSL
eth0 address (`ip addr` in WSL) or set up a portproxy.

WIRE PROTOCOL  (must match scope_control/scope_link... no -- depth_bridge.py)
--------------------------------------------------------------------------
TCP, TCP_NODELAY. One frame:
    struct "<4sHHIQ" : magic b"SDP1" | u16 height | u16 width | u32 seq | u64 sender_t_ns
    payload          : height*width*4 bytes float32 little-endian, row-major, [0,1]
"""

from __future__ import annotations

import argparse
import glob as _glob
import os
import socket
import struct
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "perception", "realtime"))

_HEADER = struct.Struct("<4sHHIQ")
_MAGIC = b"SDP1"


# --------------------------------------------------------------- frame sources

class CameraSource:
    def __init__(self, index: int, width: int, height: int):
        import cv2
        self._cv2 = cv2
        # CAP_DSHOW: the DirectShow backend -- most reliable for USB webcams on
        # Windows, and needed to set MJPG/resolution on many devices.
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise SystemExit(
                f"could not open camera index {index}.\n"
                "  The scope is OpenCV index 1 (index 2 is the broken NVIDIA Broadcast cam).\n"
                "  Try --camera-index 0/1/2, or --source <video file> for an offline test."
            )
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[win_vision] camera {index} open at {w}x{h}")

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()


class FileSource:
    """A video file or an image glob, looped -- for testing with no scope."""

    def __init__(self, path: str):
        import cv2
        self._cv2 = cv2
        self._frames = None
        self._i = 0
        self._cap = None
        if any(ch in path for ch in "*?[") or os.path.isdir(path):
            pat = os.path.join(path, "*") if os.path.isdir(path) else path
            self._frames = sorted(_glob.glob(pat))
            if not self._frames:
                raise SystemExit(f"no images match {pat}")
            print(f"[win_vision] file source: {len(self._frames)} images, looping")
        else:
            self._cap = cv2.VideoCapture(path)
            if not self._cap.isOpened():
                raise SystemExit(f"could not open video {path}")
            print(f"[win_vision] file source: video {path}, looping")

    def read(self):
        if self._frames is not None:
            img = self._cv2.imread(self._frames[self._i % len(self._frames)])
            self._i += 1
            return img
        ok, frame = self._cap.read()
        if not ok:
            self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        return frame if ok else None

    def close(self):
        if self._cap is not None:
            self._cap.release()


# --------------------------------------------------------------- tcp sender

class DepthSender:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock = None

    def _connect(self):
        while True:
            try:
                s = socket.create_connection((self.host, self.port), timeout=5.0)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.sock = s
                print(f"[win_vision] connected to depth_bridge {self.host}:{self.port}")
                return
            except OSError as e:
                print(f"[win_vision] connect {self.host}:{self.port} failed ({e}); retry in 2s")
                time.sleep(2.0)

    def send(self, obs_hw: np.ndarray, seq: int):
        if self.sock is None:
            self._connect()
        h, w = obs_hw.shape
        payload = np.ascontiguousarray(obs_hw, dtype="<f4").tobytes()
        header = _HEADER.pack(_MAGIC, h, w, seq & 0xFFFFFFFF, time.time_ns())
        try:
            self.sock.sendall(header + payload)
        except OSError as e:
            print(f"[win_vision] send failed ({e}); reconnecting")
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass


# --------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="depth_bridge host (WSL)")
    ap.add_argument("--port", type=int, default=5599)
    ap.add_argument("--camera-index", type=int, default=1)
    ap.add_argument("--source", default=None,
                    help="video file or image dir/glob instead of the camera (offline test)")
    ap.add_argument("--cap-width", type=int, default=1280)
    ap.add_argument("--cap-height", type=int, default=720)
    ap.add_argument("--obs-height", type=int, default=54)
    ap.add_argument("--obs-width", type=int, default=96)
    ap.add_argument("--checkpoint", default=None, help="DA3 fine-tuned head .pth (default: da3_runtime's)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--precision", default="bf16", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--max-fps", type=float, default=0.0, help="cap the loop rate (0 = uncapped)")
    ap.add_argument("--preview", action="store_true", help="show the DA3 obs in a window")
    args = ap.parse_args()

    try:
        from da3_runtime import Da3Depth
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"could not import da3_runtime ({e}).\n"
            "  Run this with the Python 3.11 interpreter -- DA3 is not in 3.14.\n"
            "  See README interpreter matrix."
        )

    kw = dict(device=args.device, precision=args.precision)
    if args.checkpoint:
        kw["checkpoint"] = args.checkpoint
    engine = Da3Depth(**kw)
    print(f"[win_vision] DA3 ready: {engine.describe()}")

    source = (FileSource(args.source) if args.source is not None
              else CameraSource(args.camera_index, args.cap_width, args.cap_height))
    sender = DepthSender(args.host, args.port)

    res = (args.obs_height, args.obs_width)
    period = 1.0 / args.max_fps if args.max_fps > 0 else 0.0
    seq = 0
    t_log = time.time()
    n = 0
    cv2 = None
    if args.preview:
        import cv2  # noqa: F401

    print("[win_vision] streaming. Ctrl-C to stop.")
    try:
        while True:
            t0 = time.time()
            frame = source.read()
            if frame is None:
                print("[win_vision] no frame from source; stopping")
                break

            depth = engine(frame, bgr=True)                 # (280,504) relative
            obs = Da3Depth.to_obs(depth, res=res)[0]         # (H,W) in [0,1]
            sender.send(obs, seq)
            seq += 1
            n += 1

            if args.preview:
                import cv2
                disp = (np.clip(obs, 0, 1) * 255).astype(np.uint8)
                disp = cv2.resize(disp, (args.obs_width * 6, args.obs_height * 6),
                                  interpolation=cv2.INTER_NEAREST)
                cv2.imshow("win_vision obs (54x96)", disp)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            now = time.time()
            if now - t_log >= 2.0:
                print(f"[win_vision] {n / (now - t_log):.1f} fps  (seq {seq})")
                n = 0
                t_log = now

            if period:
                dt = period - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        sender.close()
        if args.preview:
            import cv2
            cv2.destroyAllWindows()
        print("[win_vision] stopped")


if __name__ == "__main__":
    main()
