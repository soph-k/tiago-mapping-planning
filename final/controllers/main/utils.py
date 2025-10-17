import os                                                                                 # stdlib: file paths
import heapq                                                                              # stdlib: priority queue
import numpy as np                                                                        # math & arrays
from typing import Callable, Dict, List, Optional, Tuple                                   # typing helpers
from collections import deque                                                             # simple deque
from controller import Robot                                                              # Webots Robot
from config import RobotConfig                                                            # global config


# ------------------------------ Small helpers ------------------------------

def dev(robot, name):                                                                     # safe getDevice
    return getattr(robot, "getDevice", lambda _n: None)(name)

def enable(sensor, step):                                                                 # enable sensor 
    try:
        if sensor: sensor.enable(step)
    except Exception:
        pass

def rtime(robot) -> float:                                                                # sim time 
    return getattr(robot, "getTime", lambda: 0.0)()

def rstep(robot, step) -> int:                                                            # advance sim
    return getattr(robot, "step", lambda _s: -1)(step)

def safe(val, default=None):                                                              # call and catch
    try:
        return val()
    except Exception:
        return default

def clamp(v, lo, hi):                                                                     # float clamp
    return float(np.clip(v, lo, hi))

def wrap_angle(angle: float) -> float:                                                    # normalize to
    return np.arctan2(np.sin(angle), np.cos(angle))

def compute_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:          # Euclidean 2D
    return float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))

def world_to_pixel(wx: float, wy: float) -> Tuple[int, int]:                              # world to pixel
    # float math for transform; round only for indices                                       #
    fx = (wx - RobotConfig.MAP_ORIGIN_X) / RobotConfig.MAP_RESOLUTION
    if RobotConfig.MAP_Y_AXIS_UP:
        fy = (wy - RobotConfig.MAP_ORIGIN_Y) / RobotConfig.MAP_RESOLUTION                 # Y up
    else:
        fy = (RobotConfig.MAP_ORIGIN_Y - wy) / RobotConfig.MAP_RESOLUTION                 # Y down 
    px = int(np.round(np.clip(fx, 0, RobotConfig.MAP_WIDTH - 1)))                         # clamp & round
    py = int(np.round(np.clip(fy, 0, RobotConfig.MAP_SIZE  - 1)))
    return px, py

def pixel_to_world(px: int, py: int) -> Tuple[float, float]:                              # pixel  world
    wx = (float(px) * RobotConfig.MAP_RESOLUTION) + RobotConfig.MAP_ORIGIN_X
    if RobotConfig.MAP_Y_AXIS_UP:
        wy = (float(py) * RobotConfig.MAP_RESOLUTION) + RobotConfig.MAP_ORIGIN_Y          # Y up
    else:
        wy = RobotConfig.MAP_ORIGIN_Y - (float(py) * RobotConfig.MAP_RESOLUTION)          # Y down 
    return (wx, wy)

def _within_map_envelope(x: float, y: float) -> bool:                                     # inside map bounds?
    x_min = RobotConfig.MAP_ORIGIN_X
    x_max = RobotConfig.MAP_ORIGIN_X + RobotConfig.MAP_WIDTH_METERS
    if RobotConfig.MAP_Y_AXIS_UP:
        y_min = RobotConfig.MAP_ORIGIN_Y
        y_max = RobotConfig.MAP_ORIGIN_Y + RobotConfig.MAP_HEIGHT_METERS
    else:
        y_min = RobotConfig.MAP_ORIGIN_Y - RobotConfig.MAP_HEIGHT_METERS
        y_max = RobotConfig.MAP_ORIGIN_Y
    return (x_min <= x <= x_max) and (y_min <= y <= y_max)

def validate_frame_alignment(gps_pos: Tuple[float, float], compass_orientation: float) -> bool:  # basic checks
    if not RobotConfig.COORDINATE_FRAME_VALIDATION:                                              # toggle
        return True
    try:
        x, y = gps_pos
        if not _within_map_envelope(x, y):
            return False
        if not np.isfinite(compass_orientation):
            return False
        if not (-np.pi <= compass_orientation <= np.pi):
            return False
        return True
    except Exception as e:
        return False

def validate_grid_alignment() -> bool:                                                    # map/grid 
    try:
        expected_res = RobotConfig.MAP_WIDTH_METERS / RobotConfig.MAP_WIDTH
        if abs(expected_res - RobotConfig.MAP_RESOLUTION) > 1e-6:
            return False
        expected_h_px = RobotConfig.MAP_HEIGHT_METERS / RobotConfig.MAP_RESOLUTION
        if abs(expected_h_px - RobotConfig.MAP_SIZE) > 1:
            return False
        test_world = (0.0, 0.0)
        px, py = world_to_pixel(*test_world)
        back_world = pixel_to_world(px, py)
        if abs(test_world[0] - back_world[0]) > 0.01 or abs(test_world[1] - back_world[1]) > 0.01:
            return False
        return True
    except Exception as e:
        return False

def standoff(frm: Tuple[float, float], goal: Tuple[float, float], dist: float = 0.45) -> Tuple[float, float]:  # offset goal
    dx, dy = goal[0] - frm[0], goal[1] - frm[1]
    L = max(1e-6, (dx * dx + dy * dy) ** 0.5)
    if L < dist:
        return goal
    return goal[0] - dist * dx / L, goal[1] - dist * dy / L

def compute_motor_commands(alpha: float, rho: float, C=RobotConfig) -> Tuple[float, float]:  # simple base ctrl
    if abs(alpha) > (np.pi / 2.2):                                                           # large turn
        turn_factor = min(1.0, abs(alpha) / np.pi)
        turn_speed = 0.6 + (C.TURN_SPEED_MAX - 0.6) * turn_factor
        d = 1 if alpha > 0 else -1
        return float(-d * turn_speed), float(d * turn_speed)                                 # spin-in-place
    base = (C.DRIVE_SPEED_MAX * (1 - min(abs(alpha) / (np.pi / 2), 1) * 0.6)) if rho > 0.5 else C.DRIVE_SPEED_MAX * 0.6 * (rho / 0.5)
    base = max(base, 0.6)                                                                    # min speed
    corr = clamp(C.Kp_angle * alpha, -base * 0.85, base * 0.85)                              # steering
    left, right = base - corr, base + corr
    return clamp(left, -C.DRIVE_SPEED_MAX, C.DRIVE_SPEED_MAX), clamp(right, -C.DRIVE_SPEED_MAX, C.DRIVE_SPEED_MAX)

def block_reduce_max(arr: np.ndarray, kx: int, ky: int) -> np.ndarray:                       # max-pool blocks
    H, W = arr.shape
    ph, pw = (kx - (H % kx)) % kx, (ky - (W % ky)) % ky                                      # pad to blocks
    if ph or pw:
        arr = np.pad(arr, ((0, ph), (0, pw)), constant_values=0.0)
    H, W = arr.shape
    return arr.reshape(H // kx, kx, W // ky, ky).max(axis=(1, 3))


# ------------------------------ Shared Memory ------------------------------

class MemoryBoard:
    def __init__(self) -> None:
        self.data: Dict[str, object] = {}                                                   # simple dict
        self.data.update({
            "mapping_waypoints": RobotConfig.MAPPING_WAYPOINTS,                            # WPs
            "jar_positions": RobotConfig.JAR_POSITIONS,                                    # jars
            "dropoff_points": RobotConfig.DROPOFF_POINTS,                                  # drops
            "picked_positions": [],                                                        # picked
            "current_dropoff_index": 0,                                                    # idx
            "current_jar_index": 0,                                                        # idx
            "mapping_complete": False,                                                     # flags
            "cspace_complete": False,
            "navigation_active": False,
            "jar_navigation_active": False,
            "recognized_objects": []
        })
    def set(self, k: str, v: object) -> None: self.data[k] = v                             # write
    def get(self, k: str, d: Optional[object] = None) -> object: return self.data.get(k, d)  # read
    def has(self, k: str) -> bool: return k in self.data                                   # has?
    read, write = get, set                                                                  # aliases


# ------------------------------ Base Device Manager ------------------------------

class RobotDeviceManager:
    def __init__(self, robot: Robot, memory: MemoryBoard) -> None:
        self.robot, self.memory = robot, memory                                            # handles
        self.timestep = int(getattr(robot, "getBasicTimeStep", lambda: 32)())             # ms
        self.last_good_pose = (0.0, 0.0)                                                   # pose cache
        self.last_good_yaw = 0.0                                                           # yaw cache

    def _position(self) -> Tuple[float, float]:                                            # get (x,y)
        gps = self.memory.get("gps")
        if not gps:
            return self.last_good_pose
        pos = gps.getValues()[:2]
        x, y = float(pos[0]), float(pos[1])
        if not np.isfinite(x) or not np.isfinite(y) or not _within_map_envelope(x, y):
            return self.last_good_pose
        self.last_good_pose = (x, y)
        return self.last_good_pose

    def _orientation(self) -> float:                                                       # get yaw
        comp = self.memory.get("compass")
        if not comp:
            return self.last_good_yaw
        v = comp.getValues()
        orientation = float(np.arctan2(v[0], v[1]))                                       # yaw
        if not np.isfinite(orientation):
            return self.last_good_yaw
        self.last_good_yaw = orientation
        return self.last_good_yaw

    def _set_wheel_speeds(self, L: float, R: float) -> None:                               # send wheels
        mL, mR = self.memory.get("motorL"), self.memory.get("motorR")
        if mL and mR:
            mL.setVelocity(L)
            mR.setVelocity(R)

    def _stop(self) -> None:                                                               # halt base
        self._set_wheel_speeds(0.0, 0.0)
