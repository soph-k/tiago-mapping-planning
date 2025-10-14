from __future__ import annotations                # Used in webots and older versions of python.
import os                                         # For device stuff.
import time                                       # Times stamps etc.
import traceback                                  # Better looking print statements.
from enum import Enum                         
from pathlib import Path                          # For file paths.
from dataclasses import dataclass                 # Lightweight class containers for parameters.
from typing import Any, Dict, List, Optional, Tuple, Callable  # Helpers.
import numpy as np                                # For math and arrays.

################################################################################
# =============================== Math utilities ===============================
################################################################################
def NormalizeAngle(angle: float) -> float:        # Normalize angle to -pi, pi to avoid wrap around issues.
    return (angle + np.pi) % (2 * np.pi) - np.pi


def Distance2D(point1, point2) -> float:          # Euclidean distance between two points.
    return np.hypot(point1[0] - point2[0], point1[1] - point2[1])


##################################################################################
# ------------------------ World to Grid coordinate transforms -------------------
##################################################################################
def WorldToGridRaw(world_x: float, world_y: float) -> Tuple[int, int]:
    row = int(40.0 * (world_x + 2.25))            # Meters to grid rows 
    col = int(-52.9 * (world_y - 1.6633))         # Meters to grid cols 
    return row, col                               # Return unclamped

def WorldToGrid(
    world_x: float,                               # World X coordinate
    world_y: float,                               # world Y coordinate
    grid_shape: Tuple[int, int] = (200, 300),  
    clamp: bool = True,                           # Clamp out of bounds to edges
):
    row, col = WorldToGridRaw(world_x, world_y)   # Convert using raw mapping.
    if clamp:
        clamped_row = int(np.clip(row, 0, grid_shape[0] - 1))
        clamped_col = int(np.clip(col, 0, grid_shape[1] - 1))
        return clamped_row, clamped_col
    return (row, col) if (0 <= row < grid_shape[0] and 0 <= col < grid_shape[1]) else None

def GridToWorld(row: int, col: int) -> Tuple[float, float]:
    world_x = row / 40.0 - 2.25                   # Map grid row to world X.
    world_y = -col / 52.9 + 1.6633                # Map grid col to world Y.
    return world_x, world_y                       # Return world coordinates

# def IsWorldCoordInBounds(
#     world_x: float,
#     world_y: float,
#     map_shape: Tuple[int, int] = (200, 300),
# ) -> bool:
#     row, col = WorldToGridRaw(world_x, world_y)
#     return 0 <= row < map_shape[0] and 0 <= col < map_shape[1]

def BresenhamLine(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    points: List[Tuple[int, int]] = []            # Output list of x and y.
    delta_x, delta_y = abs(x1 - x0), abs(y1 - y0) # Horizontal/vertical deltas.
    current_x, current_y = x0, y0                 # Start at first point.
    step_x = 1 if x0 < x1 else -1                 # Step direction in x.
    step_y = 1 if y0 < y1 else -1                 # Step direction in y.
    error = (delta_x - delta_y) if delta_x > delta_y else (delta_y - delta_x)
    for _ in range(max(delta_x, delta_y) + 1):    # Iterate over longest axis.
        points.append((current_x, current_y))     # Add current point
        error_doubled = error * 2                 
        if error_doubled > -delta_y:
            error -= delta_y
            current_x += step_x
        if error_doubled < delta_x:
            error += delta_x
            current_y += step_y
    return points

def UpdateTrajectory(
    trajectory: Optional[List[Tuple[float, float]]],
    point: Tuple[float, float],
    min_distance: float = 0.1,
    max_points: int = 200,
) -> List[Tuple[float, float]]:
    trajectory = trajectory or []
    if not trajectory or Distance2D(trajectory[-1], point) > min_distance:
        trajectory.append(point)
    return trajectory[-max_points:]

################################################################################
# =============================== Path for files ==============================
###############################################################################
def ResolveMapPath(filename: str) -> Path:
    if ("/" in filename) or ("\\" in filename):
        raise ValueError(f"Error filename: {filename}")
    maps_dir = Path(__file__).parent / "maps"
    maps_dir.mkdir(exist_ok=True)
    return (maps_dir / filename).resolve()

def EnsureParentDirectories(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    return path_obj

################################################################################
# ================================ Type aliases ================================
#################################################################################
Position2D = Tuple[float, float]
PathType = List[Position2D]
MapArray = np.ndarray

TH_FREE_PLANNER = 0.45

################################################################################
# ============================= Parameter containers ===========================
#################################################################################
@dataclass
class MappingParams:
    th_occupied: float = 0.75
    th_free_planner: float = TH_FREE_PLANNER
    th_free_explore: float = 0.35
    robot_radius: float = 0.15
    safety_margin: float = 0.00
    map_resolution_m: float = 0.02
    default_map_shape: Tuple[int, int] = (200, 300)
    cspace_inflation_scale: float = 0.35
    cspace_core_obstacle_value: float = 1.0
    cspace_morph_closing: int = 0
    cspace_morph_iters: int = 0
    cspace_downsample: int = 2
    mapping_interval: int = 6
    lidar_offset_x: float = 0.202
    lidar_update_interval: int = 2
    world_to_grid: Optional[Callable] = None
    grid_to_world: Optional[Callable] = None


@dataclass
class PlanningParams:
    th_free_planner: float = TH_FREE_PLANNER
    max_iterations: int = 25000
    heuristic_weight: float = 1.0
    check_neighbor_safety: bool = False
    path_validation_enabled: bool = False
    default_map_shape: Tuple[int, int] = (200, 300)
    verbose: bool = False
    max_open_set_size: int = 50000
    sqrt_2: float = 1.414213562373095
    jump_point_search: bool = True
    bidirectional: bool = True
    early_exit_multiplier: float = 1.5
    adaptive_max_iterations: int = 50000
    adaptive_max_open_set_size: int = 100000
    adaptive_heuristic_weight: float = 1.5
    safe_waypoint_search_radius: int = 12
    optimize_for_differential_drive: bool = True
    differential_drive_alignment_tolerance: float = 0.5
    differential_drive_angle_tolerance: float = 15.0
    W2G: Optional[Callable] = None
    G2W: Optional[Callable] = None



################################################################################
# ================================== Logging ===================================
################################################################################
class LogLevel:
    ERROR, WARNING, INFO, DEBUG, VERBOSE = range(5)

_LOG_LEVEL_MAP = {
    "ERROR": 0,
    "WARNING": 1,
    "INFO": 2,
    "DEBUG": 3,
    "VERBOSE": 4,
}
LOG_LEVEL = _LOG_LEVEL_MAP.get(
    os.environ.get("ROBOT_LOG_LEVEL", "INFO").upper(),
    LogLevel.INFO,
)

class Logger:
    def __init__(self, name: str = "ROBOT", level: Optional[int] = None):
        self.name = name
        self.level = level

    def IsLevelEnabled(self, level: int) -> bool:
        current_level = self.level if self.level is not None else LOG_LEVEL
        return current_level >= level

    def Log(self, level: int, tag: str, message: Any) -> None:
        if self.IsLevelEnabled(level):
            print(f"[{tag}] {self.name}: {message}")

    def Error(self, message: Any) -> None:
        self.Log(LogLevel.ERROR, "ERROR", message)

    def Warning(self, message: Any) -> None:
        self.Log(LogLevel.WARNING, "WARNING", message)

    def Info(self, message: Any) -> None:
        self.Log(LogLevel.INFO, "INFO", message)

    def Debug(self, message: Any) -> None:
        self.Log(LogLevel.DEBUG, "DEBUG", message)
nav_logger = Logger("NAV")
map_logger = Logger("MAP")
plan_logger = Logger("PLAN")
main_logger = Logger("MAIN")


def SetLogLevel(level: int | str, module: Optional[str] = None) -> None:
    level_value = _LOG_LEVEL_MAP[level.upper()] if isinstance(level, str) else int(level)
    global LOG_LEVEL
    if module is None:
        LOG_LEVEL = level_value
        return
    loggers = {
        "nav": nav_logger,
        "map": map_logger,
        "plan": plan_logger,
        "main": main_logger,
    }
    if logger := loggers.get(module.lower()):
        logger.level = level_value
    else:
        main_logger.Warning(f"Unknown '{module}'; valid: {list(loggers.keys())}")

