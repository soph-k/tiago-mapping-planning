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
    points: List[Tuple[int, int]] = []            # Output list grid points along the line
    delta_x, delta_y = abs(x1 - x0), abs(y1 - y0) # Deltas along x and y between endpoints
    current_x, current_y = x0, y0                 # Initialize current position at the starting point
    step_x = 1 if x0 < x1 else -1                 # Step direction in x
    step_y = 1 if y0 < y1 else -1                 # Step direction in y
    error = (delta_x - delta_y) if delta_x > delta_y else (delta_y - delta_x)
    for _ in range(max(delta_x, delta_y) + 1):    # Iterate once per pixel along the longer axis
        points.append((current_x, current_y))     # Record the current grid point on the line
        error_doubled = error * 2                 # Double the error to avoid floating point
        if error_doubled > -delta_y:              # If error is large enough, advance in x
            error -= delta_y                      # Reduce error by delta_y
            current_x += step_x                   # Step one pixel in x toward the target
        if error_doubled < delta_x:               # If error is small enough, advance in y
            error += delta_x                      # Increase error by delta_x
            current_y += step_y                   # Step one pixel in y toward the target
    return points                                 # Return 


def UpdateTrajectory(
    trajectory: Optional[List[Tuple[float, float]]],
    point: Tuple[float, float],
    min_distance: float = 0.1,
    max_points: int = 200,
) -> List[Tuple[float, float]]:
    trajectory = trajectory or []                                 # Start with empty list if None provided
    if not trajectory or Distance2D(trajectory[-1], point) > min_distance:
        trajectory.append(point)                                   # Add current x and y position
    return trajectory[-max_points:]                                # Keep only the most recent max_points samples


################################################################################
# =============================== Path for files ===============================
################################################################################

def ResolveMapPath(filename: str) -> Path:
    if ("/" in filename) or ("\\" in filename):                   # Disallow path separators for safety
        raise ValueError(f"Error filename: {filename}")           # Guard against path traversal
    maps_dir = Path(__file__).parent / "maps"                     # Maps directory next to this file
    maps_dir.mkdir(exist_ok=True)                                 # Ensure the directory exists
    return (maps_dir / filename).resolve()                        # Return absolute path to the file


def EnsureParentDirectories(path: str | Path) -> Path:
    path_obj = Path(path)                                         # Normalize to Path object
    path_obj.parent.mkdir(parents=True, exist_ok=True)            # Create all parent directories
    return path_obj                                               # Return the normalized Path


################################################################################
# ================================ Type aliases ================================
################################################################################
Position2D = Tuple[float, float]                                   # X and Y in world coordinates
PathType = List[Position2D]                                        # Sequence of 2D positions
MapArray = np.ndarray                                              # Alias for numpy map arrays
TH_FREE_PLANNER = 0.45                                             # Threshold used by planner for free cells


################################################################################
# ============================= Parameter containers ===========================
################################################################################
@dataclass
class MappingParams:
    th_occupied: float = 0.75                                       # Prob for obstacle
    th_free_planner: float = TH_FREE_PLANNER                        # Prob free for planning
    th_free_explore: float = 0.35                                   # Slightly looser free threshold for exploration
    robot_radius: float = 0.15                                      # Robot radius in meters
    safety_margin: float = 0.00                                     # Extra margin added to radius
    map_resolution_m: float = 0.02                                  # Map cell size
    default_map_shape: Tuple[int, int] = (200, 300)                 # Fallback grid size 
    cspace_inflation_scale: float = 0.35                            # Inflation scale for c-space obstacles
    cspace_core_obstacle_value: float = 1.0                         # Value assigned to hard obstacles in c-space
    cspace_morph_closing: int = 0                                   
    cspace_morph_iters: int = 0                                     # Number of iterations
    cspace_downsample: int = 2                                      # Downsampling factor applied to c-space
    mapping_interval: int = 6                                       # Build/refresh c-space every N frames
    lidar_offset_x: float = 0.202                                   # Lidar sensor x-offset from robot base
    lidar_update_interval: int = 2                                  # Only use every Nth Lidar frame
    world_to_grid: Optional[Callable] = None                         
    grid_to_world: Optional[Callable] = None                        

@dataclass
class PlanningParams:
    th_free_planner: float = TH_FREE_PLANNER                         # Free-space threshold for planning
    max_iterations: int = 25000                                      # Max A*/JPS expansions
    heuristic_weight: float = 1.0                                    # Weight on heuristic
    check_neighbor_safety: bool = False                              # Validate neighbor transitions for safety
    path_validation_enabled: bool = False                            # Post validate path against c-space
    default_map_shape: Tuple[int, int] = (200, 300)                  # Fallback grid dimensions
    verbose: bool = False                                           
    max_open_set_size: int = 50000                                  
    sqrt_2: float = 1.414213562373095                                # sqrt(2) 
    jump_point_search: bool = True                                   # Enable JPS acceleration
    bidirectional: bool = True                                       # Enable bidirectional search
    early_exit_multiplier: float = 1.5                               # Exit early when heuristic close to goal
    adaptive_max_iterations: int = 50000                             # Raised limits if needed
    adaptive_max_open_set_size: int = 100000                         # Raised open set limit
    adaptive_heuristic_weight: float = 1.5                           # Heavier heuristic when adapting
    safe_waypoint_search_radius: int = 12                            # Radius to sample safe waypoints
    optimize_for_differential_drive: bool = True                     # Smooth/align path for diff-drive robots
    differential_drive_alignment_tolerance: float = 0.5              # Alignment tolerance
    differential_drive_angle_tolerance: float = 15.0                 # Heading tolerance
    W2G: Optional[Callable] = None                                   # World to grid mapper to inject at runtime
    G2W: Optional[Callable] = None                                   # Grid to world mapper to inject at runtime


################################################################################
# ================================== Logging ===================================
################################################################################
class LogLevel:
    ERROR, WARNING, INFO, DEBUG, VERBOSE = range(5)           

_LOG_LEVEL_MAP = {
    "ERROR": 0,                                                     # Text name to numeric level
    "WARNING": 1,
    "INFO": 2,
    "DEBUG": 3,
    "VERBOSE": 4,
}

LOG_LEVEL = _LOG_LEVEL_MAP.get(                                     # Default level from environment variables, else INFO for default
    os.environ.get("ROBOT_LOG_LEVEL", "INFO").upper(),
    LogLevel.INFO,
)

class Logger:
    def __init__(self, name: str = "ROBOT", level: Optional[int] = None):
        self.name = name                                            # Label shown in log lines
        self.level = level                                      

    def IsLevelEnabled(self, level: int) -> bool:
        current_level = self.level if self.level is not None else LOG_LEVEL  # Pick local or global level
        return current_level >= level                               # Log only if allowed by current level

    def Log(self, level: int, tag: str, message: Any) -> None:
        if self.IsLevelEnabled(level):                              # Skip if below threshold
            print(f"[{tag}] {self.name}: {message}")                # Basic console output

    def Error(self, message: Any) -> None:
        self.Log(LogLevel.ERROR, "ERROR", message)                  # ERROR: always important

    def Warning(self, message: Any) -> None:
        self.Log(LogLevel.WARNING, "WARNING", message)              # WARNING: something might be wrong

    def Info(self, message: Any) -> None:
        self.Log(LogLevel.INFO, "INFO", message)                    # INFO: normal status

    def Debug(self, message: Any) -> None:
        self.Log(LogLevel.DEBUG, "DEBUG", message)                  # DEBUG: extra details


nav_logger = Logger("NAV")                                          # Navigation logs
map_logger = Logger("MAP")                                          # Mapping logs
plan_logger = Logger("PLAN")                                        # Planning logs
main_logger = Logger("MAIN")                                        # General logs

def SetLogLevel(level: int | str, module: Optional[str] = None) -> None:
    level_value = _LOG_LEVEL_MAP[level.upper()] if isinstance(level, str) else int(level)  # Accept name or number
    global LOG_LEVEL
    if module is None:                                              # No module: set global default
        LOG_LEVEL = level_value
        return
    loggers = {                                                     # Pick a specific logger
        "nav": nav_logger,
        "map": map_logger,
        "plan": plan_logger,
        "main": main_logger,
    }
    if logger := loggers.get(module.lower()):
        logger.level = level_value                                  # Set only that logger's level
    else:
        main_logger.Warning(f"Unknown '{module}'; valid: {list(loggers.keys())}") 


################################################################################
# ================================ Blackboard keys =============================
################################################################################
class BBKey(str, Enum):
    ROBOT = "robot"
    GPS = "gps"
    COMPASS = "compass"
    LIDAR = "lidar"
    MOTOR_L = "motorL"
    MOTOR_R = "motorR"
    DISPLAY = "display"
    TIMESTEP = "timestep"
    INIT_Z = "init_z"
    PROB_MAP = "prob_map"
    CSPACE = "cspace"
    START_XY = "start_xy"
    PLANNED_PATH = "planned_path"
    NAVIGATION_GOALS = "navigation_goals"
    NAVIGATION_GOAL = "navigation_goal"
    TRAJECTORY_POINTS = "trajectory_points"
    STOP_MAPPING = "stop_mapping"
    EMERGENCY_STOP = "emergency_stop"
    MAX_MAPPING_STEPS = "max_mapping_steps"
    MAP_SAVED = "map_saved"
    MAP_READY = "map_ready"
    CSPACE_FROZEN = "cspace_frozen"
    DISPLAY_MODE = "display_mode"
    ALLOW_CSPACE_DISPLAY = "allow_cspace_display"


################################################################################
# ================================= Blackboard =================================
################################################################################
class Blackboard:
    def __init__(self):
        self.data: Dict[str, Any] = {}

    @staticmethod
    def Key(key: BBKey | str) -> str:
        return key.value if isinstance(key, BBKey) else key

    def Set(self, key: BBKey | str, value: Any) -> None:
        self.data[self.Key(key)] = value

    def Get(self, key: BBKey | str, default: Any = None) -> Any:
        return self.data.get(self.Key(key), default)

    def Has(self, key: BBKey | str) -> bool:
        return self.Key(key) in self.data

    def Incr(self, key: BBKey | str, by: int = 1) -> int:
        key_str = self.Key(key)
        new_value = (self.Get(key_str, 0) or 0) + by
        self.Set(key_str, new_value)
        return new_value

    def Remove(self, key: BBKey | str) -> None:
        self.data.pop(self.Key(key), None)

    def Clear(self) -> None:
        self.data.clear()

    def AllowCspaceDisplay(self, value: bool | None = None) -> bool:
        """Get/set whether c-space display is allowed."""
        if value is not None:
            self.Set(BBKey.ALLOW_CSPACE_DISPLAY, bool(value))
        return bool(self.Get(BBKey.ALLOW_CSPACE_DISPLAY, False))

    def ClearMissionData(self) -> None:
        """Reset mission-related entries to known defaults."""
        for k in (BBKey.PLANNED_PATH, BBKey.NAVIGATION_GOALS, BBKey.NAVIGATION_GOAL):
            self.Set(k, None)
        self.Set(BBKey.TRAJECTORY_POINTS, [])
        for k, v in [
            ("stop_mapping", False),
            ("allow_cspace_display", False),
            ("map_saved", False),
            ("display_mode", "full"),
        ]:
            self.Set(k, v)

    def GetRobot(self): return self.Get(BBKey.ROBOT)
    def GetGps(self): return self.Get(BBKey.GPS)
    def GetCompass(self): return self.Get(BBKey.COMPASS)
    def GetLidar(self): return self.Get(BBKey.LIDAR)
    def GetMotors(self): return self.Get(BBKey.MOTOR_L), self.Get(BBKey.MOTOR_R)
    def GetDisplay(self): return self.Get(BBKey.DISPLAY)
    def GetProbMap(self) -> Optional(MapArray): return self.Get(BBKey.PROB_MAP)
    def GetCspace(self) -> Optional(MapArray): return self.Get(BBKey.CSPACE)
    def GetPlannedPath(self) -> Optional(PathType): return self.Get(BBKey.PLANNED_PATH)
    def GetTrajectory(self) -> Optional(PathType): return self.Get(BBKey.TRAJECTORY_POINTS)
    def GetNavigationGoals(self) -> Optional[List[Position2D]]: return self.Get(BBKey.NAVIGATION_GOALS)

    def SetProbMap(self, probability_map: MapArray):
        self.Set(BBKey.PROB_MAP, probability_map)

    def SetCspace(self, configuration_space: MapArray):
        self.Set(BBKey.CSPACE, configuration_space)

    def SetPlannedPath(self, path: PathType):
        self.Set(BBKey.PLANNED_PATH, path)

    def SetTrajectory(self, trajectory: PathType):
        self.Set(BBKey.TRAJECTORY_POINTS, trajectory)

    def SetNavigationGoals(self, goals: List[Position2D]):
        self.Set(BBKey.NAVIGATION_GOALS, goals)

    def SetMapReady(self, ready: bool = True):
        self.Set(BBKey.MAP_READY, ready)

    def SetMapSaved(self, saved: bool = True):
        self.Set(BBKey.MAP_SAVED, saved)

blackboard = Blackboard()

def CreateBlackboard() -> Blackboard:
    return Blackboard()

def SetGlobalBlackboard(new_blackboard: Blackboard) -> None:
    """Swap out the global blackboard instance."""
    global blackboard
    blackboard = new_blackboard


################################################################################
# ================================== BT core ===================================
################################################################################
class Status(Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"

class BehaviorNode:
    def __init__(self, name: str = "Behavior"):
        self.name = name
        self.status = Status.FAILURE
        self.tick_count = 0
        self.last_tick_time: Optional[float] = None
        self.is_paused = False
        self.is_halted = False

    def execute(self) -> Status:
        return Status.FAILURE

    def tick(self) -> Status:
        if self.is_halted:
            return Status.FAILURE
        if self.is_paused:
            return Status.PAUSED
        self.tick_count += 1
        self.last_tick_time = time.time()
        try:
            result = self.execute()
            self.status = result if isinstance(result, Status) else Status.FAILURE
        except Exception as error:
            self.status = Status.FAILURE
            main_logger.Error(f"{self.name} (#{self.tick_count}): {type(error).__name__} - {error}")
            traceback.print_exc()
        return self.status

    def reset(self) -> None:
        self.status = Status.FAILURE
        self.tick_count = 0
        self.is_paused = False

    def terminate(self) -> None:
        pass

    def pause(self) -> None:
        self.is_paused = True
        main_logger.Debug(f"Paused: {self.name}")

    def resume(self) -> None:
        self.is_paused = False
        main_logger.Debug(f"Resumed: {self.name}")

    def halt(self) -> None:
        self.is_halted = True
        self.is_paused = False
        main_logger.Debug(f"Halted: {self.name}")

    def GetInfo(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "tick_count": self.tick_count,
            "last_tick_time": self.last_tick_time,
            "is_paused": self.is_paused,
            "is_halted": self.is_halted,
        }


################################################################################
# ================================= Composites =================================
################################################################################
class _Composite(BehaviorNode):
    def __init__(self, name: str, children: Optional[List[BehaviorNode]] = None):
        super().__init__(name)
        self.children = list(children or [])
        self.current_child = 0

    def reset(self) -> None:
        super().reset()
        self.current_child = 0
        for child in self.children:
            child.reset()

    def terminate(self) -> None:
        if 0 <= self.current_child < len(self.children):
            self.children[self.current_child].terminate()
        self.current_child = 0

class Selector(_Composite):
    def execute(self) -> Status:
        if not self.children:
            return Status.FAILURE
        for index in range(self.current_child, len(self.children)):
            self.current_child = index
            status = self.children[index].tick()
            if status == Status.SUCCESS:
                self.children[index].reset()
                self.current_child = 0
                return Status.SUCCESS
            if status in (Status.RUNNING, Status.PAUSED):
                return status
        if 0 <= self.current_child < len(self.children):
            self.children[self.current_child].reset()
        self.current_child = 0
        return Status.FAILURE

class Sequence(_Composite):
    def execute(self) -> Status:
        if not self.children:
            return Status.FAILURE
        for index in range(self.current_child, len(self.children)):
            self.current_child = index
            status = self.children[index].tick()
            if status == Status.FAILURE:
                self.children[index].reset()
                self.current_child = 0
                return Status.FAILURE
            if status in (Status.RUNNING, Status.PAUSED):
                return status
        if 0 <= self.current_child < len(self.children):
            self.children[self.current_child].reset()
        self.current_child = 0
        return Status.SUCCESS

class Parallel(BehaviorNode):
    def __init__(
        self,
        name: str = "Parallel",
        children: Optional[List[BehaviorNode]] = None,
        success_threshold: int = 1,
        failure_threshold: Optional[int] = None,
    ):
        super().__init__(name)
        self.children = list(children or [])
        self.success_threshold = success_threshold
        self.failure_threshold = failure_threshold or len(self.children)

    def execute(self) -> Status:
        status_counts = {status: 0 for status in Status}
        for child in self.children:
            try:
                child_status = child.tick()
                status_counts[child_status] += 1
            except Exception as error:
                main_logger.Error(f"{self.name}: {error}")
                status_counts[Status.FAILURE] += 1
        if status_counts[Status.SUCCESS] >= self.success_threshold:
            for child in self.children:
                child.reset()
            self.terminate()
            return Status.SUCCESS
        if status_counts[Status.FAILURE] >= self.failure_threshold:
            for child in self.children:
                child.reset()
            self.terminate()
            return Status.FAILURE
        return Status.PAUSED if status_counts[Status.PAUSED] else Status.RUNNING

    def reset(self) -> None:
        super().reset()
        for child in self.children:
            child.reset()

    def terminate(self) -> None:
        for child in self.children:
            child.terminate()

################################################################################
# ================================ Rate limiters ===============================
################################################################################
class TimeBasedRateLimiter:
    def __init__(self, interval_seconds: float = 5.0):
        self.interval = interval_seconds
        self.last_execution: Optional[float] = None

    def ShouldExecute(self, current_time: Optional[float] = None) -> bool:
        now = current_time if current_time is not None else time.time()
        if self.last_execution is None or (now - self.last_execution) >= self.interval:
            self.last_execution = now
            return True
        return False

    def Reset(self) -> None:
        self.last_execution = None

class CountBasedRateLimiter:
    def __init__(self, every_n_calls: int = 100):
        self.interval = max(1, int(every_n_calls))
        self.counter = 0

    def ShouldExecute(self) -> bool:
        self.counter += 1
        if self.counter >= self.interval:
            self.counter = 0
            return True
        return False
    
    def Reset(self) -> None:
        self.counter = 0

def EveryNCalls(n: int = 100) -> CountBasedRateLimiter:
    return CountBasedRateLimiter(n)

def EveryNSeconds(seconds: float = 5.0) -> TimeBasedRateLimiter:
    return TimeBasedRateLimiter(seconds)