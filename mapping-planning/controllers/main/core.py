from __future__ import annotations                                                # Used in webots and older versions of python; future annotations keep hints as strings
import os                                                                         # For device stuff; env vars, file ops
import time                                                                       # Times stamps etc; timing/rate control
import traceback                                                                  # Better looking print statements; stack traces
from enum import Enum                             
from pathlib import Path                                                          # For file paths; safer path handling
from dataclasses import dataclass                                                 # auto __init__
from typing import Any, Dict, List, Optional, Tuple, Callable                     # Helpers.  # type hints
import numpy as np                                                                # For math and arrays. 

################################################################################
# =============================== Math utilities ===============================
################################################################################
# small, pure helpers used across mapping/planning
# keep fast and dependency free for tight loops
def NormalizeAngle(angle: float) -> float:                                        # Normalize angle to -pi, pi to avoid wrap around issues. 
    return (angle + np.pi) % (2 * np.pi) - np.pi  

def Distance2D(point1, point2) -> float:                                          # Euclidean distance between two points.  # (x,y) to scalar
    return np.hypot(point1[0] - point2[0], point1[1] - point2[1])  


##################################################################################
# ==================== World to Grid coordinate transforms =======================
##################################################################################
# maps world meters ↔ grid indices for drawing/planning
# tuned scale/offsets match environment calibration
def WorldToGridRaw(world_x: float, world_y: float) -> Tuple[int, int]:              # core transform
    row = int(40.0 * (world_x + 2.25))                                              # Meters to grid rows; scale+offset
    col = int(-52.9 * (world_y - 1.6633))                                           # Meters to grid cols; scale+offset+flip
    return row, col                                                                 # Return unclamped

def WorldToGrid(
    world_x: float,                                                                 # World X coordinate; meters
    world_y: float,                                                                 # world Y coordinate; meters
    grid_shape: Tuple[int, int] = (200, 300),                                       # target grid size; rows, cols
    clamp: bool = True,                                                             # Clamp out of bounds to edges; safer drawing
):                                                                                   
    row, col = WorldToGridRaw(world_x, world_y)                                     # Convert using raw mapping; integer indices
    if clamp:
        clamped_row = int(np.clip(row, 0, grid_shape[0] - 1))                       # Row clamp
        clamped_col = int(np.clip(col, 0, grid_shape[1] - 1))                       # Col clamp
        return clamped_row, clamped_col                                             # Always in bounds
    return (row, col) if (0 <= row < grid_shape[0] and 0 <= col < grid_shape[1]) else None  # None if out of bounds

def GridToWorld(row: int, col: int) -> Tuple[float, float]:                         
    world_x = row / 40.0 - 2.25                                                     # Map grid row to world X. 
    world_y = -col / 52.9 + 1.6633                                                  # Map grid col to world Y. 
    return world_x, world_y                                                         # Return world coordinates; meters

def BresenhamLine(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:     # Int grid endpoints; list of cells
    points: List[Tuple[int, int]] = []                                              # Output list grid points along the line; will include endpoints
    delta_x, delta_y = abs(x1 - x0), abs(y1 - y0)                                   # Deltas along x and y between endpoints; none negative
    current_x, current_y = x0, y0                                                   # Initialize current position at the starting point  
    step_x = 1 if x0 < x1 else -1                                                   # Step direction in x  
    step_y = 1 if y0 < y1 else -1                                                   # Step direction in y  
    error = (delta_x - delta_y) if delta_x > delta_y else (delta_y - delta_x)       # Initial error term; avoids floats
    for _ in range(max(delta_x, delta_y) + 1):                                      # Iterate once per pixel along the longer axis  
        points.append((current_x, current_y))                                       # Record the current grid point on the line; append cell
        error_doubled = error * 2                                                   # Double the error to avoid floating point 
        if error_doubled > -delta_y:                                                # If error is large enough, advance in x; x step check
            error -= delta_y                                                        # Reduce error by delta_y; maintain slope
            current_x += step_x                                                     # Step one pixel in x toward the target; move x
        if error_doubled < delta_x:                                                 # If error is small enough, advance in y; y step check
            error += delta_x                                                        # Increase error by delta_x; maintain slope
            current_y += step_y                                                     # Step one pixel in y toward the target; move y
    return points                                                                   # Return; list of x and y

def UpdateTrajectory(                             
    trajectory: Optional[List[Tuple[float, float]]],
    point: Tuple[float, float],
    min_distance: float = 0.1,
    max_points: int = 200,
) -> List[Tuple[float, float]]:
    trajectory = trajectory or []                                                    # Start with empty list if None provided; normalize
    if not trajectory or Distance2D(trajectory[-1], point) > min_distance:
        trajectory.append(point)                                                     # Add current x and y position 
    return trajectory[-max_points:]                                                  # Keep only the most recent max_points samples; trim


################################################################################
# =============================== Path for files ===============================
################################################################################
# Safe helpers for map file placement
# Auto create directories when needed
def ResolveMapPath(filename: str) -> Path:                                           # Maps absolute path
    if ("/" in filename) or ("\\" in filename):                                      # Disallow path separators for safety
        raise ValueError(f"Error filename: {filename}")                              # Guard against path traversal; hard fail
    maps_dir = Path(__file__).parent / "maps"                                        # Maps directory next to this file; folder
    maps_dir.mkdir(exist_ok=True)                                                    # Ensure the directory exists; create if missing
    return (maps_dir / filename).resolve()                                           # Return absolute path to the file

def EnsureParentDirectories(path: str | Path) -> Path:                               # ensure parents exist
    path_obj = Path(path)                                                            # Normalize to Path object
    path_obj.parent.mkdir(parents=True, exist_ok=True)                               # Create all parent directories  # mkdir -p
    return path_obj                                                                  # Return the normalized Path


################################################################################
# ================================ Type aliases ================================
################################################################################
# Improves readability for maps and paths
# No runtime impact; purely type-level
Position2D = Tuple[float, float]                                                     # X and Y in world coordinates; meters
PathType = List[Position2D]                                                          # Sequence of 2D positions
MapArray = np.ndarray                                                                # Alias for numpy map arrays
TH_FREE_PLANNER = 0.1                                                                # Threshold used by planner for free cells


################################################################################
# ============================= Parameter containers ===========================
################################################################################
# Editable points for mapping/planning
# dataclasses keep defaults clean and explicit
@dataclass
class MappingParams:                                                                # Params used by mapping and cspace pipeline
    th_occupied: float = 0.60                                                       # Prob for obstacle 
    th_free_planner: float = TH_FREE_PLANNER                                        # Prob free for planning
    th_free_explore: float = 0.15                                                   # Slightly looser free threshold for exploration  
    robot_radius: float = 0.15                                                      # Robot radius in meters
    safety_margin: float = 0.05                                                     # Extra margin added to radius 
    map_resolution_m: float = 0.02                                                  # Map cell size; meters/cell
    default_map_shape: Tuple[int, int] = (200, 300)                                 # Fallback grid size; rows and cols
    cspace_inflation_scale: float = 1.0                                             # Inflation scale for c-space obstacles  
    cspace_core_obstacle_value: float = 1.0                                         # Value assigned to hard obstacles in c-space
    cspace_morph_closing: int = 0                                                   
    cspace_morph_iters: int = 0                                                     # Number of iterations
    cspace_downsample: int = 1                                                      # Downsampling factor applied to c-space  
    mapping_interval: int = 6                                                       # Build or refresh c-space every N frames 
    lidar_offset_x: float = 0.202                                                   # Lidar sensor x-offset from robot base 
    lidar_update_interval: int = 2                                                  # Only use every Nth Lidar frame  
    world_to_grid: Optional[Callable] = None                                        # Override W2G  
    grid_to_world: Optional[Callable] = None                                        # Override G2W  

@dataclass
class PlanningParams:                                                               # Params for planner behavior and perf
    th_free_planner: float = TH_FREE_PLANNER                                        # Free space threshold for planning  
    max_iterations: int = 25000                                                     # Max A*/JPS expansions
    heuristic_weight: float = 1.0                                                   # Weight on heuristic
    check_neighbor_safety: bool = False                                             # Validate neighbor transitions for safety; extra check
    path_validation_enabled: bool = False                                           # Post validate path against c-space  
    default_map_shape: Tuple[int, int] = (200, 300)                                 # Fallback grid dimensions; rows, cols
    verbose: bool = False                                                         
    max_open_set_size: int = 50000                                                  # Priority queue cap 
    sqrt_2: float = 1.414213562373095                                               # sqrt(2)  
    jump_point_search: bool = True                                                  # Enable JPS acceleration; speed up
    bidirectional: bool = True                                                      # Enable bidirectional search  
    early_exit_multiplier: float = 1.5                                              # Exit early when heuristic close to goal  
    adaptive_max_iterations: int = 50000                                            # Raised limits if needed; adapt mode
    adaptive_max_open_set_size: int = 100000                                        # Raised open set limit; adapt mode
    adaptive_heuristic_weight: float = 1.5                                          # Heavier heuristic when adapting 
    safe_waypoint_search_radius: int = 12                                           # Radius to sample safe waypoints  # cells
    optimize_for_differential_drive: bool = True                                    # Smooth/align path for diff-drive robots
    differential_drive_alignment_tolerance: float = 0.5                             # Alignment tolerance; meters
    differential_drive_angle_tolerance: float = 15.0                                # Heading tolerance; degrees
    W2G: Optional[Callable] = None                                                  # World to grid mapper to inject at runtime  
    G2W: Optional[Callable] = None                                                  # Grid to world mapper to inject at runtime 


################################################################################
# ================================== Logging ===================================
################################################################################
# Tiny logger with module-level overrides
# Levels can be set globally or per-subsystem
class LogLevel:                                                                     # Numeric levels for speed
    ERROR, WARNING, INFO, DEBUG, VERBOSE = range(5)           

LOG_LEVEL_MAP = {
    "ERROR": 0,                                                                     # Text name to numeric level  
    "WARNING": 1,
    "INFO": 2,
    "DEBUG": 3,
    "VERBOSE": 4,
}

LOG_LEVEL = LOG_LEVEL_MAP.get(                                                      # Default level from environment variables, else INFO for default
    os.environ.get("ROBOT_LOG_LEVEL", "INFO").upper(),
    LogLevel.INFO,
)


class Logger:                                                                       # Per-module logger
    def __init__(self, name: str = "ROBOT", level: Optional[int] = None):
        self.name = name                                                            # Label shown in log lines 
        self.level = level                                                          # optional override 

    def IsLevelEnabled(self, level: int) -> bool:
        current_level = self.level if self.level is not None else LOG_LEVEL         # Pick local or global level
        return current_level >= level                                               # Log only if allowed by current level  

    def Log(self, level: int, tag: str, message: Any) -> None:
        if self.IsLevelEnabled(level):                                              # Skip if below threshold  
            print(f"[{tag}] {self.name}: {message}")                                # Basic console output 

    def Error(self, message: Any) -> None:
        self.Log(LogLevel.ERROR, "ERROR", message)                                  # ERROR: always important 

    def Warning(self, message: Any) -> None:
        self.Log(LogLevel.WARNING, "WARNING", message)                              # WARNING: something might be wrong 

    def Info(self, message: Any) -> None:
        self.Log(LogLevel.INFO, "INFO", message)                                    # INFO: normal status; heartbeat

    def Debug(self, message: Any) -> None:
        self.Log(LogLevel.DEBUG, "DEBUG", message)                                  # DEBUG: extra details  

nav_logger = Logger("NAV")                                                          # Navigation logs  
map_logger = Logger("MAP")                                                          # Mapping logs   
plan_logger = Logger("PLAN")                                                        # Planning logs  
main_logger = Logger("MAIN")                                                        # General logs   

def SetLogLevel(level: int | str, module: Optional[str] = None) -> None:           
    level_value = LOG_LEVEL_MAP[level.upper()] if isinstance(level, str) else int(level)  # Accept name or number 
    global LOG_LEVEL
    if module is None:                                                              # No module: set global default; broadcast
        LOG_LEVEL = level_value
        return
    loggers = {                                                                     # Pick a specific logger
        "nav": nav_logger,
        "map": map_logger,
        "plan": plan_logger,
        "main": main_logger,
    }
    if logger := loggers.get(module.lower()):
        logger.level = level_value                                                  # Set only that logger's level; local override
    else:
        main_logger.Warning(f"Unknown '{module}'; valid: {list(loggers.keys())}")   # Bad key

################################################################################
# ================================ Blackboard keys =============================
################################################################################
# Keys for BB; avoid magic strings
# Grouped by subsystem for clarity
class BBKey(str, Enum):                                                             # Enum of string keys used to access data on the blackboard 
    # === ROBOT HARDWARE ===
    ROBOT = "robot"                                                                 # Robot controller; webots
    GPS = "gps"                                                                     # GPS sensor; position
    COMPASS = "compass"                                                             # Compass sensor; heading
    LIDAR = "lidar"                                                                 # Lidar sensor; range
    MOTOR_L = "motorL"                                                              # Left motor handle 
    MOTOR_R = "motorR"                                                              # Right motor handle 
    DISPLAY = "display"                                                             # Display device / UI 
    TIMESTEP = "timestep"                                                           # Simulation or control timestep; ms
    INIT_Z = "init_z"                                                               # Initial Z height/offset; meters
    
    # === MAPPING DATA ===
    PROB_MAP = "prob_map"                                                             # Probability map 
    CSPACE = "cspace"                                                                 # Cspace grid 
    MAP_SAVED = "map_saved"                                                           # Indicating map 
    MAP_READY = "map_ready"                                                           # Indicating map is ready/complete 
    CSPACE_FROZEN = "cspace_frozen"                                                   # Indicating cspace should not update
    
    # === NAVIGATION DATA ===
    START_XY = "start_xy"                                                             # Start position x and y; meters
    PLANNED_PATH = "planned_path"                                                     # Planned path data structure
    NAVIGATION_GOALS = "navigation_goals"                                             # List of navigation waypoints
    NAVIGATION_GOAL = "navigation_goal"                                               # Current navigation goal
    TRAJECTORY_POINTS = "trajectory_points"                                           # Executed trajectory points
    
    # === DISPLAY CONTROL ===
    DISPLAY_MODE = "display_mode"                                                     # UI display mode  # 'full'|'cspace'|'planning'
    ALLOW_CSPACE_DISPLAY = "allow_cspace_display"                                     # Permission to show cspace overlay
    
    # === TRANSFORM DATA ===
    WORLD_TO_GRID = "world_to_grid"                                                   # World to grid coordinate transform  
    GRID_TO_WORLD = "grid_to_world"                                                   # Grid to world coordinate transform 


################################################################################
# ================================= Blackboard =================================
################################################################################
# Lightweight shared state store between nodes
# Wrap dict for convenience + typed accessors
class Blackboard:                                                                     # Simple key/value blackboard
    def __init__(self):
        self.data: Dict[str, Any] = {}                                                # Internal storage dictionary for all key/value pairs 

    @staticmethod
    def Key(key: BBKey | str) -> str: #
        return key.value if isinstance(key, BBKey) else key                           # Normalize enums to their string values

    def Set(self, key: BBKey | str, value: Any) -> None:
        self.data[self.Key(key)] = value                                              # Set/overwrite a value for a key

    def Get(self, key: BBKey | str, default: Any = None) -> Any:
        return self.data.get(self.Key(key), default)                                  # Get value for key with optional default

    def Has(self, key: BBKey | str) -> bool:
        return self.Key(key) in self.data                                             # True if key exists

    def Incr(self, key: BBKey | str, by: int = 1) -> int:
        key_str = self.Key(key)                                                       # Normalize key  # str
        new_value = (self.Get(key_str, 0) or 0) + by                                  # Increment current value 
        self.Set(key_str, new_value)                                                  # Store incremented value 
        return new_value                                                              # Return updated value

    def Remove(self, key: BBKey | str) -> None: 
        self.data.pop(self.Key(key), None)                                            # Remove key if present; ignore if missing

    def Clear(self) -> None:
        self.data.clear()                                                             # Remove all keys/values

    def AllowCspaceDisplay(self, value: bool | None = None) -> bool:
        if value is not None:                                                         # If a value is provided, set the flag; setter
            self.Set(BBKey.ALLOW_CSPACE_DISPLAY, bool(value))
        return bool(self.Get(BBKey.ALLOW_CSPACE_DISPLAY, False))                      # Return current flag; getter

    def ClearMissionData(self) -> None:
        for k in (BBKey.PLANNED_PATH, BBKey.NAVIGATION_GOALS, BBKey.NAVIGATION_GOAL):
            self.Set(k, None)                                                         # Clear related pointers
        self.Set(BBKey.TRAJECTORY_POINTS, [])                                         # Reset trajectory to empty list
        for k, v in [
            ("allow_cspace_display", False),
            ("map_saved", False),
            ("display_mode", "full"),
        ]:
            self.Set(k, v)                                                            # Reset several flags and display mode  

    def GetRobot(self): return self.Get(BBKey.ROBOT)                                  # Convenience getter for robot  
    def GetGps(self): return self.Get(BBKey.GPS)                                      # Get for GPS  
    def GetCompass(self): return self.Get(BBKey.COMPASS)                              # Get for compass  
    def GetLidar(self): return self.Get(BBKey.LIDAR)                                  # Get for lidar 
    def GetMotors(self): return self.Get(BBKey.MOTOR_L), self.Get(BBKey.MOTOR_R)      # Get for L/R motors 
    def GetDisplay(self): return self.Get(BBKey.DISPLAY)                              # Get for display 
    def GetProbMap(self) -> Optional(MapArray): return self.Get(BBKey.PROB_MAP)       # Get for probability map
    def GetCspace(self) -> Optional(MapArray): return self.Get(BBKey.CSPACE)          # Get for cspace
    def GetPlannedPath(self) -> Optional(PathType): return self.Get(BBKey.PLANNED_PATH)  # Get for planned path 
    def GetTrajectory(self) -> Optional(PathType): return self.Get(BBKey.TRAJECTORY_POINTS)  # Get for trajectory 
    def GetNavigationGoals(self) -> Optional[List[Position2D]]: return self.Get(BBKey.NAVIGATION_GOALS)  # Get for goals

    def SetProbMap(self, probability_map: MapArray):
        self.Set(BBKey.PROB_MAP, probability_map)                                     # Set for prob map

    def SetCspace(self, configuration_space: MapArray):
        self.Set(BBKey.CSPACE, configuration_space)                                   # Setter for cspace

    def SetPlannedPath(self, path: PathType):
        self.Set(BBKey.PLANNED_PATH, path)                                            # Set for planned path 

    def SetTrajectory(self, trajectory: PathType):
        self.Set(BBKey.TRAJECTORY_POINTS, trajectory)                                 # Set for trajectory points

    def SetNavigationGoals(self, goals: List[Position2D]):
        self.Set(BBKey.NAVIGATION_GOALS, goals)                                       # Set for navigation goals

    def SetMapReady(self, ready: bool = True):
        self.Set(BBKey.MAP_READY, ready)                                              # Mark map readiness state

    def SetMapSaved(self, saved: bool = True):
        self.Set(BBKey.MAP_SAVED, saved)                                              # Mark map saved state

blackboard = Blackboard()                                                             # Global blackboard instance

def GetFromBlackboard(key, default=None):                                             # Fetch a value from a global blackboard store
    return blackboard.Get(key, default)

def CreateBlackboard() -> Blackboard:
    return Blackboard()                                                               # Fresh blackboard

def SetGlobalBlackboard(new_blackboard: Blackboard) -> None:
    global blackboard                                                                 # Use module level global
    blackboard = new_blackboard                                                       # Replace global blackboard reference


################################################################################
# ================================== BT core ===================================
################################################################################
# Minimal behavior tree base node
# Handles tick lifecycle, pause/halt, and errors
class Status(Enum):                                                                   # Execution states for behavior tree nodes
    SUCCESS = "SUCCESS"                                                               # Successfully
    FAILURE = "FAILURE"                                                               # Unsuccessfully
    RUNNING = "RUNNING"                                                               # Still in progress  
    PAUSED = "PAUSED"                                                                 # Paused 


class BehaviorNode:                                                                   # Base class for all BT nodes
    def __init__(self, name: str = "Behavior"):
        self.name = name                                                              # Node label
        self.status = Status.FAILURE                                                  # Last status 
        self.tick_count = 0                                                           # Number of times tick was called
        self.last_tick_time: Optional[float] = None                                   # Timestamp of last tick; seconds
        self.is_paused = False                                                        # Pause 
        self.is_halted = False                                                        # Halt 

    def execute(self) -> Status:                                                      # Override in subclasses 
        return Status.FAILURE                                                         # Default behavior

    def tick(self) -> Status:
        if self.is_halted:                                                            # If halted, report failure without executing; hard stop
            return Status.FAILURE
        if self.is_paused:                                                            # If paused, report pause without executing  
            return Status.PAUSED
        self.tick_count += 1                                                          # Increment tick counter; +1
        self.last_tick_time = time.time()                                             # Record tick time  
        try:
            result = self.execute()                                                   # Execute node specific logic 
            self.status = result if isinstance(result, Status) else Status.FAILURE    # Type 
        except Exception as error:                                                    # Catch runtime exceptions 
            self.status = Status.FAILURE                                              # Mark failure on exception
            main_logger.Error(f"{self.name} (#{self.tick_count}): {type(error).__name__} - {error}")  # Log error
            traceback.print_exc()                                                     
        return self.status                                                            # Return current status

    def reset(self) -> None:
        self.status = Status.FAILURE                                                  # Reset status to default
        self.tick_count = 0                                                           # Reset tick counter
        self.is_paused = False                                                        # Clear pause state

    def terminate(self) -> None:
        pass                                                                          # Cleanup

    def pause(self) -> None:
        self.is_paused = True                                                         # Set pause flag
        main_logger.Debug(f"Paused: {self.name}")                                     # Log pause

    def resume(self) -> None:
        self.is_paused = False                                                        # Clear pause flag
        main_logger.Debug(f"Resumed: {self.name}")                                    # Log resume

    def halt(self) -> None:
        self.is_halted = True                                                         # Stop permanently
        self.is_paused = False                                                        # Ensure not paused 
        main_logger.Debug(f"Halted: {self.name}")                                     # Log halt

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
# Selector/sequence/parallel building blocks
# Manage child node execution policies
class _Composite(BehaviorNode):                                                         # abstract composite w/ child list
    def __init__(self, name: str, children: Optional[List[BehaviorNode]] = None):
        super().__init__(name)                                                          # Initialize base node state
        self.children = list(children or [])                                            # Store child nodes
        self.current_child = 0                                                          # Index of currently active child

    def reset(self) -> None:
        super().reset()                                                                 # Reset own state
        self.current_child = 0                                                          # Reset child pointer
        for child in self.children:
            child.reset()                                                               # Reset all children

    def terminate(self) -> None:
        if 0 <= self.current_child < len(self.children):
            self.children[self.current_child].terminate()                               # Terminate active child if any
        self.current_child = 0                                                          # Reset child pointer


class Selector(_Composite):                                                             # Try children until one succeeds
    def execute(self) -> Status:
        if not self.children:
            return Status.FAILURE                                                       # No children means failure; nothing to do
        for index in range(self.current_child, len(self.children)):
            self.current_child = index                                                  # Update current child index  
            status = self.children[index].tick()                                        # Tick child
            if status == Status.SUCCESS:
                self.children[index].reset()                                            # Reset child on success  
                self.current_child = 0                                                  # Prepare for next cycle
                return Status.SUCCESS                                                   # Return success 
            if status in (Status.RUNNING, Status.PAUSED):
                return status                                                           # Running/paused  
        if 0 <= self.current_child < len(self.children):
            self.children[self.current_child].reset()                                   # Reset last attempted child; clean
        self.current_child = 0                                                          # Reset pointer; rewind
        return Status.FAILURE                                                           # All children failed 


class Sequence(_Composite):                                                             # Run children until one fails
    def execute(self) -> Status:
        if not self.children:
            return Status.FAILURE                                                       # No children means failure; empty
        for index in range(self.current_child, len(self.children)):
            self.current_child = index                                                  # Update current child index
            status = self.children[index].tick()                                        # Tick child; run
            if status == Status.FAILURE:
                self.children[index].reset()                                            # Reset failing child; clean
                self.current_child = 0                                                  # Reset pointer;  rewind
                return Status.FAILURE                                                   # Early exit on failure; stop
            if status in (Status.RUNNING, Status.PAUSED):
                return status                                                           # Active or paused 
        if 0 <= self.current_child < len(self.children):
            self.children[self.current_child].reset()                                   # Reset last child after success; clean
        self.current_child = 0                                                          # Reset pointer; rewind
        return Status.SUCCESS                                                           # All children succeeded; done


class Parallel(BehaviorNode):                                                           # Run children in parallel with thresholds
    def __init__(
        self,
        name: str = "Parallel",
        children: Optional[List[BehaviorNode]] = None,
        success_threshold: int = 1,
        failure_threshold: Optional[int] = None,
    ):
        super().__init__(name)                                                          # Initialize base node; label
        self.children = list(children or [])                                            # Store children; list copy
        self.success_threshold = success_threshold                                      # Number of child successes to succeed overall
        self.failure_threshold = failure_threshold or len(self.children)                # Fail after this many failures 

    def execute(self) -> Status:
        status_counts = {status: 0 for status in Status}                                # Counters for each status; tally
        for child in self.children:
            try:
                child_status = child.tick()                                             # Tick child node; run
                status_counts[child_status] += 1                                        # Increment counter for returned status; count
            except Exception as error:
                main_logger.Error(f"{self.name}: {error}")                              
                status_counts[Status.FAILURE] += 1                                      
        if status_counts[Status.SUCCESS] >= self.success_threshold:
            for child in self.children:
                child.reset()                                                           # Reset all children on success; clean
            self.terminate()                                                            # Terminate parallel node session
            return Status.SUCCESS                                                       # Overall success
        if status_counts[Status.FAILURE] >= self.failure_threshold:
            for child in self.children:
                child.reset()                                                           # Reset all children on failure
            self.terminate()                                                            # Terminate parallel session
            return Status.FAILURE                                                       # Overall failure
        return Status.PAUSED if status_counts[Status.PAUSED] else Status.RUNNING       

    def reset(self) -> None:
        super().reset()                                                                 # Reset self
        for child in self.children:
            child.reset()                                                               # Reset all children

    def terminate(self) -> None:
        for child in self.children:
            child.terminate()                                                           # Terminate all children

################################################################################
# ================================ Rate limiters ===============================
################################################################################
# utilities to throttle expensive work
# choose time-based or call-count based
class TimeBasedRateLimiter:                                                             # Allow every N seconds
    def __init__(self, interval_seconds: float = 5.0):
        self.interval = interval_seconds                                                # Minimum seconds between allowed executions; spacing
        self.last_execution: Optional[float] = None                                     # Timestamp of last allowed execution

    def ShouldExecute(self, current_time: Optional[float] = None) -> bool:
        now = current_time if current_time is not None else time.time()                 # Use provided time or current time
        if self.last_execution is None or (now - self.last_execution) >= self.interval:
            self.last_execution = now                                                   # Update last execution time 
            return True                                                                 # Allowed to execute 
        return False                                                                    # Not yet; wait

    def Reset(self) -> None:
        self.last_execution = None                                                      # Clear execution history


class CountBasedRateLimiter:                                                            # Allow every Nth call
    def __init__(self, every_n_calls: int = 100):
        self.interval = max(1, int(every_n_calls))                                      # Execute every N 
        self.counter = 0                                                                # Internal call counter; starts at 0

    def ShouldExecute(self) -> bool:
        self.counter += 1                                                               # Increment call count; +1
        if self.counter >= self.interval:                                               # If reached threshold; check
            self.counter = 0                                                            # Reset counter
            return True                                                                 # Allow execution
        return False                                                                    # Not yet; skip

    def Reset(self) -> None:
        self.counter = 0                                                                # Reset counter to zero; clear

def EveryNCalls(n: int = 100) -> CountBasedRateLimiter:
    return CountBasedRateLimiter(n)                                                     # Helper to build a count based limiter

def EveryNSeconds(seconds: float = 5.0) -> TimeBasedRateLimiter:
    return TimeBasedRateLimiter(seconds)                                                # Helper to build a time based limiter


################################################################################
# ============================ Display Manager =================================
################################################################################
# Draws maps, paths, and markers to the simulated display
# Keep calls light and rate-limited for performance
def MapToDisplayCoords(row: int, col: int, map_shape: tuple, w: int, h: int) -> tuple[int, int]:
    # map grid cell to display pixel
    return int(row * w / map_shape[0]), int(col * h / map_shape[1])                     # Simple scale


class DisplayManager:                                                                   # Rendering helper for the Webots display
    COLOR_BLACK = 0x000000
    COLOR_WHITE = 0xFFFFFF
    COLOR_BLUE  = 0x0000FF
    COLOR_RED   = 0xFF0000
    COLOR_GREEN = 0x00FF00
    COLOR_YELLOW= 0xFFFF00
    COLOR_CYAN  = 0x00FFFF
    COLOR_HOT_PINK = 0xFF69B4

    PROBMAP_MIN_DRAW = 0.35                                                             # Low clamp for map draw; darkest
    PROBMAP_MAX_SHOWN = 0.75                                                            # High clamp for map draw; brightest
    PROBMAP_GAMMA = 0.85                                                                # Gamma to boost contrast; curve

    def __init__(self, display=None):                                                   
        self.display = display or blackboard.Get("display")                             # Get from bb if not passed
        self.width = self.display.getWidth() if self.display else 0                     # Cache width; px
        self.height = self.display.getHeight() if self.display else 0                   # Cache height; px

    def InitializeDisplay(self):                                                        # Ensure display ready
        if not self.display:
            self.display = blackboard.Get("display")                                    # Try again; fetch
            if not self.display:
                return False                                                            # Cannot draw without display; abort
        self.width  = self.display.getWidth()                                           # Refresh size in case it changed; update cache
        self.height = self.display.getHeight()
        self.ClearDisplay()                                                             # Start with clean screen; wipe
        return True

    def ClearDisplay(self):                                                             # Black background
        if self.display:
            self.display.setColor(self.COLOR_BLACK)                                     # Choose black; color set
            self.display.fillRectangle(0, 0, self.width, self.height)                   # Fill full area

    def GetMapShape(self) -> tuple[int, int]:                                           # Pick best known grid shape
        cspace = blackboard.Get("cspace")                                               # Prefer cspace for shape then prob map then default
        if cspace is not None:
            return cspace.shape
        prob_map = blackboard.Get("prob_map")
        if prob_map is not None:
            return prob_map.shape
        return 200, 300

    def MapToDisplay(self, row, col, shape):
        # convert grid to pixel coords
        return MapToDisplayCoords(row, col, shape, self.width, self.height)             

    def in_bounds(self, x, y, w=2, h=2):                                                # Does the pixel rect fits screen?
        # make sure we can draw inside screen
        return 0 <= x <= self.width - w and 0 <= y <= self.height - h                   # Inclusive bounds
    
    def DrawPixel(self, x, y, color):                                                   # 2x2 block pixel
        if self.in_bounds(x, y):
            self.display.setColor(color)
            self.display.fillRectangle(x, y, 2, 2)                                      # Draw a small square

    def DrawWorldLine(self, start_point, end_point, shape, color):                      # World coords line
        if not self.display:
            return
        sx, sy = self.MapToDisplay(*WorldToGrid(*start_point, shape), shape)            # Start pixel
        ex, ey = self.MapToDisplay(*WorldToGrid(*end_point, shape), shape)              # End pixel 
        self.display.setColor(color)
        self.display.drawLine(sx, sy, ex, ey)                                           # Draw line 

    def DrawProbabilityMap(self, probability_map=None):                                 # Grayscale map
        if not self.display:
            return
        probability_map = blackboard.Get("prob_map") if probability_map is None else probability_map
        if probability_map is None:
            return                                                                      # Nothing to draw 
        p = np.clip(probability_map, self.PROBMAP_MIN_DRAW, self.PROBMAP_MAX_SHOWN)     # Clamp; reduce range
        p = (p - self.PROBMAP_MIN_DRAW) / (self.PROBMAP_MAX_SHOWN - self.PROBMAP_MIN_DRAW)  # Scale 0 to 1; normalize
        p = p ** self.PROBMAP_GAMMA                                                     # Apply gamma; contrast
        h, w = p.shape                                                                  # Grid size  # dims
        step = 2                                                                        # Skip every other cell  # stride
        rows, cols = np.mgrid[0:h:step, 0:w:step]                                      
        intensity = (p[rows, cols] * 255).astype(np.uint8)                              # 0 to 255
        px = (rows * self.width  // h).astype(int)                                      # X pixels 
        py = (cols * self.height // w).astype(int)                                      # Y pixels
        bw = max(2, self.width  // w)                                                  
        bh = max(2, self.height // h)                                                  
        valid = (px >= 0) & (px <= self.width - bw) & (py >= 0) & (py <= self.height - bh)  # Inside screen 
        vr, vc = np.where(valid)                                                        # Indices that fit; coords
        if len(vr) > 10000:                                                             # Trim if too many blocks
            s = max(1, len(vr) // 10000)
            vr, vc = vr[::s], vc[::s]
        for i, j in zip(vr, vc):
            g = int(intensity[i, j])                                                    # Gray value
            rgb = (g << 16) | (g << 8) | g                                              # Grayscale
            self.display.setColor(rgb)
            self.display.fillRectangle(px[i, j], py[i, j], bw, bh)                      

    def DrawTrajectory(self, trajectory_points=None, map_shape=None, color=None):       # Draw path taken
        if not self.display:
            return
        traj = trajectory_points or blackboard.Get("trajectory_points")                 # List of world points
        if not traj or len(traj) < 2:
            return
        shape = map_shape or self.GetMapShape()
        col = color or self.COLOR_HOT_PINK
        for a, b in zip(traj, traj[1:]):                                                # Draw line between pairs
            self.DrawWorldLine(a, b, shape, col)

    def DrawRobotPosition(self, map_shape=None, color=None, size=4):                    # Blue dot for robot
        if not self.display:
            return
        gps = blackboard.Get("gps")                      
        if not gps:
            return
        wx, wy = gps.getValues()[:2]                                                    # Current world position; x and y
        shape = map_shape or self.GetMapShape()
        x, y = self.MapToDisplay(*WorldToGrid(wx, wy, shape), shape)                    # Convert to pixel
        if 0 <= x < self.width and 0 <= y < self.height:
            self.display.setColor(color or self.COLOR_BLUE)
            hs = size // 2
            self.display.fillOval(x - hs, y - hs, size, size)                           

    def DrawPlannedPath(self, path=None, map_shape=None, color=None):                   
        if not self.display:
            return
        planned = path or blackboard.Get("planned_path")
        if not planned:
            return
        shape = map_shape or self.GetMapShape()
        col = color or self.COLOR_CYAN
        for a, b in zip(planned, planned[1:]):                                          # Draw each segment
            self.DrawWorldLine(a, b, shape, col)
        self.display.setColor(col)                                     
        for wx, wy in planned:                                                        
            x, y = self.MapToDisplay(*WorldToGrid(wx, wy, shape), shape)
            if self.in_bounds(x, y, 2, 2):
                self.display.fillOval(x - 1, y - 1, 2, 2)                               

    def DrawNavigationGoals(self, goals=None, map_shape=None, color=None):              
        if not self.display:
            return
        nav_goals = goals or blackboard.Get("navigation_goals")
        if not nav_goals:
            return
        shape = map_shape or self.GetMapShape()
        self.display.setColor(color or self.COLOR_YELLOW)              
        for gx, gy in nav_goals:
            x, y = self.MapToDisplay(*WorldToGrid(gx, gy, shape), shape)
            if self.in_bounds(x, y, 3, 3):
                s = 3
                self.display.drawLine(x - s, y - s, x + s, y + s)                       
                self.display.drawLine(x - s, y + s, x + s, y - s)                       

    def DrawCspace(self, cspace):                                                       # Black/white cspace preview
        if not (self.display and cspace is not None):
            return
        h, w = cspace.shape
        for r in range(0, h, 2):                                                        # Skip rows for speed
            row = cspace[r]
            for c in range(0, w, 2):                                                    # Skip cols for speed
                color = self.COLOR_BLACK if float(row[c]) > 0.5 else self.COLOR_WHITE   # Threshold
                x, y = self.MapToDisplay(r, c, (h, w))
                self.DrawPixel(x, y, color)                                             # Draw pixel

    def UpdateDisplay(self, mode="full"):                                               # Render one frame based on mode
        if not self.display:
            return
        self.ClearDisplay()                                                             # Fresh frame
        shape = self.GetMapShape()
        if mode in ("cspace", "planning"):                                                          
            cspace = blackboard.Get("cspace")
            if cspace is not None:
                self.DrawCspace(cspace)                                                 # Show free and blocked
            if mode == "planning":
                self.DrawPlannedPath(map_shape=shape)                                   # Show path; cyan
                self.DrawNavigationGoals(map_shape=shape)                               # Show goals; yellow
                self.DrawRobotPosition(map_shape=shape)                                 # Show robot marker; blue
            return                                                                      # Done for these modes
        prob_map = blackboard.Get("prob_map")
        if prob_map is not None:
            self.DrawProbabilityMap(prob_map)                                           
        self.DrawTrajectory(map_shape=shape)                                            # Trail
        self.DrawRobotPosition(map_shape=shape)                                         # Robot dot; blue

display_manager = None 

def GetDisplayManager():                                                                
    # singleton maker for display manager
    global display_manager
    if display_manager is None:                                       
        display_manager = DisplayManager()                                              # Create on first call; init
    return display_manager                                                              # Return cached


################################################################################
# ========================== Display Behavior Tree Nodes ======================
################################################################################
# BT nodes that update the UI display
# Rate limiters to keep everything smooth
class DisplayUpdater(BehaviorNode):                                                     # Ticks renderer
    def __init__(self):
        super().__init__("DisplayUpdater")
        self.manager = GetDisplayManager()                                              # Renderer helper
        self.fps = EveryNSeconds(0.25)                                                  

    @staticmethod
    def truthy(v):                                                                      # "Is non-empty?" check
        try:
            if isinstance(v, np.ndarray):
                return bool(np.any(v))                                                  
        except Exception:
            pass
        try:
            if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):  
                return len(v) > 0                                                       
        except Exception:
            pass
        return bool(v)                                                                  

    def execute(self):
        try:
            if not (self.manager and self.manager.display):                             # Display ready
                return Status.RUNNING
            robot = blackboard.Get("robot")                                             # For time source  
            t = robot.getTime() if robot else None
            if t is None or not self.fps.ShouldExecute(t):                              # Rate cap 
                return Status.RUNNING
            display_mode = blackboard.Get("display_mode", "full")                       # Current mode  
            allow_cspace = self.truthy(blackboard.Get("allow_cspace_display", False))  
            survey_done  = self.truthy(blackboard.Get("survey_complete", False))
            cspace = blackboard.Get("cspace")                                           # Current cspace grid  
            has_cspace = (cspace is not None) and (CalculateFreeSpacePercentage(cspace) >= 0.01) 
            if display_mode == "full" and allow_cspace and survey_done and has_cspace:  # Auto switch to cspace mode when allowed and ready
                blackboard.Set(BBKey.DISPLAY_MODE, "cspace")                            # Switch mode; auto
                display_mode = "cspace"
            self.manager.UpdateDisplay(mode=display_mode)                               # Draw frame; render
        except Exception as e:
            main_logger.Error(f"DisplayUpdater error {e}")                              # Log then continue
        return Status.RUNNING                                                           # keep ticking  # non-blocking

class SetDisplayMode(BehaviorNode):                                                     # Simple setter node
    def __init__(self, mode):
        super().__init__(f"SetDisplayMode({mode})")
        self.mode = mode                                                                

    def execute(self):
        blackboard.Set(BBKey.DISPLAY_MODE, self.mode)                                   # Set in bb
        return Status.SUCCESS                                                           # Success


class EnableCspaceDisplay(BehaviorNode):                                                # Allow cspace overlay + mark survey done
    def __init__(self):
        super().__init__("EnableCspaceDisplay")

    def execute(self):
        blackboard.Set(BBKey.ALLOW_CSPACE_DISPLAY, True)                                # Permit cspace on ui
        blackboard.Set("survey_complete", True)                                         # Mark survey done
        return Status.SUCCESS                                                           # Done

def CalculateFreeSpacePercentage(cspace: np.ndarray) -> float:                          
    free_cells = float((cspace > TH_FREE_PLANNER).sum())                                # Count cells above free
    return free_cells / float(cspace.size)                                              # Divide by total
