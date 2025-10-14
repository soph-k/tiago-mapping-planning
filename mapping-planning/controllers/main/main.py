from __future__ import annotations
from controller import Supervisor
import numpy as np
from os.path import exists
from core import (
    MappingParams,
    PlanningParams,
    main_logger,
    SetLogLevel,
    LogLevel,
    NormalizeAngle,
    Distance2D,
    WorldToGrid,
    GridToWorld,
    UpdateTrajectory,
    ResolveMapPath,
    EnsureParentDirectories,
    BehaviorNode,
    Status,
    Sequence,
    Selector,
    Parallel,
    blackboard,
    EveryNSeconds,
    BBKey,
    TH_FREE_PLANNER,
)
from navigation import ObstacleAvoidingNavigation, SafeSetMotorVelocities, StopMotors
from mapping import LidarMappingBT
from planning import validate_path, visualize_path_on_map, find_safe_positions, MultiGoalPlannerBT


def GetFromBlackboard(key, default=None):
    return blackboard.Get(key, default)

def CalculateFreeSpacePercentage(cspace: np.ndarray) -> float:
    free_cells = float((cspace > TH_FREE_PLANNER).sum())
    return free_cells / float(cspace.size)

def MapToDisplayCoords(row: int, col: int, map_shape: tuple, w: int, h: int) -> tuple[int, int]:
    return int(row * w / map_shape[0]), int(col * h / map_shape[1])


###############################################################################
# ============================== Helpers ======================================
###############################################################################
def GetFromBlackboard(key, default=None):                               # Fetch a value from a global blackboard store
    return blackboard.Get(key, default)

def CalculateFreeSpacePercentage(cspace: np.ndarray) -> float:          # % of cells above free threshold
    free_cells = float((cspace > TH_FREE_PLANNER).sum())                # Count free cells
    return free_cells / float(cspace.size)                              # Normalize by total cells

def MapToDisplayCoords(row: int, col: int, map_shape: tuple, w: int, h: int) -> tuple[int, int]:
    return int(row * w / map_shape[0]), int(col * h / map_shape[1])     # Scale indices to display size


################################################################################
# ========================= Device Initialization ==============================
################################################################################
def GetDeviceNames(robot) -> set[str]:   # Collect device names from the robot
    try:
        return {robot.getDeviceByIndex(i).getName() for i in range(robot.getNumberOfDevices())}
    except Exception:
        return set()                                                    # If failed, return empty set

def InitRobot():                                                        
    robot = Supervisor()            
    timestep = int(robot.getBasicTimeStep())                            
    blackboard.Set(BBKey.INIT_Z, robot.getSelf().getPosition()[2])
    return robot, timestep

def InitSensors(robot, timestep):                                       # Initialize and enable sensors
    gps, compass = robot.getDevice('gps'), robot.getDevice('compass')   # Get sensor handles
    if gps:
        gps.enable(timestep)                                            # Turn on GPS at controller timestep
    if compass:
        compass.enable(timestep)                                        # Turn on compass at controller timestep
    available = GetDeviceNames(robot)                                   # Set of all device names on robot
    lidar = None
    for name in ("Hokuyo URG-04LX-UG01", "lidar", "laser", "Hokuyo", "urg04lx"):
        if name in available:                                           # If this name exists on the robot
            try:
                lidar = robot.getDevice(name)                           # Get the device handle
                lidar.enable(timestep)                                  # Enable Lidar updates
                blackboard.Set(BBKey.LIDAR, lidar)                      # Store Lidar on blackboard
                break                                                   # Stop after first working Lidar
            except Exception:                                           # Ignore and try next name
                pass
    if not lidar:
        main_logger.Warning("Lidar not detected")
    if not (gps and compass):
        main_logger.Error("Sensor missing")                             # Error if GPS or compass is missing
    return gps, compass, lidar                                          # Return sensor handles

def InitMotors(robot):                                                  # Initialize drive motors
    available = GetDeviceNames(robot)                                   # All device names
    pick = lambda *c: next((n for n in c if n in available), None)
    left_name = pick("wheel_left_joint", "wheel_left_motor", "left_wheel_joint", "left_wheel_motor")
    right_name = pick("wheel_right_joint", "wheel_right_motor", "right_wheel_joint", "right_wheel_motor")
    if not left_name or not right_name:
        if not left_name:
            main_logger.Error("Left motor not found.")                  # Log missing left motor
        if not right_name:
            main_logger.Error("Right motor not found.")                 # Log missing right motor
        return None, None
    left = robot.getDevice(left_name)                                   # Get left motor handle
    right = robot.getDevice(right_name)                                 # Get right motor handle
    try:
        for m in (left, right):                                           
            m.setPosition(float('inf'))                                 # Set velocity control 
            m.setVelocity(0.0)                                          # Start motors stopped
    except Exception:
        pass
    return left, right                                                  # Return motor handles

def InitDisplay(robot):                                                 # Initialize display device and manager  
    try:
        disp = robot.getDevice("display")                               # Fetch display device
        if not disp:
            main_logger.Warning("Display error, not correct dimension") # Warn if missing/wrong
            return None                                                 # No display
        blackboard.Set(BBKey.DISPLAY, disp)                             # Store display on blackboard
        GetDisplayManager().InitializeDisplay()                         # Initialize drawing surface
        return disp                                                     # Return display handle
    except Exception as e:
        main_logger.Error(f"Display init error: {e}")                   # Log any init error
        return None

def RegisterDevices(robot, timestep, gps, compass, lidar, left_motor, right_motor, display):
    for k, v in (
        (BBKey.ROBOT, robot),                                           # Robot supervisor
        (BBKey.GPS, gps),                                               # GPS handle
        (BBKey.COMPASS, compass),                                       # Compass handle
        (BBKey.LIDAR, lidar),                                           # Left motor
        (BBKey.MOTOR_L, left_motor),                                    # Right motor
        (BBKey.MOTOR_R, right_motor),                                   # Display
        (BBKey.DISPLAY, display),                                       # Controller timestep
        (BBKey.TIMESTEP, timestep),
        ("world_to_grid", WorldToGrid),
        ("grid_to_world", GridToWorld),
        ("normalize_angle", NormalizeAngle),
        ("distance_2d", Distance2D),
    ):
        blackboard.Set(k, v)                                            # Store each on blackboard
    missing = [n for n, d in (("GPS", gps), ("Compass", compass), ("LiDAR", lidar)) if not d]
    if missing:
        main_logger.Warning(f"Missing sensors.")                        # Warn if any missing

def InitAllDevices():
    robot, timestep = InitRobot()                                       # Create supervisor + timestep
    gps, compass, lidar = InitSensors(robot, timestep)                  # Sensors
    left_motor, right_motor = InitMotors(robot)                         # Motors
    display = InitDisplay(robot)                                        # Display
    RegisterDevices(robot, timestep, gps, compass, lidar, left_motor, right_motor, display)
    return robot, timestep, gps, compass, lidar, left_motor, right_motor, display   # Return all handles


################################################################################
# ============================ Display Manager =================================
################################################################################
class DisplayManager:                   
    COLOR_BLACK = 0x000000                                               # RGB hex colors
    COLOR_WHITE = 0xFFFFFF
    COLOR_BLUE = 0x0000FF
    COLOR_RED = 0xFF0000
    COLOR_GREEN = 0x00FF00
    COLOR_YELLOW = 0xFFFF00
    COLOR_CYAN = 0x00FFFF
    COLOR_HOT_PINK = 0xFF69B4

    PROBMAP_MIN_DRAW = 0.35                                             # Lower clamp for prob map draw
    PROBMAP_MAX_SHOWN = 0.75                                            # Upper clamp for prob map draw
    PROBMAP_GAMMA = 0.85                                                # Gamma for contrast tweak

    def __init__(self, display=None):                                   
        self.display = display or GetFromBlackboard("display")          # Use provided or fetch from bb
        self.width = self.display.getWidth() if self.display else 0     # Cache width    
        self.height = self.display.getHeight() if self.display else 0   # Cache height

    def InitializeDisplay(self):                                        
        if not self.display:                                            # If display object wasn't set
            self.display = GetFromBlackboard("display")                 # Try to obtain it from blackboard
            if not self.display:                                        # If still missing
                return False                                            # Signal failure to initialize
        self.width, self.height = self.display.getWidth(), self.display.getHeight() # refresh size
        self.ClearDisplay()                                              # Blank screen
        return True

    def ClearDisplay(self):                                              
        if self.display:                                                # Only if a display is available
            self.display.setColor(self.COLOR_BLACK)                     # Black background
            self.display.fillRectangle(0, 0, self.width, self.height)   # Cover entire screen area

    def GetMapShape(self) -> tuple[int, int]:
        cspace = GetFromBlackboard("cspace")                            # Try c-space grid
        if cspace is not None:
            return cspace.shape
        prob_map = GetFromBlackboard("prob_map")                        # Else try prob map
        if prob_map is not None:
            return prob_map.shape
        return (200, 300)                                               # Fallback grid size

    def MapToDisplay(self, row, col, shape):
        return MapToDisplayCoords(row, col, shape, self.width, self.height) # Convert grid coords to display coords

    def _in_bounds(self, x, y, w=2, h=2):
        return 0 <= x < self.width - w and 0 <= y < self.height - h     # Rect fits display
    
    def DrawPixel(self, x, y, color):
        if self._in_bounds(x, y):
            self.display.setColor(color)                                # Set color
            self.display.fillRectangle(x, y, 2, 2)                      # Draw 2x2 block

    def DrawWorldLine(self, start_point, end_point, shape, color):
        if not self.display:
            return                                                      # No display, nothing to do
        sx, sy = self.MapToDisplay(*WorldToGrid(*start_point, shape), shape) # Map world start to display
        ex, ey = self.MapToDisplay(*WorldToGrid(*end_point, shape), shape)  # Map world end to display
        self.display.setColor(color)                                    # Set line color
        self.display.drawLine(sx, sy, ex, ey)                           # Draw the line in display space

    def DrawProbabilityMap(self, probability_map=None):
        if not self.display:
            return
        probability_map = GetFromBlackboard("prob_map") if probability_map is None else probability_map
        if probability_map is None:
            return
        p = np.clip(probability_map, self.PROBMAP_MIN_DRAW, self.PROBMAP_MAX_SHOWN) # Clamp values
        p = (p - self.PROBMAP_MIN_DRAW) / (self.PROBMAP_MAX_SHOWN - self.PROBMAP_MIN_DRAW)
        p = p ** self.PROBMAP_GAMMA                                     # Apply gamma correction
        h, w = p.shape                                                  # Grid height and width
        step = 2                                                        # Sample every 2 cells for speed
        rows, cols = np.mgrid[0:h:step, 0:w:step]                       # Generate sampled indices
        intensity = (p[rows, cols] * 255).astype(np.uint8)              # Convert to grayscale
        px = (rows * self.width // h).astype(int)                       # Map grid rows to display x
        py = (cols * self.height // w).astype(int)                      # Map grid cols to display y
        bw = max(2, self.width // w)                                    # Block width in pixels
        bh = max(2, self.height // h)                                   # Block height in pixels
        valid = (px >= 0) & (px < self.width - bw) & (py >= 0) & (py < self.height - bh)    # Within bounds mask
        vr, vc = np.where(valid)                                        # Indices of valid sample blocks
        if len(vr) > 10000:                                             # If too many blocks make it wait
            s = max(1, len(vr) // 10000)
            vr, vc = vr[::s], vc[::s]                                   # Downsample indices
        for i, j in zip(vr, vc):                                        # For each valid sampled block
            g = int(intensity[i, j])                                    # Grayscale intensity
            rgb = (g << 16) | (g << 8) | g
            self.display.setColor(rgb)                                  # Set fill color
            self.display.fillRectangle(px[i, j], py[i, j], bw, bh)      # Draw the block on screen

    def DrawTrajectory(self, trajectory_points=None, map_shape=None, color=None):
        if not self.display:                                            # Require a display
            return
        traj = trajectory_points or GetFromBlackboard("trajectory_points") # Use given or stored trajectory
        if not traj or len(traj) < 2:                                   # Need at least two points to draw lines
            return
        shape = map_shape or self.GetMapShape()                         # Determine grid shape
        col = color or self.COLOR_HOT_PINK                              # Default trajectory color
        for a, b in zip(traj, traj[1:]):
            self.DrawWorldLine(a, b, shape, col)

    def DrawRobotPosition(self, map_shape=None, color=None, size=4):
        if not self.display:                                            # Require a display
            return
        gps = GetFromBlackboard("gps")                      
        if not gps:
            return
        wx, wy = gps.getValues()[:2]                                    # World coordinates
        shape = map_shape or self.GetMapShape()                         # Determine grid shape
        x, y = self.MapToDisplay(*WorldToGrid(wx, wy, shape), shape)    # Map world to display
        if 0 <= x < self.width and 0 <= y < self.height:                # Only draw if on-screen
            self.display.setColor(color or self.COLOR_BLUE)             # Choose marker color
            hs = size // 2
            self.display.fillOval(x - hs, y - hs, size, size)           # Draw filled oval at robot position

    def DrawPlannedPath(self, path=None, map_shape=None, color=None):
        if not self.display:                                           # No display do nothing
            return
        planned = path or GetFromBlackboard("planned_path")            # Use provided path or fetch from blackboard
        if not planned:                                                # If there is still no path
            return                                                     # Exit
        shape = map_shape or self.GetMapShape()                        # Determine grid shape used for mapping
        col = color or self.COLOR_CYAN                                 # Choose path color
        for a, b in zip(planned, planned[1:]):                        
            self.DrawWorldLine(a, b, shape, col)                       # Draw line segment between waypoints
        self.display.setColor(col)                                     
        for wx, wy in planned:                                         # For each waypoint in world coords
            x, y = self.MapToDisplay(*WorldToGrid(wx, wy, shape), shape)  # Map world to display pixel coords
            if self._in_bounds(x, y, 2, 2):                            # Ensure a 2by2 dot fits on-screen
                self.display.fillOval(x - 1, y - 1, 2, 2)              # Draw a small dot at the waypoint

    def DrawNavigationGoals(self, goals=None, map_shape=None, color=None):
        if not self.display:                                           # Require display
            return
        nav_goals = goals or GetFromBlackboard("navigation_goals")     # Use provided list or fetch from bb
        if not nav_goals:                                              # Nothing to draw
            return
        shape = map_shape or self.GetMapShape()                        # Determine grid shape
        self.display.setColor(color or self.COLOR_YELLOW)              
        for gx, gy in nav_goals:                                       # For each goal
            x, y = self.MapToDisplay(*WorldToGrid(gx, gy, shape), shape) # Convert to display coords
            if self._in_bounds(x, y, 3, 3):                            # Ensure the cross fits on screen
                s = 3                                                  
                self.display.drawLine(x - s, y - s, x + s, y + s)      
                self.display.drawLine(x - s, y + s, x + s, y - s)      

    def DrawCspace(self, cspace):
        if not (self.display and cspace is not None):                  # Need display and a c-space
            return
        h, w = cspace.shape                                            # Grid dimensions
        for r in range(0, h, 2):                                       # Iterate every other row for speed
            row = cspace[r]                                            # Cache row to avoid repeated indexing
            for c in range(0, w, 2):                                   # Iterate every other column
                color = self.COLOR_BLACK if float(row[c]) > 0.5 else self.COLOR_WHITE  # Threshold cell value
                x, y = self.MapToDisplay(r, c, (h, w))                 # Map grid cell to display pixel
                self.DrawPixel(x, y, color)                            

    def UpdateDisplay(self, mode="full"):
        if not self.display:                                           # No display device available
            return
        self.ClearDisplay()                                            # Start from a clean background
        shape = self.GetMapShape()                                     # Determine map/grid shape for mapping
        if mode in ("cspace", "planning"):                            
            cspace = GetFromBlackboard("cspace")                       # Try to get c-space
            if cspace is not None:                                     # If available, draw it as background
                self.DrawCspace(cspace)
            if mode == "planning":                                     # In planning mode, overlay planning layers
                self.DrawPlannedPath(map_shape=shape)                  
                self.DrawNavigationGoals(map_shape=shape)              
                self.DrawRobotPosition(map_shape=shape)                # Show current robot pose
            return                                                     
        prob_map = GetFromBlackboard("prob_map")                       # Default mode probability map background
        if prob_map is not None:                                       # If available
            self.DrawProbabilityMap(prob_map)                          # Render grayscale probability field
        self.DrawTrajectory(map_shape=shape)                           # Overlay past trajectory 
        self.DrawRobotPosition(map_shape=shape)                        # Overlay current robot position marker
_display_manager = None                                                
def GetDisplayManager():
    global _display_manager                                            # Refer to the module global
    if _display_manager is None:                                       
        _display_manager = DisplayManager()                            # Create a DisplayManager 
    return _display_manager                                            # Return the cached 


################################################################################
# ============================ Waypoint Generation =============================
#################################################################################
def WorldBoundsFromConfig(shape=(200, 300)):
    h, w = shape                                                        # Unpack grid shape 
    x_min, y_max = GridToWorld(0, 0)                                    # Convert top left grid cell to world coords
    x_max, y_min = GridToWorld(h - 1, w - 1)                            # Convert bottom right grid cell to world coords
    return x_min, y_min, x_max, y_max                                   # Return world-space bounds

def BuildEllipsePoints(center=(-0.65, -1.43), rx=1.05, ry=1.25, num_points=12, rotation=0.0):
    cx, cy = center                                                     # Center of the world coords
    ang = np.linspace(0, 2 * np.pi, num_points, endpoint=False) + rotation  # Spaced angles with rotation
    return [(cx + rx * np.cos(a),                                       # X coordinate
             cy + ry * np.sin(a))                                       # Y coordinate
            for a in ang]                                               # Generate one x and y for each angle

def BuildPerimeterLoop(margin=None, include_midpoints=True):
    x_min, y_min, x_max, y_max = WorldBoundsFromConfig()                # Get world bounds of the map
    m = margin if margin is not None else max(0.6, 0.18 + 0.4)          # Use provided margin or a safe default
    left, right, bottom, top = x_min + m, x_max - m, y_min + m, y_max - m  # Inset rectangle by margin
    mid_x, mid_y = 0.5 * (left + right), 0.5 * (bottom + top)           # Midpoints along x and y
    pts = [(left, bottom)]                                              # Start at bottom left corner
    if include_midpoints: pts += [(mid_x, bottom)]            
    pts += [(right, bottom)]                                            # Bottom right corner
    if include_midpoints: pts += [(right, mid_y)]             
    pts += [(right, top)]                                               # Top-right corner
    if include_midpoints: pts += [(mid_x, top)]               
    pts += [(left, top)]                                                # Top-left corner
    if include_midpoints: pts += [(left, mid_y)]              
    return pts                                                          # Return loop as a list of waypoints

def OrderFromStart(points, start_position, close_loop=True):
    sx, sy = start_position                                             # Extract starting world coordinates
    d = [np.hypot(x - sx, y - sy) for (x, y) in points]                 # Compute Euclidean distance to each point
    i = int(np.argmin(d))                                               # Find index of nearest point to start position
    ordered = points[i:] + points[:i]                                   # Rotate list so nearest point is first
    return ordered + [start_position] if close_loop else ordered  

def ClampGoal(goal_x, goal_y, cspace=None):
    if cspace is None:                                                  # If no c-space provided, nothing to clamp
        return goal_x, goal_y                                           # Return goal 
    h, w = cspace.shape                                      
    x_min, y_min = GridToWorld(h - 1, 0)                                # World coords of bottom left grid cell
    x_max, y_max = GridToWorld(0, w - 1)                                # World coords of top right grid cell
    buf = 0.20                                                
    return (                                                  
        min(max(goal_x, x_min + buf), x_max - buf),                     # Clamp goal_x between 
        min(max(goal_y, y_min + buf), y_max - buf),                     # Clamp goal_y between
    )                                                                   # Return clamped goal coordinates


################################################################################
# ========================== Behavior Tree Nodes ===============================
################################################################################
class WaitForMapReady(BehaviorNode):
    def __init__(self):
        super().__init__("WaitForMapReady")                             
        self._start_time = None                                         # Robot time when we began waiting
        self._warned = [False] * 3                                      # Track which warnings were already sent

    def execute(self):
        robot = GetFromBlackboard("robot")                              # Get robot interface (provides time)
        if robot and self._start_time is None:                          # First tick with a robot present
            self._start_time = robot.getTime()                          # Record the start time
        if GetFromBlackboard("map_saved"):                              
            return Status.SUCCESS
        cspace = GetFromBlackboard("cspace")                            # Check if a c-space exists yet
        if cspace is None:                                              # No c-space available
            if robot and self._start_time is not None:          
                elapsed = robot.getTime() - self._start_time            # Time since start of waiting
                msgs = [                                                
                    (120, "Still no map after 120s."),
                    (60, "Map build is slow 60s."),
                    (30, "Waiting for C-space 30s"),
                ]
                for idx, (t, msg) in enumerate(msgs[::-1]):             
                    real_idx = 2 - idx                                  
                    if elapsed > t and not self._warned[real_idx]:      
                        main_logger.Warning(msg)                        # Give one time warning
                        self._warned[real_idx] = True                   # Mark that we warned at this level
            return Status.RUNNING                                       # Keep waiting for map
        return Status.SUCCESS if CalculateFreeSpacePercentage(cspace) >= 0.01 else Status.RUNNING  # Ready when there is some free space

    def reset(self):
        super().reset()                                                 # Reset base node state
        self._start_time = None                                         # Clear start timer
        self._warned = [False] * 3                                      # Reset warning flags

class MapExistsOrReady(BehaviorNode):
    def __init__(self, path="cspace.npy"):
        super().__init__("MapExistsOrReady")                            # Name the node
        self.path = str(ResolveMapPath(path))                           # Path to map file

    def execute(self):
        return Status.SUCCESS if (GetFromBlackboard("map_ready", False) or exists(self.path)) else Status.FAILURE  # Succeeds if either flag 

class LoadMap(BehaviorNode):
    def __init__(self, path="cspace.npy"):
        super().__init__("LoadMap")                                     # Name the node
        self.path = str(ResolveMapPath(path))                           # Path to map to load

    def execute(self):
        try:
            c = np.clip(np.load(self.path).astype(np.float32), 0.0, 1.0)  
            blackboard.SetCspace(c)                                     # Publish c-space to blackboard
            blackboard.Set(BBKey.CSPACE_FROZEN, True)                   # Freeze c-space to avoid accidental edits
            blackboard.SetMapReady(True)                                # Mark map as ready for use
            return Status.SUCCESS                                     
        except Exception as e:
            main_logger.Error(f"LoadMap failed {e}")                    # Log load error
            return Status.FAILURE                                       # Fail the node

class EnsureCspaceNow(BehaviorNode):
    def __init__(self, blackboard_instance=None):
        super().__init__("EnsureCspaceNow")                             # Name the node
        self.blackboard = blackboard_instance or blackboard             # Use provided bb or default global

    def execute(self):
        c = self.blackboard.GetCspace()                                 # Already have c-space?
        if c is not None:
            return Status.SUCCESS                                       # Nothing to do
        p = self.blackboard.GetProbMap()                                # Try to get probability map
        if p is None:
            return Status.RUNNING                                       # Wait for prob map
        try:
            mapper = LidarMappingBT(params=MappingParams(), bb=self.blackboard)
            c = mapper.create_cspace(p)                                # Build c-space from probability map
            if c is not None:
                self.blackboard.SetCspace(c)                           # Store c-space
                return Status.SUCCESS
            return Status.FAILURE                                      # Mapper returned None
        except Exception as e:
            main_logger.Error(f"c-space build crashed: {e}")           # Mapping pipeline crashed
            return Status.FAILURE

class SaveMap(BehaviorNode):
    def __init__(self, path="cspace.npy", threshold=None):
        super().__init__("SaveMap")                                     # Name the node
        self.path = str(ResolveMapPath(path))                           # Path
        self.threshold = 0.30 if threshold is None else threshold     
        self.done = False                                               # Guard against duplicate saves

    def PrepareMapForSaving(self) -> np.ndarray | None:
        c = GetFromBlackboard("cspace")                                 # Prefer existing c-space
        if c is None:                                                   # If none, try to build from prob map
            p = GetFromBlackboard("prob_map")                           # Fetch probability map
            if p is None:                                               # If nothing available
                main_logger.Error("no map.")                            # Log and abort
                return None
            mapper = LidarMappingBT(MappingParams())                    # Create mapping component
            c = mapper.create_cspace(p)                                 # Attempt building c-space
            if c is None:                                               # If that fails
                c = (p <= self.threshold).astype(np.float32)            # Threshold as last resort
        return np.clip(c.astype(np.float32), 0.0, 1.0)                 

    def ShouldSaveMap(self, cspace: np.ndarray) -> bool:
        return CalculateFreeSpacePercentage(cspace) * 100.0 >= 0.1      # Save if at least 0.1% of space is free (more lenient)

    def SaveMapToFile(self, cspace: np.ndarray) -> bool:
        try:
            EnsureParentDirectories(self.path)                          # Make sure folders exist
            np.save(self.path, cspace)                                  # Save as .npy array
            return True
        except Exception as e:
            main_logger.Error(f"Write failed to save map: {e}")           
            return False

    def UpdateMapState(self, cspace: np.ndarray):
        for k, v in (("cspace", cspace), ("map_ready", True), ("map_saved", True), ("cspace_frozen", True)):
            blackboard.Set(k, v)                                        # Update multiple blackboard keys
        self.done = True                                                # Mark as finished

    def execute(self):
        if self.done:                                                   # If already saved once
            return Status.SUCCESS
        c = self.PrepareMapForSaving()                                  # Obtain c-space to save
        if c is None:                                                   # Abort if none
            return Status.FAILURE
        if not self.ShouldSaveMap(c):                                   # Check quality threshold
            main_logger.Error("C-space rejected - not saving.")         # Explain why not saved
            return Status.FAILURE
        try:
            from os import remove                                       # Import for file removal
            if exists(self.path):                                       # If file already exists
                remove(self.path)                                       # Remove it first
        except Exception:                                               # Ignore deletion errors
            pass
        if not self.SaveMapToFile(c):                                 
            return Status.FAILURE
        self.UpdateMapState(c)                                        
        return Status.SUCCESS                                           # Success

class DisplayUpdater(BehaviorNode):
    def __init__(self):
        super().__init__("DisplayUpdater")                              # Name the node
        self.manager = GetDisplayManager()                              # Display manager helper
        self.fps = EveryNSeconds(0.25)                                  # Rate limit UI updates (4 Hz)

    @staticmethod
    def _truthy(v):
        try:
            if isinstance(v, np.ndarray):                               # True if any element is none zero
                return bool(np.any(v))
        except Exception:
            pass
        try:
            if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):  
                return len(v) > 0
        except Exception:
            pass
        return bool(v)                                                  # Fallback

    def execute(self):
        try:
            if not (self.manager and self.manager.display):             # Need a display to draw
                return Status.RUNNING
            robot = GetFromBlackboard("robot")                          # For timing control
            t = robot.getTime() if robot else None                      # Current robot time
            if t is None or not self.fps.ShouldExecute(t):             
                return Status.RUNNING
            display_mode = GetFromBlackboard("display_mode", "full")    # Current requested mode
            allow_cspace = self._truthy(GetFromBlackboard("allow_cspace_display", False))  
            survey_done = self._truthy(GetFromBlackboard("survey_complete", False))        # Survey completion flag
            cspace = GetFromBlackboard("cspace")                        # Current c-space map
            has_cspace = (cspace is not None) and (CalculateFreeSpacePercentage(cspace) >= 0.01)  # Basic validity
            if display_mode == "full" and allow_cspace and survey_done and has_cspace:     # Auto switch to c-space
                blackboard.Set(BBKey.DISPLAY_MODE, "cspace")          
                display_mode = "cspace"                                 # Use it this tick
            self.manager.UpdateDisplay(mode=display_mode)               # Render according to mode
        except Exception as e:
            main_logger.Error(f"DisplayUpdater error {e}")              # Log any rendering errors
        return Status.RUNNING                                           # Keep ticking forever

class SetDisplayMode(BehaviorNode):
    def __init__(self, mode):
        super().__init__(f"SetDisplayMode({mode})")                     # Name includes target mode
        self.mode = mode                                                # Desired display mode string

    def execute(self):
        blackboard.Set(BBKey.DISPLAY_MODE, self.mode)                   # Update blackboard flag
        return Status.SUCCESS                                          

class NavigateToWaypoints(BehaviorNode):
    def __init__(self):
        super().__init__("NavigateToWaypoints")                         # Name the node
        self.navigator = None                                           # Created path follower

    def execute(self):
        planned_path = GetFromBlackboard("planned_path")                # Retrieve path from planning stage
        if not planned_path:                                            # No path available is an error
            main_logger.Error("No planned path available.")
            return Status.FAILURE
        if self.navigator is None:                                      # Create controller on first run
            self.navigator = ObstacleAvoidingNavigation(planned_path, bb=blackboard, traversal="once")
        status = self.navigator.execute()                               # Step the controller
        if status == Status.SUCCESS:                                    # If finished
            self.navigator = None                                       # Reset for next time
        return status                                                   

    def reset(self):
        super().reset()                                                 # Reset base node
        self.navigator = None                                           # Drop controller instance

    def terminate(self):
        if self.navigator:                                              # If running, stop safely
            self.navigator.terminate()
            self.navigator = None

class ValidateLoadedMap(BehaviorNode):
    def __init__(self):
        super().__init__("ValidateLoadedMap")                           # Name the node

    def execute(self):
        c = GetFromBlackboard("cspace")                                 # Fetch loaded c-space
        if c is None:                                                   # If map is missing then invalid
            return Status.FAILURE
        free_pct = 100.0 * CalculateFreeSpacePercentage(c)              # Percentage of free cells
        if free_pct < 0.1:                                            
            main_logger.Error(f"Loaded map looks wrong: free={free_pct:.2f}% (<0.1%).")
            return Status.FAILURE
        return Status.SUCCESS                                           


class ContinuousFreeRoam(BehaviorNode):
    def __init__(self, params, num_waypoints=12):
        super().__init__("ContinuousFreeRoam")                          # Name the node
        self.params = params                                            # Motion/planning params
        self.num_waypoints = num_waypoints                              # Desired waypoint density for survey
        self.last_direction_change = 0                                  # Last time the heading target changed
        self.direction_change_interval = 3.0                            # Seconds between heading updates
        self.current_direction = 0                                      # Target heading angle 
        self.planning_attempts = 0                                      # Number of planning attempts so far
        self.max_planning_attempts = 2                                  # Upper bound on planning retries
        self.params.W2G = blackboard.Get("world_to_grid")               # Inject transforms into params
        self.params.G2W = blackboard.Get("grid_to_world")   
        self.log_limiter = EveryNSeconds(10.0)                          # Rate limit logs

    def execute(self):
        robot = GetFromBlackboard("robot")                              # Need robot to operate
        if not robot:
            return Status.FAILURE                                       # Without robot context we cannot run
        t = robot.getTime()                                             # Current time
        cspace = GetFromBlackboard("cspace")                            # Current c-space
        map_ready = GetFromBlackboard("map_ready", False)               # Ready flag
        if cspace is not None and self.log_limiter.ShouldExecute(t):    # Periodic status log
            main_logger.Info(f"FreeRoam: c-space ready, shape={cspace.shape}, map_ready={map_ready}")
        if self.planning_attempts < self.max_planning_attempts and cspace is not None:  # Try planning first
            self.planning_attempts += 1                                 # Count attempt
            main_logger.Info(f"FreeRoam: planning attempt {self.planning_attempts}/{self.max_planning_attempts}")
            gps = GetFromBlackboard("gps")                              # Position source
            current_pos = gps.getValues()[:2] if gps else None         
            safe_goals = find_safe_positions(                             # Sample a few safe goals
                cspace,
                self.params,
                num_positions=3,
                restrict_to_reachable_from=current_pos,
                check_neighbors=False
            )
            if safe_goals:                                               # If candidates exist
                planner = MultiGoalPlannerBT(self.params, bb=blackboard) # Build planner
                planner.params.W2G = self.params.W2G                     # Ensure transforms
                planner.params.G2W = self.params.G2W
                blackboard.SetNavigationGoals(safe_goals)                # Publish goals
                if planner.execute() == Status.SUCCESS:                  # Attempt planning
                    planned_path = GetFromBlackboard("planned_path")     # Retrieve path
                    if planned_path:                                     # If path available
                        self.current_navigator = ObstacleAvoidingNavigation(  # Start following
                            planned_path, bb=blackboard, traversal="once"
                        )
                        main_logger.Info("Planning worked, following path.")
                        return Status.RUNNING                             # Continue running navigator
            main_logger.Warning(f"Planning attempt {self.planning_attempts} failed.")  # Planning failed
        if self.log_limiter.ShouldExecute(t):                             # Switch strategy
            main_logger.Info("Switching to c-space.")
        return self.CspaceAwareReactiveRoam(t)                            # Reactive mode guided by c-space

    def _maybe_change_direction(self, now):
        if now - self.last_direction_change > self.direction_change_interval:  # Time to pick new heading
            self.current_direction = np.random.uniform(0, 2 * np.pi)      # Random heading 
            self.last_direction_change = now                              # Update timestamp
            main_logger.Info(f"New heading = {self.current_direction:.2f} rad")

    def CspaceAwareReactiveRoam(self, now):
        try:
            self._maybe_change_direction(now)                            # Occasionally change desired heading
            cspace = GetFromBlackboard("cspace")                         # Occupancy/cost map
            compass = GetFromBlackboard("compass")                       # Orientation sensor
            lidar = GetFromBlackboard("lidar")                           # Range sensor
            gps = GetFromBlackboard("gps")                               # Position sensor
            if not all([compass, gps]):                                
                if self.log_limiter.ShouldExecute(now):
                    main_logger.Warning("FreeRoam: GPS/Compass missing, falling back to Lidar.")
                return self.SimpleReactiveRoam(now)                      # Downgrade to simpler behavior
            heading = float(np.arctan2(compass.getValues()[0], compass.getValues()[1])) 
            gx, gy = gps.getValues()[:2]                                 # Current position in world
            pos = (gx, gy)
            if self.params.W2G:                                          # Have world to grid mapping
                try:
                    gr, gc = map(int, self.params.W2G(pos[0], pos[1]))   # Convert to grid indices
                except Exception as e:
                    if self.log_limiter.ShouldExecute(now):
                        main_logger.Warning(f"FreeRoam: W2G failed ({e}).")
                    return self.SimpleReactiveRoam(now)                  # Fall back on failure
            else:
                if self.log_limiter.ShouldExecute(now):
                    main_logger.Warning("No W2G available; using Lidar only.")
                return self.SimpleReactiveRoam(now)
            if (cspace is None or gr < 0 or gr >= cspace.shape[0] or gc < 0 or gc >= cspace.shape[1]):  # Bounds check
                if self.log_limiter.ShouldExecute(now):
                    main_logger.Warning(f"In free roam grid pos out of bounds ({gr}, {gc}).")
                return self.SimpleReactiveRoam(now)
            val = cspace[gr, gc]                                      
            if val < 0.6:                                              
                if self.log_limiter.ShouldExecute(now):
                    main_logger.Info(f"FreeRoam: close to obstacle (c={val:.3f}).")
                ang_v = 3.0 if np.random.random() > 0.5 else -3.0        # Turn away quickly
                lin_v = 0.2                                              # Slow linear speed
            else:                                                        # Space ahead looks clear
                if self.log_limiter.ShouldExecute(now):
                    main_logger.Info(f"FreeRoam: clear space (c={val:.3f}); moving.")
                err = (self.current_direction - heading + np.pi) % (2 * np.pi) - np.pi  # Smallest signed angle
                ang_v = err * 1.5                                        # Proportional steering
                lin_v = 1.5                                              # Nominal forward speed
                d = 0.5                                                  # Lookahead distance
                lx = pos[0] + d * np.cos(heading)                        # Lookahead x in world
                ly = pos[1] + d * np.sin(heading)                        # Lookahead y in world
                try:
                    lr, lc = map(int, self.params.W2G(lx, ly))           # Lookahead grid indices
                    if (0 <= lr < cspace.shape[0] and 0 <= lc < cspace.shape[1] and cspace[lr, lc] < 0.6):
                        ang_v = 2.5 if np.random.random() > 0.5 else -2.5  # Preemptively steer away
                        lin_v = 0.5
                except Exception:
                    pass                                                 # Ignore mapping errors for lookahead
            ml = GetFromBlackboard("motorL")                             # Left motor handle
            mr = GetFromBlackboard("motorR")                             # Right motor handle
            if ml and mr:                                                # If motors are present
                SafeSetMotorVelocities(ml, mr, lin_v - ang_v, lin_v + ang_v) 
            return Status.RUNNING                                        # Keep roaming
        except Exception as e:
            main_logger.Error(f"FreeRoam crashed: {e}")   # Log crash
            return self.SimpleReactiveRoam(now)                          # Fall back to simpler mode

    def SimpleReactiveRoam(self, now):
        try:
            self._maybe_change_direction(now)                           # Occasionally change heading target
            compass = GetFromBlackboard("compass")                      # Orientation sensor
            lidar = GetFromBlackboard("lidar")                          # Range sensor
            ml = GetFromBlackboard("motorL")                            # Left motor
            mr = GetFromBlackboard("motorR")                            # Right motor
            if not all([compass, lidar]):                               # If critical sensors missing
                if ml and mr:
                    SafeSetMotorVelocities(ml, mr, 0.5, 0.5)        
                return Status.RUNNING
            heading = float(np.arctan2(compass.getValues()[0], compass.getValues()[1]))  
            ranges = np.array(lidar.getRangeImage())                    # Raw range image
            vr = ranges[np.isfinite(ranges)]                            # Keep only finite returns
            if len(vr) == 0:                                            # No usable ranges
                if ml and mr:
                    SafeSetMotorVelocities(ml, mr, 0.5, 0.5)        
                return Status.RUNNING
            if np.min(vr) < 0.5:                                        # Obstacle detected within 0.5 m
                ang_v, lin_v = (2.0 if np.random.random() > 0.5 else -2.0), 0.2  # Turn in place-ish
            else:                                                       # Clear
                err = (self.current_direction - heading + np.pi) % (2 * np.pi) - np.pi  # Heading error
                ang_v, lin_v = err * 1.0, 1.0                           # Proportional steering and forward speed
            if ml and mr:                                               # Command motors
                SafeSetMotorVelocities(ml, mr, lin_v - ang_v, lin_v + ang_v)
            return Status.RUNNING
        except Exception as e:
            main_logger.Error(f"FreeRoam: crashed: {e}")                # Log failure
            ml = GetFromBlackboard("motorL")                            # Attempt to stop
            mr = GetFromBlackboard("motorR")
            if ml and mr:
                StopMotors(ml, mr)
            return Status.RUNNING                                       # Keep node alive

class SetTwoGoals(BehaviorNode):
    def __init__(self, goals=None, num_goals=2, use_outer_perimeter=False):
        super().__init__("SetTwoGoals")                                  # Name the node
        self.goals = goals                                              
        self.num_goals = num_goals                                       # Number of goals to choose
        self.use_outer_perimeter = use_outer_perimeter                   

    def execute(self):
        if self.goals:                                                   # If caller supplied goals
            blackboard.SetNavigationGoals(self.goals)                    # Publish directly
            return Status.SUCCESS                                        # Done
        cspace = GetFromBlackboard("cspace")                             # Need map to pick safe goals
        if cspace is None:
            main_logger.Error("SetTwoGoals: c-space not available yet.")
            return Status.FAILURE
        gps = GetFromBlackboard("gps")                                   # Use current position if available
        curr = gps.getValues()[:2] if gps else None                   
        if self.use_outer_perimeter:                                   
            start_xy = GetFromBlackboard("start_xy", curr)              # Use start or current position
            outer_perim = OrderFromStart(BuildPerimeterLoop(), start_xy, close_loop=False)  
            outer_perim = [ClampGoal(x, y, cspace) for x, y in outer_perim]  # Clamp to safe interior
            if len(outer_perim) >= 4:                                   
                idx1 = len(outer_perim) // 4
                idx2 = (3 * len(outer_perim)) // 4
                goals = [outer_perim[idx1], outer_perim[idx2]]
            else:
                goals = outer_perim[:2] if len(outer_perim) >= 2 else outer_perim  
            if len(goals) < 2:                                           # Ensure two goals exist
                main_logger.Error(f"Not enough waypoints ({len(goals)})")
                return Status.FAILURE
            main_logger.Info(f"SetTwoGoals: Goal 1=({goals[0][0]:.2f},{goals[0][1]:.2f}), Goal 2=({goals[1][0]:.2f},{goals[1][1]:.2f})")
            blackboard.SetNavigationGoals(goals)                         # Publish
            return Status.SUCCESS
        pp = PlanningParams()                                            
        pp.W2G = GetFromBlackboard("world_to_grid")                      # Inject transforms
        pp.G2W = GetFromBlackboard("grid_to_world")
        safe_goals = find_safe_positions(                                  # Ask for N safe positions
            cspace,
            pp,
            num_positions=self.num_goals,
            restrict_to_reachable_from=curr,
            check_neighbors=True
        )
        if not safe_goals or len(safe_goals) < self.num_goals:          # Validate result size
            main_logger.Error(f"SetTwoGoals: found {len(safe_goals) if safe_goals else 0} safe goals (need {self.num_goals}).")
            return Status.FAILURE
        blackboard.SetNavigationGoals(safe_goals)                        # Publish chosen goals
        main_logger.Info(f"SetTwoGoals: selected {len(safe_goals)} safe goals.")
        return Status.SUCCESS


class ValidateAndVisualizeWaypoints(BehaviorNode):
    def __init__(self):
        super().__init__("ValidateAndVisualizeWaypoints")               # Name the node
        self.done = False                                               # Run at most once

    def execute(self):
        if self.done:                                                   # Already executed successfully
            return Status.SUCCESS
        cspace = GetFromBlackboard("cspace")                            # Need c-space to validate paths
        if cspace is None:                                              # Wait for c-space
            return Status.RUNNING
        for name, path in [("INNER_12", INNER_12), ("OUTER_PERIM", OUTER_PERIM)]:  # Known waypoint sets
            try:
                if validate_path(path, cspace):                          # If path is valid through c-space
                    visualize_path_on_map(cspace, path, save_path=f"{name}_viz.npy")  # Save viz overlay for display
            except Exception:
                pass                                                     # Ignore any visualization errors
        self.done = True                                                 # Mark as finished
        return Status.SUCCESS


class BidirectionalSurveyNavigator(BehaviorNode):
    def __init__(self, survey_waypoints):
        super().__init__("BidirectionalSurveyNavigator")                # Name the node
        self.survey_waypoints = survey_waypoints                        # Waypoints to forward/back
        self.current_phase = 1                                        
        self.phase2_started_time = None                                 # Time when reverse phase begins
        self.rotation_target = None                                     # Target heading for in place rotation
        self.navigation_controller = ObstacleAvoidingNavigation(        # Path follower instance
            survey_waypoints, tolerance=0.30, start_direction=+1
        )

    def _tick_rotation(self):
        MAX_W, GAIN = 3.0, 4.0                                           # Rotation controller params
        ml, mr = GetFromBlackboard("motorL"), GetFromBlackboard("motorR")  # Motors
        compass, lidar = GetFromBlackboard("compass"), GetFromBlackboard("lidar")  # Sensors
        if not all([ml, mr, compass]):                                   # Need motors + compass to rotate
            main_logger.Error("Rotation error need motors and compass.")
            return True                                                  
        lidar_ranges = np.array(lidar.getRangeImage()) if lidar else np.array([])  # LiDAR safety
        if lidar_ranges.size and np.isfinite(lidar_ranges).any():
            if np.min(lidar_ranges[np.isfinite(lidar_ranges)]) < 0.5:    # Too close to obstacle
                SafeSetMotorVelocities(ml, mr, 0, 0)                  # Stop immediately
                return False                                             # Not safe to rotate yet
        if self.rotation_target is None:                                 # Initialize rotation goal
            vx, vy = compass.getValues()[:2]
            self.rotation_target = NormalizeAngle(np.arctan2(vx, vy) + np.pi)
        vx, vy = compass.getValues()[:2]                                 # Current heading
        heading = np.arctan2(vx, vy)
        err = NormalizeAngle(self.rotation_target - heading)             # Smallest signed error
        w = float(np.clip(GAIN * err, -MAX_W, MAX_W))                
        SafeSetMotorVelocities(ml, mr, -w, w)                         # Spin in place
        if abs(err) < np.radians(10):                             
            SafeSetMotorVelocities(ml, mr, 0, 0)                      # Stop
            self.rotation_target = None                                  # Clear for next time
            return True                                                  # Rotation complete
        return False                                                     # Continue rotating

    def _begin_reverse(self):
        if self.navigation_controller:                                   # Stop current controller 
            self.navigation_controller.terminate()
        self.current_phase = 1.5                                         # Enter rotation phase
        self.rotation_target = None                                      # Reset rotation target

    def execute(self):
        if self.current_phase == 1.5:                                    # Rotation between passes
            if self._tick_rotation():                                    # If rotation finished
                self.navigation_controller = ObstacleAvoidingNavigation(
                    self.survey_waypoints, tolerance=0.30, start_direction=-1
                )
                self.navigation_controller.reset()                        # Reset controller for reverse run
                robot = GetFromBlackboard("robot")
                self.phase2_started_time = robot.getTime() if robot else None  # Record start time of phase 2
                self.current_phase = 2                                    # Now in reverse traversal
            return Status.RUNNING
        status = self.navigation_controller.execute()                     # Step the current traversal
        gps = GetFromBlackboard("gps")                                    # GPS for position updates
        start_position = GetFromBlackboard("start_xy")                    # Starting point
        if gps:
            gx, gy, _ = gps.getValues()                                   # Current pose
            cur = (gx, gy)
            traj = UpdateTrajectory(GetFromBlackboard("trajectory_points") or [], cur)  # Append to trajectory
            blackboard.SetTrajectory(traj)                               # Store for UI
            if start_position and len(traj) > 10:                        # After enough points
                dist = Distance2D(cur, start_position)                   # Distance to origin
                if self.current_phase == 1 and dist < 0.30:              # Near origin during forward pass
                    self._begin_reverse()                                # Switch to rotation/reverse
                    return Status.RUNNING
                if self.current_phase == 2:                              # During reverse run
                    robot = GetFromBlackboard("robot")
                    time_gate = (robot and self.phase2_started_time is not None and
                                 (robot.getTime() - self.phase2_started_time) > 2.0)  # Prevent instant finish
                    if time_gate and dist < 0.30:                        # Back at start after some time
                        return Status.SUCCESS                            # Survey complete
        if status == Status.SUCCESS and self.current_phase == 1:         # Finished forward pass early
            self._begin_reverse()                                        # Force reverse
            return Status.RUNNING
        if self.current_phase == 2 and status == Status.FAILURE and gps and start_position:  # Reverse failed
            gx, gy, _ = gps.getValues()                                  # If we still ended near start
            cur = (gx, gy)
            traj = GetFromBlackboard("trajectory_points") or []
            if len(traj) > 10 and Distance2D(cur, start_position) < 0.50:
                return Status.SUCCESS                                    # Accept as success
        return status                                                    # Otherwise propagate status

    def terminate(self):
        if self.navigation_controller:                                   # Stop controller on BT shutdown
            self.navigation_controller.terminate()

class RunInBackground(BehaviorNode):
    def __init__(self, child):
        super().__init__("RunInBackground")                            
        self.child = child                                             

    def execute(self):
        if GetFromBlackboard("cspace_frozen", False):                   # If map is frozen, keep child running
            return Status.RUNNING
        child_status = self.child.tick()                                # Tick child without blocking
        return Status.RUNNING if child_status in (Status.SUCCESS, Status.RUNNING) else Status.FAILURE  # Only fail if child failed

class EnableCspaceDisplay(BehaviorNode):
    def __init__(self):
        super().__init__("EnableCspaceDisplay")                          # Name the node

    def execute(self):
        blackboard.Set(BBKey.ALLOW_CSPACE_DISPLAY, True)                 # Allow c-space visualization in UI
        blackboard.Set("survey_complete", True)                          # Mark survey as complete
        return Status.SUCCESS                                           

class OnlyOnce(BehaviorNode):
    def __init__(self, child, key_suffix=None):
        super().__init__(f"OnlyOnce({child.name})")                      # Include child name for clarity
        self.child = child                                               # Wrapped node
        self.flag_key = f"onlyonce:{key_suffix or child.name}"           # Blackboard key used to remember run

    def execute(self):
        if GetFromBlackboard(self.flag_key, False):                      # Already executed once?
            return Status.SUCCESS                                        # Return SUCCESS to keep selector on this branch
        res = self.child.tick()                                          # Otherwise, tick child
        if res in (Status.SUCCESS, Status.FAILURE):                      # When child finishes
            blackboard.Set(self.flag_key, True)                          # Set the flag
        return res                                                    

    def reset(self):
        super().reset()                                                  # Reset decorator state
        blackboard.Set(self.flag_key, False)                             # Clear once-flag
        if hasattr(self.child, 'reset'):
            self.child.reset()                                           # Reset child if supported


###############################################################################
# ================================== Main =====================================
###############################################################################
if __name__ == "__main__":
    robot, timestep, gps, compass, lidar, motor_left, motor_right, display = InitAllDevices()  # Init devices

    for k, v in [
        ("stop_mapping", False),                                        # Stop mapping loop
        ("emergency_stop", False),                                      # Global emergency stop
        ("max_mapping_steps", 5000),                                    # Mapping iterations
        ("map_ready", False),                                           # Map built and usable?
        ("map_saved", False),                                           # Map saved to disk?
        ("trajectory_points", []),                                      # Points for UI trail
        ("display_mode", "full"),                                       # Full/cspace/planning
        ("allow_cspace_display", False),                                # Allow c-space view
        ("cspace_frozen", False),                                       # Freeze c-space edits
        ("survey_complete", False),                                     # Survey done?
    ]:
        blackboard.Set(k, v)                                            # Seed defaults
    SetLogLevel(LogLevel.INFO)                                          # Set log level
    start = gps.getValues()                                             # Initial pose
    start_xy = (start[0], start[1])                    
    blackboard.Set(BBKey.START_XY, start_xy)                            # Store start
    cspace = GetFromBlackboard("cspace")                                # Preexisting map?
    INNER_12 = OrderFromStart(BuildEllipsePoints(), start_xy, close_loop=True)      # Inner loop waypoints
    OUTER_PERIM = OrderFromStart(BuildPerimeterLoop(), start_xy, close_loop=True)   # Outer loop waypoints
    if cspace is not None:                                              # Clamp to safe bounds
        INNER_12 = [ClampGoal(x, y, cspace) for x, y in INNER_12]
        OUTER_PERIM = [ClampGoal(x, y, cspace) for x, y in OUTER_PERIM]
    WAYPOINTS_SURVEY_INNER = INNER_12[:-1]                              # Drop duplicate last point
    WAYPOINTS_SURVEY_OUTER = OUTER_PERIM[:-1]                           # Kept for parity
    mapping_params = MappingParams(world_to_grid=WorldToGrid, grid_to_world=GridToWorld)  # Map params
    planning_params = PlanningParams()                                  # Planner params
    check_map = MapExistsOrReady()                              
    load_map = LoadMap()                              
    validate_loaded_map = ValidateLoadedMap()         
    lidar_mapping = LidarMappingBT(mapping_params, bb=blackboard)       # Build/refresh c-space
    bidir_survey = BidirectionalSurveyNavigator(WAYPOINTS_SURVEY_INNER)  
    navigate_to_waypoints = NavigateToWaypoints()                       # Follow planned path
    set_two_safe_goals = SetTwoGoals(goals=None, num_goals=2)          
    plan_then_go = Sequence("PlanThenGo", [                          
        set_two_safe_goals,
        MultiGoalPlannerBT(planning_params, bb=blackboard),
        navigate_to_waypoints,
    ])
    mapping_background = RunInBackground(lidar_mapping)               # Decorator: map in background
    mapping_and_survey_parallel = Parallel(                           # Parallel: survey + mapping
        "SurveyWithBackgroundMapping",
        [bidir_survey, mapping_background],
        success_threshold=1,                                          # Succeed if any child succeeds
        failure_threshold=None,                                       # Don't fail based on a single child
    )
    complete_mapping_sequence = Sequence("CompleteMappingSequence", [ # Full flow when no saved map
        mapping_and_survey_parallel,                                  # 1) Survey while mapping
        EnsureCspaceNow(blackboard_instance=blackboard),              # 2) Ensure c-space exists
        WaitForMapReady(),                                            # 3) Wait for map
        SaveMap(),                                                    # 4) Save map to disk
        validate_loaded_map,                                          # 5) Validate map
        ValidateAndVisualizeWaypoints(),                              # 6) Save viz overlays
        EnableCspaceDisplay(),                                        # 7) Allow c-space UI
        SetDisplayMode("cspace"),                                     # 8) Switch UI
        plan_then_go,                                                 # 9) Plan+go to goals
    ])
    use_existing_map_inner = Sequence("UseExistingMap", [             # Fast path if map exists
        check_map,
        load_map,
        validate_loaded_map,
        EnableCspaceDisplay(),
        SetDisplayMode("cspace"),
        plan_then_go,
    ])
    use_existing_map = OnlyOnce(use_existing_map_inner, "existing_map")  # Run fast path at most once
    main_mission_tree = Selector("MainMissionTree", [use_existing_map, complete_mapping_sequence])  # Choose path
    display_updater = DisplayUpdater()                                # Node: update UI regularly
    main_execution_tree = Parallel("MainWithDisplay", [main_mission_tree, display_updater], 
                                   success_threshold=2, failure_threshold=2)  # Mission + UI - never finishes

    last_state = None                                                 # Track state changes
    last_log_time = 0                                                 # Track last periodic log time
    try:
        while robot.step(timestep) != -1:                             # Main control loop
            state = main_execution_tree.tick()                        # Step behavior tree
            t = robot.getTime()
            if state != last_state:                                   # Log on change
                if state == Status.FAILURE:                           # Mission failed
                    main_logger.Error(f"Mission failed at t={t:.1f}s.")
                    main_execution_tree.terminate()
                    break
                elif state == Status.SUCCESS:                         # Mission succeeded
                    main_logger.Info(f"Mission completed at t={t:.1f}s.")
                last_state = state
    except KeyboardInterrupt:
        pass                                                          # Allow Ctrl+C to exit
    except Exception as e:
        main_logger.Error(f"Unhandled exception in main loop: {e}")   # Catch-all log
    finally:
        main_execution_tree.terminate()                               # Stop controllers
