from __future__ import annotations
from controller import Supervisor
import numpy as np
from os.path import exists

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
    h, w = shape                                              # Unpack grid shape 
    x_min, y_max = GridToWorld(0, 0)                          # Convert top left grid cell to world coords
    x_max, y_min = GridToWorld(h - 1, w - 1)                  # Convert bottom right grid cell to world coords
    return x_min, y_min, x_max, y_max                         # Return world-space bounds


def BuildEllipsePoints(center=(-0.65, -1.43), rx=1.05, ry=1.25, num_points=12, rotation=0.0):
    cx, cy = center                                           # Center of the world coords
    ang = np.linspace(0, 2 * np.pi, num_points, endpoint=False) + rotation  # Spaced angles with rotation
    return [(cx + rx * np.cos(a),                             # X coordinate
             cy + ry * np.sin(a))                             # Y coordinate
            for a in ang]                                     # Generate one x and y for each angle


def BuildPerimeterLoop(margin=None, include_midpoints=True):
    x_min, y_min, x_max, y_max = WorldBoundsFromConfig()      # Get world bounds of the map
    m = margin if margin is not None else max(0.6, 0.18 + 0.4) # Use provided margin or a safe default
    left, right, bottom, top = x_min + m, x_max - m, y_min + m, y_max - m  # Inset rectangle by margin
    mid_x, mid_y = 0.5 * (left + right), 0.5 * (bottom + top) # Midpoints along x and y
    pts = [(left, bottom)]                                    # Start at bottom left corner
    if include_midpoints: pts += [(mid_x, bottom)]            
    pts += [(right, bottom)]                                  # Bottom right corner
    if include_midpoints: pts += [(right, mid_y)]             
    pts += [(right, top)]                                     # Top-right corner
    if include_midpoints: pts += [(mid_x, top)]               
    pts += [(left, top)]                                      # Top-left corner
    if include_midpoints: pts += [(left, mid_y)]              
    return pts                                                # Return loop as a list of waypoints


def OrderFromStart(points, start_position, close_loop=True):
    sx, sy = start_position                                   # Extract starting world coordinates
    d = [np.hypot(x - sx, y - sy) for (x, y) in points]       # Compute Euclidean distance to each point
    i = int(np.argmin(d))                                     # Find index of nearest point to start position
    ordered = points[i:] + points[:i]                         # Rotate list so nearest point is first
    return ordered + [start_position] if close_loop else ordered  


def ClampGoal(goal_x, goal_y, cspace=None):
    if cspace is None:                                        # If no c-space provided, nothing to clamp
        return goal_x, goal_y                                 # Return goal 
    h, w = cspace.shape                                      
    x_min, y_min = GridToWorld(h - 1, 0)                      # World coords of bottom left grid cell
    x_max, y_max = GridToWorld(0, w - 1)                      # World coords of top right grid cell
    buf = 0.20                                                
    return (                                                  
        min(max(goal_x, x_min + buf), x_max - buf),           # Clamp goal_x between 
        min(max(goal_y, y_min + buf), y_max - buf),           # Clamp goal_y between
    )                                                         # Return clamped goal coordinates


################################################################################
# ========================== Behavior Tree Nodes ===============================
################################################################################
class WaitForMapReady(BehaviorNode):
    def __init__(self):
        super().__init__("WaitForMapReady")
        self._start_time = None
        self._warned = [False] * 3

    def execute(self):
        robot = GetFromBlackboard("robot")
        if robot and self._start_time is None:
            self._start_time = robot.getTime()
        if GetFromBlackboard("map_saved"):
            return Status.SUCCESS
        cspace = GetFromBlackboard("cspace")
        if cspace is None:
            if robot and self._start_time is not None:
                elapsed = robot.getTime() - self._start_time
                msgs = [
                    (120, "Still no map after 120s."),
                    (60, "Map build is slow 60s."),
                    (30, "Waiting for C-space 30s"),
                ]
                for idx, (t, msg) in enumerate(msgs[::-1]):
                    real_idx = 2 - idx
                    if elapsed > t and not self._warned[real_idx]:
                        main_logger.Warning(msg)
                        self._warned[real_idx] = True
            return Status.RUNNING
        return Status.SUCCESS if CalculateFreeSpacePercentage(cspace) >= 0.01 else Status.RUNNING

    def reset(self):
        super().reset()
        self._start_time = None
        self._warned = [False] * 3


class MapExistsOrReady(BehaviorNode):
    def __init__(self, path="cspace.npy"):
        super().__init__("MapExistsOrReady")
        self.path = str(ResolveMapPath(path))

    def execute(self):
        return Status.SUCCESS if (GetFromBlackboard("map_ready", False) or exists(self.path)) else Status.FAILURE


class LoadMap(BehaviorNode):
    def __init__(self, path="cspace.npy"):
        super().__init__("LoadMap")
        self.path = str(ResolveMapPath(path))

    def execute(self):
        try:
            c = np.clip(np.load(self.path).astype(np.float32), 0.0, 1.0)
            blackboard.SetCspace(c)
            blackboard.Set(BBKey.CSPACE_FROZEN, True)
            blackboard.SetMapReady(True)
            return Status.SUCCESS
        except Exception as e:
            main_logger.Error(f"LoadMap failed {e}")
            return Status.FAILURE


class EnsureCspaceNow(BehaviorNode):
    def __init__(self, blackboard_instance=None):
        super().__init__("EnsureCspaceNow")
        self.blackboard = blackboard_instance or blackboard

    def execute(self):
        c = self.blackboard.GetCspace()
        if c is not None:
            return Status.SUCCESS
        p = self.blackboard.GetProbMap()
        if p is None:
            return Status.RUNNING
        try:
            mapper = LidarMappingBT(params=MappingParams(), bb=self.blackboard)
            c = mapper.create_cspace(p)
            if c is not None:
                self.blackboard.SetCspace(c)
                return Status.SUCCESS
            return Status.FAILURE
        except Exception as e:
            main_logger.Error(f"c-space build crashed: {e}")
            return Status.FAILURE


class SaveMap(BehaviorNode):
    def __init__(self, path="cspace.npy", threshold=None):
        super().__init__("SaveMap")
        self.path = str(ResolveMapPath(path))
        self.threshold = 0.30 if threshold is None else threshold
        self.done = False

    def PrepareMapForSaving(self) -> np.ndarray | None:
        c = GetFromBlackboard("cspace")
        if c is None:
            p = GetFromBlackboard("prob_map")
            if p is None:
                main_logger.Error("no map.")
                return None
            mapper = LidarMappingBT(MappingParams())
            c = mapper.create_cspace(p)
            if c is None:
                c = (p <= self.threshold).astype(np.float32)
        return np.clip(c.astype(np.float32), 0.0, 1.0)

    def ShouldSaveMap(self, cspace: np.ndarray) -> bool:
        return CalculateFreeSpacePercentage(cspace) * 100.0 >= 1.0

    def SaveMapToFile(self, cspace: np.ndarray) -> bool:
        try:
            EnsureParentDirectories(self.path)
            np.save(self.path, cspace)
            return True
        except Exception as e:
            main_logger.Error(f"SaveMap: write failed: {e}")
            return False

    def UpdateMapState(self, cspace: np.ndarray):
        for k, v in (("cspace", cspace), ("map_ready", True), ("map_saved", True), ("cspace_frozen", True)):
            blackboard.Set(k, v)
        self.done = True

    def execute(self):
        if self.done:
            return Status.SUCCESS

        c = self.PrepareMapForSaving()
        if c is None:
            return Status.FAILURE

        if not self.ShouldSaveMap(c):
            main_logger.Error("C-space rejected - not saving.")
            return Status.FAILURE

        try:
            from os import remove
            if exists(self.path):
                remove(self.path)
        except Exception:
            pass

        if not self.SaveMapToFile(c):
            return Status.FAILURE

        self.UpdateMapState(c)
        return Status.SUCCESS


class DisplayUpdater(BehaviorNode):
    def __init__(self):
        super().__init__("DisplayUpdater")
        self.manager = GetDisplayManager()
        self.fps = EveryNSeconds(0.25)

    @staticmethod
    def _truthy(v):
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
            if not (self.manager and self.manager.display):
                return Status.RUNNING
            robot = GetFromBlackboard("robot")
            t = robot.getTime() if robot else None
            if t is None or not self.fps.ShouldExecute(t):
                return Status.RUNNING
            display_mode = GetFromBlackboard("display_mode", "full")
            allow_cspace = self._truthy(GetFromBlackboard("allow_cspace_display", False))
            survey_done = self._truthy(GetFromBlackboard("survey_complete", False))
            cspace = GetFromBlackboard("cspace")
            has_cspace = (cspace is not None) and (CalculateFreeSpacePercentage(cspace) >= 0.01)
            if display_mode == "full" and allow_cspace and survey_done and has_cspace:
                blackboard.Set(BBKey.DISPLAY_MODE, "cspace")
                display_mode = "cspace"
            self.manager.UpdateDisplay(mode=display_mode)
        except Exception as e:
            main_logger.Error(f"DisplayUpdater error {e}")
        return Status.RUNNING


class SetDisplayMode(BehaviorNode):
    def __init__(self, mode):
        super().__init__(f"SetDisplayMode({mode})")
        self.mode = mode

    def execute(self):
        blackboard.Set(BBKey.DISPLAY_MODE, self.mode)
        return Status.SUCCESS

