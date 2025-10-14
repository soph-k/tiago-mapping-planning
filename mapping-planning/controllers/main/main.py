from __future__ import annotations
from controller import Supervisor
import numpy as np
from os.path import exists

###############################################################################
# ============================== Helpers ======================================
###############################################################################
def GetFromBlackboard(key, default=None):   # Fetch a value from a global blackboard store
    return blackboard.Get(key, default)

def CalculateFreeSpacePercentage(cspace: np.ndarray) -> float:    # % of cells above free threshold
    free_cells = float((cspace > TH_FREE_PLANNER).sum())          # Count free cells
    return free_cells / float(cspace.size)                        # Normalize by total cells

def MapToDisplayCoords(row: int, col: int, map_shape: tuple, w: int, h: int) -> tuple[int, int]:
    return int(row * w / map_shape[0]), int(col * h / map_shape[1]) # Scale indices to display size


################################################################################
# ========================= Device Initialization ==============================
################################################################################
def GetDeviceNames(robot) -> set[str]:   # Collect device names from the robot
    try:
        return {robot.getDeviceByIndex(i).getName() for i in range(robot.getNumberOfDevices())}
    except Exception:
        return set()                                                # If failed, return empty set

def InitRobot():                                                    # Create Supervisor and basic timestep
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())                        # Simulation timestep
    blackboard.Set(BBKey.INIT_Z, robot.getSelf().getPosition()[2])
    return robot, timestep

def InitSensors(robot, timestep):                                   # Enable GPS/compass
    gps, compass = robot.getDevice('gps'), robot.getDevice('compass')   # Get sensor handles
    if gps:
        gps.enable(timestep)
    if compass:
        compass.enable(timestep)

    available = GetDeviceNames(robot)
    lidar = None
    for name in ("Hokuyo URG-04LX-UG01", "lidar", "laser", "Hokuyo", "urg04lx"):
        if name in available:
            try:
                lidar = robot.getDevice(name)
                lidar.enable(timestep)
                blackboard.Set(BBKey.LIDAR, lidar)
                break
            except Exception:
                pass
    if not lidar:
        main_logger.Warning("Lidar not detected")
    if not (gps and compass):
        main_logger.Error("Sensor missing")
    return gps, compass, lidar


def InitMotors(robot):
    available = GetDeviceNames(robot)
    pick = lambda *c: next((n for n in c if n in available), None)
    left_name = pick("wheel_left_joint", "wheel_left_motor", "left_wheel_joint", "left_wheel_motor")
    right_name = pick("wheel_right_joint", "wheel_right_motor", "right_wheel_joint", "right_wheel_motor")
    if not left_name or not right_name:
        if not left_name:
            main_logger.Error("Left motor not found.")
        if not right_name:
            main_logger.Error("Right motor not found.")
        return None, None
    left = robot.getDevice(left_name)
    right = robot.getDevice(right_name)
    try:
        for m in (left, right):
            m.setPosition(float('inf'))
            m.setVelocity(0.0)
    except Exception:
        pass
    return left, right

def InitDisplay(robot):
    try:
        disp = robot.getDevice("display")
        if not disp:
            main_logger.Warning("Display error, not correct dimension")
            return None
        blackboard.Set(BBKey.DISPLAY, disp)
        GetDisplayManager().InitializeDisplay()
        return disp
    except Exception as e:
        main_logger.Error(f"Display init error: {e}")
        return None

def RegisterDevices(robot, timestep, gps, compass, lidar, left_motor, right_motor, display):
    for k, v in (
        (BBKey.ROBOT, robot),
        (BBKey.GPS, gps),
        (BBKey.COMPASS, compass),
        (BBKey.LIDAR, lidar),
        (BBKey.MOTOR_L, left_motor),
        (BBKey.MOTOR_R, right_motor),
        (BBKey.DISPLAY, display),
        (BBKey.TIMESTEP, timestep),
        ("world_to_grid", WorldToGrid),
        ("grid_to_world", GridToWorld),
        ("normalize_angle", NormalizeAngle),
        ("distance_2d", Distance2D),
    ):
        blackboard.Set(k, v)
    missing = [n for n, d in (("GPS", gps), ("Compass", compass), ("LiDAR", lidar)) if not d]
    if missing:
        main_logger.Warning(f"Missing sensors.")

def InitAllDevices():
    robot, timestep = InitRobot()
    gps, compass, lidar = InitSensors(robot, timestep)
    left_motor, right_motor = InitMotors(robot)
    display = InitDisplay(robot)
    RegisterDevices(robot, timestep, gps, compass, lidar, left_motor, right_motor, display)
    return robot, timestep, gps, compass, lidar, left_motor, right_motor, display


################################################################################
# ============================ Display Manager =================================
################################################################################
class DisplayManager:
    COLOR_BLACK = 0x000000
    COLOR_WHITE = 0xFFFFFF
    COLOR_BLUE = 0x0000FF
    COLOR_RED = 0xFF0000
    COLOR_GREEN = 0x00FF00
    COLOR_YELLOW = 0xFFFF00
    COLOR_CYAN = 0x00FFFF
    COLOR_HOT_PINK = 0xFF69B4

    PROBMAP_MIN_DRAW = 0.35
    PROBMAP_MAX_SHOWN = 0.75
    PROBMAP_GAMMA = 0.85

    def __init__(self, display=None):
        self.display = display or GetFromBlackboard("display")
        self.width = self.display.getWidth() if self.display else 0
        self.height = self.display.getHeight() if self.display else 0

    def InitializeDisplay(self):
        if not self.display:
            self.display = GetFromBlackboard("display")
            if not self.display:
                return False
        self.width, self.height = self.display.getWidth(), self.display.getHeight()
        self.ClearDisplay()
        return True

    def ClearDisplay(self):
        if self.display:
            self.display.setColor(self.COLOR_BLACK)
            self.display.fillRectangle(0, 0, self.width, self.height)

    def GetMapShape(self) -> tuple[int, int]:
        cspace = GetFromBlackboard("cspace")
        if cspace is not None:
            return cspace.shape
        prob_map = GetFromBlackboard("prob_map")
        if prob_map is not None:
            return prob_map.shape
        return (200, 300)

    def MapToDisplay(self, row, col, shape):
        return MapToDisplayCoords(row, col, shape, self.width, self.height)

    def _in_bounds(self, x, y, w=2, h=2):
        return 0 <= x < self.width - w and 0 <= y < self.height - h
    
    def DrawPixel(self, x, y, color):
        if self._in_bounds(x, y):
            self.display.setColor(color)
            self.display.fillRectangle(x, y, 2, 2)

    def DrawWorldLine(self, start_point, end_point, shape, color):
        if not self.display:
            return
        sx, sy = self.MapToDisplay(*WorldToGrid(*start_point, shape), shape)
        ex, ey = self.MapToDisplay(*WorldToGrid(*end_point, shape), shape)
        self.display.setColor(color)
        self.display.drawLine(sx, sy, ex, ey)

    def DrawProbabilityMap(self, probability_map=None):
        if not self.display:
            return
        probability_map = GetFromBlackboard("prob_map") if probability_map is None else probability_map
        if probability_map is None:
            return
        p = np.clip(probability_map, self.PROBMAP_MIN_DRAW, self.PROBMAP_MAX_SHOWN)
        p = (p - self.PROBMAP_MIN_DRAW) / (self.PROBMAP_MAX_SHOWN - self.PROBMAP_MIN_DRAW)
        p = p ** self.PROBMAP_GAMMA
        h, w = p.shape
        step = 2
        rows, cols = np.mgrid[0:h:step, 0:w:step]
        intensity = (p[rows, cols] * 255).astype(np.uint8)
        px = (rows * self.width // h).astype(int)
        py = (cols * self.height // w).astype(int)
        bw = max(2, self.width // w)
        bh = max(2, self.height // h)
        valid = (px >= 0) & (px < self.width - bw) & (py >= 0) & (py < self.height - bh)
        vr, vc = np.where(valid)
        if len(vr) > 10000:
            s = max(1, len(vr) // 10000)
            vr, vc = vr[::s], vc[::s]
        for i, j in zip(vr, vc):
            g = int(intensity[i, j])
            rgb = (g << 16) | (g << 8) | g
            self.display.setColor(rgb)
            self.display.fillRectangle(px[i, j], py[i, j], bw, bh)

    def DrawTrajectory(self, trajectory_points=None, map_shape=None, color=None):
        if not self.display:
            return
        traj = trajectory_points or GetFromBlackboard("trajectory_points")
        if not traj or len(traj) < 2:
            return
        shape = map_shape or self.GetMapShape()
        col = color or self.COLOR_HOT_PINK
        for a, b in zip(traj, traj[1:]):
            self.DrawWorldLine(a, b, shape, col)

    def DrawRobotPosition(self, map_shape=None, color=None, size=4):
        if not self.display:
            return
        gps = GetFromBlackboard("gps")
        if not gps:
            return
        wx, wy = gps.getValues()[:2]
        shape = map_shape or self.GetMapShape()
        x, y = self.MapToDisplay(*WorldToGrid(wx, wy, shape), shape)
        if 0 <= x < self.width and 0 <= y < self.height:
            self.display.setColor(color or self.COLOR_BLUE)
            hs = size // 2
            self.display.fillOval(x - hs, y - hs, size, size)

    def DrawPlannedPath(self, path=None, map_shape=None, color=None):
        if not self.display:
            return
        planned = path or GetFromBlackboard("planned_path")
        if not planned:
            return
        shape = map_shape or self.GetMapShape()
        col = color or self.COLOR_CYAN
        for a, b in zip(planned, planned[1:]):
            self.DrawWorldLine(a, b, shape, col)
        self.display.setColor(col)
        for wx, wy in planned:
            x, y = self.MapToDisplay(*WorldToGrid(wx, wy, shape), shape)
            if self._in_bounds(x, y, 2, 2):
                self.display.fillOval(x - 1, y - 1, 2, 2)

    def DrawNavigationGoals(self, goals=None, map_shape=None, color=None):
        if not self.display:
            return
        nav_goals = goals or GetFromBlackboard("navigation_goals")
        if not nav_goals:
            return
        shape = map_shape or self.GetMapShape()
        self.display.setColor(color or self.COLOR_YELLOW)
        for gx, gy in nav_goals:
            x, y = self.MapToDisplay(*WorldToGrid(gx, gy, shape), shape)
            if self._in_bounds(x, y, 3, 3):
                s = 3
                self.display.drawLine(x - s, y - s, x + s, y + s)
                self.display.drawLine(x - s, y + s, x + s, y - s)

    def DrawCspace(self, cspace):
        if not (self.display and cspace is not None):
            return
        h, w = cspace.shape
        for r in range(0, h, 2):
            row = cspace[r]
            for c in range(0, w, 2):
                color = self.COLOR_BLACK if float(row[c]) > 0.5 else self.COLOR_WHITE
                x, y = self.MapToDisplay(r, c, (h, w))
                self.DrawPixel(x, y, color)

    def UpdateDisplay(self, mode="full"):
        if not self.display:
            return
        self.ClearDisplay()
        shape = self.GetMapShape()
        if mode in ("cspace", "planning"):
            cspace = GetFromBlackboard("cspace")
            if cspace is not None:
                self.DrawCspace(cspace)
            if mode == "planning":
                self.DrawPlannedPath(map_shape=shape)
                self.DrawNavigationGoals(map_shape=shape)
                self.DrawRobotPosition(map_shape=shape)
            return
        prob_map = GetFromBlackboard("prob_map")
        if prob_map is not None:
            self.DrawProbabilityMap(prob_map)
        self.DrawTrajectory(map_shape=shape)
        self.DrawRobotPosition(map_shape=shape)
_display_manager = None

def GetDisplayManager():
    global _display_manager
    if _display_manager is None:
        _display_manager = DisplayManager()
    return _display_manager
