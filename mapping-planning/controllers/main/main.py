from __future__ import annotations
from controller import Supervisor
import numpy as np
from os.path import exists
from core import (
    MappingParams,                 # mapping parameter bag
    PlanningParams,                # planning parameter bag
    main_logger,                   # shared logger for general info
    SetLogLevel,                   # helper to change log level
    LogLevel,                      # enum for log levels
    NormalizeAngle,                # angle wrap helper
    Distance2D,                    # euclidean distance in 2d
    WorldToGrid,                   # world to grid converter
    GridToWorld,                   # grid to world converter
    BehaviorNode,                  # base node type
    Status,                        # node status enum
    Sequence,                      # behavior tree sequence
    Selector,                      # behavior tree selector
    Parallel,                      # behavior tree parallel
    blackboard,                    # global key value store
    EveryNSeconds,                 # rate limiter by time
    BBKey,                         # keys for blackboard
    TH_FREE_PLANNER,               # threshold for free cells
)
from navigation import (
    StopMotors,                    # stop both wheels
    NavigateToWaypoints,           # bt node to follow waypoints
    BidirectionalSurveyNavigator,  # survey forward then reverse
)
from mapping import (
    LidarMappingBT,                # bt node to build map and cspace
    WaitForMapReady,               # bt node that waits for map ok
    MapExistsOrReady,              # bt node that checks saved map
    LoadMap,                       # bt node to load map file
    EnsureCspaceNow,               # bt node to force build cspace
    SaveMap,                       # bt node to save map to disk
    ValidateLoadedMap,             # bt node to sanity check map
)
from planning import (
    MultiGoalPlannerBT,            # bt node that plans to goals
    SetTwoGoals,                   # bt node to pick two goals
    ValidateAndVisualizeWaypoints, # bt node to precompute overlays
)


def GetFromBlackboard(key, default=None):            # small getter for bb
    return blackboard.Get(key, default)

def CalculateFreeSpacePercentage(cspace: np.ndarray) -> float:  # share of free cells
    free_cells = float((cspace > TH_FREE_PLANNER).sum())        # count cells above free
    return free_cells / float(cspace.size)                      # divide by total

def MapToDisplayCoords(row: int, col: int, map_shape: tuple, w: int, h: int) -> tuple[int, int]:
    # map grid cell to display pixel
    return int(row * w / map_shape[0]), int(col * h / map_shape[1])


################################################################################
# ========================= Device Initialization ==============================
################################################################################
# set up robot sensors motors and display
# store all handles in the shared blackboard
def GetDeviceNames(robot) -> set[str]:               # list all device names on robot
    try:
        return {robot.getDeviceByIndex(i).getName() for i in range(robot.getNumberOfDevices())}
    except Exception:
        return set()                                 # give empty set if query fails

def InitRobot():                                     # make supervisor and get timestep
    robot = Supervisor()                             # main webots supervisor
    timestep = int(robot.getBasicTimeStep())         # controller step in ms
    blackboard.Set(BBKey.INIT_Z, robot.getSelf().getPosition()[2])  # save initial z height
    return robot, timestep

def InitSensors(robot, timestep):                    # enable gps compass lidar
    gps = robot.getDevice('gps')                     # gps handle
    compass = robot.getDevice('compass')            # compass handle
    if gps:
        gps.enable(timestep)                         # enable gps at loop step
    if compass:
        compass.enable(timestep)                     # enable compass at loop step
    available = GetDeviceNames(robot)                # set of device names on robot
    lidar = None                                     # will hold chosen lidar
    for name in "Hokuyo URG-04LX-UG01", "lidar", "laser", "Hokuyo", "urg04lx":
        if name in available:                        # if that name exists on robot
            try:
                dev = robot.getDevice(name)          # get the device
                dev.enable(timestep)                 # turn it on
                lidar = dev                          # keep it
                blackboard.Set(BBKey.LIDAR, lidar)   # store in bb for others
                break                                # stop on first match
            except Exception:
                pass                                 # try next name if any error
    if not lidar:
        main_logger.Warning("Lidar not detected")    # warn if none found
    if not (gps and compass):
        main_logger.Error("Sensor missing")          # must have both gps and compass
    return gps, compass, lidar

def InitMotors(robot):                               # find left and right wheel motors
    available = GetDeviceNames(robot)                # names present
    pick = lambda *c: next((n for n in c if n in available), None)  # first match helper
    left_name = pick("wheel_left_joint", "wheel_left_motor", "left_wheel_joint", "left_wheel_motor")
    right_name = pick("wheel_right_joint", "wheel_right_motor", "right_wheel_joint", "right_wheel_motor")
    if not left_name or not right_name:              # if either is missing
        if not left_name:
            main_logger.Error("Left motor not found.")
        if not right_name:
            main_logger.Error("Right motor not found.")
        return None, None
    left = robot.getDevice(left_name)                # left motor handle
    right = robot.getDevice(right_name)              # right motor handle
    try:
        for m in (left, right):
            m.setPosition(float('inf'))              # velocity control mode
            m.setVelocity(0.0)                       # start with zero speed
    except Exception:
        pass                                         # ignore if motor api not ready
    return left, right

def InitDisplay(robot):                              # set up on screen drawing
    try:
        disp = robot.getDevice("display")            # fetch display device
        if not disp:
            main_logger.Warning("Display error not correct dimension")  # hint user scene issue
            return None
        blackboard.Set(BBKey.DISPLAY, disp)          # store handle for later draws
        GetDisplayManager().InitializeDisplay()      # clear and size the canvas
        return disp
    except Exception as e:
        main_logger.Error(f"Display init error: {e}")  # log any failure
        return None

def RegisterDevices(robot, timestep, gps, compass, lidar, left_motor, right_motor, display):
    # push all handles and helpers into bb
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
    # warn if any core sensor is missing
    missing = [n for n, d in (("GPS", gps), ("Compass", compass), ("LiDAR", lidar)) if not d]
    if missing:
        main_logger.Warning("Missing sensors.")

def InitAllDevices():
    # one call to set up everything
    robot, timestep = InitRobot()
    gps, compass, lidar = InitSensors(robot, timestep)
    left_motor, right_motor = InitMotors(robot)
    display = InitDisplay(robot)
    RegisterDevices(robot, timestep, gps, compass, lidar, left_motor, right_motor, display)
    return robot, timestep, gps, compass, lidar, left_motor, right_motor, display


################################################################################
# ============================ Display Manager =================================
################################################################################
# draw maps paths and markers on the screen
# keep calls light and rate limited
class DisplayManager:
    COLOR_BLACK = 0x000000
    COLOR_WHITE = 0xFFFFFF
    COLOR_BLUE  = 0x0000FF
    COLOR_RED   = 0xFF0000
    COLOR_GREEN = 0x00FF00
    COLOR_YELLOW= 0xFFFF00
    COLOR_CYAN  = 0x00FFFF
    COLOR_HOT_PINK = 0xFF69B4

    PROBMAP_MIN_DRAW = 0.35         # low clamp for map draw
    PROBMAP_MAX_SHOWN = 0.75        # high clamp for map draw
    PROBMAP_GAMMA = 0.85            # gamma to boost contrast

    def __init__(self, display=None):
        self.display = display or GetFromBlackboard("display")  # get from bb if not passed
        self.width = self.display.getWidth() if self.display else 0    # cache width
        self.height = self.display.getHeight() if self.display else 0  # cache height

    def InitializeDisplay(self):
        if not self.display:
            self.display = GetFromBlackboard("display")   # try again
            if not self.display:
                return False                              # cannot draw without display
        self.width  = self.display.getWidth()            # refresh size in case it changed
        self.height = self.display.getHeight()
        self.ClearDisplay()                               # start with clean screen
        return True

    def ClearDisplay(self):
        if self.display:
            self.display.setColor(self.COLOR_BLACK)       # choose black
            self.display.fillRectangle(0, 0, self.width, self.height)  # fill full area

    def GetMapShape(self) -> tuple[int, int]:
        # prefer cspace for shape then prob map then default
        cspace = GetFromBlackboard("cspace")
        if cspace is not None:
            return cspace.shape
        prob_map = GetFromBlackboard("prob_map")
        if prob_map is not None:
            return prob_map.shape
        return 200, 300

    def MapToDisplay(self, row, col, shape):
        # convert grid to pixel coords
        return MapToDisplayCoords(row, col, shape, self.width, self.height)

    def _in_bounds(self, x, y, w=2, h=2):
        # make sure we can draw inside screen
        return 0 <= x <= self.width - w and 0 <= y <= self.height - h

    def DrawPixel(self, x, y, color):
        if self._in_bounds(x, y):
            self.display.setColor(color)
            self.display.fillRectangle(x, y, 2, 2)        # draw a small square

    def DrawWorldLine(self, start_point, end_point, shape, color):
        if not self.display:
            return
        sx, sy = self.MapToDisplay(*WorldToGrid(*start_point, shape), shape)  # start pixel
        ex, ey = self.MapToDisplay(*WorldToGrid(*end_point, shape), shape)    # end pixel
        self.display.setColor(color)
        self.display.drawLine(sx, sy, ex, ey)                 # draw line on canvas

    def DrawProbabilityMap(self, probability_map=None):
        if not self.display:
            return
        probability_map = GetFromBlackboard("prob_map") if probability_map is None else probability_map
        if probability_map is None:
            return                                            # nothing to draw
        p = np.clip(probability_map, self.PROBMAP_MIN_DRAW, self.PROBMAP_MAX_SHOWN)  # clamp
        p = (p - self.PROBMAP_MIN_DRAW) / (self.PROBMAP_MAX_SHOWN - self.PROBMAP_MIN_DRAW)  # scale 0 to 1
        p = p ** self.PROBMAP_GAMMA                           # apply gamma
        h, w = p.shape                                        # grid size
        step = 2                                              # skip every other cell
        rows, cols = np.mgrid[0:h:step, 0:w:step]             # sample grid
        intensity = (p[rows, cols] * 255).astype(np.uint8)    # 0 to 255
        px = (rows * self.width  // h).astype(int)            # x pixels
        py = (cols * self.height // w).astype(int)            # y pixels
        bw = max(2, self.width  // w)                         # block width
        bh = max(2, self.height // h)                         # block height
        valid = (px >= 0) & (px <= self.width - bw) & (py >= 0) & (py <= self.height - bh)  # inside screen
        vr, vc = np.where(valid)                              # indices that fit
        if len(vr) > 10000:                                   # trim if too many blocks
            s = max(1, len(vr) // 10000)
            vr, vc = vr[::s], vc[::s]
        for i, j in zip(vr, vc):
            g = int(intensity[i, j])                          # gray value
            rgb = (g << 16) | (g << 8) | g                    # pack to rgb
            self.display.setColor(rgb)
            self.display.fillRectangle(px[i, j], py[i, j], bw, bh)

    def DrawTrajectory(self, trajectory_points=None, map_shape=None, color=None):
        if not self.display:
            return
        traj = trajectory_points or GetFromBlackboard("trajectory_points")  # list of world points
        if not traj or len(traj) < 2:
            return
        shape = map_shape or self.GetMapShape()
        col = color or self.COLOR_HOT_PINK
        for a, b in zip(traj, traj[1:]):                     # draw line between pairs
            self.DrawWorldLine(a, b, shape, col)

    def DrawRobotPosition(self, map_shape=None, color=None, size=4):
        if not self.display:
            return
        gps = GetFromBlackboard("gps")
        if not gps:
            return
        wx, wy = gps.getValues()[:2]                         # current world position
        shape = map_shape or self.GetMapShape()
        x, y = self.MapToDisplay(*WorldToGrid(wx, wy, shape), shape)  # convert to pixel
        if 0 <= x < self.width and 0 <= y < self.height:
            self.display.setColor(color or self.COLOR_BLUE)
            hs = size // 2
            self.display.fillOval(x - hs, y - hs, size, size)         # small dot

    def DrawPlannedPath(self, path=None, map_shape=None, color=None):
        if not self.display:
            return
        planned = path or GetFromBlackboard("planned_path")
        if not planned:
            return
        shape = map_shape or self.GetMapShape()
        col = color or self.COLOR_CYAN
        for a, b in zip(planned, planned[1:]):               # draw each segment
            self.DrawWorldLine(a, b, shape, col)
        self.display.setColor(col)
        for wx, wy in planned:                               # draw small dot at each wp
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
                self.display.drawLine(x - s, y - s, x + s, y + s)     # cross marker
                self.display.drawLine(x - s, y + s, x + s, y - s)

    def DrawCspace(self, cspace):
        if not (self.display and cspace is not None):
            return
        h, w = cspace.shape
        for r in range(0, h, 2):                           # skip rows for speed
            row = cspace[r]
            for c in range(0, w, 2):                       # skip cols for speed
                color = self.COLOR_BLACK if float(row[c]) > 0.5 else self.COLOR_WHITE
                x, y = self.MapToDisplay(r, c, (h, w))
                self.DrawPixel(x, y, color)

    def UpdateDisplay(self, mode="full"):
        if not self.display:
            return
        self.ClearDisplay()                                 # fresh frame
        shape = self.GetMapShape()
        if mode in ("cspace", "planning"):
            cspace = GetFromBlackboard("cspace")
            if cspace is not None:
                self.DrawCspace(cspace)                     # show free and blocked
            if mode == "planning":
                self.DrawPlannedPath(map_shape=shape)       # show path
                self.DrawNavigationGoals(map_shape=shape)   # show goals
                self.DrawRobotPosition(map_shape=shape)     # show robot marker
            return
        prob_map = GetFromBlackboard("prob_map")
        if prob_map is not None:
            self.DrawProbabilityMap(prob_map)               # gray map background
        self.DrawTrajectory(map_shape=shape)                # trail
        self.DrawRobotPosition(map_shape=shape)             # robot dot

_display_manager = None
def GetDisplayManager():
    # singleton maker for display manager
    global _display_manager
    if _display_manager is None:
        _display_manager = DisplayManager()
    return _display_manager


################################################################################
# ============================ Waypoint Generation =============================
#################################################################################
# make simple waypoint sets for survey and tests
# keep points inside safe world bounds
def WorldBoundsFromConfig(shape=(200, 300)):
    h, w = shape
    x_min, y_max = GridToWorld(0, 0)                        # top left world
    x_max, y_min = GridToWorld(h - 1, w - 1)                # bottom right world
    return x_min, y_min, x_max, y_max

def BuildEllipsePoints(center=(-0.65, -1.43), rx=1.05, ry=1.25, num_points=12, rotation=0.0):
    cx, cy = center                                         # center of ellipse
    ang = np.linspace(0, 2 * np.pi, num_points, endpoint=False) + rotation  # angle list
    # produce points around ellipse
    return [(cx + rx * np.cos(a), cy + ry * np.sin(a)) for a in ang]

def BuildPerimeterLoop(margin=None, include_midpoints=True):
    x_min, y_min, x_max, y_max = WorldBoundsFromConfig()    # world rectangle
    m = margin if margin is not None else max(0.6, 0.18 + 0.4)  # safe offset from walls
    left, right  = x_min + m, x_max - m                      # inner left right
    bottom, top  = y_min + m, y_max - m                      # inner bottom top
    mid_x, mid_y = 0.5 * (left + right), 0.5 * (bottom + top)
    pts = [(left, bottom)]                                   # start bottom left
    if include_midpoints: pts += [(mid_x, bottom)]           # mid bottom
    pts += [(right, bottom)]                                 # bottom right
    if include_midpoints: pts += [(right, mid_y)]            # mid right
    pts += [(right, top)]                                    # top right
    if include_midpoints: pts += [(mid_x, top)]              # mid top
    pts += [(left, top)]                                     # top left
    if include_midpoints: pts += [(left, mid_y)]             # mid left
    return pts

def OrderFromStart(points, start_position, close_loop=True):
    sx, sy = start_position                                  # start world pos
    d = [np.hypot(x - sx, y - sy) for (x, y) in points]      # distance to each
    i = int(np.argmin(d))                                    # index of nearest
    ordered = points[i:] + points[:i]                        # rotate so nearest first
    return ordered + [start_position] if close_loop else ordered

def ClampGoal(goal_x, goal_y, cspace=None):
    if cspace is None:                                       # nothing to clamp
        return goal_x, goal_y
    h, w = cspace.shape
    x_min, y_min = GridToWorld(h - 1, 0)                     # world min corner
    x_max, y_max = GridToWorld(0, w - 1)                     # world max corner
    buf = 0.20                                               # keep off the edge
    return (
        min(max(goal_x, x_min + buf), x_max - buf),
        min(max(goal_y, y_min + buf), y_max - buf),
    )


################################################################################
# ========================== Behavior Tree Nodes ===============================
################################################################################
# small helper nodes for ui and flow
# keep logic easy to follow
class DisplayUpdater(BehaviorNode):
    def __init__(self):
        super().__init__("DisplayUpdater")
        self.manager = GetDisplayManager()                    # renderer helper
        self.fps = EveryNSeconds(0.25)                        # 4 hz update cap

    @staticmethod
    def _truthy(v):
        # treat arrays and containers as true when non empty
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
            if not (self.manager and self.manager.display):   # need display ready
                return Status.RUNNING
            robot = GetFromBlackboard("robot")                # for time source
            t = robot.getTime() if robot else None
            if t is None or not self.fps.ShouldExecute(t):    # respect rate cap
                return Status.RUNNING
            display_mode = GetFromBlackboard("display_mode", "full")  # current mode
            allow_cspace = self._truthy(GetFromBlackboard("allow_cspace_display", False))
            survey_done  = self._truthy(GetFromBlackboard("survey_complete", False))
            cspace = GetFromBlackboard("cspace")              # current cspace grid
            has_cspace = (cspace is not None) and (CalculateFreeSpacePercentage(cspace) >= 0.01)
            # auto switch to cspace mode when allowed and ready
            if display_mode == "full" and allow_cspace and survey_done and has_cspace:
                blackboard.Set(BBKey.DISPLAY_MODE, "cspace")
                display_mode = "cspace"
            self.manager.UpdateDisplay(mode=display_mode)     # draw frame
        except Exception as e:
            main_logger.Error(f"DisplayUpdater error {e}")    # log then continue
        return Status.RUNNING

class SetDisplayMode(BehaviorNode):
    def __init__(self, mode):
        super().__init__(f"SetDisplayMode({mode})")
        self.mode = mode                                      # new mode string

    def execute(self):
        blackboard.Set(BBKey.DISPLAY_MODE, self.mode)         # set in bb
        return Status.SUCCESS

class RunInBackground(BehaviorNode):
    def __init__(self, child):
        super().__init__("RunInBackground")
        self.child = child                                     # wrapped node

    def execute(self):
        if GetFromBlackboard("cspace_frozen", False):          # when frozen keep child alive
            return Status.RUNNING
        child_status = self.child.tick()                       # tick child once
        # keep running unless child failed
        return Status.RUNNING if child_status in (Status.SUCCESS, Status.RUNNING) else Status.FAILURE

class EnableCspaceDisplay(BehaviorNode):
    def __init__(self):
        super().__init__("EnableCspaceDisplay")

    def execute(self):
        blackboard.Set(BBKey.ALLOW_CSPACE_DISPLAY, True)       # permit cspace on ui
        blackboard.Set("survey_complete", True)                # mark survey done
        main_logger.Info("C-space display enabled switching to c-space view")
        return Status.SUCCESS

class OnlyOnce(BehaviorNode):
    def __init__(self, child, key_suffix=None):
        super().__init__(f"OnlyOnce({child.name})")
        self.child = child                                     # target node
        self.flag_key = f"onlyonce:{key_suffix or child.name}" # bb flag name

    def execute(self):
        if GetFromBlackboard(self.flag_key, False):            # already ran
            return Status.SUCCESS
        res = self.child.tick()                                # run child
        if res in (Status.SUCCESS, Status.FAILURE):            # once it ends
            blackboard.Set(self.flag_key, True)                # remember done
        return res

    def reset(self):
        super().reset()
        blackboard.Set(self.flag_key, False)                   # clear flag
        if hasattr(self.child, 'reset'):
            self.child.reset()                                 # also reset child


###############################################################################
# ================================== Main =====================================
###############################################################################
# wire up devices build the tree and run
# stop cleanly on finish or error
if __name__ == "__main__":
    # create all hardware interfaces
    robot, timestep, gps, compass, lidar, motor_left, motor_right, display = InitAllDevices()

    # seed shared flags and defaults
    for k, v in [
        ("map_ready", False),
        ("map_saved", False),
        ("trajectory_points", []),
        ("display_mode", "full"),
        ("allow_cspace_display", False),
        ("cspace_frozen", False),
        ("survey_complete", False),
    ]:
        blackboard.Set(k, v)

    SetLogLevel(LogLevel.INFO)                            # default info logs
    main_logger.Info("Starting Tiago mapping and navigation system")

    # capture start position
    start = gps.getValues()                               # vector from webots
    start_xy = (start[0], start[1])                       # x and y
    blackboard.Set(BBKey.START_XY, start_xy)              # store for later
    main_logger.Info(f"Robot starting position: ({start_xy[0]:.2f}, {start_xy[1]:.2f})")

    # check if a map already exists
    cspace = GetFromBlackboard("cspace")
    if cspace is not None:
        main_logger.Info("Using existing map")
    else:
        main_logger.Info("No existing map found will create new map")

    # build waypoint sets around start
    INNER_12   = OrderFromStart(BuildEllipsePoints(), start_xy, close_loop=True)   # inner ring
    OUTER_PERIM= OrderFromStart(BuildPerimeterLoop(), start_xy, close_loop=True)   # outer box
    if cspace is not None:
        INNER_12    = [ClampGoal(x, y, cspace) for x, y in INNER_12]               # keep inside safe area
        OUTER_PERIM = [ClampGoal(x, y, cspace) for x, y in OUTER_PERIM]
    WAYPOINTS_SURVEY_INNER = INNER_12[:-1]     # drop duplicate last point
    WAYPOINTS_SURVEY_OUTER = OUTER_PERIM[:-1]  # kept for parity if needed

    # prepare parameter objects for subsystems
    mapping_params  = MappingParams(world_to_grid=WorldToGrid, grid_to_world=GridToWorld)
    planning_params = PlanningParams()

    ################################################################################
    # ========================== BEHAVIOR TREE CONSTRUCTION =======================
    ################################################################################
    # build leaves build sequences then combine into the root
    # run display in parallel so ui stays live

    # leaf nodes
    check_map            = MapExistsOrReady()                         # quick check for saved map file
    load_map             = LoadMap()                                  # try to load file
    validate_loaded_map  = ValidateLoadedMap()                        # sanity check after load
    lidar_mapping        = LidarMappingBT(mapping_params, bb=blackboard)  # live mapping worker
    bidir_survey         = BidirectionalSurveyNavigator(WAYPOINTS_SURVEY_INNER)  # drive inner loop both ways
    navigate_to_waypoints= NavigateToWaypoints()                      # run the navigator
    set_two_safe_goals   = SetTwoGoals(goals=None, num_goals=2)       # choose two goals automatically

    # plan then drive sequence
    plan_then_go = Sequence("PlanThenGo", [
        set_two_safe_goals,                                   # choose two safe goals
        MultiGoalPlannerBT(planning_params, bb=blackboard),   # build a path to one of them
        navigate_to_waypoints,                                 # follow the path
    ])

    # background mapping while doing survey
    mapping_background = RunInBackground(lidar_mapping)

    # full mapping route when no saved map
    mapping_and_survey_parallel = Parallel(
        "SurveyWithBackgroundMapping",
        [bidir_survey, mapping_background],                    # survey and map together
        success_threshold=1,                                   # succeed when survey ends
        failure_threshold=None,                                # do not fail on mapping alone
    )

    complete_mapping_sequence = Sequence("CompleteMappingSequence", [
        mapping_and_survey_parallel,                           # collect scans while touring
        EnsureCspaceNow(blackboard_instance=blackboard),       # make sure cspace exists
        WaitForMapReady(),                                     # wait until map is usable
        SaveMap(),                                             # write to disk
        validate_loaded_map,                                   # verify result
        ValidateAndVisualizeWaypoints(),                       # precompute overlays on map
        EnableCspaceDisplay(),                                 # allow cspace view in ui
        SetDisplayMode("cspace"),                              # switch screen to cspace
        plan_then_go,                                          # then plan and go to goals
    ])

    # fast path when file exists
    use_existing_map_inner = Sequence("UseExistingMap", [
        check_map,                                             # quick test for ready or file
        load_map,                                              # load from disk
        validate_loaded_map,                                   # confirm it looks ok
        EnableCspaceDisplay(),                                 # show cspace
        SetDisplayMode("cspace"),                              # switch ui mode
        plan_then_go,                                          # plan and drive
    ])
    use_existing_map = OnlyOnce(use_existing_map_inner, "existing_map")  # run this branch once

    # top selector chooses fast path first then full mapping if needed
    main_mission_tree = Selector("MainMissionTree", [
        use_existing_map,
        complete_mapping_sequence,
    ])

    # display updater runs forever
    display_updater = DisplayUpdater()

    # root parallel keeps ui alive while mission runs
    main_execution_tree = Parallel("MainWithDisplay", [
        main_mission_tree,
        display_updater,
    ], success_threshold=2, failure_threshold=2)               # both keep running

    last_state = None                                          # to detect state changes
    last_log_time = 0                                          # reserved for periodic logs
    mission_completed = False                                  # set once mission done

    try:
        while robot.step(timestep) != -1:                      # main control loop
            state = main_execution_tree.tick()                 # tick whole tree
            t = robot.getTime()                                # current sim time

            # check mission branch alone so we can exit when it finishes
            mission_state = main_mission_tree.tick()
            if mission_state == Status.SUCCESS and not mission_completed:
                mission_completed = True
                main_logger.Info(f"Mission completed successfully at t={t:.1f}s.")
                motor_left, motor_right = blackboard.GetMotors()   # get wheels
                if motor_left and motor_right:
                    StopMotors(motor_left, motor_right)            # halt robot
                main_logger.Info("Motors stopped. Mission finished.")
                break                                              # leave main loop

            if state != last_state:                           # log when overall state changes
                if state == Status.FAILURE:
                    main_logger.Error(f"Mission failed at t={t:.1f}s.")
                    main_execution_tree.terminate()           # stop all nodes
                    break
                last_state = state
    except KeyboardInterrupt:
        main_logger.Info("Mission interrupted")
        pass
    except Exception as e:
        main_logger.Error(f"Unhandled exception in main loop: {e}")  # catch all errors
    finally:
        main_logger.Info("Cleaning up and stopping robot")
        main_execution_tree.terminate()                     # end nodes
        motor_left, motor_right = blackboard.GetMotors()    # get wheel handles
        if motor_left and motor_right:
            StopMotors(motor_left, motor_right)             # stop robot
        main_logger.Info("Robot stopped. Controller exiting")
