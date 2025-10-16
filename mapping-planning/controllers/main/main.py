from __future__ import annotations                                              # Future annotations
from controller import Supervisor                                               # Webots supervisor 
import numpy as np                                                              # For arrays and math
from os.path import exists                                                      # File check helper
from core import (
    MappingParams,                                                              # Mapping parameter
    PlanningParams,                                                             # Planning parameter
    main_logger,                                                                # Shared logger for general info
    SetLogLevel,                                                                # Helper to change log level
    LogLevel,                                                                   # Enum for log levels
    NormalizeAngle,                                                             # Angle wrap helper
    Distance2D,                                                                 # Euclidean distance in 2d
    WorldToGrid,                                                                # World to grid converter
    GridToWorld,                                                                # Grid to world converter
    BehaviorNode,                                                               # Base node type
    Status,                                                                     # Node status enum
    Sequence,                                                                   # Behavior tree sequence
    Selector,                                                                   # Behavior tree selector
    Parallel,                                                                   # Behavior tree parallel
    blackboard,                                                                 # Global key value store
    GetFromBlackboard,                                                          # Blackboard getter function
    BBKey,                                                                      # Keys for blackboard
    GetDisplayManager,                                                          # Display manager
    DisplayUpdater,                                                             # Display behavior tree node
    SetDisplayMode,                                                             # Display mode behavior tree node
    EnableCspaceDisplay,                                                        # Enable cspace display behavior tree node
)
from navigation import (
    StopMotors,                                                                 # Stop both wheels
    BidirectionalSurveyNavigator,                                               # Survey forward then reverse
    ObstacleAvoidingNavigation,                                                 # Waypoint navigation controller
)
from mapping import (
    LidarMappingBT,                                                             # Behavior tree node to build map and cspace
    WaitForMapReady,                                                            # Behavior tree node that waits for map ok
    MapExistsOrReady,                                                           # Behavior tree node that checks saved map
    LoadMap,                                                                    # Behavior tree node to load map file
    EnsureCspaceNow,                                                            # Behavior tree node to force build cspace
    SaveMap,                                                                    # Behavior tree node to save map to disk
    ValidateLoadedMap,                                                          # Behavior tree node to sanity check map
    SaveCspaceImage,                                                            # Behavior tree node to save c-space as image
    ClearCspace,                                                                # Behavior tree node to clear existing c-space
)
from planning import (
    BuildEllipsePoints,                                                         # Waypoint generation functions
    OrderFromStart,                                                             # Reorder by start
    ClampGoal,                                                                  # Clamp waypoint to free cell
)


################################################################################
# ========================= Device Initialization ==============================
################################################################################
# Set up robot devices once at start.
# Store all handles in the shared blackboard.
def InitRobot():                                                                # Create supervisor and timestep
    robot = Supervisor()                                                        # Webots supervisor instance
    timestep = int(robot.getBasicTimeStep())                                    # Controller step; ms
    blackboard.Set(BBKey.INIT_Z, robot.getSelf().getPosition()[2])              # Cache initial Z so we can refer to ground height if needed
    return robot, timestep                                                      # Return handles

def InitSensors(robot, timestep):                                               # Enable sensors
    gps = robot.getDevice('gps')                                                # GPS handle
    compass = robot.getDevice('compass')                                        # Compass handle
    lidar = robot.getDevice('Hokuyo URG-04LX-UG01')                             # Lidar handle
    if gps:
        gps.enable(timestep)                                                    # Enable GPS updates
    if compass:
        compass.enable(timestep)                                                # Enable compass updates
    if lidar:
        lidar.enable(timestep)                                                  # Enable lidar updates
        blackboard.Set(BBKey.LIDAR, lidar)                                      # Store in blackboard
    else:
        main_logger.Warning("Lidar not detected")                               # Warn if no lidar
    if not (gps and compass):
        main_logger.Error("Sensor missing")                                     # Need gps + compass to navigate
    return gps, compass, lidar                                                  # Return sensor handles

def InitMotors(robot):                                                          # Find wheel motors
    left = robot.getDevice('wheel_left_joint')                                  # Left motor
    right = robot.getDevice('wheel_right_joint')                                # Right motor
    if not left or not right:                                                   # Check both found
        if not left:
            main_logger.Error("Left motor not found.")                          # Log missing left motor
        if not right:
            main_logger.Error("Right motor not found.")                         # Log missing right motor
        return None, None                                                       # Abort if missing
    try:
        for m in (left, right):                                                 # Put both motors in velocity mode with zero speed
            m.setPosition(float('inf'))                                         # Infinite position for velocity control
            m.setVelocity(0.0)                                                  # Start stopped
    except Exception:
        pass                                                                    # Continue execution
    return left, right                                                          # Return motors

def InitDisplay(robot):                                                         # Set up UI display
    try:
        disp = robot.getDevice("display")                                       # Get display device
        if not disp:
            main_logger.Warning("Display error not correct dimension")          # Warn about display issue
            return None                                                         # No display available
        blackboard.Set(BBKey.DISPLAY, disp)                                     # Store handle
        GetDisplayManager().InitializeDisplay()                                 # Clear canvas, set size
        return disp                                                             # return display
    except Exception as e:
        main_logger.Error(f"Display init error: {e}")                           # Report issue
        return None                                                             # Failed to initialize display

def RegisterDevices(robot, timestep, gps, compass, lidar, left_motor, right_motor, display): # Push all device handles and helpers into blackboard
    for k, v in (
        (BBKey.ROBOT, robot),                                                   # Robot core
        (BBKey.GPS, gps),                                                       # Gps
        (BBKey.COMPASS, compass),                                               # Compass
        (BBKey.LIDAR, lidar),                                                   # Lidar
        (BBKey.MOTOR_L, left_motor),                                            # Left wheel
        (BBKey.MOTOR_R, right_motor),                                           # Right wheel
        (BBKey.DISPLAY, display),                                               # UI display
        (BBKey.TIMESTEP, timestep),                                             # Control dt
        ("world_to_grid", WorldToGrid),                                         # Map transform
        ("grid_to_world", GridToWorld),                                         # Inverse transform
        ("normalize_angle", NormalizeAngle),                                    # Angle helper
        ("distance_2d", Distance2D),                                            # Distance helper
    ):
        blackboard.Set(k, v)                                                    # Store each item
    missing = [n for n, d in (("GPS", gps), ("Compass", compass), ("LiDAR", lidar)) if not d]  # Check missing core sensors and warn
    if missing:
        main_logger.Warning("Missing sensors.")                                 # Warning

def InitAllDevices():                                                           # One call to set up everything in order
    robot, timestep = InitRobot()                                               # Supervisor 
    gps, compass, lidar = InitSensors(robot, timestep)                          # Sensors
    left_motor, right_motor = InitMotors(robot)                                 # Motors
    display = InitDisplay(robot)                                                # UI display
    RegisterDevices(robot, timestep, gps, compass, lidar, left_motor, right_motor, display)  # Blackboard
    return robot, timestep, gps, compass, lidar, left_motor, right_motor, display   # Return


################################################################################
# ========================== Shared Functions ==================================
################################################################################
# Utilities for waypoint and navigation; shared across planners
def GenerateWaypoints(start_xy, cspace=None):                                   # Make inner waypoints around the start. Clamp to c-space if given
    INNER_12 = OrderFromStart(BuildEllipsePoints(), start_xy, close_loop=True)  
    if cspace is not None:                                                      # Clamp each waypoint to nearest safe cell in c-space
        INNER_12 = [ClampGoal(x, y, cspace) for x, y in INNER_12]               # Ensure waypoints are safe
    return INNER_12                                                             # Return list of (x, y)


class GenericWaypointNavigator(BehaviorNode):                                   # Generic waypoint navigator using any waypoint key in the blackboard.
    def __init__(self, waypoint_key, log_message, success_message="Waypoint navigation completed"):
        super().__init__("GenericWaypointNavigator")                            # Node name
        self.waypoint_key = waypoint_key                                        # bb key for waypoints
        self.log_message = log_message                                          # Log on start
        self.success_message = success_message                                  # Log on end
        self.navigator = None                                                   # Internal controller

    def execute(self):
        waypoints = GetFromBlackboard(self.waypoint_key)                        # Fetch waypoints
        if not waypoints:
            return Status.FAILURE                                               # nothing to do
        if self.navigator is None:
            main_logger.Info(self.log_message)                                  # Announce start
            self.navigator = ObstacleAvoidingNavigation(waypoints, bb=blackboard, traversal="once")
        status = self.navigator.execute()                                       # Step controller
        if status == Status.SUCCESS:
            main_logger.Info(self.success_message)                              # Announce done
            self.navigator = None                                               # Free controller
        return status                                                           # Bubble up status
        
    def reset(self):
        super().reset()                                                         # Reset node state
        if self.navigator:
            self.navigator.terminate()                                          # Stop controller
            self.navigator = None                                               # Clear

    def terminate(self):
        if self.navigator:
            self.navigator.terminate()                                          # Ensure stop
            self.navigator = None                                               # Clear 


################################################################################
# ========================== Behavior Tree Nodes ===============================
################################################################################
# Small helper nodes for UI and flow.
# Keep logic easy to follow and reusable.
class RunInBackground(BehaviorNode):
    def __init__(self, child):
        super().__init__("RunInBackground")                                     # Node name
        self.child = child                                                      # Wrapped node

    def execute(self):                                                          # If c-space is frozen, keep background task alive
        if GetFromBlackboard("cspace_frozen", False):
            return Status.RUNNING                                               
        child_status = self.child.tick()                                        # Tick the child
        return Status.RUNNING if child_status in (Status.SUCCESS, Status.RUNNING) else Status.FAILURE   # Keep running unless child failed hard


class OnlyOnce(BehaviorNode):
    def __init__(self, child, key_suffix=None):
        super().__init__(f"OnlyOnce({child.name})")                             
        self.child = child                                                      # Target node
        self.flag_key = f"onlyonce:{key_suffix or child.name}"                  # Unique key

    def execute(self):
        if GetFromBlackboard(self.flag_key, False):                             # Already ran before?
            return Status.SUCCESS                                               # Do nothing now
        res = self.child.tick()                                                 # Run child once
        if res in (Status.SUCCESS, Status.FAILURE):                             # Finished
            blackboard.Set(self.flag_key, True)                                 # Mark as done
        return res                                                              # Return child's result

    def reset(self):
        super().reset()                                                         # Reset self state
        blackboard.Set(self.flag_key, False)                                    # Clear only-once flag
        if hasattr(self.child, 'reset'):
            self.child.reset()                                                  # Reset child


###############################################################################
# ================================== Main =====================================
###############################################################################
# Wire up devices, build the behavior tree, and run the mission.
# Stop cleanly on finish or error.
if __name__ == "__main__":                                                      # Create all hardware interfaces first
    robot, timestep, gps, compass, lidar, motor_left, motor_right, display = InitAllDevices()
    for k, v in [                                                               # Seed shared flags and defaults in the blackboard
        ("map_ready", False),                                                   # Map not ready yet
        ("map_saved", False),                                                   # Nothing saved yet
        ("trajectory_points", []),                                              # Empty path history
        ("display_mode", "full"),                                               # Full ui mode
        ("allow_cspace_display", False),                                        # Don't show c-space yet
        ("cspace_frozen", False),                                               # Background mapping allowed
        ("survey_complete", False),                                             # Loops not done
    ]:
        blackboard.Set(k, v)                                                    # Set all defaults
    SetLogLevel(LogLevel.INFO)                                                  # Default - INFO logs
    main_logger.Info("Starting Tiago mapping and navigation system")            # Capture start position — wait for GPS to be ready
    start_xy = None                                                             # Store (x, y)
    for attempt in range(100):                                                  
        robot.step(timestep)                                                    
        start = gps.getValues()                                                 # Get GPS (x, y, z)
        if start and len(start) >= 2 and all(np.isfinite(start[:2])):           # Check we got usable data
            start_xy = (start[0], start[1])                                     # Save x, y
            break                                                               # Stop 

    if start_xy is None:                                                        # Failed to get clean GPS after many tries
        main_logger.Error("Failed to get valid GPS position after 100 attempts")
        start_xy = (0.0, 0.0)                                                   # Fallback
    blackboard.Set(BBKey.START_XY, start_xy)                                    # Store start for others

    cspace = GetFromBlackboard("cspace")                                        # Check if a map already exists in memory 
    if cspace is not None:
        main_logger.Info("Using existing map")                                  

    # build waypoint set around start (inner loop)
    INNER_12 = GenerateWaypoints(start_xy, cspace)                              # Full closed loop
    WAYPOINTS_SURVEY_INNER = INNER_12[:-1]                                      # Drop last

    # prepare parameter objects for mapping and planning subsystems
    mapping_params  = MappingParams(world_to_grid=WorldToGrid, grid_to_world=GridToWorld)  # Inject transforms
    planning_params = PlanningParams()                                          # Default planning params


    ################################################################################
    # ========================== BEHAVIOR TREE CONSTRUCTION =======================
    ################################################################################
    # Create leaves, then sequences/selectors, then the root.
    # Run display updater in parallel so UI stays live.
    # leaf nodes
    check_map            = MapExistsOrReady()                                   # Check for on disk map or ready flag
    load_map             = LoadMap()                                            # Load map file to memory
    validate_loaded_map  = ValidateLoadedMap()                                  # Basic checks on loaded data
    clear_cspace         = ClearCspace()                                        # Remove old c-space 
    save_cspace_image    = SaveCspaceImage()                                    # Write c-space image file
    lidar_mapping        = LidarMappingBT(mapping_params, bb=blackboard)        # Mapping worker node
    bidir_survey         = BidirectionalSurveyNavigator(WAYPOINTS_SURVEY_INNER)  


    class GenerateWaypointsForExistingMap(BehaviorNode):                        # Waypoint generation for existing maps
        def __init__(self):
            super().__init__("GenerateWaypointsForExistingMap")                

        def execute(self):
            cspace = GetFromBlackboard("cspace")                                # Need c-space loaded
            if cspace is None:
                return Status.FAILURE                                           # Cannot proceed
            start_xy = GetFromBlackboard("start_xy")                            # Need a start pose
            if not start_xy:
                return Status.FAILURE                                           
            existing_inner = GetFromBlackboard("waypoints_survey_inner")        # Already have waypoints?
            if existing_inner is not None:
                return Status.SUCCESS                                           # Nothing to do
            INNER_12 = GenerateWaypoints(start_xy, cspace)                      
            blackboard.Set("waypoints_survey_inner", INNER_12)                  # Store for later use by navigator
            return Status.SUCCESS                                               # Done
    
    generate_waypoints = GenerateWaypointsForExistingMap()                      


    class SimpleWaypointNavigator(BehaviorNode):                                # Simple waypoint navigator for existing maps 
        def __init__(self):
            super().__init__("SimpleWaypointNavigator")                         # Name of node
            self.navigator = None                                               # Controller ref

        def execute(self):
            waypoints = GetFromBlackboard("waypoints_survey_inner")             
            if not waypoints:
                return Status.FAILURE                                           # No waypoints yet
            if self.navigator is None:
                main_logger.Info("Starting c-space navigation with generated waypoints for existing map") 
                self.navigator = ObstacleAvoidingNavigation(waypoints, bb=blackboard, traversal="once")
            status = self.navigator.execute()                                   
            if status == Status.SUCCESS:
                main_logger.Info("Waypoint navigation completed")               # Announce milestone
                self.navigator = None                                           
            return status                                                       

        def reset(self):
            super().reset()                                                     # Reset node
            if self.navigator:
                self.navigator.terminate()                                      # Stop controller
                self.navigator = None                                           
    
    simple_navigator = SimpleWaypointNavigator()                                # Create instance
    mapping_background = RunInBackground(lidar_mapping)                         # Run mapping in background while doing loop


    mapping_and_survey_parallel = Parallel(                                     # Full mapping route when no saved map exists
        "SurveyWithBackgroundMapping",                                          # Node name
        [bidir_survey, mapping_background],                                     # Do both together
        success_threshold=1,                                                    # Success when survey finishes
        failure_threshold=None,                                                 # Don't fail just because mapping is still running
    )
 
    cspace_only_sequence = Sequence("CspaceOnlySequence", [                     # Cspace only sequence
        clear_cspace,                                                           # Clear stale c-space
        mapping_and_survey_parallel,                                            # Collect scans while touring
        EnsureCspaceNow(blackboard_instance=blackboard),                        # Force c-space compute
        WaitForMapReady(),                                                      # Wait until usable
        SaveMap(),                                                              # Save map to disk
        save_cspace_image,                                                      # Save preview image
        EnableCspaceDisplay(),                                                  # Allow c-space overlay
        SetDisplayMode("cspace"),                                               # Switch UI to c-space view
    ])

    use_existing_map_inner = Sequence("UseExistingMap", [                       # Fast path when file exists 
        check_map,                                                              
        load_map,                                                               # Load file
        validate_loaded_map,                                                    # Basic validation
        generate_waypoints,                                                   
        EnableCspaceDisplay(),                                                  # Show c-space in UI
        SetDisplayMode("cspace"),                                               # Set UI mode
        simple_navigator,                                                      
    ])

    use_existing_map = OnlyOnce(use_existing_map_inner, "existing_map")         # Ensure we don't run the 'use existing map' branch twice

    main_mission_tree = Selector("MainMissionTree", [                           # Top level selector prefers fast path; falls back to building c-space
        use_existing_map,                                                       # Try existing map
        cspace_only_sequence,                                                   # Else build new
    ])

    display_updater = DisplayUpdater()                                          # Display updater runs continuously

    main_execution_tree = Parallel("MainWithDisplay", [                         # Parallel: mission + display together
        main_mission_tree,                                                      
        display_updater,                                                        # UI branch
    ], success_threshold=2, failure_threshold=2)                                # Both should keep running
    
    last_state = None                                                           
    last_log_time = 0                                                           # For periodic logs
    mission_completed = False                                                   # Set once mission succeeds
    
    try:
        while robot.step(timestep) != -1:                                       # Main control loop
            state = main_execution_tree.tick()                                  # Tick root parallel
            t = robot.getTime()                                                 
            mission_state = main_mission_tree.tick()                            
            if mission_state == Status.SUCCESS and not mission_completed:
                mission_completed = True                                        # Mark done once
                motor_left, motor_right = blackboard.GetMotors()                # Get wheels
                if motor_left and motor_right:
                    StopMotors(motor_left, motor_right)                         # Stop motion
                if blackboard.Get("display_mode") == "cspace":                  # If we're in c-space only mode, keep UI present
                    pass                                                        # Don't break; let display keep updating
                else:
                    break                                                       # Done with mission, exit loop
            if state != last_state:                                             # Report state transitions for the root tree
                if state == Status.FAILURE:
                    main_logger.Error(f"Mission failed at t={t:.1f}s.")         # Failure info
                    main_execution_tree.terminate()                             # Stop all nodes
                    break                                                       # Exit early
                last_state = state                                              # Remember new state
    except KeyboardInterrupt:
        main_logger.Info("Mission interrupted")                                 
        pass  
    except Exception as e:
        main_logger.Error(f"Unhandled exception in main loop: {e}")             # Catch-all to avoid uncontrolled exit
    finally:
        main_logger.Info("Cleaning up and stopping robot")                      # Always try to stop cleanly
        main_execution_tree.terminate()                                         # Terminate nodes
        motor_left, motor_right = blackboard.GetMotors()                        # Wheel handles
        if motor_left and motor_right:
            StopMotors(motor_left, motor_right)                                 # Stop wheels
        main_logger.Info("Robot stopped. Controller exiting")                   # Cleanup complete, exit controller
