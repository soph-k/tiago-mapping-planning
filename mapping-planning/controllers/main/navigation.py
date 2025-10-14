###############################################################################
# ------------------------- Imports  -----------------------------------------
###############################################################################
from __future__ import annotations                    # Postpone evaluation of annotations
import numpy as np                                    # Numerical utilities
import random                                         # Random choices, etc.
from collections import deque                     
from dataclasses import dataclass, field              # Dataclass helpers
from core import BehaviorNode, Status, blackboard, nav_logger, NormalizeAngle # Project imports


###############################################################################
# ------------------------- Utilities -----------------------------------------
###############################################################################
def AngDiff(a: float, b: float) -> float:
    return NormalizeAngle(a - b)                     # Normalized angular difference 

def FindMinimumFiniteValue(a):
    a = np.asarray(a)                                # Ensure numpy array
    f = a[np.isfinite(a)]                            # Filter out infinite
    if f.size:
        return float(np.min(f))                      # Min of finite values
    return 10.0                                      # Fallback large distance if none finite

def ScanLidarSectors(ranges):
    n = len(ranges)                                  # Number of rays
    t = n // 3                                       # Third of array length
    left = FindMinimumFiniteValue(ranges[:t])        # Left sector min
    front = FindMinimumFiniteValue(ranges[t:2 * t])  # Front sector min
    right = FindMinimumFiniteValue(ranges[2 * t:])   # Right sector min
    m = FindMinimumFiniteValue(ranges)               # Global min
    return left, front, right, min(left, front, right, m)


###############################################################################
# ------------------------- Device helpers ------------------------------------
###############################################################################
def GetRobotDevicesFromBlackboard(bb, *keys):
    devs = [bb.Get(k) for k in keys]                 # Fetch devices by key
    miss = []                                        # Track missing ones
    for k, d in zip(keys, devs):
        if d is None:
            miss.append(k)                           # Collect missing key names
    if miss:
        nav_logger.Warning(f"Missing devices: {miss}")  # Warn if any missing
    return devs                                      # Return

def ClipMotorVelocity(motor, v, name):
    if motor is None:
        nav_logger.Error(f"{name}: Motor device is None")  # Error if motor missing
        return 0.0
    vmax = motor.getMaxVelocity()                    
    return float(np.clip(v, -vmax, vmax))            

def SafeSetMotorVelocities(L, R, lv, rv):
    if not (L and R):
        nav_logger.Error("Motor devices are None")  # Guard against None
        return False
    L.setVelocity(ClipMotorVelocity(L, lv, "MotorL"))  # Set safe left wheel vel
    R.setVelocity(ClipMotorVelocity(R, rv, "MotorR"))  # Set safe right wheel vel
    return True

def StopMotors(L, R):
    SafeSetMotorVelocities(L, R, 0.0, 0.0)           # Command zero velocity to both motors


###############################################################################
# ------------------------- Stuck detection state -----------------------------
################################################################################
@dataclass
class StuckDetourState:
    stuck_threshold: int = 10                        # Counts before declaring stuck
    max_detour_attempts: int = 3                     # Max number of detours to insert
    detour_count: int = 0                            # How many detours used so far
    stuck_events: int = 0                            # Number of consecutive stuck detections
    stuck_cooldown_until: float = 0.0                # Time until next stuck check allowed
    last_detour_time: float = -1.0                   # Last time a detour was inserted
    detour_active: bool = False                      # Whether currently executing a detour
    detour_start_time: float | None = None           # Start time of active detour
    heading_hist: deque = field(default_factory=lambda: deque(maxlen=32))      # Recent headings
    heading_hist_t: deque = field(default_factory=lambda: deque(maxlen=32))    # Timestamps for headings

    def reset_detours(self):
        self.detour_count = 0                        # Reset detour count
        self.last_detour_time = -1.0                 # Clear last detour time
        self.detour_active = False                   # Mark no active detour
        self.detour_start_time = None                # Clear detour start time
        self.stuck_events = 0                        # Reset stuck event streak

    def reset_all(self):
        self.reset_detours()                         # Reset detour
        self.stuck_cooldown_until = 0.0              # Remove cooldown
        self.heading_hist.clear()                    # Clear heading history
        self.heading_hist_t.clear()                  # Clear heading timestamps


###############################################################################
# ------------------------- Navigation base -----------------------------------
###############################################################################
class NavigationBase(BehaviorNode):
    def __init__(self, name, waypoints=None, maxSpeed=5.0, wheel_radius=None, distance_2wheels=None,
                 p1=0.7, p2=0.3, tolerance=0.30, traversal="once", start_direction=+1, bb=None):
        super().__init__(name)                       # Init BehaviorNode
        self.bb = bb or blackboard                   # Use provided or global blackboard
        self.waypoints = waypoints or []             # List of x and y targets
        self.dir = 1 if start_direction >= 0 else -1 # Direction of traversal
        if self.dir > 0:
            self.index = 0                           # Start at first waypoint
        else:
            self.index = max(0, len(self.waypoints) - 1)  # Start at last if reverse
        self.maxSpeed = maxSpeed                     # Max translational speed
        self.R = wheel_radius or 0.0985              # Wheel radius
        self.L = distance_2wheels or 0.404           # Wheelbase
        self.p1 = p1                                 # Tuning parameter
        self.p2 = p2                                 # Tuning parameter
        self.tolerance = tolerance                   # Distance threshold for waypoint hit
        self.start_time = None                       # Navigation start time
        self.waypoint_start_time = None              # Start time for current waypoint
        self.position_history = deque(maxlen=16)     # Recent positions for stuck detection
        self.last_position_check = None              # Last time we sampled position
        self.stuck_counter = 0                       
        self.stuck_threshold = 8                     # Threshold for local stuck counter
        self._finished = False                       # Internal completion flag
        self.avoid_state = "NORMAL"                  
        self.avoid_start_time = None                 # When avoidance started

    def _devs(self):
        return GetRobotDevicesFromBlackboard(self.bb, 'robot', 'gps', 'compass', 'motorL', 'motorR')  # Grab devices

    def _pose(self, gps, compass):
        xw, yw = gps.getValues()[:2]                 # Current x,y from GPS
        cv = compass.getValues()                     # Compass vector
        th = float(np.arctan2(cv[0], cv[1]))         # Heading angle from compass vector
        return xw, yw, th                            # Return pose 

    def current_target(self):
        if not self.waypoints:
            return None                              # No waypoints configured
        if 0 <= self.index < len(self.waypoints):
            return self.waypoints[self.index]        # Active waypoint
        return None                                  # Out of range index

    def advance_waypoint(self, _robot):
        if not self.waypoints:
            return                                   # Nothing to advance
        self.index += self.dir                       # Move to next waypoint
        self.waypoint_start_time = None              
        if hasattr(self, "sd") and self.sd.detour_active:
            self.sd.detour_active = False            # Clear detour state
            self.sd.detour_start_time = None

    def all_waypoints_completed(self):
        if not self.waypoints:
            return True                              # Empty plan counts as done
        if self.dir == 1 and self.index >= len(self.waypoints):
            return True                              # Forward traversal
        if self.dir == -1 and self.index < 0:
            return True                              # Reverse traversal
        return False

    def terminate(self):
        L, R = self.bb.GetMotors()                   # Get motors
        if L and R:
            StopMotors(L, R)                         # Stop motors safely
        else:
            nav_logger.Error("Motors missing from blackboard.")  # Log if missing

    def reset(self):
        super().reset()                              # Reset BehaviorNode state
        if self.dir >= 0:
            self.index = 0                           # Reset to first waypoint
        else:
            self.index = max(0, len(self.waypoints) - 1)  # Reset to last
        self.start_time = None                       # Clear timers
        self.waypoint_start_time = None
        self.position_history.clear()                # Clear motion history
        self.last_position_check = None
        self.stuck_counter = 0                       # Reset counters
        if hasattr(self, "sd"):
            self.sd.reset_all()                      # Reset stuck state
        self.avoid_state = "NORMAL"                  # Back to normal
        self.avoid_start_time = None
        self._finished = False


###############################################################################
# ------------------------- Obstacle avoiding navigator -----------------------
###############################################################################
class ObstacleAvoidingNavigation(NavigationBase):
    def __init__(self, waypoints=None, bb=None, **kw):
        super().__init__("ObstacleAvoidingNavigation", waypoints, bb=bb, **kw) 
        self.sd = StuckDetourState()           

    def _update_heading_hist(self, theta, t):
        self.sd.heading_hist.append(theta)            # Append heading sample
        self.sd.heading_hist_t.append(t)              # Append timestamp
        cut = t - 1.5                                 
        while self.sd.heading_hist_t and self.sd.heading_hist_t[0] < cut:
            self.sd.heading_hist_t.popleft()          # Drop old timestamp
            self.sd.heading_hist.popleft()            # Drop matching heading
        if len(self.sd.heading_hist) > 1:
            return abs(AngDiff(self.sd.heading_hist[-1], self.sd.heading_hist[0]))  # Total rotation span
        return 0.0                                    # Not enough samples

    def check_if_stuck(self, pos, t):
        if self.avoid_state != "NORMAL":
            return False                              
        if t < self.sd.stuck_cooldown_until:
            return False                              

        if (self.last_position_check is None) or (t - self.last_position_check >= 0.25):
            self.position_history.append((t, pos[0], pos[1]))  # Record position snapshot
            self.last_position_check = t                       # Update sample time
            if len(self.position_history) < 2:
                return False                          # Need at least two points

        t0, x0, y0 = self.position_history[0]        # Oldest sample
        dt = self.position_history[-1][0] - t0       # Time span covered
        if dt < 2.0:
            return False                              

        dx = self.position_history[-1][1] - x0      
        dy = self.position_history[-1][2] - y0       
        moved = float(np.hypot(dx, dy))              # Distance moved
        if moved < 0.03:                             # Very little movement
            self.stuck_counter += 1
            if self.stuck_counter >= self.sd.stuck_threshold:
                self.stuck_counter = 0               # Reset local counter
                return True                          # Declare stuck
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)  # Decay counter on progress
        return False

    def _inject_detour(self, xw, yw, theta, side, t):
        if self.sd.detour_count >= self.sd.max_detour_attempts:
            return False                            
        d = 0.35                                      # Lateral offset
        th = theta + np.pi / 2.0                      # Perpendicular to heading
        dx = side * d * np.cos(th)                    # Lateral x shift
        dy = side * d * np.sin(th)                    # Lateral y shift
        wp = (xw + dx, yw + dy)                       # New temporary waypoint
        self.waypoints[self.index:self.index] = [wp]  # Insert before current target
        self.sd.detour_count += 1                     # Increment detour count
        self.sd.detour_active = True                  # Mark active detour
        self.sd.detour_start_time = t                 # Record start time
        self.sd.last_detour_time = t                  # Update last detour time
        return True

    def execute(self):
        robot, gps, compass, L, R = GetRobotDevicesFromBlackboard(
            self.bb, 'robot', 'gps', 'compass', 'motorL', 'motorR'  # Request core devices
        )
        if not (robot and gps and compass and L and R):
            return Status.RUNNING                     # Keep ticking until devices are available
        t = robot.getTime()                           # Current time from robot
        if self.start_time is None:
            self.start_time = t                       # Initialize start time
        if self.all_waypoints_completed():
            StopMotors(L, R)                          # Stop when done
            return Status.SUCCESS
        xw, yw = gps.getValues()[:2]                  # Current position 
        cv = compass.getValues()[:2]                  # Compass vector 
        theta = float(np.arctan2(cv[0], cv[1]))       # Heading in radians
        target = self.current_target()                # Active waypoint
        if target is None:
            StopMotors(L, R)                          # Nothing to do, stop
            return Status.SUCCESS
        if self.waypoint_start_time is None:
            self.waypoint_start_time = t              # Start timing this waypoint
        if self.sd.detour_active and self.sd.detour_start_time is not None:
            if (t - self.sd.detour_start_time) > 15.0:
                nav_logger.Warning("Detour timeout - clearing")  # Detour taking too long
                self.sd.detour_active = False
                self.sd.detour_start_time = None
        lidar = self.bb.GetLidar()                    # Fetch lidar if present
        if lidar:
            ranges = np.asarray(lidar.getRangeImage())  # Convert range image to array
        else:
            ranges = np.array([])                     # Empty array if no lidar
        recent_rot = self._update_heading_hist(theta, t)  
        obstacle = False                              # Obstacle ahead
        emerg = False                                
        left = 10.0                                   # Defaults 
        front = 10.0
        right = 10.0
        min_all = 10.0
        if ranges.size and np.isfinite(ranges).any():
            left, front, right, min_all = ScanLidarSectors(ranges) 
            if (front < 0.6) or (min_all < 0.5):
                obstacle = True                       # Obstacle detected
            if (front < 0.20) and (min_all < 0.17):
                emerg = True                          # Very close = emergency
        if emerg:
            nav_logger.Error(f"Emergency stop: front={front:.2f}m min={min_all:.2f}m")  # Log details
            SafeSetMotorVelocities(L, R, -3.0, -3.0)  # Reverse quickly
            return Status.RUNNING                     # Keep control loop alive
        if self.avoid_state == "NORMAL":
            if t >= self.sd.stuck_cooldown_until and recent_rot < 0.35:  # Not rotating much
                if self.check_if_stuck((xw, yw), t):                     # Check motion over time
                    nav_logger.Warning("Stuck detected – recovery")      # Log recovery
                    self.sd.stuck_events += 1                            # Counter
                    self.sd.stuck_cooldown_until = t + 1.2               # Set cooldown
                    self.position_history.clear()                        # Reset history
                    if self.sd.stuck_events >= 3 and (target is not None):
                        side = random.choice([-1, 1])                     # Pick left/right
                        self._inject_detour(xw, yw, theta, side, t)      # Insert lateral detour
                        self.sd.stuck_events = 0                         # Reset streak
                    self.avoid_state = "REVERSING"                       # Enter reversing mode
                    self.avoid_start_time = t                            # Record start time
                    SafeSetMotorVelocities(L, R, -1.8, -1.8)             # Back up slowly
                    return Status.RUNNING
        if obstacle:
            side_left = left > right                 # Choose side with more clearance
            if front < 0.8:
                base = 2.0                           # Slow base speed in tighter space
            else:
                base = 3.0                           # Otherwise a bit faster
            if base > self.maxSpeed:
                base = self.maxSpeed                 
            steer = 0.9 if side_left else -0.9       # Steering direction/amount
            tg = 0.7                                 # Turn gain
            lv = np.clip(base - tg * steer, -self.maxSpeed, self.maxSpeed)  # Left velocity
            rv = np.clip(base + tg * steer, -self.maxSpeed, self.maxSpeed)  # Right velocity
            if front < 0.45:                         # Very close
                lv *= 0.5
                rv *= 0.5
                if side_left:
                    rv += 1.0                         # Nudge to turn right
                else:
                    lv += 1.0                         # Nudge to turn left
            SafeSetMotorVelocities(L, R, float(lv), float(rv))  # Apply wheel commands
            if (front < 0.25) and ((t - self.sd.last_detour_time) > 2.0) and (not self.sd.detour_active):
                self._inject_detour(xw, yw, theta, 1 if side_left else -1, t)  
            return Status.RUNNING
        dx = target[0] - xw                          # Vector to goal x
        dy = target[1] - yw                          # Vector to goal y
        rho = float(np.hypot(dx, dy))                # Distance to goal
        alpha = NormalizeAngle(np.arctan2(dy, dx) - theta)  # Heading error
        if rho < self.tolerance:                     # Within goal tolerance
            if abs(alpha) > 0.45:                    # If not facing roughly forward, wait to fix
                return Status.RUNNING
            if (t - (self.waypoint_start_time or t)) >= 0.3:
                self.advance_waypoint(robot)         # Advance after short wait
            if self.all_waypoints_completed():
                StopMotors(L, R)                     # Stop when plan done
                return Status.SUCCESS
            return Status.RUNNING
        turn_scale = min(abs(alpha) / np.pi, 0.75)   # Normalize turn demand
        speed = max(1.2, 4.0 * (1 - turn_scale))     # Reduce speed as turn demand increases
        turn = 2.0 * alpha                          
        lv = np.clip(speed - turn, -4.0, 4.0)         # Left wheel command
        rv = np.clip(speed + turn, -4.0, 4.0)         # Right wheel command
        SafeSetMotorVelocities(L, R, float(lv), float(rv))  # Send commands
        return Status.RUNNING                         # Continue navigating
