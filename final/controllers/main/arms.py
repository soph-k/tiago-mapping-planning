import numpy as np                                                                                                   # numeric ops
import py_trees                                                                                                     # behavior trees
from typing import Callable, Dict, List, Optional, Tuple                                                            # typing helpers

from config import RobotConfig                                                                                      # robot-wide parameters
from utils import (compute_distance, dev, enable, rtime, safe, standoff, wrap_angle, world_to_pixel)               # utilities from your stack
from utils import RobotDeviceManager                                                                                # base class that provides robot + memory handles, timestep, helpers


####################################################################################
# ------------------------------ Behavior Tree Helper ------------------------------
####################################################################################
class BTAction(py_trees.behaviour.Behaviour):                                                                       # thin adapter around a callable returning a string status
    def __init__(self, func: Callable[[], str]) -> None:
        name = getattr(func, "__name__", "BTAction")                                                                # auto-name from function
        super().__init__(name=name)                                                                                 # init parent with friendly name
        self.func = func                                                                                            # store callback

    def update(self) -> py_trees.common.Status:
        result_str = self.func()                                                                                    # call user function returning "SUCCESS"/"FAILURE"/"RUNNING"
        if result_str == "SUCCESS":
            return py_trees.common.Status.SUCCESS                                                                   # map to enum
        elif result_str == "FAILURE":
            return py_trees.common.Status.FAILURE                                                                   # map to enum
        return py_trees.common.Status.RUNNING                                                                       # default to RUNNING


############################################################################
# ------------------------------ Arm Control ------------------------------
###########################################################################
class ArmPoseController(RobotDeviceManager):                                                                        # moves joints to a named pose; special logic for 'grab'
    """Kept here to avoid circular imports with planning.py behaviors."""                                           # module structure note
    def __init__(self, robot, memory, pose_name: str = 'safe') -> None:
        super().__init__(robot, memory)                                                                             # base init for timing and memory
        self.pose_name: str = pose_name                                                                             # target pose name
        self.threshold: float = 0.05                                                                                # joint position tolerance
        self.threshold_force: float = -2.5                                                                          # contact threshold for force feedback

        self.configurations: Dict[str, Dict[str, float]] = {                                                        # predefined joint configurations
            'safe': {
                'torso_lift_joint': 0.05, 'arm_1_joint': 1.600, 'arm_2_joint': np.pi / 4, 'arm_3_joint': -2.815,
                'arm_4_joint': 0.8854, 'arm_5_joint': 0.0, 'arm_6_joint': 0.0, 'arm_7_joint': np.pi / 2,
                'gripper_left_finger_joint': 0.0, 'gripper_right_finger_joint': 0.0, 'head_1_joint': 0.0, 'head_2_joint': 0.0
            },
            'reach': {
                'torso_lift_joint': 0.11, 'arm_1_joint': 1.600, 'arm_2_joint': np.pi / 4, 'arm_3_joint': 0.0,
                'arm_4_joint': 0.8854, 'arm_5_joint': 0.0, 'arm_6_joint': 0.0, 'arm_7_joint': np.pi / 2,
                'gripper_left_finger_joint': 0.045, 'gripper_right_finger_joint': 0.045, 'head_1_joint': 0.0, 'head_2_joint': 0.0
            },
            'reach_open': {                                                                                         # same as reach here
                'torso_lift_joint': 0.11, 'arm_1_joint': 1.600, 'arm_2_joint': np.pi / 4, 'arm_3_joint': 0.0,
                'arm_4_joint': 0.8854, 'arm_5_joint': 0.0, 'arm_6_joint': 0.0, 'arm_7_joint': np.pi / 2,
                'gripper_left_finger_joint': 0.045, 'gripper_right_finger_joint': 0.045, 'head_1_joint': 0.0, 'head_2_joint': 0.0
            },
            'grab': {
                'torso_lift_joint': 0.11, 'arm_1_joint': 1.600, 'arm_2_joint': np.pi / 4, 'arm_3_joint': 0.0,
                'arm_4_joint': 0.8854, 'arm_5_joint': 0.0, 'arm_6_joint': 0.0, 'arm_7_joint': np.pi / 2,
                'gripper_left_finger_joint': 0.0, 'gripper_right_finger_joint': 0.0
            },
            'place': {
                'torso_lift_joint': 0.09, 'arm_1_joint': 1.6, 'arm_2_joint': 0.9, 'arm_3_joint': 0.0,
                'arm_4_joint': 0.8854, 'arm_5_joint': 0.0, 'arm_6_joint': 0.0, 'arm_7_joint': 1.576,
                'gripper_left_finger_joint': 0.0, 'gripper_right_finger_joint': 0.0, 'head_1_joint': 0.0, 'head_2_joint': 0.0
            }
        }

        if self.pose_name not in self.configurations:                                                             # unknown pose; fallback
            print(f" Unknown pose '{self.pose_name}', defaulting to 'safe'.")                                     # helpful log
            self.pose_name = 'safe'                                                                               # default pose

        self.target_positions: Dict[str, float] = dict(self.configurations[self.pose_name])                       # copy per-instance
        self.joint_motors: Dict[str, object] = {}                                                                 # resolved in setup
        self.joint_sensors: Dict[str, object] = {}                                                                # resolved in setup

    def setup(self) -> None:
        for joint_name in self.target_positions:                                                                  # resolve motors
            motor = dev(self.robot, joint_name)                                                                    # device lookup
            if motor is not None:
                self.joint_motors[joint_name] = motor                                                             # store motor

        sensor_names = [                                                                                          #  sensors to enable
            'torso_lift_joint_sensor','arm_1_joint_sensor','arm_2_joint_sensor','arm_3_joint_sensor',
            'arm_4_joint_sensor','arm_5_joint_sensor','arm_6_joint_sensor','arm_7_joint_sensor',
            'gripper_left_sensor_finger_joint','gripper_right_sensor_finger_joint',
            'head_1_joint_sensor','head_2_joint_sensor'
        ]
        for s_name in sensor_names:
            sensor = dev(self.robot, s_name)                                                                       # device lookup
            enable(sensor, self.timestep)                                                                          # safe enable if present
            if sensor is not None:
                self.joint_sensors[s_name] = sensor                                                                # store sensor

    def initialise(self) -> None:
        for joint_name, goal in self.target_positions.items():                                                     # send position + velocity
            motor = self.joint_motors.get(joint_name)                                                              # fetch motor
            if motor is None:
                continue                                                                                           # skip if missing

            motor.setPosition(goal)                                                                                # position target

            if 'torso' in joint_name:
                speed = RobotConfig.TORSO_SPEED                                                                    # torso speed
            elif 'gripper' in joint_name:
                speed = RobotConfig.GRIPPER_SPEED                                                                  # gripper speed
            else:
                speed = 0.2 if self.pose_name == 'place' else RobotConfig.ARM_SPEED                                # gentle on place
            motor.setVelocity(speed)                                                                               # set velocity profile

        if self.pose_name == 'grab':                                                                               # enable force feedback
            for nm in ('gripper_left_finger_joint', 'gripper_right_finger_joint'):
                safe(lambda: self.joint_motors.get(nm).enableForceFeedback(self.timestep))                         # guarded call

    def update(self) -> str:
        mismatched_count = 0                                                                                       # count joints outside tolerance
        for joint_name, goal in self.target_positions.items():
            sensor = self.joint_sensors.get(f"{joint_name}_sensor")                                                # paired sensor name
            error = abs(sensor.getValue() - goal) if sensor is not None else 0.0                                   # 0 if missing
            if error > self.threshold:
                mismatched_count += 1                                                                              # accumulate

        if mismatched_count > 0 and self.pose_name in ['grab', 'reach']:                                           # slow down when close
            worst_error = 0.0                                                                                      # track maximum joint error
            for joint_name, goal in self.target_positions.items():
                sensor = self.joint_sensors.get(f"{joint_name}_sensor")                                            # sensor lookup
                if sensor is not None:
                    worst_error = max(worst_error, abs(sensor.getValue() - goal))                                   # update worst
            if worst_error < 0.03:                                                                                 # close enough to creep
                for joint_name, motor in self.joint_motors.items():
                    if motor is not None and 'gripper' not in joint_name:
                        safe(lambda: motor.setVelocity(0.05))                                                      # gentle final alignment

        if self.pose_name == 'grab':                                                                               # special force check
            if mismatched_count != 0:
                return "RUNNING"                                                                                   # still moving into pose
            if not hasattr(self, 'grab_settle_time'):
                self.grab_settle_time = rtime(self.robot)                                                          # record first-arrival time
                print("Reached pose, waiting before closing gripper...")                                   # log
                return "RUNNING"                                                                                   # allow settle
            if rtime(self.robot) - self.grab_settle_time >= 0.03:                                                  # short delay
                left_motor = self.joint_motors.get('gripper_left_finger_joint')                                    # left gripper motor
                right_motor = self.joint_motors.get('gripper_right_finger_joint')                                  # right gripper motor
                try:
                    lf = left_motor.getForceFeedback() if left_motor else 0.0                                      # left force
                    rf = right_motor.getForceFeedback() if right_motor else 0.0                                    # right force
                    if lf < self.threshold_force and rf < self.threshold_force:
                        return "SUCCESS"                                                                            # contact on both fingers
                except Exception:
                    pass                                                                                            # force feedback not available
            return "RUNNING"                                                                                        # still checking

        if mismatched_count == 0:
            return "SUCCESS"                                                                                        # non-grab pose reached

        return "RUNNING"                                                                                            # keep going


####################################################################################
# ------------------------------ Generic BT Behaviors ------------------------------ 
####################################################################################
class MoveToPose(py_trees.behaviour.Behaviour):
    def __init__(self, rm: RobotDeviceManager, pose: str, name: Optional[str] = None, speed: Optional[float] = None) -> None:
        super().__init__(name or f"Move to {pose}")                                                                 # default name
        self.rm = rm                                                                                                # device manager
        self.pose = pose                                                                                            # pose name
        self.ctrl: Optional[ArmPoseController] = None                                                               # pose controller created on initialise
        self.custom_speed = speed                                                                                   # placeholder for future tuning

    def initialise(self) -> None:
        print(f"Moving arm to {self.pose} pose...")
        self.rm.memory.set("suppress_reactive", True)                                                               # avoid reflexes during arm move
        self.ctrl = ArmPoseController(self.rm.robot, self.rm.memory, self.pose)                                     # create controller
        self.ctrl.setup(); self.ctrl.initialise()                                                                   # resolve + command

    def update(self) -> py_trees.common.Status:
        self.rm._stop()                                                                                             # stop base while moving arm
        if self.ctrl is None:
            return py_trees.common.Status.FAILURE                                                                   # defensive
        result = self.ctrl.update()                                                                                 # poll controller
        if result == "SUCCESS":
            print(f"Reached {self.pose} pose")
            return py_trees.common.Status.SUCCESS                                                                   # done
        return py_trees.common.Status.RUNNING                                                                       # continue

class RotateToTarget(py_trees.behaviour.Behaviour):
    def __init__(self, rm: RobotDeviceManager, target: Tuple[float, float], name="Rotate to Target") -> None:
        super().__init__(name)                                                                                      # name
        self.rm = rm                                                                                                # device manager
        self.target = target                                                                                        # target (x, y)

    def initialise(self) -> None:
        print(f"Rotating to face target at {self.target}")
        self.rm.memory.set("suppress_reactive", True)                                                               # rotation should not be interrupted

    def update(self) -> py_trees.common.Status:
        (x, y) = self.rm._position()                                                                                # current position
        heading = self.rm._orientation()                                                                            # current heading (rad)
        dx, dy = self.target[0] - x, self.target[1] - y                                                             # vector to target
        desired_heading = np.arctan2(dy, dx)                                                                        # desired heading
        error = wrap_angle(desired_heading - heading)                                                               # heading error

        if abs(error) < 0.02:                                                                                       # close enough
            self.rm._stop()                                                                                         # stop wheels
            return py_trees.common.Status.SUCCESS                                                                   # done

        turn_dir = np.sign(error)                                                                                   # direction to turn
        turn_mag = max(min(abs(error) / np.pi, 1.0), 0.15)                                                          # bounded magnitude
        s = 1.0 * turn_dir * turn_mag                                                                               # wheel speed command

        self.rm._set_wheel_speeds(-s, s)                                                                            # differential in-place turn
        return py_trees.common.Status.RUNNING                                                                       # keep turning

class MoveToTarget(py_trees.behaviour.Behaviour):
    def __init__(self, rm: RobotDeviceManager, target: Tuple[float, float], max_speed=1.5, name="Move to Target") -> None:
        super().__init__(name)                                                                                      # name
        self.rm = rm                                                                                                # device manager
        self.target = target                                                                                        # goal position
        self.max_speed = max_speed                                                                                  # max travel speed

    def update(self) -> py_trees.common.Status:
        if hasattr(self.rm, 'move_towards'):
            reached = self.rm.move_towards(self.target, self.max_speed)  # type: ignore[attr-defined]                # platform helper
        else:
            reached = compute_distance(self.rm._position(), self.target) < 0.5                                      # fallback radius

        return py_trees.common.Status.SUCCESS if reached else py_trees.common.Status.RUNNING                        # done/continue

class RetreatFromPosition(py_trees.behaviour.Behaviour):
    def __init__(self, rm: RobotDeviceManager, distance=0.48, name="Retreat") -> None:
        super().__init__(name)                                                                                      # name
        self.rm = rm                                                                                                # device manager
        self.distance = distance                                                                                    # retreat distance
        self.start: Optional[Tuple[float, float]] = None                                                            # start position

    def initialise(self) -> None:
        self.start = self.rm._position()                                                                            # remember where we started
        self.rm.memory.set("suppress_reactive", True)                                                               # deliberate reverse motion

    def update(self) -> py_trees.common.Status:
        if self.start is None:
            return py_trees.common.Status.FAILURE                                                                   # defensive (should not happen)
        if compute_distance(self.rm._position(), self.start) > self.distance:                                       # backed up enough
            self.rm._stop()                                                                                         # stop
            return py_trees.common.Status.SUCCESS                                                                   # done
        self.rm._set_wheel_speeds(-0.25, -0.25)                                                                     # gentle reverse
        return py_trees.common.Status.RUNNING                                                                       # keep reversing

class WaitAndOpenGripper(py_trees.behaviour.Behaviour):
    def __init__(self, rm: RobotDeviceManager, wait_time=0.5, name="Wait and Open Gripper") -> None:
        super().__init__(name)                                                                                      # name
        self.rm = rm                                                                                                # device manager
        self.wait_time = wait_time                                                                                  # delay before open
        self.start_time: Optional[float] = None                                                                     # timer start
        self.opened: bool = False                                                                                   # first open flag

    def initialise(self) -> None:
        self.start_time = rtime(self.rm.robot)                                                                      # start timer
        self.opened = False                                                                                         # reset

    def update(self) -> py_trees.common.Status:
        if self.start_time is None:
            return py_trees.common.Status.FAILURE                                                                   # defensive

        if rtime(self.rm.robot) - self.start_time < self.wait_time:                                                 # still waiting to settle
            return py_trees.common.Status.RUNNING                                                                   # keep waiting

        if not self.opened:                                                                                         # issue open once
            for nm in ('gripper_left_finger_joint','gripper_right_finger_joint'):
                m = dev(self.rm.robot, nm)                                                                          # motor lookup
                if m is not None:
                    m.setPosition(0.045)                                                                            # open width
                    m.setVelocity(RobotConfig.GRIPPER_SPEED)                                                        # gentle
            self.opened = True                                                                                      # mark sent

        ls = dev(self.rm.robot,'gripper_left_sensor_finger_joint')                                                  # optional sensor
        rs = dev(self.rm.robot,'gripper_right_sensor_finger_joint')                                                 # optional sensor
        try:
            if ls and rs and ls.getValue() < 0.02 and rs.getValue() < 0.02:                                        # confirm opened
                self.rm._stop()                                                                                     # stop
                return py_trees.common.Status.SUCCESS                                                               # done
        except Exception:
            pass                                                                                                    # ignore sensor errors

        if rtime(self.rm.robot) - self.start_time > self.wait_time + 1.0:                                           # fallback timeout
            self.rm._stop()                                                                                         # stop
            return py_trees.common.Status.SUCCESS                                                                   # done
        return py_trees.common.Status.RUNNING                                                                       # keep waiting

class WaitSeconds(py_trees.behaviour.Behaviour):
    def __init__(self, rm: RobotDeviceManager, duration: float, name="Wait") -> None:
        super().__init__(name)                                                                                      # name
        self.rm = rm                                                                                                # device manager
        self.duration = duration                                                                                    # wait duration
        self.start_time: Optional[float] = None                                                                     # timer start

    def initialise(self) -> None:
        self.start_time = rtime(self.rm.robot)                                                                      # capture start

    def update(self) -> py_trees.common.Status:
        if self.start_time is None:
            return py_trees.common.Status.FAILURE                                                                   # defensive
        return (py_trees.common.Status.SUCCESS
                if rtime(self.rm.robot) - self.start_time >= self.duration
                else py_trees.common.Status.RUNNING)                                                                # success when time elapsed

class ComputeStandoffToTable(py_trees.behaviour.Behaviour):
    def __init__(self, rm: RobotDeviceManager, jar_index: int, name="Compute Standoff"):
        super().__init__(name)                                                                                      # name
        self.rm = rm                                                                                                # device manager
        self.idx = jar_index                                                                                        # select which drop zone

    def update(self) -> py_trees.common.Status:
        cur = self.rm._position()                                                                                   # robot position
        drop = RobotConfig.DROPOFF_POINTS[self.idx][:2]                                                             # (x,y) of drop
        so = standoff(cur, drop, dist=0.35)                                                                         # compute approach point
        nav = self.rm.memory.get("navigation")                                                                      # planner iface
        cspace = self.rm.memory.get("cspace")                                                                       # occupancy grid
        if nav and cspace is not None:
            px, py = world_to_pixel(*so)                                                                            # world->grid
            if 0 <= px < cspace.shape[0] and 0 <= py < cspace.shape[1] and cspace[px, py] < 0.5:                    # occupied?
                so = drop                                                                                           # fallback to direct
        self.rm.memory.set(f"table_standoff_{self.idx}", so)                                                        # publish for planner
        return py_trees.common.Status.SUCCESS                                                                       # done

class MoveAlongPlannedPath(py_trees.behaviour.Behaviour):
    def __init__(self, rm: RobotDeviceManager, goal_pos: Tuple[float, float] = None,
                 memory_key: str = None, wp_tol: float = 0.35, name="Follow A* Path"):
        super().__init__(name)                                                                                      # name
        self.rm = rm                                                                                                # device manager
        self.goal: Optional[Tuple[float, float]] = goal_pos                                                         # goal position
        self.mem_key = memory_key                                                                                   # memory key to read goal
        self.wp_tol = wp_tol                                                                                        # waypoint tolerance
        self.path: List[Tuple[float, float]] = []                                                                   # current path
        self.idx: int = 0                                                                                           # waypoint index
        self.stuck: int = 0                                                                                         # stuck counter
        self.last: Optional[float] = None                                                                           # last distance-to-goal

    def initialise(self) -> None:
        self.rm.memory.set("suppress_reactive", False)                                                              # allow reflexes while driving
        self.stuck, self.last = 0, None                                                                             # reset progress tracking
        if self.mem_key:                                                                                            # pull goal from memory
            self.goal = self.rm.memory.get(self.mem_key)                                                            # memory read
            if not self.goal:
                self.path, self.idx = [], 0                                                                         # nothing to do
                return
        nav = self.rm.memory.get("navigation")                                                                      # planner iface
        if not nav:
            self.path, self.idx = ([self.goal] if self.goal else []), 0                                             # no planner -> direct
            return
        cur = self.rm._position()                                                                                   # current pose
        self.path = nav.plan_path_to_goal(cur, self.goal) or [self.goal]                                            # plan or fallback
        self.idx = 0                                                                                                # start at first
        self.rm.memory.set("navigation_target", self.goal)                                                          # for UI/debug

    def update(self) -> py_trees.common.Status:
        if self.idx >= len(self.path):                                                                              # path done
            self.rm._stop()                                                                                         # stop
            self.rm.memory.set("suppress_reactive", False)                                                          # restore reflexes
            self.rm.memory.set("navigation_target", None)                                                           # clear UI target
            return py_trees.common.Status.SUCCESS                                                                   # done

        target = self.path[self.idx]                                                                                # current waypoint
        pos = self.rm._position()                                                                                   # current position
        dist = compute_distance(pos, target)                                                                        # to waypoint
        dist_goal = compute_distance(pos, self.goal) if self.goal else float('inf')                                 # to goal

        if self.last is not None:
            self.stuck = self.stuck + 1 if abs(dist_goal - self.last) < 0.02 else 0                                 # progress check
        self.last = dist_goal                                                                                       # update last

        if self.stuck > 50 and dist_goal < 0.25:                                                                    # accept success if near and stalled
            self.rm._stop(); self.rm.memory.set("suppress_reactive", False); self.rm.memory.set("navigation_target", None)  # cleanup
            return py_trees.common.Status.SUCCESS                                                                   # done

        eff_tol = (0.6 if dist_goal < 0.25 else (0.50 if self.idx >= len(self.path) - 2 else self.wp_tol))          # adaptive tol

        if dist < eff_tol:                                                                                          # reached waypoint
            self.idx += 1                                                                                           # advance
            if self.idx >= len(self.path):                                                                          # finished
                self.rm._stop(); self.rm.memory.set("suppress_reactive", False); self.rm.memory.set("navigation_target", None)  # cleanup
                return py_trees.common.Status.SUCCESS                                                               # done
            return py_trees.common.Status.RUNNING                                                                   # continue

        if self.idx >= len(self.path) - 1 and dist_goal < 0.5 and hasattr(self.rm, '_lidar_front_min'):             # safety stop near goal
            fm = self.rm._lidar_front_min()                                                                         # type: ignore[attr-defined]
            if fm is not None and fm < 0.55:
                self.rm._stop(); self.rm.memory.set("suppress_reactive", False); self.rm.memory.set("navigation_target", None)  # cleanup
                return py_trees.common.Status.SUCCESS                                                               # done

        if hasattr(self.rm, 'move_towards'):                                                                         # drive towards waypoint
            speed = 0.15 if dist_goal < 0.3 else 0.20 if dist_goal < 0.6 else 0.30                                  # slow down near goal
            self.rm.move_towards(target, max_speed=speed)                                                            # type: ignore[attr-defined]
        return py_trees.common.Status.RUNNING                                                                        # keep going

class WaitForContact(py_trees.behaviour.Behaviour):
    def __init__(self, rm: RobotDeviceManager, max_wait=1.5, threshold=-1.0, name="Wait For Contact") -> None:
        super().__init__(name)                                                                                      # name
        self.rm = rm                                                                                                # device manager
        self.max_wait = max_wait                                                                                    # maximum wait seconds
        self.threshold = threshold                                                                                  # contact threshold
        self.start: Optional[float] = None                                                                           # start time

    def initialise(self) -> None:
        self.start = rtime(self.rm.robot)                                                                           # start counting
        for nm in ('gripper_left_finger_joint', 'gripper_right_finger_joint'):
            m = dev(self.rm.robot, nm)                                                                              # motor
            safe(lambda: m.enableForceFeedback(self.rm.timestep))                                                   # enable force sensing

    def update(self) -> py_trees.common.Status:
        if self.start is None:
            return py_trees.common.Status.FAILURE                                                                   # defensive
        if rtime(self.rm.robot) - self.start > self.max_wait:                                                       # timeout succeeds
            return py_trees.common.Status.SUCCESS                                                                   # done

        lm = dev(self.rm.robot,'gripper_left_finger_joint')                                                         # left motor
        rmot = dev(self.rm.robot,'gripper_right_finger_joint')                                                      # right motor
        lf = safe(lambda: lm.getForceFeedback(), 0.0) if lm else 0.0                                                # left force
        rf = safe(lambda: rmot.getForceFeedback(), 0.0) if rmot else 0.0                                            # right force

        return (py_trees.common.Status.SUCCESS
                if (lf < self.threshold or rf < self.threshold)
                else py_trees.common.Status.RUNNING)                                                                # contact?/keep waiting

class DriveForwardTime(py_trees.behaviour.Behaviour):
    def __init__(self, rm: RobotDeviceManager, duration: float, speed: float = 0.2, name="Drive Forward") -> None:
        super().__init__(name)                                                                                      # name
        self.rm = rm                                                                                                # device manager
        self.duration = duration                                                                                    # drive time
        self.speed = speed                                                                                          # forward speed
        self.start: Optional[float] = None                                                                           # start time

    def initialise(self) -> None:
        self.start = rtime(self.rm.robot)                                                                           # timestamp start

    def update(self) -> py_trees.common.Status:
        if self.start is None:
            return py_trees.common.Status.FAILURE                                                                   # defensive
        if rtime(self.rm.robot) - self.start >= self.duration:                                                      # done by time
            self.rm._stop()                                                                                         # stop
            return py_trees.common.Status.SUCCESS                                                                   # done
        self.rm._set_wheel_speeds(self.speed, self.speed)                                                           # constant forward
        return py_trees.common.Status.RUNNING                                                                       # keep moving

class DriveUntilGripperContact(py_trees.behaviour.Behaviour):
    def __init__(self, rm: RobotDeviceManager, max_duration=3.0, speed=0.12, name="Drive Until Contact") -> None:
        super().__init__(name)                                                                                      # name
        self.rm = rm                                                                                                # device manager
        self.max_duration = max_duration                                                                            # safety timeout
        self.speed = speed                                                                                          # creep speed
        self.start_time: Optional[float] = None                                                                      # start time
        self.start_pos: Optional[Tuple[float, float]] = None                                                         # start pos
        self.init_wait = 0.2                                                                                        # small spin-up

    def initialise(self) -> None:
        self.start_time, self.start_pos = rtime(self.rm.robot), self.rm._position()                                 # capture state
        self.rm.memory.set("suppress_reactive", True)                                                                # avoid reflex brake
        for nm in ('gripper_left_finger_joint','gripper_right_finger_joint'):
            m = dev(self.rm.robot, nm)                                                                              # motor
            if m:
                safe(lambda: m.enableForceFeedback(self.rm.timestep))                                               # ensure sensing on

    def update(self) -> py_trees.common.Status:
        if self.start_time is None or self.start_pos is None:
            return py_trees.common.Status.FAILURE                                                                   # defensive

        elapsed = rtime(self.rm.robot) - self.start_time                                                            # elapsed seconds
        if elapsed < self.init_wait:                                                                                # begin rolling first
            self.rm._set_wheel_speeds(self.speed, self.speed)                                                       # gentle start
            return py_trees.common.Status.RUNNING                                                                   # continue

        lm = dev(self.rm.robot,'gripper_left_finger_joint')                                                         # left motor
        rmot = dev(self.rm.robot,'gripper_right_finger_joint')                                                      # right motor
        lf = safe(lambda: lm.getForceFeedback(), 0.0) if lm else 0.0                                                # left force
        rf = safe(lambda: rmot.getForceFeedback(), 0.0) if rmot else 0.0                                            # right force

        if lf < -2.5 or rf < -2.5:                                                                                  # contact threshold
            self.rm._stop()                                                                                         # stop
            return py_trees.common.Status.SUCCESS                                                                   # done
        if compute_distance(self.start_pos, self.rm._position()) > 0.5:                                             # distance safety
            self.rm._stop()                                                                                         # stop
            return py_trees.common.Status.SUCCESS                                                                   # done
        if elapsed > self.max_duration:                                                                             # time safety
            self.rm._stop()                                                                                         # stop
            return py_trees.common.Status.SUCCESS                                                                   # done

        self.rm._set_wheel_speeds(self.speed, self.speed)                                                           # keep creeping
        return py_trees.common.Status.RUNNING                                                                       # continue

class CheckJarDetection(py_trees.behaviour.Behaviour):
    def __init__(self, rm: RobotDeviceManager, name="Check Jar Detection") -> None:
        super().__init__(name)                                                                                      # name
        self.rm = rm                                                                                                # device manager

    def update(self) -> py_trees.common.Status:
        det = getattr(self.rm, "_jar_in_front", lambda: False)()                                                    # default False if missing
        return py_trees.common.Status.SUCCESS if det else py_trees.common.Status.FAILURE                            # status

class OpenGripper(py_trees.behaviour.Behaviour):
    def __init__(self, rm: RobotDeviceManager, name="Open Gripper"):
        super().__init__(name)                                                                                      # name
        self.rm = rm                                                                                                # device manager

    def update(self) -> py_trees.common.Status:
        print("Opening gripper to release object...")
        if hasattr(self.rm, "_set_gripper"):
            self.rm._set_gripper(open=True)                                                                        # open if available
        return py_trees.common.Status.SUCCESS                                                                       # continue regardless

class LiftBy(py_trees.behaviour.Behaviour):
    def __init__(self, rm: RobotDeviceManager, dz=0.05, name="Lift By"):
        super().__init__(name)                                                                                      # name
        self.rm = rm                                                                                                # device manager
        self.dz = dz                                                                                                # relative lift command
        self.start: Optional[float] = None                                                                           # start height
        self.target: Optional[float] = None                                                                          # target height

    def initialise(self) -> None:
        self.start = getattr(self.rm, "_get_lift_height", lambda: 0.0)()                                            # current height
        self.target = self.start + self.dz                                                                          # absolute target
        getattr(self.rm, "_set_lift_velocity", lambda v: None)(0.15)                                                # gentle speed
        getattr(self.rm, "_set_lift_target", lambda p: None)(self.target)                                           # send target

    def update(self) -> py_trees.common.Status:
        h = getattr(self.rm, "_get_lift_height", lambda: 0.0)()                                                     # current height
        return (py_trees.common.Status.SUCCESS
                if (self.target is not None and abs(h - self.target) < 0.005)
                else py_trees.common.Status.RUNNING)                                                                # within 5mm?

class MarkJarPlaced(py_trees.behaviour.Behaviour):
    def __init__(self, rm: RobotDeviceManager, idx: int, name="Mark Placed"):
        super().__init__(name)                                                                                      # name
        self.rm = rm                                                                                                # device manager
        self.idx = idx                                                                                              # jar index

    def update(self) -> py_trees.common.Status:
        placed = self.rm.memory.get("placed_count", 0) + 1                                                          # increment counter
        self.rm.memory.set("placed_count", placed)                                                                  # write back
        self.rm.memory.set("suppress_reactive", False)                                                              # restore reflexes
        self.rm.memory.set("navigation_target", None)                                                               # clear UI target
        print(f"Jar {self.idx+1} placed successfully! ({placed}/3 total)")                                                              # progress log
        return py_trees.common.Status.SUCCESS                                                                       # done
