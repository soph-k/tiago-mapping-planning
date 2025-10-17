from typing import Dict, List, Optional, Tuple                     # type hints
import numpy as np                                                # math helpers
import py_trees                                                   # behavior trees

from config import RobotConfig                                    # robot constants
from utils import clamp, compute_distance, wrap_angle, standoff, rtime, dev, safe
from utils import RobotDeviceManager                              # base helper (robot + memory)
from arms import (                                                # small BT building blocks
    MoveToPose, RotateToTarget, MoveAlongPlannedPath, DriveForwardTime,
    RetreatFromPosition, WaitSeconds, CheckJarDetection, OpenGripper,
    LiftBy, MarkJarPlaced, DriveUntilGripperContact, ComputeStandoffToTable
)

class PickPlaceController(RobotDeviceManager):
    def __init__(self, robot, memory) -> None:
        super().__init__(robot, memory)                           # init base fields
        self.tree = None                                          # py_trees root
        self.setup_behavior_tree()                                # build tree once

    # ----------------------------- Base motion helper -----------------------------
    def move_towards(self, target: Tuple[float, float], max_speed=1.5, turn_gain=2.0) -> bool:
        # Drive the base toward (x,y); return True when close enough.                 #
        cur, yaw = self._position(), self._orientation()                              # current pose
        dx, dy = target[0] - cur[0], target[1] - cur[1]                               # to goal
        dist = float(np.hypot(dx, dy))                                                # distance
        if dist < 0.6: self._stop(); return True                                      # arrived?
        desired = np.arctan2(dy, dx)                                                  # target bearing
        ang_err = wrap_angle(desired - yaw)                                           # -pi,pi
        speed_cap = min(max_speed, 0.4) if dist < 0.8 else max_speed                  # slow near goal
        base = speed_cap * min(dist / 1.5, 1.0)                                       # forward term
        L = clamp(base - turn_gain * ang_err, -1.5, 1.5)                              # left wheel
        R = clamp(base + turn_gain * ang_err, -1.5, 1.5)                              # right wheel
        # reactive avoidance                                  #
        from navigation import apply_reactive_avoidance
        L, R = apply_reactive_avoidance(L, R, yaw, self.memory, self._position)
        self._set_wheel_speeds(L, R)                                                  # command drive
        return False                                                                   # not there yet

    # ----------------------------- Simple jar detector ----------------------------
    def _jar_in_front(self) -> bool:
        # Lidar center check                                                           #
        lidar = self.memory.get("lidar")
        if lidar:
            try:
                rng = lidar.getRangeImage()                                           # list of ranges
                if rng:
                    s, e = int(len(rng) * 0.4), int(len(rng) * 0.6)                   # center slice
                    vals = [r for r in rng[s:e] if 0.01 < r < 10.0]                    # valid hits
                    if vals and min(vals) < 0.45: return True                          # within 45 cm
            except Exception:
                pass                                                                   # ignore sensor errors
        # Camera proximity check                                                       #
        cam = self.memory.get("camera")
        if cam:
            try:
                for obj in cam.getRecognitionObjects():                                # recognized objs
                    p = obj.getPosition()                                              # camera frame
                    if p and np.linalg.norm(p) < 0.3: return True                      # < 30 cm
            except Exception:
                pass
        return False                                                                    # nothing close

    # ----------------------------- Per-jar sequence --------------------------------
    def create_jar_sequence(self, i: int) -> py_trees.composites.Sequence:
        # Sequence for one jar         #
        jar_pos  = RobotConfig.JAR_POSITIONS[i]                                        
        drop_pos = RobotConfig.DROPOFF_POINTS[min(i, len(RobotConfig.DROPOFF_POINTS)-1)]  
        print(f"Starting sequence for Jar {i+1} at {jar_pos[:2]}")
        seq = py_trees.composites.Sequence(f"Process Jar {i}", memory=True)            # sequence node
        seq.add_child(MoveToPose(self, 'safe',  f"Safe Start Jar {i}"))                 # tuck arm
        seq.add_child(RotateToTarget(self, jar_pos[:2], f"Rotate to Jar {i}"))          # face jar
        seq.add_child(MoveToPose(self, 'reach', f"Extend to Jar {i}"))                  # reach pose
        sel = py_trees.composites.Selector(f"Move or Detect Jar {i}", memory=False)
        sel.add_child(CheckJarDetection(self, f"Check Jar {i} Detection"))              # success if seen
        sel.add_child(MoveAlongPlannedPath(self, jar_pos[:2], name=f"A* to Jar {i}"))   # else walk path
        seq.add_child(sel)

        # gentle nudge to seat gripper                                                 #
        print(f"Approaching Jar {i+1} for pickup...")
        seq.add_child(DriveForwardTime(self,
                                       duration=RobotConfig.DRIVE_INTO_JAR_DURATION[i],
                                       speed=0.12,
                                       name=f"Drive Into Jar {i}"))

        print(f"Grabbing Jar {i+1}...")
        seq.add_child(MoveToPose(self, 'grab', f"Grab Jar {i}"))                        # close fingers
        seq.add_child(RetreatFromPosition(self, distance=0.7 if i == 2 else 0.48,
                                          name=f"Retreat from Jar {i}"))                # back out
        seq.add_child(MoveToPose(self, 'safe', f"Safe Carry Jar {i}"))                  # tuck for drive

        print(f"Moving to drop-off location for Jar {i+1}...")
        seq.add_child(ComputeStandoffToTable(self, i, name=f"Compute Standoff {i}"))    # pick standoff
        seq.add_child(RotateToTarget(self, drop_pos[:2], f"Face Table {i}"))            # face table
        seq.add_child(MoveAlongPlannedPath(self, memory_key=f"table_standoff_{i}",
                                           name=f"A* to Table Standoff {i}"))           # go to standoff
        seq.add_child(RotateToTarget(self, drop_pos[:2], f"Align to Table {i}"))        # square up

        # careful place for jar #3                                                     #
        if i == 2:
            print(f"Placing Jar {i+1} (special careful handling)...")
            seq.add_child(DriveUntilGripperContact(self, max_duration=3.0, speed=0.06,
                                                   name=f"Slow Nudge to Table {i}"))    # slow push
            seq.add_child(WaitSeconds(self, 1.0, name=f"Long Settle before place {i}")) # longer settle
            seq.add_child(MoveToPose(self, 'place', f"Slow Extend to Table {i}", speed=0.15))  # slow extend
            seq.add_child(WaitSeconds(self, 0.5, name=f"Stabilize before release {i}")) # stop wiggles
            seq.add_child(OpenGripper(self, name=f"Careful Open Gripper {i}"))          # open fingers
            seq.add_child(WaitSeconds(self, 0.3, name=f"Wait for jar to settle {i}"))   # let settle
        else:
            print(f"Placing Jar {i+1} (standard handling)...")
            seq.add_child(DriveUntilGripperContact(self, max_duration=2.0, speed=0.10,
                                                   name=f"Nudge to Table {i}"))         # normal push
            seq.add_child(WaitSeconds(self, 0.5, name=f"Settle before place {i}"))      # quick settle
            seq.add_child(MoveToPose(self, 'place', f"Extend to Table {i}"))             # extend
            seq.add_child(OpenGripper(self, name=f"Open Gripper {i}"))                  # release

        print(f"Releasing Jar {i+1} and retreating...")
        seq.add_child(WaitSeconds(self, 0.3, name=f"Settle after open {i}"))            # avoid snag
        seq.add_child(LiftBy(self, dz=0.06, name=f"Retreat Up {i}"))                    # clear vertically
        seq.add_child(MoveToPose(self, 'safe', f"Back to Safe {i}"))                    # tuck again
        seq.add_child(MarkJarPlaced(self, i, name=f"Mark Placed {i}"))                  # count placed

        return seq                                                                      # finished sub-tree

    # ----------------------------- Build BT once -----------------------------------
    def setup_behavior_tree(self) -> None:
        # Root: retry each jar sequence, then ensure final safe pose.                  #
        print(f"Building behavior tree for {len(RobotConfig.JAR_POSITIONS)} jars...")
        root = py_trees.composites.Sequence("Pick Place All Jars", memory=True)         # root seq
        for i in range(len(RobotConfig.JAR_POSITIONS)):                                  # for each jar
            attempt = py_trees.decorators.Retry(child=self.create_jar_sequence(i),
                                                num_failures=2,
                                                name=f"Retry Jar {i}")                  # up to 2 retries
            root.add_child(attempt)                                                     # add sub-seq
        root.add_child(MoveToPose(self, 'safe', "Final Safe Position"))                 # always end safe
        self.tree = py_trees.trees.BehaviourTree(root)                                  # build tree
        safe(lambda: self.tree.setup(timeout=15))                                       # guarded setup
        print("Behavior tree setup complete!")

    # ----------------------------- Tick interface -----------------------------------
    def run(self) -> str:
        # Tick the tree once; return 'SUCCESS' / 'FAILURE' / 'RUNNING'.                #
        if not self.tree: return "FAILURE"                                              # missing tree
        self.tree.tick()                                                                # advance BT
        st = self.tree.root.status                                                      # enum
        if st == py_trees.common.Status.SUCCESS: return "SUCCESS"                       # done
        if st == py_trees.common.Status.FAILURE: return "FAILURE"                       # failed
        return "RUNNING"                                                                # keep going

    # ----------------------------- Gripper & lift helpers ---------------------------
    def _set_gripper(self, open: bool = True) -> None:
        # Open/close both fingers 
        pos = 0.045 if open else 0.0                                                    # target pos
        for nm in ('gripper_left_finger_joint', 'gripper_right_finger_joint'):          # both sides
            m = dev(self.robot, nm)
            if m: m.setPosition(pos); m.setVelocity(RobotConfig.GRIPPER_SPEED)         # gentle speed

    def _get_lift_height(self) -> float:
        # Current torso height, 0 if not available.                               
        s = dev(self.robot, 'torso_lift_joint_sensor')                                  # sensor
        return safe(lambda: s.getValue(), 0.0) if s else 0.0                            # guarded read

    def _set_lift_velocity(self, v: float) -> None:
        # Set lift velocity .                                           
        m = dev(self.robot, 'torso_lift_joint')
        if m: m.setVelocity(v)                                                          # set vel

    def _set_lift_target(self, p: float) -> None:
        # Set lift target height m.                                                   
        m = dev(self.robot, 'torso_lift_joint')
        if m: m.setPosition(p)                                                          # set pos

    def _lidar_front_min(self) -> Optional[float]:
        # Min forward lidar range , or None if unavailable.               
        lidar = self.memory.get("lidar")
        try:
            rng = lidar.getRangeImage() if lidar else None                              # scan
            if not rng: return None                                                     # no data
            s, e = len(rng)//3, 2*len(rng)//3                                           # center slice
            vals = [r for r in rng[s:e] if 0.01 < r < 10.0]                             # valid hits
            return min(vals) if vals else None                                          # nearest
        except Exception:
            return None                                                                  # on error
