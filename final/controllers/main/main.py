import copy
import time
from collections import deque
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from controller import Robot
import py_trees
import heapq

class RobotConfig:                  # Collect all of the constants for the robot in one place
    MAP_SIZE: int = 300             # Map height in pixels 
    MAP_WIDTH: int = 200            # Map width in pixels 
    MAPPING_INTERVAL: int = 3       # Ticks between mapping updates
    TRAJECTORY_INTERVAL: int = 8    # Ticks between trajectory updates
    TURN_SPEED_MAX: float = 4.0     # Max turning speed when rotating
    DRIVE_SPEED_MAX: float = 3.0    # Max driving speed when moving forward
    DIST_TOL: float = 0.4           # Distance threshold to consider a waypoint reached 
    Kp_angle: float = 0.8           # Proportional gain for steering towards a waypoint
    GRIPPER_SPEED: float = 0.05     # Speed for the gripper fingers
    TORSO_SPEED: float = 0.07       # Speed for the torso lift
    ARM_SPEED: float = 0.5          # Speed for the arm joints

    MAPPING_WAYPOINTS: List[Tuple[float, float]] = [    # Waypoints that guide the robot through the environment during mapping.
        (0.34, -0.19), (0.47, -0.94), (0.8, -1.8), (0.60, -2.4),
        (0.4, -2.8), (-0.51, -3.0), (-1.2, -2.7), (-1.4, -2.50),
        (-1.5, -1.25), (-1.7, -0.50), (-1.5, 0.10), (-0.70, 0.2),
        (-0.6, 0.3), (-0.20, 0.1), (-0.1, 0.01), (0.24, -0.09)
    ]  # Sequence of (x, y) waypoints for exploration
  
    JAR_POSITIONS: List[Tuple[float, float, float]] = [ # Known jar positions in the world as (x, y, z) coordinates. 
        (1.7143, 0.7042, 0.8894),     # Jar 1
        (1.7123, -0.3007, 0.8894),    # Jar 2
        (1.98, 0.49, 0.8894)          # Jar 3 Edited this so its not the true jar location but only way to ensure the arm reaches the jar.
    ]
   
    DROPOFF_POINTS: List[Tuple[float, float, float]] = [ # Coordinates on the table where jars should be placed after pickup. 
        (-0.08, -0.3, 0.8),
        (-0.01, -0.25, 0.8),
        (-0.13, -0.33, 0.8)
    ]  

def wrap_angle(angle: float) -> float:  # Wrap an angle in radians into the range pi and -pi.
    while angle > np.pi:                # While the angle is greater than pi, subtract 2pi.
        angle -= 2 * np.pi
    while angle < -np.pi:               # While the angle is less than -pi, add 2pi.
        angle += 2 * np.pi 
    return angle

def world_to_pixel(                     # Convert world coordinates to pixel coordinates on the map display.
    x: float,  
    y: float,
    map_width: int = RobotConfig.MAP_WIDTH, # The width of the map in pixels. 
    map_size: int = RobotConfig.MAP_SIZE, # The height of the map in pixels.
) -> Tuple[int, int]:                   # Return the pixel coordinates on the map.
    px = int(40 * (x + 2.25))           # Convert metres into pixel indices.
    py = int(-300 / 5.6666 * (y - 1.6633)) # Convert metres into pixel indices.
    px = min(max(px, 0), map_width - 1) # Lie within the bounds of the map.
    py = min(max(py, 0), map_size - 1)
    return px, py

def compute_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float: # Compute the Euclidean distance between two flat points.
    return float(np.linalg.norm(np.array(pos1) - np.array(pos2)))

def compute_motor_commands(             # Based on the angle error and distance to the target.
    alpha: float,
    rho: float,
    config: RobotConfig = RobotConfig,
) -> Tuple[float, float]:               # Return the left and right wheel commands.
    if abs(alpha) > (np.pi / 2.2):      # If the angle error is large, rotate in place.
        turn_factor = min(1.0, abs(alpha) / np.pi) # Scale the turn speed based on how far off we are.
        turn_speed = 0.6 + (config.TURN_SPEED_MAX - 0.6) * turn_factor # The value ranges from 0.6 to turn max.
        direction = 1 if alpha > 0 else -1 # The direction of the turn.
        left_cmd = -direction * turn_speed # The left wheel command.
        right_cmd = direction * turn_speed # The right wheel command.
    else: 
        if rho > 0.5: 
            base_speed = config.DRIVE_SPEED_MAX * ( # Drive at nearly max speed, but slow down when turning sharply. 
                1 - min(abs(alpha) / (np.pi / 2), 1) * 0.6
            )
        else:                               # Close to goal, then reduce speed linearly with distance.
            base_speed = config.DRIVE_SPEED_MAX * 0.6 * (rho / 0.5)
        base_speed = max(base_speed, 0.6)   # Don't let the base speed drop below 0.6 to prevent stalling.

        correction = config.Kp_angle * alpha  
        max_correction = base_speed * 0.85 
        correction = np.clip(correction, -max_correction, max_correction) 
        left_cmd = base_speed - correction  # The left wheel command.
        right_cmd = base_speed + correction # The right wheel command.

        left_cmd = np.clip(left_cmd, -config.DRIVE_SPEED_MAX, config.DRIVE_SPEED_MAX) # Puts the wheel commands to the maximum drive speed.
        right_cmd = np.clip(right_cmd, -config.DRIVE_SPEED_MAX, config.DRIVE_SPEED_MAX) # Puts the wheel commands to the maximum drive speed.
    return float(left_cmd), float(right_cmd) # Return the left and right wheel commands.
class MemoryBoard:                          # A simple key/value store for sharing state between subsystems.
    def __init__(self) -> None:             # Store everything in a simple dictionary.
        self.data: Dict[str, object] = {} 
        self._init_defaults()               # Populate the memory board with a set of default entries.

    def _init_defaults(self) -> None:       # Populate the memory board with a set of default entries.
        self.data.update({      
            "mapping_waypoints": RobotConfig.MAPPING_WAYPOINTS,             # The exploration waypoints defined in RobotConfig.
            "jar_positions": RobotConfig.JAR_POSITIONS,                     # Known jar positions.
            "dropoff_points": RobotConfig.DROPOFF_POINTS,                   # Dropoff points for jars.
            "picked_positions": [],
            "current_dropoff_index": 0,                                     # Indices to track progress through jars and drop points.
            "current_jar_index": 0,
            "mapping_complete": False,                                      # Indicate completion of various tasks.
            "cspace_complete": False,
            "navigation_active": False,                                     # Indicate whether navigation or jar navigation is active.
            "jar_navigation_active": False,
            "recognized_objects": []                                        # A list of objects recognised by the system.
        })

    def set(self, key: str, value: object) -> None: # Store an object under the given key.
        self.data[key] = value

    def get(self, key: str, default: Optional[object] = None) -> object: # Retrieve an object from the board, returning default if absent.
        return self.data.get(key, default)

    def has(self, key: str) -> bool:        # Return true if the board contains a value for key.
        return key in self.data

    def read(self, key: str, default: Optional[object] = None) -> object: # Retrieve an object from the board, returning default if absent.
        return self.get(key, default)

    def write(self, key: str, value: object) -> None: # Store an object under the given key.    
        self.set(key, value)


class RobotDeviceManager:                    # A base class for managing robot devices.
    def __init__(self, robot: Robot, memory: MemoryBoard) -> None: # Initialize the robot device manager.
        self.robot = robot
        self.memory = memory
        self.timestep = int(getattr(robot, "getBasicTimeStep", lambda: 32)()) # Get the simulation time step in milliseconds.

    def _position(self) -> Tuple[float, float]: # Return the current x/y position of the robot. 
        gps = self.memory.get("gps")            # Get the GPS sensor.
        return gps.getValues()[:2] if gps else (0.0, 0.0) 

    def _orientation(self) -> float:            # Return the robot's heading as an angle in radians.
        compass = self.memory.get("compass")    # Get the compass sensor.
        if compass:
            values = compass.getValues() 
            return float(np.arctan2(values[0], values[1])) # Return the heading as an angle in radians.
        return 0.0

    def _set_wheel_speeds(self, left: float, right: float) -> None: # Set the velocities of the left and right wheels.
        motorL = self.memory.get("motorL")      # Get the left motor.
        motorR = self.memory.get("motorR")      # Get the right motor.
        if motorL and motorR:
            motorL.setVelocity(left)            # Set the velocity of the left motor.
            motorR.setVelocity(right)           # Set the velocity of the right motor.

    def _stop(self) -> None:                    # Function to stop both wheels.
        self._set_wheel_speeds(0.0, 0.0)        # Set the velocity of the left and right wheels to 0.

    def _front_clear(self, thresh=0.35):        # Check if the front is clear of obstacles.
        lidar = self.memory.get("lidar")
        try:
            ranges = lidar.getRangeImage()
            if not ranges: return True
            c0, c1 = int(len(ranges)*0.45), int(len(ranges)*0.55)
            fr = [r for r in ranges[c0:c1] if np.isfinite(r) and r>0]
            return (min(fr) if fr else float('inf')) > thresh
        except Exception:
            return True


class BTAction(py_trees.behaviour.Behaviour):   # Wrap a function into a py_trees behavior.
    def __init__(self, func: Callable[[], str]) -> None: 
        super().__init__(name=getattr(func, "__name__", "BTAction")) # Initialize the behavior.
        self.func = func

    def update(self) -> py_trees.common.Status: 
        result = self.func()                    # Call the underlying function.
        if result == "SUCCESS":
            return py_trees.common.Status.SUCCESS 
        elif result == "FAILURE":
            return py_trees.common.Status.FAILURE # Return failure.
        else:
            return py_trees.common.Status.RUNNING # Return running.


class ArmPoseController(RobotDeviceManager):    # Move the arm into named poses.
    def __init__(self, robot: Robot, memory: MemoryBoard, pose_name: str = 'safe') -> None: 
        super().__init__(robot, memory)         # Initialize the robot device manager.  
        self.pose_name = pose_name              # The name of the pose to move to.
        self.threshold = 0.05                   # Allow a small error tolerance when checking if joints have reached their targets.  
        self.threshold_force = -2.5             # was -5.0             
        self.configurations: Dict[str, Dict[str, float]] = { # Define joint targets for each named pose. 
            'safe': {                           
                'torso_lift_joint': 0.05,      
                'arm_1_joint': 1.600,
                'arm_2_joint': np.pi / 4,
                'arm_3_joint': -2.815, 
                'arm_4_joint': 0.8854, 
                'arm_5_joint': 0.0, 
                'arm_6_joint': 0.0, 
                'arm_7_joint': np.pi / 2, 
                'gripper_left_finger_joint': 0.0, 
                'gripper_right_finger_joint': 0.0, 
                'head_1_joint': 0.0, 
                'head_2_joint': 0.0 
            },
            'reach': {
                'torso_lift_joint': 0.11,
                'arm_1_joint': 1.600,
                'arm_2_joint': np.pi / 4,
                'arm_3_joint': 0.0,
                'arm_4_joint': 0.8854,
                'arm_5_joint': 0.0,
                'arm_6_joint': 0.0,
                'arm_7_joint': np.pi / 2,
                'gripper_left_finger_joint': 0.045,
                'gripper_right_finger_joint': 0.045,
                'head_1_joint': 0.0,
                'head_2_joint': 0.0
            },
            'reach_open': {
                'torso_lift_joint': 0.11,
                'arm_1_joint': 1.600,
                'arm_2_joint': np.pi / 4,
                'arm_3_joint': 0.0,
                'arm_4_joint': 0.8854,
                'arm_5_joint': 0.0,
                'arm_6_joint': 0.0,
                'arm_7_joint': np.pi / 2,
                'gripper_left_finger_joint': 0.045,  
                'gripper_right_finger_joint': 0.045,  
                'head_1_joint': 0.0,
                'head_2_joint': 0.0
            },
            'grab': {
                'torso_lift_joint': 0.11,
                'arm_1_joint': 1.600,
                'arm_2_joint': np.pi / 4,
                'arm_3_joint': 0.0,
                'arm_4_joint': 0.8854,
                'arm_5_joint': 0.0,
                'arm_6_joint': 0.0,
                'arm_7_joint': np.pi / 2,
                'gripper_left_finger_joint': 0.0,   
                'gripper_right_finger_joint': 0.0   
            },
            'place': {
                'torso_lift_joint': 0.09,   
                'arm_1_joint': 1.6,
                'arm_2_joint': 0.9,
                'arm_3_joint': 0.0,
                'arm_4_joint': 0.8854,
                'arm_5_joint': 0.0,
                'arm_6_joint': 0.0,
                'arm_7_joint': 1.576,
                'gripper_left_finger_joint': 0.0,
                'gripper_right_finger_joint': 0.0,
                'head_1_joint': 0.0,
                'head_2_joint': 0.0
            }
        }

        if self.pose_name not in self.configurations:   # If the requested pose is unknown, default to safe.
            print(f" Unknown pose '{self.pose_name}', defaulting to 'safe'.") # Print a message to the console.
            self.pose_name = 'safe'                     # Set the pose name to safe.
        
        self.target_positions: Dict[str, float] = copy.deepcopy( # Store the desired joint targets for the chosen pose.
            self.configurations[self.pose_name]         # Get the desired joint targets for the chosen pose.
        )
        self.joint_motors: Dict[str, object] = {}       # Dictionary for motors.
        self.joint_sensors: Dict[str, object] = {}      # Dictionary for position sensors.

    def setup(self) -> None:                            # Retrieve motor devices for each joint mentioned in the target.
        for joint_name in self.target_positions:        # For each joint in the target positions.
            motor = getattr(self.robot, "getDevice", lambda name: None)(joint_name) # Get the motor device for the joint.
            if motor:                                    # If the motor device is found.
                self.joint_motors[joint_name] = motor    # Store the motor device in the dictionary.

        sensor_names = [
            'torso_lift_joint_sensor', 'arm_1_joint_sensor', 'arm_2_joint_sensor',
            'arm_3_joint_sensor', 'arm_4_joint_sensor', 'arm_5_joint_sensor',
            'arm_6_joint_sensor', 'arm_7_joint_sensor',
            'gripper_left_sensor_finger_joint', 'gripper_right_sensor_finger_joint',
            'head_1_joint_sensor', 'head_2_joint_sensor'
        ]
        for s_name in sensor_names:                        # For each sensor in the sensor names.
            sensor = getattr(self.robot, "getDevice", lambda name: None)(s_name) # Get the sensor device for the sensor.
            if sensor:                                      # If the sensor device is found.
                try:
                    sensor.enable(self.timestep)            # Enable the sensor.
                except Exception:
                    pass
                self.joint_sensors[s_name] = sensor         # Store the sensor device in the dictionary.


    def initialise(self) -> None:                           # Send target positions and velocities to each motor.
        for joint, goal in self.target_positions.items():    # For each joint in the target positions.
            motor = self.joint_motors.get(joint)             # Get the motor device for the joint.
            if motor:                                        # If the motor device is found.
                motor.setPosition(goal)                      # Set the desired position for this motor.  Webots will move the joint towards this angle over time.
                if 'torso' in joint:                         # If the joint is a torso joint.
                    motor.setVelocity(RobotConfig.TORSO_SPEED) # Set the velocity of the motor.
                elif 'gripper' in joint:                    # If the joint is a gripper joint.
                    motor.setVelocity(RobotConfig.GRIPPER_SPEED) # Set the velocity of the motor.
                else:                                       # If the joint is an arm joint.
                    if self.pose_name == 'place':
                        motor.setVelocity(0.2)  # instead of RobotConfig.ARM_SPEED for arm joints
                    else:
                        motor.setVelocity(RobotConfig.ARM_SPEED) # Set the velocity of the motor.
        if self.pose_name == 'grab': # When grabbing, enable force feedback so we can detect a firm grip.
            left_motor = self.joint_motors.get('gripper_left_finger_joint') # Get the left motor device for the joint.
            right_motor = self.joint_motors.get('gripper_right_finger_joint') # Get the right motor device for the joint.
            if left_motor:                                    # If the left motor device is found.
                try:
                    left_motor.enableForceFeedback(self.timestep) # Enable force feedback for the left motor.
                except Exception:
                    pass
            if right_motor:
                try:
                    right_motor.enableForceFeedback(self.timestep) # Enable force feedback for the right motor.
                except Exception:
                    pass

    def update(self) -> str:                                # Update the pose controller.
        mismatch = 0                                        # Initialize the mismatch counter.
        for joint, goal in self.target_positions.items():   # Check each joint sensor against its target and count mismatches.
            sensor = self.joint_sensors.get(f"{joint}_sensor") # Get the sensor device for the joint.
            if not sensor:                                   # If the sensor device is not found.
                continue
            try:
                error = abs(sensor.getValue() - goal)       # Get the error between the sensor and the goal.
            except Exception:
                error = 0.0                                 # If reading the sensor fails we assume zero error.
            if error > self.threshold:                      # If the error is greater than the threshold.
                mismatch += 1                               # Increment the mismatch counter.
        if self.pose_name == 'grab':                        
            if mismatch == 0:                               # If the mismatch is 0.
                left_motor = self.joint_motors.get('gripper_left_finger_joint') # Get the left motor device for the joint.
                right_motor = self.joint_motors.get('gripper_right_finger_joint') # Get the right motor device for the joint.
                if left_motor and right_motor:              # If the left and right motor devices are found.
                    try:
                        left_force = left_motor.getForceFeedback() # Get the force feedback for the left motor.
                        right_force = right_motor.getForceFeedback() # Get the force feedback for the right motor.
                        if (
                            left_force < self.threshold_force and 
                            right_force < self.threshold_force
                        ):
                            return "SUCCESS"
                    except Exception:                       # If an error occurs.
                        pass
                return "RUNNING"
        if mismatch == 0:                                   # If the mismatch is 0.
            print(f"Pose '{self.pose_name}' SUCCESS - All joints reached target") # Print a message to the console.
            return "SUCCESS"                                # Return success.
        else:                                               # If the mismatch is not 0.
            return "RUNNING"


class MoveToPose(py_trees.behaviour.Behaviour):             # Move the robot arm to a specific pose.
    def __init__(                                            # Initialize the move to pose behavior.
        self, 
        robot_manager: RobotDeviceManager,
        pose_name: str,
        name: Optional[str] = None,
    ) -> None: 
        super().__init__(name or f"Move to {pose_name}")    # Initialize the behavior.
        self.robot_manager = robot_manager                  # The robot manager.
        self.pose_name = pose_name                          # The name of the pose to move to.
        self.pose_controller: Optional[ArmPoseController] = None # The pose controller.

    def initialise(self) -> None: 
        print(f"Starting to move to pose: {self.pose_name}") # Print a message to the console.
        self.pose_controller = ArmPoseController(
            self.robot_manager.robot,                        # The robot.
            self.robot_manager.memory,                       # The memory.
            self.pose_name                                   # The name of the pose to move to.
        )  
        self.pose_controller.setup()                         # Setup the pose controller.
        self.pose_controller.initialise()                    # Initialise the pose controller.

    def update(self) -> py_trees.common.Status:             # Update the pose controller.   
        # At top of MoveToPose.update()
        self.robot_manager._stop()  # keep base still while joints settle
        if self.pose_controller is None:                    # If the pose controller hasn't been created, fail this behavior.
            return py_trees.common.Status.FAILURE
        result = self.pose_controller.update()              # Update the pose controller.
        if result == "SUCCESS":                             # If the pose controller has reached the target.
            print(f"Completed pose: {self.pose_name}")      # Print a message to the console.
            return py_trees.common.Status.SUCCESS           # Return success.
        else:                                               # If the pose controller has not reached the target.
            return py_trees.common.Status.RUNNING
            

class RotateToTarget(py_trees.behaviour.Behaviour):         # Rotate the robot to face a target position.
    def __init__(                                           # Initialize the rotate to target behavior.
        self, 
        robot_manager: RobotDeviceManager,
        target_pos: Tuple[float, float],
        name: str = "Rotate to Target",                     # The name of the behavior.
    ) -> None:
        super().__init__(name)                              # Initialize the behavior.
        self.robot_manager = robot_manager                  # Robot manager.
        self.target_pos = target_pos                        # The target position.

    def update(self) -> py_trees.common.Status:             # Update the rotate to target behavior.
        cur_pos = self.robot_manager._position()            # Compute current position.
        heading = self.robot_manager._orientation()         # Compute current orientation.
        dx = self.target_pos[0] - cur_pos[0]                # Compute the difference in x.
        dy = self.target_pos[1] - cur_pos[1]                # Compute the difference in y.
        target_angle = np.arctan2(dy, dx)                   # Compute the target angle.
        err = wrap_angle(target_angle - heading)            # Compute the error.
        if abs(err) < 0.02:                                 # If the error is less than 0.02.
            self.robot_manager._stop()                      # Stop the robot.
            return py_trees.common.Status.SUCCESS           # Return success.
        turn_speed = 1.5 * np.sign(err)                     # Compute the turn speed.
        turn_speed *= max(min(abs(err) / np.pi, 1.0), 0.2)  # Scale the turn speed relative to the error but never below 0.2.
        self.robot_manager._set_wheel_speeds(-turn_speed, turn_speed) # Set the wheel speeds.
        return py_trees.common.Status.RUNNING


class MoveToTarget(py_trees.behaviour.Behaviour):   # Drive the robot towards a target position.
    def __init__(                                   # Initialize the move to target behavior.
        self,
        robot_manager: RobotDeviceManager,          # The robot manager.
        target_pos: Tuple[float, float],            # The target position.
        max_speed: float = 1.5,                     # The maximum speed.
        name: str = "Move to Target",               # The name of the behavior.
    ) -> None:
        super().__init__(name)                      # Initialize the behavior.
        self.robot_manager = robot_manager          # The robot manager.
        self.target_pos = target_pos                # The target position.
        self.max_speed = max_speed                  # The maximum speed.

    def update(self) -> py_trees.common.Status:     # Update the move to target behavior.
        if isinstance(self.robot_manager, PickPlaceController): 
            reached = self.robot_manager.move_towards(self.target_pos, self.max_speed) # Use the robot manager's move_towards helper if available.
        else: 
            reached = compute_distance(self.robot_manager._position(), self.target_pos) < 0.5 # Compute the distance between the robot and the target.
        return py_trees.common.Status.SUCCESS if reached else py_trees.common.Status.RUNNING # Return success if the robot has reached the target, otherwise return running.


class RetreatFromPosition(py_trees.behaviour.Behaviour): # Back away a short distance from the starting point.
    def __init__(                                   # Initialize the retreat from position behaviour.
        self,
        robot_manager: RobotDeviceManager,
        distance: float = 0.48,
        name: str = "Retreat",
    ) -> None:
        super().__init__(name)                      # Initialize the behaviour.
        self.robot_manager = robot_manager          # The robot manager.
        self.distance = distance                    # The distance to retreat.
        self.start_pos: Optional[Tuple[float, float]] = None

    def initialise(self) -> None: 
        self.start_pos = self.robot_manager._position() # Record the position where the retreat begins.

    def update(self) -> py_trees.common.Status:     # Update the retreat from position behaviour.
        if self.start_pos is None:                  # If the start position is not found.
            return py_trees.common.Status.FAILURE   # Return failure.
        cur = self.robot_manager._position()        # Compute the current position.
        dist = compute_distance(cur, self.start_pos) # Compute the distance travelled backwards so far.
        if dist > self.distance:                     # If the distance travelled backwards is greater than the distance to retreat.
            self.robot_manager._stop()               # Stop the robot.
            return py_trees.common.Status.SUCCESS    # Return success.
        else:                                        # If the distance travelled backwards is less than the distance to retreat.
            self.robot_manager._set_wheel_speeds(-0.3, -0.3) # Set the wheel speeds.
            return py_trees.common.Status.RUNNING    # Return running.


class WaitAndOpenGripper(py_trees.behaviour.Behaviour): # Wait for a short period then open the gripper.
    def __init__(                                   # Initialize the wait and open gripper behaviour.
        self,
        robot_manager: RobotDeviceManager,
        wait_time: float = 0.5,                     # The time to wait before opening the gripper.
        name: str = "Wait and Open Gripper",        # The name of the behaviour.
    ) -> None:
        super().__init__(name)                      # Initialize the behaviour.
        self.robot_manager = robot_manager          # The robot manager.
        self.wait_time = wait_time                  # The time to wait before opening the gripper.
        self.start_time: Optional[float] = None     # The time when the wait begins.
        self.gripper_opened: bool = False           # Whether the gripper has been opened.

    def initialise(self) -> None:                   # Record the time when the wait begins.
        self.start_time = getattr(self.robot_manager.robot, "getTime", lambda: 0.0)() 
        self.gripper_opened = False                 # Set the gripper to closed.

    def update(self) -> py_trees.common.Status:     # Update the wait and open gripper behaviour.
        if self.start_time is None:                 # If the start time is not found.
            return py_trees.common.Status.FAILURE   # Return failure.
        current_time = getattr(self.robot_manager.robot, "getTime", lambda: 0.0)() # Get the current time.
        elapsed = current_time - self.start_time    # Compute the elapsed time.
        if elapsed < self.wait_time:                # If the elapsed time is less than the wait time.
            return py_trees.common.Status.RUNNING   # Return running.
        if not self.gripper_opened:                 # If the gripper is not already opened.
            left_motor = getattr(self.robot_manager.robot, "getDevice", lambda name: None)('gripper_left_finger_joint') # Get the left motor device for the joint.
            right_motor = getattr(self.robot_manager.robot, "getDevice", lambda name: None)('gripper_right_finger_joint') # Get the right motor device for the joint.
            if left_motor and right_motor:          # If the left and right motor devices are found.
                left_motor.setPosition(0.045)       # Set the position of the left motor.
                right_motor.setPosition(0.045)      # Set the position of the right motor.
                left_motor.setVelocity(RobotConfig.GRIPPER_SPEED) # Set the velocity of the left motor (max 0.05)
                right_motor.setVelocity(RobotConfig.GRIPPER_SPEED) # Set the velocity of the right motor (max 0.05)
            self.gripper_opened = True               # Set the gripper to opened.
        left_sensor = getattr(self.robot_manager.robot, "getDevice", lambda name: None)('gripper_left_sensor_finger_joint') # Get the left sensor device for the joint.
        right_sensor = getattr(self.robot_manager.robot, "getDevice", lambda name: None)('gripper_right_sensor_finger_joint') # Get the right sensor device for the joint.
        try:
            if left_sensor and right_sensor:        # If the left and right sensor devices are found.
                if left_sensor.getValue() < 0.02 and right_sensor.getValue() < 0.02: 
                    self.robot_manager._stop()      # Stop the robot.
                    return py_trees.common.Status.SUCCESS # Return success.
        except Exception:                           # If an error occurs.
            pass
        if elapsed > self.wait_time + 1.0:          # If the elapsed time is greater than the wait time plus 1 second.
            self.robot_manager._stop()              # Stop the robot.
            return py_trees.common.Status.SUCCESS   # Return success.
        return py_trees.common.Status.RUNNING       # Return running.


class WaitSeconds(py_trees.behaviour.Behaviour):    # Wait for a fixed duration.
    def __init__(self, robot_manager: RobotDeviceManager, duration: float, name: str = "Wait") -> None: 
        super().__init__(name)                      # Initialize the behaviour.
        self.robot_manager = robot_manager          # The robot manager.
        self.duration = duration                    # The duration to wait.
        self.start_time: Optional[float] = None     # The time when the wait begins.

    def initialise(self) -> None:                   # Record the time when the wait begins.
        self.start_time = getattr(self.robot_manager.robot, "getTime", lambda: 0.0)() 

    def update(self) -> py_trees.common.Status:     # Update the wait seconds behaviour.
        if self.start_time is None:                 # If the start time is not found.
            return py_trees.common.Status.FAILURE   # Return failure.
        current_time = getattr(self.robot_manager.robot, "getTime", lambda: 0.0)() # Get the current time.
        if current_time - self.start_time >= self.duration: # If the elapsed time is greater than or equal to the duration.
            return py_trees.common.Status.SUCCESS   # Return success.
        return py_trees.common.Status.RUNNING       # Return running.


class WaitForContact(py_trees.behaviour.Behaviour): # Wait until the gripper contacts an object.
    def __init__(self, robot_manager: RobotDeviceManager, 
                 max_wait: float = 1.5, threshold: float = -1.0,
                 name: str = "Wait For Contact") -> None:
        super().__init__(name)                  # Initialize the behaviour.
        self.robot_manager = robot_manager      # The robot manager.
        self.max_wait = max_wait                # The maximum wait time.
        self.threshold = threshold              # The threshold for contact.
        self.start_time: Optional[float] = None # The time when the wait begins.

    def initialise(self) -> None:               # Record the time when the wait begins.
        self.start_time = getattr(self.robot_manager.robot, "getTime", lambda: 0.0)() 
        left_motor = getattr(self.robot_manager.robot, "getDevice", lambda name: None)('gripper_left_finger_joint') 
        right_motor = getattr(self.robot_manager.robot, "getDevice", lambda name: None)('gripper_right_finger_joint') 
        if left_motor:                          # If the left motor device is found.
            try:                                # Try to enable force feedback on the left motor.
                left_motor.enableForceFeedback(self.robot_manager.timestep) # Enable force feedback on the left motor.
            except Exception:                   # If an error occurs.
                pass
        if right_motor:                         # If the right motor device is found.
            try: # Try to enable force feedback on the right motor.
                right_motor.enableForceFeedback(self.robot_manager.timestep) # Enable force feedback on the right motor.
            except Exception:
                pass

    def update(self) -> py_trees.common.Status: # Update the wait for contact behaviour.
        if self.start_time is None:             # If the start time is not found.
            return py_trees.common.Status.FAILURE # Return failure.
        current_time = getattr(self.robot_manager.robot, "getTime", lambda: 0.0)() # Get the current time.
        if current_time - self.start_time > self.max_wait: # If the elapsed time is greater than the maximum wait time.
            return py_trees.common.Status.SUCCESS # Return success.
        left_motor = getattr(self.robot_manager.robot, "getDevice", lambda name: None)('gripper_left_finger_joint') # Get the left motor device for the joint.
        right_motor = getattr(self.robot_manager.robot, "getDevice", lambda name: None)('gripper_right_finger_joint') # Get the right motor device for the joint.
        try:
            left_force = left_motor.getForceFeedback() if left_motor else 0.0 # Get the force feedback for the left motor.
        except Exception:
            left_force = 0.0                        # If an error occurs, set the left force to 0.0.
        try:
            right_force = right_motor.getForceFeedback() if right_motor else 0.0 # Get the force feedback for the right motor.
        except Exception:
            right_force = 0.0                       # If an error occurs, set the right force to 0.0.
        if (left_force < self.threshold) or (right_force < self.threshold): # If the left or right force is less than the threshold.
            return py_trees.common.Status.SUCCESS # Return success.
        return py_trees.common.Status.RUNNING       # Return running.


class DriveForwardTime(py_trees.behaviour.Behaviour):
    """Drive forward at constant speed for a fixed time"""
    def __init__(self, robot_manager: RobotDeviceManager, duration: float, speed: float = 0.2, name: str = "Drive Forward") -> None:
        super().__init__(name)
        self.robot_manager = robot_manager
        self.duration = duration
        self.speed = speed
        self.start_time: Optional[float] = None

    def initialise(self) -> None:
        self.start_time = getattr(self.robot_manager.robot, "getTime", lambda: 0.0)()

    def update(self) -> py_trees.common.Status:
        if self.start_time is None:
            return py_trees.common.Status.FAILURE
        
        current_time = getattr(self.robot_manager.robot, "getTime", lambda: 0.0)()
        elapsed = current_time - self.start_time
        
        if elapsed >= self.duration:
            self.robot_manager._stop()
            return py_trees.common.Status.SUCCESS
        
        self.robot_manager._set_wheel_speeds(self.speed, self.speed)
        return py_trees.common.Status.RUNNING


class DriveUntilGripperContact(py_trees.behaviour.Behaviour):
    """Drive forward slowly until gripper fingers detect contact with jar"""
    def __init__(self, robot_manager: RobotDeviceManager, max_duration: float = 3.0, speed: float = 0.12, name: str = "Drive Until Contact") -> None:
        super().__init__(name)
        self.robot_manager = robot_manager
        self.max_duration = max_duration
        self.speed = speed
        self.start_time: Optional[float] = None
        self.start_pos: Optional[Tuple[float, float]] = None

    def initialise(self) -> None:
        self.start_time = getattr(self.robot_manager.robot, "getTime", lambda: 0.0)()
        self.start_pos = self.robot_manager._position()
        
        # Enable force feedback on gripper
        left_motor = getattr(self.robot_manager.robot, "getDevice", lambda name: None)('gripper_left_finger_joint')
        right_motor = getattr(self.robot_manager.robot, "getDevice", lambda name: None)('gripper_right_finger_joint')
        if left_motor:
            try:
                left_motor.enableForceFeedback(self.robot_manager.timestep)
            except Exception:
                pass
        if right_motor:
            try:
                right_motor.enableForceFeedback(self.robot_manager.timestep)
            except Exception:
                pass

    def update(self) -> py_trees.common.Status:
        if self.start_time is None or self.start_pos is None:
            return py_trees.common.Status.FAILURE
        
        current_time = getattr(self.robot_manager.robot, "getTime", lambda: 0.0)()
        elapsed = current_time - self.start_time
        
        # Check if gripper fingers are touching the jar
        left_motor = getattr(self.robot_manager.robot, "getDevice", lambda name: None)('gripper_left_finger_joint')
        right_motor = getattr(self.robot_manager.robot, "getDevice", lambda name: None)('gripper_right_finger_joint')
        
        try:
            left_force = left_motor.getForceFeedback() if left_motor else 0.0
            right_force = right_motor.getForceFeedback() if right_motor else 0.0
        
            # Higher threshold = need more force = drive farther before stopping
            if left_force < -2.5 or right_force < -2.5:
                self.robot_manager._stop()
                print(f"[GRIPPER CONTACT] Detected! L:{left_force:.2f}N R:{right_force:.2f}N after {elapsed:.2f}s")
                return py_trees.common.Status.SUCCESS
        except Exception as e:
            print(f"[GRIPPER CONTACT] Error reading force: {e}")
            pass
        
        # Safety checks
        current_pos = self.robot_manager._position()
        distance = compute_distance(self.start_pos, current_pos)
        
        if distance > 0.7:  # Driven too far 
            self.robot_manager._stop()
            print(f"[GRIPPER CONTACT] Distance limit reached: {distance:.2f}m - NO CONTACT DETECTED")
            return py_trees.common.Status.FAILURE
        
        if elapsed > self.max_duration:  # Taken too long
            self.robot_manager._stop()
            print(f"[GRIPPER CONTACT] Time limit reached: {elapsed:.2f}s - NO CONTACT DETECTED")
            return py_trees.common.Status.FAILURE
        
        # Keep driving forward slowly
        self.robot_manager._set_wheel_speeds(self.speed, self.speed)
        return py_trees.common.Status.RUNNING
