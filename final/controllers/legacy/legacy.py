import copy
import time
from collections import deque
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from controller import Supervisor
import py_trees

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
        (1.9763, 0.4943, 0.8894)      # Jar 3
    ]
   
    DROPOFF_POINTS: List[Tuple[float, float, float]] = [ # Coordinates on the table where jars should be placed after pickup. 
        (-0.08, -0.3, 0.8),
        (-0.07, -0.15, 0.8),
        (-0.03, -0.3, 0.8)
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
    def __init__(self, robot: Supervisor, memory: MemoryBoard) -> None: # Initialize the robot device manager.
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


class BTAction(py_trees.behaviour.Behaviour):   # Wrap a function into a py_trees behaviour.
    def __init__(self, func: Callable[[], str]) -> None: 
        super().__init__(name=getattr(func, "__name__", "BTAction")) # Initialize the behaviour.
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
    def __init__(self, robot: Supervisor, memory: MemoryBoard, pose_name: str = 'safe') -> None: 
        super().__init__(robot, memory)         # Initialize the robot device manager.  
        self.pose_name = pose_name              # The name of the pose to move to.
        self.threshold = 0.05                   # Allow a small error tolerance when checking if joints have reached their targets.  
        self.threshold_force = -5.0             
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
                'gripper_left_finger_joint': 0.08,
                'gripper_right_finger_joint': 0.08,
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
                'gripper_left_finger_joint': 0.08,
                'gripper_right_finger_joint': -0.08
            },
            'place': {
                'torso_lift_joint': 0.03,
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
    def __init__(                                            # Initialize the move to pose behaviour.
        self, 
        robot_manager: RobotDeviceManager,
        pose_name: str,
        name: Optional[str] = None,
    ) -> None: 
        super().__init__(name or f"Move to {pose_name}")    # Initialize the behaviour.
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
        if self.pose_controller is None:                    # If the pose controller hasn't been created, fail this behaviour.
            return py_trees.common.Status.FAILURE
        result = self.pose_controller.update()              # Update the pose controller.
        if result == "SUCCESS":                             # If the pose controller has reached the target.
            print(f"Completed pose: {self.pose_name}")      # Print a message to the console.
            return py_trees.common.Status.SUCCESS           # Return success.
        else:                                               # If the pose controller has not reached the target.
            return py_trees.common.Status.RUNNING
            

class RotateToTarget(py_trees.behaviour.Behaviour):         # Rotate the robot to face a target position.
    def __init__(                                           # Initialize the rotate to target behaviour.
        self, 
        robot_manager: RobotDeviceManager,
        target_pos: Tuple[float, float],
        name: str = "Rotate to Target",                     # The name of the behaviour.
    ) -> None:
        super().__init__(name)                              # Initialize the behaviour.
        self.robot_manager = robot_manager                  # Robot manager.
        self.target_pos = target_pos                        # The target position.

    def update(self) -> py_trees.common.Status:             # Update the rotate to target behaviour.
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
    def __init__(                                   # Initialize the move to target behaviour.
        self,
        robot_manager: RobotDeviceManager,          # The robot manager.
        target_pos: Tuple[float, float],            # The target position.
        max_speed: float = 1.5,                     # The maximum speed.
        name: str = "Move to Target",               # The name of the behaviour.
    ) -> None:
        super().__init__(name)                      # Initialize the behaviour.
        self.robot_manager = robot_manager          # The robot manager.
        self.target_pos = target_pos                # The target position.
        self.max_speed = max_speed                  # The maximum speed.

    def update(self) -> py_trees.common.Status:     # Update the move to target behaviour.
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
                left_motor.setVelocity(RobotConfig.GRIPPER_SPEED * 2) # Set the velocity of the left motor.
                right_motor.setVelocity(RobotConfig.GRIPPER_SPEED * 2) # Set the velocity of the right motor.
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


class CheckJarDetection(py_trees.behaviour.Behaviour): # Check if a jar is detected in front of the robot.
    def __init__(
        self,
        robot_manager: RobotDeviceManager,
        name: str = "Check Jar Detection",
    ) -> None:
        super().__init__(name)                        # Initialize the behaviour.
        self.robot_manager = robot_manager            # The robot manager.

    def update(self) -> py_trees.common.Status:       # Update the check jar detection behaviour.
        if isinstance(self.robot_manager, PickPlaceController): # If the robot manager is a PickPlaceController.
            detected = self.robot_manager._jar_in_front() # Ask the robot manager whether a jar is directly ahead.
        else:
            detected = False
        return (
            py_trees.common.Status.SUCCESS
            if detected
            else py_trees.common.Status.FAILURE
        )


class PickPlaceController(RobotDeviceManager):          # Controller to coordinate picking up jars and placing them on the table.
    def __init__(self, robot: Supervisor, memory: MemoryBoard) -> None: # Initialize the pick place controller.
        super().__init__(robot, memory)                 # Initialize the robot device manager.
        self.tree: Optional[py_trees.trees.BehaviourTree] = None # The behaviour tree for handling all jars.
        self.setup_behavior_tree()                      # Setup the behaviour tree.

    def move_towards( 
        self, 
        target: Tuple[float, float], # The target position.
        max_speed: float = 1.5, # The maximum speed.
        turn_gain: float = 2.0, # The gain for the turn.
    ) -> bool:
        cur = self._position() # Get the current position.
        heading = self._orientation() # Get the current heading.
        dx = target[0] - cur[0] # Compute the difference in x.
        dy = target[1] - cur[1] # Compute the difference in y.
        dist = np.hypot(dx, dy) # Compute the distance to the target.
        if dist < 0.6: # If the distance to the target is less than 0.6.
            self._stop() # Stop the robot.
            return True # Return True.
        desired = np.arctan2(dy, dx) # Compute the desired heading.
        angle_err = wrap_angle(desired - heading) # Compute the angle error.
        speed_limit = max_speed # Set the speed limit.
        if dist < 0.8: # If the distance to the target is less than 0.8.
            speed_limit = min(speed_limit, 0.4) # Set the speed limit to 0.4.
        base_speed = speed_limit * min(dist / 1.5, 1.0) # Compute the base speed.
        left = base_speed - turn_gain * angle_err # Compute the left wheel speed.
        right = base_speed + turn_gain * angle_err # Compute the right wheel speed.
        left = float(np.clip(left, -1.5, 1.5)) # Clamp the left wheel speed.
        right = float(np.clip(right, -1.5, 1.5)) # Clamp the right wheel speed.
        self._set_wheel_speeds(left, right) # Set the wheel speeds.
        return False # Return False.

    def _jar_in_front(self) -> bool: # Return True if a jar is detected directly ahead of the robot.
        lidar = self.memory.get("lidar") # Get the lidar device from memory.
        if lidar: # If the lidar device is found.
            try: # Try to get the range image from the lidar.
                ranges = lidar.getRangeImage() # Get the range image from the lidar.
                if ranges: # If the range image is found.
                    start = int(len(ranges) * 0.4) # Compute the start index.
                    end = int(len(ranges) * 0.6) # Compute the end index.
                    front = ranges[start:end] # Get the front part of the range image.
                    if front and min(front) < 0.45: # If the front part of the range image is found and the minimum value is less than 0.45.
                        return True # Return True.
            except Exception: # If an error occurs.
                pass
        camera = self.memory.get("camera")          # Get the camera device from memory.
        if camera: # If the camera device is found.
            try: # Try to get the recognition objects from the camera.
                objs = camera.getRecognitionObjects() # Get the recognition objects from the camera.
                for obj in objs: # For each object.
                    pos = obj.getPosition() # Get the position of the object.
                    if pos and np.linalg.norm(pos) < 0.3: 
                        return True # Return True.
            except Exception: # If an error occurs.
                pass
        return False # Return False.

    def create_jar_sequence(self, jar_index: int) -> py_trees.composites.Sequence: # Create a behaviour sequence for processing a single jar.
        jar_pos = RobotConfig.JAR_POSITIONS[jar_index] # Get the position of the jar.
        drop_pos = RobotConfig.DROPOFF_POINTS[min(jar_index, len(RobotConfig.DROPOFF_POINTS) - 1)] # Get the drop off point.
        sequence = py_trees.composites.Sequence(f"Process Jar {jar_index}", memory=True) # Create a sequence for processing a single jar.
        sequence.add_child(MoveToPose(self, 'safe', f"Safe Start Jar {jar_index}")) # Move the arm to the safe starting position.
        sequence.add_child(
            RotateToTarget(self, jar_pos[:2], f"Rotate to Jar {jar_index}")
        )
        if jar_index == 2: # If the jar index is 2.
            reach_pose = 'reach_open' # Use a wider grasp for the third jar.
        else:
            reach_pose = 'reach' # Use a standard grasp.
        sequence.add_child(MoveToPose(self, reach_pose, f"Extend to Jar {jar_index}"))
        move_or_detect = py_trees.composites.Selector(f"Move or Detect Jar {jar_index}", memory=False) # Create a selector for moving or detecting the jar.
        move_or_detect.add_child(CheckJarDetection(self, f"Check Jar {jar_index} Detection")) # Check if a jar is detected in front of the robot.
        move_or_detect.add_child(MoveToTarget(self, jar_pos[:2], max_speed=0.3, name=f"Move to Jar {jar_index}")) # Move the arm towards the jar.
        sequence.add_child(move_or_detect)
        if jar_index == 2: # If the jar index is 2.
            sequence.add_child(WaitForContact(self, max_wait=9.0, threshold=-0.5, name=f"Wait for Contact Jar {jar_index}")) # Wait for contact with the jar.
        sequence.add_child(MoveToPose(self, 'grab', f"Grab Jar {jar_index}")) # Close the gripper to grab the jar.
        sequence.add_child(RetreatFromPosition(self, name=f"Retreat from Jar {jar_index}")) # Back away from the jar.
        sequence.add_child(MoveToPose(self, 'safe', f"Safe Carry Jar {jar_index}")) # Return the arm to safe carry position.
        sequence.add_child(
            RotateToTarget(self, drop_pos[:2], f"Rotate to Table {jar_index}")
        )
        sequence.add_child(MoveToPose(self, 'place', f"Extend to Table {jar_index}")) # Extend arm to place the jar on the table.
        sequence.add_child(
            MoveToTarget(self, drop_pos[:2], max_speed=1.0, name=f"Move to Table {jar_index}")
        )
        if jar_index == 2: # If the jar index is 2.
            sequence.add_child(WaitSeconds(self, 1.5, f"Stabilise at Table {jar_index}")) # For the third jar we add an extra pause to ensure the arm stabilises above the placement spot.  Without this delay the jar might be released prematurely or not quite above the table.
        sequence.add_child(
            WaitAndOpenGripper(self, wait_time=0.5, name=f"Place Jar {jar_index}")
        )
        return sequence

    def setup_behavior_tree(self) -> None: # Build the behaviour tree for processing all jars in sequence.
        root = py_trees.composites.Sequence("Pick Place All Jars", memory=True) # Use a memoryful sequence here so that the controller remembers which jar it is processing across ticks.  Without memory the root would restart from the first jar every time run() is called.
        for i in range(len(RobotConfig.JAR_POSITIONS)):
            jar_sequence = self.create_jar_sequence(i)
            root.add_child(jar_sequence)
        root.add_child(MoveToPose(self, 'safe', "Final Safe Position")) # After all jars are processed, move the arm back to safe position.
        self.tree = py_trees.trees.BehaviourTree(root)
        try: # Try to setup the behaviour tree.
            self.tree.setup(timeout=15) # Setup the behaviour tree.
        except Exception: # If an error occurs.
            pass

    def run(self) -> str: # Run the behaviour tree and return its status.
        if self.tree is None: # If the behaviour tree is not found.
            return "FAILURE"
        self.tree.tick()
        status = self.tree.root.status # Get the status of the behaviour tree.
        if status == py_trees.common.Status.SUCCESS: # If the behaviour tree is successful.
            print("All pick and place operations completed!") # Print a message to the console.
            return "SUCCESS" # Return success.
        elif status == py_trees.common.Status.FAILURE: # If the behaviour tree is failed.
            print("Pick and place operations failed!") # Print a message to the console.
            return "FAILURE" # Return failure.
        else: # If the behaviour tree is running.
            return "RUNNING" # Return running.


class NavigationController(RobotDeviceManager):
    def __init__(self, robot: Supervisor, memory: MemoryBoard) -> None:
        super().__init__(robot, memory)
        self.prob_map: np.ndarray = np.zeros((RobotConfig.MAP_WIDTH, RobotConfig.MAP_SIZE), dtype=np.float32) 
        self.trajectory_points: deque[Tuple[float, float]] = deque(maxlen=3000) # A deque to store the robot's recent trajectory for display. 
        self.LIDAR_OFFSET_X = 0.202 
        self.LIDAR_START_IDX = 80
        self.LIDAR_END_IDX = -80
        self.waypoints = RobotConfig.MAPPING_WAYPOINTS # Waypoints for exploration.
        self.current_wp_idx = 0
        self.start_time: Optional[float] = None
        self.max_mapping_time = 90.0 # Maximum time allocated for mapping
        self.max_mapping_time = 90.0
        self.sensor_array: Dict[str, object] = {}   # Sensor array for LIDAR, GPS and compass. 
        self.arm_motors: Dict[str, object] = {} # Placeholders for arm motors 
        self.sensor_array_positioned = False # Flags to ensure the sensor array has been moved out of the way.

    def initialize_sensor_array(self) -> bool: 
        lidar = self.memory.get("lidar") # Get the lidar device from memory.
        if lidar: # If the lidar device is found.
            self.sensor_array['lidar'] = lidar # Bind the lidar device to the sensor array.
        joint_names = [
            'torso_lift_joint', 'arm_1_joint', 'arm_2_joint', 'arm_3_joint',
            'arm_4_joint', 'arm_5_joint', 'arm_6_joint', 'arm_7_joint',
            'gripper_left_finger_joint', 'gripper_right_finger_joint',
            'head_1_joint', 'head_2_joint'
        ]
        for joint_name in joint_names: # For each joint name.
            motor = getattr(self.robot, "getDevice", lambda name: None)(joint_name) # Get the motor device for the joint.
            if motor: # If the motor device is found.
                self.arm_motors[joint_name] = motor # Bind the motor device to the arm motors.
        gps = self.memory.get("gps") # Get the gps device from memory.
        compass = self.memory.get("compass") # Get the compass device from memory.
        if gps and compass: # If the gps and compass devices are found.
            self.sensor_array['gps'] = gps # Bind the gps device to the sensor array.
            self.sensor_array['compass'] = compass # Bind the compass device to the sensor array.
        return self._set_sensor_array_position() # Set the sensor array position.

    def _set_sensor_array_position(self) -> bool: # Set the sensor array position.
        if self.sensor_array_positioned: # If the sensor array has been moved out of the way.
            return True # Return True.
        if self.memory.get("jar_picking_started", False): # If the jar picking has started.
            return False # Return False.
        system_instance = self.memory.get("system_instance") # Get the system instance from memory.
        if system_instance and getattr(system_instance, 'state', None) == "MANIPULATION": # If the system instance is found and the state is MANIPULATION.
            return False # Return False.
        sensor_array_config = {
            'torso_lift_joint': 0.25,
            'arm_1_joint': 1.5708,
            'arm_2_joint': 1.0472,
            'arm_3_joint': 1.5708,
            'arm_4_joint': 0.0,
            'arm_5_joint': 0.1745,
            'arm_6_joint': -1.5708,
            'arm_7_joint': -0.0175,
            'gripper_left_finger_joint': 0.045,
            'gripper_right_finger_joint': 0.045,
            'head_1_joint': 0.0,
            'head_2_joint': 0.0
        }
        success = 0 # Initialize the success counter.
        for joint, position in sensor_array_config.items(): # For each joint and position.
            motor = self.arm_motors.get(joint) # Get the motor device for the joint.
            if motor: # If the motor device is found.
                motor.setPosition(position) # Set the position of the motor.
                motor.setVelocity(0.5) # Set the velocity of the motor.
                success += 1 # Increment the success counter.
        self.sensor_array_positioned = success > 0 # Set the sensor array positioned.
        self.memory.set("sensor_array_positioned", True) 
        self.memory.set("navigation_arm_ready", True) 
        return self.sensor_array_positioned # Return the sensor array positioned.

    def execute_mapping(self) -> str: 
        if not self.sensor_array_positioned: # If the sensor array is not positioned.
            print("Positioning sensor array for navigation") # Print a message to the console.
            self.initialize_sensor_array() # Initialize the sensor array.
        if self.start_time is None: # If the start time is not found.
            self.start_time = getattr(self.robot, "getTime", lambda: 0.0)() # Record the start time.
            print("Beginning navigation and mapping phase") # Print a message to the console.
        elapsed = getattr(self.robot, "getTime", lambda: 0.0)() - self.start_time # Get the elapsed time.
        # Stop mapping after the maximum time has elapsed.
        if elapsed > self.max_mapping_time: # If the elapsed time is greater than the maximum mapping time.
            return self._finish_mapping() # Finish the mapping.
        # Stop mapping after all waypoints have been visited.
        if self.current_wp_idx >= len(self.waypoints): # If the current waypoint index is greater than or equal to the number of waypoints.
            print("All navigation waypoints visited, finishing mapping.") # Print a message to the console.
            return self._finish_mapping() # Finish the mapping.
        current_pos = self._position() # Get the current position.
        current_angle = self._orientation() # Get the current angle.
        if len(current_pos) < 2: # If the current position is less than 2.
            return "FAILURE" # Return failure.
        xw, yw = current_pos # Get the current position.
        self.trajectory_points.append((xw, yw)) # Record the robot's trajectory for display.
        self._update_map(xw, yw, current_angle) # Update the probability map with LIDAR data.
        goal_x, goal_y = self.waypoints[self.current_wp_idx] # Get the current waypoint.
        dx, dy = goal_x - xw, goal_y - yw # Get the distance to the current waypoint.
        rho = np.sqrt(dx * dx + dy * dy) # Get the distance to the current waypoint.
        if rho < RobotConfig.DIST_TOL: # If the distance to the current waypoint is less than the distance tolerance.
            self.current_wp_idx += 1 # Advance to the next waypoint.
            print(f"Reached waypoint {self.current_wp_idx-1}, moving to next waypoint") # Print a message to the console.
            return "RUNNING" # Return running.
        goal_theta = np.arctan2(dy, dx) # Get the goal angle.
        alpha = wrap_angle(goal_theta - current_angle) # Get the angle to the current waypoint.
        left_cmd, right_cmd = compute_motor_commands(alpha, rho) # Compute the motor commands.
        # Apply simple obstacle avoidance based on sensor readings.
        if self.sensor_array_positioned: # If the sensor array is positioned.
            left_cmd, right_cmd = self._apply_sensor_array_avoidance(left_cmd, right_cmd, current_angle) # Apply the sensor array avoidance.
        self._set_wheel_speeds(left_cmd, right_cmd) # Set the wheel speeds.
        self.memory.set("prob_map", self.prob_map) # Store the probability map in memory.
        self.memory.set("trajectory_points", list(self.trajectory_points)) # Store the trajectory in memory.
        return "RUNNING"

    def _apply_sensor_array_avoidance(
        self,
        left_cmd: float,
        right_cmd: float,
        robot_heading: float,
    ) -> Tuple[float, float]:
        try: # Try to apply the sensor array avoidance.
            obstacles = self.check_obstacles()
            # If something is directly in front, turn in place away from it.
            if obstacles['front_center'] or (
                obstacles['front_left'] and obstacles['front_right'] 
            ):
                if obstacles['front_left'] and not obstacles['front_right']:
                    return 2.0, -2.0
                elif obstacles['front_right'] and not obstacles['front_left']:
                    return -2.0, 2.0
                else:
                    return 2.0, -2.0
            if obstacles['left']:
                right_cmd *= 1.2
            if obstacles['right']:
                left_cmd *= 1.2
            return left_cmd, right_cmd
        except Exception:
            return left_cmd, right_cmd

    def check_obstacles(self) -> Dict[str, bool]: # Check for obstacles.
        readings = self.get_sensor_readings() # Get the sensor readings.
        return {
            'front_center': readings.get('front_center', float('inf')) < 0.6, # Front center obstacle.
            'front_left': readings.get('front_left', float('inf')) < 0.6, # Front left obstacle.
            'front_right': readings.get('front_right', float('inf')) < 0.6, # Front right obstacle.
            'left': readings.get('left', float('inf')) < 0.6, # Left obstacle.
            'right': readings.get('right', float('inf')) < 0.6
        }

    def get_sensor_readings(self) -> Dict[str, float]: # Get the sensor readings.
        readings: Dict[str, float] = {} # Initialize the readings dictionary.
        lidar = self.sensor_array.get('lidar') # Get the lidar device from the sensor array.
        if lidar: # If the lidar device is found.
            try:
                lidar_ranges = np.array(lidar.getRangeImage()) # Get the lidar ranges.
                readings['lidar'] = lidar_ranges # Store the lidar ranges in the readings dictionary.
                readings['front_center'] = self._get_lidar_sector(lidar_ranges, -15, 15) # Front center obstacle.
                readings['front_left'] = self._get_lidar_sector(lidar_ranges, -45, -15) # Front left obstacle.
                readings['front_right'] = self._get_lidar_sector(lidar_ranges, 15, 45) # Front right obstacle.
                readings['left'] = self._get_lidar_sector(lidar_ranges, -90, -45) # Left obstacle.
                readings['right'] = self._get_lidar_sector(lidar_ranges, 45, 90) # Right obstacle.
            except Exception:
                pass
        gps = self.sensor_array.get('gps') # Get the gps device from the sensor array.
        compass = self.sensor_array.get('compass') # Get the compass device from the sensor array.
        if gps and compass: # If the gps and compass devices are found.
            readings['position'] = gps.getValues() # Store the gps values in the readings dictionary.
            readings['orientation'] = np.arctan2(compass.getValues()[0], compass.getValues()[1])
        return readings

    def _get_lidar_sector(
        self, ranges: np.ndarray, start_angle: float, end_angle: float
    ) -> float: # Return the minimum range in a sector of the LIDAR scan.
        if len(ranges) == 0: # If the ranges are empty.
            return float('inf') # Return infinity.
        total_angle = 240  
        center_index = len(ranges) // 2
        start_idx = center_index + int(start_angle * len(ranges) / total_angle) # Calculate the start index.
        end_idx = center_index + int(end_angle * len(ranges) / total_angle) # Calculate the end index.
        start_idx = max(0, min(start_idx, len(ranges) - 1)) # Calculate the start index.
        end_idx = max(0, min(end_idx, len(ranges) - 1)) # Calculate the end index.
        if start_idx > end_idx: # If the start index is greater than the end index.
            start_idx, end_idx = end_idx, start_idx # Swap the start and end indices.
        sector_ranges = ranges[start_idx:end_idx + 1] # Get the sector ranges.
        valid_ranges = [r for r in sector_ranges if np.isfinite(r) and r > 0.1] # Get the valid ranges.
        return min(valid_ranges) if valid_ranges else float('inf')

    def _update_map(self, robot_x: float, robot_y: float, robot_theta: float) -> None: # Update the map.
        lidar = self.memory.get("lidar") # Get the lidar device from memory.
        if not lidar: # If the lidar device is not found.
            return 
        try:
            lidar_ranges = np.array(lidar.getRangeImage()) # Get the lidar ranges.
            if len(lidar_ranges) == 0: # If the lidar ranges are empty.
                return 
            angles = np.linspace(2 * np.pi / 3, -2 * np.pi / 3, len(lidar_ranges)) # Get the angles.
            valid_start = max(0, self.LIDAR_START_IDX) # Get the valid start index.
            valid_end = min(len(lidar_ranges), len(lidar_ranges) + self.LIDAR_END_IDX) # Get the valid end index.
            valid_ranges: List[float] = [] # Initialize the valid ranges list.
            valid_angles: List[float] = [] # Initialize the valid angles list.
            for i in range(valid_start, valid_end): # For each valid index.
                range_val = lidar_ranges[i] # Get the range value.
                if not np.isfinite(range_val): # If the range value is not finite.
                    range_val = 100.0 # Set the range value to 100.0.
                if 0.12 < range_val < 8.0: 
                    valid_ranges.append(range_val) # Add the range value to the valid ranges list.
                    valid_angles.append(angles[i]) # Add the angle to the valid angles list.
            if not valid_ranges: # If the valid ranges list is empty.
                return 
            w_T_r = np.array([
                [np.cos(robot_theta), -np.sin(robot_theta), robot_x], # Rotation matrix.
                [np.sin(robot_theta), np.cos(robot_theta), robot_y], # Rotation matrix.
                [0, 0, 1] # Rotation matrix.
            ])
            X_r = np.array([
                np.array(valid_ranges) * np.cos(valid_angles) + self.LIDAR_OFFSET_X, 
                np.array(valid_ranges) * np.sin(valid_angles), 
                np.ones(len(valid_ranges)) 
            ])
            world_points = w_T_r @ X_r
            for i, (x_world, y_world) in enumerate(zip(world_points[0], world_points[1])): # For each world point.
                px, py = world_to_pixel(x_world, y_world) # Convert the world point to a pixel.
                if 0 <= px < RobotConfig.MAP_WIDTH and 0 <= py < RobotConfig.MAP_SIZE: # If the pixel is within the map.
                    distance = valid_ranges[i] # Get the distance.
                    weight = 0.008 if distance < 1.0 else 0.006 if distance < 3.0 else 0.004 # Get the weight.
                    self.prob_map[px, py] += weight # Add the weight to the probability map.
                    self.prob_map[px, py] = min(self.prob_map[px, py], 1.0) # Add the probability map.
        except Exception:
            pass

    def _finish_mapping(self) -> str: 
        print("Finishing mapping phase, generating configuration space") # Print a message to the console.
        self._set_wheel_speeds(0.0, 0.0) # Set the wheel speeds to 0.0.
        self.memory.set("prob_map", self.prob_map) # Store the probability map in memory.
        self.memory.set("trajectory_points", list(self.trajectory_points)) # Store the trajectory in memory.
        try:
            cspace_map = self.generate_cspace() # Generate the configuration space.
            self.memory.set("cspace", cspace_map) # Store the configuration space in memory.
        except Exception:
            pass
        self.memory.set("mapping_complete", True)
        print("Configuration space generated, mapping phase complete.")
        return "SUCCESS"

    def generate_cspace(self) -> np.ndarray: # Generate the configuration space.
        print("Generating configuration space") # Print a message to the console.
        radius_pixels = 9 # Get the radius in pixels.
        down_factor = 4 # Get the down factor.
        prob_small = self.prob_map[::down_factor, ::down_factor] # Get the probability map.
        radius_small = max(1, radius_pixels // down_factor) # Get the radius in pixels.
        small_offsets: List[Tuple[int, int]] = [] # Initialize the small offsets list.
        for dx in range(-radius_small, radius_small + 1): # For each dx.
            for dy in range(-radius_small, radius_small + 1): # For each dy.
                if dx * dx + dy * dy <= radius_small * radius_small: # If the distance is less than the radius.
                    small_offsets.append((dx, dy)) # Add the offset to the small offsets list.
        cspace_small = np.ones_like(prob_small, dtype=np.float32) # Initialize the small configuration space.
        obstacles_small = np.where(prob_small > 0.05) # Find the obstacles.
        for idx in range(len(obstacles_small[0])): # For each obstacle.
            x, y = obstacles_small[0][idx], obstacles_small[1][idx] # Get the obstacle.
            for dx, dy in small_offsets: # For each small offset.
                nx, ny = x + dx, y + dy # Get the new position.
                if 0 <= nx < cspace_small.shape[0] and 0 <= ny < cspace_small.shape[1]: # If the new position is within the small configuration space.
                    cspace_small[nx, ny] = 0.0 # Set the small configuration space to 0.0.
        cspace_map = np.repeat(np.repeat(cspace_small, down_factor, axis=0), down_factor, axis=1) # Repeat the small configuration space.
        cspace_map = cspace_map[: self.prob_map.shape[0], : self.prob_map.shape[1]] # Repeat the small configuration space.
        self.memory.set("cspace_complete", True) # Set the cspace complete flag in memory.
        print("Configuration space generation complete") # Print a message to the console.
        return cspace_map # Return the configuration space.


class PerceptionController(RobotDeviceManager): # Perception controller.
    def __init__(self, robot: Supervisor, memory: MemoryBoard) -> None: # Initialize the perception controller.
        super().__init__(robot, memory) # Initialize the perception controller.
        gps = getattr(robot, "getDevice", lambda name: None)('gps') # Get the gps device.
        if gps: # If the gps device is found.
            try: # Try to enable the gps device.
                gps.enable(self.timestep) # Enable the gps device.
            except Exception: # If the gps device is not found.
                pass
            self.memory.set("gps", gps) # Store the gps device in memory.
        compass = getattr(robot, "getDevice", lambda name: None)('compass') # Get the compass device.
        if compass: # If the compass device is found.
            try: # Try to enable the compass device.
                compass.enable(self.timestep) # Enable the compass device.
            except Exception: # If the compass device is not found.
                pass
            self.memory.set("compass", compass) # Store the compass device in memory.
        # Enable the camera and its recognition feature.
        self.camera = getattr(robot, "getDevice", lambda name: None)("camera")
        if self.camera:
            try:
                self.camera.enable(self.timestep) # Enable the camera device. 
                # Enable object recognition if supported.
                try:
                    self.camera.recognitionEnable(self.timestep)  
                except Exception:
                    pass
            except Exception:
                self.camera = None
        self.recognized_objects: List[Dict[str, object]] = [] # Initialize the recognized objects list.
        self.distance_threshold = 0.1 # Initialize the distance threshold.

    def update(self) -> str: # Update the perception controller.
        if not self.camera: # If the camera device is not found.
            return "FAILURE" # Return failure.
        try: # Try to update the perception controller.
            T_world_camera = self._get_camera_transform() # Get the camera transform.
            objects = self.camera.getRecognitionObjects() # Get the recognition objects.
            added = 0 # Initialize the added variable.
            for obj in objects: # For each object.
                pos_cam = np.array(list(obj.getPosition()) + [1]) # Get the position of the object.
                pos_world = (T_world_camera @ pos_cam)[:3] # Get the position of the object in the world.
                is_duplicate = any( # Check if the object is a duplicate.
                    compute_distance(item["position"], pos_world) < self.distance_threshold # Check if the object is too close to the already recorded objects.
                    for item in self.recognized_objects
                )
                if not is_duplicate: # If the object is not a duplicate.
                    self.recognized_objects.append({
                        "position": pos_world,
                        "name": getattr(obj, 'model', 'Unknown'),
                        "id": getattr(obj, 'id', -1)
                    })
                    added += 1
            self.memory.set("recognized_objects", self.recognized_objects) # Store the recognized objects in memory.
            return "SUCCESS" if added > 0 else "RUNNING" # Return success if the object is added, otherwise return running.
        except Exception: # If the object is not found.
            return "FAILURE" # Return failure.

    def _get_camera_transform(self) -> np.ndarray: # Get the camera transform.
        gps = self.memory.get("gps") # Get the gps device from memory.
        compass = self.memory.get("compass") # Get the compass device from memory.
        if not gps or not compass:
            # Return identity if sensors aren't available.
            return np.eye(4)
        gps_values = gps.getValues() # Get the gps values.
        compass_values = compass.getValues()
        robot_x, robot_y, robot_z = gps_values
        robot_theta = np.arctan2(compass_values[0], compass_values[1]) 
        # Offset of the camera relative to the robot's base.
        camera_offset_x = 0.1
        camera_offset_z = 1.2
        return np.array([
            [np.cos(robot_theta), 0, np.sin(robot_theta), robot_x + camera_offset_x * np.cos(robot_theta)],
            [0, 1, 0, robot_y],
            [-np.sin(robot_theta), 0, np.cos(robot_theta), robot_z + camera_offset_z],
            [0, 0, 0, 1]
        ])


class MapDisplay:
    def __init__(self, memory: MemoryBoard) -> None:
        self.memory = memory
        self.display = None
        self.width = 0
        self.height = 0
        self.COLOR_BLACK = 0x000000 # Black color.
        self.COLOR_WHITE = 0xFFFFFF # White color.
        self.COLOR_RED = 0xFF0000 # Red color.
        self.COLOR_GREEN = 0x00FF00 # Green color.
        self.COLOR_BLUE = 0x0000FF

    def update(self) -> None: # Update the display.
        self.display = self.memory.get("display") # Get the display device from memory.
        if not self.display: # If the display device is not found.
            return
        self.width = self.display.getWidth() # Get the width of the display.
        self.height = self.display.getHeight() # Get the height of the display.
        self.display.setColor(self.COLOR_BLACK)
        self.display.fillRectangle(0, 0, self.width, self.height) # Clear the display.
        cspace = self.memory.get("cspace") # Get the configuration space from memory.
        if cspace is not None: # If the configuration space is found.
            self._draw_cspace(cspace) # Draw the configuration space.
        else:
            prob_map = self.memory.get("prob_map") # Get the probability map from memory.
            if prob_map is not None: # If the probability map is found.
                self._draw_probability_map(prob_map) # Draw the probability map.
        trajectory = self.memory.get("trajectory_points") # Get the trajectory from memory.
        if trajectory: # If the trajectory is found.
            self._draw_trajectory(trajectory) # Draw the trajectory.
        self._draw_robot() # Draw the robot.

    def _draw_probability_map(self, prob_map: np.ndarray) -> None: # Draw the probability map.
        for x in range(0, prob_map.shape[0], 2): # For each x.
            for y in range(0, prob_map.shape[1], 2): # For each y.
                if prob_map[x, y] > 0.001: # If the probability is greater than 0.001.
                    intensity = int(255 * min(prob_map[x, y] * 2.5, 0.8)) # Get the intensity.
                    color = (intensity << 16) | (intensity << 8) | intensity # Get the color.
                    self.display.setColor(color) # Set the color.
                    dx, dy = self._map_to_display(x, y, prob_map.shape) # Get the display coordinates.
                    if 0 <= dx < self.width and 0 <= dy < self.height: # If the display coordinates are within the display.
                        self.display.fillRectangle(dx, dy, 2, 2) # Draw the probability map.

    def _draw_cspace(self, cspace: np.ndarray) -> None: # Draw the configuration space.
        step = 2 # Get the step.
        for x in range(0, cspace.shape[0], step): # For each x.
            for y in range(0, cspace.shape[1], step): # For each y.
                if cspace[x, y] < 0.5: # If the configuration space is less than 0.5.
                    self.display.setColor(self.COLOR_WHITE) # Set the color to white.
                    dx, dy = self._map_to_display(x, y, cspace.shape) # Get the display coordinates.
                    if 0 <= dx < self.width and 0 <= dy < self.height: # If the display coordinates are within the display.
                        self.display.fillRectangle(dx, dy, step, step) # Draw the configuration space.

    def _draw_trajectory(self, trajectory: List[Tuple[float, float]]) -> None: # Draw the trajectory.
        self.display.setColor(self.COLOR_RED) # Set the color to red.
        for i in range(len(trajectory) - 1): # For each trajectory.
            x1, y1 = trajectory[i] # Get the trajectory.
            x2, y2 = trajectory[i + 1]
            px1, py1 = world_to_pixel(x1, y1)
            px2, py2 = world_to_pixel(x2, y2)
            dx1, dy1 = self._map_to_display(px1, py1, (RobotConfig.MAP_WIDTH, RobotConfig.MAP_SIZE))
            dx2, dy2 = self._map_to_display(px2, py2, (RobotConfig.MAP_WIDTH, RobotConfig.MAP_SIZE))
            if (
                0 <= dx1 < self.width and 0 <= dy1 < self.height and
                0 <= dx2 < self.width and 0 <= dy2 < self.height
            ):
                self.display.drawLine(dx1, dy1, dx2, dy2)

    def _draw_robot(self) -> None:
        gps = self.memory.get("gps") # Get the gps device from memory.
        compass = self.memory.get("compass") # Get the compass device from memory.
        if not gps or not compass: # If the gps or compass device is not found.
            return
        pos = gps.getValues() # Get the gps values.
        px, py = world_to_pixel(pos[0], pos[1]) # Get the pixel coordinates.
        dx, dy = self._map_to_display(px, py, (RobotConfig.MAP_WIDTH, RobotConfig.MAP_SIZE)) # Get the display coordinates.
        if 0 <= dx < self.width and 0 <= dy < self.height: # If the display coordinates are within the display.
            self.display.setColor(self.COLOR_GREEN) # Set the color to green.
            self.display.fillRectangle(dx - 2, dy - 2, 5, 5) # Draw the robot.
            angle = np.arctan2(compass.getValues()[0], compass.getValues()[1]) # Get the angle.
            end_x = dx + int(10 * np.cos(angle)) # Get the end x.
            end_y = dy + int(10 * np.sin(angle)) # Get the end y.
            self.display.setColor(self.COLOR_WHITE) # Set the color to white.
            self.display.drawLine(dx, dy, end_x, end_y) # Draw the robot.

    def _map_to_display(
        self, px: int, py: int, map_shape: Tuple[int, int]
    ) -> Tuple[int, int]: # Map the probability/c-space coordinates to display coordinates.
        dx = int(px * self.width / map_shape[0]) # Get the display x.
        dy = int(py * self.height / map_shape[1]) # Get the display y.
        return dx, dy


class RobotSupervisor:
    def __init__(self) -> None: # Initialize the robot supervisor.
        self.robot = Supervisor() # Create a supervisor instance from Webots.
        self.timestep = int(getattr(self.robot, "getBasicTimeStep", lambda: 32)()) # Get the timestep.
        self.memory = MemoryBoard() # Create a memory board.
        self.memory.set("robot", self.robot) # Set the robot in memory.
        self.memory.set("timestep", self.timestep) # Set the timestep in memory.
        self.navigation: Optional[NavigationController] = None # Initialize the navigation controller.
        self.pickplace: Optional[PickPlaceController] = None # Initialize the pick place controller.
        self.perception: Optional[PerceptionController] = None # Initialize the perception controller.
        self.display: Optional[MapDisplay] = None # Initialize the map display.
        self.state = "MAPPING"
        self.step_counter = 0
        self.display_static = False
        self.behavior_tree: Optional[py_trees.trees.BehaviourTree] = None

    def _initialize_devices(self) -> None: # Initialize the devices.
        gps = getattr(self.robot, "getDevice", lambda name: None)('gps') # Get the gps device.
        if gps: # If the gps device is found.
            try: # Try to enable the gps device.
                gps.enable(self.timestep) # Enable the gps device.
            except Exception: # If the gps device is not found.
                pass
            self.memory.set("gps", gps) # Set the gps device in memory.
        compass = getattr(self.robot, "getDevice", lambda name: None)('compass') # Get the compass device.
        if compass: # If the compass device is found.
            try: # Try to enable the compass device.
                compass.enable(self.timestep)
            except Exception:
                pass
            self.memory.set("compass", compass) # Set the compass device in memory.
        motorL = getattr(self.robot, "getDevice", lambda name: None)("wheel_left_joint") # Get the left motor.
        if motorL: # If the left motor is found.
            try: # Try to set the left motor to velocity control mode.
                motorL.setPosition(float('inf'))
            except Exception:
                pass
        motorR = getattr(self.robot, "getDevice", lambda name: None)("wheel_right_joint")
        if motorR:
            try:
                motorR.setPosition(float('inf'))
            except Exception:
                pass
        self.memory.set("motorL", motorL)
        self.memory.set("motorR", motorR)
        for name in ["Hokuyo URG-04LX-UG01", "lidar"]: # For each lidar device.
            try: # Try to enable the lidar device.
                lidar = getattr(self.robot, "getDevice", lambda name: None)(name) # Get the lidar device.
                if lidar: # If the lidar device is found.
                    try: # Try to enable the lidar device.
                        lidar.enable(self.timestep) # Enable the lidar device.
                    except Exception: # If the lidar device is not found.
                        pass
                    self.memory.set("lidar", lidar)
                    break
            except Exception:
                continue
        # Display for map rendering.
        display = getattr(self.robot, "getDevice", lambda name: None)("display") # Get the display device.
        if display: # If the display device is found.
            self.memory.set("display", display) # Set the display device in memory.
        # Camera device.
        camera = getattr(self.robot, "getDevice", lambda name: None)("camera") # Get the camera device.
        if camera: # If the camera device is found.
            try: # Try to enable the camera device.
                camera.enable(self.timestep) # Enable the camera device.
            except Exception:
                pass
            self.memory.set("camera", camera) # Set the camera device in memory.
        # Bind arm motors and enable position sensors.
        arm_motors: Dict[str, object] = {} # Initialize the arm motors.
        joint_names = [
            'torso_lift_joint', 'arm_1_joint', 'arm_2_joint', 'arm_3_joint',
            'arm_4_joint', 'arm_5_joint', 'arm_6_joint', 'arm_7_joint',
            'gripper_left_finger_joint', 'gripper_right_finger_joint',
            'head_1_joint', 'head_2_joint'
        ]
        for joint in joint_names:
            try:
                motor = getattr(self.robot, "getDevice", lambda name: None)(joint)
                if motor:
                    arm_motors[joint] = motor
                    try:
                        sensor = motor.getPositionSensor()
                        if sensor:
                            sensor.enable(self.timestep)
                    except Exception:
                        pass
            except Exception:
                pass
        # Enable dedicated sensors on the gripper fingers if available.
        for sensor_name in ['gripper_left_sensor_finger_joint', 'gripper_right_sensor_finger_joint']:
            try:
                sensor = getattr(self.robot, "getDevice", lambda name: None)(sensor_name)
                if sensor:
                    sensor.enable(self.timestep)
            except Exception:
                pass
        self.memory.set("arm_motors", arm_motors)

    def initialize(self) -> bool:
        """Initialise subsystems and behaviour tree."""
        try:
            self._initialize_devices()
            # Create subsystem controllers.
            self.navigation = NavigationController(self.robot, self.memory)
            self.pickplace = PickPlaceController(self.robot, self.memory)
            self.perception = PerceptionController(self.robot, self.memory)
            self.display = MapDisplay(self.memory)
            # Build a behaviour tree: first run mapping, then pick/place.
            root = py_trees.composites.Sequence(name="RootSequence", memory=True)
            mapping_action = BTAction(self._bt_mapping)
            pick_action = BTAction(self._bt_pickplace)
            root.add_children([mapping_action, pick_action])
            self.behavior_tree = py_trees.trees.BehaviourTree(root)
            try:
                self.behavior_tree.setup(timeout=5)
            except Exception as e:
                print(f"[Behaviour tree setup failed: {e}")
            return True
        except Exception as e:
            print(f"Initialization failed: {e}")
            return False

    def run(self) -> bool:
        """Main control loop.  Returns False if initialization fails."""
        if not self.initialize():
            return False
        print("Starting main control loop.")
        # Expose this instance on the memory board for other subsystems.
        self.memory.set("system_instance", self)
        try:
            while getattr(self.robot, "step", lambda x: -1)(self.timestep) != -1:
                self.step_counter += 1
                # Update the perception controller on every step.
                if self.perception:
                    self.perception.update()
                # Tick the behaviour tree.
                if self.behavior_tree:
                    self.behavior_tree.tick()
                    # When the tree completes, update the state.
                    if self.behavior_tree.root.status == py_trees.common.Status.SUCCESS:
                        self.state = "COMPLETE"
                # Periodically draw to the display.
                if (
                    self.display and not self.display_static and
                    self.step_counter % 15 == 0
                ):
                    self.display.update()
                # Exit after a large number of steps to avoid infinite loops.
                if self.step_counter > 20000:
                    print("Maximum step count reached, stopping execution.")
                    break
                # When complete, stop the robot.
                if self.state == "COMPLETE":
                    print("Task complete. Robot is now idle in safe position.")
                    break
        except KeyboardInterrupt:
            print("Interrupted by user.")
            pass
        finally:
            self._cleanup()
        return True

    def _bt_mapping(self) -> str:
        """Behaviour tree leaf: run mapping until complete."""
        if not self.navigation:
            return "FAILURE"
        result = self.navigation.execute_mapping()
        if result == "SUCCESS":
            self.state = "MANIPULATION"
            print("Navigation and mapping phase completed successfully.")
            # Take a snapshot of the map and stop updating it.
            if self.display and not self.display_static:
                self.display.update()
                self.display_static = True
            return "SUCCESS"
        elif result == "FAILURE":
            print("Navigation and mapping phase failed.")
            return "FAILURE"
        return "RUNNING"

    def _bt_pickplace(self) -> str:
        if not self.pickplace:
            return "FAILURE"
        result = self.pickplace.run()
        if result == "SUCCESS":
            print("Pick and place operations completed successfully.")
            return "SUCCESS"
        elif result == "FAILURE":
            print("Pick and place operations failed.")
            return "FAILURE"
        return "RUNNING"

    def _cleanup(self) -> None:
        print("Cleaning up and stopping all motors")
        motorL = self.memory.get("motorL")
        motorR = self.memory.get("motorR")
        if motorL and motorR:
            try:
                motorL.setVelocity(0.0)
                motorR.setVelocity(0.0)
                print("Wheel motors stopped successfully")
            except Exception:
                print("Warning - could not stop wheel motors")
                pass
        print("Cleanup complete")

    # Convenience method for testing poses manually.  Not used by the behaviour tree.
    def test_pose(self, pose_name: str) -> bool:
        if self.pickplace:
            # We can instantiate a temporary ArmPoseController to test the pose.
            controller = ArmPoseController(self.robot, self.memory, pose_name)
            controller.setup()
            controller.initialise()
            # Wait until the pose is reached.
            while getattr(self.robot, "step", lambda x: -1)(self.timestep) != -1:
                status = controller.update()
                if status == "SUCCESS":
                    break
            return True
        return False

def main() -> None: # Main function to run the robot.
    print("Starting autonomous navigation and manipulation system")
    supervisor = RobotSupervisor()
    supervisor.run()
    print("System shutdown complete.")

if __name__ == "__main__": # If the script is run directly, run the main function.
    main()