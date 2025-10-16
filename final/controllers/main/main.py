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
        (1.98, 0.49, 0.8894)        # Jar 3 - moved back 0.026m for easier access
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
                'gripper_left_finger_joint': 0.045,  # Max allowed is 0.045
                'gripper_right_finger_joint': 0.045,  # Max allowed is 0.045
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
                'gripper_left_finger_joint': 0.0,   # was 0.01 - close fully
                'gripper_right_finger_joint': 0.0   # was 0.01 - close fully
            },
            'place': {
                'torso_lift_joint': 0.09,   # was 0.03
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