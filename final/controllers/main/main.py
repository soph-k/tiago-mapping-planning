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
