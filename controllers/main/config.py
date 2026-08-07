from typing import List, Tuple
import numpy as np

class RobotConfig:
    MAP_SIZE = 300                                                                                 # grid height 
    MAP_WIDTH = 200                                                                                # grid width 
    MAPPING_INTERVAL = 3                                                                           # map update throttle (ticks)
    TRAJECTORY_INTERVAL = 8                                                                        
    TURN_SPEED_MAX = 4.0                                                                           # max angular speed rad/s
    DRIVE_SPEED_MAX = 3.0                                                                          # max linear speed m/s
    DIST_TOL = 0.4                                                                                 # goal tolerance meters
    Kp_angle = 0.8                                                                                 # P gain for heading correction
    GRIPPER_SPEED = 0.05                                                                           # gentle gripper motion
    TORSO_SPEED = 0.05                                                                             # slow lift for stability
    ARM_SPEED = 0.3                                                                                # general arm joint speed
    DRIVE_INTO_JAR_DURATION = [1.5, 1.5, 11.0]                                                     # push-in durations per jar s
    # Feature flags
    REACTIVE_AVOIDANCE = True                                                                      # enable lidar based reflexes
    # Frame/map settings
    STATIC_MAP_MODE = True                                                                         # use static  map
    MAP_FRAME_ALIGNED = True                                                                       # align map frame to robot frame
    COORDINATE_FRAME_VALIDATION = True                                                             
    # Grid origin & resolution
    MAP_ORIGIN_X = -2.25                                                                           # world X at map left edge
    MAP_ORIGIN_Y = 1.6633                                                                          # world Y at map top/bottom edge
    MAP_RESOLUTION = 0.025                                                                         # meters per pixel 
    MAP_WIDTH_METERS = 5.0                                                                         # map width in meters
    MAP_HEIGHT_METERS = 7.5                                                                        # map height in meters
    # Orientation
    MAP_Y_AXIS_UP = False                                                                          # image-style Y axis if False
    WORLD_FRAME_ORIGIN = "bottom_left"                                                             
    MAPPING_WAYPOINTS: List[Tuple[float, float]] = [
        (0.34, -0.19), (0.47, -0.94), (0.8, -1.8), (0.60, -2.4), (0.4, -2.8),
        (-0.51, -3.0), (-1.2, -2.7), (-1.4, -2.50), (-1.5, -1.25), (-1.7, -0.50),
        (-1.5, 0.10), (-0.70, 0.2), (-0.6, 0.3), (-0.20, 0.1), (-0.1, 0.01), (0.24, -0.09)
    ]                                                                                               # coverage loop for SLAM
    JAR_POSITIONS: List[Tuple[float, float, float]] = [
        (1.7143, 0.7042, 0.8894), (1.7123, -0.3007, 0.8894), (1.98, 0.509, 0.8894)
    ]                                                                                               # (x, y, z) pickup targets
    DROPOFF_POINTS: List[Tuple[float, float, float]] = [
        (-0.08, -0.3, 0.8), (-0.09, -0.35, 0.8), (-0.13, -0.33, 0.8)
    ]                                                                                               # slight XY offsets; z = surface
