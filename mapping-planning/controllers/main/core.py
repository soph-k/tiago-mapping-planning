from __future__ import annotations            # Used in webots and older versions of python
import os                                     # For device stuff
import time                                   # Times stamps etc
import traceback                              # Better looking print statements
from enum import Enum                         
from pathlib import Path                      # For file paths
from dataclasses import dataclass             # Lightweight class containers for parameters.
from typing import Any, Dict, List, Optional, Tuple, Callable  # Helpers
import numpy as np                            # For math 

################################################################################
# =============================== Math utilities ===============================
################################################################################
def NormalizeAngle(angle: float) -> float:    # Normalize angle to [-pi, pi] to avoid wrap-around issues.
    return (angle + np.pi) % (2 * np.pi) - np.pi


def Distance2D(point1, point2) -> float:      # Euclidean distance between two points.
    return np.hypot(point1[0] - point2[0], point1[1] - point2[1])


##################################################################################
# ------------------------ World to Grid coordinate transforms -------------------
##################################################################################
def WorldToGridRaw(world_x: float, world_y: float) -> Tuple[int, int]:
    row = int(40.0 * (world_x + 2.25))         # Meters to grid rows 
    col = int(-52.9 * (world_y - 1.6633))      # Meters to grid cols 
    return row, col                            # Return unclamped

def WorldToGrid(
    world_x: float,
    world_y: float,
    grid_shape: Tuple[int, int] = (200, 300),
    clamp: bool = True,
):
    row, col = WorldToGridRaw(world_x, world_y) # Convert using raw mapping.
    if clamp:
        clamped_row = int(np.clip(row, 0, grid_shape[0] - 1))
        clamped_col = int(np.clip(col, 0, grid_shape[1] - 1))
        return clamped_row, clamped_col
    return (row, col) if (0 <= row < grid_shape[0] and 0 <= col < grid_shape[1]) else None

def GridToWorld(row: int, col: int) -> Tuple[float, float]:
    world_x = row / 40.0 - 2.25
    world_y = -col / 52.9 + 1.6633
    return world_x, world_y

def IsWorldCoordInBounds(
    world_x: float,
    world_y: float,
    map_shape: Tuple[int, int] = (200, 300),
) -> bool:
    row, col = WorldToGridRaw(world_x, world_y)
    return 0 <= row < map_shape[0] and 0 <= col < map_shape[1]

def BresenhamLine(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    points: List[Tuple[int, int]] = []
    delta_x, delta_y = abs(x1 - x0), abs(y1 - y0)
    current_x, current_y = x0, y0
    step_x = 1 if x0 < x1 else -1
    step_y = 1 if y0 < y1 else -1
    error = (delta_x - delta_y) if delta_x > delta_y else (delta_y - delta_x)
    for _ in range(max(delta_x, delta_y) + 1):
        points.append((current_x, current_y))
        error_doubled = error * 2
        if error_doubled > -delta_y:
            error -= delta_y
            current_x += step_x
        if error_doubled < delta_x:
            error += delta_x
            current_y += step_y
    return points

def UpdateTrajectory(
    trajectory: Optional[List[Tuple[float, float]]],
    point: Tuple[float, float],
    min_distance: float = 0.1,
    max_points: int = 200,
) -> List[Tuple[float, float]]:
    trajectory = trajectory or []
    if not trajectory or Distance2D(trajectory[-1], point) > min_distance:
        trajectory.append(point)
    return trajectory[-max_points:]

