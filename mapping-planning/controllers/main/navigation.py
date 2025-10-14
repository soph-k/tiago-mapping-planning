from __future__ import annotations
import numpy as np
import random
from collections import deque
from dataclasses import dataclass, field
from core import BehaviorNode, Status, blackboard, nav_logger, NormalizeAngle, TH_FREE_PLANNER

# ------------------------- Utilities -----------------------------------------

def AngDiff(a: float, b: float) -> float:
    return NormalizeAngle(a - b)

def FindMinimumFiniteValue(a):
    a = np.asarray(a)
    f = a[np.isfinite(a)]
    if f.size:
        return float(np.min(f))
    return 10.0

def ScanLidarSectors(ranges):
    n = len(ranges)
    t = n // 3
    left = FindMinimumFiniteValue(ranges[:t])
    front = FindMinimumFiniteValue(ranges[t:2 * t])
    right = FindMinimumFiniteValue(ranges[2 * t:])
    m = FindMinimumFiniteValue(ranges)
    return left, front, right, min(left, front, right, m)

# ------------------------- Device helpers ------------------------------------

def GetRobotDevicesFromBlackboard(bb, *keys):
    devs = [bb.Get(k) for k in keys]
    miss = []
    for k, d in zip(keys, devs):
        if d is None:
            miss.append(k)
    if miss:
        nav_logger.Warning(f"Missing devices: {miss}")
    return devs

def ClipMotorVelocity(motor, v, name):
    if motor is None:
        nav_logger.Error(f"{name}: Motor device is None")
        return 0.0
    vmax = motor.getMaxVelocity()
    return float(np.clip(v, -vmax, vmax))

def SafeSetMotorVelocities(L, R, lv, rv):
    if not (L and R):
        nav_logger.Error("safe_set_motor_velocities: Motor devices are None")
        return False
    L.setVelocity(ClipMotorVelocity(L, lv, "MotorL"))
    R.setVelocity(ClipMotorVelocity(R, rv, "MotorR"))
    return True

def StopMotors(L, R):
    SafeSetMotorVelocities(L, R, 0.0, 0.0)
