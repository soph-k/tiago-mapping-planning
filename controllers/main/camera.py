from typing import Dict, List, Optional, Tuple                                         # typing
import numpy as np                                                                     # math
from utils import RobotDeviceManager, dev, enable, safe, compute_distance              # helpers


class PerceptionController(RobotDeviceManager):                                        # perception manager
    def __init__(self, robot, memory) -> None:
        super().__init__(robot, memory)                                                # base init
        gps = dev(robot, 'gps')
        enable(gps, self.timestep)                                                     # safe enable
        if gps: self.memory.set("gps", gps)                                            # stash gps
        comp = dev(robot, 'compass')
        enable(comp, self.timestep)                                                    # safe enable
        if comp: self.memory.set("compass", comp)                                      # stash compass
        self.camera = dev(robot, "camera")                                             # camera (opt)
        if self.camera:
            try:
                self.camera.enable(self.timestep)                                      # images on
                safe(lambda: self.camera.recognitionEnable(self.timestep))             # recog on
            except Exception:
                self.camera = None                                                     # disable on error
        self.recognized_objects: List[Dict[str, object]] = []                          # detections
        self.distance_threshold: float = 0.1                                           # de-dup radius

    def update(self) -> str:
        if not self.camera:                                                            # no camera
            return "FAILURE"
        try:
            T = self._get_camera_transform()                                           # cam→world
            added = 0
            for obj in self.camera.getRecognitionObjects():                            # iter objs
                pos_cam = np.array(list(obj.getPosition()) + [1])                     # [x,y,z,1]
                pos_world = (T @ pos_cam)[:3]                                          # world xyz
                duplicate_found = any(
                    compute_distance(item["position"], pos_world) < self.distance_threshold
                    for item in self.recognized_objects
                )
                if not duplicate_found:                                                # keep new
                    self.recognized_objects.append({
                        "position": pos_world,                                         # world pos
                        "name": getattr(obj, 'model', 'Unknown'),                      # model
                        "id": getattr(obj, 'id', -1)                                   # id
                    })
                    added += 1
            self.memory.set("recognized_objects", self.recognized_objects)             # publish
            return "SUCCESS" if added > 0 else "RUNNING"                               # status
        except Exception:
            return "FAILURE"                                                           # transient err

    def _get_camera_transform(self) -> np.ndarray:
        gps = self.memory.get("gps")                                                   # gps (opt)
        comp = self.memory.get("compass")                                              # compass (opt)
        if not gps or not comp:                                                        # missing?
            return np.eye(4)                                                           # identity
        gx, gy, gz = gps.getValues()                                                   # base pos
        vals = comp.getValues()                                                        # compass vec
        theta = np.arctan2(vals[0], vals[1])                                           # yaw
        cx = 0.1                                                                        # cam fwd
        cz = 1.2                                                                        # cam height
        return np.array([
            [ np.cos(theta), 0,  np.sin(theta), gx + cx * np.cos(theta) ],             # X row
            [ 0,             1,  0,             gy                        ],           # Y row
            [-np.sin(theta), 0,  np.cos(theta), gz + cz                 ],             # Z row
            [ 0,             0,  0,             1                         ]            # homog
        ], dtype=float)
