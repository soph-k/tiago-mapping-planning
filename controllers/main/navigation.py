import os, json, shutil                                        # file + JSON utils
import numpy as np                                             # arrays / math
from typing import Callable, Dict, List, Optional, Tuple       # type hints
from collections import deque                                  # simple FIFO
from PIL import Image                                          # MapDisplay PNG saving
from scipy import ndimage                                      # light image ops

from config import RobotConfig                                 # robot/map settings
from utils import (                                            # shared helpers
    MemoryBoard, RobotDeviceManager, world_to_pixel, pixel_to_world, compute_distance, block_reduce_max,
    rtime, safe, validate_grid_alignment, _within_map_envelope, standoff
)
from utils import compute_motor_commands                       # wheel cmd helper

###########################################################################################
# ------------------------------ Reactive Avoidance ---------------------------------------
###########################################################################################
def apply_reactive_avoidance(L: float, R: float, heading: float, memory, position_func) -> Tuple[float, float]:
    # Adjust (L,R) wheel speeds using lidar to avoid obstacles                           #
    if not RobotConfig.REACTIVE_AVOIDANCE: return L, R                                  # feature toggle
    if memory.get("suppress_reactive", False): return L, R                              # temporarily off

    tgt = memory.get("navigation_target", None)                                         # current goal (if any)
    if tgt is not None:                                                                 # near-goal easing
        cur = position_func()                                                           # current (x,y)
        dx, dy = tgt[0]-cur[0], tgt[1]-cur[1]                                           # vector to goal
        dist = float(np.hypot(dx, dy))                                                  # distance to goal
        ang_err = abs(np.arctan2(np.sin(np.arctan2(dy, dx) - heading),
                                  np.cos(np.arctan2(dy, dx) - heading)))                # small heading err
        if dist < 1.0 and ang_err < 0.35:                                              # close + aligned
            lidar = memory.get("lidar")                                                 # center brake only
            if lidar:
                try:
                    rng = lidar.getRangeImage()
                    if rng:
                        n = len(rng); c0, c1 = int(n*0.45), int(n*0.55)                 # center slice
                        fc_vals = [r for r in rng[c0:c1] if np.isfinite(r) and r > 0.1] # valid readings
                        fc = min(fc_vals) if fc_vals else float('inf')                  # nearest ahead
                        if fc < 0.12: return 0.0, 0.0                                   # hard stop
                except Exception:
                    pass
            return L, R                                                                  # keep commands

    lidar = memory.get("lidar")                                                         # general reflex
    try:
        rng = lidar.getRangeImage()
        if not rng: return L, R                                                         # no data -> pass

        def smin(a, b):                                                                 # safe min in slice
            s = rng[a:b]; v = [r for r in s if np.isfinite(r) and r > 0.1]; return min(v) if v else float('inf')

        n = len(rng)                                                                     # slice bands
        fc = smin(int(n*0.45), int(n*0.55))                                             # front-center
        fl = smin(int(n*0.35), int(n*0.45))                                             # front-left
        fr = smin(int(n*0.55), int(n*0.65))                                             # front-right
        ls = smin(int(n*0.15), int(n*0.35))                                             # left side
        rs = smin(int(n*0.65), int(n*0.85))                                             # right side
        thr = 0.22                                                                       # proximity thr
        obs = { 'front_center': fc < thr, 'front_left': fl < thr, 'front_right': fr < thr, 'left': ls < thr, 'right': rs < thr }  # flags

        if tgt is not None:                                                             # relax sides near goal
            cur = position_func(); dx, dy = tgt[0]-cur[0], tgt[1]-cur[1]
            dist = float(np.hypot(dx, dy))
            ang_err = abs(np.arctan2(np.sin(np.arctan2(dy, dx) - heading),
                                      np.cos(np.arctan2(dy, dx) - heading)))
            if dist < 0.70 and ang_err < 0.18:
                obs['front_left'] = obs['front_right'] = obs['left'] = obs['right'] = False

        if obs['front_center'] or (obs['front_left'] and obs['front_right']):           # blocked ahead?
            if obs['front_left'] and not obs['front_right']:  return  1.5, -1.5         # turn right
            if obs['front_right'] and not obs['front_left']:  return -1.5,  1.5         # turn left
            return 0.0, 0.0                                                              # both sides -> stop

        if obs['left']:  R *= 1.1                                                       # bias away left
        if obs['right']: L *= 1.1                                                       # bias away right
        return L, R                                                                      # adjusted cmds
    except Exception:
        return L, R                                                                      # fail-safe

###########################################################################################
# ------------------------------ Navigation & Mapping -------------------------------------
###########################################################################################
class NavigationController(RobotDeviceManager):
    def __init__(self, robot, memory: MemoryBoard) -> None:
        super().__init__(robot, memory)                                                 # base init
        self.prob_map = np.zeros((RobotConfig.MAP_WIDTH, RobotConfig.MAP_SIZE), np.float32)   # prob grid
        self.trajectory_points: deque[Tuple[float, float]] = deque(maxlen=3000)         # recent path
        self.LIDAR_OFFSET_X, self.LIDAR_START_IDX, self.LIDAR_END_IDX = 0.202, 80, -80  # lidar params
        self.waypoints, self.current_wp_idx = RobotConfig.MAPPING_WAYPOINTS, 0          # mapping WPs
        self.start_time, self.max_mapping_time = None, 90.0                             # time budget
        self.sensor_array, self.arm_motors, self.sensor_array_positioned = {}, {}, False  # arm pose
        self.planned_path: List[Tuple[float, float]] = []                                # A* path
        self.current_path_idx, self.use_path_planning = 0, False                         # A* progress
        self.using_loaded_cspace = False                                                 # loaded vs live
        if not validate_grid_alignment(): pass                                           # non-fatal sanity

    # ======== Save/Load Maps (+ metadata) ========
    def save_maps(self, cspace_map: np.ndarray, map_dir="map") -> None:
        try:
            print(f"Saving maps to {map_dir} directory...")
            os.makedirs(map_dir, exist_ok=True)                                          # ensure folder
            print(f"Created/verified directory: {os.path.abspath(map_dir)}")
            cspace_path = os.path.join(map_dir, "cspace.npy")
            prob_path = os.path.join(map_dir, "prob_map.npy")
            print(f"Saving cspace to {cspace_path}")
            np.save(cspace_path, cspace_map)                                             # save cspace
            print(f"Saving probability map to {prob_path}")
            np.save(prob_path, self.prob_map)                                            # save prob
            metadata = { "map_origin_x": RobotConfig.MAP_ORIGIN_X, "map_origin_y": RobotConfig.MAP_ORIGIN_Y,
                         "map_resolution": RobotConfig.MAP_RESOLUTION, "map_width_meters": RobotConfig.MAP_WIDTH_METERS,
                         "map_height_meters": RobotConfig.MAP_HEIGHT_METERS, "map_y_axis_up": RobotConfig.MAP_Y_AXIS_UP,
                         "world_frame_origin": RobotConfig.WORLD_FRAME_ORIGIN, "map_width_pixels": RobotConfig.MAP_WIDTH,
                         "map_height_pixels": RobotConfig.MAP_SIZE, "creation_timestamp": rtime(self.robot) if hasattr(self, 'robot') else 0.0 }
            
            metadata_path = os.path.join(map_dir, "cspace_metadata.json")
            print(f"Saving metadata to {metadata_path}")
            with open(metadata_path, 'w') as f: json.dump(metadata, f, indent=2)  # write meta
            print("Maps saved ")
        except Exception as e:
            print(f"Failed to save maps: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")

    def load_maps(self, map_dir="map") -> bool:
        try:
            print(f"Attempting to load saved maps from {map_dir}...")
            cspace_path = os.path.join(map_dir, "cspace.npy"); prob_path = os.path.join(map_dir, "prob_map.npy")  # file paths
            if not os.path.exists(cspace_path): 
                print("No saved maps found")
                return False                               # nothing saved

            cs = np.load(cspace_path)                                                      # load cspace
            if not self._validate_cspace_integrity(cs): return False                       # sanity check

            if os.path.exists(prob_path):                                                  # load prob map
                pm = np.load(prob_path)
                if not self._validate_probability_map_integrity(pm): return False
                self.prob_map = pm; self.memory.set("prob_map", pm)                        # publish

            metadata_path = os.path.join(map_dir, "cspace_metadata.json")                  # metadata
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f: saved_metadata = json.load(f)      # read meta
                    for k in ["MAP_ORIGIN_X","MAP_ORIGIN_Y","MAP_RESOLUTION","MAP_WIDTH_METERS","MAP_HEIGHT_METERS","MAP_Y_AXIS_UP"]:
                        meta_key = k.lower()
                        if meta_key in saved_metadata: setattr(RobotConfig, k, saved_metadata[meta_key])  # adopt saved
                except Exception:
                    pass

            if not self._verify_cspace_coordinate_alignment(cs):                           # alignment?
                if not self._regenerate_cspace_with_current_transforms(): return False     # try rebuild
                cs = self.memory.get("cspace")                                             # refreshed map

            self.memory.set("cspace", cs); self.memory.set("cspace_complete", True)       # publish map
            self.memory.set("mapping_complete", True); self.using_loaded_cspace = True    # flags
            print("Successfully loaded saved maps")                         # milestone: loaded maps successfully
            return True
        except Exception:
            return False                                                                   # failed load

    # ======== Sensor array pose for mapping ========
    def initialize_sensor_array(self) -> bool:
        l = self.memory.get("lidar"); l and (self.sensor_array.__setitem__('lidar', l))   # keep lidar ref
        for j in ['torso_lift_joint','arm_1_joint','arm_2_joint','arm_3_joint','arm_4_joint','arm_5_joint','arm_6_joint','arm_7_joint',
                  'gripper_left_finger_joint','gripper_right_finger_joint','head_1_joint','head_2_joint']:
            m = dev(self.robot, j) if (dev := getattr(__import__('utils', fromlist=['dev']), 'dev')) else None  # lazy import
            if m: self.arm_motors[j] = m                                                  # stash motors
        gps, comp = self.memory.get("gps"), self.memory.get("compass")                    # pose sensors
        if gps and comp: self.sensor_array['gps'], self.sensor_array['compass'] = gps, comp
        return self._set_sensor_array_position()                                          # move pose

    def _set_sensor_array_position(self) -> bool:
        if self.sensor_array_positioned or self.memory.get("jar_picking_started", False): return self.sensor_array_positioned  # already set
        sys = self.memory.get("system_instance")
        if sys and getattr(sys, 'state', None) == "MANIPULATION": return False            # skip if manipulating
        cfg = { 'torso_lift_joint': 0.25, 'arm_1_joint': 1.5708, 'arm_2_joint': 1.0472, 'arm_3_joint': 1.5708,
                'arm_4_joint': 0.0, 'arm_5_joint': 0.1745, 'arm_6_joint': -1.5708, 'arm_7_joint': -0.0175,
                'gripper_left_finger_joint': 0.045, 'gripper_right_finger_joint': 0.045, 'head_1_joint': 0.0, 'head_2_joint': 0.0 }  # simple mapping pose
        ok = 0
        from utils import dev                                                            # local import
        for j, p in cfg.items():
            m = self.arm_motors.get(j)
            if m: m.setPosition(p); m.setVelocity(0.5); ok += 1                          # set + count
        self.sensor_array_positioned = ok > 0                                            # mark ready
        self.memory.set("sensor_array_positioned", True); self.memory.set("navigation_arm_ready", True)  # publish flags
        return self.sensor_array_positioned

    # ======== Mapping loop ========
    def execute_mapping(self) -> str:
        if not self.sensor_array_positioned:
            self.initialize_sensor_array()                                               # ensure arm pose
        if self.start_time is None:
            self.start_time = rtime(self.robot); print("Mapping phase started")    
        if rtime(self.robot) - self.start_time > self.max_mapping_time:                  # time cap
            return self._finish_mapping()
        if self.current_wp_idx >= len(self.waypoints):                                   # all WPs done
            return self._finish_mapping()

        (xw, yw), th = self._position(), self._orientation()                             # robot pose
        if len((xw, yw)) < 2: return "FAILURE"                                           # invalid pose
        if not np.all(np.isfinite([xw, yw, th])) or not _within_map_envelope(xw, yw):    # out-of-bounds
            self._stop(); return "RUNNING"                                               # wait safely

        self.trajectory_points.append((xw, yw))                                          # log path
        self._update_map(xw, yw, th)                                                     # fuse lidar
        self._update_path_planning((xw, yw))                                             # refresh A*

        tx, ty = self._get_current_target((xw, yw))                                      # next target
        dx, dy = tx - xw, ty - yw; rho = np.hypot(dx, dy)                                # distance

        if rho < RobotConfig.DIST_TOL:                                                   # target reached
            if self.use_path_planning and self.planned_path and self.current_path_idx < len(self.planned_path) - 1:
                self.current_path_idx += 1                                               # next A* wp
            else:
                self.current_wp_idx += 1; self.current_path_idx = 0                      # next mapping WP
                self.planned_path = []; self.use_path_planning = False
                print(f"Reached waypoint {self.current_wp_idx}")                    # MILESTONE
            return "RUNNING"

        alpha = np.arctan2(np.sin(np.arctan2(dy, dx) - th), np.cos(np.arctan2(dy, dx) - th))  # bearing err
        L, R = compute_motor_commands(alpha, rho)                                        # base (L,R)
        if self.sensor_array_positioned: L, R = apply_reactive_avoidance(L, R, th, self.memory, self._position)  # reflex
        if self.using_loaded_cspace: L = np.clip(L, -0.8, 0.8); R = np.clip(R, -0.8, 0.8)   # gentle on loaded map

        self._set_wheel_speeds(L, R)                                                     # drive
        self.memory.set("prob_map", self.prob_map); self.memory.set("trajectory_points", list(self.trajectory_points))  # publish
        return "RUNNING"

    def _update_map(self, x: float, y: float, th: float) -> None:
        lidar = self.memory.get("lidar")
        if not lidar: return                                                             # need lidar
        try:
            lr = np.array(lidar.getRangeImage())                                        # ranges
            if len(lr) == 0: return
            ang = np.linspace(2*np.pi/3, -2*np.pi/3, len(lr))                           # angles
            vs, ve = max(0, self.LIDAR_START_IDX), min(len(lr), len(lr)+self.LIDAR_END_IDX)  # slice
            vr, va = [], []                                                             # valid hits
            for i in range(vs, ve):
                v = lr[i]; v = 100.0 if not np.isfinite(v) else v                        # far for NaN
                if 0.12 < v < 8.0:
                    a_local = ang[i]
                    if abs(a_local) < np.pi/2: vr.append(v); va.append(a_local)          # forward-only
            if not vr: return
            wTr = np.array([[np.cos(th), -np.sin(th), x], [np.sin(th), np.cos(th), y], [0,0,1]])  # SE(2)
            Xr = np.array([np.array(vr)*np.cos(va)+self.LIDAR_OFFSET_X, np.array(vr)*np.sin(va), np.ones(len(vr))])  # rays→robot
            W = wTr @ Xr                                                                 # to world
            traj_check = len(self.trajectory_points) > 10 and len(self.trajectory_points) % 5 == 0  # optional filter
            recent_traj = list(self.trajectory_points)[-10:] if traj_check else []
            for i,(wx,wy) in enumerate(zip(W[0],W[1])):
                if traj_check and recent_traj:
                    if any(compute_distance((wx, wy), t) < 0.20 for t in recent_traj): continue   # skip echoes
                px, py = world_to_pixel(wx, wy)                                          # world->pixel
                if 0 <= px < RobotConfig.MAP_WIDTH and 0 <= py < RobotConfig.MAP_SIZE:
                    d = vr[i]; w = 0.008 if d < 1.0 else 0.006 if d < 3.0 else 0.004     # dist weight
                    self.prob_map[px, py] = min(self.prob_map[px, py] + w, 1.0)          # clamp
        except Exception:
            pass                                                                          # stay robust

    def _finish_mapping(self) -> str:
        self._stop()                                                                      # halt
        self.memory.set("prob_map", self.prob_map); self.memory.set("trajectory_points", list(self.trajectory_points))  # publish
        print("Finishing mapping -- building C-space")                                
        try:
            cs = self.generate_cspace(); self.memory.set("cspace", cs)                   # build + stash
            display = self.memory.get("display"); display and display.save_display_png("mapping_complete.png", "map")  # snapshot
        except Exception as e:
            print(f"Failed to generate cspace: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
        self.memory.set("mapping_complete", True)                                        # flag done
        print("Mapping complete!")                                                   
        return "SUCCESS"

    def generate_cspace(self) -> np.ndarray:
        print("Generating configuration space from probability map...")
        if np.max(self.prob_map) == 0:
            print("Warning - probability map is empty")
        radius_pixels, down = 4, 4                                                       # inflate + downsample
        prob_small = block_reduce_max(self.prob_map, down, down)                         # pooled map
        r = max(1, int(np.ceil(radius_pixels / down)))                                   # disk radius
        offs = [(dx,dy) for dx in range(-r,r+1) for dy in range(-r,r+1) if dx*dx+dy*dy <= r*r]  # struct elem
        thresh = 0.008; c_small = np.ones_like(prob_small, np.float32)                   # start free
        ox, oy = np.where(prob_small > thresh)                                           # obstacle seeds
        for x,y in zip(ox,oy):
            for dx,dy in offs:
                nx, ny = x+dx, y+dy
                if 0 <= nx < c_small.shape[0] and 0 <= ny < c_small.shape[1]: c_small[nx, ny] = 0.0  # inflate
        cspace = np.kron(c_small, np.ones((down, down), np.float32))[:self.prob_map.shape[0], :self.prob_map.shape[1]]  # upsample
        if not self._validate_cspace_integrity(cspace): return None                      
        loaded_cs = self.memory.get("cspace")                                            # compare if had one
        if loaded_cs is not None and not self.using_loaded_cspace:
            if not self._compare_live_vs_loaded(cspace, loaded_cs): self._repair_corrupted_map("map")  # best-effort repair

        self.memory.set("cspace_complete", True); self.save_maps(cspace, "map")         # publish + persist
        print("C-space generated")                                                   
        return cspace

    # ======== A* path planning ========
    def _find_nearest_valid(self, px: Tuple[int, int], cspace: np.ndarray, max_search=30) -> Optional[Tuple[int, int]]:
        h, w = cspace.shape; x0, y0 = px                                                # bounds
        if 0 <= y0 < h and 0 <= x0 < w and cspace[y0, x0] > 0.5: return (x0, y0)        # already free
        q, visited = deque([(x0, y0, 0)]), {(x0, y0)}                                   # BFS frontier
        dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]                  
        while q:
            x, y, d = q.popleft()
            if d > max_search: break
            for dx, dy in dirs:
                nx, ny = x+dx, y+dy
                if (nx, ny) in visited: continue
                visited.add((nx, ny))
                if 0 <= ny < h and 0 <= nx < w and cspace[ny, nx] > 0.5: return (nx, ny)
                q.append((nx, ny, d+1))
        return None                                                                      # none found

    def _astar(self, start_px: Tuple[int,int], goal_px: Tuple[int,int], cspace: np.ndarray) -> List[Tuple[float,float]]:
        def px2w(px: int, py: int) -> Tuple[float, float]: return pixel_to_world(px, py)  
        def valid(x: int, y: int) -> bool:
            if not (0 <= x < cspace.shape[0] and 0 <= y < cspace.shape[1]): return False # in-bounds
            margin = 5
            if x < margin or x >= cspace.shape[0]-margin or y < margin or y >= cspace.shape[1]-margin: return False  # margin
            return cspace[x, y] > 0.5                                                    # free cell

        if not valid(*start_px):
            ns = self._find_nearest_valid(start_px, cspace, 30)
            if ns: start_px = ns
            else: return [px2w(*goal_px)]                                               # degrade to goal

        if not valid(*goal_px):
            ng = self._find_nearest_valid(goal_px, cspace, 30)
            if ng: goal_px = ng
            else: return [px2w(*goal_px)]                                               # degrade to goal

        def h(x1,y1,x2,y2): return np.hypot(x1-x2, y1-y2)                               # heuristic
        neighbors = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]             # moves
        heapq = __import__('heapq'); open_set = []                                      # frontier
        heapq.heappush(open_set, (0, start_px[0], start_px[1]))
        came_from, g_score, f_score = {}, {start_px: 0}, {start_px: h(*start_px, *goal_px)}  # maps

        while open_set:
            _, cx, cy = heapq.heappop(open_set); current = (cx, cy)                     # best
            if current == goal_px:
                path = []; cur = current                                                # reconstruct
                while cur in came_from: path.append(px2w(*cur)); cur = came_from[cur]
                path.append(px2w(*start_px)); path.reverse(); return path               # world path
            for dx, dy in neighbors:
                nx, ny = cx+dx, cy+dy
                if not valid(nx, ny): continue
                move_cost = 1.414 if abs(dx)==1 and abs(dy)==1 else 1.0                 # diag vs axial
                ng = g_score[current] + move_cost; nb = (nx, ny)
                if nb not in g_score or ng < g_score[nb]:
                    came_from[nb] = current; g_score[nb] = ng; f_score[nb] = ng + h(nx, ny, *goal_px)
                    heapq.heappush(open_set, (f_score[nb], nx, ny))
        return [px2w(*goal_px)]                                                         # fallback

    def _is_within_safe_bounds(self, pos: Tuple[float, float]) -> bool:
        x, y = pos; xmin, xmax, ymin, ymax = -2.0, 1.0, -3.2, 0.5                        # sandbox box
        return xmin <= x <= xmax and ymin <= y <= ymax                                   # inside?

    def _is_pixel_within_bounds(self, px: Tuple[int, int], shape: Tuple[int, int]) -> bool:
        x, y = px; w, h = shape; return 0 <= x < w and 0 <= y < h                        # px bounds

    def _validate_static_map_frame_alignment(self) -> bool:
        from utils import validate_frame_alignment, world_to_pixel
        try:
            pos = self._position(); ori = self._orientation()                            # pose
            if not validate_frame_alignment(pos, ori): return False                      # invalid frame
            cspace = self.memory.get("cspace"); 
            if cspace is None: return False                                              # no map
            px, py = world_to_pixel(*pos)
            if not self._is_pixel_within_bounds((px, py), cspace.shape): return False    # oob
            if cspace[px, py] < 0.5: return False                                        # not free
            return True
        except Exception:
            return False

    # ======== C-space validation (quiet; no spam) ========
    def _validate_cspace_integrity(self, cs: np.ndarray) -> bool:
        try:
            if not isinstance(cs, np.ndarray) or cs.ndim != 2: return False             # type/dims
            if cs.shape != (RobotConfig.MAP_WIDTH, RobotConfig.MAP_SIZE): return False  # shape
            if not np.all((cs >= 0.0) & (cs <= 1.0)) or not np.all(np.isfinite(cs)): return False  # values
            fr = np.mean(cs > 0.5)                                                      # free ratio
            if fr < 0.1 or fr > 0.9: return False                                      # plausible?
            if fr < 0.01 or fr > 0.99: return False                                    # extreme
            return True
        except Exception:
            return False

    def _validate_probability_map_integrity(self, pm: np.ndarray) -> bool:
        try:
            if not isinstance(pm, np.ndarray) or pm.ndim != 2: return False             # type/dims
            if pm.shape != (RobotConfig.MAP_WIDTH, RobotConfig.MAP_SIZE): return False  # shape
            if not np.all((pm >= 0.0) & (pm <= 1.0)) or not np.all(np.isfinite(pm)): return False  # values
            return True
        except Exception:
            return False

    def _analyze_cspace_quality(self, cs: np.ndarray) -> None:
        try:
            _ = np.sum(cs > 0.5); _ = cs.size                                           # kept quiet (dev use)
            edge_margin = 10; _ = np.mean(cs[edge_margin:-edge_margin, edge_margin:-edge_margin] > 0.5)  # center free
            _ = ndimage.label(cs < 0.5)                                                 # components
        except Exception:
            pass

    def _compare_live_vs_loaded(self, live_cs: np.ndarray, loaded_cs: np.ndarray) -> bool:
        try:
            if live_cs.shape != loaded_cs.shape: return False                            # shape mismatch
            diff = np.abs(live_cs - loaded_cs)                                           # abs diff
            return not (np.mean(diff) > 0.1 or np.max(diff) > 0.5)                       # consistent?
        except Exception:
            return False

    def _repair_corrupted_map(self, map_dir="map") -> bool:
        try:
            backup_dir = f"{map_dir}_backup"                                            # backup path
            if os.path.exists(map_dir): shutil.copytree(map_dir, backup_dir, dirs_exist_ok=True)  # backup
            for fname in ("cspace.npy", "prob_map.npy"):                                 # remove bad
                p = os.path.join(map_dir, fname)
                if os.path.exists(p): os.remove(p)
            self.memory.set("cspace", None); self.memory.set("cspace_complete", False)   # reset flags
            self.memory.set("mapping_complete", False)
            print(" Corrupted maps removed - remap needed")                           
            return True
        except Exception:
            return False

    # ======== Alignment/Misalignment helpers (quiet) ========
    def _verify_cspace_coordinate_alignment(self, cspace: np.ndarray) -> bool:
        from utils import world_to_pixel
        try:
            pos = self._position()
            if not pos or pos == (0.0, 0.0): return True                                 # not fatal
            px, py = world_to_pixel(*pos)
            if not self._is_pixel_within_bounds((px, py), cspace.shape): return False    # oob
            if cspace[px, py] < 0.5: return False                                        # not free
            tests = [(0.0,0.0),(0.5,0.0),(0.0,-0.5),(-0.5,0.0)]                          # quick probes
            bad = 0
            for t in tests:
                tx, ty = world_to_pixel(*t)
                if self._is_pixel_within_bounds((tx, ty), cspace.shape) and cspace[tx, ty] < 0.5: bad += 1
            return bad <= len(tests)//2                                                  # allow some
        except Exception:
            return False

    def _regenerate_cspace_with_current_transforms(self) -> bool:
        try:
            pm = self.memory.get("prob_map")
            if pm is None: return False                                                  # nothing to regen
            new_cs = self.generate_cspace()
            if new_cs is None: return False
            if not self._verify_cspace_coordinate_alignment(new_cs): return False
            self.memory.set("cspace", new_cs); self.using_loaded_cspace = False          # publish
            return True
        except Exception:
            return False

    def _visualize_robot_position_in_cspace(self, cspace: np.ndarray, robot_pos: Tuple[float, float]) -> None:
        try:
            import matplotlib.pyplot as plt
            from utils import world_to_pixel
            px, py = world_to_pixel(*robot_pos)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.imshow(cspace.T, origin='lower', cmap='gray_r'); ax1.set_title('C-space (White=Free, Black=Obstacle)')
            if 0 <= px < cspace.shape[0] and 0 <= py < cspace.shape[1]:
                ax1.plot(px, py, 'ro', markersize=10)
            margin = 20; x_min = max(0, px - margin); x_max = min(cspace.shape[0], px + margin)
            y_min = max(0, py - margin); y_max = min(cspace.shape[1], py + margin)
            zoomed = cspace[x_min:x_max, y_min:y_max]
            ax2.imshow(zoomed.T, origin='lower', cmap='gray_r'); ax2.set_title('Zoomed view')
            plt.tight_layout(); plt.savefig('cspace_debug.png', dpi=150, bbox_inches='tight'); plt.close()
        except Exception:
            pass

    def _debug_robot_position_in_cspace(self) -> None:
        try:
            cspace = self.memory.get("cspace"); 
            if cspace is None: return                                                    # no map
            pos = self._position(); 
            if not pos or pos == (0.0, 0.0): return                                      # no pose
            from utils import world_to_pixel
            px, py = world_to_pixel(*pos)                                                # px coords
            _ = (px, py, cspace.shape)                                                   # quiet diagnostics
        except Exception:
            pass

    def _handle_frame_misalignment(self) -> bool:
        try:
            original_static = RobotConfig.STATIC_MAP_MODE; RobotConfig.STATIC_MAP_MODE = False  # temp off
            self.memory.set("original_static_map_mode", original_static); self.memory.set("frame_misalignment_detected", True)
            return True
        except Exception:
            return False

    def _restore_static_map_mode(self) -> bool:
        try:
            if not self.memory.get("frame_misalignment_detected", False): return True    # nothing to do
            if self._validate_static_map_frame_alignment():
                RobotConfig.STATIC_MAP_MODE = self.memory.get("original_static_map_mode", True)
                self.memory.set("frame_misalignment_detected", False)
                return True
            return False
        except Exception:
            return False

    def _detect_grid_misalignment(self) -> bool:
        try:
            if len(self.trajectory_points) < 10: return False                            # need history
            recent = list(self.trajectory_points)[-10:]
            if len(recent) < 3: return False
            moves = [(recent[i][0]-recent[i-1][0], recent[i][1]-recent[i-1][1]) for i in range(1, len(recent))]  # deltas
            flips = 0
            for i in range(1, len(moves)):
                a0 = np.arctan2(moves[i-1][1], moves[i-1][0]); a1 = np.arctan2(moves[i][1], moves[i][0])
                if abs(np.arctan2(np.sin(a1-a0), np.cos(a1-a0))) > np.pi/2: flips += 1   # large change
            return flips > len(moves)*0.5                                                # erratic?
        except Exception:
            return False

    def _correct_grid_misalignment(self) -> bool:
        try:
            from utils import validate_grid_alignment
            if not RobotConfig.MAP_Y_AXIS_UP:
                RobotConfig.MAP_Y_AXIS_UP = True
                if validate_grid_alignment(): return True
                RobotConfig.MAP_Y_AXIS_UP = False
            ox, oy = RobotConfig.MAP_ORIGIN_X, RobotConfig.MAP_ORIGIN_Y
            for dx in [-0.1, 0.1, -0.2, 0.2]:
                for dy in [-0.1, 0.1, -0.2, 0.2]:
                    RobotConfig.MAP_ORIGIN_X = ox + dx; RobotConfig.MAP_ORIGIN_Y = oy + dy
                    if validate_grid_alignment(): return True
            RobotConfig.MAP_ORIGIN_X, RobotConfig.MAP_ORIGIN_Y = ox, oy                  # restore
            return False
        except Exception:
            return False

    # ======== Planning API used by behaviors ========
    def plan_path(self, start: Tuple[float,float], goal: Tuple[float,float]) -> List[Tuple[float,float]]:
        cspace = self.memory.get("cspace")                                               # quick planner
        if cspace is None: return [goal]                                                 # direct
        return self._astar(world_to_pixel(*start), world_to_pixel(*goal), cspace)        # A*

    def plan_path_to_goal(self, cur: Tuple[float,float], goal: Tuple[float,float]) -> List[Tuple[float,float]]:
        print(f"Planning path from {cur} to {goal}")
        cspace = self.memory.get("cspace")
        if cspace is None: return [goal]                                                 # no cspace
        if not np.all(np.isfinite(cur)) or not np.all(np.isfinite(goal)): return []      # bad inputs
        if not _within_map_envelope(cur[0], cur[1]) or not _within_map_envelope(goal[0], goal[1]): return []  # oob
        if self.using_loaded_cspace: self._debug_robot_position_in_cspace()              # quiet check
        if RobotConfig.STATIC_MAP_MODE and not self._validate_static_map_frame_alignment():
            if not self._handle_frame_misalignment(): return [goal]                      # fallback
        if not self._is_within_safe_bounds(cur) or not self._is_within_safe_bounds(goal): return [goal]  # clamp to box
        sgoal = standoff(cur, goal, dist=0.45)                                           # approach point
        start_px = world_to_pixel(*cur); goal_px = world_to_pixel(*sgoal)                # to pixels
        if not self._is_pixel_within_bounds(start_px, cspace.shape): return [goal]       # guard
        if not self._is_pixel_within_bounds(goal_px, cspace.shape): return [goal]        # guard
        path = self._astar(start_px, goal_px, cspace)                                    # plan
        if len(path) <= 1:                                                               # try direct
            goal_px_direct = world_to_pixel(*goal)
            if self._is_pixel_within_bounds(goal_px_direct, cspace.shape):
                path = self._astar(start_px, goal_px_direct, cspace)
        if len(path) <= 1: path = [goal]                                                 # last resort
        safe_path = []
        for p in path:
            if self._is_within_safe_bounds(p): safe_path.append(p)                       # keep safe
            else: break
        if not safe_path: safe_path = [goal]                                             # ensure non-empty
        print(f" Planned path with {len(safe_path)} waypoints")                       
        self.planned_path, self.current_path_idx, self.use_path_planning = safe_path[:], 0, True  # publish
        return safe_path

    # ======== Mapping path helpers ========
    def _update_path_planning(self, _cur: Tuple[float, float]) -> None:
        if not self.memory.get("cspace_complete", False): return                         # need cspace
        if self.current_wp_idx < len(self.waypoints) and (not self.planned_path or self.current_path_idx >= len(self.planned_path)):
            goal = self.waypoints[self.current_wp_idx]                                   # next WP
            self.planned_path = self.plan_path(_cur, goal); self.current_path_idx = 0; self.use_path_planning = True  # seed path

    def _get_current_target(self, cur: Tuple[float, float]) -> Tuple[float, float]:
        if self.use_path_planning and self.planned_path and self.current_path_idx < len(self.planned_path):  # follow A*
            return self.planned_path[self.current_path_idx]                              # current A* wp
        if self.current_wp_idx < len(self.waypoints): return self.waypoints[self.current_wp_idx]  # mapping WP
        return cur                                                                       # no target
