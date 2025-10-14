from __future__ import annotations                                    
import numpy as np                                                    # Numerical arrays and math
from scipy import ndimage as ndi                                      # Image processing 
from core import (                                                    # Project specific utilities and logging
    BehaviorNode, Status, blackboard, EveryNCalls, EveryNSeconds,
    WorldToGrid, BresenhamLine, map_logger
)


###############################################################################
# ------------------------- C-space builder -----------------------------------
###############################################################################
def build_cspace(                                                   # Build cspace from prob map
    probability_map: np.ndarray,                                    # Occupancy probabilities per cell
    occupied_threshold: float,                                      
    resolution_meters: float,                                       # Meters per pixel in the map
    robot_radius_meters: float,                                     # Robot radius in meters
    safety_margin: float,                                           # Extra margin around robot
    morph_closing: int,                                             # Structuring element size for closing
    morph_iterations: int,                                          # Number of closing iterations
    downsample: int = 1,                                            # Optional downsampling factor for speed
    inflation_scale: float = 1.0                                    # Scale factor on inflation radius
) -> np.ndarray:
    pm = np.asarray(probability_map, dtype=np.float32)              # Ensure float array
    if downsample > 1:                                              # If decimating resolution
        hh = pm.shape[0] // downsample                              # Downsampled height
        ww = pm.shape[1] // downsample                              # Downsampled width
        cut = pm[:hh * downsample, :ww * downsample]                # Crop to multiple of factor
        pm_ds = cut.reshape(hh, downsample, ww, downsample).mean(axis=(1, 3)) 
    else:
        pm_ds = pm                                                 
    core = (pm_ds >= occupied_threshold).astype(np.uint8)           
    labels, num = ndi.label(core)                                   # Connected components in obstacle core
    if num:                                                         # If any components exist
        res_px = max(1, downsample)                                 # Pixels per original pixel in ds space
        m_per_px = max(1e-6, resolution_meters * res_px)            # Meters per downsampled pixel
        min_side_m = 0.05                                          
        solidity_min = 0.05                                        
        min_area_px = int((min_side_m / m_per_px) ** 2)             # Area threshold in pixels
        objects = ndi.find_objects(labels)                          
        for i, sl in enumerate(objects, start=1):                   
            if not sl:                                              # Skip empty slices
                continue
            h = sl[0].stop - sl[0].start                            # Height of bbox
            w = sl[1].stop - sl[1].start                            # Width of bbox
            area = h * w                                          
            if area < min_area_px:                                  # Skip small components
                continue
            comp = (labels[sl] == i)                                # Component mask inside bbox
            solidity = float(comp.sum()) / float(area)              # Occupancy fraction of bbox
            if solidity >= solidity_min:                            # If solid enough, pad and fill
                pad_h = max(1, h // 3) + max(1, min(h, w) // 6)     # Vertical padding
                pad_w = max(1, w // 3) + max(1, min(h, w) // 6)     # Horizontal padding
                r0 = max(0, sl[0].start - pad_h)                   
                r1 = min(core.shape[0], sl[0].stop + pad_h)         
                c0 = max(0, sl[1].start - pad_w)                    
                c1 = min(core.shape[1], sl[1].stop + pad_w)         
                core[r0:r1, c0:c1] = 1                              # Mark padded region occupied
    if morph_closing > 0 and morph_iterations > 0:                  
        se = np.ones((morph_closing, morph_closing), dtype=np.uint8)  # Square structuring element
        core = ndi.binary_closing(core, structure=se, iterations=morph_iterations)  #   close gaps
    free_mask = (~core.astype(bool))                                # Free = not occupied
    dist_px = ndi.distance_transform_edt(free_mask).astype(np.float32)  # Euclidean distance to nearest obstacle
    inflate_px = (robot_radius_meters + safety_margin) / max(resolution_meters, 1e-6)  # Convert meters to pixels
    inflate_px = float(max(0.0, inflate_px * inflation_scale))        # Scale and clamp non-negative
    denom = max(inflate_px, 1e-6)                                    # Avoid divide-by-zero
    cspace_ds = np.clip((dist_px - inflate_px) / denom, 0.0, 1.0)    
    halo_strength = 0.65                                             
    if halo_strength > 0.0:                                          # Apply halo if enabled
        halo_radius = inflate_px * 5.0                               # Radius where halo fades out
        halo_denom = max(halo_radius - inflate_px, 1e-6)             # Avoid zero division
        halo_dist = np.clip((dist_px - inflate_px) / halo_denom, 0.0, 1.0)  # Normalized halo distance
        halo = (1.0 - halo_dist) * halo_strength                     # Stronger near obstacles
        cspace_ds = np.clip(cspace_ds - halo, 0.0, 1.0)              # Subtract halo, clamp

    # upsample back to original size if needed
    if downsample > 1:                                               # If we reduced resolution earlier
        up = ndi.zoom(cspace_ds, zoom=downsample, order=0)         
        return up[:pm.shape[0], :pm.shape[1]]                        # Crop to original shape
    return cspace_ds                                                 # Return cspace at current resolution

###############################################################################
# ------------------------- Helpers -------------------------------------------
###############################################################################
def is_cspace_frozen(bb) -> bool:                                    # Check if C-space updates are frozen
    val = bb.Get("cspace_frozen", False)                             # Read flag from blackboard
    return bool(val)                                                 # Cast to bool and return

def validate_inflation_parameters():                                 # Heuristic checks
    s = max(52.9, 40.0)                                              # pixels per meter for y/x scales
    rpx = 0.16 * s * 0.55                     
    r_pix = int(max(1, round(rpx)))                                  # Integer pixel radius
    bubble = int(r_pix + 5)                                          # Halo/buffer radius
    core_m = r_pix / s                                               # Core radius in meters
    margin = core_m - 0.16                                           # Margin beyond robot radius
    # Tips removed - no more annoying suggestions!


###############################################################################
# ------------------------- BT: Lidar Mapping ---------------------------------
################################################################################
class LidarMappingBT(BehaviorNode):                                   # Behavior tree node for building map and c-space
    def __init__(self, params, bb=None):                              # Initialize mapping behavior
        super().__init__("LidarMappingBT")                            #   set node name
        self.params = params                                          #   store parameters object
        self.bb = bb or blackboard                                    #   use provided or global blackboard
        self.step_counter = 0                                          #   iteration counter
        self.mission_complete = False                                  #   whether mapping should complete
        self.min_scan_ticks = 25                                       #   minimal ticks before completing
        self.robot_radius = params.robot_radius                        #   cache robot radius
        self.mapping_interval_ms = 100                                 #   target interval between updates (ms)
        self.last_mapping_time = 0.0                                   #   last update time
        self.current_decimation = 4                                    #   lidar ray decimation factor
        self.cspace_log_limiter = EveryNCalls(20)                      #   log cspace less frequently
        self.cspace_build_limiter = EveryNSeconds(0.5)                 #   build cspace at most twice/sec
        self.lidar_update_limiter = EveryNCalls(2)                     #   (unused here) for lidar updates
        self.error_log_limiter = EveryNSeconds(5.0)                    #   throttle error logs
        validate_inflation_parameters()                                #   run inflation sanity check

    def create_cspace(self, prob_map: np.ndarray) -> np.ndarray | None:  # Build cspace from prob map
        if prob_map is None:                                           #   guard against missing map
            return None
        return build_cspace(                                           #   delegate to builder
            probability_map=prob_map,
            occupied_threshold=self.params.th_occupied,
            resolution_meters=self.params.map_resolution_m,
            robot_radius_meters=self.params.robot_radius,
            safety_margin=self.params.safety_margin,
            morph_closing=self.params.cspace_morph_closing,
            morph_iterations=self.params.cspace_morph_iters,
            downsample=self.params.cspace_downsample,
            inflation_scale=self.params.cspace_inflation_scale
        )

    def should_stop_mapping(self) -> bool:                             # Decide if mapping should stop soon
        if self.bb.Get("stop_mapping"):                                #   explicit user stop
            return True
        if self.bb.Get("emergency_stop"):                              #   emergency stop condition
            return True
        if self.step_counter >= self.bb.Get("max_mapping_steps", 5000):#   step cap reached
            return True
        if hasattr(self, "start_time"):                                 #   time-based stop if long enough
            robot = self.bb.GetRobot()
            if robot:
                if (robot.getTime() - self.start_time) > 120.0 and self.step_counter >= self.min_scan_ticks:
                    return True
        if self.step_counter > 2000 and self.is_robot_stationary():     #   if stationary long enough
            return True
        return False                                                    # Otherwise continue

    def is_robot_stationary(self) -> bool:                              # Detect if robot hasn't moved much
        gps = self.bb.GetGps()                                          #   access GPS
        if not gps:
            return False
        cur = gps.getValues()                                           #   current position
        if not hasattr(self, "last_pos"):                               #   initialize state on first call
            self.last_pos = cur
            self.stationary_count = 0
            return False
        dx = float(cur[0] - self.last_pos[0])                           #   delta x
        dy = float(cur[1] - self.last_pos[1])                           #   delta y
        dist = float(np.hypot(dx, dy))                                  #   distance moved
        if dist < 0.05:                                                 #   small -> increment stationary counter
            self.stationary_count += 1
        else:
            self.stationary_count = 0                                   #   reset if moved
        self.last_pos = cur                                             #   update last position
        enough_scans = (self.step_counter >= self.min_scan_ticks)       #   ensure enough samples collected
        return (self.stationary_count > 200) and enough_scans           #   stationary for many ticks

    def assess_map_readiness(self, prob_map: np.ndarray, cspace: np.ndarray) -> bool:  # Check if map is usable
        if prob_map is None:                                            #   must have prob map
            return False
        if cspace is None:                                              #   and cspace
            return False
        total = int(prob_map.size)                                      #   total number of cells
        known_mask = (prob_map != 0.5) & np.isfinite(prob_map)          #   cells updated from prior (not 0.5)
        coverage = float(np.sum(known_mask)) / float(total)             #   fraction of known cells
        free_frac = float(np.sum(cspace > self.params.th_free_planner)) / float(total)  # fraction free in cspace
        return (coverage >= 0.01) and (free_frac >= 0.01)               #   minimal readiness criteria

    def execute(self) -> Status:                                        # Behavior tick
        self.step_counter += 1                                          #   increment iteration count
        robot = self.bb.GetRobot()                                      #   get robot handle
        if (not hasattr(self, "start_time")) and robot:                 #   initialize start time once
            self.start_time = robot.getTime()

        if self.should_stop_mapping():                                   #   check stopping conditions
            if self.step_counter >= self.min_scan_ticks:                 #     if enough scans, mark mission complete
                self.mission_complete = True
            return Status.RUNNING                                        #   keep node alive to finalize
        if robot:                                                        #   time-based throttle
            dt_ms = (robot.getTime() - self.last_mapping_time) * 1000.0
            if dt_ms < self.mapping_interval_ms:                         #   enforce mapping interval
                return Status.RUNNING
        if (self.step_counter % self.params.mapping_interval) != 0:      #   decimate by mapping_interval
            return Status.RUNNING

        try:
            gps = self.bb.GetGps()                                       #   fetch sensors
            compass = self.bb.GetCompass()
            lidar = self.bb.GetLidar()
            if not (gps and compass and lidar):                          #   if missing, wait
                map_logger.Warning("Sensors not available - waiting...")
                return Status.RUNNING
            xw, yw = gps.getValues()[:2]                                 #   world x,y
            cv = compass.getValues()                                     #   compass vector
            theta = float(np.arctan2(cv[0], cv[1]))                      #   heading angle
            prob_map = self.bb.GetProbMap()                              #   get map from blackboard
            if prob_map is None:                                         #   create if missing
                prob_map = np.zeros((200, 300), dtype=np.float32)
                self.bb.Set("prob_map", prob_map)
            try:
                ranges = np.asarray(lidar.getRangeImage(), dtype=float)  #   convert to numpy array
                if ranges.size == 0 or (not np.isfinite(ranges).any()):  #   check for valid numbers
                    map_logger.Warning("LiDAR empty/invalid - skipping")
                    return Status.RUNNING

                good = np.isfinite(ranges) & (ranges > 0.0)              #   valid range mask
                if int(good.sum()) < int(ranges.size * 0.1):             #   skip if too few valid readings
                    map_logger.Warning(f"LiDAR mostly invalid ({int(good.sum())}/{int(ranges.size)}) - skipping")
                    return Status.RUNNING
            except Exception as e:                                       #   handle sensor exceptions
                if robot and self.error_log_limiter.ShouldExecute(robot.getTime()):
                    map_logger.Error(f"LiDAR acquisition failed: {e}")
                return Status.RUNNING
            n = int(ranges.size)                                         #   number of lidar rays
            half_fov = np.radians(240.0) / 2.0                           #   half field of view in radians
            angles = np.linspace(+half_fov, -half_fov, n)                #   angles from left to right
            r0, c0 = WorldToGrid(xw, yw, prob_map.shape)                 #   robot cell in grid
            H, W = prob_map.shape                                        #   map dimensions
            self.update_prob_map_vectorized(prob_map, ranges, angles, theta, xw, yw, r0, c0, H, W)  #   update map
            if robot:
                self.last_mapping_time = robot.getTime()                 #   store last mapping time
            self.bb.SetProbMap(prob_map)                                 #   write map back to blackboard
            if (not is_cspace_frozen(self.bb)) and robot:                #   only if not frozen
                if self.cspace_build_limiter.ShouldExecute(robot.getTime()):  #   rate limited
                    cspace = self.create_cspace(prob_map)                #   build cspace
                    if cspace is not None:
                        self.bb.SetCspace(cspace)                        #   publish cspace
            if self.mission_complete:                                    #   after completion decision
                cs = self.bb.GetCspace()                                 #     read cspace
                if cs is not None:
                    frac = (cs > self.params.th_free_planner).sum() / float(cs.size)  # free fraction
                    if frac >= 0.01:
                        self.bb.SetMapReady(True)                        #   mark map ready
        except Exception as e:                                           # Fail-safe for unexpected errors
            if robot and self.error_log_limiter.ShouldExecute(robot.getTime()):
                map_logger.Error(f"Exception: {e}")                      #   log throttled error
            return Status.FAILURE                                        #   indicate failure this tick
        return Status.RUNNING                                            # Continue mapping

    def update_prob_map_vectorized(                                     # Vectorized update of probabilistic map
        self,
        prob_map,                                                       #   map to update
        lidar_ranges,                                                   #   raw lidar ranges
        angles,                                                         #   per-ray angles in robot frame
        theta,                                                          #   robot heading in world frame
        xw,                                                             #   robot x in world
        yw,                                                             #   robot y in world
        row0,                                                           #   robot row in grid
        col0,                                                           #   robot col in grid
        H,                                                              #   map height
        W                                                               #   map width
    ):
        valid = np.isfinite(lidar_ranges) & (lidar_ranges > 0.1) & (lidar_ranges < 8.0)  # keep reasonable hits
        if not valid.any():                                             # nothing to process
            return
        dec = int(self.current_decimation)                              # decimation factor
        if dec > 1:                                                     # stride mask for decimation
            mask = np.zeros_like(valid, dtype=bool)
            mask[::dec] = True
            valid = valid & mask
        near = (lidar_ranges < 4.0)                                     # emphasize close obstacles
        if near.any():
            od = max(1, dec // 3)                                       #   oversample factor for near hits
            add_mask = np.zeros_like(valid, dtype=bool)
            idx = np.nonzero(near & np.isfinite(lidar_ranges))[0]       #   indices of good near hits
            add_mask[idx[::od]] = True                                  #   add some of them back
            valid = valid | add_mask                                    #   union with base mask
        if not valid.any():                                             # after masking, may be empty
            return
        rng = lidar_ranges[valid]                                       # filtered ranges
        ang = angles[valid]                                             # filtered angles
        ct = float(np.cos(theta))                                       # cos heading
        st = float(np.sin(theta))                                       # sin heading
        ca = np.cos(theta + ang)                                        # cos of absolute ray angle
        sa = np.sin(theta + ang)                                        # sin of absolute ray angle
        xh = xw + rng * ca + 0.202 * ct                                 # hit x in world
        yh = yw + rng * sa + 0.202 * st                                 # hit y in world
        rows1 = np.round(40.0 * (xh + 2.25)).astype(int)                # map world x to row
        cols1 = np.round(-52.9 * (yh - 1.6633)).astype(int)             # map world y to col
        rows1 = np.clip(rows1, 0, H - 1)                                # clamp to grid
        cols1 = np.clip(cols1, 0, W - 1)                                # clamp to grid
        batch = min(400, len(rows1))                                    # choose batch size
        for start in range(0, len(rows1), batch):                       # iterate in chunks
            re = rows1[start:start + batch]                             #   end rows chunk
            ce = cols1[start:start + batch]                             #   end cols chunk
            free_r = []                                                 #   rows of free cells along rays
            free_c = []                                                 #   cols of free cells along rays
            for r1, c1 in zip(re, ce):                                  #   for each endpoint
                pts = BresenhamLine(row0, col0, r1, c1)                 #     grid line from origin to hit
                if not pts:
                    continue
                if len(pts) > 1:
                    pts = pts[:-1]                                      #     exclude the hit cell itself
                stride = 6                                              #     sample every Nth along ray
                if len(pts) > stride:
                    pts = [pts[0], *pts[stride:-1:stride], pts[-1]]     #     keep endpoints and sparse interior
                for rr, cc in pts:                                      #     append valid free cells
                    if (0 <= rr < H) and (0 <= cc < W):
                        free_r.append(rr)
                        free_c.append(cc)
            if free_r:                                                  # If any free cells
                fr = np.asarray(free_r, dtype=int)
                fc = np.asarray(free_c, dtype=int)
                prob_map[fr, fc] = np.maximum(prob_map[fr, fc] - 0.004, 0.0)  # Lower occupancy probability
            inb = (re >= 0) & (re < H) & (ce >= 0) & (ce < W)          # Endpoints within bounds
            if not inb.any():
                continue
            hr = re[inb]                                                # Hit rows
            hc = ce[inb]                                                # Hit cols
            dr = np.arange(-1, 2)                                       # Offsets for 3x3 patch
            dc = np.arange(-1, 2)
            drg, dcg = np.meshgrid(dr, dc, indexing='ij')               # Grid of offsets
            r_idx = np.clip(hr[:, None] + drg.flatten(), 0, H - 1).flatten()  # Neighbor rows
            c_idx = np.clip(hc[:, None] + dcg.flatten(), 0, W - 1).flatten()  # Neighbor cols
            prob_map[r_idx, c_idx] = np.minimum(prob_map[r_idx, c_idx] + 0.020, 1.0)  # Raise occupancy prob

    def reset(self):                                                    # Reset node state
        super().reset()                                                 # Base reset
        self.step_counter = 0                                           # Clear counters
        self.mission_complete = False                                   # Clear completion flag
        if hasattr(self, "completion_reported"):                        # Remove optional field if present
            delattr(self, "completion_reported")
        if not is_cspace_frozen(self.bb):                               # Only clear maps if not frozen
            self.bb.SetProbMap(None)
            self.bb.SetCspace(None)
            self.bb.Set("obstacle_field", None)

    def terminate(self):                                                # Called when node is ended
        self.mission_complete = True                                    # Mark mission complete
        cs = self.bb.GetCspace()                                        # Try to freeze existing cspace
        if cs is not None:                                              # If cspace exists
            free_frac = (cs > self.params.th_free_planner).sum() / float(cs.size)  # Measure free fraction
            if free_frac >= 0.01:                                       # Minimal quality threshold
                self.bb.Set("cspace_frozen", True)                      # Freeze cspace updates
                self.bb.SetMapReady(True)                               # Mark ready
                return
        pm = self.bb.GetProbMap()                                       # Get latest probability map
        if pm is None:                                                  # If missing, cannot build
            map_logger.Warning("terminate(): no prob_map to build c-space from")
            return
        cspace = self.create_cspace(pm)                                 # Attempt to build cspace once
        if cspace is None:                                             
            return
        if not self.bb.Get("map_ready", False):                         # If map not yet ready
            self.bb.Set("cspace_live", cspace)                          # Stage live cspace
        else:
            self.bb.Set("cspace", self.bb.Get("cspace_live", cspace))   # Commit staged cspace
            self.bb.Set("map_ready", True)                              # Mark ready
        self.bb.Set("cspace_frozen", True)                              # Freeze cspace