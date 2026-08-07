from __future__ import annotations
import numpy as np                                                          # Arrays and math
from scipy import ndimage as ndi                                            # Image processing 
from core import (                                                          # Project specific utilities and logging 
    BehaviorNode, Status, blackboard, GetFromBlackboard, EveryNCalls, EveryNSeconds,
    WorldToGrid, BresenhamLine, map_logger, main_logger,
    ResolveMapPath, EnsureParentDirectories,
    BBKey, MappingParams, TH_FREE_PLANNER, CalculateFreeSpacePercentage
)
from os.path import exists                                                  # File existence check  


###############################################################################
# ------------------------- C-space builder -----------------------------------
###############################################################################
# Converts a probability map into a binary configuration space.
def build_cspace(                                                           # Build cspace from prob map; main generator
    probability_map: np.ndarray,                                            # Occupancy probabilities per cell  
    occupied_threshold: float,                                              # Threshold for core obstacle mask
    resolution_meters: float,                                               # Meters per pixel in the map
    robot_radius_meters: float,                                             # Robot radius in meters 
    safety_margin: float,                                                   # Extra margin around robot  
    morph_closing: int,                                                     # Structuring element size for closing 
    morph_iterations: int,                                                  # Number of closing iterations 
    downsample: int = 1,                                             
    inflation_scale: float = 1.0                                            # Scale factor on inflation radius  
) -> np.ndarray:
    pm = np.asarray(probability_map, dtype=np.float32)                      # Ensure float array  
    if downsample > 1:                                                      
        hh = pm.shape[0] // downsample                                      # Downsampled height  
        ww = pm.shape[1] // downsample                                      # Downsampled width   
        cut = pm[:hh * downsample, :ww * downsample]                        # Crop to multiple of factor; clean edges
        pm_ds = cut.reshape(hh, downsample, ww, downsample).mean(axis=(1, 3))  
    else:
        pm_ds = pm                                                          
    core = (pm_ds >= occupied_threshold).astype(np.uint8)                   # Core obstacle 
    labels, num = ndi.label(core)                                           # Connected components in obstaclecore
    if num:                                                                 # If any components exist; non-zero
        res_px = max(1, downsample)                                         # Pixels per original pixel in ds space; scale
        m_per_px = max(1e-6, resolution_meters * res_px)                    # Meters per downsampled pixel; avoid 0
        min_side_m = 0.05                                                   # Ignore tiny specks; meters
        solidity_min = 0.05                                                 # Ignore very sparse boxes; min fill
        min_area_px = int((min_side_m / m_per_px) ** 2)                     # Area threshold in pixels 
        objects = ndi.find_objects(labels)                                  # Bounding slices 
        for i, sl in enumerate(objects, start=1):                           
            if not sl:                                                      # Skip empty slices
                continue
            h = sl[0].stop - sl[0].start                                    # Height of bbox; px
            w = sl[1].stop - sl[1].start                                    # Width of bbox;px
            area = h * w                                                    # Box area; px^2
            if area < min_area_px:                                          # Skip small components; noise
                continue
            comp = (labels[sl] == i)                                        # Component mask inside bbox
            solidity = float(comp.sum()) / float(area)                      # Occupancy fraction of bbox 
            if solidity >= solidity_min:                                    # If solid enough, pad and fill 
                pad_h = max(1, h // 3) + max(1, min(h, w) // 6)             # Vertical padding; px
                pad_w = max(1, w // 3) + max(1, min(h, w) // 6)             # Horizontal padding; px
                r0 = max(0, sl[0].start - pad_h)                            # Clamp start row 
                r1 = min(core.shape[0], sl[0].stop + pad_h)                 # Clamp end row   
                c0 = max(0, sl[1].start - pad_w)                            # Clamp start col 
                c1 = min(core.shape[1], sl[1].stop + pad_w)                 # Clamp end col   
                core[r0:r1, c0:c1] = 1                                      # Mark padded region occupied; fill
    if morph_closing > 0 and morph_iterations > 0:                          
        se = np.ones((morph_closing, morph_closing), dtype=np.uint8)        # Square structuring element 
        core = ndi.binary_closing(core, structure=se, iterations=morph_iterations)  # Close gaps; smooth
    free_mask = (~core.astype(bool))                                        # Free = not occupied
    dist_px = ndi.distance_transform_edt(free_mask).astype(np.float32)      # Euclidean distance to nearest obstacle 
    inflate_px = (robot_radius_meters + safety_margin) / max(resolution_meters, 1e-6)  # Convert meters to pixels; scale
    inflate_px = float(max(0.0, inflate_px * inflation_scale))              # Scale and clamp non-negative 
    safety_buffer = inflate_px * 0.18                                       # C-space generation with reduced safety margins for thinner c-space                                    
    safe_distance = inflate_px + safety_buffer                              # Minimal inflation to prevent collisions while making c-space thinner
    cspace_ds = np.where(dist_px >= safe_distance, 1.0, 0.0).astype(np.float32)  
    uncertain_threshold = 0.4                                               # Mark areas near uncertain probability values as occupied
    uncertain_mask = (pm >= uncertain_threshold) & (pm <= (1.0 - uncertain_threshold))  # This helps prevent collisions in areas where the map is unclear
    cspace_ds = np.where(uncertain_mask, 0.0, cspace_ds)                    # Mark uncertain areas as occupied; more conservative
    if downsample > 1:                                                      # If we reduced resolution earlier 
        up = ndi.zoom(cspace_ds, zoom=downsample, order=0)                  # nearest-neighbor
        return up[:pm.shape[0], :pm.shape[1]]                               # Crop to original shape
    return cspace_ds                                                        # Return cspace at current resolution


###############################################################################
# ------------------------- Helpers -------------------------------------------
###############################################################################
# Small utilities to gate updates and pre check parameters.
def is_cspace_frozen(bb) -> bool:                                           # Check if C-space updates are frozen 
    # --- returns True when further c-space writes should be skipped ---
    val = bb.Get("cspace_frozen", False)                                    # Read flag from blackboard; default False
    return bool(val)                                                        # Cast to bool and return; normalize

def validate_inflation_parameters():                                        # Heuristic checks
    # --- computes nominal pixel radius for debug---
    s = max(52.9, 40.0)                                                     # Pixels per meter for y/x scales
    rpx = 0.16 * s * 0.55                                                   # Rough pixel radius estimate
    r_pix = int(max(1, round(rpx)))                                         # Integer pixel radius
    bubble = int(r_pix + 5)                                                 # Halo/buffer radius; px
    core_m = r_pix / s                                                      # Core radius in meters
    margin = core_m - 0.16                                                  # Margin beyond robot radius


###############################################################################
# ------------------------- BT: Lidar Mapping ---------------------------------
################################################################################
# Behavior node that fuses LiDAR into a prob-map and derives c-space.
# Runs periodically, throttled to avoid overwork.
class LidarMappingBT(BehaviorNode):                                         # Behavior tree node for building map and c-space
    # --- manages LiDAR processing, map updates, and c-space builds ---
    def __init__(self, params, bb=None):                                    # Initialize mapping behavior
        super().__init__("LidarMappingBT")                                  # Set node name; label
        self.params = params                                                # Store parameters object
        self.bb = bb or blackboard                                          # Use provided or global blackboard
        self.step_counter = 0                                               # Iteration counter
        self.mission_complete = False                                       # Whether mapping should complete; flag
        self.min_scan_ticks = 25                                            # Minimal ticks before completing 
        self.robot_radius = params.robot_radius                             # Cache robot radius; meters
        self.mapping_interval_ms = 100                                      # Target interval between updates; ms
        self.last_mapping_time = 0.0                                        # Last update time; seconds
        self.current_decimation = 4                                       
        self.cspace_log_limiter = EveryNCalls(20)                           # Log cspace less frequently
        self.cspace_build_limiter = EveryNSeconds(0.5)                      # Build cspace at most x2 per sec; rate limiter
        self.error_log_limiter = EveryNSeconds(5.0)                         # Throttle error logs; avoid spam
        validate_inflation_parameters()                                     # Run inflation check 

    def create_cspace(self, prob_map: np.ndarray) -> np.ndarray | None:     # Build cspace from prob map; helper
        # --- wrapper to call builder with current params; returns ndarray or None ---
        if prob_map is None:                                                # Guard against missing map
            return None
        return build_cspace(                                                # Cspace builder
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

    def should_stop_mapping(self) -> bool:                                  # Decide if mapping should stop soon
        if hasattr(self, "start_time"):                                     # Time based stop if mapping has been running long enough
            robot = self.bb.GetRobot()
            if robot:
                elapsed_time = robot.getTime() - self.start_time            # Elapsed seconds
                if elapsed_time > 120.0 and self.step_counter >= self.min_scan_ticks:
                    return True                                             # Time and min scans
        if self.step_counter > 2000 and self.is_robot_stationary():         # Stop if robot has been stationary for too long
            return True                                                     # Stationary too long
        return False                                                        # Otherwise continue

    def is_robot_stationary(self) -> bool:                                  # Detect if robot hasn't moved much; idle
        # --- tracks distance between ticks; counts "still" frames ---
        gps = self.bb.GetGps()                                              # Access GPS; sensor
        if not gps:
            return False                                                    # No sensor; assume moving
        cur = gps.getValues()                                               # Current position; xyz
        if not hasattr(self, "last_pos"):                                   # Initialize state on first call
            self.last_pos = cur
            self.stationary_count = 0
            return False
        dx = float(cur[0] - self.last_pos[0])                               # Delta x; meters
        dy = float(cur[1] - self.last_pos[1])                               # Delta y; meters
        dist = float(np.hypot(dx, dy))                                      # Distance moved; euclid
        if dist < 0.05:                                                     # Increment stationary counter; 5cm
            self.stationary_count += 1
        else:
            self.stationary_count = 0                                       # Reset if moved
        self.last_pos = cur                                                 # Update last position
        enough_scans = (self.step_counter >= self.min_scan_ticks)           # Ensure enough samples collected
        return (self.stationary_count > 200) and enough_scans               # Stationary for many ticks

    def assess_map_readiness(self, prob_map: np.ndarray, cspace: np.ndarray) -> bool:  # Check if map is usable
        # --- minimal coverage + free fraction check ---
        if prob_map is None:                                                # Must have prob map; required
            return False
        if cspace is None:                                                  # And cspace; required
            return False
        total = int(prob_map.size)                                          # Total number of cells  
        known_mask = (prob_map != 0.5) & np.isfinite(prob_map)              # Cells updated from prior
        coverage = float(np.sum(known_mask)) / float(total)                 # Fraction of known cells
        free_frac = float(np.sum(cspace > self.params.th_free_planner)) / float(total)  # Fraction free in cspace 
        return (coverage >= 0.01) and (free_frac >= 0.005)                  # Minimal readiness 

    def execute(self) -> Status:                                            # Behavior tick; main loop
        self.step_counter += 1                                              # Increment iteration count; tick++
        robot = self.bb.GetRobot()                                          # Get robot handle; API
        if (not hasattr(self, "start_time")) and robot:                     # Initialize start time once; first tick
            self.start_time = robot.getTime()                              
        if self.should_stop_mapping():                                      # Check stopping conditions
            if self.step_counter >= self.min_scan_ticks:                    # If enough scans, mark mission complete 
                if not self.mission_complete:
                    map_logger.Info("Mapping completed - creating final c-space")  
                self.mission_complete = True
            return Status.RUNNING                                           # Keep node alive to finalize
        if robot:                                                           # Time based throttle
            dt_ms = (robot.getTime() - self.last_mapping_time) * 1000.0
            if dt_ms < self.mapping_interval_ms:                            # Enforce mapping interval
                return Status.RUNNING
        if (self.step_counter % self.params.mapping_interval) != 0:         
            return Status.RUNNING
        try:
            gps = self.bb.GetGps()                                          # Fetch sensors; GPS
            compass = self.bb.GetCompass()                                  # Compass
            lidar = self.bb.GetLidar()                                      # Lidar
            if not (gps and compass and lidar):                             # If missing, wait 
                return Status.RUNNING
            xw, yw = gps.getValues()[:2]                                    # World x,y; meters
            cv = compass.getValues()                                        # Compass vector; x,z
            theta = float(np.arctan2(cv[0], cv[1]))                         # Heading angle; radians
            prob_map = self.bb.GetProbMap()                                 # Get map from blackboard
            if prob_map is None:                                            # Create if missing 
                prob_map = np.zeros((200, 300), dtype=np.float32)
                self.bb.Set("prob_map", prob_map)
            try:
                ranges = np.asarray(lidar.getRangeImage(), dtype=float)     # Convert to numpy array; 1D
                if ranges.size == 0 or (not np.isfinite(ranges).any()):     # Check for valid numbers 
                    return Status.RUNNING

                good = np.isfinite(ranges) & (ranges > 0.0)                 # Valid range mask  
                if int(good.sum()) < int(ranges.size * 0.1):                # Skip if too few valid readings
                    return Status.RUNNING
            except Exception as e:                                          # Handle sensor exceptions
                if robot and self.error_log_limiter.ShouldExecute(robot.getTime()):
                    map_logger.Error(f"LiDAR acquisition failed: {e}")      # Rate limited log 
                return Status.RUNNING
            n = int(ranges.size)                                            # Number of lidar rays
            half_fov = np.radians(240.0) / 2.0                              
            angles = np.linspace(+half_fov, -half_fov, n)                   # Angles from left to right  # vector
            r0, c0 = WorldToGrid(xw, yw, prob_map.shape)                    # Robot cell in grid  # row,col
            H, W = prob_map.shape                                           
            self.update_prob_map_vectorized(prob_map, ranges, angles, theta, xw, yw, r0, c0, H, W)  # Update map
            if robot:
                self.last_mapping_time = robot.getTime()                    # Store last mapping time
            self.bb.SetProbMap(prob_map)                                    # Write map back to blackboard 
            if (not is_cspace_frozen(self.bb)) and robot:                   # Only if not frozen
                if self.cspace_build_limiter.ShouldExecute(robot.getTime()):  # Rate limited; 2 Hz
                    cspace = self.create_cspace(prob_map)                   # Build cspace
                    if cspace is not None:
                        self.bb.SetCspace(cspace)                           # Publish cspace
            if self.mission_complete:                                       
                cs = self.bb.GetCspace()                                    # Read cspace
                if cs is not None:
                    frac = (cs > self.params.th_free_planner).sum() / float(cs.size)  # Free fraction; ratio
                    if frac >= 0.01:
                        self.bb.SetMapReady(True)                           # Mark map ready 
        except Exception as e:                                              # Fail safe for unexpected errors
            if robot and self.error_log_limiter.ShouldExecute(robot.getTime()):
                map_logger.Error(f"Exception: {e}")                         # Log throttled error; message
            return Status.FAILURE                                           # Indicate failure this tick
        return Status.RUNNING                                               # Continue mapping

    def update_prob_map_vectorized(                                         # Vectorized update of probabilistic map
        self,
        prob_map,                                                           # Map to update 
        lidar_ranges,                                                       # Raw lidar ranges; 1D
        angles,                                                             # Per-ray angles in robot frame
        theta,                                                              # Robot heading in world frame; rad
        xw,                                                                 # Robot x in world; m
        yw,                                                                 # Robot y in world; m
        row0,                                                               # Robot row in grid  
        col0,                                                               # Robot col in grid 
        H,                                                                  # Map height; rows
        W                                                                   # Map width; cols
    ):
        valid = np.isfinite(lidar_ranges) & (lidar_ranges > 0.1) & (lidar_ranges < 8.0)  # Keep reasonable hits 
        if not valid.any():                                                 # Nothing to process; skip
            return
        dec = int(self.current_decimation)                                  # Decimation factor
        if dec > 1:                                                         
            mask = np.zeros_like(valid, dtype=bool)
            mask[::dec] = True
            valid = valid & mask
        near = (lidar_ranges < 4.0)                                         # Emphasize close obstacles
        if near.any():
            od = max(1, dec // 3)                                           # Oversample factor for near hits
            add_mask = np.zeros_like(valid, dtype=bool)
            idx = np.nonzero(near & np.isfinite(lidar_ranges))[0]           # Indices of good near hits
            add_mask[idx[::od]] = True                                      # Add some of them back 
            valid = valid | add_mask                                        # Union with base mask; combine
        if not valid.any():                                                 # After masking, may be empty 
            return
        rng = lidar_ranges[valid]                                           # Filtered ranges
        ang = angles[valid]                                                 # Filtered angles
        ct = float(np.cos(theta))                                           # Cos heading; scalar
        st = float(np.sin(theta))                                           # Sin heading; scalar
        ca = np.cos(theta + ang)                                            # Cos of absolute ray angle; vector
        sa = np.sin(theta + ang)                                            # Sin of absolute ray angle; vector
        xh = xw + rng * ca + 0.202 * ct                                     # Hit x in world; add lidar offset
        yh = yw + rng * sa + 0.202 * st                                     # Hit y in world; add lidar offset
        rows1 = np.round(40.0 * (xh + 2.25)).astype(int)                    # Map world x to row  
        cols1 = np.round(-52.9 * (yh - 1.6633)).astype(int)                 # Map world y to col 
        rows1 = np.clip(rows1, 0, H - 1)                                    # Clamp to grid; bounds
        cols1 = np.clip(cols1, 0, W - 1)                                    # Clamp to grid; bounds
        batch = min(400, len(rows1))                                        # Choose batch size; throttle
        for start in range(0, len(rows1), batch):                           # Iterate in chunks 
            re = rows1[start:start + batch]                                 # End rows chunk 
            ce = cols1[start:start + batch]                                 # End cols chunk
            free_r = []                                                     # Rows of free cells along rays
            free_c = []                                                     # Cols of free cells along rays 
            for r1, c1 in zip(re, ce):                                      # For each endpoint
                pts = BresenhamLine(row0, col0, r1, c1)                     # Grid line from origin to hit 
                if not pts:
                    continue
                if len(pts) > 1:
                    pts = pts[:-1]                                          # Exclude the hit cell itself
                stride = 6                                                  # Sample every Nth along ray
                if len(pts) > stride:
                    pts = [pts[0], *pts[stride:-1:stride], pts[-1]]         # Keep endpoints 
                for rr, cc in pts:                                          # Append valid free cells 
                    if (0 <= rr < H) and (0 <= cc < W):
                        free_r.append(rr)
                        free_c.append(cc)
            if free_r:                                                      # If any free cells; update free
                fr = np.asarray(free_r, dtype=int)
                fc = np.asarray(free_c, dtype=int)
                prob_map[fr, fc] = np.maximum(prob_map[fr, fc] - 0.004, 0.0)  # Lower occupancy probability
            inb = (re >= 0) & (re < H) & (ce >= 0) & (ce < W)               # Endpoints within bounds
            if not inb.any():
                continue
            hr = re[inb]                                                    # Hit rows  
            hc = ce[inb]                                                    # Hit cols 
            dr = np.arange(-1, 2)                                           # Offsets for 3x3 patch 
            dc = np.arange(-1, 2)
            drg, dcg = np.meshgrid(dr, dc, indexing='ij')                   # Grid of offsets; 3x3
            r_idx = np.clip(hr[:, None] + drg.flatten(), 0, H - 1).flatten()  # Neighbor rows 
            c_idx = np.clip(hc[:, None] + dcg.flatten(), 0, W - 1).flatten()  # Neighbor cols 
            prob_map[r_idx, c_idx] = np.minimum(prob_map[r_idx, c_idx] + 0.020, 1.0)  # Raise occupancy prob

    def reset(self):                                                        # Reset node state
        # --- clears progress counters and map stuff ---
        super().reset()                                                     # Base reset
        self.step_counter = 0                                               # Clear counters
        self.mission_complete = False                                       # Clear completion flag
        if hasattr(self, "completion_reported"):                            # Remove optional field if present; cleanup
            delattr(self, "completion_reported")
        if not is_cspace_frozen(self.bb):                                   # Only clear maps if not frozen; preserve if frozen
            self.bb.SetProbMap(None)
            self.bb.SetCspace(None)
            self.bb.Set("obstacle_field", None)

    def terminate(self):                                                    # Called when node is ended 
        # --- on termination, try to freeze a usable c-space snapshot ---
        self.mission_complete = True                                        # Mark mission complete
        cs = self.bb.GetCspace()                                            # Try to freeze existing cspace
        if cs is not None:                                                  # If cspace exists; check
            free_frac = (cs > self.params.th_free_planner).sum() / float(cs.size)  # Measure free fraction; ratio
            if free_frac >= 0.005:                                          # Minimal quality threshold
                self.bb.Set("cspace_frozen", True)                          # Freeze cspace updates; lock
                self.bb.SetMapReady(True)                                   # Mark ready
                return
        pm = self.bb.GetProbMap()                                           # Get latest probability map; fallback
        if pm is None:                                                      # If missing, cannot build; abort
            map_logger.Warning("No prob map to build c-space from.")
            return
        cspace = self.create_cspace(pm)                                     # Attempt to build cspace once
        if cspace is None:
            return
        if not self.bb.Get("map_ready", False):                             # If map not yet ready
            self.bb.Set("cspace_live", cspace)                              # Stage live cspace 
        else:
            self.bb.Set("cspace", self.bb.Get("cspace_live", cspace))       
            self.bb.Set("map_ready", True)                                  # Mark ready
        self.bb.Set("cspace_frozen", True)                                  # Freeze cspace


###############################################################################
# ------------------------- Behavior Tree Nodes ------------------------------
###############################################################################
# Mapping behavior tree nodes that handle map creation and validation.
# Ensure a ready/usable c-space is available for planning.
class WaitForMapReady(BehaviorNode):                                        # Waits until c-space exists and passes a minimal free space check
    def __init__(self):
        super().__init__("WaitForMapReady")                            
        self._start_time = None                                             # Robot time when we began waiting  
        self._warned = [False] * 3                                          # Track which warnings were already sent

    def execute(self):                                                      # Log time-based warnings; succeed when map ready
        robot = GetFromBlackboard("robot")                                  # Get robot interface; API
        if robot and self._start_time is None:                              # First tick with a robot present
            self._start_time = robot.getTime()                              # Record the start time
        if GetFromBlackboard("map_saved"):                                  
            return Status.SUCCESS
        cspace = GetFromBlackboard("cspace")                                # Check if a c-space exists yet
        if cspace is None:                                                  # No c-space available; wait
            if robot and self._start_time is not None:          
                elapsed = robot.getTime() - self._start_time                # Time since start of waiting; s
                msgs = [                                                    
                    (120, "Still no map after 120s."),
                    (60, "Map build is slow 60s."),
                    (30, "Waiting for C-space 30s"),
                ]
                for idx, (t, msg) in enumerate(msgs[::-1]):                 # Iterate backwards
                    real_idx = 2 - idx                                      # Map index
                    if elapsed > t and not self._warned[real_idx]:      
                        main_logger.Warning(msg)                            # Give one time warning 
                        self._warned[real_idx] = True                       # Mark that we warned at this level
            return Status.RUNNING                                           # Keep waiting for map  
        return Status.SUCCESS if CalculateFreeSpacePercentage(cspace) >= 0.005 else Status.RUNNING  # Ready when some free space

    def reset(self):                                                        # Clears timers and warning state so we can wait again fresh
        super().reset()                                                     # Reset base node state
        self._start_time = None                                             # Clear start timer
        self._warned = [False] * 3                                          # Reset warning flags; clear


class MapExistsOrReady(BehaviorNode):
    def __init__(self, path="cspace.npy"):                                  # Passes if map_ready flag set or on-disk file exists; else fails
        super().__init__("MapExistsOrReady")                                # Name the node
        self.path = str(ResolveMapPath(path))                               # Path to map file

    def execute(self):
        if GetFromBlackboard("map_ready", False):                           # Check if map is ready or file exists                      
            return Status.SUCCESS                                           # Ready flag
        if exists(self.path):                                               # File present; on-disk
            return Status.SUCCESS
        main_logger.Info("No existing map found - will create new map")     # Info; plan to build
        return Status.FAILURE                                               # Signal need to build; fail


class LoadMap(BehaviorNode):
    def __init__(self, path="cspace.npy"):                                  # Loads c-space from disk if not already on blackboard
        super().__init__("LoadMap")                                         # Name the node 
        self.path = str(ResolveMapPath(path))                               # Path to map to load  

    def execute(self):                                                      # Check if c-space already exists in blackboard
        cspace = GetFromBlackboard("cspace")                                # C-space already exists, just mark as ready
        if cspace is not None:
            main_logger.Info("Map found in blackboard - using existing c-space") 
            blackboard.Set(BBKey.CSPACE_FROZEN, True)                       # Freeze c-space to avoid accidental edits  
            blackboard.SetMapReady(True)                                    # Mark map as ready for use  
            return Status.SUCCESS
        try:                                                                # Try to load from file
            c = np.clip(np.load(self.path).astype(np.float32), 0.0, 1.0)    # Load npy 
            main_logger.Info("Map found and loaded")                        # Success  
            blackboard.SetCspace(c)                                         # Publish cspace to blackboard  
            blackboard.Set(BBKey.CSPACE_FROZEN, True)                       # Freeze cspace to avoid accidental edits 
            blackboard.SetMapReady(True)                                    # Mark map as ready for use
            return Status.SUCCESS                                     
        except Exception as e:
            main_logger.Error(f"LoadMap failed {e}")                        # Log load error
            return Status.FAILURE                                           # Fail the node; no file


class EnsureCspaceNow(BehaviorNode):                                        # Ensures c-space exists now; builds from prob map if needed
    def __init__(self, blackboard_instance=None):
        super().__init__("EnsureCspaceNow")                                 # Name the node
        self.blackboard = blackboard_instance or blackboard                 # Use provided bb or default global

    def execute(self):
        c = self.blackboard.GetCspace()                                     # Already have c-space?; check
        if c is not None:
            return Status.SUCCESS                                           # Nothing to do
        p = self.blackboard.GetProbMap()                                    # Try to get probability map
        if p is None:
            return Status.RUNNING                                           # Wait for prob map
        try:
            mapper = LidarMappingBT(params=MappingParams(), bb=self.blackboard)  # Temp mapper; default params
            c = mapper.create_cspace(p)                                     # Build c-space from probability map
            if c is not None:
                self.blackboard.SetCspace(c)                                # Store c-space
                return Status.SUCCESS
            return Status.FAILURE                                           # Mapper returned None
        except Exception as e:
            main_logger.Error(f"c-space build crashed: {e}")                # Mapping pipeline crashed; error
            return Status.FAILURE


class SaveMap(BehaviorNode):                                                # Saves a usable c-space to disk and marks state as ready
    def __init__(self, path="cspace.npy", threshold=None):
        super().__init__("SaveMap")                                         # Name the node 
        self.path = str(ResolveMapPath(path))                               # Path
        self.threshold = 0.30 if threshold is None else threshold             
        self.done = False                                                   # Guard against duplicate saves

    def PrepareMapForSaving(self) -> np.ndarray | None:                     # Prefer c-space; otherwise build from prob map; final fallback
        c = GetFromBlackboard("cspace")                                     # Prefer existing c-space
        if c is None:                                                       # If none, try to build from prob map
            p = GetFromBlackboard("prob_map")                               # Fetch probability map
            if p is None:                                                   # If nothing available; abort
                main_logger.Error("no map.")                                # Log and abort; error
                return None
            mapper = LidarMappingBT(MappingParams())                        # Create mapping component
            c = mapper.create_cspace(p)                                     # Attempt building c-space
            if c is None:                                                   # If that fails 
                c = (p <= self.threshold).astype(np.float32)                # Threshold as last resort  
        return np.clip(c.astype(np.float32), 0.0, 1.0)                      

    def ShouldSaveMap(self, cspace: np.ndarray) -> bool:
        return CalculateFreeSpacePercentage(cspace) * 100.0 >= 0.1          

    def SaveMapToFile(self, cspace: np.ndarray) -> bool:
        try:
            EnsureParentDirectories(self.path)                              # Make sure folders exist; mkdir -p
            np.save(self.path, cspace)                                      # Save as .npy array; write
            return True
        except Exception as e:
            main_logger.Error(f"Write failed to save map: {e}")             
            return False

    def UpdateMapState(self, cspace: np.ndarray):
        for k, v in (("cspace", cspace), ("map_ready", True), ("map_saved", True), ("cspace_frozen", True)):
            blackboard.Set(k, v)                                            # Update multiple blackboard keys
        self.done = True                                                    # Mark as finished

    def execute(self):                                                      # Returns SUCCESS only after a write
        if self.done:                                                       # If already saved once
            return Status.SUCCESS
        c = self.PrepareMapForSaving()                                      # Obtain c-space to save
        if c is None:                                                       # Abort if none; fail
            return Status.FAILURE
        if not self.ShouldSaveMap(c):                                       # Check quality threshold
            main_logger.Error("C-space rejected - not saving.")             # Explain why not saved
            return Status.FAILURE
        try:
            from os import remove                                           # Import for file removal 
            if exists(self.path):                                           # If file already exists
                remove(self.path)                                           # Remove it first; replace
        except Exception:                                                   # Ignore deletion errors
            pass
        if not self.SaveMapToFile(c):                                       # Write file
            return Status.FAILURE
        self.UpdateMapState(c)                                              # Update bb 
        return Status.SUCCESS                                               # Success


class ClearCspace(BehaviorNode):                                            # Clears existing cspace to force regeneration on next cycle 
    def __init__(self):
        super().__init__("ClearCspace")                                     # Name the node

    def execute(self):                                                      # Clear any existing c-space from blackboard to force regeneration
        blackboard.Set("cspace", None)                                      # Drop map; reset
        main_logger.Info("Cleared existing c-space - will generate fresh c-space")  # Log info
        return Status.SUCCESS                                           


class SaveCspaceImage(BehaviorNode):                                        #Writes a grayscale  visualization of the current c-space
    def __init__(self):
        super().__init__("SaveCspaceImage")                                 # Name the node

    def execute(self):
        cspace = blackboard.Get("cspace")                                   # Fetch cspace
        if cspace is None:
            return Status.FAILURE                                           # Nothing to save; fail
        try:
            from PIL import Image                                           
            import os                                                       
            img_array = ((1.0 - cspace) * 255).astype(np.uint8)             # Convert c-space to image to match display
            img = Image.fromarray(img_array, mode='L')                      # Grayscale image; PIL image
            os.makedirs("maps", exist_ok=True)                              # Create maps directory if it doesn't exist
            img_path = "maps/cspace.png"                                    # Save in maps folder, overwriting previous image
            img.save(img_path)                                              # Write PNG; save
            return Status.SUCCESS                                           # Done
        except ImportError:
            main_logger.Error("space image not saved")                      
            return Status.FAILURE
        except Exception as e:
            main_logger.Error(f"Failed to save c-space image: {e}")         
            return Status.FAILURE


class ValidateLoadedMap(BehaviorNode):                                      # Quick check on loaded c-space using free % 
    def __init__(self):
        super().__init__("ValidateLoadedMap")                               # Name the node 

    def execute(self):
        c = GetFromBlackboard("cspace")                                     # Fetch loaded cspace
        if c is None:                                                       # If map is missing then invalid; no map
            return Status.FAILURE
        free_pct = 100.0 * CalculateFreeSpacePercentage(c)                  # Percentage of free cells 
        if free_pct < 0.05:                                                 # Very low free might be corrupt
            main_logger.Error(f"Loaded map looks wrong: free={free_pct:.2f}% (<0.05%).")  
            return Status.FAILURE
        return Status.SUCCESS                                               # Accept
