from __future__ import annotations
import numpy as np                                                    # Numerical arrays and math
from scipy import ndimage as ndi                                      # Image processing 
from core import (                                                    # Project specific utilities and logging
    BehaviorNode, Status, blackboard, EveryNCalls, EveryNSeconds,
    WorldToGrid, BresenhamLine, map_logger, main_logger,
    ResolveMapPath, EnsureParentDirectories,
    BBKey, MappingParams, TH_FREE_PLANNER
)
from os.path import exists


###############################################################################
# ------------------------- Utilities -----------------------------------------
###############################################################################
# Small helper tools for shared state and math
# Keep names short and easy to read
def GetFromBlackboard(key, default=None):                               # Fetch a value from a global blackboard store
    return blackboard.Get(key, default)

def CalculateFreeSpacePercentage(cspace: np.ndarray) -> float:          # percent of cells above free threshold
    free_cells = float((cspace > TH_FREE_PLANNER).sum())                # count free cells
    return free_cells / float(cspace.size)                              # turn count into fraction


###############################################################################
# ------------------------- C-space builder -----------------------------------
###############################################################################
# Turn probability map into a safe navigation map
# Inflate obstacles and add soft borders near walls
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
    pm = np.asarray(probability_map, dtype=np.float32)              # ensure float array
    if downsample > 1:                                              # decimate for speed
        hh = pm.shape[0] // downsample                              # downsampled height
        ww = pm.shape[1] // downsample                              # downsampled width
        cut = pm[:hh * downsample, :ww * downsample]                # crop to fit factor
        pm_ds = cut.reshape(hh, downsample, ww, downsample).mean(axis=(1, 3))  # average blocks
    else:
        pm_ds = pm
    core = (pm_ds >= occupied_threshold).astype(np.uint8)           # hard obstacle core
    labels, num = ndi.label(core)                                   # connected components
    if num:                                                         # if any components exist
        res_px = max(1, downsample)                                 # pixels per original pixel in ds space
        m_per_px = max(1e-6, resolution_meters * res_px)            # meters per ds pixel
        min_side_m = 0.05                                           # tiny objects filter
        solidity_min = 0.05                                         # fill only solid blobs
        min_area_px = int((min_side_m / m_per_px) ** 2)             # area threshold
        objects = ndi.find_objects(labels)                           # slices for blobs
        for i, sl in enumerate(objects, start=1):
            if not sl:                                              # skip empty slices
                continue
            h = sl[0].stop - sl[0].start                            # box height
            w = sl[1].stop - sl[1].start                            # box width
            area = h * w
            if area < min_area_px:                                  # skip tiny blobs
                continue
            comp = (labels[sl] == i)                                # mask inside box
            solidity = float(comp.sum()) / float(area)              # fill ratio
            if solidity >= solidity_min:                            # fill and pad solid areas
                pad_h = max(1, h // 3) + max(1, min(h, w) // 6)     # vertical pad
                pad_w = max(1, w // 3) + max(1, min(h, w) // 6)     # horizontal pad
                r0 = max(0, sl[0].start - pad_h)
                r1 = min(core.shape[0], sl[0].stop + pad_h)
                c0 = max(0, sl[1].start - pad_w)
                c1 = min(core.shape[1], sl[1].stop + pad_w)
                core[r0:r1, c0:c1] = 1                              # mark padded area occupied
    if morph_closing > 0 and morph_iterations > 0:                  # optional closing
        se = np.ones((morph_closing, morph_closing), dtype=np.uint8)  # square kernel
        core = ndi.binary_closing(core, structure=se, iterations=morph_iterations)  # close gaps
    free_mask = (~core.astype(bool))                                 # free where not core
    dist_px = ndi.distance_transform_edt(free_mask).astype(np.float32)  # distance to obstacles
    inflate_px = (robot_radius_meters + safety_margin) / max(resolution_meters, 1e-6)  # meters to pixels
    inflate_px = float(max(0.0, inflate_px * inflation_scale))        # scale inflation

    # conservative cspace free if far enough from obstacles
    cspace_ds = np.where(dist_px >= inflate_px, 1.0, 0.0).astype(np.float32)

    # add soft gradient at the edge of obstacles
    gradient_zone = inflate_px * 0.5                                  # half radius for ramp
    gradient_mask = (dist_px < inflate_px) & (dist_px >= inflate_px - gradient_zone)
    if gradient_zone > 0:
        gradient_values = (dist_px - (inflate_px - gradient_zone)) / gradient_zone  # 0 to 1
        cspace_ds = np.where(gradient_mask, np.clip(gradient_values, 0.0, 1.0), cspace_ds)

    # gentle halo to discourage tight paths near walls
    halo_strength = 0.15                                              # small soften near walls
    if halo_strength > 0.0:
        halo_radius = inflate_px * 3.0                                # halo extent
        halo_denom = max(halo_radius - inflate_px, 1e-6)              # normalize range
        halo_dist = np.clip((dist_px - inflate_px) / halo_denom, 0.0, 1.0)
        halo = (1.0 - halo_dist) * halo_strength                      # stronger near boundary
        cspace_ds = np.where(cspace_ds > 0.8, cspace_ds, np.clip(cspace_ds - halo, 0.0, 1.0))

    # upsample back to original size if needed
    if downsample > 1:                                                # restore original grid size
        up = ndi.zoom(cspace_ds, zoom=downsample, order=0)            # nearest upsample
        return up[:pm.shape[0], :pm.shape[1]]                         # crop to shape
    return cspace_ds                                                  # return cspace grid


###############################################################################
# ------------------------- Helpers -------------------------------------------
###############################################################################
# Small checks and quick sanity helpers
# Keep mapping safe and robust
def is_cspace_frozen(bb) -> bool:                                      # check if cspace updates are paused
    val = bb.Get("cspace_frozen", False)                               # read flag
    return bool(val)                                                   # convert to bool

def validate_inflation_parameters():                                   # quick heuristic check
    s = max(52.9, 40.0)                                                # pixels per meter
    rpx = 0.16 * s * 0.55                                              # tuned pixel radius guess
    r_pix = int(max(1, round(rpx)))                                    # integer radius
    bubble = int(r_pix + 5)                                            # buffer radius
    core_m = r_pix / s                                                 # meters of core radius
    margin = core_m - 0.16                                             # extra margin beyond robot
    # tips removed on purpose


###############################################################################
# ------------------------- BT: Lidar Mapping ---------------------------------
################################################################################
# Build the map and cspace from live lidar
# Throttle work and declare ready when good enough
class LidarMappingBT(BehaviorNode):                                    # behavior tree node for building map and cspace
    def __init__(self, params, bb=None):                               # init mapping node
        super().__init__("LidarMappingBT")                             # node name
        self.params = params                                           # store params
        self.bb = bb or blackboard                                     # use provided or global bb
        self.step_counter = 0                                          # tick counter
        self.mission_complete = False                                  # done flag
        self.min_scan_ticks = 25                                       # minimum samples before finish
        self.robot_radius = params.robot_radius                        # cache robot radius
        self.mapping_interval_ms = 100                                 # target loop spacing in ms
        self.last_mapping_time = 0.0                                   # last update timestamp
        self.current_decimation = 4                                    # lidar decimation
        self.cspace_log_limiter = EveryNCalls(20)                      # log limiter
        self.cspace_build_limiter = EveryNSeconds(0.5)                 # build limiter
        self.lidar_update_limiter = EveryNCalls(2)                     # reserved limiter
        self.error_log_limiter = EveryNSeconds(5.0)                    # error throttle
        validate_inflation_parameters()                                # run quick sanity

    def create_cspace(self, prob_map: np.ndarray) -> np.ndarray | None:  # build cspace from prob map
        if prob_map is None:                                           # guard missing map
            return None
        return build_cspace(                                           # delegate to builder
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

    def should_stop_mapping(self) -> bool:                              # decide if mapping should stop
        # time based stop when enough ticks have passed
        if hasattr(self, "start_time"):
            robot = self.bb.GetRobot()
            if robot:
                elapsed_time = robot.getTime() - self.start_time        # seconds since start
                if elapsed_time > 120.0 and self.step_counter >= self.min_scan_ticks:
                    return True
        # stop if robot is stationary too long
        if self.step_counter > 2000 and self.is_robot_stationary():
            return True
        return False                                                    # default keep going

    def is_robot_stationary(self) -> bool:                              # measure small motion over time
        gps = self.bb.GetGps()                                          # need gps
        if not gps:
            return False
        cur = gps.getValues()                                           # current xyz
        if not hasattr(self, "last_pos"):                               # first call
            self.last_pos = cur
            self.stationary_count = 0
            return False
        dx = float(cur[0] - self.last_pos[0])                           # delta x
        dy = float(cur[1] - self.last_pos[1])                           # delta y
        dist = float(np.hypot(dx, dy))                                  # distance moved
        if dist < 0.05:                                                 # tiny movement
            self.stationary_count += 1
        else:
            self.stationary_count = 0                                   # reset on movement
        self.last_pos = cur                                             # update history
        enough_scans = (self.step_counter >= self.min_scan_ticks)       # do not stop too early
        return (self.stationary_count > 200) and enough_scans           # true when still for a while

    def assess_map_readiness(self, prob_map: np.ndarray, cspace: np.ndarray) -> bool:  # is map usable
        if prob_map is None:                                            # need prob map
            return False
        if cspace is None:                                              # need cspace too
            return False
        total = int(prob_map.size)                                      # cell count
        known_mask = (prob_map != 0.5) & np.isfinite(prob_map)          # cells that were updated
        coverage = float(np.sum(known_mask)) / float(total)             # updated fraction
        free_frac = float(np.sum(cspace > self.params.th_free_planner)) / float(total)  # free fraction
        return (coverage >= 0.01) and (free_frac >= 0.005)              # minimal bar

    def execute(self) -> Status:                                        # one tick of mapping
        self.step_counter += 1                                          # tick count
        robot = self.bb.GetRobot()                                      # robot time
        if (not hasattr(self, "start_time")) and robot:                 # init start time
            self.start_time = robot.getTime()

        if self.should_stop_mapping():                                  # maybe end soon
            if self.step_counter >= self.min_scan_ticks:                # enough samples
                if not self.mission_complete:
                    map_logger.Info("Mapping completed - creating final c-space")
                self.mission_complete = True
            return Status.RUNNING                                       # keep ticking to finalize
        if robot:                                                       # spacing between updates
            dt_ms = (robot.getTime() - self.last_mapping_time) * 1000.0
            if dt_ms < self.mapping_interval_ms:                        # too soon
                return Status.RUNNING
        if (self.step_counter % self.params.mapping_interval) != 0:     # decimate work
            return Status.RUNNING

        try:
            gps = self.bb.GetGps()                                      # fetch sensors
            compass = self.bb.GetCompass()
            lidar = self.bb.GetLidar()
            if not (gps and compass and lidar):                         # wait if missing
                return Status.RUNNING
            xw, yw = gps.getValues()[:2]                                # world x y
            cv = compass.getValues()                                    # compass vector
            theta = float(np.arctan2(cv[0], cv[1]))                     # heading angle
            prob_map = self.bb.GetProbMap()                             # read map
            if prob_map is None:                                        # create if missing
                prob_map = np.zeros((200, 300), dtype=np.float32)
                self.bb.Set("prob_map", prob_map)
            try:
                ranges = np.asarray(lidar.getRangeImage(), dtype=float) # lidar ranges
                if ranges.size == 0 or (not np.isfinite(ranges).any()): # no usable data
                    return Status.RUNNING

                good = np.isfinite(ranges) & (ranges > 0.0)             # valid mask
                if int(good.sum()) < int(ranges.size * 0.1):            # too few hits
                    return Status.RUNNING
            except Exception as e:                                      # sensor error
                if robot and self.error_log_limiter.ShouldExecute(robot.getTime()):
                    map_logger.Error(f"LiDAR acquisition failed: {e}")
                return Status.RUNNING

            n = int(ranges.size)                                        # ray count
            half_fov = np.radians(240.0) / 2.0                          # half field of view
            angles = np.linspace(+half_fov, -half_fov, n)               # angle per ray
            r0, c0 = WorldToGrid(xw, yw, prob_map.shape)                # robot cell
            H, W = prob_map.shape                                       # grid size

            # integrate this scan into the probability map
            self.update_prob_map_vectorized(prob_map, ranges, angles, theta, xw, yw, r0, c0, H, W)

            if robot:
                self.last_mapping_time = robot.getTime()                # store last time
            self.bb.SetProbMap(prob_map)                                # publish map

            # build cspace at a controlled rate when not frozen
            if (not is_cspace_frozen(self.bb)) and robot:
                if self.cspace_build_limiter.ShouldExecute(robot.getTime()):
                    cspace = self.create_cspace(prob_map)               # build cspace
                    if cspace is not None:
                        self.bb.SetCspace(cspace)                       # publish cspace

            # after mission complete set ready when free area exists
            if self.mission_complete:
                cs = self.bb.GetCspace()
                if cs is not None:
                    frac = (cs > self.params.th_free_planner).sum() / float(cs.size)
                    if frac >= 0.01:
                        self.bb.SetMapReady(True)                       # mark ready
        except Exception as e:                                          # global catch
            if robot and self.error_log_limiter.ShouldExecute(robot.getTime()):
                map_logger.Error(f"Exception: {e}")                     # throttled error
            return Status.FAILURE                                       # fail this tick
        return Status.RUNNING                                           # continue

    def update_prob_map_vectorized(                                     # vectorized update of probability map
        self,
        prob_map,                                                       # target map
        lidar_ranges,                                                   # raw ranges
        angles,                                                         # ray angles
        theta,                                                          # robot heading
        xw,                                                             # robot x world
        yw,                                                             # robot y world
        row0,                                                           # robot row
        col0,                                                           # robot col
        H,                                                              # map height
        W                                                               # map width
    ):
        valid = np.isfinite(lidar_ranges) & (lidar_ranges > 0.1) & (lidar_ranges < 8.0)  # keep reasonable hits
        if not valid.any():                                             # nothing to do
            return
        dec = int(self.current_decimation)                              # decimation factor
        if dec > 1:                                                     # stride mask
            mask = np.zeros_like(valid, dtype=bool)
            mask[::dec] = True
            valid = valid & mask
        near = (lidar_ranges < 4.0)                                     # emphasize close hits
        if near.any():
            od = max(1, dec // 3)                                       # oversample factor
            add_mask = np.zeros_like(valid, dtype=bool)
            idx = np.nonzero(near & np.isfinite(lidar_ranges))[0]       # indices of usable near hits
            add_mask[idx[::od]] = True                                  # add some near hits back
            valid = valid | add_mask                                    # merge masks
        if not valid.any():                                             # all filtered out
            return
        rng = lidar_ranges[valid]                                       # selected ranges
        ang = angles[valid]                                             # selected angles
        ct = float(np.cos(theta))                                       # cos heading
        st = float(np.sin(theta))                                       # sin heading
        ca = np.cos(theta + ang)                                        # cos absolute ray angle
        sa = np.sin(theta + ang)                                        # sin absolute ray angle
        xh = xw + rng * ca + 0.202 * ct                                 # hit x world
        yh = yw + rng * sa + 0.202 * st                                 # hit y world
        rows1 = np.round(40.0 * (xh + 2.25)).astype(int)                # world x to row
        cols1 = np.round(-52.9 * (yh - 1.6633)).astype(int)             # world y to col
        rows1 = np.clip(rows1, 0, H - 1)                                # clamp r
        cols1 = np.clip(cols1, 0, W - 1)                                # clamp c
        batch = min(400, len(rows1))                                    # batch size
        for start in range(0, len(rows1), batch):                       # process in chunks
            re = rows1[start:start + batch]                             # end rows
            ce = cols1[start:start + batch]                             # end cols
            free_r = []                                                 # free rows along rays
            free_c = []                                                 # free cols along rays
            for r1, c1 in zip(re, ce):                                  # each hit endpoint
                pts = BresenhamLine(row0, col0, r1, c1)                 # ray cells
                if not pts:
                    continue
                if len(pts) > 1:
                    pts = pts[:-1]                                      # drop hit cell
                stride = 6                                              # sample step
                if len(pts) > stride:
                    pts = [pts[0], *pts[stride:-1:stride], pts[-1]]     # sparse along ray
                for rr, cc in pts:                                      # add free cells
                    if (0 <= rr < H) and (0 <= cc < W):
                        free_r.append(rr)
                        free_c.append(cc)
            if free_r:                                                  # apply free updates
                fr = np.asarray(free_r, dtype=int)
                fc = np.asarray(free_c, dtype=int)
                prob_map[fr, fc] = np.maximum(prob_map[fr, fc] - 0.004, 0.0)  # lower prob
            inb = (re >= 0) & (re < H) & (ce >= 0) & (ce < W)          # endpoint in bounds
            if not inb.any():
                continue
            hr = re[inb]                                                # hit rows
            hc = ce[inb]                                                # hit cols
            dr = np.arange(-1, 2)                                       # offset rows
            dc = np.arange(-1, 2)                                       # offset cols
            drg, dcg = np.meshgrid(dr, dc, indexing='ij')               # 3 by 3 patch
            r_idx = np.clip(hr[:, None] + drg.flatten(), 0, H - 1).flatten()  # neighbor rows
            c_idx = np.clip(hc[:, None] + dcg.flatten(), 0, W - 1).flatten()  # neighbor cols
            prob_map[r_idx, c_idx] = np.minimum(prob_map[r_idx, c_idx] + 0.020, 1.0)  # raise prob near hits

    def reset(self):                                                    # reset node state
        super().reset()                                                 # base reset
        self.step_counter = 0                                           # clear counters
        self.mission_complete = False                                   # clear flag
        if hasattr(self, "completion_reported"):                        # remove optional field
            delattr(self, "completion_reported")
        if not is_cspace_frozen(self.bb):                               # only clear when not frozen
            self.bb.SetProbMap(None)
            self.bb.SetCspace(None)
            self.bb.Set("obstacle_field", None)

    def terminate(self):                                                # end of node
        self.mission_complete = True                                    # mark done
        cs = self.bb.GetCspace()                                        # try freeze existing cspace
        if cs is not None:
            free_frac = (cs > self.params.th_free_planner).sum() / float(cs.size)  # free fraction
            if free_frac >= 0.005:                                      # minimal quality
                self.bb.Set("cspace_frozen", True)                      # freeze updates
                self.bb.SetMapReady(True)                               # mark ready
                return
        pm = self.bb.GetProbMap()                                       # latest prob map
        if pm is None:                                                  # cannot build
            map_logger.Warning("terminate(): no prob_map to build c-space from")
            return
        cspace = self.create_cspace(pm)                                 # build once
        if cspace is None:
            return
        if not self.bb.Get("map_ready", False):                         # stage if not ready
            self.bb.Set("cspace_live", cspace)
        else:
            self.bb.Set("cspace", self.bb.Get("cspace_live", cspace))   # commit staged cspace
            self.bb.Set("map_ready", True)                              # mark ready
        self.bb.Set("cspace_frozen", True)                              # freeze cspace


###############################################################################
# ------------------------- Behavior Tree Nodes ------------------------------
###############################################################################
# Nodes for map readiness and persistence
# Load save validate and wait for maps
class WaitForMapReady(BehaviorNode):
    def __init__(self):
        super().__init__("WaitForMapReady")                             
        self._start_time = None                                         # when we began waiting
        self._warned = [False] * 3                                      # which warnings sent

    def execute(self):
        robot = GetFromBlackboard("robot")                              # robot time source
        if robot and self._start_time is None:                          # first tick with robot
            self._start_time = robot.getTime()                          # record start time
        if GetFromBlackboard("map_saved"):                              # already saved
            return Status.SUCCESS
        cspace = GetFromBlackboard("cspace")                            # do we have cspace
        if cspace is None:                                              # none yet
            if robot and self._start_time is not None:          
                elapsed = robot.getTime() - self._start_time            # seconds elapsed
                msgs = [                                                # rising warnings
                    (120, "Still no map after 120s."),
                    (60, "Map build is slow 60s."),
                    (30, "Waiting for C-space 30s"),
                ]
                for idx, (t, msg) in enumerate(msgs[::-1]):            # newest last
                    real_idx = 2 - idx
                    if elapsed > t and not self._warned[real_idx]:
                        main_logger.Warning(msg)                        # send one warning
                        self._warned[real_idx] = True                   # mark sent
            return Status.RUNNING                                       # keep waiting
        # have cspace now check free fraction
        return Status.SUCCESS if CalculateFreeSpacePercentage(cspace) >= 0.005 else Status.RUNNING

    def reset(self):
        super().reset()                                                 # base reset
        self._start_time = None                                         # clear timer
        self._warned = [False] * 3                                      # clear flags


class MapExistsOrReady(BehaviorNode):
    def __init__(self, path="cspace.npy"):
        super().__init__("MapExistsOrReady")                            # node name
        self.path = str(ResolveMapPath(path))                           # file location

    def execute(self):
        # succeed if map is ready or file is present
        return Status.SUCCESS if (GetFromBlackboard("map_ready", False) or exists(self.path)) else Status.FAILURE


class LoadMap(BehaviorNode):
    def __init__(self, path="cspace.npy"):
        super().__init__("LoadMap")                                     # node name
        self.path = str(ResolveMapPath(path))                           # map to load

    def execute(self):
        try:
            c = np.clip(np.load(self.path).astype(np.float32), 0.0, 1.0)  # read and clamp
            blackboard.SetCspace(c)                                     # publish cspace
            blackboard.Set(BBKey.CSPACE_FROZEN, True)                   # freeze changes
            blackboard.SetMapReady(True)                                # mark ready
            return Status.SUCCESS                                     
        except Exception as e:
            main_logger.Error(f"LoadMap failed {e}")                    # load error
            return Status.FAILURE                                       # fail node


class EnsureCspaceNow(BehaviorNode):
    def __init__(self, blackboard_instance=None):
        super().__init__("EnsureCspaceNow")                             # node name
        self.blackboard = blackboard_instance or blackboard             # pick bb

    def execute(self):
        c = self.blackboard.GetCspace()                                 # already have
        if c is not None:
            return Status.SUCCESS                                       # nothing to do
        p = self.blackboard.GetProbMap()                                # try prob map
        if p is None:
            return Status.RUNNING                                       # wait for map
        try:
            mapper = LidarMappingBT(params=MappingParams(), bb=self.blackboard)  # temp mapper
            c = mapper.create_cspace(p)                                 # build cspace
            if c is not None:
                self.blackboard.SetCspace(c)                            # store cspace
                return Status.SUCCESS
            return Status.FAILURE                                       # no output
        except Exception as e:
            main_logger.Error(f"c-space build crashed: {e}")            # crash path
            return Status.FAILURE


class SaveMap(BehaviorNode):
    def __init__(self, path="cspace.npy", threshold=None):
        super().__init__("SaveMap")                                     # node name
        self.path = str(ResolveMapPath(path))                           # save path
        self.threshold = 0.30 if threshold is None else threshold       # fallback threshold
        self.done = False                                               # guard flag

    def PrepareMapForSaving(self) -> np.ndarray | None:
        c = GetFromBlackboard("cspace")                                 # prefer cspace
        if c is None:                                                   # if missing try to build
            p = GetFromBlackboard("prob_map")                           # get prob map
            if p is None:
                main_logger.Error("no map.")                            # nothing to save
                return None
            mapper = LidarMappingBT(MappingParams())                    # build helper
            c = mapper.create_cspace(p)                                 # build once
            if c is None:                                               # fallback hard threshold
                c = (p <= self.threshold).astype(np.float32)
        return np.clip(c.astype(np.float32), 0.0, 1.0)                  # clamp output

    def ShouldSaveMap(self, cspace: np.ndarray) -> bool:
        return CalculateFreeSpacePercentage(cspace) * 100.0 >= 0.1      # minimal free space

    def SaveMapToFile(self, cspace: np.ndarray) -> bool:
        try:
            EnsureParentDirectories(self.path)                          # ensure folders
            np.save(self.path, cspace)                                  # save npy
            return True
        except Exception as e:
            main_logger.Error(f"Write failed to save map: {e}")         # write error
            return False

    def UpdateMapState(self, cspace: np.ndarray):
        for k, v in (("cspace", cspace), ("map_ready", True), ("map_saved", True), ("cspace_frozen", True)):
            blackboard.Set(k, v)                                        # update keys
        self.done = True                                                # mark done

    def execute(self):
        if self.done:                                                   # already saved
            return Status.SUCCESS
        c = self.PrepareMapForSaving()                                  # get map
        if c is None:                                                   # nothing to do
            return Status.FAILURE
        if not self.ShouldSaveMap(c):                                   # quality check
            main_logger.Error("C-space rejected - not saving.")         # explain
            return Status.FAILURE
        try:
            from os import remove                                       # delete old file
            if exists(self.path):
                remove(self.path)
        except Exception:
            pass                                                        # ignore delete errors
        if not self.SaveMapToFile(c):                                   # write failed
            return Status.FAILURE
        self.UpdateMapState(c)                                          # publish state
        return Status.SUCCESS                                           # success


class ValidateLoadedMap(BehaviorNode):
    def __init__(self):
        super().__init__("ValidateLoadedMap")                           # node name

    def execute(self):
        c = GetFromBlackboard("cspace")                                 # read map
        if c is None:                                                   # missing map
            return Status.FAILURE
        free_pct = 100.0 * CalculateFreeSpacePercentage(c)              # free percent
        if free_pct < 0.05:                                            # too little free area
            main_logger.Error(f"Loaded map looks wrong: free={free_pct:.2f}% (<0.05%).")
            return Status.FAILURE
        return Status.SUCCESS                                           # looks fine
