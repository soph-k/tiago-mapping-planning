from __future__ import annotations
import numpy as np
import heapq
from collections import defaultdict
from typing import List, Tuple, Optional

from core import (
    BehaviorNode, Status, blackboard, plan_logger, main_logger,
    PlanningParams, BBKey,
    ResolveMapPath as RESMAP, EnsureParentDirectories as ENSURE
)

###############################################################################
# --- Utilities ---------------------------------------------------------------
###############################################################################
# Small helpers for grid math and checks
# Readable names and short logic
def GetFromBlackboard(key, default=None):                 # Pull value from shared store
    return blackboard.Get(key, default)

def Inb(h, w, r, c):                                      # Inside grid bounds
    return (0 <= r < h) and (0 <= c < w)

def heuristic(a, b):                                      # Euclidean distance
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))

def Thr(t, p: PlanningParams):                            # Pick threshold with default
    if t is None:
        return p.th_free_planner
    return t

def is_free(cs, r, c, p: PlanningParams, thr=None):       # Free if prob above threshold
    h, w = cs.shape
    if not Inb(h, w, r, c):
        return False
    return cs[r, c] > Thr(thr, p)

def is_free_with_neighbors(cs, r, c, p: PlanningParams, threshold=None, check_neighbors=True):
    if not is_free(cs, r, c, p, threshold):               # Cell must be free
        return False
    if not check_neighbors:                                # Skip neighbor check
        return True
    h, w = cs.shape
    t = Thr(threshold, p)
    for dr in (-1, 0, 1):                                  # Check 8-neighborhood
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if Inb(h, w, rr, cc) and cs[rr, cc] <= t:      # Any neighbor blocked
                return False
    return True

def NearestFreeCell(cs, r, c, p: PlanningParams, threshold=None, max_radius=12, check_neighbors=True):
    if is_free_with_neighbors(cs, r, c, p, threshold, check_neighbors):
        return (r, c)                                      # Start already safe
    h, w = cs.shape
    for R in range(1, max_radius + 1):                     # Grow ring by ring
        for dr in range(-R, R + 1):
            for dc in range(-R, R + 1):
                if max(abs(dr), abs(dc)) != R:             # Only ring border
                    continue
                rr = r + dr
                cc = c + dc
                ok = Inb(h, w, rr, cc)
                if ok and is_free_with_neighbors(cs, rr, cc, p, threshold, check_neighbors):
                    return (rr, cc)                        # First safe cell
    return None                                            # None found

def Neighbors(r, c, cs, p: PlanningParams):
    h, w = cs.shape
    for dr in (-1, 0, 1):                                  # 8 directions
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if Inb(h, w, rr, cc) and cs[rr, cc] > p.th_free_planner:
                move_cost = p.sqrt_2 if (dr != 0 and dc != 0) else 1.0
                yield (rr, cc, move_cost)                  # Neighbor and cost

def Reconstruct(came, goal):
    path = []                                              # Backtrack path
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = came[cur]
    path.reverse()
    return path

###############################################################################
# --- Jump Point Search -------------------------------------------------------
###############################################################################
# Prune neighbors and jump across empty runs
# Speeds up long open area searches
def Forced(r, c, dr, dc, cs, p):                          # Check forced neighbor
    h, w = cs.shape
    nr = r + dr
    nc = c + dc
    if not Inb(h, w, nr, nc) or cs[nr, nc] <= p.th_free_planner:
        return False
    checks = ((nr + dr, nc), (nr, nc + dc), (nr + dr, nc + dc))
    for rr, cc in checks:
        if Inb(h, w, rr, cc) and cs[rr, cc] <= p.th_free_planner:
            return True
    return False

def IsJp(r, c, dr, dc, cs, p):                            # Is this a jump point
    if dr != 0 and dc != 0:                               # Diagonal case
        return Forced(r, c, dr, 0, cs, p) or Forced(r, c, 0, dc, cs, p)
    diags = ((1, 1), (1, -1), (-1, 1), (-1, -1))          # Straight case
    for u, v in diags:
        if Forced(r, c, u, v, cs, p):
            return True
    return False

def Jump(r, c, dr, dc, cs, p):                            # Walk until stop or jp
    h, w = cs.shape
    sr = r                                                # Start row
    sc = c                                                # Start col
    cap = min(h, w) // 2                                  # Hard cap on run
    while True:
        r += dr
        c += dc
        if (not Inb(h, w, r, c)) or cs[r, c] <= p.th_free_planner:
            return (None, None)                           # Hit wall or out
        if IsJp(r, c, dr, dc, cs, p):
            return (r, c)                                 # Found jump point
        if abs(r - sr) > cap or abs(c - sc) > cap:
            return (r, c)                                 # Bail near cap

def JpsNbrs(r, c, cs, goal, p):                           # JPS neighbor gen
    out = []
    gr, gc = goal
    dr = (gr > r) - (gr < r)                              # Step sign row
    dc = (gc < c) - (gc > c)                              # Step sign col
    dirs = ((dr, -dc), (dr, 0), (0, -dc), (-dr, -dc),
            (dr, dc), (-dr, 0), (0, dc), (-dr, dc))       # Biased order
    for rr, cc in dirs:
        if rr == 0 and cc == 0:
            continue
        jr, jc = Jump(r, c, rr, cc, cs, p)
        if jr is not None:
            cost = p.sqrt_2 if (rr != 0 and cc != 0) else 1.0
            out.append((jr, jc, cost))                    # Jump target
    return out

###############################################################################
# --- A* variants -------------------------------------------------------------
###############################################################################
# Standard A star and bidirectional search
# Optional jump point search for speed
def Astar(cs, start, goal, max_it, max_open, h_w, use_jps, early_exit, p: PlanningParams):
    openq = [(heuristic(start, goal) * h_w, 0.0, start, None)]  # f g node parent
    came = {}                                                   # Closed set as parents
    g = defaultdict(lambda: float("inf"))                       # Cost so far
    g[start] = 0.0
    it = 0
    while openq and it < max_it:                                # Main loop
        it += 1
        if len(openq) > max_open:                               # Guard memory
            return []
        f, gg, cur, parent = heapq.heappop(openq)               # Best first
        if gg > early_exit:                                     # Early stop gate
            return []
        if cur in came:                                         # Skip if closed
            continue
        came[cur] = parent                                      # Close it
        if cur == goal:                                         # Reached goal
            break
        if use_jps:
            nbrs = JpsNbrs(cur[0], cur[1], cs, goal, p)         # Pruned nbrs
        else:
            nbrs = Neighbors(cur[0], cur[1], cs, p)             # Full nbrs
        for nr, nc, mc in nbrs:
            nxt = (nr, nc)
            ng = gg + mc                                        # New cost
            if nxt in came:                                     # Skip closed
                continue
            if ng >= g[nxt]:                                    # Not better
                continue
            g[nxt] = ng
            h = heuristic(nxt, goal) * h_w                      # Weighted h
            heapq.heappush(openq, (ng + h, ng, nxt, cur))       # Push
    if goal in came:
        return Reconstruct(came, goal)                          # Build path
    return []                                                   # No path

def BiAstar(cs, start, goal, max_it, max_open, h_w, p: PlanningParams):
    f_open = [(heuristic(start, goal) * h_w, 0.0, start, None)] # Front queue
    b_open = [(heuristic(goal, start) * h_w, 0.0, goal, None)]  # Back queue
    f_c, b_c = {}, {}                                           # Closed sets
    f_g = defaultdict(lambda: float("inf"))                     # Costs front
    b_g = defaultdict(lambda: float("inf"))                     # Costs back
    f_g[start] = 0.0
    b_g[goal] = 0.0
    it = 0
    meet = None                                                 # Meet point
    while (f_open or b_open) and it < max_it:
        it += 1
        if (it % 2) == 1 and f_open:                            # Alternate sides
            q = f_open
            close = f_c
            gcost = f_g
            tgt = goal
            def hx(n): return heuristic(n, goal) * h_w          # Heuristic front
        else:
            q = b_open
            close = b_c
            gcost = b_g
            tgt = start
            def hx(n): return heuristic(n, start) * h_w         # Heuristic back
        _, gval, cur, par = heapq.heappop(q)                    # Expand one
        if cur in close:
            continue
        close[cur] = par
        if cur in (b_c if close is f_c else f_c):               # Met other side
            meet = cur
            break
        for nr, nc, mc in Neighbors(cur[0], cur[1], cs, p):     # Explore nbrs
            nxt = (nr, nc)
            ng = gval + mc
            if nxt in close:
                continue
            if ng >= gcost[nxt]:
                continue
            gcost[nxt] = ng
            heapq.heappush(q, (ng + hx(nxt), ng, nxt, cur))
        if len(f_open) + len(b_open) > max_open:                # Memory cap
            return []
    if meet is None:
        return []                                               # No meet
    f_path = []                                                 # Front part
    cur = meet
    while cur is not None:
        f_path.append(cur)
        cur = f_c.get(cur)
    f_path.reverse()
    b_path = []                                                 # Back part
    cur = meet
    while cur is not None:
        b_path.append(cur)
        cur = b_c.get(cur)
    return f_path + b_path[1:]                                  # Merge

def a_star(cs, start, goal, p: PlanningParams, max_iterations=None, max_open_set_size=None, heuristic_weight=None):
    max_it = max_iterations if max_iterations is not None else p.max_iterations
    max_open = max_open_set_size if max_open_set_size is not None else p.max_open_set_size
    h_w = heuristic_weight if heuristic_weight is not None else p.heuristic_weight
    if not is_free(cs, start[0], start[1], p) or not is_free(cs, goal[0], goal[1], p):
        plan_logger.Warning(f"A*: Start or goal not free - start: {is_free(cs, start[0], start[1], p)}, goal: {is_free(cs, goal[0], goal[1], p)}")
        return []
    if start == goal:
        plan_logger.Info("A*: Start equals goal, returning single point")
        return [start]
    dist = heuristic(start, goal)                               # Grid distance
    mindim = min(cs.shape)                                      # Size hint
    use_jps = bool(p.jump_point_search and (dist >= mindim * 0.15))
    if p.bidirectional and dist > mindim * 0.8:                 # Long trips
        return BiAstar(cs, start, goal, max_it, max_open, h_w, p)
    early = dist * p.early_exit_multiplier                      # Early gate
    return Astar(cs, start, goal, max_it, max_open, h_w, use_jps, early, p)

###############################################################################
# --- Path smoothing ----------------------------------------------------------
###############################################################################
# Keep path short and straight when safe
# Uses line of sight checks across segments
def has_line_of_sight(a, b, cs, p: PlanningParams):
    r0, c0 = a
    r1, c1 = b
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r = r0
    c = c0
    while True:
        if not is_free(cs, r, c, p):                            # Hit blocked
            return False
        if (r, c) == (r1, c1):                                  # Reached end
            return True
        e2 = err << 1                                           # Bresenham step
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc

def smooth_path(path, cs, p: PlanningParams, look=15):
    if len(path) <= 2:
        return path
    out = [path[0]]                                            # Always keep start
    i = 0
    n = len(path)
    while i < n - 1:
        far = i + 1                                            # Farthest we can skip to
        upto = min(n, i + look + 1)                            # Window limit
        j = i + 2
        while j < upto:
            if has_line_of_sight(path[i], path[j], cs, p):     # If clear, jump
                far = j
                j += 1
            else:
                break
        out.append(path[far])                                  # Add new point
        i = far                                                # Continue from here
    return out

###############################################################################
# --- Differential drive optimization -----------------------------------------
###############################################################################
# Shape path to favor straight and 45 degree runs
# Better for two wheel robots
def Collinear(a, b, c, p: PlanningParams):
    cp = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))  # Cross product area
    d = np.hypot(c[0] - a[0], c[1] - a[1])                                    # Span
    if d < 1e-6:
        return True
    return (cp / d) < p.differential_drive_alignment_tolerance                # Close to line

def AlignCardinal(pt, prev, nxt, p: PlanningParams):
    dx = nxt[0] - prev[0]                                    # Direction x
    dy = nxt[1] - prev[1]                                    # Direction y
    L = np.hypot(dx, dy)                                     # Length
    if L < 1e-6:
        return pt
    dx /= L                                                  # Unit dir
    dy /= L
    s = 1 / np.sqrt(2.0)                                     # 45 degree scale
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1),
            (s, s), (-s, s), (s, -s), (-s, -s)]              # 8 preferred axes
    best_i = 0
    best_dot = -1e9
    for i, (ux, uy) in enumerate(dirs):                      # Pick closest axis
        dot = dx * ux + dy * uy
        if dot > best_dot:
            best_dot = dot
            best_i = i
    ux, uy = dirs[best_i]
    ang = np.degrees(np.arccos(np.clip(ux * dx + uy * dy, -1, 1)))  # Angle diff
    if ang < p.differential_drive_angle_tolerance:          # Within tolerance
        dist = (pt[0] - prev[0]) * ux + (pt[1] - prev[1]) * uy      # Project
        h, w = p.default_map_shape                                  # Clamp to grid
        rr = int(round(dist * ux + prev[0]))
        cc = int(round(dist * uy + prev[1]))
        rr = max(0, min(h - 1, rr))
        cc = max(0, min(w - 1, cc))
        return (rr, cc)
    return pt

def optimize_path_for_differential_drive(path, cs, p: PlanningParams):
    if len(path) <= 2:
        return path
    out = [path[0]]
    for i in range(1, len(path) - 1):                      # Walk inner points
        cur = path[i]
        prev = out[-1]
        nxt = path[i + 1]
        if not Collinear(prev, cur, nxt, p):               # If bend, try snap
            cur = AlignCardinal(cur, prev, nxt, p)
        out.append(cur)
    out.append(path[-1])
    return out

def Downsample(pth, cap=40):
    if len(pth) <= cap:
        return pth
    step = max(1, len(pth) // cap)                         # Even thinning
    slim = pth[::step]
    if slim[-1] != pth[-1]:
        slim = slim + [pth[-1]]
    return slim

###############################################################################
# --- Adaptive planning -------------------------------------------------------
###############################################################################
# Adjust limits and weights to fit the map
# Try several settings until one works
def AdaptiveParams(cs, start, goal, p: PlanningParams):
    free = int((cs > p.th_free_planner).sum())             # Count free cells
    total = cs.size
    occ_pct = (100.0 * free / total)                       # Free percent
    dense = (100 - occ_pct) > 40                           # Many obstacles
    sparse = (100 - occ_pct) < 10                          # Few obstacles
    dist = heuristic(start, goal)
    out = {
        "max_iterations": p.max_iterations,
        "max_open_set_size": p.max_open_set_size,
        "heuristic_weight": p.heuristic_weight
    }
    if dense:                                              # Tight maps
        out["max_iterations"] = int(p.max_iterations * 0.7)
        out["max_open_set_size"] = int(p.max_open_set_size * 0.8)
        out["heuristic_weight"] = 1.1
    elif sparse:                                           # Open maps
        out["max_iterations"] = int(p.max_iterations * 1.5)
        out["heuristic_weight"] = 0.9
    if dist > min(cs.shape) * 0.5:                         # Long distance
        out["heuristic_weight"] = min(out["heuristic_weight"] * 1.2, 1.5)
    return out

def adaptive_a_star(cs, start, goal, p: PlanningParams):
    tries = [                                              # Ordered attempts
        AdaptiveParams(cs, start, goal, p),
        {
            "max_iterations": p.adaptive_max_iterations,
            "max_open_set_size": p.adaptive_max_open_set_size,
            "heuristic_weight": p.adaptive_heuristic_weight
        },
        {
            "max_iterations": p.adaptive_max_iterations,
            "max_open_set_size": p.adaptive_max_open_set_size,
            "heuristic_weight": 2.0
        },
    ]
    for i, kw in enumerate(tries):
        path = a_star(cs, start, goal, p, **kw)            # Try settings
        if path:
            return path
    plan_logger.Error("All adaptive A* attempts failed")   # Nothing worked
    return []

###############################################################################
# --- Public API --------------------------------------------------------------
###############################################################################
# Main planning call used by other modules
# Returns waypoints in world units
def plan_path(cs, start_w, goal_w, p: PlanningParams, smooth=True,
              optimize_for_differential_drive=True, check_neighbors=None,
              max_search_radius=None, **_):
    if check_neighbors is None:
        check_neighbors = p.check_neighbor_safety
    if max_search_radius is None:
        max_search_radius = p.safe_waypoint_search_radius
    W2G = getattr(p, 'W2G', blackboard.Get("world_to_grid"))  # World to grid
    G2W = getattr(p, 'G2W', blackboard.Get("grid_to_world"))  # Grid to world
    sr, sc = W2G(*start_w, cs.shape)                          # Grid start
    gr, gc = W2G(*goal_w, cs.shape)                           # Grid goal
    ns = NearestFreeCell(cs, sr, sc, p, max_radius=max_search_radius, check_neighbors=check_neighbors)
    if ns is None:
        plan_logger.Error(f"No safe start found at ({sr}, {sc})")
        return []
    sr, sc = ns
    ng = NearestFreeCell(cs, gr, gc, p, max_radius=max_search_radius, check_neighbors=check_neighbors)
    if ng is None:
        plan_logger.Error(f"No safe goal found at ({gr}, {gc})")
        return []
    gr, gc = ng
    if has_line_of_sight((sr, sc), (gr, gc), cs, p):         # Direct shot
        return [start_w, goal_w]
    path = adaptive_a_star(cs, (sr, sc), (gr, gc), p)        # Grid path
    if not path:
        plan_logger.Error(f"Planning failed from ({sr}, {sc}) to ({gr}, {gc})")
        return []
    if optimize_for_differential_drive and p.optimize_for_differential_drive and len(path) > 2:
        path = optimize_path_for_differential_drive(path, cs, p)  # Snap bends
    if smooth and len(path) > 2:
        path = smooth_path(path, cs, p)                    # Reduce points
    path = Downsample(path, cap=40)                        # Trim length
    out = []
    for r, c in path:
        out.append(G2W(r, c))                              # Back to world
    if out[-1] != goal_w:                                  # Ensure end is goal
        out.append(goal_w)
    return out

###############################################################################
# --- Reachability analysis ---------------------------------------------------
###############################################################################
# Flood fill style reach mask
# Can limit random goal sampling
def reachable_mask_from(cs, start_world, p: PlanningParams, threshold=None, connectivity=8):
    if start_world is None:
        return None
    W2G = getattr(p, 'W2G', blackboard.Get("world_to_grid"))
    r0, c0 = W2G(*start_world, cs.shape)                    # Seed cell
    t = Thr(threshold, p)
    h, w = cs.shape
    if (not Inb(h, w, r0, c0)) or cs[r0, c0] <= t:          # If blocked, nudge
        near = NearestFreeCell(cs, r0, c0, p, threshold=t, max_radius=6, check_neighbors=False)
        if near is None:
            return np.zeros((h, w), dtype=bool)
        r0, c0 = near
    vis = np.zeros((h, w), dtype=bool)                      # Seen mask
    stack = [(r0, c0)]                                      # DFS stack
    while stack:
        r, c = stack.pop()
        if (not Inb(h, w, r, c)) or vis[r, c] or cs[r, c] <= t:
            continue
        vis[r, c] = True
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if connectivity == 4 and dr != 0 and dc != 0:  # No diagonals
                    continue
                stack.append((r + dr, c + dc))
    return vis

def find_safe_positions(cs, p: PlanningParams, num_positions=10, threshold=None,
                        restrict_to_reachable_from=None, check_neighbors=True, max_attempts=None):
    out = []
    h, w = cs.shape
    t = min(1.0, Thr(threshold, p) + 0.05)                 # Slightly stricter
    if max_attempts is None:
        max_attempts = num_positions * 20                   # Try budget
    if restrict_to_reachable_from is not None:
        reach = reachable_mask_from(cs, restrict_to_reachable_from, p, threshold=t)
    else:
        reach = None
    G2W = getattr(p, 'G2W', blackboard.Get("grid_to_world"))
    tries = 0
    while tries < max_attempts:
        tries += 1
        r = np.random.randint(10, h - 10)                   # Avoid borders
        c = np.random.randint(10, w - 10)
        if reach is not None and not reach[r, c]:
            continue
        if cs[r, c] > t and is_free_with_neighbors(cs, r, c, p, threshold=t, check_neighbors=check_neighbors):
            xw, yw = G2W(r, c)                              # World coords
            too_close = False
            for ex, ey in out:
                if np.hypot(xw - ex, yw - ey) < 0.5:        # Keep spread out
                    too_close = True
                    break
            if not too_close:
                out.append((xw, yw))
                if len(out) >= num_positions:
                    break
    return out

###############################################################################
# --- Validation and visualization --------------------------------------------
###############################################################################
# Check waypoints against the map
# Optionally draw a mask for display
def validate_path(path_w, cs, p: PlanningParams):
    if len(path_w) < 2:
        return True
    W2G = getattr(p, 'W2G', blackboard.Get("world_to_grid"))
    for i, (x, y) in enumerate(path_w):                    # Each waypoint
        r, c = W2G(x, y, cs.shape)
        ok = is_free_with_neighbors(cs, r, c, p, check_neighbors=p.check_neighbor_safety)
        if not ok:
            plan_logger.Error(f"Invalid waypoint {i} at ({x:.2f}, {y:.2f})")
            return False
    for i in range(len(path_w) - 1):                       # Each segment
        r0, c0 = W2G(*path_w[i], cs.shape)
        r1, c1 = W2G(*path_w[i + 1], cs.shape)
        if not has_line_of_sight((r0, c0), (r1, c1), cs, p):
            plan_logger.Error(f"No LoS between waypoints {i} and {i + 1}")
            return False
    return True

def visualize_path_on_map(cs, path_w, save_path="path_visualization.npy"):
    vis = cs.copy()                                        # Copy for draw
    W2G = blackboard.Get("world_to_grid")
    for x, y in path_w:
        r, c = W2G(x, y, cs.shape)
        if Inb(cs.shape[0], cs.shape[1], r, c):
            vis[r, c] = 2.0                                # Mark path cells
    pth = RESMAP(save_path)                                # Safe path
    ENSURE(pth)                                            # Make folders
    np.save(pth, vis)                                      # Save array
    return vis

###############################################################################
# --- BT node -----------------------------------------------------------------
###############################################################################
# Behavior tree nodes for path planning
# Pick goals run planner and publish results
class MultiGoalPlannerBT(BehaviorNode):
    def __init__(self, params: PlanningParams, goals_key="navigation_goals",
                 path_key="planned_path", bb=None):
        super().__init__("MultiGoalPlanner")
        self.params = params                               # Planning params
        self.goals_key = goals_key                         # Goals key
        self.path_key = path_key                           # Path key
        self.bb = bb or blackboard                         # Blackboard
        self.params.W2G = self.bb.Get("world_to_grid")     # Inject maps
        self.params.G2W = self.bb.Get("grid_to_world")

    def reset(self):
        super().reset()
        self.bb.Set(self.path_key, None)                   # Clear path

    def execute(self):
        try:
            gps = self.bb.GetGps()                         # Need pose
            goals = self.bb.Get(self.goals_key)            # Goal list
            cs = self.bb.GetCspace()                       # Map grid
            if not gps:
                plan_logger.Error("GPS not available")
                return Status.FAILURE
            if not goals:
                plan_logger.Error("No goals set")
                return Status.FAILURE
            if cs is None:
                plan_logger.Error("No cspace")
                return Status.FAILURE
            x, y = gps.getValues()[:2]                     # Current xy
            goals_sorted = sorted(goals, key=lambda pxy: np.hypot(pxy[0] - x, pxy[1] - y))  # Nearest first
            successful_paths = 0
            failed_goals = []
            for i, g in enumerate(goals_sorted):           # Try each goal
                try:
                    seg = plan_path(cs, (x, y), g, self.params, smooth=True)
                except Exception as e:
                    failed_goals.append((g, f"Exception: {e}"))
                    continue
                if not seg:
                    failed_goals.append((g, "Empty path"))
                    continue
                successful_paths += 1
                if self.params.path_validation_enabled and (not validate_path(seg, cs, self.params)):
                    continue
                self.bb.Set(self.path_key, seg)            # Publish path
                return Status.SUCCESS
            # All goals failed give summary
            plan_logger.Error(f"Planning failed - no candidate goal yielded a valid path ({successful_paths}/{len(goals_sorted)} goals had valid paths)")
            return Status.FAILURE
        except Exception as e:
            plan_logger.Error(f"Exception: {e}")
            return Status.FAILURE

###############################################################################
# ------------------------- Behavior Tree Nodes ------------------------------
###############################################################################
# Planner side BT nodes for goals and checks
# Glue between map goals and path logic
class SetTwoGoals(BehaviorNode):
    def __init__(self, goals=None, num_goals=2, use_outer_perimeter=False):
        super().__init__("SetTwoGoals")
        self.goals = goals                                 # Pre set goals
        self.num_goals = num_goals                         # How many to pick
        self.use_outer_perimeter = use_outer_perimeter     # Use outer loop

    def execute(self):
        if self.goals:                                     # Caller provided
            blackboard.SetNavigationGoals(self.goals)
            return Status.SUCCESS
        cspace = GetFromBlackboard("cspace")               # Need map
        if cspace is None:
            main_logger.Error("SetTwoGoals: c-space not available yet.")
            return Status.FAILURE
        gps = GetFromBlackboard("gps")                     # Current pose
        curr = gps.getValues()[:2] if gps else None
        if self.use_outer_perimeter:                       # Use map loop
            # Import helpers from main
            from main import OrderFromStart, BuildPerimeterLoop, ClampGoal
            start_xy = GetFromBlackboard("start_xy", curr) # Start or now
            outer_perim = OrderFromStart(BuildPerimeterLoop(), start_xy, close_loop=False)
            outer_perim = [ClampGoal(x, y, cspace) for x, y in outer_perim]
            if len(outer_perim) >= 4:
                idx1 = len(outer_perim) // 4               # Opposite spots
                idx2 = (3 * len(outer_perim)) // 4
                goals = [outer_perim[idx1], outer_perim[idx2]]
            else:
                goals = outer_perim[:2] if len(outer_perim) >= 2 else outer_perim
            if len(goals) < 2:
                main_logger.Error(f"Not enough waypoints ({len(goals)})")
                return Status.FAILURE
            blackboard.SetNavigationGoals(goals)
            return Status.SUCCESS
        pp = PlanningParams()                               # Temp params
        pp.W2G = GetFromBlackboard("world_to_grid")
        pp.G2W = GetFromBlackboard("grid_to_world")
        safe_goals = find_safe_positions(                   # Sample safe goals
            cspace,
            pp,
            num_positions=self.num_goals,
            restrict_to_reachable_from=curr,
            check_neighbors=True
        )
        if not safe_goals or len(safe_goals) < self.num_goals:
            main_logger.Error(f"SetTwoGoals: found {len(safe_goals) if safe_goals else 0} safe goals (need {self.num_goals}).")
            return Status.FAILURE
        blackboard.SetNavigationGoals(safe_goals)           # Publish picks
        return Status.SUCCESS

class ValidateAndVisualizeWaypoints(BehaviorNode):
    def __init__(self):
        super().__init__("ValidateAndVisualizeWaypoints")
        self.done = False                                   # Only once

    def execute(self):
        if self.done:
            return Status.SUCCESS
        cspace = GetFromBlackboard("cspace")               # Need map grid
        if cspace is None:
            return Status.RUNNING
        # Create waypoint sets using main helpers
        try:
            from main import BuildEllipsePoints, BuildPerimeterLoop, OrderFromStart, ClampGoal
            start_xy = GetFromBlackboard("start_xy")       # Start point
            if start_xy:
                inner_12 = OrderFromStart(BuildEllipsePoints(), start_xy, close_loop=True)
                outer_perim = OrderFromStart(BuildPerimeterLoop(), start_xy, close_loop=True)
                # Clamp inside safe area
                inner_12 = [ClampGoal(x, y, cspace) for x, y in inner_12]
                outer_perim = [ClampGoal(x, y, cspace) for x, y in outer_perim]
                # Validate and draw
                for name, path in [("INNER_12", inner_12), ("OUTER_PERIM", outer_perim)]:
                    try:
                        if validate_path(path, cspace):     # Check geometry
                            visualize_path_on_map(cspace, path, save_path=f"{name}_viz.npy")
                    except Exception:
                        pass                                 # Ignore viz errors
        except Exception as e:
            main_logger.Warning(f"Could not create waypoints: {e}")
        self.done = True
        return Status.SUCCESS
