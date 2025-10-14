from __future__ import annotations
import numpy as np
import heapq
from collections import defaultdict
from typing import List, Tuple, Optional

from core import (
    BehaviorNode, Status, blackboard, plan_logger,
    PlanningParams,
    ResolveMapPath as RESMAP, EnsureParentDirectories as ENSURE
)

###############################################################################
# --- Utilities ---------------------------------------------------------------
###############################################################################
def Inb(h, w, r, c):
    return (0 <= r < h) and (0 <= c < w)

def heuristic(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))

def Thr(t, p: PlanningParams):
    if t is None:
        return p.th_free_planner
    return t

def is_free(cs, r, c, p: PlanningParams, thr=None):
    h, w = cs.shape
    if not Inb(h, w, r, c):
        return False
    return cs[r, c] > Thr(thr, p)

def is_free_with_neighbors(cs, r, c, p: PlanningParams, threshold=None, check_neighbors=True):
    if not is_free(cs, r, c, p, threshold):
        return False
    if not check_neighbors:
        return True
    h, w = cs.shape
    t = Thr(threshold, p)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if Inb(h, w, rr, cc) and cs[rr, cc] <= t:
                return False
    return True

def NearestFreeCell(cs, r, c, p: PlanningParams, threshold=None, max_radius=12, check_neighbors=True):
    if is_free_with_neighbors(cs, r, c, p, threshold, check_neighbors):
        return (r, c)
    h, w = cs.shape
    for R in range(1, max_radius + 1):
        for dr in range(-R, R + 1):
            for dc in range(-R, R + 1):
                if max(abs(dr), abs(dc)) != R:
                    continue
                rr = r + dr
                cc = c + dc
                ok = Inb(h, w, rr, cc)
                if ok and is_free_with_neighbors(cs, rr, cc, p, threshold, check_neighbors):
                    return (rr, cc)
    return None

def Neighbors(r, c, cs, p: PlanningParams):
    h, w = cs.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if Inb(h, w, rr, cc) and cs[rr, cc] > p.th_free_planner:
                move_cost = p.sqrt_2 if (dr != 0 and dc != 0) else 1.0
                yield (rr, cc, move_cost)

def Reconstruct(came, goal):
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = came[cur]
    path.reverse()
    return path

###############################################################################
# --- Jump Point Search -------------------------------------------------------
###############################################################################
def Forced(r, c, dr, dc, cs, p):
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

def IsJp(r, c, dr, dc, cs, p):
    if dr != 0 and dc != 0:
        return Forced(r, c, dr, 0, cs, p) or Forced(r, c, 0, dc, cs, p)
    diags = ((1, 1), (1, -1), (-1, 1), (-1, -1))
    for u, v in diags:
        if Forced(r, c, u, v, cs, p):
            return True
    return False

def Jump(r, c, dr, dc, cs, p):
    h, w = cs.shape
    sr = r
    sc = c
    cap = min(h, w) // 2
    while True:
        r += dr
        c += dc
        if (not Inb(h, w, r, c)) or cs[r, c] <= p.th_free_planner:
            return (None, None)
        if IsJp(r, c, dr, dc, cs, p):
            return (r, c)
        if abs(r - sr) > cap or abs(c - sc) > cap:
            return (r, c)

def JpsNbrs(r, c, cs, goal, p):
    out = []
    gr, gc = goal
    dr = (gr > r) - (gr < r)
    dc = (gc < c) - (gc > c)
    dirs = ((dr, -dc), (dr, 0), (0, -dc), (-dr, -dc), (dr, dc), (-dr, 0), (0, dc), (-dr, dc))
    for rr, cc in dirs:
        if rr == 0 and cc == 0:
            continue
        jr, jc = Jump(r, c, rr, cc, cs, p)
        if jr is not None:
            cost = p.sqrt_2 if (rr != 0 and cc != 0) else 1.0
            out.append((jr, jc, cost))
    return out

###############################################################################
# --- A* variants -------------------------------------------------------------
###############################################################################
def Astar(cs, start, goal, max_it, max_open, h_w, use_jps, early_exit, p: PlanningParams):
    openq = [(heuristic(start, goal) * h_w, 0.0, start, None)]
    came = {}
    g = defaultdict(lambda: float("inf"))
    g[start] = 0.0
    it = 0
    while openq and it < max_it:
        it += 1
        if len(openq) > max_open:
            return []
        f, gg, cur, parent = heapq.heappop(openq)
        if gg > early_exit:
            return []
        if cur in came:
            continue
        came[cur] = parent
        if cur == goal:
            break
        if use_jps:
            nbrs = JpsNbrs(cur[0], cur[1], cs, goal, p)
        else:
            nbrs = Neighbors(cur[0], cur[1], cs, p)
        for nr, nc, mc in nbrs:
            nxt = (nr, nc)
            ng = gg + mc
            if nxt in came:
                continue
            if ng >= g[nxt]:
                continue
            g[nxt] = ng
            h = heuristic(nxt, goal) * h_w
            heapq.heappush(openq, (ng + h, ng, nxt, cur))
    if goal in came:
        return Reconstruct(came, goal)
    return []

def BiAstar(cs, start, goal, max_it, max_open, h_w, p: PlanningParams):
    f_open = [(heuristic(start, goal) * h_w, 0.0, start, None)]
    b_open = [(heuristic(goal, start) * h_w, 0.0, goal, None)]
    f_c, b_c = {}, {}
    f_g = defaultdict(lambda: float("inf"))
    b_g = defaultdict(lambda: float("inf"))
    f_g[start] = 0.0
    b_g[goal] = 0.0
    it = 0
    meet = None
    while (f_open or b_open) and it < max_it:
        it += 1
        if (it % 2) == 1 and f_open:
            q = f_open
            close = f_c
            gcost = f_g
            tgt = goal
            def hx(n): return heuristic(n, goal) * h_w
        else:
            q = b_open
            close = b_c
            gcost = b_g
            tgt = start
            def hx(n): return heuristic(n, start) * h_w
        _, gval, cur, par = heapq.heappop(q)
        if cur in close:
            continue
        close[cur] = par
        if cur in (b_c if close is f_c else f_c):
            meet = cur
            break
        for nr, nc, mc in Neighbors(cur[0], cur[1], cs, p):
            nxt = (nr, nc)
            ng = gval + mc
            if nxt in close:
                continue
            if ng >= gcost[nxt]:
                continue
            gcost[nxt] = ng
            heapq.heappush(q, (ng + hx(nxt), ng, nxt, cur))
        if len(f_open) + len(b_open) > max_open:
            return []
    if meet is None:
        return []
    f_path = []
    cur = meet
    while cur is not None:
        f_path.append(cur)
        cur = f_c.get(cur)
    f_path.reverse()
    b_path = []
    cur = meet
    while cur is not None:
        b_path.append(cur)
        cur = b_c.get(cur)
    return f_path + b_path[1:]

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
    dist = heuristic(start, goal)
    mindim = min(cs.shape)
    use_jps = bool(p.jump_point_search and (dist >= mindim * 0.15))
    if p.bidirectional and dist > mindim * 0.8:
        return BiAstar(cs, start, goal, max_it, max_open, h_w, p)
    early = dist * p.early_exit_multiplier
    return Astar(cs, start, goal, max_it, max_open, h_w, use_jps, early, p)


###############################################################################
# --- Path smoothing ----------------------------------------------------------
###############################################################################
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
        if not is_free(cs, r, c, p):
            return False
        if (r, c) == (r1, c1):
            return True
        e2 = err << 1
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc

def smooth_path(path, cs, p: PlanningParams, look=15):
    if len(path) <= 2:
        return path
    out = [path[0]]
    i = 0
    n = len(path)
    while i < n - 1:
        far = i + 1
        upto = min(n, i + look + 1)
        j = i + 2
        while j < upto:
            if has_line_of_sight(path[i], path[j], cs, p):
                far = j
                j += 1
            else:
                break
        out.append(path[far])
        i = far
    return out


###############################################################################
# --- Differential drive optimization -----------------------------------------
###############################################################################
def Collinear(a, b, c, p: PlanningParams):
    cp = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    d = np.hypot(c[0] - a[0], c[1] - a[1])
    if d < 1e-6:
        return True
    return (cp / d) < p.differential_drive_alignment_tolerance

def AlignCardinal(pt, prev, nxt, p: PlanningParams):
    dx = nxt[0] - prev[0]
    dy = nxt[1] - prev[1]
    L = np.hypot(dx, dy)
    if L < 1e-6:
        return pt
    dx /= L
    dy /= L
    s = 1 / np.sqrt(2.0)
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (s, s), (-s, s), (s, -s), (-s, -s)]
    best_i = 0
    best_dot = -1e9
    for i, (ux, uy) in enumerate(dirs):
        dot = dx * ux + dy * uy
        if dot > best_dot:
            best_dot = dot
            best_i = i
    ux, uy = dirs[best_i]
    ang = np.degrees(np.arccos(np.clip(ux * dx + uy * dy, -1, 1)))
    if ang < p.differential_drive_angle_tolerance:
        dist = (pt[0] - prev[0]) * ux + (pt[1] - prev[1]) * uy
        h, w = p.default_map_shape
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
    for i in range(1, len(path) - 1):
        cur = path[i]
        prev = out[-1]
        nxt = path[i + 1]
        if not Collinear(prev, cur, nxt, p):
            cur = AlignCardinal(cur, prev, nxt, p)
        out.append(cur)
    out.append(path[-1])
    return out

def Downsample(pth, cap=40):
    if len(pth) <= cap:
        return pth
    step = max(1, len(pth) // cap)
    slim = pth[::step]
    if slim[-1] != pth[-1]:
        slim = slim + [pth[-1]]
    return slim


###############################################################################
# --- Adaptive planning -------------------------------------------------------
###############################################################################
def AdaptiveParams(cs, start, goal, p: PlanningParams):
    free = int((cs > p.th_free_planner).sum())
    total = cs.size
    occ_pct = (100.0 * free / total)
    dense = (100 - occ_pct) > 40
    sparse = (100 - occ_pct) < 10
    dist = heuristic(start, goal)
    out = {
        "max_iterations": p.max_iterations,
        "max_open_set_size": p.max_open_set_size,
        "heuristic_weight": p.heuristic_weight
    }
    if dense:
        out["max_iterations"] = int(p.max_iterations * 0.7)
        out["max_open_set_size"] = int(p.max_open_set_size * 0.8)
        out["heuristic_weight"] = 1.1
    elif sparse:
        out["max_iterations"] = int(p.max_iterations * 1.5)
        out["heuristic_weight"] = 0.9
    if dist > min(cs.shape) * 0.5:
        out["heuristic_weight"] = min(out["heuristic_weight"] * 1.2, 1.5)
    return out

def adaptive_a_star(cs, start, goal, p: PlanningParams):
    tries = [
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
        plan_logger.Info(f"Trying adaptive_a_star attempt {i+1}/3 with params: {kw}")
        path = a_star(cs, start, goal, p, **kw)
        if path:
            plan_logger.Info(f"Adaptive A* succeeded on attempt {i+1} with {len(path)} waypoints")
            return path
        else:
            plan_logger.Warning(f"Adaptive A* attempt {i+1} failed")
    plan_logger.Error("All adaptive A* attempts failed")
    return []

###############################################################################
# --- Public API --------------------------------------------------------------
###############################################################################
def plan_path(cs, start_w, goal_w, p: PlanningParams, smooth=True,
              optimize_for_differential_drive=True, check_neighbors=None,
              max_search_radius=None, **_):
    if check_neighbors is None:
        check_neighbors = p.check_neighbor_safety
    if max_search_radius is None:
        max_search_radius = p.safe_waypoint_search_radius
    W2G = getattr(p, 'W2G', blackboard.Get("world_to_grid"))
    G2W = getattr(p, 'G2W', blackboard.Get("grid_to_world"))
    sr, sc = W2G(*start_w, cs.shape)
    gr, gc = W2G(*goal_w, cs.shape)
    ns = NearestFreeCell(cs, sr, sc, p, max_radius=max_search_radius, check_neighbors=check_neighbors)
    if ns is None:
        plan_logger.Error(f"No safe start found at ({sr}, {sc})")
        return []
    sr, sc = ns
    plan_logger.Info(f"Found safe start at ({sr}, {sc})")
    ng = NearestFreeCell(cs, gr, gc, p, max_radius=max_search_radius, check_neighbors=check_neighbors)
    if ng is None:
        plan_logger.Error(f"No safe goal found at ({gr}, {gc})")
        return []
    gr, gc = ng
    plan_logger.Info(f"Found safe goal at ({gr}, {gc})")
    if has_line_of_sight((sr, sc), (gr, gc), cs, p):
        return [start_w, goal_w]
    path = adaptive_a_star(cs, (sr, sc), (gr, gc), p)
    if not path:
        plan_logger.Error(f"Planning failed from ({sr}, {sc}) to ({gr}, {gc})")
        return []
    plan_logger.Info(f"Planning succeeded with {len(path)} waypoints")
    if optimize_for_differential_drive and p.optimize_for_differential_drive and len(path) > 2:
        path = optimize_path_for_differential_drive(path, cs, p)
    if smooth and len(path) > 2:
        path = smooth_path(path, cs, p)
    path = Downsample(path, cap=40)
    out = []
    for r, c in path:
        out.append(G2W(r, c))
    if out[-1] != goal_w:
        out.append(goal_w)
    return out

###############################################################################
# --- Reachability analysis ---------------------------------------------------
###############################################################################
def reachable_mask_from(cs, start_world, p: PlanningParams, threshold=None, connectivity=8):
    if start_world is None:
        return None
    W2G = getattr(p, 'W2G', blackboard.Get("world_to_grid"))
    r0, c0 = W2G(*start_world, cs.shape)
    t = Thr(threshold, p)
    h, w = cs.shape
    if (not Inb(h, w, r0, c0)) or cs[r0, c0] <= t:
        near = NearestFreeCell(cs, r0, c0, p, threshold=t, max_radius=6, check_neighbors=False)
        if near is None:
            return np.zeros((h, w), dtype=bool)
        r0, c0 = near
    vis = np.zeros((h, w), dtype=bool)
    stack = [(r0, c0)]
    while stack:
        r, c = stack.pop()
        if (not Inb(h, w, r, c)) or vis[r, c] or cs[r, c] <= t:
            continue
        vis[r, c] = True
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if connectivity == 4 and dr != 0 and dc != 0:
                    continue
                stack.append((r + dr, c + dc))
    return vis

def find_safe_positions(cs, p: PlanningParams, num_positions=10, threshold=None,
                        restrict_to_reachable_from=None, check_neighbors=True, max_attempts=None):
    out = []
    h, w = cs.shape
    t = min(1.0, Thr(threshold, p) + 0.05)
    if max_attempts is None:
        max_attempts = num_positions * 20
    if restrict_to_reachable_from is not None:
        reach = reachable_mask_from(cs, restrict_to_reachable_from, p, threshold=t)
    else:
        reach = None
    G2W = getattr(p, 'G2W', blackboard.Get("grid_to_world"))
    tries = 0
    while tries < max_attempts:
        tries += 1
        r = np.random.randint(10, h - 10)
        c = np.random.randint(10, w - 10)
        if reach is not None and not reach[r, c]:
            continue
        if cs[r, c] > t and is_free_with_neighbors(cs, r, c, p, threshold=t, check_neighbors=check_neighbors):
            xw, yw = G2W(r, c)
            too_close = False
            for ex, ey in out:
                if np.hypot(xw - ex, yw - ey) < 0.5:
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
def validate_path(path_w, cs, p: PlanningParams):
    if len(path_w) < 2:
        return True
    W2G = getattr(p, 'W2G', blackboard.Get("world_to_grid"))
    for i, (x, y) in enumerate(path_w):
        r, c = W2G(x, y, cs.shape)
        ok = is_free_with_neighbors(cs, r, c, p, check_neighbors=p.check_neighbor_safety)
        if not ok:
            plan_logger.Error(f"Invalid waypoint {i} at ({x:.2f}, {y:.2f})")
            return False
    for i in range(len(path_w) - 1):
        r0, c0 = W2G(*path_w[i], cs.shape)
        r1, c1 = W2G(*path_w[i + 1], cs.shape)
        if not has_line_of_sight((r0, c0), (r1, c1), cs, p):
            plan_logger.Error(f"No LoS between waypoints {i} and {i + 1}")
            return False
    return True

def visualize_path_on_map(cs, path_w, save_path="path_visualization.npy"):
    vis = cs.copy()
    W2G = blackboard.Get("world_to_grid")
    for x, y in path_w:
        r, c = W2G(x, y, cs.shape)
        if Inb(cs.shape[0], cs.shape[1], r, c):
            vis[r, c] = 2.0
    pth = RESMAP(save_path)
    ENSURE(pth)
    np.save(pth, vis)
    return vis


###############################################################################
# --- BT node -----------------------------------------------------------------
###############################################################################
class MultiGoalPlannerBT(BehaviorNode):
    def __init__(self, params: PlanningParams, goals_key="navigation_goals",
                 path_key="planned_path", bb=None):
        super().__init__("MultiGoalPlanner")
        self.params = params
        self.goals_key = goals_key
        self.path_key = path_key
        self.bb = bb or blackboard
        self.params.W2G = self.bb.Get("world_to_grid")
        self.params.G2W = self.bb.Get("grid_to_world")

    def reset(self):
        super().reset()
        self.bb.Set(self.path_key, None)

    def execute(self):
        try:
            gps = self.bb.GetGps()
            goals = self.bb.Get(self.goals_key)
            cs = self.bb.GetCspace()
            if not gps:
                plan_logger.Error("GPS not available")
                return Status.FAILURE
            if not goals:
                plan_logger.Error("No goals set")
                return Status.FAILURE
            if cs is None:
                plan_logger.Error("No cspace")
                return Status.FAILURE
            x, y = gps.getValues()[:2]
            plan_logger.Info(f"Planning with {len(goals)} goals")
            if self.params.verbose:
                for i, (gx, gy) in enumerate(goals[:3]):
                    plan_logger.Info(f" Goal {i + 1}: ({gx:.3f}, {gy:.3f})")
                if len(goals) > 3:
                    plan_logger.Info(f" ... and {len(goals) - 3} more goals")
            goals_sorted = sorted(goals, key=lambda pxy: np.hypot(pxy[0] - x, pxy[1] - y))
            for g in goals_sorted:
                try:
                    seg = plan_path(cs, (x, y), g, self.params, smooth=True)
                except Exception:
                    continue
                if not seg:
                    continue
                if self.params.path_validation_enabled and (not validate_path(seg, cs, self.params)):
                    continue
                self.bb.Set(self.path_key, seg)
                plan_logger.Info(f"Planned path with {len(seg)} waypoints.")
                if self.params.verbose:
                    plan_logger.Info(f" Path start: ({seg[0][0]:.3f},{seg[0][1]:.3f}) → end: ({seg[-1][0]:.3f},{seg[-1][1]:.3f})")
                return Status.SUCCESS
            plan_logger.Error("Planning failed - no candidate goal yielded a valid path.")
            return Status.FAILURE
        except Exception as e:
            plan_logger.Error(f"Exception: {e}")
            return Status.FAILURE
