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

