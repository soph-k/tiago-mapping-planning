from __future__ import annotations                                              
import numpy as np                                                              # Numpy arrays for fast math
import heapq                                                                    # Binary heap structure powering A* open set operations
from collections import defaultdict                                             # Dict with default factory

from core import (                                                              # ProjectB local
     blackboard,  plan_logger,                                                  # Shared state handle and planner-scoped logger
    PlanningParams,                                                             # Planner thresholds, caps, and feature toggles
    ResolveMapPath as RESMAP, EnsureParentDirectories as ENSURE,                # Safe filesystem utils 
    GridToWorld,                                                                # Grid TO world conversion helper for bounds and outputs
)                                                                               

###############################################################################  
# --- Utilities ---------------------------------------------------------------  
###############################################################################  
# Lightweight helpers used throughout the planner. 
# Centralize bounds checks, thresholds, and basic geometry.

def Inb(h, w, r, c):                                                            # Inside grid bounds
    return (0 <= r < h) and (0 <= c < w)                                        # Return True only when both indices are valid

def heuristic(a, b):                                                            
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))                            # Straight-line distance in grid units

def Thr(t, p: PlanningParams):                                                  # Resolve threshold against planner default
    if t is None:                                                               # If caller threshold not there, fall back cleanly
        return p.th_free_planner                                                # Use configured free cutoff for comparisons
    return t                                                                    # Otherwise override

def is_free(cs, r, c, p: PlanningParams, thr=None):                             
    h, w = cs.shape                                                             # Cache shape to avoid repeats
    if not Inb(h, w, r, c):                                                     # Reject out-of-bounds indices 
        return False                                                            # Treat OOB as blocked to stay safe
    return cs[r, c] > Thr(thr, p)                                               # Compare cell value against  threshold

def is_free_with_neighbors(cs, r, c, p: PlanningParams, threshold=None, check_neighbors=True):  
    if not is_free(cs, r, c, p, threshold):                                     # Center must be free before scanning neighbors
        return False                                                            # Early exit keeps loops tight
    if not check_neighbors:                                                     
        return True                                                             # Accept center cell
    h, w = cs.shape                                                          
    t = Thr(threshold, p)                                                       # Effective threshold used across the neighborhood
    for dr in (-1, 0, 1):                                                       # Iterate neighbor row 
        for dc in (-1, 0, 1):                                                   # Iterate neighbor column 
            if dr == 0 and dc == 0:                                             # Skip the center; we checked it already
                continue
            rr, cc = r + dr, c + dc                                             # Compute neighbor indices relative to center
            if Inb(h, w, rr, cc) and cs[rr, cc] <= t:                           # Any at or below threshold is considered unsafe
                return False                                                    # Reject 
    return True                                                                 # All neighbors acceptable 


###############################################################################  
# --- Waypoint Generation -----------------------------------------------------  
###############################################################################  
# Quick builders for survey sets in world coordinates. 
def WorldBoundsFromConfig(shape=(200, 300)):                                    
    h, w = shape                                                                # Grid rows and cols used by conversion
    x_min, y_max = GridToWorld(0, 0)                                            # Top left corner in world frame
    x_max, y_min = GridToWorld(h - 1, w - 1)                                    # Bottom right corner in world frame
    return x_min, y_min, x_max, y_max                                           # Return tuple

def BuildEllipsePoints(center=(-0.65, -1.43), rx=1.05, ry=1.25, num_points=12, rotation=0.0):  # Evenly spaced ellipse
    cx, cy = center                                                             # World space center for generated shape
    ang = np.linspace(0, 2 * np.pi, num_points, endpoint=False) + rotation      
    return [(cx + rx * np.cos(a), cy + ry * np.sin(a)) for a in ang]            

def BuildPerimeterLoop(margin=None, include_midpoints=True):                    
    x_min, y_min, x_max, y_max = WorldBoundsFromConfig()                        
    m = margin if margin is not None else max(0.6, 0.18 + 0.4)                  # Offset to avoid hugging the walls
    left, right  = x_min + m, x_max - m                                         # X interval 
    bottom, top  = y_min + m, y_max - m                                         # Y interval 
    mid_x, mid_y = 0.5 * (left + right), 0.5 * (bottom + top)                   # Midpoints
    pts = [(left, bottom)]                                                      
    if include_midpoints: pts += [(mid_x, bottom)]                              # Add midpoint on the bottom edge if requested
    pts += [(right, bottom)]
    if include_midpoints: pts += [(right, mid_y)]                               # Add midpoint on the right edge 
    pts += [(right, top)]
    if include_midpoints: pts += [(mid_x, top)]                                 # Add midpoint on the top edge 
    pts += [(left, top)]
    if include_midpoints: pts += [(left, mid_y)]                                # Add midpoint on the left edge before closing
    return pts                                                                  # Ordered loop in world coordinates

def OrderFromStart(points, start_position, close_loop=True):                    # Rotate sequence so nearest becomes first
    sx, sy = start_position                                                     # Reference origin for distance computate
    d = [np.hypot(x - sx, y - sy) for (x, y) in points]                         # Distances from start to all
    i = int(np.argmin(d))                                                       # Index of closest waypoint improves first point
    ordered = points[i:] + points[:i]                                           # Rotate list so the nearest point is first
    return ordered + [start_position] if close_loop else ordered                

def ClampGoal(goal_x, goal_y, cspace=None):                                     # Keep goal inside world bounds
    if cspace is None:                                                          # No map implies no reliable bounds
        return goal_x, goal_y                                                   # Pass through when clamping not possible
    h, w = cspace.shape                                                         
    x_min, y_min = GridToWorld(h - 1, 0)                                        # World min corner from bottom left grid
    x_max, y_max = GridToWorld(0, w - 1)                                        # World max corner from top-right grid
    buf = 0.20                                                                  # Small margin to avoid edge
    return (
        min(max(goal_x, x_min + buf), x_max - buf),                             # Clamp x 
        min(max(goal_y, y_min + buf), y_max - buf),                             # Clamp y 
    )

def NearestFreeCell(cs, r, c, p: PlanningParams, threshold=None, max_radius=12, check_neighbors=True):  # Snap to safe cell
    if is_free_with_neighbors(cs, r, c, p, threshold, check_neighbors):         
        return (r, c)                                                           # No movement needed; reuse cell
    h, w = cs.shape                                                             # Cache for bounds checks 
    for R in range(1, max_radius + 1):                                          
        for dr in range(-R, R + 1):                                             # Iterate candidate rows
            for dc in range(-R, R + 1):                                         # Iterate candidate cols
                if max(abs(dr), abs(dc)) != R:                                 
                    continue
                rr, cc = r + dr, c + dc                                         # Candidate cell relative to origin
                if Inb(h, w, rr, cc) and is_free_with_neighbors(cs, rr, cc, p, threshold, check_neighbors):
                    return (rr, cc)                                             # First safe cell found on expansion
    return None                                                                 # Not found within the search radius

def Neighbors(r, c, cs, p: PlanningParams):                                     
    h, w = cs.shape                                                             # Dimensions for bounds
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:                                             # Skip center; not a neighbor
                continue
            rr, cc = r + dr, c + dc                                             # Neighbor indices
            if Inb(h, w, rr, cc) and cs[rr, cc] > p.th_free_planner:            # Accept only free cells
                move_cost = p.sqrt_2 if (dr != 0 and dc != 0) else 1.0          
                yield (rr, cc, move_cost)                                       # Remove neighbor with cost for A*

def Reconstruct(came, goal):                                                    # Follow parent pointers to build path
    path, cur = [], goal                                                        # Start at goal node and walk backward
    while cur is not None:                                                     
        path.append(cur); cur = came[cur]                                       # Append then move 
    path.reverse()                                                              # Convert to start to goal order 
    return path                                                                 # Final node list in grid coordinates


###############################################################################  
# --- Jump Point Search -------------------------------------------------------  
###############################################################################  
# JPS straight runs, keeping only forced turns and goals. 
def Forced(r, c, dr, dc, cs, p):                                                # Detect forced neighbor
    h, w = cs.shape                                                             # Grid size for safe indexing
    nr, nc = r + dr, c + dc                                                     # Next cell after one step along direction
    if not Inb(h, w, nr, nc) or cs[nr, nc] <= p.th_free_planner:                # If step is invalid or blocked
        return False                                                            
    checks = ((nr + dr, nc), (nr, nc + dc), (nr + dr, nc + dc))                 # Imply a forced turn
    for rr, cc in checks:                                                       # Examine neighbors around the stepped cell
        if Inb(h, w, rr, cc) and cs[rr, cc] <= p.th_free_planner:               # Obstacle near path creates forced scenario
            return True                                                         
    return False                                                                # No forced neighbors detected

def IsJp(r, c, dr, dc, cs, p):                                                  # Determine if current cell is a jump point
    if dr != 0 and dc != 0:                                                     # Diagonal motion case
        return Forced(r, c, dr, 0, cs, p) or Forced(r, c, 0, dc, cs, p)         
    for u, v in ((1,1),(1,-1),(-1,1),(-1,-1)):                                  # Checks diagonal forced turns
        if Forced(r, c, u, v, cs, p):                                           
            return True
    return False                                                                # Not considered a jump point

def Jump(r, c, dr, dc, cs, p):                                                  # Walk along direction until JP, obstacle, or cap
    h, w = cs.shape; sr, sc = r, c                                              # Bounds plus start cell for distance cap
    cap = min(h, w) // 2                                                        
    while True:
        r += dr; c += dc                                                        # Advance one cell in the chosen direction
        if (not Inb(h, w, r, c)) or cs[r, c] <= p.th_free_planner:              # Out of bounds or hit obstacle
            return (None, None)                                                 # Terminate without a JP
        if IsJp(r, c, dr, dc, cs, p):                                           # Hit a valid jump point under this direction
            return (r, c)                                                       # Return JP coordinates
        if abs(r - sr) > cap or abs(c - sc) > cap:                              # Traveled too far without JP
            return (r, c)                                                       # Bail with last valid cell

def JpsNbrs(r, c, cs, goal, p):                                                 # Compute pruned neighbors via JPS
    out = []; gr, gc = goal                                                     # Goal indices
    dr = (gr > r) - (gr < r)                                                    # Row sign toward goal
    dc = (gc < c) - (gc > c)                                                    # Column sign toward goal
    dirs = ((dr,-dc),(dr,0),(0,-dc),(-dr,-dc),(dr,dc),(-dr,0),(0,dc),(-dr,dc))  # Exploration order 
    for rr, cc in dirs:
        if rr == 0 and cc == 0:                                                 
            continue
        jr, jc = Jump(r, c, rr, cc, cs, p)                                      # Jump along that direction
        if jr is not None:                                                      # Valid target returned
            cost = p.sqrt_2 if (rr != 0 and cc != 0) else 1.0                   
            out.append((jr, jc, cost))                                          
    return out                                                                  # Return reduced neighbor set


###############################################################################  
# --- A* variants -------------------------------------------------------------  
###############################################################################  
# A* and a bidirectiona, support heuristics and cap on iterations and open set size for better performance.
def Astar(cs, start, goal, max_it, max_open, h_w, use_jps, early_exit, p: PlanningParams):  # Single direction A*
    openq = [(heuristic(start, goal) * h_w, 0.0, start, None)]                  
    came = {}                                                                   # Parent mapping 
    g = defaultdict(lambda: float("inf"))                                       
    g[start] = 0.0                                                              
    it = 0                                                                      # Iteration counter 
    while openq and it < max_it:                                                # Loop open set empty
        it += 1                                                                 # Step iteration counter
        if len(openq) > max_open:                                               # Abort if open set grows beyond cap
            return []                                                           # Treat as failure 
        f, gg, cur, parent = heapq.heappop(openq)                               # Pop node with smallest f-score
        if gg > early_exit:                                                     
            return []                                                           # Abort this attempt early
        if cur in came:                                                         # Skip if already closed
            continue
        came[cur] = parent                                                      # Record 
        if cur == goal:                                                         # Reached target cell
            break                                                               # Exit loop
        nbrs = JpsNbrs(cur[0], cur[1], cs, goal, p) if use_jps else Neighbors(cur[0], cur[1], cs, p)  # Choose neighbor 
        for nr, nc, mc in nbrs:                                                 # For each candidate neighbor
            nxt = (nr, nc); ng = gg + mc                                        # Compute next state and cost
            if nxt in came or ng >= g[nxt]:                                     # Ignore closed paths
                continue
            g[nxt] = ng; h = heuristic(nxt, goal) * h_w                         # Update cost and heuristic
            heapq.heappush(openq, (ng + h, ng, nxt, cur))                       # Push with updated f and parent
    if goal in came:                                                            # If goal was closed, a path exists
        return Reconstruct(came, goal)                                          # Build path by following parents
    return []                                                                   # Failure under current settings

def BiAstar(cs, start, goal, max_it, max_open, h_w, p: PlanningParams):         # Alternate forward/backward A*
    f_open = [(heuristic(start, goal) * h_w, 0.0, start, None)]                 # Forward 
    b_open = [(heuristic(goal, start) * h_w, 0.0, goal, None)]                  # Backward 
    f_c, b_c = {}, {}                                                           # Parent sets
    f_g = defaultdict(lambda: float("inf")); b_g = defaultdict(lambda: float("inf"))  # Cost 
    f_g[start] = 0.0; b_g[goal] = 0.0                                           # Initialize costs 
    it = 0; meet = None                                                         
    while (f_open or b_open) and it < max_it:                                   
        it += 1                                                                 # Flip side each step
        if (it % 2) == 1 and f_open:
            q, close, gcost = f_open, f_c, f_g                                  # Forward
            def hx(n): return heuristic(n, goal) * h_w                          # Heuristic toward goal
        else:
            q, close, gcost = b_open, b_c, b_g                                  # Go backward
            def hx(n): return heuristic(n, start) * h_w                         # Heuristic toward start
        _, gval, cur, par = heapq.heappop(q)                                    # Pop best on chosen 
        if cur in close:                                                        # Already closed 
            continue
        close[cur] = par                                                        # Record parent linkage
        if cur in (b_c if close is f_c else f_c):                               # Saw node on opposite side
            meet = cur; break                                                   # Stop expanding
        for nr, nc, mc in Neighbors(cur[0], cur[1], cs, p):                     # Expand neighbors
            nxt = (nr, nc); ng = gval + mc                                      # Candidate cost
            if nxt in close or ng >= gcost[nxt]:                                # Skip  
                continue
            gcost[nxt] = ng                                                     # Improve and queue
            heapq.heappush(q, (ng + hx(nxt), ng, nxt, cur))                     # Push with f, g, parent
        if len(f_open) + len(b_open) > max_open:                                
            return []                                                           # Failure
    if meet is None:                                                            
        return []                                                               # No path found
    f_path, cur = [], meet                                                      # Rebuild forward half
    while cur is not None: f_path.append(cur); cur = f_c.get(cur)
    f_path.reverse()
    b_path, cur = [], meet                                                      # Rebuild backward half
    while cur is not None: b_path.append(cur); cur = b_c.get(cur)
    return f_path + b_path[1:]                                                  

def a_star(cs, start, goal, p: PlanningParams, max_iterations=None, max_open_set_size=None, heuristic_weight=None):     # Config wrapper
    max_it   = max_iterations     if max_iterations     is not None else p.max_iterations
    max_open = max_open_set_size  if max_open_set_size  is not None else p.max_open_set_size
    h_w      = heuristic_weight   if heuristic_weight   is not None else p.heuristic_weight
    if not is_free(cs, start[0], start[1], p) or not is_free(cs, goal[0], goal[1], p):                                  # Validate endpoints
        plan_logger.Warning(f"A*: Start or goal not free - start: {is_free(cs, start[0], start[1], p)}, goal: {is_free(cs, goal[0], goal[1], p)}")
        return []                                                               # Abort early 
    if start == goal:                                                           # Already at goal
        plan_logger.Info("A*: Start equals goal, returning single point")
        return [start]                                                          # Single node path
    dist = heuristic(start, goal); mindim = min(cs.shape)                       # Scale for JPS and early exit
    use_jps = bool(p.jump_point_search and (dist >= mindim * 0.15))             # Use JPS on long runs
    if p.bidirectional and dist > mindim * 0.8:                                 # Prefer Bi-A* 
        return BiAstar(cs, start, goal, max_it, max_open, h_w, p)
    early = dist * p.early_exit_multiplier                                      # Early bound 
    return Astar(cs, start, goal, max_it, max_open, h_w, use_jps, early, p)     # Execute single-direction A*


###############################################################################  
# --- Path smoothing ----------------------------------------------------------  
###############################################################################  
# Cut redundant bends with line-of-sight checks. Keeps validity while reducing
# waypoint count and turn complexity.

def has_line_of_sight(a, b, cs, p: PlanningParams):                             # Bresenham-style with collision checks
    r0, c0 = a; r1, c1 = b                                                      # Endpoints
    dr, dc = abs(r1 - r0), abs(c1 - c0)                                         # Integer deltas
    sr = 1 if r0 < r1 else -1; sc = 1 if c0 < c1 else -1                        # Step directions for each axis
    err = dr - dc; r, c = r0, c0                                                # Initialize loop state
    while True:
        if not is_free(cs, r, c, p):                                            # Hit obstacle 
            return False                                                        # Not visible directly
        if (r, c) == (r1, c1):                                                  # Arrived at target cell
            return True                                                         # Segment is clear
        e2 = err << 1                                                           # Twice error for integer decision
        if e2 > -dc: err -= dc; r += sr                                         # Step row and adjust error
        if e2 <  dr: err += dr; c += sc                                         # Step column similarly

def smooth_path(path, cs, p: PlanningParams, look=15):                          
    if len(path) <= 2:                                                          # Nothing to simplify
        return path                                                             # Return unchanged
    out = [path[0]]; i = 0; n = len(path)                                       
    while i < n - 1:
        far = i + 1; upto = min(n, i + look + 1)                                
        j = i + 2                                                               # Start at least one forward
        while j < upto:
            if has_line_of_sight(path[i], path[j], cs, p):                      # Directly visible 
                far = j; j += 1                                                 # Extend shortcut forward
            else:
                break                                                           # Blocked; stop extending
        out.append(path[far]); i = far                                          # Commit and move
    return out                                                                  # Smoothed grid path


###############################################################################  
# --- Differential drive optimization -----------------------------------------  
############################################################################### 
# Make paths for drive motion and reduce intense heading changes before starting.
def Collinear(a, b, c, p: PlanningParams):                                      # Near-collinearity via area/length ratio
    cp = abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))                 # Twice triangle area
    d  = np.hypot(c[0]-a[0], c[1]-a[1])                                         # Baseline segment length
    if d < 1e-6:                                                                # Degenerate configuration
        return True                                                             # Treat as collinear to be safe
    return (cp / d) < p.differential_drive_alignment_tolerance                  # Thresholded ratio indicates near-straight

def AlignCardinal(pt, prev, nxt, p: PlanningParams):                            # Snap interior point toward axis/diag if close
    dx, dy = nxt[0]-prev[0], nxt[1]-prev[1]                                     # Spanning segment direction
    L = np.hypot(dx, dy)                                                        # Length for normalization
    if L < 1e-6:                                                                # Too small to reason about
        return pt                                                               # Leave untouched
    dx /= L; dy /= L                                                            # Unit direction vector
    s = 1 / np.sqrt(2.0)                                                        # Diagonal basis scaling
    dirs = [(1,0),(-1,0),(0,1),(0,-1),(s,s),(-s,s),(s,-s),(-s,-s)]              # Candidate basis directions
    best_i, best_dot = 0, -1e9                                                  # Track best-aligned basis
    for i, (ux, uy) in enumerate(dirs):
        dot = dx*ux + dy*uy                                                     # Alignment score with basis
        if dot > best_dot: best_dot, best_i = dot, i                            # Keep strongest
    ux, uy = dirs[best_i]                                                       # Best-fit direction
    ang = np.degrees(np.arccos(np.clip(ux*dx + uy*dy, -1, 1)))                  # Angular deviation from best
    if ang < p.differential_drive_angle_tolerance:                              # Only snap when sufficiently aligned
        dist = (pt[0]-prev[0])*ux + (pt[1]-prev[1])*uy                          # Project along basis
        h, w = p.default_map_shape                                              # Clamp to nominal map size
        rr = int(round(dist*ux + prev[0])); cc = int(round(dist*uy + prev[1]))  # Reconstruct snapped indices
        rr = max(0, min(h-1, rr)); cc = max(0, min(w-1, cc))                    # Clamp to bounds defensively
        return (rr, cc)                                                         # Return adjusted point
    return pt                                                                   # Outside tolerance; keep original

def optimize_path_for_differential_drive(path, cs, p: PlanningParams):          # Apply collinearity and snapping across path
    if len(path) <= 2:                                                          # Two points or less is trivial
        return path                                                             # Nothing to optimize
    out = [path[0]]                                                             # Always retain start
    for i in range(1, len(path) - 1):                                           # Iterate interior vertices
        cur, prev, nxt = path[i], out[-1], path[i+1]                            # Pull neighbors for testing
        if not Collinear(prev, cur, nxt, p):                                    # If noticeably bent here
            cur = AlignCardinal(cur, prev, nxt, p)                              # Try snapping to friendlier heading
        out.append(cur)                                                         # Append kept or modified waypoint
    out.append(path[-1])                                                        # Always retain end
    return out                                                                  # Optimized grid path

def Downsample(pth, cap=40):                                                    # Reduce waypoint count with uniform stride
    if len(pth) <= cap:                                                         # Already under limit
        return pth                                                              # No action needed
    step = max(1, len(pth) // cap)                                              # Compute stride that hits cap-ish
    slim = pth[::step]                                                          # Sample every k-th waypoint
    if slim[-1] != pth[-1]: slim = slim + [pth[-1]]                             # Preserve exact terminal node
    return slim                                                                 # Return thinned sequence


############################################################################### 
# --- Adaptive planning -------------------------------------------------------  
###############################################################################  
# Tune search budgets and heuristic weight from map density and route length.
# Try a couple of presets from conservative toward aggressive as needed.

def AdaptiveParams(cs, start, goal, p: PlanningParams):                         # Derive settings from map stats and distance
    free  = int((cs > p.th_free_planner).sum())                                 # Count free cells at planner threshold
    total = cs.size                                                             # Total number of cells in cspace
    occ_pct = (100.0 * free / total)                                            # Percent free space estimate
    dense  = (100 - occ_pct) > 40                                               # Considered obstacle-dense terrain
    sparse = (100 - occ_pct) < 10                                               # Considered open terrain region
    dist = heuristic(start, goal)                                               # Grid distance for scaling heuristics
    out = {"max_iterations": p.max_iterations,                                   # Start with baseline caps
           "max_open_set_size": p.max_open_set_size,
           "heuristic_weight": p.heuristic_weight}
    if dense:                                                                   # Heavier clutter case
        out["max_iterations"]   = int(p.max_iterations * 0.7)                   # Trim expansions modestly
        out["max_open_set_size"]= int(p.max_open_set_size * 0.8)                # Reduce memory pressure
        out["heuristic_weight"] = 1.1                                           # Slightly greedier drive
    elif sparse:                                                                # Open space case
        out["max_iterations"]   = int(p.max_iterations * 1.5)                   # Allow more breadth
        out["heuristic_weight"] = 0.9                                           # Temper greediness a touch
    if dist > min(cs.shape) * 0.5:                                              # Very long run across map
        out["heuristic_weight"] = min(out["heuristic_weight"] * 1.2, 1.5)       # Cap final weight reasonably
    return out                                                                  # Return tuned params dict

def adaptive_a_star(cs, start, goal, p: PlanningParams):                        # Multi-try planner from mild to bold
    tries = [
        AdaptiveParams(cs, start, goal, p),                                     # Data-driven first shot
        {"max_iterations": p.adaptive_max_iterations,                           # Larger budgets
         "max_open_set_size": p.adaptive_max_open_set_size,
         "heuristic_weight": p.adaptive_heuristic_weight},
        {"max_iterations": p.adaptive_max_iterations,                           # Aggressive fallback
         "max_open_set_size": p.adaptive_max_open_set_size,
         "heuristic_weight": 2.0},
    ]
    for i, kw in enumerate(tries):                                              # Iterate through presets
        path = a_star(cs, start, goal, p, **kw)                                 # Attempt planning with overrides
        if path: return path                                                    # Return first success immediately
    plan_logger.Error("All adaptive A* attempts failed")                        # Exhausted all strategies
    return []                                                                   # Signal failure to caller


###############################################################################  
# --- Public API --------------------------------------------------------------  
###############################################################################  
# Orchestrate transforms, snap endpoints, plan, optionally smooth and optimize.
# Returns a world-frame path ready for controller consumption.

def plan_path(cs, start_w, goal_w, p: PlanningParams, smooth=True,              # Main entry point for planning
              optimize_for_differential_drive=True, check_neighbors=None,       # Optional features and safety policy
              max_search_radius=None, **_):                                     # Unused kwargs for forward-compat
    if check_neighbors is None:                                                 # Use configured neighbor policy by default
        check_neighbors = p.check_neighbor_safety
    if max_search_radius is None:                                               # Use conservative search radius by default
        max_search_radius = p.safe_waypoint_search_radius
    W2G = getattr(p, 'W2G', blackboard.Get("world_to_grid"))                    # World → grid converter
    G2W = getattr(p, 'G2W', blackboard.Get("grid_to_world"))                    # Grid → world converter
    sr, sc = W2G(*start_w, cs.shape); gr, gc = W2G(*goal_w, cs.shape)           # Convert endpoints into indices
    ns = NearestFreeCell(cs, sr, sc, p, max_radius=max_search_radius, check_neighbors=check_neighbors)  # Snap safe start
    if ns is None:
        plan_logger.Error(f"No safe start found at ({sr}, {sc})"); return []    # Abort if no viable start cell
    sr, sc = ns                                                                  # Use snapped start
    ng = NearestFreeCell(cs, gr, gc, p, max_radius=max_search_radius, check_neighbors=check_neighbors)  # Snap safe goal
    if ng is None:
        plan_logger.Error(f"No safe goal found at ({gr}, {gc})"); return []     # Abort if no viable goal cell
    gr, gc = ng                                                                  # Use snapped goal
    if has_line_of_sight((sr, sc), (gr, gc), cs, p):                            # Directly connected endpoints
        return [start_w, goal_w]                                                # Trivial two-point path
    path = adaptive_a_star(cs, (sr, sc), (gr, gc), p)                           # Plan in grid coordinates
    if not path:
        plan_logger.Error(f"Planning failed from ({sr}, {sc}) to ({gr}, {gc})")
        return []                                                               # Signal failure early
    if optimize_for_differential_drive and p.optimize_for_differential_drive and len(path) > 2:
        path = optimize_path_for_differential_drive(path, cs, p)                # Kinematic tweaks for diff-drive
    if smooth and len(path) > 2:
        path = smooth_path(path, cs, p)                                         # LoS shortcutting for fewer turns
    path = Downsample(path, cap=40)                                             # Keep waypoint count manageable
    out = [G2W(r, c) for r, c in path]                                          # Convert to world coordinates
    if out[-1] != goal_w: out.append(goal_w)                                    # Ensure exact final goal is present
    return out                                                                  # World-frame waypoint list


###############################################################################  
# --- Reachability analysis ---------------------------------------------------  
###############################################################################  
# Simple fill for reachability and sampling. 
# Useful for previews goals to connected free space.
def reachable_mask_from(cs, start_world, p: PlanningParams, threshold=None, connectivity=8): 
    if start_world is None:                                                     # No seed provided 
        return None                                                             # Nothing to compute here
    W2G = getattr(p, 'W2G', blackboard.Get("world_to_grid"))                    # Converter for world to grid
    r0, c0 = W2G(*start_world, cs.shape)                                        # Seed location in grid indices
    t = Thr(threshold, p); h, w = cs.shape                                      # Threshold and map size
    if (not Inb(h, w, r0, c0)) or cs[r0, c0] <= t:                              # Invalid blocked cell
        near = NearestFreeCell(cs, r0, c0, p, threshold=t, max_radius=6, check_neighbors=False)  # Pull to nearby free
        if near is None:                                                        # No alternative found
            return np.zeros((h, w), dtype=bool)                                 # Empty result
        r0, c0 = near                                                           # Use recovered seed cell
    vis = np.zeros((h, w), dtype=bool); stack = [(r0, c0)]                      
    while stack:                                                                # Depth first traversal
        r, c = stack.pop()                                                      # Pop next cell
        if (not Inb(h, w, r, c)) or vis[r, c] or cs[r, c] <= t:                 # Skip invalid or already visited
            continue
        vis[r, c] = True                                                        # Mark reachable position
        for dr in (-1, 0, 1):                                                   # Explore row offsets
            for dc in (-1, 0, 1):                                               # Explore column offsets
                if dr == 0 and dc == 0: continue                                # Skip center cell
                if connectivity == 4 and dr != 0 and dc != 0: continue          
                stack.append((r + dr, c + dc))                                  # Push neighbor onto stack
    return vis                                                                  # Boolean 

def find_safe_positions(cs, p: PlanningParams, num_positions=10, threshold=None,  # Random safe samples in world frame
                        restrict_to_reachable_from=None, check_neighbors=True, max_attempts=None):
    out = []; h, w = cs.shape                                                   # Collected results 
    t = min(1.0, Thr(threshold, p) + 0.05)                                      
    if max_attempts is None: max_attempts = num_positions * 20                  # Simple budget heuristic
    reach = reachable_mask_from(cs, restrict_to_reachable_from, p, threshold=t) if restrict_to_reachable_from is not None else None
    G2W = getattr(p, 'G2W', blackboard.Get("grid_to_world"))                    # Converter for grid to world
    tries = 0
    while tries < max_attempts:                                                 # Rejection loop
        tries += 1
        r = np.random.randint(10, h - 10); c = np.random.randint(10, w - 10)    # Avoid tight edges 
        if reach is not None and not reach[r, c]:                               
            continue
        if cs[r, c] > t and is_free_with_neighbors(cs, r, c, p, threshold=t, check_neighbors=check_neighbors):
            xw, yw = G2W(r, c)                                                  # Convert candidate to world
            too_close = False                                                   # Enforce spacing 
            for ex, ey in out:
                if np.hypot(xw - ex, yw - ey) < 0.5:                            # Minimum threshold
                    too_close = True; break
            if not too_close:                                                   
                out.append((xw, yw))                                            # Keep world position
                if len(out) >= num_positions: break                             # Reached target count
    return out                                                                  


###############################################################################  
# --- Validation and visualization --------------------------------------------  
############################################################################### 
# Validate waypoints against local segment LoS. 
# Quick visualization writes a .npy map overlay for debugging.

def validate_path(path_w, cs, p: PlanningParams):                                # World-frame path
    if len(path_w) < 2: return True                                              # Valid when < 2 points
    W2G = getattr(p, 'W2G', blackboard.Get("world_to_grid"))                     # World to grid transform
    for i, (x, y) in enumerate(path_w):                                          # Validate each vertex
        r, c = W2G(x, y, cs.shape)                                               # Convert to grid indices
        ok = is_free_with_neighbors(cs, r, c, p, check_neighbors=p.check_neighbor_safety)
        if not ok:
            plan_logger.Error(f"Invalid waypoint {i} at ({x:.2f}, {y:.2f})")     
            return False                                                         # Reject entire path
    for i in range(len(path_w) - 1):                                             # Validate segments for LoS
        r0, c0 = W2G(*path_w[i], cs.shape); r1, c1 = W2G(*path_w[i+1], cs.shape)
        if not has_line_of_sight((r0, c0), (r1, c1), cs, p):
            plan_logger.Error(f"No LoS between waypoints {i} and {i + 1}")       # Identify failing segment
            return False                                                         # Fail 
    return True                                                                  # All checks passed

def visualize_path_on_map(cs, path_w, save_path="path_visualization.npy"):       # Write simple path overlay to .npy
    vis = cs.copy()                                                              # Copy base cspace
    W2G = blackboard.Get("world_to_grid")                                        # Grab converter from blackboard
    for x, y in path_w:                                                          # Mark each waypoint cell
        r, c = W2G(x, y, cs.shape)
        if Inb(cs.shape[0], cs.shape[1], r, c):
            vis[r, c] = 2.0                                                      # Distinct value for path overlay
    pth = RESMAP(save_path); ENSURE(pth)                                         # Resolve path and ensure folders
    np.save(pth, vis)                                                            
    return vis                                                                   # Return visualization array
