# Webots: TIAGo Mapping & Navigation

[![Made by Soph](https://img.shields.io/badge/Made%20by-Soph-ff69b4?style=for-the-badge)](https://github.com/soph-k)
[![License](https://img.shields.io/badge/license-MIT-yellow?style=for-the-badge)](https://github.com/soph-k/tiago-mapping-planning/blob/main/LICENSE)
[![Last commit](https://img.shields.io/badge/last%20commit-see%20history-informational?style=for-the-badge)](https://github.com/soph-k/tiago-mapping-planning/commits)

Build a LiDAR map, inflate to C-space, plan with A*, and navigate multiple goals via a simple behavior tree.

![Robot Demo](assets/images/demo.gif)

---

## Overview

A Webots controller for a TIAGo-like robot that:

- Builds a 2D occupancy map from LiDAR  
- Inflates it into a traversable C-space  
- Plans paths with A*/Jump Point Search (optionally bidirectional)  
- Follows waypoints with obstacle-aware navigation and stuck recovery  
- Visualizes map/C-space, pose, goals, and path on the Webots Display

---

## Features

- World to Grid transforms (default grid 200×300)  
- Distance-transform inflation with tunable safety margins  
- A* + JPS, optional bidirectional search & smoothing  
- Safe velocity limits, detours, and stuck detection  
- Lightweight behavior tree orchestration and logging

---

## Built With

- Webots  
- Python 3.8+  
- NumPy  
- SciPy
- PIL (for C-space image export)

---

## Quickstart

### Prerequisites

- Webots installed  
- Python 3.8+ selected in **Webots → Preferences → Python**  
- scipy for fast distance transforms

### Install
```bash
git clone https://github.com/soph-k/tiago-mapping-planning.git
cd tiago-mapping-planning
pip install numpy scipy pillow
```

### Run (in Webots)

1. Load your world
2. Set the controller to `main.py`
3. Press Run

---

## Project Structure
```
main.py         # Device init, BT, display
core.py         # Params, transforms, logging, helpers
mapping.py      # Occupancy → C-space
planning.py     # A*, JPS, bidirectional, smoothing
navigation.py   # Waypoint follower, obstacle checks, recovery
maps/           # Saved maps and C-space images
assets/
  +-- images/   # Screenshots and demo animation
```

---

## How It Works

1. **Mapping**: Lidar updates an occupancy grid; thresholds define free/occupied
2. **C-space**: Euclidean distance transform inflates obstacles
3. **Planning**: A* with JPS acceleration; optional bidirectional for long routes; optional smoothing
4. **Navigation**: Heading+distance controller with obstacle checks, safe speed caps, and detours when stuck
5. **Display**: Renders map/C-space, path, goals, pose, and trajectory

---

## Key Parameters 

- **Mapping**: `th_occupied`, `th_free_planner`, `robot_radius`, `safety_margin`, `map_resolution_m`, `cspace_*`
- **Planning**: `jump_point_search`, `bidirectional`, `heuristic_weight`, `max_iterations`, `max_open_set_size`
- **Navigation**: `tolerance`, `p1/p2` gains, `maxSpeed`
- **Logging**: `ROBOT_LOG_LEVEL = ERROR|WARNING|INFO|DEBUG|VERBOSE`

---

## Configuration Space (C-Space)

The robot generates a configuration space that accounts for its physical dimensions and safety. This C-space is built from the probabilistic occupancy map using Euclidean distance transforms.

![Configuration Space](assets/images//cspace.png)

**C-Space Characteristics:**
- **Black areas**: Navigable free space where the robot center can safely travel
- **White areas**: No-go zones including inflated obstacles and walls
- **Inflation parameters**: Robot radius (0.15m) + safety margin (0.05m) with configurable scale factor
- **Uncertainty**: Regions with unknown probability treated as occupied
- **Update**: Rebuilt every 0.5 seconds during mapping, frozen after forward loop completed


The C-space generation uses an 18% additional safety buffer beyond the minimum required distance, creating a thinner but safe navigation space while preventing collisions.

---

## Behavior Tree Structure
```
Root: MainWithDisplay (Parallel)
|
+-- MainMissionTree (Selector)
|   |
|   +-- UseExistingMap (OnlyOnce wrapper)
|   |   |
|   |   +-- UseExistingMapSequence (Sequence)
|   |       |
|   |       +-- MapExistsOrReady
|   |       +-- LoadMap
|   |       +-- ValidateLoadedMap
|   |       +-- EnableCspaceDisplay
|   |       +-- SetDisplayMode(cspace)
|   |       +-- PlanThenGo (Sequence)
|   |           |
|   |           +-- SetTwoGoals
|   |           +-- MultiGoalPlannerBT
|   |           +-- NavigateToWaypoints
|   |
|   +-- CompleteMappingSequence (Sequence)
|       |
|       +-- SurveyWithBackgroundMapping (Parallel)
|       |   |
|       |   +-- BidirectionalSurveyNavigator
|       |   +-- LidarMappingBT (RunInBackground wrapper)
|       |
|       +-- EnsureCspaceNow
|       +-- WaitForMapReady
|       +-- SaveMap
|       +-- SaveCspaceImage
|       +-- ValidateLoadedMap
|       +-- EnableCspaceDisplay
|       +-- SetDisplayMode(cspace)
|       +-- PlanThenGo (Sequence)
|           |
|           +-- SetTwoGoals
|           +-- MultiGoalPlannerBT
|           +-- NavigateToWaypoints
|
+-- DisplayUpdater (continuous)
```

### Node Type Legend
- **Selector**: Tries children left-to-right until one succeeds
- **Sequence**: Executes children left-to-right, fails if any child fails
- **Parallel**: Runs multiple children simultaneously
- **OnlyOnce**: Decorator that ensures child runs only once per session
- **RunInBackground**: Decorator that keeps child running without blocking parent

### Execution Flow
1. System attempts to load existing map
2. If no map exists, performs complete mapping sequence
3. Once map is ready, plans path to goals
4. Navigates to goals while avoiding obstacles
5. Display updates continuously throughout mission

---

## License

MIT — see [LICENSE](LICENSE).

---

## Acknowledgments

Webots, Python, NumPy, SciPy, PIL, and coolsymbols.com.

Badges workflow adapted by Soph with help from Claude AI.