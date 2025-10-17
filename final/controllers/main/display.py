from typing import List, Tuple                                                                                       # types
import numpy as np                                                                                                   # math
from PIL import Image                                                                                                # image 
from config import RobotConfig                                                                                        # config constants
from utils import world_to_pixel                                                                                      # worldtogrid helper

class MapDisplay:
    def __init__(self, memory) -> None:
        self.memory, self.display, self.width, self.height = memory, None, 0, 0                                      # shared refs + size cache
        self.COLOR_BLACK, self.COLOR_WHITE = 0x000000, 0xFFFFFF                                                       # basic colors
        self.COLOR_RED, self.COLOR_GREEN, self.COLOR_BLUE = 0xFF0000, 0x00FF00, 0x0000FF                              # overlay colors

    def update(self) -> None:
        self.display = self.memory.get("display")                                                                     # Webots Display handle
        if not self.display:                                                                                          
            return                                                                                                    # nothing to draw
        self.width, self.height = self.display.getWidth(), self.display.getHeight()                                   # cache dims
        self.display.setColor(self.COLOR_BLACK)                                                                       
        self.display.fillRectangle(0, 0, self.width, self.height)                                                     # clear frame
        cs = self.memory.get("cspace")                                                                                # binary/free map
        if cs is not None:
            self._draw_cspace(cs)                                                                                     # draw cspace
        else:
            pm = self.memory.get("prob_map")                                                                          # grayscale map
            if pm is not None:
                self._draw_probability_map(pm)                                                                        # draw prob map
            traj = self.memory.get("trajectory_points")                                                               # past commanded path
            if traj:
                self._draw_trajectory(traj)                                                                           # draw traj
            self._draw_robot()                                                                                        
        sys = self.memory.get("system_instance")                                                                       # system ref
        if sys and getattr(sys, 'navigation', None) and sys.navigation.planned_path:                                  # planned path?
            self._draw_planned_path(sys.navigation.planned_path)                                                      # overlay plan

    def _draw_probability_map(self, pm: np.ndarray) -> None:
        for x in range(0, pm.shape[0], 2):                                                                            # subsample X
            for y in range(0, pm.shape[1], 2):                                                                        # subsample Y
                if pm[x, y] > 0.001:                                                                                  # ignore noise
                    I = int(255 * min(pm[x, y] * 2.5, 0.8))                                                           
                    col = (I << 16) | (I << 8) | I                                                                     
                    self.display.setColor(col)                                                                         # set color
                    dx, dy = self._map_to_display(x, y, pm.shape)                                                     
                    if 0 <= dx < self.width and 0 <= dy < self.height:                                                # on-screen
                        self.display.fillRectangle(dx, dy, 2, 2)                                                      # fast block
    def _draw_cspace(self, cs: np.ndarray) -> None:
        for x in range(cs.shape[0]):                                                                                  # iterate X
            for y in range(cs.shape[1]):                                                                              # iterate Y
                if cs[x, y] < 0.5:                                                                                    # occupied/edge
                    self.display.setColor(self.COLOR_WHITE)                                                           # white pixel
                    dx, dy = self._map_to_display(x, y, cs.shape)                                                     # gridto screen
                    if 0 <= dx < self.width and 0 <= dy < self.height:                                                # bounds check
                        self.display.fillRectangle(dx, dy, 1, 1)                                                      # precise dot

    def _draw_trajectory(self, traj: List[Tuple[float, float]]) -> None:
        self.display.setColor(self.COLOR_RED)                                                                         # red overlay
        for i in range(len(traj) - 1):                                                                                # segment pairs
            px1, py1 = world_to_pixel(*traj[i])                                                                       # world to grid
            px2, py2 = world_to_pixel(*traj[i + 1])                                                                   
            dx1, dy1 = self._map_to_display(px1, py1, (RobotConfig.MAP_WIDTH, RobotConfig.MAP_SIZE))                  
            dx2, dy2 = self._map_to_display(px2, py2, (RobotConfig.MAP_WIDTH, RobotConfig.MAP_SIZE))                  
            if 0 <= dx1 < self.width and 0 <= dy1 < self.height and 0 <= dx2 < self.width and 0 <= dy2 < self.height: # visible
                self.display.drawLine(dx1, dy1, dx2, dy2)                                                             # draw segment

    def _draw_planned_path(self, path: List[Tuple[float, float]]) -> None:
        self.display.setColor(self.COLOR_BLUE)                                                                        # blue overlay
        for i in range(len(path) - 1):                                                                                # segment pairs
            px1, py1 = world_to_pixel(*path[i])                                                                       
            px2, py2 = world_to_pixel(*path[i + 1])                                                                   
            dx1, dy1 = self._map_to_display(px1, py1, (RobotConfig.MAP_WIDTH, RobotConfig.MAP_SIZE))                  # grid to screen
            dx2, dy2 = self._map_to_display(px2, py2, (RobotConfig.MAP_WIDTH, RobotConfig.MAP_SIZE))                  
            if 0 <= dx1 < self.width and 0 <= dy1 < self.height and 0 <= dx2 < self.width and 0 <= dy2 < self.height: # visible
                self.display.drawLine(dx1, dy1, dx2, dy2)                                                             # draw segment

    def _draw_robot(self) -> None:
        gps, comp = self.memory.get("gps"), self.memory.get("compass")                                                # sensors
        if not gps or not comp:                                                                                       # need both
            return                                                                                                    # skip
        pos = gps.getValues()                                                                                         # robot world pos
        px, py = world_to_pixel(pos[0], pos[1])                                                                       # world→grid
        dx, dy = self._map_to_display(px, py, (RobotConfig.MAP_WIDTH, RobotConfig.MAP_SIZE))                          # grid→screen
        if 0 <= dx < self.width and 0 <= dy < self.height:                                                            # on-screen
            self.display.setColor(self.COLOR_GREEN)                                                                   # green glyph
            self.display.fillRectangle(dx - 2, dy - 2, 5, 5)                                                          # small square
            ang = np.arctan2(comp.getValues()[0], comp.getValues()[1])                                               # yaw from compass
            ex, ey = dx + int(10 * np.cos(ang)), dy + int(10 * np.sin(ang))                                          # arrow tip
            self.display.setColor(self.COLOR_WHITE)                                                                   # white arrow
            self.display.drawLine(dx, dy, ex, ey)                                                                     # heading line

    def _map_to_display(self, px: int, py: int, shape: Tuple[int, int]) -> Tuple[int, int]:
        return int(px * self.width / shape[0]), int(py * self.height / shape[1])                                     # linear scale

    def save_display_png(self, filename="map_display.png", map_dir="map") -> None:
        try:
            import os                                                                                                
            os.makedirs(map_dir, exist_ok=True)                                                                       # ensure dir
            path = os.path.join(map_dir, filename)                                                                    # full path
            if not (self.display and self.width > 0 and self.height > 0):                                             # valid canvas?
                print(f"Cannot save display - no display available or invalid dimensions")                            # log and exit
                return                                                                                                # nothing to save
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)                                              # black RGB canvas
            cs = self.memory.get("cspace")                                                                            # prefer cspace
            if cs is not None:
                for x in range(0, cs.shape[0], 2):                                                                     # subsample X
                    for y in range(0, cs.shape[1], 2):                                                                 # subsample Y
                        if cs[x, y] < 0.5:                                                                             # occupied/edge
                            dx, dy = self._map_to_display(x, y, cs.shape)                                              # grid to screen
                            if 0 <= dx < self.width and 0 <= dy < self.height:                                         # bounds
                                img[dy:dy + 2, dx:dx + 2] = [255, 255, 255]                                           # white block
            else:
                pm = self.memory.get("prob_map")                                                                       # grayscale map
                if pm is not None:
                    for x in range(0, pm.shape[0], 2):                                                                 # subsample X
                        for y in range(0, pm.shape[1], 2):                                                             # subsample Y
                            if pm[x, y] > 0.001:                                                                       # ignore noise
                                I = int(255 * min(pm[x, y] * 2.5, 0.8))                                                # map to gray
                                dx, dy = self._map_to_display(x, y, pm.shape)                                          
                                if 0 <= dx < self.width and 0 <= dy < self.height:                                     # bounds
                                    img[dy:dy + 2, dx:dx + 2] = [I, I, I]                                              # gray block
                traj = self.memory.get("trajectory_points")                                                            # red overlay
                if traj:
                    for i in range(len(traj) - 1):                                                                     # segment pairs
                        px1, py1 = world_to_pixel(*traj[i])                                                            
                        px2, py2 = world_to_pixel(*traj[i + 1])                                                        
                        dx1, dy1 = self._map_to_display(px1, py1, (RobotConfig.MAP_WIDTH, RobotConfig.MAP_SIZE))       
                        dx2, dy2 = self._map_to_display(px2, py2, (RobotConfig.MAP_WIDTH, RobotConfig.MAP_SIZE))       
                        if 0 <= dx1 < self.width and 0 <= dy1 < self.height and 0 <= dx2 < self.width and 0 <= dy2 < self.height:  # visible
                            steps = max(abs(dx2 - dx1), abs(dy2 - dy1))                                                # line steps
                            for s in range(steps + 1):                                                                 # sample line
                                t = s / steps if steps > 0 else 0                                                      # param t
                                x = int(dx1 + t * (dx2 - dx1))                                                         # interp x
                                y = int(dy1 + t * (dy2 - dy1))                                                         # interp y
                                if 0 <= x < self.width and 0 <= y < self.height:                                       # bounds
                                    img[y, x] = [255, 0, 0]                                                            # red pixel
            if cs is None:                                                                                            # draw robot only in prob map mode
                gps, comp = self.memory.get("gps"), self.memory.get("compass")                                        # sensors
                if gps and comp:                                                                                      # both present
                    pos = gps.getValues()                                                                              # world pos
                    px, py = world_to_pixel(pos[0], pos[1])                                                            
                    dx, dy = self._map_to_display(px, py, (RobotConfig.MAP_WIDTH, RobotConfig.MAP_SIZE))              
                    if 0 <= dx < self.width and 0 <= dy < self.height:                                                # bounds
                        img[max(0, dy - 2):min(self.height, dy + 3), max(0, dx - 2):min(self.width, dx + 3)] = [0, 255, 0]  # green box
                        ang = np.arctan2(comp.getValues()[0], comp.getValues()[1])                                     # yaw
                        ex, ey = int(dx + 10 * np.cos(ang)), int(dy + 10 * np.sin(ang))                                # arrow tip
                        steps = max(abs(ex - dx), abs(ey - dy))                                                         # line steps
                        for s in range(steps + 1):                                                                      # sample line
                            t = s / steps if steps > 0 else 0                                                           # param t
                            x = int(dx + t * (ex - dx))                                                                 # interp x
                            y = int(dy + t * (ey - dy))                                                                 # interp y
                            if 0 <= x < self.width and 0 <= y < self.height:                                           # bounds
                                img[y, x] = [255, 255, 255]                                                            # white pixel
            Image.fromarray(img).save(path)                                                                            # write PNG
            print(f"Saved display image'")                                                                            # log success
        except Exception as e:
            print(f"Failed to save display PNG")                                                                      # log error
