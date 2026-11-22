"""
Navigation, Line-of-Sight, and Cover System for MechRoyale.
Uses the precomputed 2D occupancy grid (1m resolution) with 2m A* pathfinding.
"""
import json
import os
import heapq
import math
from typing import List, Tuple, Optional
import numpy as np

class NavMesh:
    def __init__(self, map_dir: str):
        here = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(here, "data")
        
        # Priority 1: Bundled precomputed data inside bot directory
        if os.path.exists(os.path.join(data_dir, "occupancy_meta.json")):
            meta_path = os.path.join(data_dir, "occupancy_meta.json")
            bin_path = os.path.join(data_dir, "occupancy.bin")
        # Priority 2: In arena directory
        elif os.path.exists(os.path.join(map_dir, "v2", "occupancy_meta.json")):
            meta_path = os.path.join(map_dir, "v2", "occupancy_meta.json")
            bin_path = os.path.join(map_dir, "v2", "occupancy.bin")
        elif os.path.exists(os.path.join(map_dir, "arena", "v2", "occupancy_meta.json")):
            meta_path = os.path.join(map_dir, "arena", "v2", "occupancy_meta.json")
            bin_path = os.path.join(map_dir, "arena", "v2", "occupancy.bin")
        elif os.path.exists("/arena/v2/occupancy_meta.json"):
            meta_path = "/arena/v2/occupancy_meta.json"
            bin_path = "/arena/v2/occupancy.bin"
        else:
            meta_path = "arena/v2/occupancy_meta.json"
            bin_path = "arena/v2/occupancy.bin"
            
        with open(meta_path, "r") as f:
            self.meta = json.load(f)
            
        self.nx = int(self.meta["nx"])
        self.ny = int(self.meta["ny"])
        self.min_x = float(self.meta["minX"])
        self.min_y = float(self.meta["minY"])
        self.step = float(self.meta["step"])
        
        with open(bin_path, "rb") as f:
            # 0=walkable (clearance >= 2.5m), 1=solid wall, 2=near wall buffer (<2.5m)
            self.grid = np.frombuffer(f.read(), dtype=np.uint8).reshape((self.ny, self.nx))
            
        # 2m downsampled grid (step = 2.0m)
        self.scale = 2
        walkable_1m = (self.grid == 0)
        self.grid_2m = ~walkable_1m[::self.scale, ::self.scale]  # True = blocked
        self.h_2m, self.w_2m = self.grid_2m.shape

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        col = int(round((x - self.min_x) / self.step))
        row = int(round((y - self.min_y) / self.step))
        return max(0, min(self.nx - 1, col)), max(0, min(self.ny - 1, row))

    def grid_to_world(self, col: int, row: int) -> Tuple[float, float]:
        return self.min_x + col * self.step, self.min_y + row * self.step

    def is_walkable(self, x: float, y: float) -> bool:
        col, row = self.world_to_grid(x, y)
        return self.grid[row, col] == 0

    def has_los(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """Check if straight line between (x0, y0) and (x1, y1) has no solid walls using zero-allocation integer Bresenham."""
        c0, r0 = self.world_to_grid(x0, y0)
        c1, r1 = self.world_to_grid(x1, y1)

        dx = abs(c1 - c0)
        dy = abs(r1 - r0)
        sx = 1 if c0 < c1 else -1
        sy = 1 if r0 < r1 else -1
        err = dx - dy

        grid = self.grid
        c, r = c0, r0
        while True:
            # grid == 1 is solid wall; 0 and 2 allow vision/shots
            if grid[r, c] == 1:
                return False
            if c == c1 and r == r1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                c += sx
            if e2 < dx:
                err += dx
                r += sy
        return True

    def find_nearest_walkable_2m(self, col_2m: int, row_2m: int, max_radius: int = 15) -> Tuple[int, int]:
        c = max(0, min(self.w_2m - 1, col_2m))
        r = max(0, min(self.h_2m - 1, row_2m))
        if not self.grid_2m[r, c]:
            return c, r
        for radius in range(1, max_radius + 1):
            for dr in range(-radius, radius + 1):
                for dc in (-radius, radius):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.h_2m and 0 <= nc < self.w_2m and not self.grid_2m[nr, nc]:
                        return nc, nr
            for dc in range(-radius + 1, radius):
                for dr in (-radius, radius):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.h_2m and 0 <= nc < self.w_2m and not self.grid_2m[nr, nc]:
                        return nc, nr
        return c, r

    def plan_path(self, start_pos: Tuple[float, float], goal_pos: Tuple[float, float]) -> List[Tuple[float, float]]:
        """A* path on 2m grid with line-of-sight shortcutting."""
        sx, sy = start_pos[0], start_pos[1]
        gx, gy = goal_pos[0], goal_pos[1]
        
        # If direct line of sight exists, go straight!
        if self.has_los(sx, sy, gx, gy):
            return [(gx, gy)]

        # Convert to 2m grid coords
        c0 = int(round((sx - self.min_x) / (self.step * self.scale)))
        r0 = int(round((sy - self.min_y) / (self.step * self.scale)))
        c1 = int(round((gx - self.min_x) / (self.step * self.scale)))
        r1 = int(round((gy - self.min_y) / (self.step * self.scale)))

        c0, r0 = self.find_nearest_walkable_2m(c0, r0)
        c1, r1 = self.find_nearest_walkable_2m(c1, r1)

        if (c0, r0) == (c1, r1):
            return [(gx, gy)]

        # 8-direction A*
        pq = [(0.0, r0, c0)]
        cost = {(r0, c0): 0.0}
        parent = {}
        dirs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]

        nodes_visited = 0
        max_nodes = 2500  # Cap search to stay under 5ms
        reached = False
        end_node = (r1, c1)

        while pq and nodes_visited < max_nodes:
            nodes_visited += 1
            f, r, c = heapq.heappop(pq)
            if (r, c) == (r1, c1) or (abs(r - r1) <= 1 and abs(c - c1) <= 1):
                reached = True
                end_node = (r, c)
                break
                
            g = cost[(r, c)]
            for dr, dc, dcost in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.h_2m and 0 <= nc < self.w_2m and not self.grid_2m[nr, nc]:
                    ng = g + dcost
                    if ng < cost.get((nr, nc), 1e9):
                        cost[(nr, nc)] = ng
                        h = math.hypot(r1 - nr, c1 - nc)
                        heapq.heappush(pq, (ng + h, nr, nc))
                        parent[(nr, nc)] = (r, c)

        if not reached:
            # Fallback to closest explored node toward goal
            if parent:
                best_n = min(parent.keys(), key=lambda n: math.hypot(r1 - n[0], c1 - n[1]))
                end_node = best_n
            else:
                return [(gx, gy)]

        # Reconstruct path
        path_nodes = []
        curr = end_node
        while curr in parent:
            path_nodes.append(curr)
            curr = parent[curr]
        path_nodes.reverse()

        if not path_nodes:
            return [(gx, gy)]

        # Convert back to world coordinates
        raw_waypoints = []
        for r, c in path_nodes:
            wx = self.min_x + c * self.step * self.scale
            wy = self.min_y + r * self.step * self.scale
            raw_waypoints.append((wx, wy))
        raw_waypoints.append((gx, gy))

        # String-pulling (shortcut line-of-sight)
        if len(raw_waypoints) <= 2:
            return raw_waypoints

        smooth_path = [raw_waypoints[0]]
        i = 0
        while i < len(raw_waypoints) - 1:
            furthest = i + 1
            for j in range(len(raw_waypoints) - 1, i, -1):
                if self.has_los(raw_waypoints[i][0], raw_waypoints[i][1], raw_waypoints[j][0], raw_waypoints[j][1]):
                    furthest = j
                    break
            smooth_path.append(raw_waypoints[furthest])
            i = furthest

        return self.round_corner_waypoints(smooth_path)

    def round_corner_waypoints(
        self,
        waypoints: List[Tuple[float, float]],
        fillet_radius: float = 4.5,
        min_angle_deg: float = 35.0
    ) -> List[Tuple[float, float]]:
        """
        Rounds sharp corners (e.g. 90-degree building turns) into smooth wall-sliding arcs.
        Replaces sharp vertices with tangential fillet waypoints that maintain safe wall clearance.
        """
        if len(waypoints) < 3:
            return waypoints

        result: List[Tuple[float, float]] = [waypoints[0]]
        for i in range(1, len(waypoints) - 1):
            p_prev = waypoints[i - 1]
            p_curr = waypoints[i]
            p_next = waypoints[i + 1]

            v1 = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
            v2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])
            l1 = math.hypot(v1[0], v1[1])
            l2 = math.hypot(v2[0], v2[1])

            if l1 < 2.0 or l2 < 2.0:
                result.append(p_curr)
                continue

            u1 = (v1[0] / l1, v1[1] / l1)
            u2 = (v2[0] / l2, v2[1] / l2)
            dot = max(-1.0, min(1.0, u1[0] * u2[0] + u1[1] * u2[1]))
            turn_angle_deg = math.acos(dot) * 180.0 / math.pi

            if turn_angle_deg >= min_angle_deg:
                d_cut = min(fillet_radius * math.tan(turn_angle_deg * math.pi / 360.0), 0.45 * l1, 0.45 * l2)
                p_in = (p_curr[0] - d_cut * u1[0], p_curr[1] - d_cut * u1[1])
                p_out = (p_curr[0] + d_cut * u2[0], p_curr[1] + d_cut * u2[1])

                bisect = (u2[0] - u1[0], u2[1] - u1[1])
                blen = math.hypot(bisect[0], bisect[1])
                if blen > 1e-4:
                    ub = (bisect[0] / blen, bisect[1] / blen)
                    apex = (p_curr[0] + (d_cut * 0.35) * ub[0], p_curr[1] + (d_cut * 0.35) * ub[1])
                    if self.is_walkable(apex[0], apex[1]):
                        result.extend([p_in, apex, p_out])
                    elif self.is_walkable(p_in[0], p_in[1]) and self.is_walkable(p_out[0], p_out[1]):
                        result.extend([p_in, p_out])
                    else:
                        result.append(p_curr)
                else:
                    result.extend([p_in, p_out])
            else:
                result.append(p_curr)

        result.append(waypoints[-1])
        return result

    def compute_wall_slide_vector(
        self,
        px: float,
        py: float,
        desired_vx: float,
        desired_vy: float,
        probe_dist: float = 3.5
    ) -> Tuple[float, float]:
        """
        Wall-sliding along concave and convex wall contours:
        When moving near walls, projects velocity along the tangent of the wall normal.
        In concave corners (inside pockets), deflects velocity outward along the concave bisector.
        """
        normals = []
        for deg in range(0, 360, 30):
            rad = math.radians(deg)
            dx, dy = math.cos(rad), math.sin(rad)
            for d in (1.5, 2.5, probe_dist):
                qx, qy = px + dx * d, py + dy * d
                col, row = self.world_to_grid(qx, qy)
                if self.grid[row, col] == 1:
                    normals.append((-dx, -dy))
                    break
        if not normals:
            return desired_vx, desired_vy

        avg_nx = sum(n[0] for n in normals)
        avg_ny = sum(n[1] for n in normals)
        nlen = math.hypot(avg_nx, avg_ny)
        if nlen < 1e-4:
            return desired_vx, desired_vy
        nx_out = avg_nx / nlen
        ny_out = avg_ny / nlen

        dot = desired_vx * nx_out + desired_vy * ny_out
        if dot < 0.0:
            slide_x = desired_vx - dot * nx_out
            slide_y = desired_vy - dot * ny_out
            if len(normals) >= 3:
                slide_x += nx_out * 0.35
                slide_y += ny_out * 0.35
            slen = math.hypot(slide_x, slide_y)
            if slen > 1e-4:
                return slide_x / slen, slide_y / slen
        return desired_vx, desired_vy

    def find_cover_spot(self, my_pos: Tuple[float, float], threat_pos: Tuple[float, float], min_dist: float = 30.0, max_dist: float = 60.0) -> Optional[Tuple[float, float]]:
        """Find a nearby cell that breaks line-of-sight from threat_pos."""
        my_col, my_row = self.world_to_grid(my_pos[0], my_pos[1])
        radius_cells = int(max_dist / self.step)
        step_stride = 3

        best_spot = None
        best_score = -1e9

        for dr in range(-radius_cells, radius_cells + 1, step_stride):
            for dc in range(-radius_cells, radius_cells + 1, step_stride):
                nr, nc = my_row + dr, my_col + dc
                if 0 <= nr < self.ny and 0 <= nc < self.nx and self.grid[nr, nc] == 0:
                    cand_x, cand_y = self.grid_to_world(nc, nr)
                    dist_to_threat = math.hypot(cand_x - threat_pos[0], cand_y - threat_pos[1])
                    dist_to_me = math.hypot(cand_x - my_pos[0], cand_y - my_pos[1])
                    
                    if min_dist <= dist_to_threat and dist_to_me <= max_dist:
                        if not self.has_los(cand_x, cand_y, threat_pos[0], threat_pos[1]):
                            score = -dist_to_me + dist_to_threat * 0.5
                            if score > best_score:
                                best_score = score
                                best_spot = (cand_x, cand_y)
                                
        return best_spot
