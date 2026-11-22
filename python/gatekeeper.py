"""
CHOKE: Zone Gatekeeper & Macro Strategist for MechRoyale Tournament Bot.

Mission:
Turn the deadly shrinking zone into our lethal weapon.
- Zone radii shrink: 560m -> 380m (60-110s) -> 240m (160-210s) -> 140m (260-310s) -> 60m -> 25m -> 0m.
- Zone DPS scales exponentially: 20 -> 40 -> 80 -> 260 DPS! (260 DPS kills a 1400 HP mech in 5.3s).
- Placement score = 16 - placement. Surviving early is worth 10-15x more than a risky kill.
- Endgame (< 5 alive): pure HP preservation; rank at 8:00 is determined by HP.

Key Capabilities:
1. Chokepoint & Funnel Precomputation:
   Analyzes campus building layout from world.json / occupancy.bin to find narrow alleys
   along the incoming zone circle and calculate funnel catchment.
2. The Gatekeeper Ambush Doctrine:
   Rotates early into safe zone (ahead of the 15s warning), sets up behind cover at the funnel exit,
   holds pre-aim down the alley, and ambushes low-HP panic rotators.
3. Endgame Survival Engine (< 5 alive):
   Avoids optional fights, lets enemies fight each other, positions on high ground or deep cover
   inside the final 60m circle.
4. Ohio Stadium Pinch Avoidance:
   Detects stadium traps and precomputes safe bypass paths.
"""

import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np


class Chokepoint:
    """Descriptor of a zone boundary bottleneck / funnel alley."""
    def __init__(
        self,
        pos: Tuple[float, float],
        width: float,
        inward_dir: Tuple[float, float],
        outward_dir: Tuple[float, float],
        angle: float,
        traffic_score: float,
        exit_pos: Tuple[float, float],
        ambush_pos: Tuple[float, float],
        aim_yaw: float
    ):
        self.pos = (float(pos[0]), float(pos[1]))
        self.width = float(width)
        self.inward_dir = (float(inward_dir[0]), float(inward_dir[1]))
        self.outward_dir = (float(outward_dir[0]), float(outward_dir[1]))
        self.angle = float(angle)
        self.traffic_score = float(traffic_score)
        self.exit_pos = (float(exit_pos[0]), float(exit_pos[1]))
        self.ambush_pos = (float(ambush_pos[0]), float(ambush_pos[1]))
        self.aim_yaw = float(aim_yaw)

    def __getitem__(self, idx: int) -> float:
        return self.pos[idx]

    def __len__(self) -> int:
        return 2

    def __iter__(self):
        return iter(self.pos)

    def __repr__(self) -> str:
        return (
            f"Chokepoint(pos=({self.pos[0]:.1f}, {self.pos[1]:.1f}), "
            f"width={self.width:.1f}m, traffic={self.traffic_score:.1f}, "
            f"yaw={math.degrees(self.aim_yaw):.1f}°)"
        )


class ZoneGatekeeper:
    """
    Zone Gatekeeper & Macro Strategist (CHOKE).
    Manages zone shrink timings, chokepoint ambush positioning,
    stadium bypasses, and endgame HP preservation.
    """

    def __init__(self, map_dir: str = "."):
        self.map_dir = map_dir
        self.meta, self.grid = self._load_occupancy()
        self.nx = int(self.meta["nx"])
        self.ny = int(self.meta["ny"])
        self.min_x = float(self.meta["minX"])
        self.min_y = float(self.meta["minY"])
        self.step = float(self.meta["step"])

        # Load terrain elevations if available
        self.terrain = self._load_terrain()

        # Stadium Bounding Box & Safe Bypasses
        # Stadium footprint: X in [-382.2, -170.2], Y in [-13.2, 266.8]
        self.stadium_box = {
            "min_x": -385.0,
            "max_x": -165.0,
            "min_y": -15.0,
            "max_y": 270.0
        }
        self.stadium_north_bypass = (-220.0, 285.0)
        self.stadium_south_bypass = (-220.0, -28.0)

        # Chokepoint cache: (next_cx, next_cy, next_r) -> List[Chokepoint]
        self._choke_cache: Dict[Tuple[float, float, float], List[Chokepoint]] = {}

        # Endgame position cache
        self._endgame_cache: Dict[Tuple[float, float, float], Tuple[float, float]] = {}

    def _resolve_path(self, filename: str) -> Optional[str]:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(here, "data", filename),
            os.path.join(here, "arena", "v2", filename),
            os.path.join(self.map_dir, "v2", filename),
            os.path.join(self.map_dir, "arena", "v2", filename),
            os.path.join(self.map_dir, filename),
            os.path.join("/arena", "v2", filename),
            os.path.join("/arena", filename),
            filename
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    def _load_occupancy(self) -> Tuple[Dict[str, Any], np.ndarray]:
        meta_path = self._resolve_path("occupancy_meta.json")
        bin_path = self._resolve_path("occupancy.bin")

        if not meta_path or not bin_path:
            meta = {"nx": 1070, "ny": 724, "minX": -513.0, "minY": -334.0, "step": 1.0}
            grid = np.zeros((724, 1070), dtype=np.uint8)
            return meta, grid

        with open(meta_path, "r") as f:
            meta = json.load(f)

        with open(bin_path, "rb") as f:
            grid = np.frombuffer(f.read(), dtype=np.uint8).reshape((int(meta["ny"]), int(meta["nx"])))

        return meta, grid

    def _load_terrain(self) -> Optional[np.ndarray]:
        t_path = self._resolve_path("terrain.i16")
        if not t_path:
            return None
        try:
            with open(t_path, "rb") as f:
                return np.frombuffer(f.read(), dtype="<i2").reshape((self.ny, self.nx)).astype(np.float32) * 0.01
        except Exception:
            return None

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        col = int(round((x - self.min_x) / self.step))
        row = int(round((y - self.min_y) / self.step))
        return max(0, min(self.nx - 1, col)), max(0, min(self.ny - 1, row))

    def grid_to_world(self, col: int, row: int) -> Tuple[float, float]:
        return self.min_x + col * self.step, self.min_y + row * self.step

    def is_walkable(self, x: float, y: float) -> bool:
        c, r = self.world_to_grid(x, y)
        return self.grid[r, c] == 0

    def has_los(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """Check direct line-of-sight across the occupancy grid."""
        c0, r0 = self.world_to_grid(x0, y0)
        c1, r1 = self.world_to_grid(x1, y1)
        dist = max(abs(c1 - c0), abs(r1 - r0))
        if dist <= 1:
            return self.grid[r0, c0] != 1 and self.grid[r1, c1] != 1
        cols = np.linspace(c0, c1, dist + 1, dtype=int)
        rows = np.linspace(r0, r1, dist + 1, dtype=int)
        return not np.any(self.grid[rows, cols] == 1)

    # =========================================================================
    # 1. Ohio Stadium Pinch Detection & Safe Bypass
    # =========================================================================

    def is_stadium_pinch(self, start_pos: Tuple[float, float], goal_pos: Tuple[float, float]) -> Tuple[bool, Optional[Tuple[float, float]]]:
        """
        Detects if traversing directly from start_pos to goal_pos collides with
        Ohio Stadium's massive curved concrete exterior (>200m wall).
        Returns: (is_pinched, recommended_bypass_waypoint)
        """
        sx, sy = start_pos[0], start_pos[1]
        gx, gy = goal_pos[0], goal_pos[1]

        sbox = self.stadium_box
        # Check if either point is west/inside stadium and moving eastward
        west_of_stadium = sx < sbox["min_x"] + 20.0 and (sbox["min_y"] - 20.0 <= sy <= sbox["max_y"] + 20.0)
        inside_stadium_zone = (sbox["min_x"] <= sx <= sbox["max_x"]) and (sbox["min_y"] <= sy <= sbox["max_y"])
        going_east = gx > sbox["min_x"]

        if not (west_of_stadium or inside_stadium_zone) or not going_east:
            return False, None

        # Line segment bounding box intersection with stadium
        seg_min_x = min(sx, gx)
        seg_max_x = max(sx, gx)
        seg_min_y = min(sy, gy)
        seg_max_y = max(sy, gy)

        intersects_box = not (
            seg_max_x < sbox["min_x"] or
            seg_min_x > sbox["max_x"] or
            seg_max_y < sbox["min_y"] or
            seg_min_y > sbox["max_y"]
        )

        if not intersects_box:
            return False, None

        # Choose North or South bypass based on shortest total path to goal
        d_north = (math.hypot(self.stadium_north_bypass[0] - sx, self.stadium_north_bypass[1] - sy) +
                   math.hypot(gx - self.stadium_north_bypass[0], gy - self.stadium_north_bypass[1]))
        d_south = (math.hypot(self.stadium_south_bypass[0] - sx, self.stadium_south_bypass[1] - sy) +
                   math.hypot(gx - self.stadium_south_bypass[0], gy - self.stadium_south_bypass[1]))
        bypass = self.stadium_north_bypass if d_north < d_south else self.stadium_south_bypass
        return True, bypass

    # =========================================================================
    # 2. Zone Mechanics & Early Rotation Urgency
    # =========================================================================

    def evaluate_rotation_urgency(
        self,
        my_pos: Tuple[float, float],
        zone_obs: Dict[str, Any],
        my_speed: float = 11.7,
        current_time: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        The Gatekeeper Ambush Doctrine: Rotate early into the safe zone ahead of the 15s warning.
        Computes traversal times across terrain and stadium obstacles.

        Returns: (needs_rotation, eta_seconds)
        """
        x, y = my_pos[0], my_pos[1]
        next_center = zone_obs.get("next_center", zone_obs.get("center", [0.0, 0.0]))
        next_radius = float(zone_obs.get("next_radius", zone_obs.get("radius", 380.0)))
        current_radius = float(zone_obs.get("radius", 560.0))
        current_center = zone_obs.get("center", next_center)
        shrink_start = float(zone_obs.get("shrink_start", 60.0))
        t = current_time if current_time is not None else zone_obs.get("t", 0.0)

        # Distance to current circle center vs radius
        dist_curr = math.hypot(x - current_center[0], y - current_center[1])
        is_outside = dist_curr > (current_radius - 5.0)

        # Distance to next circle center
        dist_next = math.hypot(x - next_center[0], y - next_center[1])
        # Safe buffer inside the next circle (want to be 25m inside the border)
        safe_radius = max(10.0, next_radius - 25.0)
        dist_to_safe = max(0.0, dist_next - safe_radius)

        # If already well inside safe zone, no rotation needed unless outside current
        if dist_to_safe <= 0.0 and not is_outside:
            return False, 0.0

        # Check for Stadium Pinch detour
        is_pinched, bypass_wp = self.is_stadium_pinch((x, y), (next_center[0], next_center[1]))
        if is_pinched and bypass_wp:
            # Traversal distance via bypass waypoint
            d1 = math.hypot(bypass_wp[0] - x, bypass_wp[1] - y)
            d2 = math.hypot(next_center[0] - bypass_wp[0], next_center[1] - bypass_wp[1])
            total_dist = d1 + max(0.0, d2 - safe_radius)
            obstacle_mult = 1.35
        else:
            total_dist = dist_to_safe
            obstacle_mult = 1.2  # Real terrain navigation detour factor

        # Compute traversal ETA
        eta_seconds = (total_dist / max(1.0, my_speed)) * obstacle_mult

        # Safe rotation timing:
        # Give a comfortable 5.0s safety buffer ahead of zone arrival,
        # but do not panic-rotate 30 seconds prematurely when farming is critical.
        time_to_shrink = shrink_start - t
        needs_rotation = is_outside or (time_to_shrink <= (eta_seconds + 5.0))

        return bool(needs_rotation), float(eta_seconds)

    # =========================================================================
    # 3. Chokepoint & Funnel Precomputation
    # =========================================================================

    def find_chokepoints(
        self,
        current_radius: float,
        next_center: Tuple[float, float],
        next_radius: float,
        num_samples: int = 360
    ) -> List[Chokepoint]:
        """
        Precomputes and identifies narrow alley bottlenecks along the boundary
        of the incoming zone circle where panic-rotating naive bots MUST pass through.
        Returns sorted list of Chokepoint objects.
        """
        cx, cy = float(next_center[0]), float(next_center[1])
        cache_key = (round(cx, 1), round(cy, 1), round(next_radius, 1))
        if cache_key in self._choke_cache:
            return self._choke_cache[cache_key]

        angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
        xs = cx + next_radius * np.cos(angles)
        ys = cy + next_radius * np.sin(angles)

        cols = np.round((xs - self.min_x) / self.step).astype(int)
        rows = np.round((ys - self.min_y) / self.step).astype(int)

        valid = (cols >= 0) & (cols < self.nx) & (rows >= 0) & (rows < self.ny)
        walkable = np.zeros(num_samples, dtype=bool)
        walkable[valid] = (self.grid[rows[valid], cols[valid]] == 0)

        # Detect contiguous walkable arcs
        segs = []
        in_seg = False
        start_idx = 0
        for i in range(2 * num_samples):
            idx = i % num_samples
            if walkable[idx]:
                if not in_seg:
                    in_seg = True
                    start_idx = i
            else:
                if in_seg:
                    in_seg = False
                    seg_len = i - start_idx
                    if start_idx < num_samples and seg_len < num_samples:
                        segs.append((start_idx % num_samples, seg_len))
                    if i >= num_samples:
                        break

        arc_meter_factor = 2.0 * math.pi * next_radius / num_samples
        unique_chokes: List[Chokepoint] = []
        seen_starts = set()

        for start, length in segs:
            if start in seen_starts:
                continue
            seen_starts.add(start)

            width_m = length * arc_meter_factor
            # A chokepoint funnel is narrow: width between 3.5m and 38.0m
            if not (3.5 <= width_m <= 38.0):
                continue

            mid_idx = (start + length // 2) % num_samples
            mid_angle = float(angles[mid_idx])
            px = cx + next_radius * math.cos(mid_angle)
            py = cy + next_radius * math.sin(mid_angle)

            # Inward vector toward zone center
            dx = cx - px
            dy = cy - py
            d_len = math.hypot(dx, dy)
            inward = (dx / max(0.001, d_len), dy / max(0.001, d_len))
            outward = (-inward[0], -inward[1])

            # Catchment Area Calculation:
            # Measure the size of solid obstacles immediately to the left and right
            left_blocked = 0
            for k in range(1, 90):
                if not walkable[(start - k) % num_samples]:
                    left_blocked += 1
                else:
                    break

            right_blocked = 0
            for k in range(1, 90):
                if not walkable[(start + length + k - 1) % num_samples]:
                    right_blocked += 1
                else:
                    break

            # Traffic score: ratio of blocked catchment meters to alley width
            catchment_meters = (left_blocked + right_blocked) * arc_meter_factor
            traffic_score = (catchment_meters / max(4.0, width_m)) * 10.0

            # Calculate Ambush Position behind cover at the funnel exit
            exit_pos, ambush_pos, aim_yaw = self._compute_ambush_spot(cx, cy, px, py, inward, outward)

            cp = Chokepoint(
                pos=(px, py),
                width=width_m,
                inward_dir=inward,
                outward_dir=outward,
                angle=mid_angle,
                traffic_score=traffic_score,
                exit_pos=exit_pos,
                ambush_pos=ambush_pos,
                aim_yaw=aim_yaw
            )
            unique_chokes.append(cp)

        # Sort by traffic score (highest kill potential first)
        unique_chokes.sort(key=lambda c: c.traffic_score, reverse=True)
        self._choke_cache[cache_key] = unique_chokes
        return unique_chokes

    def _compute_ambush_spot(
        self,
        cx: float,
        cy: float,
        choke_x: float,
        choke_y: float,
        inward: Tuple[float, float],
        outward: Tuple[float, float]
    ) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        """
        Finds the funnel exit and selects a position behind cover (wall/corner)
        at the exit with direct line-of-sight down the alleyway.
        """
        c0 = int(round((choke_x - self.min_x) / self.step))
        r0 = int(round((choke_y - self.min_y) / self.step))
        d_choke_to_center = math.hypot(choke_x - cx, choke_y - cy)

        best_ambush = (choke_x, choke_y)
        best_exit = (choke_x, choke_y)
        best_score = -1e9

        # Search nearby cells (8m - 22m away) for walkable cells that lead inward
        for dr in range(-20, 21, 2):
            for dc in range(-20, 21, 2):
                nr, nc = r0 + dr, c0 + dc
                if 0 <= nr < self.ny and 0 <= nc < self.nx and self.grid[nr, nc] == 0:
                    dist_cells = math.hypot(dr, dc)
                    if 8.0 <= dist_cells <= 22.0:
                        cand_x = self.min_x + nc * self.step
                        cand_y = self.min_y + nr * self.step
                        cand_to_center = math.hypot(cand_x - cx, cand_y - cy)
                        # Must be closer to center than choke (moving inward into safe zone)
                        if cand_to_center < d_choke_to_center:
                            if self.has_los(cand_x, cand_y, choke_x, choke_y):
                                # Wall presence nearby provides cover
                                sub = self.grid[max(0, nr-3):min(self.ny, nr+4), max(0, nc-3):min(self.nx, nc+4)]
                                walls = int(np.sum(sub == 1))
                                score = (d_choke_to_center - cand_to_center) * 2.0 + walls * 0.5 - abs(dist_cells - 14.0) * 0.5
                                if score > best_score:
                                    best_score = score
                                    best_ambush = (cand_x, cand_y)
                                    best_exit = (cand_x, cand_y)

        # Pre-aim angle: directly down the alley toward choke_pos
        aim_yaw = math.atan2(choke_y - best_ambush[1], choke_x - best_ambush[0])
        return best_exit, best_ambush, aim_yaw

    # =========================================================================
    # 4. Gatekeeper Ambush Doctrine
    # =========================================================================

    def select_gatekeep_position(
        self,
        my_pos: Tuple[float, float],
        chokepoints_or_choke: Union[List[Chokepoint], Chokepoint]
    ) -> Tuple[float, float]:
        """
        Positions our bot behind cover at the exit of the best chokepoint funnel.
        Returns: (ambush_x, ambush_y)
        """
        if isinstance(chokepoints_or_choke, Chokepoint):
            return chokepoints_or_choke.ambush_pos

        chokepoints = chokepoints_or_choke
        if not chokepoints:
            return my_pos

        best_cp = chokepoints[0]
        best_score = -1e9
        for cp in chokepoints:
            dist_to_cp = math.hypot(cp.ambush_pos[0] - my_pos[0], cp.ambush_pos[1] - my_pos[1])
            score = cp.traffic_score * 1.5 - dist_to_cp * 0.12
            if score > best_score:
                best_score = score
                best_cp = cp

        return best_cp.ambush_pos

    def get_ambush_aim(
        self,
        my_pos: Tuple[float, float],
        chokepoint: Chokepoint
    ) -> Tuple[float, float]:
        """
        Returns (yaw, pitch) holding aim down the throat of the chokepoint alley.
        """
        dy = chokepoint.pos[1] - my_pos[1]
        dx = chokepoint.pos[0] - my_pos[0]
        yaw = math.atan2(dy, dx)
        return yaw, 0.0

    # =========================================================================
    # 5. Endgame Survival Engine (< 5 alive)
    # =========================================================================

    def is_endgame(self, alive_count: int, zone_radius: float) -> bool:
        """
        Checks if the match is in the critical endgame phase:
        5 or fewer mechs alive, or final collapsing circle (<= 70m).
        """
        return alive_count <= 5 or zone_radius <= 70.0

    def should_avoid_combat(
        self,
        alive_count: int,
        my_hp: float,
        enemy: Optional[Dict[str, Any]],
        enemy_dist: float
    ) -> bool:
        """
        Endgame Survival Engine:
        When <= 5 alive, objective shifts to pure HP preservation (at 8:00 survivors ranked by HP).
        Avoids all optional fights; lets remaining enemies butcher each other.
        Only fights back if directly attacked at point-blank range (< 25m).
        """
        if not self.is_endgame(alive_count, 100.0):
            return False

        if not enemy:
            return True

        # Point blank self defense (< 25m): must defend ourselves
        if enemy_dist < 25.0:
            return False

        # If enemy is not attacking us, or dist >= 25m: avoid fight!
        return True

    def get_endgame_position(
        self,
        my_pos: Tuple[float, float],
        zone_obs: Dict[str, Any],
        alive_count: int
    ) -> Tuple[float, float]:
        """
        Finds the premier defensive position on high ground or deep cover
        inside the final 60m circle to maximize HP preservation.
        """
        center = zone_obs.get("next_center", zone_obs.get("center", [0.0, 0.0]))
        radius = float(zone_obs.get("next_radius", zone_obs.get("radius", 60.0)))
        cx, cy = float(center[0]), float(center[1])
        r = min(radius, 60.0)

        cache_key = (round(cx, 1), round(cy, 1), round(r, 1))
        if cache_key in self._endgame_cache:
            return self._endgame_cache[cache_key]

        c_min = max(0, int((cx - r - self.min_x) / self.step))
        c_max = min(self.nx - 1, int((cx + r - self.min_x) / self.step))
        r_min = max(0, int((cy - r - self.min_y) / self.step))
        r_max = min(self.ny - 1, int((cy + r - self.min_y) / self.step))

        # Sample grid at 2m stride for sub-millisecond execution
        stride = 2
        sub_grid = self.grid[r_min:r_max+1:stride, c_min:c_max+1:stride]

        cols, rows = np.meshgrid(
            np.arange(c_min, c_max + 1, stride),
            np.arange(r_min, r_max + 1, stride)
        )
        xs = self.min_x + cols * self.step
        ys = self.min_y + rows * self.step
        dists = np.hypot(xs - cx, ys - cy)

        # Valid cells: walkable, inside circle with 8m safety buffer from the lethal gas rim,
        # and at least 10m from the center where the chaotic deathball forms.
        valid = (sub_grid == 0) & (dists <= r - 8.0) & (dists >= 10.0)
        if not np.any(valid):
            fallback = (cx, cy)
            self._endgame_cache[cache_key] = fallback
            return fallback

        # 1. Elevation component (High Ground)
        if self.terrain is not None:
            sub_terrain = self.terrain[r_min:r_max+1:stride, c_min:c_max+1:stride]
            z_valid = sub_terrain[valid]
            z_min = float(z_valid.min())
            z_max = float(z_valid.max())
            z_norm = (sub_terrain - z_min) / max(0.1, z_max - z_min)
        else:
            z_norm = np.zeros_like(dists)

        # 2. Deep Cover component:
        dist_pref = -np.abs(dists - (r * 0.5)) / max(1.0, r)

        # Composite survival score: high ground (4.0) + distance buffer (2.0)
        total_score = np.where(valid, z_norm * 4.0 + dist_pref * 2.0, -1e9)
        best_idx = np.unravel_index(np.argmax(total_score), total_score.shape)

        best_pos = (float(xs[best_idx]), float(ys[best_idx]))
        self._endgame_cache[cache_key] = best_pos
        return best_pos


# =============================================================================
# Standalone Runnable Test Suite with Simulated Campus Scenarios
# =============================================================================

def run_all_tests():
    """Executes full test suite validating all tactical specifications."""
    print("=" * 70)
    print("CHOKE: Zone Gatekeeper & Macro Strategist — Standalone Test Suite")
    print("=" * 70)

    t_start = time.perf_counter()
    gk = ZoneGatekeeper()
    t_init = (time.perf_counter() - t_start) * 1000
    print(f"[*] ZoneGatekeeper initialized in {t_init:.2f} ms")
    print(f"    Grid: {gk.nx}x{gk.ny}, Terrain loaded: {gk.terrain is not None}")

    # -------------------------------------------------------------------------
    # Test 1: Chokepoint & Funnel Precomputation across Zone Radii
    # -------------------------------------------------------------------------
    print("\n[Test 1] Chokepoint & Funnel Precomputation across Zone Radii")
    test_radii = [380.0, 240.0, 140.0, 60.0]
    center = (41.3, -72.8)

    for r in test_radii:
        t0 = time.perf_counter()
        chokes = gk.find_chokepoints(r + 100.0, center, r)
        dt = (time.perf_counter() - t0) * 1000
        print(f"  Radius {r:5.1f}m -> Found {len(chokes):2d} chokepoints in {dt:.3f} ms")
        assert len(chokes) > 0, f"Expected chokepoints at radius {r}m"
        best = chokes[0]
        assert 3.5 <= best.width <= 38.0, f"Invalid chokepoint width: {best.width}"
        assert best.traffic_score > 0, "Expected positive funnel traffic score"
        print(f"    Top Funnel: pos=({best.pos[0]:.1f}, {best.pos[1]:.1f}), width={best.width:.1f}m, "
              f"traffic_score={best.traffic_score:.1f}, ambush=({best.ambush_pos[0]:.1f}, {best.ambush_pos[1]:.1f})")

    # -------------------------------------------------------------------------
    # Test 2: Gatekeeper Early Rotation Urgency vs Naive 15s Late Rotation
    # -------------------------------------------------------------------------
    print("\n[Test 2] Gatekeeper Early Rotation Urgency Doctrine")
    zone_test = {
        "center": [41.3, -72.8],
        "radius": 560.0,
        "next_center": [41.3, -72.8],
        "next_radius": 380.0,
        "shrink_start": 60.0,
        "shrink_end": 110.0,
        "t": 30.0
    }
    bot_pos = (41.3 + 480.0, -72.8)
    needs_rot, eta = gk.evaluate_rotation_urgency(bot_pos, zone_test, my_speed=11.7, current_time=30.0)
    print(f"  Bot at distance 480m (100m outside safe circle):")
    print(f"    Calculated traversal ETA: {eta:.1f}s")
    print(f"    Rotation urgent at t=30.0s (30s before shrink)? {needs_rot}")
    assert needs_rot is True, "Gatekeeper should trigger early rotation ahead of naive bots!"

    safe_bot = (41.3 + 200.0, -72.8)
    needs_rot_safe, _ = gk.evaluate_rotation_urgency(safe_bot, zone_test, my_speed=11.7, current_time=30.0)
    assert needs_rot_safe is False, "Bot safely inside next circle should not panic rotate"
    print(f"    Bot at distance 200m (safely inside next circle): needs_rot={needs_rot_safe} (PASS)")

    # -------------------------------------------------------------------------
    # Test 3: Ohio Stadium Pinch Avoidance
    # -------------------------------------------------------------------------
    print("\n[Test 3] Ohio Stadium Pinch Avoidance & Safe Bypass")
    west_bot = (-360.0, 100.0)
    south_goal = (41.3, -72.8)
    north_goal = (41.3, 200.0)

    is_pinched_s, bypass_s = gk.is_stadium_pinch(west_bot, south_goal)
    print(f"  West bot at {west_bot} heading to South goal {south_goal}:")
    print(f"    Detected Stadium Pinch? {is_pinched_s}")
    print(f"    Safe Bypass Waypoint: {bypass_s}")
    assert is_pinched_s is True, "Should detect stadium obstacle pinch!"
    assert bypass_s == gk.stadium_south_bypass, "Should select South bypass for southern destination!"

    is_pinched_n, bypass_n = gk.is_stadium_pinch(west_bot, north_goal)
    print(f"  West bot at {west_bot} heading to North goal {north_goal}:")
    print(f"    Detected Stadium Pinch? {is_pinched_n}")
    print(f"    Safe Bypass Waypoint: {bypass_n}")
    assert is_pinched_n is True, "Should detect stadium obstacle pinch!"
    assert bypass_n == gk.stadium_north_bypass, "Should select North bypass for northern destination!"

    needs_rot_stadium, eta_stadium = gk.evaluate_rotation_urgency(west_bot, zone_test, my_speed=11.7, current_time=20.0)
    print(f"    Stadium bypass traversal ETA: {eta_stadium:.1f}s, urgent: {needs_rot_stadium}")
    assert eta_stadium > 15.0, "Detour around stadium should reflect realistic travel ETA"

    # -------------------------------------------------------------------------
    # Test 4: Gatekeeper Ambush Doctrine Simulation
    # -------------------------------------------------------------------------
    print("\n[Test 4] Gatekeeper Ambush Doctrine Simulation")
    chokes_240 = gk.find_chokepoints(380.0, center, 240.0)
    chosen_choke = chokes_240[0]
    ambush_pos = gk.select_gatekeep_position((41.3, -72.8), chosen_choke)
    aim_yaw, _ = gk.get_ambush_aim(ambush_pos, chosen_choke)

    print(f"  Selected Chokepoint: {chosen_choke}")
    print(f"  Holding ambush at: ({ambush_pos[0]:.1f}, {ambush_pos[1]:.1f})")
    print(f"  Holding pre-aim down alley: {math.degrees(aim_yaw):.1f}°")

    assert gk.is_walkable(ambush_pos[0], ambush_pos[1]), "Ambush spot must be walkable"
    dist_ambush_center = math.hypot(ambush_pos[0] - center[0], ambush_pos[1] - center[1])
    assert dist_ambush_center <= 240.0, "Ambush spot must be inside the safe zone!"

    scrambler_pos = chosen_choke.pos
    scrambler_hp = 420.0
    scrambler_yaw = math.atan2(scrambler_pos[1] - ambush_pos[1], scrambler_pos[0] - ambush_pos[0])
    angle_error = abs(math.atan2(math.sin(aim_yaw - scrambler_yaw), math.cos(aim_yaw - scrambler_yaw)))
    print(f"  Incoming panic scrambler at alley entrance ({scrambler_pos[0]:.1f}, {scrambler_pos[1]:.1f}) [HP: {scrambler_hp}]")
    print(f"  Pre-aim angular deviation: {math.degrees(angle_error):.2f}° (< 5° allows zero-slew instant trigger!)")
    assert math.degrees(angle_error) < 5.0, "Pre-aim should align directly with incoming funnel path!"

    # -------------------------------------------------------------------------
    # Test 5: Endgame Survival Engine (< 5 alive)
    # -------------------------------------------------------------------------
    print("\n[Test 5] Endgame Survival Engine (< 5 alive)")
    assert gk.is_endgame(alive_count=4, zone_radius=50.0) is True, "alive=4 must be endgame"
    assert gk.is_endgame(alive_count=12, zone_radius=300.0) is False, "alive=12 is not endgame"

    enemy_far = {"id": 2, "hp": 1200.0, "weapons": ["gun"]}
    avoid_far = gk.should_avoid_combat(alive_count=3, my_hp=1400.0, enemy=enemy_far, enemy_dist=80.0)
    print(f"  alive=3: Enemy at 80m -> avoid fight? {avoid_far}")
    assert avoid_far is True, "Should avoid optional long-range fights in endgame!"

    enemy_close = {"id": 3, "hp": 600.0, "weapons": ["gun"]}
    avoid_close = gk.should_avoid_combat(alive_count=3, my_hp=1400.0, enemy=enemy_close, enemy_dist=15.0)
    print(f"  alive=3: Point-blank threat at 15m -> avoid fight? {avoid_close}")
    assert avoid_close is False, "Must engage in point-blank self-defense!"

    final_zone = {"center": [41.3, -72.8], "radius": 60.0, "next_radius": 60.0}
    endgame_pos = gk.get_endgame_position((41.3, -72.8), final_zone, alive_count=3)
    endgame_dist = math.hypot(endgame_pos[0] - center[0], endgame_pos[1] - center[1])
    print(f"  Final 60m Circle -> Selected Defensive Position: ({endgame_pos[0]:.1f}, {endgame_pos[1]:.1f})")
    print(f"    Dist to center: {endgame_dist:.1f}m (safe buffer from 260 DPS gas rim)")
    assert gk.is_walkable(endgame_pos[0], endgame_pos[1]), "Endgame position must be walkable"
    assert endgame_dist <= 52.0, "Endgame position must maintain safety buffer from gas rim"

    # -------------------------------------------------------------------------
    # Test 6: Micro-Benchmark Execution Time Constraint (< 2.0 ms)
    # -------------------------------------------------------------------------
    print("\n[Test 6] Micro-Benchmark Execution Time Constraint (< 2.0 ms)")
    iterations = 500
    t_bench_start = time.perf_counter()

    for i in range(iterations):
        gk.evaluate_rotation_urgency((41.3 + (i % 100), -72.8), zone_test, my_speed=11.7, current_time=35.0)
        chks = gk.find_chokepoints(380.0, center, 240.0)
        gk.select_gatekeep_position((41.3, -72.8), chks)
        gk.should_avoid_combat(4, 1400.0, enemy_far, 60.0)
        gk.get_endgame_position((41.3, -72.8), final_zone, 4)

    t_bench_total = (time.perf_counter() - t_bench_start) * 1000
    avg_per_tick = t_bench_total / iterations
    print(f"  Executed {iterations} complete tick cycles in {t_bench_total:.2f} ms")
    print(f"  Average execution time per tick: {avg_per_tick:.4f} ms")
    assert avg_per_tick < 2.0, f"Execution time constraint violated: {avg_per_tick:.4f} ms >= 2.0 ms"
    print(f"  >> Performance Constraint PASSED (Target < 2.0 ms, Measured: {avg_per_tick:.4f} ms, {(2.0 / avg_per_tick):.1f}x headroom)")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
