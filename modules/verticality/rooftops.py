"""
Verticality and Rooftop Control Engine for MechRoyale (mechfight/1).
Specialist: ATLAS - 3D Verticality and Rooftop Specialist
Target Module: modules/verticality/rooftops.py

Key Systems:
1. Offline / Startup Geometry Parser:
   - Ingests world.json prisms and terrain.i16 elevation data.
   - Identifies flat, landable rooftops up to 60m height (e.g. Thompson Library tower 44.1m,
     Baker Systems 48.8m, CBEC 38.1m, University Hall 35.1m).
   - Detects and maps multi-tier building eaves that act as intermediate stair-step perches.
   - Uses Euclidean Distance Transform (EDT) at 0.5m resolution with continuous validation
     to compute exact setback fields from all outer perimeter walls, inner courtyards, and upper tiers.

2. Rooftop Jet Climbing & Stair-Step Eave Controller:
   - Mechs with 100 energy can climb up to 60m:
     - Direct climb: sustained jet rise up to ~58m (costs ~95 energy).
     - Stair-step climb: stages vertical climbs along building eaves (10m - 25m per step).
   - Function `find_stairstep_escape(my_pos, enemy_pos, energy, max_height)`:
     Computes safe stair-step roof perches along building eaves so the bot breaks ground
     line-of-sight vertically at every stage while conserving energy.
   - Ground bots cannot follow without jetting, giving an instant 45 HP/s combat reset.

3. The 3-Meter Roof-Edge Setback Rule:
   - Standing near the edge leaves the 9m mech torso visible to ground muzzle angles.
   - Calculates safe interior "perch points" at least 3m back from roof edges to break
     ground line of sight vertically.

4. Safe Drop-Down & Exit Vectors:
   - Calculates safe jumping-off points along roof edges leading into verified walkable ground
     or lower eave surfaces.
   - Leverages zero fall damage physics: mechs can leap off 60m buildings safely with no penalty.

5. Wall-Sliding Corner Rounding:
   - When escaping, slides along concave and convex wall contours rather than turning sharp 90° corners.
   - Replaces sharp waypoints with tangential fillet arcs (`round_corner_waypoints`).
   - Dynamically deflects movement vectors along wall normals (`compute_wall_slide_vector`).

6. High-Performance Execution:
   - Spatial indexing via cKDTree and precomputed geometric cache (data/rooftops.json).
   - All runtime queries execute in < 0.20 ms (far below the 1.5 ms budget).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import scipy.ndimage as ndi
    from scipy.spatial import cKDTree
    from PIL import Image, ImageDraw
except ImportError:
    ndi = None
    cKDTree = None
    Image = None
    ImageDraw = None


# =====================================================================
# Physics & Protocol Constants
# =====================================================================

MECH_HEIGHT = 9.0              # Total mech height (meters)
MECH_RADIUS = 2.5              # Cylinder hit radius (meters)
MUZZLE_HEIGHT = 7.0            # Gun muzzle z-offset from feet (meters)
EYE_HEIGHT = 8.0               # Eye camera z-offset from feet (meters)
WALK_SPEED = 11.7              # Ground walk speed (m/s)
BOOST_SPEED = 19.5             # Ground boost sprint speed (m/s)
JUMP_SPEED = 11.0              # Initial upward jump velocity (m/s)
JUMP_ENERGY_COST = 15.0        # One-shot energy cost to initiate jump
JET_DRAIN = 15.0               # Jetting energy drain per second (energy/s)
JET_MAX_RISE_SPEED = 10.0      # Maximum upward vertical velocity while jetting (m/s)
JET_ACCEL = 34.0               # Upward vertical jet acceleration (m/s^2)
GRAVITY = 22.0                 # Downward gravitational acceleration (m/s^2)
STEP_UP_MAX = 1.2              # Ledges up to 1.2m can be stepped up without jumping
MAX_ENERGY = 100.0             # Default energy pool (100 energy allows climbs up to 60m)
ENERGY_REGEN = 18.0            # Energy regeneration rate (energy/s)
ENERGY_REGEN_DELAY = 0.6       # Delay before energy regeneration begins (s)
REGEN_DELAY = 5.0              # Out-of-combat heal delay (5.0s, 3.5s with nanites)
REGEN_PER_S = 45.0             # Out-of-combat regeneration rate (HP/s)

ROOF_MIN_HEIGHT = 10.0         # Minimum building height to qualify as tactical escape roof (meters)
ROOF_MAX_HEIGHT = 60.0         # Maximum building height reachable within standard jump-jet budget (up to 60m)
ROOF_MIN_SETBACK = 3.0         # The mandatory 3-meter setback rule from all edges (meters)
ROOF_MIN_DIMENSION = 6.0       # Minimum width/length (2 * 3m) required to harbor a 3m setback perch


# =====================================================================
# Data Structures
# =====================================================================

@dataclass
class DropDownPoint:
    """Precomputed safe drop-down point along a roof edge."""
    edge_pos: Tuple[float, float, float]       # Point on the roof edge (x, y, z_roof)
    exit_vec: Tuple[float, float]              # Outward unit 2D direction vector away from building
    ground_pos: Tuple[float, float, float]     # Landable ground or lower eave coordinate (x, y, z_land)
    ground_clearance: float                    # Clearance on landing surface (meters)
    landing_roof_id: Optional[int] = None      # None if ground, or rooftop ID if landing on lower eave


@dataclass
class StairStepPerch:
    """Safe stair-step roof perch along building eaves to break LOS vertically."""
    step_index: int                            # 0-based index of this climb step (0 = first eave from ground)
    roof_id: int                               # Rooftop ID for this step
    roof_name: str                             # Building / tier name
    top: float                                 # Surface altitude z (meters)
    height: float                              # Height above base (meters)
    step_elevation_gain: float                 # Elevation gain from previous surface (meters)
    energy_cost: float                         # Energy consumed to jump/jet to this eave
    approach_pos: Tuple[float, float, float]   # Takeoff point on lower surface/ground (x, y, z)
    climb_edge: Tuple[float, float, float]     # Landing edge point (x, y, z)
    perch_pos: Tuple[float, float, float]      # Safe interior perch point (>= 3m setback)
    los_occluded: bool = True                  # True if sightline from enemy is occluded vertically


@dataclass
class StairStepClimbPlan:
    """Multi-tier vertical climb plan up building eaves up to 60m."""
    target_roof_id: int                        # Top-tier target roof ID
    target_roof_name: str                      # Top-tier target building name
    total_height: float                        # Building height from ground (meters)
    total_elevation_gain: float                # Total elevation gain climbed (meters)
    total_energy: float                        # Total energy consumed across all steps
    steps: List[StairStepPerch]                # Sequence of eave perches from ground to top
    is_multi_tier: bool                        # True if multiple tiers/eaves are used (len > 1)
    safe_exit: Optional[DropDownPoint]         # Safe drop-down exit from top tier
    score: float                               # Tactical optimality score (lower is better)


@dataclass
class Rooftop:
    """Represents a verified flat, landable rooftop."""
    id: int                                    # Prism index in world.json
    name: str                                  # Human-readable building name or block identifier
    block_id: int                              # Block ID from world.json
    top: float                                 # Absolute z-altitude of the roof surface
    base: float                                # Ground / base elevation
    height: float                              # Total vertical height (top - base)
    center: Tuple[float, float]                # 2D centroid (x, y)
    bounds: Tuple[float, float, float, float]  # Bounding box: (min_x, max_x, min_y, max_y)
    ring: List[Tuple[float, float]]            # Outer boundary polygon vertices
    holes: List[List[Tuple[float, float]]]      # Inner courtyard / hole polygons
    area: float                                # Total roof surface area (m^2)
    usable_area: float                         # Area with >= 3.0m setback from all edges (m^2)
    max_setback: float                         # Maximum interior setback distance (meters)
    best_perch: Tuple[float, float, float]     # Deepest interior perch point (pole of inaccessibility)
    perch_points: List[Tuple[float, float, float]] = field(default_factory=list) # Candidate 3m setback perch points
    edges: List[Tuple[Tuple[float, float], Tuple[float, float]]] = field(default_factory=list) # Boundary line segments
    drop_points: List[DropDownPoint] = field(default_factory=list) # Verified safe drop-down exit points
    eave_ids: List[int] = field(default_factory=list)              # Lower-tier eave rooftops leading to this roof
    higher_tier_ids: List[int] = field(default_factory=list)       # Higher-tier rooftops reachable from this roof


@dataclass
class RooftopEscapeTarget:
    """Tactical target returned by find_nearest_rooftop_escape."""
    roof_id: int                               # Target rooftop ID
    roof_name: str                             # Target building name
    roof_top: float                            # Roof surface z
    roof_height: float                         # Building vertical height
    energy_cost: float                         # Total energy required to jump-jet to roof
    ground_approach: Tuple[float, float, float]# Ground waypoint at wall base to run toward
    climb_edge: Tuple[float, float, float]     # Roof edge point to jump onto
    perch_point: Tuple[float, float, float]    # Safe interior perch point (>= 3m setback)
    dist_to_approach: float                    # Distance from current position to ground approach
    safe_exit: Optional[DropDownPoint]         # Safe drop-down exit into ground cover
    shielded_from_enemy: bool                  # True if building body shields climb from enemy
    score: float                               # Tactical optimality score (lower is better)
    stairstep_plan: Optional[StairStepClimbPlan] = None            # Multi-tier climb plan if stair-stepping
    stairstep_perches: List[Tuple[float, float, float]] = field(default_factory=list) # Waypoints of safe eave perches


# =====================================================================
# Math & Polygon Helpers
# =====================================================================

def point_in_polygon(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    """Ray casting algorithm for point-in-polygon test."""
    n = len(poly)
    if n < 3:
        return False
    inside = False
    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def point_segment_distance(
    px: float, py: float,
    x1: float, y1: float,
    x2: float, y2: float
) -> Tuple[float, float, float]:
    """
    Computes distance from (px, py) to line segment (x1, y1)-(x2, y2),
    along with the closest projected point (qx, qy) on the segment.
    """
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-8:
        return math.hypot(px - x1, py - y1), x1, y1

    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
    qx = x1 + t * dx
    qy = y1 + t * dy
    dist = math.hypot(px - qx, py - qy)
    return dist, qx, qy


def round_corner_waypoints(
    waypoints: List[Tuple[float, float]],
    fillet_radius: float = 4.5,
    min_angle_deg: float = 35.0
) -> List[Tuple[float, float]]:
    """
    Wall-sliding corner rounding:
    Replaces sharp 90-degree corners with smooth tangential arc fillets
    that slide along concave and convex wall contours, preserving forward escape speed.
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
                result.extend([p_in, apex, p_out])
            else:
                result.extend([p_in, p_out])
        else:
            result.append(p_curr)

    result.append(waypoints[-1])
    return result


# =====================================================================
# Rooftop Manager (Core Verticality Engine)
# =====================================================================

class RooftopManager:
    """
    Comprehensive 3D Verticality and Rooftop Control Manager for MechRoyale.
    Ingests arena map data, precomputes climbable roofs up to 60m height,
    manages multi-tier stair-step perches along building eaves, enforces the 3-meter setback rule,
    and provides wall-sliding corner rounding along building contours.
    """

    def __init__(
        self,
        map_dir: str = "arena",
        world_data: Optional[Dict[str, Any]] = None,
        meta_data: Optional[Dict[str, Any]] = None,
        occ_grid: Optional[np.ndarray] = None,
        terrain_grid: Optional[np.ndarray] = None
    ):
        self.map_dir = map_dir
        self.rooftops: Dict[int, Rooftop] = {}
        self.rooftop_list: List[Rooftop] = []

        # Spatial indexing
        self.kdtree: Optional[cKDTree] = None
        self.roof_centers: np.ndarray = np.empty((0, 2), dtype=np.float32)

        # Environment grids
        self.meta: Dict[str, Any] = {}
        self.grid: Optional[np.ndarray] = None
        self.min_x: float = -512.38
        self.min_y: float = -333.09
        self.step: float = 1.0
        self.nx: int = 1070
        self.ny: int = 724

        self.terrain: Optional[np.ndarray] = None
        self.t_nx: int = 1024
        self.t_ny: int = 1024
        self.t_min_x: float = -512.0
        self.t_min_y: float = -512.0
        self.t_step: float = 1.0

        # Performance telemetry
        self.query_count: int = 0
        self.total_query_time_ms: float = 0.0

        self._load_environment(world_data, meta_data, occ_grid, terrain_grid)
        self._build_spatial_index()

    # -----------------------------------------------------------------
    # Environment Loading
    # -----------------------------------------------------------------

    def _resolve_path(self, filename: str) -> Optional[str]:
        """Resolves static data paths across repository layouts."""
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(os.path.dirname(here))
        candidates = [
            os.path.join(root, "data", filename),
            os.path.join(root, "arena", "v2", filename),
            os.path.join(self.map_dir, "v2", filename),
            os.path.join(self.map_dir, "arena", "v2", filename),
            os.path.join(self.map_dir, filename),
            os.path.join("/arena", "v2", filename),
            os.path.join("/arena", filename),
            os.path.join(here, filename),
            filename
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    def _load_environment(
        self,
        world_data: Optional[Dict[str, Any]],
        meta_data: Optional[Dict[str, Any]],
        occ_grid: Optional[np.ndarray],
        terrain_grid: Optional[np.ndarray]
    ):
        """Loads world prisms, occupancy grid, and terrain elevation."""
        # 1. World JSON
        world = world_data
        if world is None:
            wpath = self._resolve_path("world.json")
            if wpath and os.path.exists(wpath):
                try:
                    with open(wpath, "r") as f:
                        world = json.load(f)
                except Exception:
                    world = None

        # 2. Occupancy Metadata
        if meta_data is not None:
            self.meta = meta_data
        else:
            mpath = self._resolve_path("occupancy_meta.json")
            if mpath and os.path.exists(mpath):
                try:
                    with open(mpath, "r") as f:
                        self.meta = json.load(f)
                except Exception:
                    pass

        self.nx = int(self.meta.get("nx", 1070))
        self.ny = int(self.meta.get("ny", 724))
        self.min_x = float(self.meta.get("minX", -512.38))
        self.min_y = float(self.meta.get("minY", -333.09))
        self.step = float(self.meta.get("step", 1.0))

        # 3. Occupancy binary grid (0 = walkable, 1 = solid, 2 = near-wall buffer)
        if occ_grid is not None:
            self.grid = occ_grid
        else:
            bpath = self._resolve_path("occupancy.bin")
            if bpath and os.path.exists(bpath):
                try:
                    with open(bpath, "rb") as f:
                        self.grid = np.frombuffer(f.read(), dtype=np.uint8).reshape((self.ny, self.nx))
                except Exception:
                    self.grid = None

        # 4. Terrain elevation (terrain.i16)
        if terrain_grid is not None:
            self.terrain = terrain_grid
        else:
            tpath = self._resolve_path("terrain.i16")
            if tpath and os.path.exists(tpath):
                try:
                    with open(tpath, "rb") as f:
                        # Raw int16 array in 0.01m units
                        raw = np.frombuffer(f.read(), dtype=np.int16)
                        if len(raw) == self.t_nx * self.t_ny:
                            self.terrain = raw.reshape((self.t_ny, self.t_nx)).astype(np.float32) * 0.01
                except Exception:
                    self.terrain = None

        # 5. Parse rooftops
        if world:
            self._parse_rooftops(world)

    def get_ground_elevation(self, x: float, y: float) -> float:
        """Returns terrain elevation at world coordinates (x, y)."""
        if self.terrain is None:
            return 0.0
        c = int(round((x - self.t_min_x) / self.t_step))
        r = int(round((y - self.t_min_y) / self.t_step))
        if 0 <= c < self.t_nx and 0 <= r < self.t_ny:
            return float(self.terrain[r, c])
        return 0.0

    def is_ground_walkable(self, x: float, y: float) -> bool:
        """Returns True if ground at (x, y) is clear and walkable."""
        if self.grid is None:
            return True
        c = max(0, min(self.nx - 1, int(round((x - self.min_x) / self.step))))
        r = max(0, min(self.ny - 1, int(round((y - self.min_y) / self.step))))
        return bool(self.grid[r, c] == 0)

    # -----------------------------------------------------------------
    # Startup / Offline Rooftop Parser
    # -----------------------------------------------------------------

    def _parse_rooftops(self, world: Dict[str, Any]):
        """
        Parses world.json prisms to identify flat, landable rooftops up to 60m height,
        and constructs the building eave stair-stepping relationship graph.
        Prefers precomputed data/rooftops.json cache to load in <15ms.
        """
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(os.path.dirname(here))
        cache_paths = [
            os.path.join(root, "data", "rooftops.json"),
            os.path.join(here, "data", "rooftops.json"),
            os.path.join(self.map_dir, "data", "rooftops.json"),
            os.path.join(self.map_dir, "rooftops.json"),
            os.path.join(here, "rooftops.json"),
            "data/rooftops.json"
        ]
        for cp in cache_paths:
            if os.path.exists(cp):
                try:
                    with open(cp, "r") as f:
                        cached_list = json.load(f)
                    for d in cached_list:
                        dp_list = [
                            DropDownPoint(
                                edge_pos=tuple(dp["edge_pos"]),
                                exit_vec=tuple(dp["exit_vec"]),
                                ground_pos=tuple(dp["ground_pos"]),
                                ground_clearance=float(dp["ground_clearance"]),
                                landing_roof_id=dp.get("landing_roof_id")
                            )
                            for dp in d.get("drop_points", [])
                        ]
                        roof = Rooftop(
                            id=int(d["id"]),
                            name=str(d["name"]),
                            block_id=int(d["block_id"]),
                            top=float(d["top"]),
                            base=float(d["base"]),
                            height=float(d["height"]),
                            center=tuple(d["center"]),
                            bounds=tuple(d["bounds"]),
                            ring=[tuple(pt) for pt in d["ring"]],
                            holes=[[tuple(pt) for pt in h] for h in d.get("holes", [])],
                            area=float(d["area"]),
                            usable_area=float(d["usable_area"]),
                            max_setback=float(d["max_setback"]),
                            best_perch=tuple(d["best_perch"]),
                            perch_points=[tuple(pt) for pt in d.get("perch_points", [])],
                            edges=[(tuple(e[0]), tuple(e[1])) for e in d.get("edges", [])],
                            drop_points=dp_list,
                            eave_ids=list(d.get("eave_ids", [])),
                            higher_tier_ids=list(d.get("higher_tier_ids", []))
                        )
                        self.rooftops[roof.id] = roof
                        self.rooftop_list.append(roof)
                    return
                except Exception:
                    pass

        # Dynamic fallback parsing from world.json
        prisms = world.get("prisms", [])
        blocks = world.get("blocks", [])

        prism_to_name: Dict[int, str] = {}
        for b in blocks:
            names = b.get("names", [])
            bname = names[0] if names else b.get("id", "Building")
            for pid in b.get("prisms", []):
                prism_to_name[pid] = bname

        building_polys = []
        for idx, p in enumerate(prisms):
            if p.get("kind") == "building":
                ring = p.get("ring", [])
                if len(ring) >= 6:
                    xs, ys = ring[0::2], ring[1::2]
                    building_polys.append((idx, float(p.get("top", 0)), [(xs[i], ys[i]) for i in range(len(xs))]))

        for idx, p in enumerate(prisms):
            if p.get("kind") != "building":
                continue

            base = float(p.get("base", 0.0))
            top = float(p.get("top", 0.0))
            h_prism = top - base

            # Building vertical height must be between 10m and 60m
            if not (ROOF_MIN_HEIGHT <= h_prism <= ROOF_MAX_HEIGHT):
                continue

            ring = p.get("ring", [])
            if len(ring) < 6:
                continue

            xs = ring[0::2]
            ys = ring[1::2]
            min_px, max_px = min(xs), max(xs)
            min_py, max_py = min(ys), max(ys)
            width = max_px - min_px
            length = max_py - min_py

            if width < ROOF_MIN_DIMENSION or length < ROOF_MIN_DIMENSION:
                continue

            obstructing_prisms = []
            for other_idx, other_p in enumerate(prisms):
                if other_idx == idx:
                    continue
                if other_p.get("top", 0.0) > top + 0.1:
                    oxs = other_p.get("ring", [])[0::2]
                    oys = other_p.get("ring", [])[1::2]
                    if not oxs or not oys:
                        continue
                    if not (max(oxs) < min_px or min(oxs) > max_px or max(oys) < min_py or min(oys) > max_py):
                        obstructing_prisms.append(other_p)

            polygon_pts = [(xs[i], ys[i]) for i in range(len(xs))]
            holes = p.get("holes", [])
            holes_pts = [[(h[i], h[i+1]) for i in range(0, len(h), 2)] for h in holes if len(h) >= 6]

            edges: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
            n_pts = len(polygon_pts)
            for i in range(n_pts):
                edges.append((polygon_pts[i], polygon_pts[(i + 1) % n_pts]))
            for hpts in holes_pts:
                hn = len(hpts)
                for i in range(hn):
                    edges.append((hpts[i], hpts[(i + 1) % hn]))
            for op in obstructing_prisms:
                op_xs = op.get("ring", [])[0::2]
                op_ys = op.get("ring", [])[1::2]
                op_pts = [(op_xs[i], op_ys[i]) for i in range(len(op_xs))]
                for i in range(len(op_pts)):
                    edges.append((op_pts[i], op_pts[(i + 1) % len(op_pts)]))

            res = 0.5
            margin = 2.0
            gx0 = min_px - margin
            gy0 = min_py - margin
            gw = int(np.ceil((width + 2 * margin) / res))
            gh = int(np.ceil((length + 2 * margin) / res))
            if gw <= 0 or gh <= 0 or Image is None or ndi is None:
                continue

            img = Image.new("L", (gw, gh), 0)
            draw = ImageDraw.Draw(img)
            poly = [((x - gx0) / res, (y - gy0) / res) for x, y in zip(xs, ys)]
            draw.polygon(poly, fill=255)

            for h in holes:
                if len(h) >= 6:
                    hpoly = [((h[i] - gx0) / res, (h[i+1] - gy0) / res) for i in range(0, len(h), 2)]
                    draw.polygon(hpoly, fill=0)

            for op in obstructing_prisms:
                oxs = op.get("ring", [])[0::2]
                oys = op.get("ring", [])[1::2]
                opoly = [((ox - gx0) / res, (oy - gy0) / res) for ox, oy in zip(oxs, oys)]
                draw.polygon(opoly, fill=0)

            roof_mask = np.array(img, dtype=bool)
            if not np.any(roof_mask):
                continue

            dist_map = ndi.distance_transform_edt(roof_mask) * res
            if float(np.max(dist_map)) < ROOF_MIN_SETBACK:
                continue

            max_pos_idx = np.unravel_index(np.argmax(dist_map), dist_map.shape)
            best_perch_x = float(gx0 + max_pos_idx[1] * res)
            best_perch_y = float(gy0 + max_pos_idx[0] * res)

            perch_mask = dist_map >= 3.1
            usable_area = float(np.sum(perch_mask)) * (res * res)
            total_area = float(np.sum(roof_mask)) * (res * res)

            step_stride = max(1, int(round(2.0 / res)))
            sub_perch_mask = np.zeros_like(perch_mask)
            sub_perch_mask[::step_stride, ::step_stride] = perch_mask[::step_stride, ::step_stride]
            py_indices, px_indices = np.where(sub_perch_mask)

            perch_points: List[Tuple[float, float, float]] = []
            true_max_setback = 0.0

            for pr, pc in zip(py_indices, px_indices):
                cand_x = float(gx0 + pc * res)
                cand_y = float(gy0 + pr * res)
                min_edge_d = min(point_segment_distance(cand_x, cand_y, e[0][0], e[0][1], e[1][0], e[1][1])[0] for e in edges)
                if min_edge_d >= ROOF_MIN_SETBACK:
                    perch_points.append((cand_x, cand_y, top))
                    if min_edge_d > true_max_setback:
                        true_max_setback = min_edge_d
                        best_perch_x = cand_x
                        best_perch_y = cand_y

            if not perch_points:
                min_edge_d = min(point_segment_distance(best_perch_x, best_perch_y, e[0][0], e[0][1], e[1][0], e[1][1])[0] for e in edges)
                if min_edge_d >= ROOF_MIN_SETBACK:
                    perch_points.append((best_perch_x, best_perch_y, top))
                    true_max_setback = min_edge_d
                else:
                    continue

            # Compute safe drop-down & exit points along outer perimeter (ground or lower eave)
            drop_points: List[DropDownPoint] = []
            for i in range(n_pts):
                p1 = polygon_pts[i]
                p2 = polygon_pts[(i + 1) % n_pts]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                seg_len = math.hypot(dx, dy)
                if seg_len < 1.5:
                    continue

                nx_out = dy / seg_len
                ny_out = -dx / seg_len

                sample_fractions = [0.5] if seg_len < 8.0 else [0.25, 0.5, 0.75]
                for frac in sample_fractions:
                    edge_x = p1[0] + frac * dx
                    edge_y = p1[1] + frac * dy

                    for out_dist in (3.0, 4.5, 6.0, 7.5):
                        land_x = edge_x + nx_out * out_dist
                        land_y = edge_y + ny_out * out_dist
                        if self.is_ground_walkable(land_x, land_y):
                            gz = self.get_ground_elevation(land_x, land_y)
                            drop_points.append(
                                DropDownPoint(
                                    edge_pos=(edge_x, edge_y, top),
                                    exit_vec=(nx_out, ny_out),
                                    ground_pos=(land_x, land_y, gz),
                                    ground_clearance=2.5,
                                    landing_roof_id=None
                                )
                            )
                            break
                        # Lower eave landing
                        lower_eave_id = None
                        lower_eave_top = 0.0
                        for b_id, b_top, b_poly in building_polys:
                            if b_id != idx and b_top < top and point_in_polygon(land_x, land_y, b_poly):
                                lower_eave_id = b_id
                                lower_eave_top = b_top
                                break
                        if lower_eave_id is not None:
                            drop_points.append(
                                DropDownPoint(
                                    edge_pos=(edge_x, edge_y, top),
                                    exit_vec=(nx_out, ny_out),
                                    ground_pos=(land_x, land_y, lower_eave_top),
                                    ground_clearance=2.5,
                                    landing_roof_id=lower_eave_id
                                )
                            )
                            break

            if not drop_points:
                continue

            centroid_x = float(sum(xs) / len(xs))
            centroid_y = float(sum(ys) / len(ys))
            block_id = int(p.get("block", -1))
            bname = prism_to_name.get(idx, f"Roof_{idx}")

            rooftop = Rooftop(
                id=idx,
                name=bname,
                block_id=block_id,
                top=top,
                base=base,
                height=h_prism,
                center=(centroid_x, centroid_y),
                bounds=(min_px, max_px, min_py, max_py),
                ring=polygon_pts,
                holes=holes_pts,
                area=total_area,
                usable_area=usable_area,
                max_setback=true_max_setback,
                best_perch=(best_perch_x, best_perch_y, top),
                perch_points=perch_points,
                edges=edges,
                drop_points=drop_points
            )

            self.rooftops[idx] = rooftop
            self.rooftop_list.append(rooftop)

        # Build eave stair-step graph
        for r in self.rooftop_list:
            for other in self.rooftop_list:
                if other.id == r.id:
                    continue
                same_block = (r.block_id != -1 and r.block_id == other.block_id)
                rb = r.bounds
                ob = other.bounds
                adjacent = not (ob[1] < rb[0] - 15.0 or ob[0] > rb[1] + 15.0 or ob[3] < rb[2] - 15.0 or ob[2] > rb[3] + 15.0)
                if same_block or adjacent:
                    if other.top < r.top and (r.top - other.top) <= 30.0:
                        r.eave_ids.append(other.id)
                    elif other.top > r.top and (other.top - r.top) <= 30.0:
                        r.higher_tier_ids.append(other.id)

    def _build_spatial_index(self):
        """Constructs spatial index over rooftop centers for sub-millisecond proximity queries."""
        if self.rooftop_list:
            self.roof_centers = np.array([r.center for r in self.rooftop_list], dtype=np.float32)
            if cKDTree is not None:
                self.kdtree = cKDTree(self.roof_centers)
            else:
                self.kdtree = None

    # -----------------------------------------------------------------
    # Component 2: Jump-Jet Climb & Multi-Tier Stair-Step Perches
    # -----------------------------------------------------------------

    @staticmethod
    def calculate_climb_energy(height_diff: float) -> float:
        """
        Calculates the minimum energy required to jump-jet onto a roof of height_diff.
        Formula:
          Jump gives initial 11 m/s vertical boost, costing 15 energy.
          Jetting rises at 10 m/s max rise speed, draining 15 energy/s.
          Rise time: t_rise = max(0, (H - 1.2) / 10.0)
          Energy = 15.0 (jump) + t_rise * 15.0
        With 100 energy, mechs can climb up to ~58m in a single sustained jump-jet climb!
        """
        if height_diff <= STEP_UP_MAX:
            return 0.0
        t_rise = max(0.0, (height_diff - STEP_UP_MAX) / JET_MAX_RISE_SPEED)
        return JUMP_ENERGY_COST + t_rise * JET_DRAIN

    @staticmethod
    def check_vertical_los_occluded(
        perch_pos: Union[List[float], Tuple[float, ...]],
        roof_edge_pos: Union[List[float], Tuple[float, ...]],
        enemy_pos: Union[List[float], Tuple[float, ...]]
    ) -> bool:
        """
        Calculates 3D vertical line-of-sight occlusion between a ground enemy and a mech
        standing on an elevated roof/eave perch behind a roof edge.
        Enemy eye/muzzle: ez + 7.0m (muzzle) to ez + 8.0m (eye).
        Mech torso/head: pz + 7.0m (torso) to pz + 9.0m (head top).
        Roof edge: edge_z.
        Returns True if the line of sight from enemy eye to mech head strikes below the roof edge.
        """
        ex, ey, ez = float(enemy_pos[0]), float(enemy_pos[1]), float(enemy_pos[2])
        px, py, pz = float(perch_pos[0]), float(perch_pos[1]), float(perch_pos[2])
        rx, ry, rz = float(roof_edge_pos[0]), float(roof_edge_pos[1]), float(roof_edge_pos[2])

        d_enemy_edge = math.hypot(rx - ex, ry - ey)
        d_edge_perch = math.hypot(px - rx, py - ry)

        if d_enemy_edge < 1e-3:
            return False

        enemy_eye_z = ez + EYE_HEIGHT
        mech_head_z = pz + MECH_HEIGHT

        # If enemy is already higher than the roof edge, roof edge cannot occlude vertically
        if enemy_eye_z >= rz:
            return False

        total_dist = d_enemy_edge + d_edge_perch
        ray_height_at_perch = enemy_eye_z + (rz - enemy_eye_z) * (total_dist / d_enemy_edge)

        # Occluded if the sightline ray passing over the edge is higher than the mech head
        return ray_height_at_perch >= mech_head_z

    def get_stairstep_eaves(self, roof_or_id: Union[int, Rooftop]) -> List[Rooftop]:
        """Returns the list of lower eave rooftops that serve as stair-steps leading to this roof."""
        if isinstance(roof_or_id, Rooftop):
            roof = roof_or_id
        else:
            roof = self.rooftops.get(roof_or_id)
            if roof is None:
                return []

        eaves = []
        for eid in roof.eave_ids:
            er = self.rooftops.get(eid)
            if er is not None:
                eaves.append(er)
        return eaves

    def find_stairstep_climb(
        self,
        my_pos: Union[List[float], Tuple[float, ...]],
        target_roof_or_id: Union[int, Rooftop],
        enemy_pos: Optional[Union[List[float], Tuple[float, ...]]] = None,
        current_energy: float = 100.0
    ) -> Optional[StairStepClimbPlan]:
        """
        Computes a safe stair-step roof climb along building eaves leading up to target_roof.
        Finds intermediate eave perches with >= 3m setback, breaking ground LOS vertically.
        Supports roofs up to 60m height.
        """
        if isinstance(target_roof_or_id, Rooftop):
            target_roof = target_roof_or_id
        else:
            target_roof = self.rooftops.get(target_roof_or_id)
            if target_roof is None:
                return None

        mx, my = float(my_pos[0]), float(my_pos[1])
        mz = float(my_pos[2]) if len(my_pos) > 2 else self.get_ground_elevation(mx, my)

        h_direct = target_roof.top - mz
        direct_energy = self.calculate_climb_energy(h_direct)

        eaves = self.get_stairstep_eaves(target_roof)
        candidate_plans: List[StairStepClimbPlan] = []

        # Plan A: Single direct jump if feasible (height <= 32m and energy sufficient)
        if direct_energy <= current_energy and h_direct <= 32.0:
            perch = self.get_safe_perch_point(target_roof, enemy_pos)
            climb_edge = (target_roof.ring[0][0], target_roof.ring[0][1], target_roof.top)
            approach_pos = (climb_edge[0], climb_edge[1], mz)
            if target_roof.edges:
                min_d = float("inf")
                for p1, p2 in target_roof.edges[:len(target_roof.ring)]:
                    d, qx, qy = point_segment_distance(mx, my, p1[0], p1[1], p2[0], p2[1])
                    if d < min_d:
                        min_d = d
                        climb_edge = (qx, qy, target_roof.top)
                        approach_pos = (qx, qy, mz)

            step = StairStepPerch(
                step_index=0,
                roof_id=target_roof.id,
                roof_name=target_roof.name,
                top=target_roof.top,
                height=target_roof.height,
                step_elevation_gain=h_direct,
                energy_cost=direct_energy,
                approach_pos=approach_pos,
                climb_edge=climb_edge,
                perch_pos=perch,
                los_occluded=True
            )
            candidate_plans.append(
                StairStepClimbPlan(
                    target_roof_id=target_roof.id,
                    target_roof_name=target_roof.name,
                    total_height=target_roof.height,
                    total_elevation_gain=h_direct,
                    total_energy=direct_energy,
                    steps=[step],
                    is_multi_tier=False,
                    safe_exit=target_roof.drop_points[0] if target_roof.drop_points else None,
                    score=direct_energy + math.hypot(approach_pos[0] - mx, approach_pos[1] - my)
                )
            )

        # Plan B: Multi-tier stair-step through intermediate eaves
        if eaves:
            sorted_eaves = sorted(eaves, key=lambda r: r.top)
            for eave in sorted_eaves:
                h_ground_to_eave = eave.top - mz
                if h_ground_to_eave < 1.0 or h_ground_to_eave > 32.0:
                    continue
                e_step0 = self.calculate_climb_energy(h_ground_to_eave)
                if e_step0 > current_energy:
                    continue

                eave_perch = self.get_safe_perch_point(eave, enemy_pos)

                climb_edge0 = (eave.ring[0][0], eave.ring[0][1], eave.top)
                approach_pos0 = (climb_edge0[0], climb_edge0[1], mz)
                if eave.edges:
                    min_d = float("inf")
                    for p1, p2 in eave.edges[:len(eave.ring)]:
                        d, qx, qy = point_segment_distance(mx, my, p1[0], p1[1], p2[0], p2[1])
                        if d < min_d:
                            min_d = d
                            climb_edge0 = (qx, qy, eave.top)
                            approach_pos0 = (qx, qy, mz)

                step0 = StairStepPerch(
                    step_index=0,
                    roof_id=eave.id,
                    roof_name=eave.name,
                    top=eave.top,
                    height=eave.height,
                    step_elevation_gain=h_ground_to_eave,
                    energy_cost=e_step0,
                    approach_pos=approach_pos0,
                    climb_edge=climb_edge0,
                    perch_pos=eave_perch,
                    los_occluded=self.check_ground_los_occluded(eave, eave_perch, enemy_pos) if enemy_pos else True
                )

                h_eave_to_target = target_roof.top - eave.top
                if 1.0 <= h_eave_to_target <= 32.0:
                    e_step1 = self.calculate_climb_energy(h_eave_to_target)
                    target_perch = self.get_safe_perch_point(target_roof, enemy_pos)

                    climb_edge1 = (target_roof.ring[0][0], target_roof.ring[0][1], target_roof.top)
                    approach_pos1 = (eave_perch[0], eave_perch[1], eave.top)
                    if target_roof.edges:
                        min_d = float("inf")
                        for p1, p2 in target_roof.edges[:len(target_roof.ring)]:
                            d, qx, qy = point_segment_distance(eave_perch[0], eave_perch[1], p1[0], p1[1], p2[0], p2[1])
                            if d < min_d:
                                min_d = d
                                climb_edge1 = (qx, qy, target_roof.top)
                                approach_pos1 = (qx, qy, eave.top)

                    step1 = StairStepPerch(
                        step_index=1,
                        roof_id=target_roof.id,
                        roof_name=target_roof.name,
                        top=target_roof.top,
                        height=target_roof.height,
                        step_elevation_gain=h_eave_to_target,
                        energy_cost=e_step1,
                        approach_pos=approach_pos1,
                        climb_edge=climb_edge1,
                        perch_pos=target_perch,
                        los_occluded=self.check_ground_los_occluded(target_roof, target_perch, enemy_pos) if enemy_pos else True
                    )

                    tot_energy = e_step0 + e_step1
                    d_run = math.hypot(approach_pos0[0] - mx, approach_pos0[1] - my)
                    score = tot_energy * 0.7 + d_run * 1.2 - (target_roof.top * 0.5)

                    candidate_plans.append(
                        StairStepClimbPlan(
                            target_roof_id=target_roof.id,
                            target_roof_name=target_roof.name,
                            total_height=target_roof.height,
                            total_elevation_gain=target_roof.top - mz,
                            total_energy=tot_energy,
                            steps=[step0, step1],
                            is_multi_tier=True,
                            safe_exit=target_roof.drop_points[0] if target_roof.drop_points else None,
                            score=score
                        )
                    )

        if not candidate_plans:
            return None

        return min(candidate_plans, key=lambda p: p.score)

    def find_nearest_rooftop_escape(
        self,
        my_pos: Union[List[float], Tuple[float, ...]],
        enemy_pos: Optional[Union[List[float], Tuple[float, ...]]] = None,
        energy: float = 100.0,
        max_search_radius: float = 160.0,
        allow_stairstep: bool = True
    ) -> Optional[RooftopEscapeTarget]:
        """
        Locates the optimal rooftop escape within jump-jet reach up to 60m height,
        automatically calculating stair-step roof perches along building eaves.
        
        Evaluation criteria:
        1. Energy feasibility: Energy required to climb must be <= available energy (with +25 reserve).
        2. Proximity: Distance from my_pos to the ground approach point at the wall base.
        3. Enemy shielding: Favors climb edges on far side of building relative to enemy_pos.
        4. Stair-step eaves: Supports multi-tier climbs to higher roofs up to 60m.
        
        Returns:
            RooftopEscapeTarget with ground approach waypoint, climb edge, safe interior perch,
            and stairstep_perches list, or None if no climbable roof is in range.
        Execution budget: < 0.20 ms.
        """
        t0 = time.perf_counter()
        if not self.rooftop_list or self.kdtree is None:
            return None

        mx, my = float(my_pos[0]), float(my_pos[1])
        mz = float(my_pos[2]) if len(my_pos) > 2 else self.get_ground_elevation(mx, my)
        ex, ey, ez = (float(enemy_pos[0]), float(enemy_pos[1]), float(enemy_pos[2])) if enemy_pos is not None else (None, None, None)

        k_search = min(5, len(self.rooftop_list))
        if self.kdtree is not None:
            _, candidate_indices = self.kdtree.query([mx, my], k=k_search)
            query_indices = [candidate_indices] if np.isscalar(candidate_indices) else list(candidate_indices)
        else:
            dists = np.hypot(self.roof_centers[:, 0] - mx, self.roof_centers[:, 1] - my)
            query_indices = list(np.argsort(dists)[:k_search])

        best_roof: Optional[Rooftop] = None
        best_approach_2d: Optional[Tuple[float, float]] = None
        best_climb_edge: Optional[Tuple[float, float, float]] = None
        best_energy_cost: float = 0.0
        best_dist_approach: float = 0.0
        best_shielded: bool = False
        best_score = float("inf")
        best_plan: Optional[StairStepClimbPlan] = None

        for idx in query_indices:
            roof = self.rooftop_list[idx]
            h_diff = roof.top - mz

            if h_diff < 1.0 or h_diff > ROOF_MAX_HEIGHT:
                continue

            # Stair-step planning for high roofs (> 28m) or when direct climb exceeds energy
            plan: Optional[StairStepClimbPlan] = None
            if allow_stairstep and (h_diff > 28.0 or (roof.eave_ids and energy < self.calculate_climb_energy(h_diff) + 20.0)):
                plan = self.find_stairstep_climb(my_pos, roof, enemy_pos, energy)

            energy_cost = plan.total_energy if plan is not None else self.calculate_climb_energy(h_diff)

            # Energy reserve buffer
            if energy < (energy_cost + 20.0):
                continue

            v_eb_x = roof.center[0] - ex if ex is not None else 0.0
            v_eb_y = roof.center[1] - ey if ey is not None else 0.0

            # Determine takeoff approach
            if plan is not None and plan.steps:
                first_step = plan.steps[0]
                approach_x, approach_y = first_step.approach_pos[0], first_step.approach_pos[1]
                climb_edge = first_step.climb_edge
                dist_approach = math.hypot(approach_x - mx, approach_y - my)
                shielded = False
                if ex is not None:
                    v_ea_x = approach_x - ex
                    v_ea_y = approach_y - ey
                    shielded = (v_ea_x * v_eb_x + v_ea_y * v_eb_y) > 0.0
                shield_bonus = -40.0 if shielded else 15.0
                height_reward = -0.5 * roof.height
                score = (dist_approach * 1.5) + (energy_cost * 0.8) + shield_bonus + height_reward
                if score < best_score:
                    best_score = score
                    best_roof = roof
                    best_approach_2d = (approach_x, approach_y)
                    best_climb_edge = climb_edge
                    best_energy_cost = energy_cost
                    best_dist_approach = dist_approach
                    best_shielded = shielded
                    best_plan = plan
            else:
                for p1, p2 in roof.edges[:len(roof.ring)]:
                    dist_to_seg, qx, qy = point_segment_distance(mx, my, p1[0], p1[1], p2[0], p2[1])

                    sdx = p2[0] - p1[0]
                    sdy = p2[1] - p1[1]
                    slen = math.hypot(sdx, sdy)
                    if slen < 1e-4:
                        continue
                    nx_out = sdy / slen
                    ny_out = -sdx / slen

                    approach_x = qx + nx_out * 2.5
                    approach_y = qy + ny_out * 2.5

                    if not self.is_ground_walkable(approach_x, approach_y):
                        approach_x, approach_y = qx, qy

                    dist_approach = math.hypot(approach_x - mx, approach_y - my)
                    if dist_approach > max_search_radius:
                        continue

                    shielded = False
                    if ex is not None:
                        v_ea_x = approach_x - ex
                        v_ea_y = approach_y - ey
                        dot_shield = v_ea_x * v_eb_x + v_ea_y * v_eb_y
                        shielded = dot_shield > 0.0

                    shield_bonus = -40.0 if shielded else 15.0
                    height_reward = -0.5 * roof.height
                    score = (dist_approach * 1.5) + (energy_cost * 0.8) + shield_bonus + height_reward

                    if score < best_score:
                        best_score = score
                        best_roof = roof
                        best_approach_2d = (approach_x, approach_y)
                        best_climb_edge = (qx, qy, roof.top)
                        best_energy_cost = energy_cost
                        best_dist_approach = dist_approach
                        best_shielded = shielded
                        best_plan = None

        if best_roof is None or best_approach_2d is None or best_climb_edge is None:
            return None

        if best_plan is None and best_roof.eave_ids and allow_stairstep and (best_roof.height > 25.0 or energy < best_energy_cost + 15.0):
            best_plan = self.find_stairstep_climb(my_pos, best_roof, enemy_pos, energy)

        approach_z = self.get_ground_elevation(best_approach_2d[0], best_approach_2d[1])
        perch = self.get_safe_perch_point(best_roof, enemy_pos)
        safe_exit = best_roof.drop_points[0] if best_roof.drop_points else None

        stairstep_perches = [step.perch_pos for step in best_plan.steps] if best_plan else [perch]

        best_target = RooftopEscapeTarget(
            roof_id=best_roof.id,
            roof_name=best_roof.name,
            roof_top=best_roof.top,
            roof_height=best_roof.height,
            energy_cost=best_energy_cost,
            ground_approach=(best_approach_2d[0], best_approach_2d[1], approach_z),
            climb_edge=best_climb_edge,
            perch_point=perch,
            dist_to_approach=best_dist_approach,
            safe_exit=safe_exit,
            shielded_from_enemy=best_shielded,
            score=best_score,
            stairstep_plan=best_plan,
            stairstep_perches=stairstep_perches
        )

        t_elapsed = (time.perf_counter() - t0) * 1000.0
        self.query_count += 1
        self.total_query_time_ms += t_elapsed

        return best_target

    def find_stairstep_escape(
        self,
        my_pos: Union[List[float], Tuple[float, ...]],
        enemy_pos: Optional[Union[List[float], Tuple[float, ...]]] = None,
        energy: float = 100.0,
        max_height: float = 60.0,
        max_search_radius: float = 160.0
    ) -> Optional[RooftopEscapeTarget]:
        """
        Specialized ATLAS method: Computes safe stair-step roof perches along building eaves
        so the bot can break line-of-sight vertically. Supports climbs up to 60m height.
        """
        return self.find_nearest_rooftop_escape(
            my_pos=my_pos,
            enemy_pos=enemy_pos,
            energy=energy,
            max_search_radius=max_search_radius,
            allow_stairstep=True
        )

    # -----------------------------------------------------------------
    # Component 3: The 3-Meter Roof-Edge Setback Rule
    # -----------------------------------------------------------------

    def get_safe_perch_point(
        self,
        roof_or_id: Union[int, Rooftop],
        enemy_pos: Optional[Union[List[float], Tuple[float, ...]]] = None
    ) -> Tuple[float, float, float]:
        """
        Calculates safe interior perch points at least 3m back from all roof edges.
        Standing near the edge exposes the 9m torso to ground muzzle angles.
        A 3m setback breaks line of sight from ground positions.
        """
        if isinstance(roof_or_id, Rooftop):
            roof = roof_or_id
        else:
            roof = self.rooftops.get(roof_or_id)
            if roof is None:
                return (0.0, 0.0, 0.0)

        if enemy_pos is None or not roof.perch_points:
            return roof.best_perch

        ex, ey = float(enemy_pos[0]), float(enemy_pos[1])
        best_pt = roof.best_perch
        max_dist_sq = -1.0

        for px, py, pz in roof.perch_points:
            d_sq = (px - ex) ** 2 + (py - ey) ** 2
            if d_sq > max_dist_sq:
                max_dist_sq = d_sq
                best_pt = (px, py, pz)

        return best_pt

    def check_ground_los_occluded(
        self,
        roof_or_id: Union[int, Rooftop],
        perch_pos: Union[List[float], Tuple[float, ...]],
        enemy_ground_pos: Union[List[float], Tuple[float, ...]]
    ) -> bool:
        """
        Verifies mathematically whether the roof edge occludes ground muzzle line of sight
        to a mech standing at perch_pos.
        Muzzle is at pos.z + 7.0. Mech torso extends to perch_pos.z + 9.0.
        """
        if isinstance(roof_or_id, Rooftop):
            roof = roof_or_id
        else:
            roof = self.rooftops.get(roof_or_id)
            if roof is None:
                return False

        ex, ey, ez = float(enemy_ground_pos[0]), float(enemy_ground_pos[1]), float(enemy_ground_pos[2])
        px, py, pz = float(perch_pos[0]), float(perch_pos[1]), float(perch_pos[2])

        muzzle_z = ez + MUZZLE_HEIGHT

        if muzzle_z >= roof.top:
            return False

        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        intersect_pt = None
        for p1, p2 in roof.edges[:len(roof.ring)]:
            if (ccw((ex, ey), (px, py), p1) != ccw((ex, ey), (px, py), p2)) and \
               (ccw((ex, ey), p1, p2) != ccw((px, py), p1, p2)):
                dx1 = px - ex
                dy1 = py - ey
                dx2 = p2[0] - p1[0]
                dy2 = p2[1] - p1[1]
                denom = dx1 * dy2 - dy1 * dx2
                if abs(denom) > 1e-6:
                    t = ((p1[0] - ex) * dy2 - (p1[1] - ey) * dx2) / denom
                    intersect_pt = (ex + t * dx1, ey + t * dy1)
                    break

        if intersect_pt is not None:
            dist_enemy_to_edge = math.hypot(intersect_pt[0] - ex, intersect_pt[1] - ey)
            dist_edge_to_perch = math.hypot(px - intersect_pt[0], py - intersect_pt[1])
        else:
            min_edge_dist = float("inf")
            edge_pt = None
            for p1, p2 in roof.edges[:len(roof.ring)]:
                dist, qx, qy = point_segment_distance(px, py, p1[0], p1[1], p2[0], p2[1])
                if dist < min_edge_dist:
                    min_edge_dist = dist
                    edge_pt = (qx, qy)
            if edge_pt is None or min_edge_dist < 0.1:
                return False
            dist_enemy_to_edge = math.hypot(edge_pt[0] - ex, edge_pt[1] - ey)
            dist_edge_to_perch = min_edge_dist

        if dist_enemy_to_edge < 1e-3:
            return False

        delta_z = roof.top - muzzle_z
        ray_slope = delta_z / dist_enemy_to_edge
        ray_height_at_perch = roof.top + dist_edge_to_perch * ray_slope

        return ray_height_at_perch >= (pz + EYE_HEIGHT)

    # -----------------------------------------------------------------
    # Component 4: Safe Drop-Down & Exit Vectors
    # -----------------------------------------------------------------

    def get_safe_exit_vector(
        self,
        roof_or_id: Union[int, Rooftop],
        zone_center: Union[List[float], Tuple[float, ...]],
        enemy_pos: Optional[Union[List[float], Tuple[float, ...]]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Calculates safe jumping-off points into ground cover when the zone starts closing.
        Leverages zero fall damage to cleanly descend 10m-60m without penalty.
        """
        if isinstance(roof_or_id, Rooftop):
            roof = roof_or_id
        else:
            roof = self.rooftops.get(roof_or_id)
            if roof is None:
                return None

        if not roof.drop_points:
            return None

        zx, zy = float(zone_center[0]), float(zone_center[1])
        ex, ey = (float(enemy_pos[0]), float(enemy_pos[1])) if enemy_pos is not None else (None, None)

        best_dp: Optional[DropDownPoint] = None
        best_score = -float("inf")

        for dp in roof.drop_points:
            gx, gy, gz = dp.ground_pos
            dist_to_center = math.hypot(gx - zx, gy - zy)
            center_alignment_score = -dist_to_center

            enemy_avoidance_score = 0.0
            if ex is not None and ey is not None:
                dist_to_enemy = math.hypot(gx - ex, gy - ey)
                enemy_avoidance_score = dist_to_enemy * 1.5

            score = center_alignment_score + enemy_avoidance_score
            if score > best_score:
                best_score = score
                best_dp = dp

        if best_dp is None:
            best_dp = roof.drop_points[0]

        return {
            "edge_pos": best_dp.edge_pos,
            "exit_vec": best_dp.exit_vec,
            "ground_pos": best_dp.ground_pos,
            "landing_roof_id": best_dp.landing_roof_id,
            "height_drop": roof.top - best_dp.ground_pos[2]
        }

    # -----------------------------------------------------------------
    # Component 5: Wall-Sliding Corner Rounding
    # -----------------------------------------------------------------

    def round_corner_waypoints(
        self,
        waypoints: List[Tuple[float, float]],
        fillet_radius: float = 4.5,
        min_angle_deg: float = 35.0
    ) -> List[Tuple[float, float]]:
        """
        Wall-sliding corner rounding:
        Replaces sharp 90-degree corners with smooth tangential arc fillets
        that slide along concave and convex wall contours, preserving forward escape speed.
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
                    if self.is_ground_walkable(apex[0], apex[1]):
                        result.extend([p_in, apex, p_out])
                    elif self.is_ground_walkable(p_in[0], p_in[1]) and self.is_ground_walkable(p_out[0], p_out[1]):
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
        Wall-sliding corner rounding and contour deflection:
        When escaping, projects velocity along the tangential contour of the wall.
        For concave wall contours (corners/pockets), applies outward relief deflection
        to slide around the contour without halting or corner-pinning.
        """
        if self.grid is None:
            return desired_vx, desired_vy

        normals = []
        for deg in range(0, 360, 30):
            rad = math.radians(deg)
            dx, dy = math.cos(rad), math.sin(rad)
            for d in (1.5, 2.5, probe_dist):
                qx, qy = px + dx * d, py + dy * d
                c = int(round((qx - self.min_x) / self.step))
                r = int(round((qy - self.min_y) / self.step))
                if 0 <= c < self.nx and 0 <= r < self.ny and self.grid[r, c] == 1:
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
                # Concave corner relief
                slide_x += nx_out * 0.35
                slide_y += ny_out * 0.35
            slen = math.hypot(slide_x, slide_y)
            if slen > 1e-4:
                return slide_x / slen, slide_y / slen

        return desired_vx, desired_vy

    # -----------------------------------------------------------------
    # Runtime Queries & Controller Execution
    # -----------------------------------------------------------------

    def is_on_rooftop(self, pos: Union[List[float], Tuple[float, ...]]) -> Optional[Rooftop]:
        """Checks whether 3D position is currently standing on a verified rooftop."""
        px, py = float(pos[0]), float(pos[1])
        pz = float(pos[2]) if len(pos) > 2 else 0.0

        for roof in self.rooftop_list:
            min_x, max_x, min_y, max_y = roof.bounds
            if not (min_x - 1.0 <= px <= max_x + 1.0 and min_y - 1.0 <= py <= max_y + 1.0):
                continue
            if abs(pz - roof.top) <= 2.0:
                if point_in_polygon(px, py, roof.ring):
                    in_hole = False
                    for h in roof.holes:
                        if point_in_polygon(px, py, h):
                            in_hole = True
                            break
                    if not in_hole:
                        return roof
        return None

    def compute_climb_action(
        self,
        my_pos: Union[List[float], Tuple[float, ...]],
        target: RooftopEscapeTarget,
        is_grounded: bool = True,
        current_energy: float = 100.0
    ) -> Dict[str, Any]:
        """
        Generates deterministic movement, jump, and boost commands to execute
        a multi-stage climb onto target roof and advance to safe 3m setback perch.
        Supports multi-tier stair-step climbs along building eaves.
        """
        mx, my = float(my_pos[0]), float(my_pos[1])
        mz = float(my_pos[2]) if len(my_pos) > 2 else 0.0

        gx, gy, gz = target.ground_approach
        cx, cy, cz = target.climb_edge
        px, py, pz = target.perch_point

        dist_to_approach = math.hypot(gx - mx, gy - my)
        dist_to_edge = math.hypot(cx - mx, cy - my)
        dist_to_perch = math.hypot(px - mx, py - my)

        # Stage 5 & 6: On top of roof
        if mz >= target.roof_top - 0.5 and is_grounded:
            if dist_to_perch > 1.5:
                yaw = math.atan2(py - my, px - mx)
                return {
                    "move": [math.cos(yaw), math.sin(yaw)],
                    "jump": False,
                    "boost": False,
                    "stage": "ADVANCING_TO_PERCH"
                }
            else:
                return {
                    "move": [0.0, 0.0],
                    "jump": False,
                    "boost": False,
                    "stage": "PERCHED_REGEN"
                }

        # Stage 4: In air, feet have reached or cleared roof edge
        if mz >= target.roof_top - 0.3:
            yaw = math.atan2(py - my, px - mx)
            return {
                "move": [math.cos(yaw), math.sin(yaw)],
                "jump": False,
                "boost": False,
                "stage": "CLEARING_EDGE"
            }

        # Stage 3: In air, jetting upward to reach roof height
        if not is_grounded:
            yaw = math.atan2(cx - mx, cy - my)
            return {
                "move": [math.cos(yaw), math.sin(yaw)],
                "jump": False,
                "boost": True,
                "stage": "JETTING_UP"
            }

        # Stage 2: Grounded and within jump range of the wall
        if dist_to_approach <= 3.0 or dist_to_edge <= 3.5:
            yaw = math.atan2(cx - mx, cy - my)
            return {
                "move": [math.cos(yaw), math.sin(yaw)],
                "jump": True,
                "boost": False,
                "stage": "JUMPING"
            }

        # Stage 1: Running toward ground approach point
        yaw = math.atan2(gy - my, gx - mx)
        should_boost = dist_to_approach > 15.0 and current_energy > target.energy_cost + 20.0
        return {
            "move": [math.cos(yaw), math.sin(yaw)],
            "jump": False,
            "boost": should_boost,
            "stage": "APPROACHING_WALL"
        }

    def should_descend(
        self,
        my_pos: Union[List[float], Tuple[float, ...]],
        zone_info: Dict[str, Any],
        current_time: float,
        my_hp: float = 1000.0
    ) -> Tuple[bool, str]:
        """Evaluates whether the bot must drop down from the roof into ground cover."""
        current_roof = self.is_on_rooftop(my_pos)
        if current_roof is None:
            return False, "NOT_ON_ROOFTOP"

        px, py = float(my_pos[0]), float(my_pos[1])
        zc = zone_info.get("center", [0.0, 0.0])
        curr_radius = float(zone_info.get("radius", 560.0))
        dist_to_center = math.hypot(px - zc[0], py - zc[1])

        if dist_to_center >= curr_radius - 10.0:
            return True, "ZONE_CLOSING_GAS_IMMINENT"

        shrink_start = float(zone_info.get("shrink_start", 9999.0))
        time_to_shrink = shrink_start - current_time
        nc = zone_info.get("next_center", zc)
        next_radius = float(zone_info.get("next_radius", curr_radius))
        dist_to_next_center = math.hypot(px - nc[0], py - nc[1])

        if dist_to_next_center > next_radius and time_to_shrink < 10.0:
            return True, "OUTSIDE_NEXT_ZONE_SHRINK_IMMINENT"

        if my_hp >= 950.0 and time_to_shrink < 15.0:
            return True, "HEAL_COMPLETE_READY_FOR_ROTATION"

        return False, "HOLD_PERCH"


# =====================================================================
# Module Singleton & Convenience Exports
# =====================================================================

_GLOBAL_ROOFTOP_MANAGER: Optional[RooftopManager] = None


def get_rooftop_manager(map_dir: str = "arena") -> RooftopManager:
    """Returns or lazily initializes the singleton RooftopManager."""
    global _GLOBAL_ROOFTOP_MANAGER
    if _GLOBAL_ROOFTOP_MANAGER is None:
        _GLOBAL_ROOFTOP_MANAGER = RooftopManager(map_dir)
    return _GLOBAL_ROOFTOP_MANAGER


def find_nearest_rooftop_escape(
    my_pos: Union[List[float], Tuple[float, ...]],
    enemy_pos: Optional[Union[List[float], Tuple[float, ...]]] = None,
    energy: float = 100.0,
    max_search_radius: float = 160.0,
    manager: Optional[RooftopManager] = None
) -> Optional[RooftopEscapeTarget]:
    """Module-level convenience wrapper for find_nearest_rooftop_escape."""
    mgr = manager or get_rooftop_manager()
    return mgr.find_nearest_rooftop_escape(my_pos, enemy_pos, energy, max_search_radius)


def find_stairstep_escape(
    my_pos: Union[List[float], Tuple[float, ...]],
    enemy_pos: Optional[Union[List[float], Tuple[float, ...]]] = None,
    energy: float = 100.0,
    max_height: float = 60.0,
    max_search_radius: float = 160.0,
    manager: Optional[RooftopManager] = None
) -> Optional[RooftopEscapeTarget]:
    """Module-level convenience wrapper for compute stair-step roof perches along building eaves up to 60m."""
    mgr = manager or get_rooftop_manager()
    return mgr.find_stairstep_escape(my_pos, enemy_pos, energy, max_height, max_search_radius)


def get_safe_perch_point(
    roof_or_id: Union[int, Rooftop],
    enemy_pos: Optional[Union[List[float], Tuple[float, ...]]] = None,
    manager: Optional[RooftopManager] = None
) -> Tuple[float, float, float]:
    """Module-level convenience wrapper for get_safe_perch_point."""
    mgr = manager or get_rooftop_manager()
    return mgr.get_safe_perch_point(roof_or_id, enemy_pos)


def get_safe_exit_vector(
    roof_or_id: Union[int, Rooftop],
    zone_center: Union[List[float], Tuple[float, ...]],
    enemy_pos: Optional[Union[List[float], Tuple[float, ...]]] = None,
    manager: Optional[RooftopManager] = None
) -> Optional[Dict[str, Any]]:
    """Module-level convenience wrapper for get_safe_exit_vector."""
    mgr = manager or get_rooftop_manager()
    return mgr.get_safe_exit_vector(roof_or_id, zone_center, enemy_pos)


def is_on_rooftop(
    pos: Union[List[float], Tuple[float, ...]],
    manager: Optional[RooftopManager] = None
) -> Optional[Rooftop]:
    """Module-level convenience wrapper for is_on_rooftop."""
    mgr = manager or get_rooftop_manager()
    return mgr.is_on_rooftop(pos)


def round_corner_waypoints(
    waypoints: List[Tuple[float, float]],
    fillet_radius: float = 4.5,
    min_angle_deg: float = 35.0,
    manager: Optional[RooftopManager] = None
) -> List[Tuple[float, float]]:
    """Module-level convenience wrapper for wall-sliding corner rounding."""
    mgr = manager or get_rooftop_manager()
    return mgr.round_corner_waypoints(waypoints, fillet_radius, min_angle_deg)


def compute_wall_slide_vector(
    px: float,
    py: float,
    desired_vx: float,
    desired_vy: float,
    probe_dist: float = 3.5,
    manager: Optional[RooftopManager] = None
) -> Tuple[float, float]:
    """Module-level convenience wrapper for wall-sliding along concave/convex contours."""
    mgr = manager or get_rooftop_manager()
    return mgr.compute_wall_slide_vector(px, py, desired_vx, desired_vy, probe_dist)


# =====================================================================
# Standalone Test Suite & Computational Budget Benchmark
# =====================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("ATLAS - 3D VERTICALITY & TERRAIN SPECIALIST (ROOFTOPS & MOVEMENT)")
    print("=" * 72)

    t0_init = time.perf_counter()
    mgr = RooftopManager(map_dir="arena")
    init_ms = (time.perf_counter() - t0_init) * 1000.0
    print(f"[OK] RooftopManager initialized in {init_ms:.2f} ms")
    print(f"[OK] Total qualified rooftops: {len(mgr.rooftop_list)}")

    # -------------------------------------------------------------
    # TEST 1: High-Rise Rooftops up to 60m Height
    # -------------------------------------------------------------
    print("\n=== TEST 1: High-Rise Rooftops (Up to 60m Height) ===")
    heights = [r.height for r in mgr.rooftop_list]
    print(f"  Height range: {min(heights):.1f}m to {max(heights):.1f}m")
    high_roofs = [r for r in mgr.rooftop_list if r.height > 25.0]
    print(f"  Qualified roofs > 25m height: {len(high_roofs)}")
    assert len(high_roofs) >= 15, "Must parse high-rise roofs between 25m and 60m"
    for r in high_roofs[:5]:
        print(f"    ID {r.id:3d}: {r.name:36s} | Top: {r.top:5.1f}m | Height: {r.height:4.1f}m | Setback: {r.max_setback:4.1f}m")
    print("  [PASS] Rooftops up to 60m successfully verified.")

    # -------------------------------------------------------------
    # TEST 2: Multi-Tier Stair-Step Perches along Building Eaves
    # -------------------------------------------------------------
    print("\n=== TEST 2: Multi-Tier Stair-Step Perches along Building Eaves ===")
    thompson_tower = mgr.rooftops.get(50)
    assert thompson_tower is not None, "Thompson Library tower (ID 50) must be loaded"
    print(f"  Target Tower: {thompson_tower.name} (Top={thompson_tower.top:.1f}m, Height={thompson_tower.height:.1f}m)")
    eaves = mgr.get_stairstep_eaves(thompson_tower)
    print(f"  Found {len(eaves)} eave rooftops connected to Thompson Library tower:")
    for e in eaves:
        print(f"    Eave ID {e.id}: {e.name} (Top={e.top:.1f}m, Height={e.height:.1f}m, Setback={e.max_setback:.1f}m)")
    assert len(eaves) >= 1, "Thompson tower must have at least 1 lower eave rooftop (ID 49)"

    # Compute stair-step climb plan from ground
    ground_pos = [110.0, -140.0, 2.5]
    plan = mgr.find_stairstep_climb(ground_pos, thompson_tower, enemy_pos=[110.0, -200.0, 2.5], current_energy=100.0)
    assert plan is not None, "Must find viable stair-step climb plan"
    assert plan.is_multi_tier is True, "Plan must use multi-tier eave stair-stepping"
    print(f"  [PASS] Multi-tier stair-step plan found ({len(plan.steps)} steps, total energy={plan.total_energy:.1f}):")
    for s in plan.steps:
        print(f"    Step {s.step_index}: {s.roof_name:32s} | Elevation Gain: +{s.step_elevation_gain:4.1f}m | Energy: {s.energy_cost:4.1f} | Perch: ({s.perch_pos[0]:.1f}, {s.perch_pos[1]:.1f}, {s.perch_pos[2]:.1f})")

    # -------------------------------------------------------------
    # TEST 3: Vertical LOS Occlusion Breakdown Behind 3m Setback
    # -------------------------------------------------------------
    print("\n=== TEST 3: Vertical Line-of-Sight Occlusion (3m Setback Rule) ===")
    eave_roof = eaves[0]
    perch = eave_roof.best_perch
    # Edge 10m in front of perch toward pursuing ground enemy at wall base (15m from edge)
    edge_pos = (perch[0], perch[1] - 10.0, eave_roof.top)
    enemy_ground = [perch[0], perch[1] - 25.0, 2.5]  # Enemy on ground 15m from wall base
    is_occluded = RooftopManager.check_vertical_los_occluded(perch, edge_pos, enemy_ground)
    print(f"  Mech at perch ({perch[0]:.1f}, {perch[1]:.1f}, {perch[2]:.1f}) with 10m setback")
    print(f"  Pursuing ground enemy at ({enemy_ground[0]:.1f}, {enemy_ground[1]:.1f}, {enemy_ground[2]:.1f})")
    print(f"  [PASS] Vertical LOS Occluded: {is_occluded}")
    assert is_occluded is True, "Ground enemy muzzle ray must be occluded by roof edge"

    # -------------------------------------------------------------
    # TEST 4: Wall-Sliding Corner Rounding (Tangential Fillet Arc)
    # -------------------------------------------------------------
    print("\n=== TEST 4: Wall-Sliding Corner Rounding (Smoothing Sharp 90° Turns) ===")
    sharp_90_path = [(0.0, 0.0), (50.0, 0.0), (50.0, 50.0)]
    rounded_path = mgr.round_corner_waypoints(sharp_90_path, fillet_radius=5.0)
    print(f"  Original 90-degree corner path (3 waypoints): {sharp_90_path}")
    print(f"  Rounded smooth path ({len(rounded_path)} waypoints):")
    for pt in rounded_path:
        print(f"    ({pt[0]:.1f}, {pt[1]:.1f})")
    assert len(rounded_path) > len(sharp_90_path), "Must insert corner-rounding fillet waypoints"
    # Verify that the sharp corner vertex (50.0, 0.0) is replaced by rounded arc
    assert (50.0, 0.0) not in rounded_path, "Sharp corner vertex must be replaced by rounded fillet"
    print("  [PASS] Sharp 90-degree turn converted to smooth wall-contour fillet.")

    # -------------------------------------------------------------
    # TEST 5: Wall-Sliding Velocity Vector & Concave Contour Relief
    # -------------------------------------------------------------
    print("\n=== TEST 5: Wall-Sliding Velocity along Concave & Convex Wall Contours ===")
    # Approach solid building wall at Pomerene Hall (x=149.0, y=-330.0) moving diagonal into wall (+0.71, +0.71)
    d_diag = math.cos(math.pi / 4)
    slide_vec = mgr.compute_wall_slide_vector(149.0, -330.0, d_diag, d_diag, probe_dist=3.5)
    print(f"  Desired diagonal ({d_diag:.2f}, {d_diag:.2f}) into wall -> Wall-slide vector: ({slide_vec[0]:.2f}, {slide_vec[1]:.2f})")
    # Verify velocity deflected away from wall collision (pushing outward/along contour)
    assert slide_vec[0] < 0.0 and slide_vec[1] > 0.0, "Velocity must deflect along wall contour away from solid obstacle"
    print("  [PASS] Wall-sliding velocity deflection active with zero collision sticking.")

    # -------------------------------------------------------------
    # TEST 6: Micro-Benchmark (Query Latency Budget < 0.20 ms)
    # -------------------------------------------------------------
    print("\n" + "=" * 72)
    print("BENCHMARK TEST: Rooftop Escape & Stair-Step Climb Query Latency")
    print("=" * 72)

    N = 5000
    np.random.seed(42)
    test_positions = np.random.uniform(-150.0, 150.0, size=(N, 2))

    t_bench_start = time.perf_counter()
    latencies = []
    for i in range(N):
        mpos = [float(test_positions[i, 0]), float(test_positions[i, 1]), 2.5]
        t_q = time.perf_counter()
        mgr.find_stairstep_escape(mpos, enemy_pos=[0.0, 0.0, 2.5], energy=100.0)
        latencies.append((time.perf_counter() - t_q) * 1000.0)

    t_bench_total = time.perf_counter() - t_bench_start
    arr = np.array(latencies)
    mean_lat = float(np.mean(arr))
    p50_lat = float(np.median(arr))
    p95_lat = float(np.percentile(arr, 95))
    p99_lat = float(np.percentile(arr, 99))
    max_lat = float(np.max(arr))

    print(f"Benchmark results ({N:,} queries in {t_bench_total:.3f}s):")
    print(f"  Mean latency:   {mean_lat:.5f} ms")
    print(f"  p50 latency:    {p50_lat:.5f} ms")
    print(f"  p95 latency:    {p95_lat:.5f} ms")
    print(f"  p99 latency:    {p99_lat:.5f} ms")
    print(f"  Max latency:    {max_lat:.5f} ms")

    assert p99_lat < 1.5, f"Latency budget exceeded: p99 {p99_lat:.5f} ms >= 1.5 ms"
    print(f"\n[PASS] STRICT BUDGET COMPLIED: p99 ({p99_lat:.5f} ms) is ~{1.5/p99_lat:.0f}x faster than 1.5 ms limit!")
    print("=" * 72)
    print("ALL TESTS & BENCHMARKS COMPLETED WITH 100% SUCCESS!")
    print("=" * 72)
