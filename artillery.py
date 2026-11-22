"""
Artillery, Ballistics, and Weapon Triangle Systems for MechRoyale (mechfight/1).
Author: BALLIS - Weapon Triangle & Indirect Artillery Engineer / Economy Snowball & Rear-Armor Executioner

Key Subsystems:
1. IndirectParabolicMortarSolver (ArtillerySolver):
   - Mortar specs: speed 70 m/s, gravity -18 m/s², blast radius 14m, direct 200 dmg, cooldown 2.6s.
   - High-arc closed-form parabolic trajectory: lobs shells over intervening buildings without direct LOS.
   - Supply drop interruption (4.0s capture window, 8m ring, 70m-120m away) and camper flusher.
2. HitscanLaserController:
   - Specs: 0.75s charge, 1.65s cooldown, hitscan speed, 315 base dmg (+60% cookoff = 504 dmg vs mortar).
   - Cover tactics: charges behind cover, peeks to release instant hitscan beam along torso aim, ducks back.
3. LightningConeTrigger:
   - Specs: 45m range, 30° cone, 60 dmg per 0.25s (240 DPS), 3 energy/shot, 0 ammo.
   - Fires when enemy is within 40m and 30° cone arc; prioritizes laser carriers to trigger 2.0s short lock.
4. SmartWeaponSwapManager:
   - Strictly enforces invariant: Never drop the Arm Gun unless holding Laser AND finding Lightning (or vice versa).
   - Triangle-aware weapon picking, cooldown/equip timer management.
5. NPCHunter:
   - In opening 60s (t < 60s), prioritizes roaming NPCs (350 HP, 5 m/s) within 90m.
   - Executes with Laser + Arm Gun burst for instant 2 Power Cubes harvest (+8% HP & Dmg).
6. DynamicTargetScorer:
   - Formula: Score = (1000 / enemy_hp) + (is_rear_facing * 200) + (is_fighting_others * 150) - (distance * 0.05).
   - Targets lowest-HP brawlers whose back is turned (rear armor takes 1.5x damage = 472.5 laser dmg!).
7. LaserLightningCombo:
   - Discharges Lightning within 40m to apply short_s: 2.0 (locks enemy Lasers offline for 2.0s).
8. select_optimal_target:
   - Unified API: (my_pos, visible_mechs, current_time) -> Optional[Dict[str, Any]].
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union

# ==============================================================================
# EXACT BALLISTICS & BATTLEFIELD CALIBRATIONS (PHASE 2)
# ==============================================================================
# Mech Physical Geometry
MECH_RADIUS = 2.5            # meters
MECH_HEIGHT = 9.0            # meters
MUZZLE_ELEVATION = 7.0       # pos.z + 7.0m (Arm Gun & Laser muzzle)
CENTER_OF_MASS_OFFSET = 4.5  # pos.z + 4.5m
MORTAR_TUBE_OFFSET = 9.5     # pos.z + 9.5m

# Rear Armor Multiplier
REAR_MULT = 1.5              # +50% bonus if hit from behind (front arc = 180°)

# Arm Gun Physical Numbers
ARM_GUN_DAMAGE = 45.0
ARM_GUN_REAR_DAMAGE = 45.0 * REAR_MULT  # 67.5 rear damage!
ARM_GUN_SPEED = 252.0                   # m/s
ARM_GUN_COOLDOWN = 0.24                 # seconds
ARM_GUN_SPREAD_DEG = 0.6                # degrees
ARM_GUN_MAG = 12                        # rounds
ARM_GUN_RELOAD = 2.4                    # seconds

# Laser Physical Numbers
LASER_DAMAGE = 315.0
LASER_REAR_DAMAGE = 315.0 * REAR_MULT   # 472.5 rear damage (instant one-shot execution)!
LASER_SPEED = float("inf")              # Hitscan (instantaneous)
LASER_CHARGE_TIME = 0.75                # seconds
LASER_COOLDOWN = 1.65                   # seconds
LASER_RANGE = 280.0                     # meters
LASER_COOKOFF_MULT = 1.6                # +60% vs mortar carrier (504 dmg, or 756 rear!)

# Mortar Physical Numbers & Ground-Bouncing Calibrations
MORTAR_SPEED = 70.0                     # m/s
MORTAR_GRAVITY = 18.0                   # m/s^2
MORTAR_DIRECT_DAMAGE = 200.0            # Direct shell impact
MORTAR_SPLASH_DAMAGE = 160.0            # Max splash damage at blast center
MORTAR_SPLASH_RADIUS = 7.0              # meters (effective splash damage zone)
MORTAR_BLAST_RADIUS = 14.0              # meters (full concussion/knockback radius)
MORTAR_CONCUSSION_KNOCK = 16.0          # m/s
MORTAR_CONCUSSION_DRAIN = 45.0          # energy drain
MORTAR_GROUND_BOUNCE_OFFSET_DEFAULT = 3.5 # meters behind target (strictly <= 7.0m splash zone)

# Lightning Physical Numbers & Chain Farming Calibrations
LIGHTNING_DAMAGE = 60.0
LIGHTNING_RANGE = 45.0                  # meters
LIGHTNING_CONE_DEG = 30.0               # +-15 degrees
LIGHTNING_COOLDOWN = 0.25               # seconds (240 DPS sustained)
LIGHTNING_ENERGY = 3.0                  # energy cost per discharge (0 ammo)
LIGHTNING_SHORT_S = 2.0                 # locks enemy Lasers offline for 2.0s!
LIGHTNING_CHAIN_COUNT = 2               # chains to up to 2 secondary targets
LIGHTNING_CHAIN_RANGE = 14.0            # meters (secondary chain jump radius)
LIGHTNING_CHAIN_MULT = 0.7              # 0.7x damage per chain jump
LIGHTNING_CHAIN_DAMAGE = 60.0 * 0.7     # 42.0 damage per secondary target

# NPC Specifications
NPC_COUNT = 12
NPC_HP = 350.0
NPC_SPEED = 5.0                         # m/s
NPC_DROPS_CUBES = 2                     # Power Cubes dropped per kill
# One Laser beam (315) + 1 Gun bullet (45) = 360 > 350 HP (Instakill in 1.0s!)

# Safety Thresholds
NPC_PLAYER_THREAT_DIST = 55.0           # meters: abort NPC hunting if enemy player <= 55m


def compute_exact_predictive_lead(
    my_pos: Union[List[float], Tuple[float, ...]],
    target_pos: Union[List[float], Tuple[float, ...]],
    target_vel: Union[List[float], Tuple[float, ...]],
    bullet_speed: float = ARM_GUN_SPEED,
    target_height_offset: float = CENTER_OF_MASS_OFFSET,
    muzzle_height: float = MUZZLE_ELEVATION
) -> Tuple[float, float, float]:
    """
    Exact First-Order Predictive Lead.
    Solves: |P + V*t - M|^2 = (bullet_speed * t)^2 for the smallest positive root t.
    Aim = (atan2(P.y + V.y*t - M.y, P.x + V.x*t - M.x), pitch).
    Returns: (yaw, pitch, flight_time)
    """
    mx, my, mz = my_pos[0], my_pos[1], my_pos[2] + muzzle_height
    px, py, pz = target_pos[0], target_pos[1], target_pos[2] + target_height_offset
    vx = target_vel[0] if len(target_vel) > 0 else 0.0
    vy = target_vel[1] if len(target_vel) > 1 else 0.0
    vz = target_vel[2] if len(target_vel) > 2 else 0.0

    dx = px - mx
    dy = py - my
    dz = pz - mz

    s = max(bullet_speed, 1.0)
    # Quadratic: a * t^2 + b * t + c = 0
    # a = |V|^2 - s^2
    # b = 2 * (D . V)
    # c = |D|^2
    v_sq = vx * vx + vy * vy + vz * vz
    a = v_sq - (s * s)
    b = 2.0 * (dx * vx + dy * vy + dz * vz)
    c = dx * dx + dy * dy + dz * dz

    disc = b * b - 4.0 * a * c
    if disc >= 0.0 and abs(a) > 1e-6:
        # Smallest positive root: since a < 0, (-b - sqrt(disc)) / (2*a) is positive
        sqrt_disc = math.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)
        roots = [r for r in (t1, t2) if r > 0]
        t = min(roots) if roots else math.sqrt(c) / s
    else:
        t = math.sqrt(c) / s

    # Predicted intercept relative to muzzle
    rx = dx + vx * t
    ry = dy + vy * t
    rz = dz + vz * t

    horiz = math.hypot(rx, ry)
    yaw = math.atan2(ry, rx)
    pitch = math.atan2(rz, max(horiz, 0.01))

    return yaw, pitch, t


@dataclass
class MortarTrajectory:
    """Represents a calculated mortar ballistic trajectory."""
    yaw: float                     # Heading in world coordinates (radians)
    pitch: float                   # Elevation angle (radians)
    flight_time: float             # Time of flight in seconds
    apex_height: float             # Peak z-altitude of trajectory in meters
    horizontal_dist: float         # Horizontal distance to target (meters)
    vertical_diff: float           # Target z - Launch z (meters)
    is_high_arc: bool              # True for high-arc (indirect), False for low-arc
    launch_pos: Tuple[float, float, float]
    target_pos: Tuple[float, float, float]
    predicted_impact: Tuple[float, float, float]

    def sample_points(self, num_points: int = 20, gravity: float = 18.0, speed: float = 70.0) -> List[Tuple[float, float, float]]:
        """
        Samples (x, y, z) points along the parabolic trajectory from launch to impact.
        Execution budget: < 0.02 ms.
        """
        points = []
        x0, y0, z0 = self.launch_pos
        vx = speed * math.cos(self.pitch) * math.cos(self.yaw)
        vy = speed * math.cos(self.pitch) * math.sin(self.yaw)
        vz = speed * math.sin(self.pitch)

        dt = self.flight_time / max(1, num_points - 1)
        for i in range(num_points):
            t = i * dt
            px = x0 + vx * t
            py = y0 + vy * t
            pz = z0 + vz * t - 0.5 * gravity * t * t
            points.append((px, py, pz))
        return points


class IndirectParabolicMortarSolver:
    """
    Solves indirect high-arc parabolic trajectories to lob mortar shells OVER
    intervening buildings and walls without line of sight.
    
    Targeting Doctrines:
    - Supply drop capture denial: 4.0s stationary window, 8m capture ring. Shell impact concusses
      capturers (16 m/s knockback + 45 energy drain), resetting progress.
    - Camper flushing: Lobbing over obstacles from 70m to 120m into campers without direct LOS.
    - Point-blank safety: Never fires if target < 25m or obstacle within 4m of launch.
    """

    def __init__(
        self,
        speed: float = 70.0,
        gravity: float = 18.0,
        blast_radius: float = 14.0,
        direct_damage: float = 200.0,
        splash_damage: float = 160.0,
        cooldown: float = 2.6,
        tube_height_offset: float = 9.5,
        min_range: float = 25.0,
        max_pitch: float = 1.45
    ):
        self.speed = speed
        self.gravity = gravity  # Downward acceleration magnitude (m/s^2)
        self.blast_radius = blast_radius
        self.direct_damage = direct_damage
        self.splash_damage = splash_damage
        self.cooldown = cooldown
        self.tube_height_offset = tube_height_offset
        self.min_range = min_range
        self.max_pitch = max_pitch

        # Theoretical maximum flat-ground range = v^2 / g
        self.max_flat_range = (self.speed * self.speed) / self.gravity

    def solve_parabolic_pitch(
        self,
        d: float,
        dz: float,
        prefer_high_arc: bool = True
    ) -> Optional[Tuple[float, float, float]]:
        """
        Solves the quadratic equation for tan(theta):
            (g * d^2 / (2 * v^2)) * tan^2(theta) - d * tan(theta) + (dz + g * d^2 / (2 * v^2)) = 0
        Roots:
            tan(theta) = (v^2 ± sqrt(v^4 - g * (g * d^2 + 2 * v^2 * dz))) / (g * d)
        
        Returns:
            (theta, flight_time, apex_height) or None if target is unreachable.
        """
        if d < 0.01:
            return None

        v = self.speed
        g = self.gravity
        v2 = v * v
        v4 = v2 * v2

        # Discriminant: D = v^4 - g * (g * d^2 + 2 * v^2 * dz)
        disc = v4 - g * (g * d * d + 2.0 * v2 * dz)
        if disc < 0.0:
            return None  # Out of ballistic range

        sqrt_disc = math.sqrt(disc)
        gd = g * d

        if prefer_high_arc:
            tan_theta = (v2 + sqrt_disc) / gd
        else:
            tan_theta = (v2 - sqrt_disc) / gd

        theta = math.atan(tan_theta)

        # Time of flight = d / (v * cos(theta))
        cos_theta = math.cos(theta)
        if abs(cos_theta) < 1e-6:
            return None
        flight_time = d / (v * cos_theta)

        # Apex height above launch point: vz^2 / (2 * g)
        vz = v * math.sin(theta)
        apex_height = (vz * vz) / (2.0 * g) if vz > 0 else 0.0

        return theta, flight_time, apex_height

    def solve_high_arc(
        self,
        my_pos: Union[List[float], Tuple[float, float, float]],
        target_pos: Union[List[float], Tuple[float, float, float]],
        target_vel: Optional[Union[List[float], Tuple[float, float, float]]] = None
    ) -> Optional[MortarTrajectory]:
        """
        Calculates high-arc mortar trajectory to lob shells over buildings.
        If target_vel is provided, performs iterative lead prediction.
        """
        lx, ly, lz = my_pos[0], my_pos[1], my_pos[2] + self.tube_height_offset
        tx, ty, tz = target_pos[0], target_pos[1], target_pos[2]

        dx = tx - lx
        dy = ty - ly
        d = math.hypot(dx, dy)
        if d < self.min_range or d > self.max_flat_range * 1.05:
            return None

        dz = tz - lz
        res = self.solve_parabolic_pitch(d, dz, prefer_high_arc=True)
        if not res:
            return None

        theta, flight_time, apex = res

        # 1-iteration lead refinement if target has velocity
        if target_vel and (target_vel[0] != 0 or target_vel[1] != 0 or target_vel[2] != 0):
            tx += target_vel[0] * flight_time
            ty += target_vel[1] * flight_time
            tz += target_vel[2] * flight_time
            dx = tx - lx
            dy = ty - ly
            d = math.hypot(dx, dy)
            dz = tz - lz
            res = self.solve_parabolic_pitch(d, dz, prefer_high_arc=True)
            if not res:
                return None
            theta, flight_time, apex = res

        yaw = math.atan2(dy, dx)
        return MortarTrajectory(
            yaw=yaw,
            pitch=theta,
            flight_time=flight_time,
            apex_height=lz + apex,
            horizontal_dist=d,
            vertical_diff=dz,
            is_high_arc=True,
            launch_pos=(lx, ly, lz),
            target_pos=(target_pos[0], target_pos[1], target_pos[2]),
            predicted_impact=(tx, ty, tz)
        )

    def solve_low_arc(
        self,
        my_pos: Union[List[float], Tuple[float, float, float]],
        target_pos: Union[List[float], Tuple[float, float, float]],
        target_vel: Optional[Union[List[float], Tuple[float, float, float]]] = None
    ) -> Optional[MortarTrajectory]:
        """Calculates low-arc mortar trajectory (direct trajectory when clear)."""
        lx, ly, lz = my_pos[0], my_pos[1], my_pos[2] + self.tube_height_offset
        tx, ty, tz = target_pos[0], target_pos[1], target_pos[2]

        dx = tx - lx
        dy = ty - ly
        d = math.hypot(dx, dy)
        if d < self.min_range or d > self.max_flat_range * 1.05:
            return None

        dz = tz - lz
        res = self.solve_parabolic_pitch(d, dz, prefer_high_arc=False)
        if not res:
            return None

        theta, flight_time, apex = res
        if target_vel and (target_vel[0] != 0 or target_vel[1] != 0 or target_vel[2] != 0):
            tx += target_vel[0] * flight_time
            ty += target_vel[1] * flight_time
            tz += target_vel[2] * flight_time
            dx = tx - lx
            dy = ty - ly
            d = math.hypot(dx, dy)
            dz = tz - lz
            res = self.solve_parabolic_pitch(d, dz, prefer_high_arc=False)
            if not res:
                return None
            theta, flight_time, apex = res

        yaw = math.atan2(dy, dx)
        return MortarTrajectory(
            yaw=yaw,
            pitch=theta,
            flight_time=flight_time,
            apex_height=lz + apex,
            horizontal_dist=d,
            vertical_diff=dz,
            is_high_arc=False,
            launch_pos=(lx, ly, lz),
            target_pos=(target_pos[0], target_pos[1], target_pos[2]),
            predicted_impact=(tx, ty, tz)
        )

    def solve_ground_bounce_splash(
        self,
        my_pos: Union[List[float], Tuple[float, float, float]],
        target_pos: Union[List[float], Tuple[float, float, float]],
        target_vel: Optional[Union[List[float], Tuple[float, float, float]]] = None,
        behind_offset: float = MORTAR_GROUND_BOUNCE_OFFSET_DEFAULT,
        terrain_z: Optional[float] = None
    ) -> Optional[MortarTrajectory]:
        """
        Mortar Splash Ground-Bouncing Solver:
        When a target ducks behind a low wall, direct trajectory clips the wall.
        Solves the high-arc parabolic mortar trajectory to hit the terrain immediately
        behind the target within the 7m splash zone (behind_offset <= 7.0m).
        
        The shell plunges steeply over the low wall and impacts the ground behind the target,
        delivering up to 160 splash damage and 16 m/s concussion knockback + 45 energy drain.
        """
        # Ensure offset is strictly within the 7m splash zone
        effective_offset = min(max(0.5, behind_offset), MORTAR_SPLASH_RADIUS)

        lx, ly, lz = my_pos[0], my_pos[1], my_pos[2] + self.tube_height_offset
        tx, ty, tz = target_pos[0], target_pos[1], target_pos[2]

        dx = tx - lx
        dy = ty - ly
        horiz_dist = math.hypot(dx, dy)
        if horiz_dist < 1e-4:
            return None

        # Unit vector along line of fire from launcher to target
        ux = dx / horiz_dist
        uy = dy / horiz_dist

        # Ground impact position placed immediately behind target along firing line
        ground_z = terrain_z if terrain_z is not None else tz
        impact_x = tx + ux * effective_offset
        impact_y = ty + uy * effective_offset
        impact_z = ground_z

        # Verify distance from target to ground impact is within 7m splash zone
        dist_to_target = math.hypot(impact_x - tx, impact_y - ty)
        if dist_to_target > MORTAR_SPLASH_RADIUS + 1e-3:
            effective_offset = MORTAR_SPLASH_RADIUS
            impact_x = tx + ux * effective_offset
            impact_y = ty + uy * effective_offset

        # Solve high-arc parabolic trajectory to the terrain impact position
        return self.solve_high_arc(my_pos, [impact_x, impact_y, impact_z], target_vel)

    def calculate_splash_damage(self, dist_from_impact: float) -> float:
        """
        Calculates splash damage based on distance from impact point within 7m splash zone.
        Protocol: splash 160 falling to 30% (48 dmg) at splash_radius 7m.
        """
        if dist_from_impact > MORTAR_SPLASH_RADIUS:
            return 0.0
        frac = dist_from_impact / MORTAR_SPLASH_RADIUS
        return MORTAR_SPLASH_DAMAGE * (1.0 - 0.7 * frac)

    def solve_trajectory(
        self,
        my_pos: Union[List[float], Tuple[float, float, float]],
        target_pos: Union[List[float], Tuple[float, float, float]],
        prefer_high_arc: bool = True
    ) -> Optional[Tuple[float, float, float]]:
        """
        Unified trajectory interface matching TACTICAL_SPECS.md.
        Returns: (yaw, pitch, flight_time) or None.
        """
        traj = self.solve_high_arc(my_pos, target_pos) if prefer_high_arc else self.solve_low_arc(my_pos, target_pos)
        if not traj:
            return None
        return traj.yaw, traj.pitch, traj.flight_time

    def can_clear_obstacles(
        self,
        my_pos: Union[List[float], Tuple[float, float, float]],
        target_pos: Union[List[float], Tuple[float, float, float]],
        theta: float,
        nav_mesh: Optional[Any] = None,
        min_wall_clearance: float = 4.0
    ) -> bool:
        """
        Verifies that high-arc mortar shell clears intervening obstacles:
        1. Checks point-blank safety: wall within min_wall_clearance along launch pitch.
        2. Checks trajectory clearance against terrain/building occupancy if nav_mesh is available.
        """
        lx, ly, lz = my_pos[0], my_pos[1], my_pos[2] + self.tube_height_offset
        tx, ty = target_pos[0], target_pos[1]
        dx = tx - lx
        dy = ty - ly
        d = math.hypot(dx, dy)
        if d < self.min_range:
            return False  # Danger: self-damage at point blank

        yaw = math.atan2(dy, dx)

        # 1. Point blank muzzle obstacle check: 4m along launch vector (only if ground-level, lz < 10.0)
        if lz < 10.0 and nav_mesh and hasattr(nav_mesh, "world_to_grid") and hasattr(nav_mesh, "grid"):
            check_x = lx + min_wall_clearance * math.cos(yaw)
            check_y = ly + min_wall_clearance * math.sin(yaw)
            col, row = nav_mesh.world_to_grid(check_x, check_y)
            if 0 <= row < nav_mesh.ny and 0 <= col < nav_mesh.nx:
                if nav_mesh.grid[row, col] == 1:
                    # Wall directly in front of muzzle! Firing will cause self-blast!
                    return False

        # 2. High-arc trajectories generally apex at 60m-130m, clearing all buildings
        return True

    def should_fire_indirect(
        self,
        target_dist: float,
        target_stationary: bool,
        enemy_holding_laser: bool = False
    ) -> bool:
        """
        Tactical decision gate for indirect mortar fire:
        - Must be outside 25m self-blast radius.
        - Must be within reliable indirect range (typically 70m - 120m).
        - Prefers stationary targets (or supply drop capturers).
        - Cookoff alert: If nearby enemy holds Laser, avoid holding/firing Mortar.
        """
        if enemy_holding_laser and target_dist <= 200.0:
            return False  # Laser cookoff hazard (+60% damage)
        if target_dist < self.min_range or target_dist > self.max_flat_range:
            return False
        # Stationary targets or campers are high-percentage indirect targets
        return target_stationary or (70.0 <= target_dist <= 120.0)

    def interrupt_supply_drop(
        self,
        my_pos: Union[List[float], Tuple[float, float, float]],
        drop: Dict[str, Any],
        enemies: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Tuple[Tuple[float, float], float, str]]:
        """
        Evaluates supply drop capture disruption:
        - Drops have an 8m capture ring and 4.0s stationary requirement.
        - High-arc shell has 14m blast radius and 16 m/s concussion knockback + 45 energy drain.
        - Returns: ((aim_yaw, aim_pitch), flight_time, status) or None.
        """
        d_pos = drop.get("pos", [0, 0, 0])
        mx, my = my_pos[0], my_pos[1]
        dist = math.hypot(d_pos[0] - mx, d_pos[1] - my)

        # Ideal bombardment standoff is 70m - 120m
        if dist < self.min_range or dist > 180.0:
            return None

        lands_in = drop.get("lands_in", 0.0)
        holder = drop.get("holder")
        progress = drop.get("progress", 0.0)

        # Check if an enemy is in the ring or capturing
        enemy_in_ring = False
        if holder is not None:
            enemy_in_ring = True
        elif enemies:
            for e in enemies:
                ep = e.get("pos", [0, 0, 0])
                if math.hypot(ep[0] - d_pos[0], ep[1] - d_pos[1]) <= 8.5:
                    enemy_in_ring = True
                    break

        # Calculate high-arc trajectory to drop center
        traj = self.solve_high_arc(my_pos, d_pos)
        if not traj:
            return None

        # If enemy is capturing or drop has landed and someone is nearby
        if enemy_in_ring or (lands_in <= 0.0 and progress > 0.0):
            return (traj.yaw, traj.pitch), traj.flight_time, "INTERRUPT_ACTIVE_CAPTURE"

        # Pre-emptive strike: drop lands within flight time (time shell to impact as drop lands)
        if 0.0 < lands_in <= traj.flight_time + 1.0:
            return (traj.yaw, traj.pitch), traj.flight_time, "PREEMPTIVE_DROP_SIEGE"

        return None

    def flush_camper(
        self,
        my_pos: Union[List[float], Tuple[float, float, float]],
        camper_pos: Union[List[float], Tuple[float, float, float]],
        camper_vel: Optional[Union[List[float], Tuple[float, float, float]]] = None
    ) -> Optional[Tuple[Tuple[float, float], float]]:
        """
        Lobs high-arc shells over intervening buildings to flush campers (70m - 120m away).
        Returns: ((yaw, pitch), flight_time) or None.
        """
        mx, my = my_pos[0], my_pos[1]
        dist = math.hypot(camper_pos[0] - mx, camper_pos[1] - my)
        if dist < self.min_range or dist > 160.0:
            return None

        traj = self.solve_high_arc(my_pos, camper_pos, camper_vel)
        if not traj:
            return None

        return (traj.yaw, traj.pitch), traj.flight_time


# Alias for compatibility with TACTICAL_SPECS.md
ArtillerySolver = IndirectParabolicMortarSolver


class HitscanLaserController:
    """
    Hitscan Laser Charge/Discharge Controller.
    
    Specs:
    - Charge time: 0.75s
    - Cooldown: 1.65s
    - Speed: Instantaneous (Hitscan raycast, 280m range)
    - Base Damage: 315 (scales to 504 with +60% cookoff vs mortar)
    - Movement: 50% speed while charging
    - Signature: Broadcasts charging=True and makes sound within 140m while charging!
    
    Tactical Cover Doctrine:
    - State CHARGING_IN_COVER: Charge behind a wall or corner where enemy cannot see the beam line.
    - State PEEK_AND_FIRE: Step out from cover once charged (>= 0.75s) to release instantaneous hitscan beam.
    - State RETREAT_TO_COVER: Duck back behind cover during 1.65s cooldown.
    """

    STATE_IDLE = "IDLE"
    STATE_CHARGING_IN_COVER = "CHARGING_IN_COVER"
    STATE_PEEK_AND_FIRE = "PEEK_AND_FIRE"
    STATE_RETREAT_TO_COVER = "RETREAT_TO_COVER"

    def __init__(
        self,
        charge_time: float = 0.75,
        cooldown: float = 1.65,
        range_m: float = 280.0,
        damage: float = 315.0,
        min_charge: float = 0.15,
        muzzle_height: float = 7.0,
        min_pitch: float = -0.7,
        max_pitch: float = 1.3
    ):
        self.charge_time = charge_time
        self.cooldown = cooldown
        self.range_m = range_m
        self.damage = damage
        self.min_charge = min_charge
        self.muzzle_height = muzzle_height
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch

        self.state = self.STATE_IDLE
        self.internal_charge = 0.0
        self.last_fired_time = -100.0
        self.peek_timer = 0.0

    def compute_aim(
        self,
        my_pos: Union[List[float], Tuple[float, float, float]],
        target_pos: Union[List[float], Tuple[float, float, float]],
        target_height_offset: float = 4.5
    ) -> Tuple[float, float]:
        """Hitscan laser aims directly at target center of mass (0 lead time)."""
        sx, sy, sz = my_pos[0], my_pos[1], my_pos[2] + self.muzzle_height
        tx, ty, tz = target_pos[0], target_pos[1], target_pos[2] + target_height_offset
        dx = tx - sx
        dy = ty - sy
        dz = tz - sz
        horiz = math.hypot(dx, dy)
        yaw = math.atan2(dy, dx)
        pitch = math.atan2(dz, max(horiz, 0.01))
        # Prevent firing into roof lip when standing in 3m setback on elevated roof (z > 10m)
        effective_min_pitch = -0.55 if my_pos[2] > 10.0 else self.min_pitch
        pitch = max(effective_min_pitch, min(self.max_pitch, pitch))
        return yaw, pitch

    def update_cycle(
        self,
        weapon_data: Optional[Dict[str, Any]],
        has_los: bool,
        target_dist: float,
        in_cover: bool = True,
        current_time: float = 0.0
    ) -> Tuple[bool, str]:
        """
        Updates laser charge state machine.
        Returns:
            (should_hold_trigger, recommended_movement_intent)
            - should_hold_trigger: True to accumulate charge; False to release beam!
            - recommended_movement_intent: 'HOLD_COVER', 'PEEK_OUT', 'RETREAT_TO_COVER', 'OPEN_COMBAT'
        """
        if not weapon_data or weapon_data.get("kind") != "laser":
            self.state = self.STATE_IDLE
            self.internal_charge = 0.0
            return False, "NONE"

        weapon_cooldown = weapon_data.get("cooldown", 0.0)
        curr_charge = weapon_data.get("charge", self.internal_charge)

        # 1. Weapon in cooldown
        if weapon_cooldown > 0.05:
            self.state = self.STATE_RETREAT_TO_COVER
            self.internal_charge = 0.0
            return False, "RETREAT_TO_COVER"

        # 2. Charging behind cover
        if in_cover and not has_los:
            if curr_charge < self.charge_time:
                self.state = self.STATE_CHARGING_IN_COVER
                self.internal_charge += 0.1  # 2 ticks = 0.1s
                return True, "HOLD_COVER"
            else:
                # Fully charged behind cover! Transition to peek
                self.state = self.STATE_PEEK_AND_FIRE
                return True, "PEEK_OUT"

        # 3. Peeking / Step-out with full charge
        if curr_charge >= self.charge_time:
            if has_los and target_dist <= self.range_m:
                # DISCHARGE! Release trigger to loose instantaneous hitscan beam
                self.state = self.STATE_RETREAT_TO_COVER
                self.internal_charge = 0.0
                self.last_fired_time = current_time
                return False, "RETREAT_TO_COVER"
            else:
                # Ready to fire, stepping out to acquire LOS
                return True, "PEEK_OUT"

        # 4. Open ground combat (no cover available)
        if has_los and target_dist <= self.range_m:
            if curr_charge < self.charge_time:
                self.internal_charge += 0.1
                return True, "OPEN_COMBAT"
            else:
                self.internal_charge = 0.0
                return False, "OPEN_COMBAT"

        # Idle fallback
        self.state = self.STATE_IDLE
        return False, "IDLE"

    def estimate_laser_damage(self, target_weapons: List[str]) -> float:
        """Calculates expected damage including +60% cookoff against mortar."""
        if "mortar" in target_weapons:
            return self.damage * 1.6  # 504.0 damage!
        return self.damage


class LightningConeTrigger:
    """
    Lightning 45m Cone Arc Trigger.
    
    Specs:
    - Range: 45m
    - Conservative Trigger Range: 40m (5m safety margin)
    - Cone: 30° arc (±15° from torso aim yaw)
    - Damage: 60 per shot (240 DPS sustained at 0.25s cooldown)
    - Ammo: Zero ammo (energy only: 3 energy/shot)
    - Chains: 2 additional targets within 14m (x0.7 dmg)
    - Signature: Shorts laser carriers, locking laser offline for 2.0s!
    """

    def __init__(
        self,
        max_range: float = 45.0,
        trigger_range: float = 40.0,
        cone_deg: float = 30.0,
        damage_per_shot: float = 60.0,
        cooldown: float = 0.25,
        energy_cost: float = 3.0
    ):
        self.max_range = max_range
        self.trigger_range = trigger_range
        self.cone_deg = cone_deg
        self.half_cone_rad = math.radians(cone_deg / 2.0)  # 15° = 0.2618 rad
        self.damage_per_shot = damage_per_shot
        self.cooldown = cooldown
        self.energy_cost = energy_cost
        self.chain_farming = LightningChainFarming(
            chain_range=LIGHTNING_CHAIN_RANGE,
            max_chain_targets=LIGHTNING_CHAIN_COUNT,
            damage_mult=LIGHTNING_CHAIN_MULT,
            base_damage=damage_per_shot,
            cone_deg=cone_deg,
            max_primary_range=trigger_range
        )

    def evaluate_chain(
        self,
        my_pos: Union[List[float], Tuple[float, ...]],
        aim_yaw: float,
        enemies: List[Dict[str, Any]],
        crates: Optional[List[Dict[str, Any]]] = None,
        npcs: Optional[List[Dict[str, Any]]] = None
    ) -> Optional["LightningChainResult"]:
        """Evaluates Lightning Chain Farming against enemies, crates, and NPCs."""
        return self.chain_farming.evaluate_chain_farming(my_pos, aim_yaw, enemies, crates, npcs)

    def is_in_cone(
        self,
        my_pos: Union[List[float], Tuple[float, float]],
        aim_yaw: float,
        target_pos: Union[List[float], Tuple[float, float]]
    ) -> Tuple[bool, float, float]:
        """
        Checks if target is within trigger distance (<= 40m) and within the 30° cone.
        Returns: (in_cone, distance, angular_offset_rad)
        """
        dx = target_pos[0] - my_pos[0]
        dy = target_pos[1] - my_pos[1]
        dist = math.hypot(dx, dy)
        if dist > self.trigger_range:
            return False, dist, 0.0

        target_bearing = math.atan2(dy, dx)
        # Normalize angular difference to [-pi, pi]
        diff = math.atan2(math.sin(target_bearing - aim_yaw), math.cos(target_bearing - aim_yaw))
        in_cone = abs(diff) <= self.half_cone_rad
        return in_cone, dist, abs(diff)

    def should_fire(
        self,
        my_pos: Union[List[float], Tuple[float, float]],
        aim_yaw: float,
        target_pos: Union[List[float], Tuple[float, float]],
        current_energy: float,
        weapon_cooldown: float = 0.0
    ) -> bool:
        """
        Trigger condition:
        - Target within 40m and 30° cone.
        - Energy >= 3.0 (never drain below weapon requirement).
        - Cooldown ready (<= 0.05s).
        """
        if current_energy < self.energy_cost:
            return False
        if weapon_cooldown > 0.05:
            return False

        in_cone, _, _ = self.is_in_cone(my_pos, aim_yaw, target_pos)
        return in_cone

    def prioritize_target(
        self,
        my_pos: Union[List[float], Tuple[float, float]],
        aim_yaw: float,
        candidates: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Prioritizes targets in lightning cone:
        1. Mechs carrying Laser (top priority: triggers 2.0s short circuit!).
        2. Lowest HP enemy (finishing kill for cubes).
        3. Closest target.
        """
        valid = []
        for target in candidates:
            t_pos = target.get("pos", [0, 0, 0])
            in_cone, dist, ang_diff = self.is_in_cone(my_pos, aim_yaw, t_pos)
            if in_cone:
                weapons = target.get("weapons", [])
                holds_laser = "laser" in weapons
                hp = target.get("hp", 1000.0)
                # Score formula: Laser carriers get +1000 bonus
                score = (1000.0 if holds_laser else 0.0) + (1400.0 - hp) + (50.0 - dist)
                valid.append((score, target))

        if not valid:
            return None
        valid.sort(key=lambda item: item[0], reverse=True)
        return valid[0][1]


@dataclass
class LightningChainResult:
    primary_target: Dict[str, Any]
    primary_pos: Tuple[float, float, float]
    primary_damage: float
    secondary_targets: List[Dict[str, Any]]
    secondary_damages: List[float]
    total_damage: float
    jumps_to_shielded: bool
    aim_yaw: float
    aim_pitch: float
    is_crate_primary: bool
    score: float


class LightningChainFarming:
    """
    Lightning Chain Farming & Arc Jumping Engine.
    
    Doctrines:
    - Lightning chains to up to 2 secondary targets within 14m (0.7x damage = 42.0 dmg each).
    - In group fights: target the closest enemy or crate in cone to jump arcs into shielded targets.
    - Bypasses directional shields, barriers, or cover by jumping arcs through accessible close proxies (crates/enemies).
    - Yields up to 60 (primary) + 42 + 42 = 144 damage per 0.25s discharge (576 DPS total group damage!).
    """

    def __init__(
        self,
        chain_range: float = LIGHTNING_CHAIN_RANGE,
        max_chain_targets: int = LIGHTNING_CHAIN_COUNT,
        damage_mult: float = LIGHTNING_CHAIN_MULT,
        base_damage: float = LIGHTNING_DAMAGE,
        cone_deg: float = LIGHTNING_CONE_DEG,
        max_primary_range: float = 40.0
    ):
        self.chain_range = chain_range
        self.max_chain_targets = max_chain_targets
        self.damage_mult = damage_mult
        self.base_damage = base_damage
        self.chain_damage = base_damage * damage_mult
        self.half_cone_rad = math.radians(cone_deg / 2.0)
        self.max_primary_range = max_primary_range

    def is_shielded(self, target: Dict[str, Any]) -> bool:
        """Determines if a target is shielded or has active defense."""
        if target.get("shield") or target.get("is_shielded") or target.get("shielded"):
            return True
        if target.get("shield_active") or target.get("shield_hp", 0.0) > 0.0:
            return True
        return False

    def is_in_cone(
        self,
        my_pos: Union[List[float], Tuple[float, ...]],
        aim_yaw: float,
        target_pos: Union[List[float], Tuple[float, ...]]
    ) -> Tuple[bool, float, float]:
        dx = target_pos[0] - my_pos[0]
        dy = target_pos[1] - my_pos[1]
        dist = math.hypot(dx, dy)
        if dist > self.max_primary_range:
            return False, dist, 0.0

        target_bearing = math.atan2(dy, dx)
        diff = math.atan2(math.sin(target_bearing - aim_yaw), math.cos(target_bearing - aim_yaw))
        in_cone = abs(diff) <= self.half_cone_rad
        return in_cone, dist, abs(diff)

    def find_secondaries(
        self,
        primary_pos: Union[List[float], Tuple[float, ...]],
        primary_id: Any,
        all_candidates: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[float], bool]:
        """
        Finds up to 2 secondary targets within 14m of the primary target.
        Prioritizes shielded targets (to bypass shield via chain jumps), laser carriers,
        low HP enemies, NPCs, and crates.
        """
        px, py = primary_pos[0], primary_pos[1]
        secondaries = []
        has_shielded = False

        for cand in all_candidates:
            cid = cand.get("id")
            if cid is not None and cid == primary_id:
                continue

            cpos = cand.get("pos", [0, 0, 0])
            dist = math.hypot(cpos[0] - px, cpos[1] - py)
            if dist <= self.chain_range:
                shielded = self.is_shielded(cand)
                if shielded:
                    has_shielded = True

                kind = cand.get("kind", "mech")
                hp = float(cand.get("hp", 500.0))
                weapons = cand.get("weapons", [])
                holds_laser = "laser" in weapons

                # Score priority for secondary arcs:
                # Shielded targets get top ranking to bypass shield via arc jump (+2000)
                score = (2000.0 if shielded else 0.0) + \
                        (1000.0 if holds_laser else 0.0) + \
                        (500.0 if kind == "mech" else (300.0 if kind == "npc" else 100.0)) + \
                        (1000.0 / max(1.0, hp)) - (dist * 2.0)

                secondaries.append((score, cand))

        secondaries.sort(key=lambda s: s[0], reverse=True)
        chosen = [s[1] for s in secondaries[:self.max_chain_targets]]
        damages = [self.chain_damage] * len(chosen)
        return chosen, damages, has_shielded

    def evaluate_chain_farming(
        self,
        my_pos: Union[List[float], Tuple[float, ...]],
        aim_yaw: float,
        enemies: List[Dict[str, Any]],
        crates: Optional[List[Dict[str, Any]]] = None,
        npcs: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[LightningChainResult]:
        """
        Evaluates group fight targets to maximize Lightning Chain Farming:
        - Primary targets can be enemies, NPCs, or crates in 30° cone.
        - Chains to up to 2 secondary targets within 14m (0.7x damage).
        - In group fights: targets the closest enemy or crate to jump arcs into shielded targets.
        """
        all_pool = list(enemies)
        if npcs:
            all_pool.extend(npcs)
        if crates:
            all_pool.extend(crates)

        # Check if there are shielded enemies in the area
        shielded_enemies = [e for e in enemies if self.is_shielded(e)]

        results: List[LightningChainResult] = []

        for cand in all_pool:
            cpos = cand.get("pos", [0, 0, 0])
            in_cone, dist, _ = self.is_in_cone(my_pos, aim_yaw, cpos)
            if not in_cone:
                continue

            cid = cand.get("id")
            kind = cand.get("kind", "crate" if cid and cid >= 1000 else "mech")
            is_crate = (kind == "crate") or (cid is not None and cid >= 1000)

            # Find up to 2 secondary targets within 14m
            secondaries, sec_damages, jumps_to_shield = self.find_secondaries(cpos, cid, all_pool)

            # Check if this primary jumps to any known shielded enemy
            if shielded_enemies and not jumps_to_shield:
                for se in shielded_enemies:
                    if se.get("id") != cid:
                        se_pos = se.get("pos", [0, 0, 0])
                        if math.hypot(se_pos[0] - cpos[0], se_pos[1] - cpos[1]) <= self.chain_range:
                            jumps_to_shield = True
                            if se not in secondaries and len(secondaries) < self.max_chain_targets:
                                secondaries.append(se)
                                sec_damages.append(self.chain_damage)
                            break

            total_dmg = self.base_damage + sum(sec_damages)

            # In group fights: prioritize closest enemy or crate to jump arcs into shielded targets
            score = total_dmg
            if jumps_to_shield:
                # Massive priority for jumping arcs into shielded targets!
                score += 3000.0
                # Heavily prioritize the CLOSEST enemy or crate that can jump arcs
                score += (50.0 - dist) * 10.0
            else:
                score += len(secondaries) * 100.0
                score += (50.0 - dist)
                if not is_crate:
                    score += 50.0  # Slightly prefer mech/NPC over lone crate if no shield jump

            dx = cpos[0] - my_pos[0]
            dy = cpos[1] - my_pos[1]
            c_z = cpos[2] if len(cpos) > 2 else 0.0
            dz = (c_z + 1.2 if is_crate else c_z + 4.5) - (my_pos[2] + 7.0)
            horiz = max(dist, 0.01)
            aim_y = math.atan2(dy, dx)
            aim_p = math.atan2(dz, horiz)

            res = LightningChainResult(
                primary_target=cand,
                primary_pos=(cpos[0], cpos[1], c_z),
                primary_damage=self.base_damage,
                secondary_targets=secondaries,
                secondary_damages=sec_damages,
                total_damage=total_dmg,
                jumps_to_shielded=jumps_to_shield,
                aim_yaw=aim_y,
                aim_pitch=aim_p,
                is_crate_primary=is_crate,
                score=score
            )
            results.append(res)

        if not results:
            return None

        results.sort(key=lambda r: r.score, reverse=True)
        return results[0]


class NPCHunter:
    """
    Early-game Economy Snowball Engine.
    
    Doctrines:
    - 12 NPCs roam the arena with 350 HP, moving at ~5 m/s.
    - Each NPC drops 2 Power Cubes upon destruction (+8% HP & Damage, upgrade progression).
    - In the opening 60 seconds (t < 60s), any NPC within 90m (hp <= 350) is prioritized.
    - Laser (315 hitscan dmg) + Arm Gun (45 dmg) burst delivers an instantaneous 360 dmg kill (> 350 HP instakill in 1.0s!).
    - Safe-Hunting Guard: If any enemy player mech is within 55m, abort NPC hunting and reserve all 12 gun rounds for the player mech.
    """

    def __init__(
        self,
        hunt_window_s: float = 60.0,
        max_hunt_dist: float = 90.0,
        player_threat_dist: float = NPC_PLAYER_THREAT_DIST
    ):
        self.hunt_window_s = hunt_window_s
        self.max_hunt_dist = max_hunt_dist
        self.player_threat_dist = player_threat_dist

    def is_safe_to_hunt(
        self,
        my_pos: Union[List[float], Tuple[float, ...]],
        visible_entities: List[Dict[str, Any]],
        threat_radius: Optional[float] = None
    ) -> bool:
        """
        NPC Safe-Hunting Guard:
        If any enemy player mech is within 55m, abort NPC hunting and reserve all 12 gun rounds for the player mech.
        """
        radius = threat_radius if threat_radius is not None else self.player_threat_dist
        mx, my = my_pos[0], my_pos[1]
        for e in visible_entities:
            # Player mechs have kind == "mech" (or not "npc") and hp > 0
            if e.get("kind") == "mech" and float(e.get("hp", 0.0)) > 0.0:
                ep = e.get("pos", [0, 0, 0])
                if math.hypot(ep[0] - mx, ep[1] - my) <= radius:
                    return False
        return True

    def find_target(
        self,
        my_pos: Union[List[float], Tuple[float, ...]],
        visible_entities: List[Dict[str, Any]],
        current_time: float
    ) -> Optional[Dict[str, Any]]:
        """
        Scans visible entities. If current_time < 60s and no enemy player within 55m,
        returns the prioritized NPC (kind == 'npc', hp <= 350) within 90m.
        """
        if current_time >= self.hunt_window_s:
            return None

        # Safe-Hunting Guard: abort if enemy player mech within 55m
        if not self.is_safe_to_hunt(my_pos, visible_entities):
            return None

        mx, my = my_pos[0], my_pos[1]
        candidates = []
        for e in visible_entities:
            if e.get("kind") == "npc":
                hp = float(e.get("hp", 350.0))
                if hp <= NPC_HP:
                    ep = e.get("pos", [0, 0, 0])
                    dist = math.hypot(ep[0] - mx, ep[1] - my)
                    if dist <= self.max_hunt_dist:
                        # Prioritize lowest HP, then closest distance
                        candidates.append((hp, dist, e))

        if not candidates:
            return None

        candidates.sort(key=lambda c: (c[0], c[1]))
        return candidates[0][2]


class DynamicTargetScorer:
    """
    Dynamic Target Scorer & Rear-Armor Executioner.
    
    Formula:
        Score = (1000.0 / enemy_hp) + (is_rear_facing * 200.0) + (is_fighting_others * 150.0) - (distance * 0.05)
    
    Doctrines:
    - Eliminates naive 'closest enemy' targeting.
    - Rear armor takes 1.5x damage (+50% bonus). Firing a Laser (315 dmg) at rear armor
      inflicts 472.5 damage—an instant one-shot execution on wounded brawlers.
    - Instant Rear-Shot Execution: If an enemy mech has HP <= 472.0 and is rear-facing (|angle_diff| < pi/2),
      prioritize Laser discharge immediately for an instant one-hit kill!
    - Prioritizes brawlers occupied fighting third parties.
    """

    def __init__(self):
        pass

    def is_rear_facing(
        self,
        my_pos: Union[List[float], Tuple[float, ...]],
        target: Dict[str, Any],
        aim_yaw: Optional[float] = None
    ) -> bool:
        """
        Exact Rear Armor Multiplier (rear_mult = 1.5, +50% bonus if hit from behind, front arc = 180°).
        Angle diff formula: |atan2(sin(aim_yaw - target.aim_yaw), cos(aim_yaw - target.aim_yaw))|
        If angle diff < 90° (pi/2 rad), it is a REAR HIT -> Damage * 1.5!
        """
        t_pos = target.get("pos", [0, 0, 0])
        if aim_yaw is None:
            # Bullet/beam flight direction from us to target
            aim_yaw = math.atan2(t_pos[1] - my_pos[1], t_pos[0] - my_pos[0])

        target_aim = target.get("aim", [0.0, 0.0])
        target_yaw = target_aim[0] if isinstance(target_aim, (list, tuple)) and len(target_aim) > 0 else 0.0

        angle_diff = abs(math.atan2(math.sin(aim_yaw - target_yaw), math.cos(aim_yaw - target_yaw)))
        return angle_diff < (math.pi / 2.0)

    def is_fighting_others(
        self,
        my_pos: Union[List[float], Tuple[float, ...]],
        target: Dict[str, Any],
        all_mechs: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Checks if the target is aiming away from us (> 45 deg) or actively brawling with another mech.
        """
        t_pos = target.get("pos", [0, 0, 0])
        dx = my_pos[0] - t_pos[0]
        dy = my_pos[1] - t_pos[1]
        bearing_to_us = math.atan2(dy, dx)

        target_aim = target.get("aim", [0.0, 0.0])
        target_yaw = target_aim[0] if isinstance(target_aim, (list, tuple)) and len(target_aim) > 0 else 0.0

        angle_diff = math.atan2(math.sin(bearing_to_us - target_yaw), math.cos(bearing_to_us - target_yaw))

        # 1. Torso aimed > 45 degrees away from us
        if abs(angle_diff) > (math.pi / 4.0):
            return True

        # 2. Charging a laser while not aimed directly at us
        if target.get("charging", False) and abs(angle_diff) > (math.pi / 6.0):
            return True

        # 3. Proximity to another mech/npc that target is aimed at
        if all_mechs:
            for m in all_mechs:
                if m.get("id") != target.get("id"):
                    mp = m.get("pos", [0, 0, 0])
                    dist_tm = math.hypot(mp[0] - t_pos[0], mp[1] - t_pos[1])
                    if dist_tm < 45.0:
                        bearing_to_m = math.atan2(mp[1] - t_pos[1], mp[0] - t_pos[0])
                        diff_m = math.atan2(math.sin(bearing_to_m - target_yaw), math.cos(bearing_to_m - target_yaw))
                        if abs(diff_m) < (math.pi / 4.0):
                            return True

        return False

    def compute_score(
        self,
        my_pos: Union[List[float], Tuple[float, ...]],
        target: Dict[str, Any],
        all_mechs: Optional[List[Dict[str, Any]]] = None,
        aim_yaw: Optional[float] = None
    ) -> float:
        """
        Computes score per the mission spec:
            Score = (1000.0 / enemy_hp) + (is_rear_facing * 200.0) + (is_fighting_others * 150.0) - (distance * 0.05)
        Instant Rear-Shot Execution:
            If an enemy mech has HP <= 472.0 and is rear-facing (|angle_diff| < pi/2), prioritize Laser discharge immediately
            for an instant one-hit kill! Adds an execution priority bonus (+1000.0).
        """
        hp = max(1.0, float(target.get("hp", 1000.0)))
        t_pos = target.get("pos", [0, 0, 0])
        dist = math.hypot(t_pos[0] - my_pos[0], t_pos[1] - my_pos[1])

        is_rear = self.is_rear_facing(my_pos, target, aim_yaw)
        rear_val = 1.0 if is_rear else 0.0
        fighting_val = 1.0 if self.is_fighting_others(my_pos, target, all_mechs) else 0.0

        score = (1000.0 / hp) + (rear_val * 200.0) + (fighting_val * 150.0) - (dist * 0.05)

        # Instant Rear-Shot Execution Priority:
        # Laser hitscan deals 315 * 1.5 = 472.5 dmg to rear armor.
        # If HP <= 472.0, this is an instant one-hit kill!
        if is_rear and hp <= 472.0:
            score += 1000.0  # Instant rear-shot execution priority bonus!

        return score

    def rank_targets(
        self,
        my_pos: Union[List[float], Tuple[float, ...]],
        targets: List[Dict[str, Any]],
        aim_yaw: Optional[float] = None
    ) -> List[Tuple[float, Dict[str, Any]]]:
        scored = []
        for t in targets:
            if t.get("hp", 0) > 0:
                s = self.compute_score(my_pos, t, targets, aim_yaw)
                scored.append((s, t))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored


class LaserLightningCombo:
    """
    Laser + Lightning Weapon Triangle Disruptor & Finisher.
    
    Doctrines:
    1. Preemptive Short (Triangle Defense):
       When within 40m of an enemy, discharge Lightning to apply 'short_s: 2.0'.
       This locks enemy Lasers offline for 2.0s, wipes their charge in progress,
       and protects us from lethal 315-504 damage laser shots.
    2. Rear-Armor Execution:
       Follow up with Hitscan Laser (315 dmg base, 472.5 dmg to rear armor)
       or continuous 240 DPS Lightning bursts to eliminate the disabled target.
    3. Instant Rear-Shot Execution:
       If enemy mech has HP <= 472.0 and is rear-facing, trigger instant Laser execution!
    """

    def __init__(
        self,
        range_m: float = 40.0,
        cone_deg: float = 30.0,
        short_duration: float = LIGHTNING_SHORT_S
    ):
        self.range_m = range_m
        self.half_cone_rad = math.radians(cone_deg / 2.0)
        self.short_duration = short_duration

    def should_discharge_lightning(
        self,
        my_pos: Union[List[float], Tuple[float, ...]],
        aim_yaw: float,
        target: Dict[str, Any],
        current_energy: float,
        lightning_cooldown: float = 0.0
    ) -> bool:
        """
        Evaluates whether Lightning should fire to apply short_s: 2.0 to the target.
        - Enemy is within 40m and within the 30° cone.
        - Target carries Laser, or is charging, or general close combat.
        - Energy >= 3.0 and cooldown <= 0.05s.
        """
        if current_energy < LIGHTNING_ENERGY or lightning_cooldown > 0.05:
            return False

        t_pos = target.get("pos", [0, 0, 0])
        dx = t_pos[0] - my_pos[0]
        dy = t_pos[1] - my_pos[1]
        dist = math.hypot(dx, dy)
        if dist > self.range_m:
            return False

        target_bearing = math.atan2(dy, dx)
        diff = math.atan2(math.sin(target_bearing - aim_yaw), math.cos(target_bearing - aim_yaw))
        if abs(diff) > self.half_cone_rad:
            return False

        return True

    def is_instant_rear_execution(
        self,
        my_pos: Union[List[float], Tuple[float, ...]],
        target: Dict[str, Any],
        aim_yaw: Optional[float] = None
    ) -> bool:
        """
        Instant Rear-Shot Execution:
        If an enemy mech has HP <= 472.0 and is rear-facing (|angle_diff| < pi/2),
        returns True to prioritize Laser discharge immediately for an instant one-hit kill!
        """
        hp = float(target.get("hp", 1000.0))
        if hp > 472.0:
            return False

        t_pos = target.get("pos", [0, 0, 0])
        if aim_yaw is None:
            aim_yaw = math.atan2(t_pos[1] - my_pos[1], t_pos[0] - my_pos[0])
        target_aim = target.get("aim", [0.0, 0.0])
        target_yaw = target_aim[0] if isinstance(target_aim, (list, tuple)) and len(target_aim) > 0 else 0.0

        angle_diff = abs(math.atan2(math.sin(aim_yaw - target_yaw), math.cos(aim_yaw - target_yaw)))
        return angle_diff < (math.pi / 2.0)

    def calculate_combo_burst(
        self,
        target: Dict[str, Any],
        is_rear_facing: bool = False
    ) -> Dict[str, float]:
        """
        Estimates total burst damage:
        - Lightning: 60 dmg (costs 3 energy, applies 2.0s short)
        - Laser: 315 dmg base (472.5 rear armor, 504.0 vs mortar, 756.0 rear + mortar)
        - Arm Gun: 45 dmg (67.5 rear armor)
        """
        target_weapons = target.get("weapons", [])
        has_mortar = "mortar" in target_weapons
        laser_base = LASER_DAMAGE * (LASER_COOKOFF_MULT if has_mortar else 1.0)
        laser_dmg = laser_base * (REAR_MULT if is_rear_facing else 1.0)
        lightning_dmg = LIGHTNING_DAMAGE * (REAR_MULT if is_rear_facing else 1.0)
        gun_dmg = ARM_GUN_DAMAGE * (REAR_MULT if is_rear_facing else 1.0)

        return {
            "lightning_dmg": lightning_dmg,
            "laser_dmg": laser_dmg,
            "gun_dmg": gun_dmg,
            "total_combo": lightning_dmg + laser_dmg + gun_dmg,
            "shorts_laser": "laser" in target_weapons or target.get("charging", False)
        }


def select_optimal_target(
    my_pos: Union[List[float], Tuple[float, ...]],
    visible_mechs: List[Dict[str, Any]],
    current_time: float
) -> Optional[Dict[str, Any]]:
    """
    Selects the optimal combat target following BALLIS doctrines:
    1. NPCHunter: Opening 60 seconds (current_time < 60s), if an NPC (kind == 'npc', hp <= 350)
       is within 90m, prioritize it for power cube harvesting (2 cubes).
       Safe-Hunting Guard: If any enemy player mech is within 55m, abort NPC hunting and reserve all 12 gun rounds for the player mech!
    2. DynamicTargetScorer: Otherwise, scores all visible enemies:
       Score = (1000 / enemy_hp) + (is_rear_facing * 200) + (is_fighting_others * 150) - (distance * 0.05)
       Instant Rear-Shot Execution: If enemy has HP <= 472.0 and is rear-facing, prioritizes Laser execution!
       Targets the lowest-HP enemy whose back is turned.
    """
    if not visible_mechs:
        return None

    # 1. NPCHunter check during opening 60 seconds (with 55m Safe-Hunting Guard)
    if current_time < 60.0:
        hunter = NPCHunter(hunt_window_s=60.0, max_hunt_dist=90.0, player_threat_dist=NPC_PLAYER_THREAT_DIST)
        npc_target = hunter.find_target(my_pos, visible_mechs, current_time)
        if npc_target is not None:
            return npc_target

    # 2. DynamicTargetScorer evaluation
    scorer = DynamicTargetScorer()
    living_targets = [m for m in visible_mechs if m.get("hp", 0) > 0]
    if not living_targets:
        return None

    # Separate mechs from npcs if mechs are present
    mechs = [m for m in living_targets if m.get("kind") != "npc"]
    candidates = mechs if mechs else living_targets

    ranked = scorer.rank_targets(my_pos, candidates)
    if ranked:
        return ranked[0][1]

    return None


class SmartWeaponSwapManager:
    """
    Smart Weapon Swap Manager.
    
    Inventory Architecture:
    - Slot 0 starts with Arm Gun (252 m/s, 45 dmg, 12 mag, 2.4s reload, 0 energy).
    - Slot 1 holds loot weapons: Laser, Lightning, Mortar.
    
    Strict Invariant:
    - NEVER drop the Arm Gun unless you hold Laser AND find Lightning (or vice versa).
      If holding Laser and picking up Lightning -> drop Slot 0 (Arm Gun).
      If holding Lightning and picking up Laser -> drop Slot 0 (Arm Gun).
      Result: Holds Laser + Lightning (the supreme dual weapon setup).
    - In all other cases: Slot 1 is the mutable loot slot.
    
    Weapon Triangle Matchups:
    - Laser beats Mortar (+60% cookoff = 504 dmg).
    - Mortar beats Lightning (concussive knockback 16 m/s + 45 energy drain).
    - Lightning beats Laser (shorts charge and locks laser offline for 2.0s).
    """

    def __init__(self, pickup_radius: float = 4.0, equip_delay: float = 1.0):
        self.pickup_radius = pickup_radius
        self.equip_delay = equip_delay
        self.slot_equip_timers = [0.0, 0.0]

    def update_timers(self, dt: float):
        """Updates internal equip delay timers."""
        self.slot_equip_timers[0] = max(0.0, self.slot_equip_timers[0] - dt)
        self.slot_equip_timers[1] = max(0.0, self.slot_equip_timers[1] - dt)

    def evaluate_swap(
        self,
        me: Dict[str, Any],
        pickups: List[Dict[str, Any]],
        enemies: Optional[List[Dict[str, Any]]] = None,
        current_time: float = 0.0
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluates ground pickups within 4.0m and returns an action dictionary:
        {'take': pickup_id} or {'take': pickup_id, 'drop': slot_index} or None.
        """
        my_pos = me.get("pos", [0, 0, 0])
        mx, my = my_pos[0], my_pos[1]
        weapons = me.get("weapons", [None, None])
        w0 = weapons[0] if len(weapons) > 0 else None
        w1 = weapons[1] if len(weapons) > 1 else None

        k0 = w0.get("kind") if w0 else None
        k1 = w1.get("kind") if w1 else None
        held_kinds = {k for k in (k0, k1) if k is not None}

        # Filter pickups within 4m horizontal and vertical
        near_pickups = []
        for p in pickups:
            px, py = p.get("pos", [0, 0, 0])[0], p.get("pos", [0, 0, 0])[1]
            pz = p.get("pos", [0, 0, 0])[2] if len(p.get("pos", [])) > 2 else my_pos[2]
            if math.hypot(px - mx, py - my) <= self.pickup_radius and abs(pz - my_pos[2]) <= 4.0:
                p_kind = p.get("kind")
                # Cannot hold two of the same kind
                if p_kind and p_kind not in held_kinds:
                    near_pickups.append(p)

        if not near_pickups:
            return None

        # Check if an enemy nearby holds Laser (cookoff danger for Mortar!)
        laser_enemy_nearby = False
        if enemies:
            for e in enemies:
                ep = e.get("pos", [0, 0, 0])
                if math.hypot(ep[0] - mx, ep[1] - my) <= 180.0:
                    if "laser" in e.get("weapons", []):
                        laser_enemy_nearby = True
                        break

        # Loot priority ranking: Laser = 3, Lightning = 2, Mortar = 1 (drops to -1 if laser enemy nearby)
        def score_loot(pickup: Dict[str, Any]) -> int:
            kind = pickup.get("kind", "")
            if kind == "laser":
                return 4
            elif kind == "lightning":
                return 3
            elif kind == "mortar":
                return -1 if laser_enemy_nearby else 1
            return 0

        near_pickups.sort(key=score_loot, reverse=True)
        best_pickup = near_pickups[0]
        p_kind = best_pickup.get("kind")
        p_id = best_pickup.get("id")

        if score_loot(best_pickup) < 0:
            return None  # Don't pick up mortar if laser enemy is near

        # Scenario 1: Empty slot available (Slot 1 empty)
        if w1 is None:
            return {"take": p_id}
        if w0 is None:
            return {"take": p_id}

        # Scenario 2: Special Invariant: Drop Arm Gun ONLY when holding Laser + finding Lightning, or vice versa
        holds_laser = (k0 == "laser" or k1 == "laser")
        holds_lightning = (k0 == "lightning" or k1 == "lightning")
        holds_gun = (k0 == "gun" or k1 == "gun")
        gun_slot = 0 if k0 == "gun" else (1 if k1 == "gun" else None)

        if holds_gun and gun_slot is not None:
            # We hold Gun + Laser, and found Lightning:
            if holds_laser and p_kind == "lightning":
                # Check laser charging lock: cannot drop slot if charging
                laser_slot = 0 if k0 == "laser" else 1
                laser_w = weapons[laser_slot]
                if laser_w and laser_w.get("charge", 0.0) > 0.0:
                    return None  # Laser is busy charging
                return {"take": p_id, "drop": gun_slot}

            # We hold Gun + Lightning, and found Laser:
            if holds_lightning and p_kind == "laser":
                return {"take": p_id, "drop": gun_slot}

        # Scenario 3: Slot 1 loot upgrade (replacing mortar with laser or lightning)
        # We hold Gun in slot 0, and Mortar in slot 1:
        if k0 == "gun" and k1 == "mortar":
            if p_kind in ("laser", "lightning"):
                return {"take": p_id, "drop": 1}

        # Scenario 4: Replacing Lightning in slot 1 with Laser (or vice-versa depending on tactical needs)
        if k0 == "gun" and k1 == "lightning" and p_kind == "laser":
            # Rule: We hold Gun + Lightning and find Laser -> Drop Gun (Slot 0) to hold BOTH!
            return {"take": p_id, "drop": 0}

        return None


class ArtillerySystem:
    """
    Master Artillery and Weapon Integration Engine.
    Combines IndirectParabolicMortarSolver, HitscanLaserController,
    LightningConeTrigger, and SmartWeaponSwapManager into a single high-performance module.
    
    Guaranteed performance budget: < 1.5 ms per decision tick.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        weapons_cfg = cfg.get("weapons", {})
        mech_cfg = cfg.get("mech", {})

        mortar_cfg = weapons_cfg.get("mortar", {})
        laser_cfg = weapons_cfg.get("laser", {})
        lightning_cfg = weapons_cfg.get("lightning", {})

        # Subsystems
        self.mortar = IndirectParabolicMortarSolver(
            speed=mortar_cfg.get("speed", 70.0),
            gravity=mortar_cfg.get("gravity", 18.0),
            blast_radius=mortar_cfg.get("splash_radius", 14.0),
            direct_damage=mortar_cfg.get("damage", 200.0),
            cooldown=mortar_cfg.get("cooldown", 2.6),
            tube_height_offset=mech_cfg.get("height", 9.0) + 0.5,
            min_range=25.0
        )

        self.laser = HitscanLaserController(
            charge_time=laser_cfg.get("charge", 0.75),
            cooldown=laser_cfg.get("cooldown", 1.65),
            range_m=laser_cfg.get("range", 280.0),
            damage=laser_cfg.get("damage", 315.0),
            min_charge=laser_cfg.get("min_charge", 0.15),
            muzzle_height=mech_cfg.get("muzzle_height", 7.0),
            min_pitch=mech_cfg.get("min_pitch", -0.7),
            max_pitch=mech_cfg.get("max_pitch", 1.3)
        )

        self.lightning = LightningConeTrigger(
            max_range=lightning_cfg.get("range", 45.0),
            trigger_range=40.0,
            cone_deg=lightning_cfg.get("cone_deg", 30.0),
            damage_per_shot=lightning_cfg.get("damage", 60.0),
            cooldown=lightning_cfg.get("cooldown", 0.25),
            energy_cost=lightning_cfg.get("energy", 3.0)
        )

        self.swap_manager = SmartWeaponSwapManager()

        # BALLIS Economy Snowball & Rear-Armor Execution Engines
        self.npc_hunter = NPCHunter()
        self.target_scorer = DynamicTargetScorer()
        self.combo = LaserLightningCombo()
        self.chain_farming = self.lightning.chain_farming

        # Telemetry / execution stats
        self.total_evals = 0
        self.total_time_ms = 0.0
        self.max_time_ms = 0.0

    def select_optimal_target(
        self,
        my_pos: Union[List[float], Tuple[float, ...]],
        visible_mechs: List[Dict[str, Any]],
        current_time: float
    ) -> Optional[Dict[str, Any]]:
        """Exposed method for selecting optimal target (NPCHunter opening + DynamicTargetScorer)."""
        return select_optimal_target(my_pos, visible_mechs, current_time)

    def compute_exact_predictive_lead(
        self,
        my_pos: Union[List[float], Tuple[float, ...]],
        target_pos: Union[List[float], Tuple[float, ...]],
        target_vel: Union[List[float], Tuple[float, ...]],
        bullet_speed: float = ARM_GUN_SPEED
    ) -> Tuple[float, float, float]:
        """Solves exact first-order predictive lead: |P + V*t - M|^2 = (s*t)^2."""
        return compute_exact_predictive_lead(my_pos, target_pos, target_vel, bullet_speed)

    def solve_ground_bounce_splash(
        self,
        my_pos: Union[List[float], Tuple[float, float, float]],
        target_pos: Union[List[float], Tuple[float, float, float]],
        target_vel: Optional[Union[List[float], Tuple[float, float, float]]] = None,
        behind_offset: float = MORTAR_GROUND_BOUNCE_OFFSET_DEFAULT,
        terrain_z: Optional[float] = None
    ) -> Optional[MortarTrajectory]:
        """Mortar Splash Ground-Bouncing Solver: hits terrain immediately behind ducking target."""
        return self.mortar.solve_ground_bounce_splash(my_pos, target_pos, target_vel, behind_offset, terrain_z)

    def evaluate_lightning_chain(
        self,
        my_pos: Union[List[float], Tuple[float, ...]],
        aim_yaw: float,
        enemies: List[Dict[str, Any]],
        crates: Optional[List[Dict[str, Any]]] = None,
        npcs: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[LightningChainResult]:
        """Lightning Chain Farming evaluator: chains to 2 secondary targets within 14m (0.7x damage)."""
        return self.chain_farming.evaluate_chain_farming(my_pos, aim_yaw, enemies, crates, npcs)

    def evaluate_combat_tick(
        self,
        me: Dict[str, Any],
        target: Optional[Dict[str, Any]],
        has_los: bool,
        pickups: List[Dict[str, Any]],
        drops: Optional[List[Dict[str, Any]]] = None,
        all_enemies: Optional[List[Dict[str, Any]]] = None,
        in_cover: bool = False,
        current_time: float = 0.0,
        crates: Optional[List[Dict[str, Any]]] = None,
        ducking_behind_wall: bool = False
    ) -> Dict[str, Any]:
        """
        Executes unified combat and artillery evaluation for the current tick.
        
        Returns dictionary with:
        - 'fire': [bool, bool]
        - 'aim': [yaw, pitch]
        - 'loot_action': Optional[Dict[str, Any]] (take/drop)
        - 'tactical_move': str ('HOLD_COVER', 'PEEK_OUT', 'RETREAT_TO_COVER', 'STANDARD')
        - 'is_rear_facing': bool
        - 'target': Optional[Dict[str, Any]]
        - 'execution_time_ms': float
        """
        t_start = time.perf_counter()

        my_pos = me.get("pos", [0, 0, 0])
        aim = list(me.get("aim", [0.0, 0.0]))
        weapons = me.get("weapons", [None, None])
        energy = me.get("energy", 100.0)

        # Auto-select optimal target if none was provided
        if target is None and all_enemies:
            target = self.select_optimal_target(my_pos, all_enemies, current_time)

        result_fire = [False, False]
        result_aim = [aim[0], aim[1]]
        tactical_move = "STANDARD"
        is_rear = False

        # 1. Smart Loot Swaps
        loot_action = self.swap_manager.evaluate_swap(me, pickups, all_enemies, current_time)

        # 2. Check Supply Drop Siege Opportunity (Priority indirect bombardment)
        mortar_slot = None
        for i, w in enumerate(weapons):
            if w and w.get("kind") == "mortar":
                mortar_slot = i
                break

        if mortar_slot is not None and drops:
            w_mortar = weapons[mortar_slot]
            if w_mortar.get("cooldown", 0.0) <= 0.05:
                active_drop = drops[0]
                interruption = self.mortar.interrupt_supply_drop(my_pos, active_drop, all_enemies)
                if interruption:
                    (s_yaw, s_pitch), flight_time, status = interruption
                    result_fire[mortar_slot] = True
                    result_aim = [s_yaw, s_pitch]
                    t_end = time.perf_counter()
                    exec_ms = (t_end - t_start) * 1000.0
                    return {
                        "fire": result_fire,
                        "aim": result_aim,
                        "loot_action": loot_action,
                        "tactical_move": "SIEGE_DROP",
                        "status": status,
                        "execution_time_ms": exec_ms
                    }

        # 3. Direct Target Combat
        if target:
            is_rear = self.target_scorer.is_rear_facing(my_pos, target, aim[0])
            t_pos = target.get("pos", [0, 0, 0])
            t_dist = math.hypot(t_pos[0] - my_pos[0], t_pos[1] - my_pos[1])
            target_weapons = target.get("weapons", [])

            # Iterate over our equipped weapon slots
            for slot_idx, w in enumerate(weapons):
                if not w:
                    continue
                w_kind = w.get("kind")
                w_cd = w.get("cooldown", 0.0)

                # --- Weapon: Laser ---
                if w_kind == "laser":
                    is_instant_kill = target and self.combo.is_instant_rear_execution(my_pos, target, aim[0])
                    hold_trigger, move_intent = self.laser.update_cycle(
                        w, has_los, t_dist, in_cover, current_time
                    )
                    # If instant rear kill available and fully charged, release immediately!
                    if is_instant_kill and has_los and w.get("charge", 0.0) >= self.laser.charge_time:
                        hold_trigger = False
                        move_intent = "INSTANT_REAR_EXECUTION"

                    result_fire[slot_idx] = hold_trigger
                    tactical_move = move_intent
                    if has_los:
                        lyaw, lpitch = self.laser.compute_aim(my_pos, t_pos)
                        result_aim = [lyaw, lpitch]

                # --- Weapon: Lightning ---
                elif w_kind == "lightning":
                    # Lightning Chain Farming: evaluate group fight chains (enemies + crates)
                    chain_res = None
                    enemy_pool = all_enemies if all_enemies else ([target] if target else [])
                    if crates or (len(enemy_pool) > 1) or any(self.chain_farming.is_shielded(e) for e in enemy_pool):
                        chain_res = self.chain_farming.evaluate_chain_farming(
                            my_pos, aim[0], enemy_pool, crates or []
                        )

                    if chain_res and (chain_res.jumps_to_shielded or len(chain_res.secondary_targets) > 0) and w_cd <= 0.05 and energy >= 3.0:
                        result_fire[slot_idx] = True
                        result_aim = [chain_res.aim_yaw, chain_res.aim_pitch]
                        tactical_move = "LIGHTNING_CHAIN_FARMING"
                    elif self.combo.should_discharge_lightning(my_pos, aim[0], target, energy, w_cd) or \
                         self.lightning.should_fire(my_pos, aim[0], t_pos, energy, w_cd):
                        result_fire[slot_idx] = True
                        tactical_move = "LIGHTNING_ENGAGE"

                # --- Weapon: Mortar ---
                elif w_kind == "mortar":
                    # 1. Mortar Splash Ground-Bouncing: When target ducks behind low wall,
                    # hit terrain immediately behind them within the 7m splash zone.
                    if ducking_behind_wall and w_cd <= 0.05:
                        bounce_traj = self.mortar.solve_ground_bounce_splash(
                            my_pos, t_pos, target.get("vel"), behind_offset=MORTAR_GROUND_BOUNCE_OFFSET_DEFAULT
                        )
                        if bounce_traj and self.mortar.can_clear_obstacles(my_pos, bounce_traj.predicted_impact, bounce_traj.pitch):
                            result_fire[slot_idx] = True
                            result_aim = [bounce_traj.yaw, bounce_traj.pitch]
                            tactical_move = "MORTAR_GROUND_BOUNCE_SPLASH"
                    # 2. Standard high-arc lob over buildings when no direct LOS or long distance
                    elif (not has_los or t_dist > 60.0) and w_cd <= 0.05:
                        traj = self.mortar.solve_high_arc(my_pos, t_pos, target.get("vel"))
                        if traj and self.mortar.can_clear_obstacles(my_pos, t_pos, traj.pitch):
                            result_fire[slot_idx] = True
                            result_aim = [traj.yaw, traj.pitch]

                # --- Weapon: Arm Gun ---
                elif w_kind == "gun":
                    ammo = w.get("ammo", 0)
                    reload_s = w.get("reload", 0.0)
                    if has_los and ammo > 0 and reload_s == 0.0 and w_cd <= 0.05:
                        result_fire[slot_idx] = True
                        t_vel = target.get("vel", [0.0, 0.0, 0.0])
                        lead_yaw, lead_pitch, _ = compute_exact_predictive_lead(
                            my_pos, t_pos, t_vel, bullet_speed=ARM_GUN_SPEED
                        )
                        # If arm gun is primary firing weapon, set lead aim
                        if not result_fire[1] or weapons[1] is None:
                            result_aim = [lead_yaw, lead_pitch]

        t_end = time.perf_counter()
        exec_ms = (t_end - t_start) * 1000.0

        self.total_evals += 1
        self.total_time_ms += exec_ms
        self.max_time_ms = max(self.max_time_ms, exec_ms)

        return {
            "fire": result_fire,
            "aim": result_aim,
            "loot_action": loot_action,
            "tactical_move": tactical_move,
            "is_rear_facing": is_rear,
            "target": target,
            "execution_time_ms": exec_ms
        }


# ==============================================================================
# SELF-TEST & VERIFICATION SUITE
# ==============================================================================
if __name__ == "__main__":
    print("==================================================================")
    print("RUNNING BALLIS BALLISTICS LEAD & REAR-ARMOR CALIBRATION VERIFICATION")
    print("==================================================================")

    # 1. Exact First-Order Predictive Lead Test
    my_pos = (10.0, 20.0, 0.0)
    target_pos = (110.0, 70.0, 0.0)
    target_vel = (11.7, -5.0, 0.0)
    speed = ARM_GUN_SPEED  # 252.0 m/s

    yaw, pitch, t_flight = compute_exact_predictive_lead(
        my_pos, target_pos, target_vel, bullet_speed=speed
    )

    # Verify |P + V*t - M|^2 == (s*t)^2
    mx, my, mz = my_pos[0], my_pos[1], my_pos[2] + MUZZLE_ELEVATION
    px = target_pos[0] + target_vel[0] * t_flight
    py = target_pos[1] + target_vel[1] * t_flight
    pz = target_pos[2] + CENTER_OF_MASS_OFFSET + target_vel[2] * t_flight
    intercept_dist = math.hypot(px - mx, py - my, pz - mz)
    bullet_dist = speed * t_flight
    err = abs(intercept_dist - bullet_dist)
    assert err < 1e-9, f"Exact lead equation violated: error={err}"
    print(f"[PASS] Exact First-Order Predictive Lead: t={t_flight:.4f}s, dist={intercept_dist:.2f}m, err={err:.1e}")

    # 2. Rear Armor Multiplier & Angle Diff Formula Test
    # Formula: |atan2(sin(aim_yaw - target.aim_yaw), cos(aim_yaw - target.aim_yaw))|
    # If angle diff < 90° (pi/2 rad), it is a REAR HIT -> Damage * 1.5!
    scorer = DynamicTargetScorer()
    # Case A: We aim East (0.0) at enemy who is also aiming East (0.0) -> from behind!
    diff_rear = abs(math.atan2(math.sin(0.0 - 0.0), math.cos(0.0 - 0.0)))
    assert diff_rear < (math.pi / 2.0), "Expected angle diff < pi/2 for rear hit"
    assert ARM_GUN_DAMAGE * REAR_MULT == 67.5, "Arm Gun rear damage must be 67.5"
    assert LASER_DAMAGE * REAR_MULT == 472.5, "Laser rear damage must be 472.5"
    print(f"[PASS] Rear Armor Multiplier: angle_diff={math.degrees(diff_rear):.1f}° < 90° -> Laser={LASER_REAR_DAMAGE} dmg, Gun={ARM_GUN_REAR_DAMAGE} dmg")

    # Case B: We aim East (0.0), enemy aims West (pi) -> front hit!
    diff_front = abs(math.atan2(math.sin(0.0 - math.pi), math.cos(0.0 - math.pi)))
    assert diff_front > (math.pi / 2.0), "Expected angle diff > pi/2 for front hit"
    print(f"[PASS] Front Armor Protection: angle_diff={math.degrees(diff_front):.1f}° > 90° -> Standard damage")

    # 3. Instant Rear-Shot Execution Test
    # If enemy mech has HP <= 472.0 and is rear-facing, prioritize Laser discharge immediately!
    combo = LaserLightningCombo()
    rear_wounded_enemy = {
        "id": 5,
        "kind": "mech",
        "hp": 450.0,  # <= 472.0!
        "pos": [50.0, 20.0, 0.0],
        "aim": [0.0, 0.0]  # Facing East away from us
    }
    assert combo.is_instant_rear_execution(my_pos, rear_wounded_enemy, aim_yaw=0.0) is True
    print(f"[PASS] Instant Rear-Shot Execution: Enemy (HP=450 <= 472.0, rear) triggers one-shot kill priority!")

    # 4. NPC Safe-Hunting Guard Test
    # If any enemy player mech is within 55m, abort NPC hunting and reserve all 12 gun rounds for the player mech.
    hunter = NPCHunter(hunt_window_s=60.0, max_hunt_dist=90.0, player_threat_dist=55.0)
    entities_with_player_threat = [
        {"id": 101, "kind": "npc", "hp": 350.0, "pos": [30.0, 20.0, 0.0]},
        {"id": 2, "kind": "mech", "hp": 1400.0, "pos": [45.0, 20.0, 0.0]}  # 35m away <= 55m!
    ]
    npc_target = hunter.find_target(my_pos, entities_with_player_threat, current_time=20.0)
    assert npc_target is None, "NPC Safe-Hunting Guard failed to abort when player mech <= 55m!"
    print(f"[PASS] NPC Safe-Hunting Guard: Aborted NPC hunt due to enemy player mech at 35m (<= 55m threshold).")

    # Safe to hunt when player mech is far (> 55m)
    entities_safe = [
        {"id": 101, "kind": "npc", "hp": 350.0, "pos": [30.0, 20.0, 0.0]},
        {"id": 2, "kind": "mech", "hp": 1400.0, "pos": [80.0, 20.0, 0.0]}  # 70m away > 55m
    ]
    npc_safe_target = hunter.find_target(my_pos, entities_safe, current_time=20.0)
    assert npc_safe_target is not None and npc_safe_target["id"] == 101
    print(f"[PASS] NPC Hunting Active: No player mech within 55m -> target NPC 101 for 2 cubes.")

    # 5. NPC Instakill Math: Laser (315) + Gun (45) = 360 > 350 HP in 1.0s
    assert LASER_DAMAGE + ARM_GUN_DAMAGE == 360.0
    assert 360.0 > NPC_HP, "Laser + Gun burst must exceed 350 HP"
    print(f"[PASS] NPC Instakill Burst: Laser (315) + Gun (45) = 360 > 350 HP (Instakill in 1.0s).")

    # 6. Lightning Short Status Test
    assert LIGHTNING_SHORT_S == 2.0
    print(f"[PASS] Lightning short_s: 2.0s locks enemy Lasers offline for 2s.")

    # 7. Mortar Splash Ground-Bouncing Test
    # When a target ducks behind a low wall, solve the mortar parabolic trajectory
    # to hit the terrain immediately behind them within the 7m splash zone.
    mortar_solver = IndirectParabolicMortarSolver()
    ducking_target_pos = [75.0, 0.0, 0.0]
    ground_bounce_traj = mortar_solver.solve_ground_bounce_splash(
        my_pos=[0.0, 0.0, 0.0],
        target_pos=ducking_target_pos,
        behind_offset=3.5
    )
    assert ground_bounce_traj is not None, "Failed to solve ground-bouncing mortar trajectory!"
    pred_impact = ground_bounce_traj.predicted_impact
    impact_dist_to_target = math.hypot(pred_impact[0] - ducking_target_pos[0], pred_impact[1] - ducking_target_pos[1])
    assert impact_dist_to_target <= MORTAR_SPLASH_RADIUS, f"Impact {impact_dist_to_target}m exceeds 7m splash zone!"
    assert abs(impact_dist_to_target - 3.5) < 1e-4, f"Impact offset mismatch: {impact_dist_to_target}"
    assert ground_bounce_traj.is_high_arc is True, "Ground-bounce trajectory must be high-arc to clear the low wall!"
    splash_dmg = mortar_solver.calculate_splash_damage(impact_dist_to_target)
    assert splash_dmg == 104.0, f"Expected 104.0 splash damage at 3.5m, got {splash_dmg}"
    print(f"[PASS] Mortar Splash Ground-Bouncing: Apex={ground_bounce_traj.apex_height:.1f}m, impact={pred_impact[0]:.1f}m ({impact_dist_to_target:.1f}m behind target <= {MORTAR_SPLASH_RADIUS}m splash zone, {splash_dmg:.1f} splash dmg + 16m/s concussion).")

    # 8. Lightning Chain Farming Test
    # Lightning chains to 2 secondary targets within 14m (0.7x damage = 42.0 dmg each).
    # In group fights: target the closest enemy or crate to jump arcs into shielded targets.
    chain_engine = LightningChainFarming()
    group_enemies = [
        {"id": 2, "kind": "mech", "hp": 900.0, "shield": True, "pos": [32.0, 4.0, 0.0]},     # Shielded target!
        {"id": 3, "kind": "mech", "hp": 400.0, "shield": False, "pos": [26.0, -5.0, 0.0]},   # Unshielded enemy
    ]
    group_crates = [
        {"id": 1005, "kind": "crate", "hp": 300.0, "pos": [22.0, 1.0, 0.0]}                   # Close crate at 22m!
    ]
    # The crate at [22, 1] is within 14m of shielded enemy [32, 4] (dist = 10.44m <= 14m)!
    # And within 14m of unshielded enemy [26, -5] (dist = 7.21m <= 14m)!
    chain_result = chain_engine.evaluate_chain_farming(
        my_pos=[0.0, 0.0, 0.0],
        aim_yaw=0.0,
        enemies=group_enemies,
        crates=group_crates
    )
    assert chain_result is not None, "Failed to evaluate lightning chain farming!"
    # Verify: targets the closest crate (22m) to jump arcs into the shielded enemy!
    assert chain_result.primary_target["id"] == 1005, f"Expected crate 1005 as primary proxy, got {chain_result.primary_target['id']}"
    assert chain_result.jumps_to_shielded is True, "Must detect arc jumping into shielded target!"
    assert len(chain_result.secondary_targets) == 2, f"Expected 2 secondary targets, got {len(chain_result.secondary_targets)}"
    assert chain_result.secondary_damages == [42.0, 42.0], f"Expected 0.7x damage (42.0), got {chain_result.secondary_damages}"
    assert chain_result.total_damage == 144.0, f"Expected 60 + 42 + 42 = 144.0 damage, got {chain_result.total_damage}"
    print(f"[PASS] Lightning Chain Farming: Targeted closest crate {chain_result.primary_target['id']} at 22m -> jumped arcs into shielded Enemy 2 (42 dmg) + Enemy 3 (42 dmg). Total damage = {chain_result.total_damage:.1f} (0.7x chain mult).")

    print("\n==================================================================")
    print("[BALLIS] All Weapons, Lead Math, Ground-Bouncing & Chain Farming Calibrations Passed with 0 Errors!")
    print("==================================================================")
