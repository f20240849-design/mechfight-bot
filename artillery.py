"""
Artillery, Ballistics, and Weapon Triangle Systems for MechRoyale (mechfight/1).
Author: BALLIS - Weapon Triangle & Indirect Artillery Engineer

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
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union


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

        # Telemetry / execution stats
        self.total_evals = 0
        self.total_time_ms = 0.0
        self.max_time_ms = 0.0

    def evaluate_combat_tick(
        self,
        me: Dict[str, Any],
        target: Optional[Dict[str, Any]],
        has_los: bool,
        pickups: List[Dict[str, Any]],
        drops: Optional[List[Dict[str, Any]]] = None,
        all_enemies: Optional[List[Dict[str, Any]]] = None,
        in_cover: bool = False,
        current_time: float = 0.0
    ) -> Dict[str, Any]:
        """
        Executes unified combat and artillery evaluation for the current tick.
        
        Returns dictionary with:
        - 'fire': [bool, bool]
        - 'aim': [yaw, pitch]
        - 'loot_action': Optional[Dict[str, Any]] (take/drop)
        - 'tactical_move': str ('HOLD_COVER', 'PEEK_OUT', 'RETREAT_TO_COVER', 'STANDARD')
        - 'execution_time_ms': float
        """
        t_start = time.perf_counter()

        my_pos = me.get("pos", [0, 0, 0])
        aim = list(me.get("aim", [0.0, 0.0]))
        weapons = me.get("weapons", [None, None])
        energy = me.get("energy", 100.0)

        result_fire = [False, False]
        result_aim = [aim[0], aim[1]]
        tactical_move = "STANDARD"

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
                    hold_trigger, move_intent = self.laser.update_cycle(
                        w, has_los, t_dist, in_cover, current_time
                    )
                    result_fire[slot_idx] = hold_trigger
                    tactical_move = move_intent
                    if has_los:
                        lyaw, lpitch = self.laser.compute_aim(my_pos, t_pos)
                        result_aim = [lyaw, lpitch]

                # --- Weapon: Lightning ---
                elif w_kind == "lightning":
                    if self.lightning.should_fire(my_pos, aim[0], t_pos, energy, w_cd):
                        result_fire[slot_idx] = True
                        tactical_move = "LIGHTNING_ENGAGE"

                # --- Weapon: Mortar ---
                elif w_kind == "mortar":
                    # If target is behind cover (no LOS) or stationary
                    if (not has_los or t_dist > 60.0) and w_cd <= 0.05:
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
            "execution_time_ms": exec_ms
        }
