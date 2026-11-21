"""
Ballistics, Targeting, and Weapon Controllers for MechRoyale.
Handles lead prediction, laser charge/release cycle, lightning cone, and mortar trajectory.
Integrated with BALLIS artillery systems.
"""
import math
from typing import Dict, List, Optional, Tuple, Any

from artillery import (
    IndirectParabolicMortarSolver,
    HitscanLaserController,
    LightningConeTrigger,
    SmartWeaponSwapManager,
    ArtillerySystem,
    MortarTrajectory,
)

class Ballistics:
    def __init__(self, config: Dict[str, Any]):
        self.weapons_cfg = config.get("weapons", {})
        self.mech_cfg = config.get("mech", {})
        
        self.gun_speed = self.weapons_cfg.get("gun", {}).get("speed", 252.0)
        self.mortar_speed = self.weapons_cfg.get("mortar", {}).get("speed", 70.0)
        self.mortar_gravity = self.weapons_cfg.get("mortar", {}).get("gravity", 18.0)
        self.laser_charge_time = self.weapons_cfg.get("laser", {}).get("charge", 0.75)
        self.min_pitch = self.mech_cfg.get("min_pitch", -0.7)
        self.max_pitch = self.mech_cfg.get("max_pitch", 1.3)
        self.muzzle_height = self.mech_cfg.get("muzzle_height", 7.0)

        # High-arc artillery solver
        self.mortar_solver = IndirectParabolicMortarSolver(
            speed=self.mortar_speed,
            gravity=self.mortar_gravity,
            tube_height_offset=self.mech_cfg.get("height", 9.0) + 0.5,
            min_range=25.0
        )
        self.laser_controller = HitscanLaserController(
            charge_time=self.laser_charge_time,
            cooldown=self.weapons_cfg.get("laser", {}).get("cooldown", 1.65),
            range_m=self.weapons_cfg.get("laser", {}).get("range", 280.0),
            damage=self.weapons_cfg.get("laser", {}).get("damage", 315.0),
            muzzle_height=self.muzzle_height,
            min_pitch=self.min_pitch,
            max_pitch=self.max_pitch
        )
        self.lightning_trigger = LightningConeTrigger(
            max_range=self.weapons_cfg.get("lightning", {}).get("range", 45.0),
            trigger_range=40.0,
            cone_deg=self.weapons_cfg.get("lightning", {}).get("cone_deg", 30.0),
            damage_per_shot=self.weapons_cfg.get("lightning", {}).get("damage", 60.0),
            cooldown=self.weapons_cfg.get("lightning", {}).get("cooldown", 0.25),
            energy_cost=self.weapons_cfg.get("lightning", {}).get("energy", 3.0)
        )

        # Internal laser state tracking
        self.laser_charging = False
        self.laser_charge_ticks = 0

    def compute_lead_aim(
        self,
        my_pos: List[float],
        target_pos: List[float],
        target_vel: List[float],
        bullet_speed: float,
        target_height_offset: float = 4.5
    ) -> Tuple[float, float, float]:
        """
        Solves first-order predictive aim for a projectile with constant velocity.
        Returns: (yaw, pitch, flight_time)
        """
        sx, sy, sz = my_pos[0], my_pos[1], my_pos[2] + self.muzzle_height
        tx, ty, tz = target_pos[0], target_pos[1], target_pos[2] + target_height_offset
        vx, vy, vz = target_vel[0], target_vel[1], target_vel[2]

        dx = tx - sx
        dy = ty - sy
        dz = tz - sz
        dist = math.hypot(dx, dy, dz)

        t_flight = dist / max(bullet_speed, 1.0)
        # 1-iteration refinement
        pred_x = tx + vx * t_flight
        pred_y = ty + vy * t_flight
        pred_z = tz + vz * t_flight

        pdx = pred_x - sx
        pdy = pred_y - sy
        pdz = pred_z - sz

        horiz_dist = math.hypot(pdx, pdy)
        yaw = math.atan2(pdy, pdx)
        pitch = math.atan2(pdz, max(horiz_dist, 0.01))
        pitch = max(self.min_pitch, min(self.max_pitch, pitch))

        return yaw, pitch, t_flight

    def compute_laser_aim(self, my_pos: List[float], target_pos: List[float]) -> Tuple[float, float]:
        """Laser is hitscan: aim directly at target center of mass."""
        return self.laser_controller.compute_aim(my_pos, target_pos)

    def compute_mortar_aim(self, my_pos: List[float], target_pos: List[float]) -> Optional[Tuple[float, float]]:
        """Solves low-arc ballistic arc for mortar."""
        traj = self.mortar_solver.solve_low_arc(my_pos, target_pos)
        if traj:
            pitch = max(self.min_pitch, min(self.max_pitch, traj.pitch))
            return traj.yaw, pitch
        return None

    def compute_mortar_high_arc_aim(
        self,
        my_pos: List[float],
        target_pos: List[float],
        target_vel: Optional[List[float]] = None
    ) -> Optional[Tuple[float, float, float]]:
        """Solves indirect high-arc ballistic trajectory to lob shells over buildings."""
        traj = self.mortar_solver.solve_high_arc(my_pos, target_pos, target_vel)
        if traj:
            return traj.yaw, traj.pitch, traj.flight_time
        return None

    def decide_weapon_triggers(
        self,
        obs_weapons: List[Optional[Dict[str, Any]]],
        target: Optional[Dict[str, Any]],
        target_dist: float,
        has_los: bool,
        current_energy: float = 100.0,
        current_aim_yaw: float = 0.0
    ) -> List[bool]:
        """
        Determines [fire_slot_0, fire_slot_1] based on weapon triangle and ranges.
        """
        fire = [False, False]

        if not target and not has_los:
            self.laser_charging = False
            self.laser_charge_ticks = 0
            return fire

        t_pos = target.get("pos", [0, 0, 0]) if target else [0, 0, 0]

        # Iterate over our 2 weapon slots
        for slot_idx, w in enumerate(obs_weapons):
            if w is None:
                continue
            kind = w.get("kind")
            cooldown = w.get("cooldown", 0.0)

            if kind == "laser":
                # Laser charge cycle:
                # Hold trigger while charging; release trigger to fire!
                curr_charge = w.get("charge", 0.0)
                if cooldown > 0.0:
                    fire[slot_idx] = False
                    self.laser_charging = False
                    self.laser_charge_ticks = 0
                elif curr_charge >= self.laser_charge_time:
                    # FULLY CHARGED! Release trigger to discharge beam
                    fire[slot_idx] = False
                    self.laser_charging = False
                    self.laser_charge_ticks = 0
                elif has_los and target_dist <= 280.0:
                    # Keep charging
                    fire[slot_idx] = True
                    self.laser_charging = True
                    self.laser_charge_ticks += 1

            elif kind == "lightning":
                # 40m conservative trigger within 30° cone
                if target:
                    if self.lightning_trigger.should_fire([0, 0], current_aim_yaw, [target_dist, 0], current_energy, cooldown):
                        fire[slot_idx] = True

            elif kind == "gun":
                ammo = w.get("ammo", 0)
                reload_time = w.get("reload", 0.0)
                if has_los and ammo > 0 and reload_time == 0.0 and cooldown <= 0.05:
                    fire[slot_idx] = True

            elif kind == "mortar":
                # Fire indirect or direct if within 25m to 200m
                if 25.0 <= target_dist <= 220.0 and cooldown <= 0.05:
                    fire[slot_idx] = True

        return fire
