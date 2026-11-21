"""
Ballistics, Targeting, and Weapon Controllers for MechRoyale.
Handles lead prediction, laser charge/release cycle, lightning cone, and mortar trajectory.
"""
import math
from typing import Dict, List, Optional, Tuple, Any

class Ballistics:
    def __init__(self, config: Dict[str, Any]):
        self.weapons_cfg = config.get("weapons", {})
        self.mech_cfg = config.get("mech", {})
        
        self.gun_speed = self.weapons_cfg.get("gun", {}).get("speed", 252.0)
        self.mortar_speed = self.weapons_cfg.get("mortar", {}).get("speed", 72.0)
        self.mortar_gravity = self.weapons_cfg.get("mortar", {}).get("gravity", 22.0)
        self.laser_charge_time = self.weapons_cfg.get("laser", {}).get("charge", 0.75)
        self.min_pitch = self.mech_cfg.get("min_pitch", -0.7)
        self.max_pitch = self.mech_cfg.get("max_pitch", 1.3)
        self.muzzle_height = self.mech_cfg.get("muzzle_height", 7.0)

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
        sx, sy, sz = my_pos[0], my_pos[1], my_pos[2] + self.muzzle_height
        tx, ty, tz = target_pos[0], target_pos[1], target_pos[2] + 4.5
        dx = tx - sx
        dy = ty - sy
        dz = tz - sz
        yaw = math.atan2(dy, dx)
        pitch = math.atan2(dz, max(math.hypot(dx, dy), 0.01))
        return yaw, max(self.min_pitch, min(self.max_pitch, pitch))

    def compute_mortar_aim(self, my_pos: List[float], target_pos: List[float]) -> Optional[Tuple[float, float]]:
        """Solves ballistic arc for mortar: range R = (v^2 / g) * sin(2*theta)."""
        sx, sy, sz = my_pos[0], my_pos[1], my_pos[2] + self.muzzle_height
        tx, ty, tz = target_pos[0], target_pos[1], target_pos[2]
        dx = tx - sx
        dy = ty - sy
        r = math.hypot(dx, dy)
        v = self.mortar_speed
        g = self.mortar_gravity

        val = (g * r) / (v * v)
        if val > 1.0:  # Out of range (> ~235m)
            return None

        # Low trajectory solution
        theta = 0.5 * math.asin(val)
        yaw = math.atan2(dy, dx)
        pitch = max(self.min_pitch, min(self.max_pitch, theta))
        return yaw, pitch

    def decide_weapon_triggers(
        self,
        obs_weapons: List[Optional[Dict[str, Any]]],
        target: Optional[Dict[str, Any]],
        target_dist: float,
        has_los: bool
    ) -> Tuple[List[bool], Tuple[float, float]]:
        """
        Determines [fire_slot_0, fire_slot_1] and commanded aim [yaw, pitch].
        Follows Claude's weapon triangle and trigger management doctrines.
        """
        fire = [False, False]
        default_aim = [0.0, 0.0]

        if not target or not has_los:
            # If laser was charging, release it so charge isn't wasted
            self.laser_charging = False
            self.laser_charge_ticks = 0
            return fire, default_aim

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
                    # Weapon in cooldown
                    fire[slot_idx] = False
                    self.laser_charging = False
                    self.laser_charge_ticks = 0
                elif curr_charge >= self.laser_charge_time:
                    # FULLY CHARGED! Release trigger to discharge 315-504 damage beam!
                    fire[slot_idx] = False
                    self.laser_charging = False
                    self.laser_charge_ticks = 0
                else:
                    # Keep charging
                    fire[slot_idx] = True
                    self.laser_charging = True
                    self.laser_charge_ticks += 1

            elif kind == "lightning":
                # Cone arc 45m. Rapid fire every 0.25s (costs 3 energy)
                if target_dist <= 45.0 and cooldown <= 0.05:
                    fire[slot_idx] = True

            elif kind == "gun":
                # Arm gun: fire when cooldown allows and mag has ammo
                ammo = w.get("ammo", 0.0)
                reload_time = w.get("reload", 0.0)
                if ammo > 0 and reload_time == 0.0 and cooldown <= 0.05:
                    fire[slot_idx] = True

            elif kind == "mortar":
                # Fire if in range (30m to 200m) and cooldown ready
                if 25.0 <= target_dist <= 220.0 and cooldown <= 0.05:
                    fire[slot_idx] = True

        return fire
