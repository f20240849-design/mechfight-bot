"""
Combat, Threat Assessment, and Movement Controllers for MechRoyale.
Implements the 'Disciplined Opportunist' fight-score engine, kiting, and unpredictable strafing.
"""
import math
import random
from typing import Dict, List, Optional, Tuple, Any

class CombatSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.weapons_cfg = config.get("weapons", {})
        self.mech_cfg = config.get("mech", {})
        
        # Strafe oscillation timer
        self.strafe_dir = 1.0
        self.last_strafe_tick = 0
        self.strafe_period = 5  # Switch every 5 decisions (0.5s)

    def estimate_dps(self, weapon_kind: str, my_cubes: float, is_enemy_holding_mortar: bool = False) -> float:
        cube_mult = 1.0 + 0.04 * min(my_cubes, 10.0)
        if weapon_kind == "gun":
            # 45 dmg * 12 shots / (12 * 0.24 + 2.4) ≈ 102 sustained DPS
            return 102.0 * cube_mult
        elif weapon_kind == "laser":
            # 315 dmg / (0.75 charge + 1.65 cd) ≈ 131 sustained DPS
            # Cookoff against mortar: +60% dmg (210 DPS!)
            mult = 1.6 if is_enemy_holding_mortar else 1.0
            return 131.0 * cube_mult * mult
        elif weapon_kind == "lightning":
            # 60 dmg / 0.25s cd = 240 burst DPS
            return 240.0 * cube_mult
        elif weapon_kind == "mortar":
            # 200 dmg / 2.6s cd ≈ 77 direct DPS + splash
            return 90.0 * cube_mult
        return 80.0

    def evaluate_fight(
        self,
        my_hp: float,
        my_weapons: List[Optional[Dict[str, Any]]],
        my_cubes: float,
        enemy: Dict[str, Any]
    ) -> Tuple[bool, float]:
        """
        Fight-Score Engine:
        Estimates TTK (time-to-kill) both ways.
        Returns: (should_engage, confidence_ratio)
        """
        enemy_hp = enemy.get("hp", 1000.0)
        enemy_weapons = enemy.get("weapons", ["gun"])
        enemy_cubes = enemy.get("cubes", 0.0)

        enemy_holds_mortar = "mortar" in enemy_weapons

        # Calculate my effective DPS
        my_dps = 0.0
        for w in my_weapons:
            if w:
                my_dps += self.estimate_dps(w.get("kind", "gun"), my_cubes, enemy_holds_mortar)
        my_dps = max(my_dps, 50.0)

        # Estimate enemy DPS
        enemy_dps = 0.0
        for ew in enemy_weapons:
            enemy_dps += self.estimate_dps(ew, enemy_cubes, False)
        enemy_dps = max(enemy_dps, 50.0)

        ttk_them = enemy_hp / my_dps
        ttk_me = my_hp / enemy_dps

        # Ratio of survival time vs kill time
        ratio = ttk_me / max(ttk_them, 0.01)

        # Engage if we have a viable fight (ratio >= 0.85)
        should_engage = ratio >= 0.85
        return should_engage, ratio

    def compute_kiting_move(
        self,
        my_pos: List[float],
        threat_pos: List[float],
        goal_pos: Optional[Tuple[float, float]],
        tick: int
    ) -> List[float]:
        """
        Decoupled Kiting:
        Moves away from threat (or toward safe goal) with unpredictable lateral strafing.
        """
        # Periodic strafe direction toggle (every 0.4 - 0.6s)
        if tick - self.last_strafe_tick > self.strafe_period:
            self.strafe_dir = -self.strafe_dir
            self.last_strafe_tick = tick

        dx = threat_pos[0] - my_pos[0]
        dy = threat_pos[1] - my_pos[1]
        dist = math.hypot(dx, dy)
        if dist < 0.01:
            return [1.0, 0.0]

        # Normal vector away from threat
        away_x = -dx / dist
        away_y = -dy / dist

        # Perpendicular vector for strafing
        perp_x = -away_y * self.strafe_dir
        perp_y = away_x * self.strafe_dir

        if goal_pos:
            # Blend retreat vector toward cover/zone goal
            gx = goal_pos[0] - my_pos[0]
            gy = goal_pos[1] - my_pos[1]
            gdist = math.hypot(gx, gy)
            if gdist > 1.0:
                gx /= gdist
                gy /= gdist
            else:
                gx, gy = 0.0, 0.0
            move_x = away_x * 0.4 + gx * 0.4 + perp_x * 0.2
            move_y = away_y * 0.4 + gy * 0.4 + perp_y * 0.2
        else:
            move_x = away_x * 0.7 + perp_x * 0.3
            move_y = away_y * 0.7 + perp_y * 0.3

        # Clamp length to 1.0
        mlen = math.hypot(move_x, move_y)
        if mlen > 1.0:
            move_x /= mlen
            move_y /= mlen

        return [move_x, move_y]

    def compute_combat_move(
        self,
        my_pos: List[float],
        enemy_pos: List[float],
        tick: int = 0
    ) -> List[float]:
        """
        Calculates transverse strafing against naive bots (who aim without lead).
        Maintains desired standoff distance (50m - 80m) while oscillating laterally.
        At 60m+, moving perpendicularly at 11.7 m/s displaces the bot by > 3.0m during
        bullet flight time, completely dodging direct-aimed unled gun shells.
        """
        if tick - self.last_strafe_tick > self.strafe_period:
            self.strafe_dir = -self.strafe_dir
            self.last_strafe_tick = tick

        dx = enemy_pos[0] - my_pos[0]
        dy = enemy_pos[1] - my_pos[1]
        dist = math.hypot(dx, dy)
        if dist < 0.01:
            return [1.0, 0.0]

        # Unit vector toward enemy
        to_x = dx / dist
        to_y = dy / dist

        # Perpendicular vector (tangential orbit)
        perp_x = -to_y * self.strafe_dir
        perp_y = to_x * self.strafe_dir

        if dist < 45.0:
            rad_weight = -0.7  # Backpedal away to maintain range
            perp_weight = 0.5
        elif dist > 85.0:
            rad_weight = 0.5   # Close in
            perp_weight = 0.6
        else:
            rad_weight = 0.0   # Pure perpendicular strafe!
            perp_weight = 1.0

        move_x = to_x * rad_weight + perp_x * perp_weight
        move_y = to_y * rad_weight + perp_y * perp_weight

        mlen = math.hypot(move_x, move_y)
        if mlen > 0.01:
            move_x /= mlen
            move_y /= mlen
        return [move_x, move_y]
