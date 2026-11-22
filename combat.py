"""
Combat, Threat Assessment, and Movement Controllers for MechRoyale.
Implements the 'Disciplined Opportunist' fight-score engine, kiting, wall-aware strafing,
weapon triangle dynamics, and peek-and-shoot cover management.
"""
import math
import random
from typing import Dict, List, Optional, Tuple, Any

class CombatSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.weapons_cfg = config.get("weapons", {})
        self.mech_cfg = config.get("mech", {})
        
        # Jittered strafe oscillation timer
        self.strafe_dir = 1.0
        self.last_strafe_tick = 0
        self.strafe_period = 4  # Initial period (4-7 ticks = 0.4s - 0.7s)

        # Tactical state for peek-and-shoot
        self.peek_phase = "SEEK_COVER"  # "SEEK_COVER", "PEEK", "DUCK"
        self.peek_timer = 0.0
        self.cover_spot: Optional[Tuple[float, float]] = None

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
            return 95.0 * cube_mult
        return 80.0

    def evaluate_fight(
        self,
        my_hp: float,
        my_weapons: List[Optional[Dict[str, Any]]],
        my_cubes: float,
        enemy: Dict[str, Any]
    ) -> Tuple[bool, float]:
        """
        Fight-Score Engine with Weapon Triangle Matrix:
        - Laser destroys Mortar (+60% cookoff bonus)
        - Lightning shuts down Laser (2s disable)
        - Mortar repels Lightning (concussion + energy drain)
        Returns: (should_engage, confidence_ratio)
        """
        enemy_hp = enemy.get("hp", 1000.0)
        enemy_weapons = enemy.get("weapons", ["gun"])
        enemy_cubes = enemy.get("cubes", 0.0)

        enemy_holds_mortar = "mortar" in enemy_weapons
        enemy_holds_laser = "laser" in enemy_weapons
        enemy_holds_lightning = "lightning" in enemy_weapons

        my_holds_mortar = any(w and w.get("kind") == "mortar" for w in my_weapons)
        my_holds_laser = any(w and w.get("kind") == "laser" for w in my_weapons)
        my_holds_lightning = any(w and w.get("kind") == "lightning" for w in my_weapons)

        # Calculate my effective DPS
        my_dps = 0.0
        for w in my_weapons:
            if w:
                k = w.get("kind", "gun")
                my_dps += self.estimate_dps(k, my_cubes, enemy_holds_mortar)
        my_dps = max(my_dps, 50.0)

        # Estimate enemy DPS against me
        enemy_dps = 0.0
        for ew in enemy_weapons:
            enemy_dps += self.estimate_dps(ew, enemy_cubes, my_holds_mortar)
        enemy_dps = max(enemy_dps, 50.0)

        # Tactical Triangle Adjustment
        if my_holds_lightning and enemy_holds_laser:
            # Lightning disables enemy laser for 2s every time it connects
            enemy_dps *= 0.65

        if enemy_holds_lightning and my_holds_laser:
            # Enemy lightning shuts down our laser
            my_dps *= 0.65

        ttk_them = enemy_hp / my_dps
        ttk_me = my_hp / enemy_dps

        # Ratio of survival time vs kill time
        ratio = ttk_me / max(ttk_them, 0.01)

        # Favorable fight condition:
        # If ratio >= 0.85, or if enemy is low HP (<400)
        should_engage = (ratio >= 0.85) or (enemy_hp < 400.0 and my_hp > 450.0)
        return should_engage, ratio

    def compute_kiting_move(
        self,
        my_pos: List[float],
        threat_pos: List[float],
        goal_pos: Optional[Tuple[float, float]],
        tick: int,
        nav: Optional[Any] = None
    ) -> List[float]:
        """
        Decoupled Kiting:
        Moves away from threat (or toward safe goal/cover) with unpredictable lateral strafing.
        """
        # Stochastic strafe toggle (every 3 to 6 ticks)
        if tick - self.last_strafe_tick > self.strafe_period:
            self.strafe_dir = -self.strafe_dir
            self.last_strafe_tick = tick
            self.strafe_period = random.randint(3, 6)

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

        # Wall-awareness check: if strafe direction hits a wall, flip strafe dir!
        if nav:
            test_x = my_pos[0] + perp_x * 4.0
            test_y = my_pos[1] + perp_y * 4.0
            if not nav.is_walkable(test_x, test_y):
                self.strafe_dir = -self.strafe_dir
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
            move_x = away_x * 0.45 + gx * 0.40 + perp_x * 0.15
            move_y = away_y * 0.45 + gy * 0.40 + perp_y * 0.15
        else:
            move_x = away_x * 0.75 + perp_x * 0.25
            move_y = away_y * 0.75 + perp_y * 0.25

        # Normalize
        mlen = math.hypot(move_x, move_y)
        if mlen > 0.01:
            move_x /= mlen
            move_y /= mlen

        return [move_x, move_y]

    def compute_combat_move(
        self,
        my_pos: List[float],
        enemy_pos: List[float],
        tick: int = 0,
        ideal_min_dist: float = 45.0,
        ideal_max_dist: float = 75.0,
        nav: Optional[Any] = None
    ) -> List[float]:
        """
        Calculates wall-aware transverse strafing with variable cadence.
        Maintains standoff distance while oscillating laterally.
        Checks nav walkability to prevent getting pinned against geometry.
        """
        if tick - self.last_strafe_tick > self.strafe_period:
            self.strafe_dir = -self.strafe_dir
            self.last_strafe_tick = tick
            self.strafe_period = random.randint(3, 6)

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

        # Wall avoidance check: test 3m ahead along strafe
        if nav:
            check_dist = 4.0
            if not nav.is_walkable(my_pos[0] + perp_x * check_dist, my_pos[1] + perp_y * check_dist):
                # Reverse strafe direction to stay in open space
                self.strafe_dir = -self.strafe_dir
                perp_x = -to_y * self.strafe_dir
                perp_y = to_x * self.strafe_dir

        if dist < ideal_min_dist:
            rad_weight = -0.65  # Backpedal
            perp_weight = 0.55
        elif dist > ideal_max_dist:
            rad_weight = 0.55   # Close in
            perp_weight = 0.65
        else:
            rad_weight = 0.0    # Pure perpendicular transverse strafe
            perp_weight = 1.0

        move_x = to_x * rad_weight + perp_x * perp_weight
        move_y = to_y * rad_weight + perp_y * perp_weight

        mlen = math.hypot(move_x, move_y)
        if mlen > 0.01:
            move_x /= mlen
            move_y /= mlen
        return [move_x, move_y]
