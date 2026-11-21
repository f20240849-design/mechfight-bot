"""
Zone Gatekeeper & Endgame Macro Strategist for MechRoyale (mechfight/1).
Author: CHOKE - Zone Gatekeeper & Macro Strategist

Responsibilities:
1. Chokepoint & Funnel Precomputation:
   - Identifies building alleys and narrow paths along zone shrink perimeters.
   - Ambush positions to intercept panic-rotating bots scrambling inward.
2. Rotation Urgency & Stadium Pinch Avoidance:
   - Calculates exact ETA vs shrink timer (20 -> 40 -> 80 -> 260 DPS zone).
   - Prevents getting trapped in Ohio Stadium concourse/horseshoe.
3. Endgame Survival Engine (< 5 alive):
   - Shifts objective to pure HP preservation.
   - Avoids optional fights; allows survivors to whittle each other down.
"""

import math
import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

@dataclass
class Chokepoint:
    id: int
    pos: Tuple[float, float]
    width: float
    normal: Tuple[float, float]
    incoming_dir: Tuple[float, float]

class ZoneGatekeeper:
    def __init__(self, map_dir: str = "arena"):
        self.map_dir = map_dir
        
        # Ohio Stadium boundary box (approximate center and extents in campus coords)
        # Stadium prism coordinates from world.json
        self.stadium_center = (-260.0, 40.0)
        self.stadium_radius = 90.0

        # Pre-calculated primary campus building funnels / alleys
        self.default_chokes = [
            Chokepoint(1, (120.0, -80.0), 12.0, (-0.8, 0.6), (0.8, -0.6)),   # Engineering Quad Alley
            Chokepoint(2, (-40.0, -150.0), 14.0, (0.0, 1.0), (0.0, -1.0)),   # South Campus Walkway
            Chokepoint(3, (-120.0, 140.0), 15.0, (0.7, -0.7), (-0.7, 0.7)),  # North River Gate
            Chokepoint(4, (180.0, 60.0), 10.0, (-1.0, 0.0), (1.0, 0.0)),     # East Campus Gate
            Chokepoint(5, (-50.0, 50.0), 16.0, (0.0, -1.0), (0.0, 1.0)),     # Central Oval Funnel
        ]

    def evaluate_rotation_urgency(
        self,
        my_pos: Tuple[float, float],
        zone: Dict[str, Any],
        my_speed: float = 11.7,
        current_time: float = 0.0
    ) -> Tuple[bool, float]:
        """
        Evaluates whether the bot must rotate toward safe zone now.
        Returns: (needs_rotation, eta_seconds)
        """
        next_center = zone.get("next_center", zone.get("center", [0.0, 0.0]))
        next_radius = zone.get("next_radius", zone.get("radius", 380.0))
        shrink_start = zone.get("shrink_start", 60.0)

        dist_to_center = math.hypot(my_pos[0] - next_center[0], my_pos[1] - next_center[1])
        dist_to_safe_border = max(0.0, dist_to_center - (next_radius - 20.0))
        
        eta_seconds = dist_to_safe_border / max(1.0, my_speed)
        time_until_shrink = max(0.0, shrink_start - current_time)

        # Urgent rotation required if outside, or if shrink starts before we can walk in with buffer
        is_outside = dist_to_center > next_radius
        shrink_imminent = time_until_shrink <= (eta_seconds + 12.0) and dist_to_safe_border > 10.0

        needs_rotation = is_outside or shrink_imminent
        return needs_rotation, eta_seconds

    def is_endgame(self, alive_count: int, next_radius: float) -> bool:
        """Endgame triggered when <= 4 mechs alive or zone is closing to final circles."""
        return alive_count <= 4 or next_radius <= 140.0

    def get_endgame_position(
        self,
        my_pos: Tuple[float, float],
        zone: Dict[str, Any],
        alive_count: int
    ) -> Tuple[float, float]:
        """
        Calculates safe endgame positioning:
        Avoids the exact center brawl point; hugs the inner edge with cover.
        """
        next_center = zone.get("next_center", zone.get("center", [0.0, 0.0]))
        next_radius = zone.get("next_radius", zone.get("radius", 100.0))

        # Position at ~40% radius from center in direction of our current approach
        dx = my_pos[0] - next_center[0]
        dy = my_pos[1] - next_center[1]
        dist = math.hypot(dx, dy)

        target_r = max(15.0, next_radius * 0.4)
        if dist > 0.01:
            target_x = next_center[0] + (dx / dist) * target_r
            target_y = next_center[1] + (dy / dist) * target_r
        else:
            target_x = next_center[0] + target_r
            target_y = next_center[1]

        return (target_x, target_y)

    def should_avoid_combat(
        self,
        alive_count: int,
        my_hp: float,
        closest_enemy: Dict[str, Any],
        enemy_dist: float
    ) -> bool:
        """
        In endgame (< 5 alive), HP preservation dictates score.
        Avoid fights unless enemy is extremely weak or pressing into point-blank.
        """
        enemy_hp = closest_enemy.get("hp", 1000.0)
        
        # Self defense: must fight if within 25m
        if enemy_dist < 25.0:
            return False

        # Finisher: punish low-health survivor (< 400 HP) if we are healthy (> 900 HP)
        if enemy_hp < 400.0 and my_hp > 900.0:
            return False

        # Otherwise, let other survivors fight each other!
        return True

    def find_chokepoints(
        self,
        zone_radius: float,
        zone_center: Tuple[float, float],
        next_radius: float
    ) -> List[Chokepoint]:
        """Returns chokepoints that fall along the transition between current and next zone."""
        valid_chokes = []
        for c in self.default_chokes:
            dist = math.hypot(c.pos[0] - zone_center[0], c.pos[1] - zone_center[1])
            if dist <= zone_radius and dist >= (next_radius - 40.0):
                valid_chokes.append(c)
        return valid_chokes if valid_chokes else self.default_chokes

    def select_gatekeep_position(
        self,
        my_pos: Tuple[float, float],
        chokes: List[Chokepoint]
    ) -> Tuple[float, float]:
        """
        Selects the best ambush spot behind the chokepoint funnel on the safe side.
        """
        if not chokes:
            return (0.0, 0.0)

        # Sort chokes by proximity to our mech
        sorted_chokes = sorted(
            chokes,
            key=lambda c: math.hypot(c.pos[0] - my_pos[0], c.pos[1] - my_pos[1])
        )
        best_choke = sorted_chokes[0]

        # Ambush position is 18m inside safe zone along the choke normal
        ambush_x = best_choke.pos[0] + best_choke.normal[0] * 18.0
        ambush_y = best_choke.pos[1] + best_choke.normal[1] * 18.0
        return (ambush_x, ambush_y)

    def is_stadium_pinch(
        self,
        my_pos: Tuple[float, float],
        target_pos: Tuple[float, float]
    ) -> Tuple[bool, Optional[Tuple[float, float]]]:
        """
        Detects if path traverses through Ohio Stadium walls and provides bypass waypoint.
        """
        # Distance of mid-point to stadium center
        mid_x = (my_pos[0] + target_pos[0]) * 0.5
        mid_y = (my_pos[1] + target_pos[1]) * 0.5
        d_stadium = math.hypot(mid_x - self.stadium_center[0], mid_y - self.stadium_center[1])

        if d_stadium < self.stadium_radius:
            # Route around stadium via east concourse bypass
            bypass_wp = (self.stadium_center[0] + 120.0, self.stadium_center[1])
            return True, bypass_wp
        return False, None

    def get_ambush_aim(
        self,
        my_pos: Tuple[float, float],
        choke: Chokepoint
    ) -> Tuple[float, float]:
        """Calculates aim yaw directly pointing down the chokepoint funnel."""
        dx = choke.pos[0] - my_pos[0]
        dy = choke.pos[1] - my_pos[1]
        yaw = math.atan2(dy, dx)
        return yaw, 0.0
