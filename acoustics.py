"""
Acoustic Intelligence & Conflict Heatmap for MechRoyale.
Tracks sound bearings, clusters active gunfights, and steers macro-rotations.
"""
import math
from typing import Dict, List, Optional, Tuple, Any

class AcousticRadar:
    def __init__(self):
        # List of sound events: {x, y, kind, time, weight}
        self.sound_events: List[Dict[str, Any]] = []
        self.dist_estimates = {
            "near": 40.0,
            "mid": 110.0,
            "far": 250.0
        }

    def update(self, sounds: List[Dict[str, Any]], my_pos: List[float], current_time: float):
        """Ingests new heard sounds and prunes expired sounds (> 5.0s old)."""
        x, y = my_pos[0], my_pos[1]

        # Add new sounds
        for s in sounds:
            kind = s.get("kind", "")
            bearing = s.get("bearing", 0.0)
            dist_cat = s.get("dist", "mid")
            age = s.get("age", 0.0)
            
            est_dist = self.dist_estimates.get(dist_cat, 110.0)
            sx = x + est_dist * math.cos(bearing)
            sy = y + est_dist * math.sin(bearing)

            # Weighting: death and explosions indicate high value / high lethality
            weight = 2.0 if kind in ("death", "explosion") else 1.0

            self.sound_events.append({
                "x": sx,
                "y": sy,
                "kind": kind,
                "time": current_time - age,
                "weight": weight
            })

        # Prune events older than 5.0 seconds
        self.sound_events = [ev for ev in self.sound_events if current_time - ev["time"] <= 5.0]

    def get_danger_vector(self, my_pos: List[float]) -> Tuple[float, float]:
        """
        Computes a repulsive 2D vector pointing AWAY from nearby sound clusters.
        Magnitude scales with sound proximity and frequency.
        """
        x, y = my_pos[0], my_pos[1]
        repulse_x = 0.0
        repulse_y = 0.0

        for ev in self.sound_events:
            dx = x - ev["x"]
            dy = y - ev["y"]
            dist = math.hypot(dx, dy)
            if 1.0 < dist < 150.0:
                # Inverse distance weighting
                force = ev["weight"] / (dist * dist)
                repulse_x += (dx / dist) * force
                repulse_y += (dy / dist) * force

        rlen = math.hypot(repulse_x, repulse_y)
        if rlen > 0.001:
            return repulse_x / rlen, repulse_y / rlen
        return 0.0, 0.0

    def find_recent_death_event(self) -> Optional[Tuple[float, float]]:
        """Returns the location of the most recent death sound (fresh wounded target opportunity)."""
        for ev in reversed(self.sound_events):
            if ev["kind"] == "death":
                return ev["x"], ev["y"]
        return None
