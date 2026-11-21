"""
Disciplined Opportunist: Tournament-winning MechRoyale Bot.
Implements the high-level brain, out-of-combat 45 HP/s resets, rear-armor protection,
ballistic lead prediction, and supply-drop ambushes.
"""
import math
import os
import sys
from typing import Dict, List, Optional, Tuple, Any

import mechfight as mf
from nav import NavMesh
from ballistics import Ballistics
from combat import CombatSystem

class Bot:
    name = "opportunist-v1"

    def __init__(self, hello: mf.Hello):
        self.config = hello.get("config", {})
        map_dir = hello.get("map", {}).get("dir", ".")
        
        # Core engines
        self.nav = NavMesh(map_dir)
        self.ballistics = Ballistics(self.config)
        self.combat = CombatSystem(self.config)

        # Tactical memory
        self.seen_enemies: Dict[int, Dict[str, Any]] = {}
        self.last_damaged_time = -100.0
        self.last_attack_time = -100.0
        self.current_waypoints: List[Tuple[float, float]] = []
        self.active_state = "SPAWN"

        # Upgrade priority ranking
        self.upgrade_ranks = {
            "nanites": 100,       # +60% out-of-combat regen, starts at 3.5s!
            "plating": 90,        # +15% max HP
            "armor": 85,          # -8% damage taken
            "servos": 80,         # +8% speed (outrunning rushers)
            "reactor": 75,        # +30% energy regen
            "laser_charge": 70,   # -25% laser charge time
            "laser_damage": 68,   # +18% laser damage
            "gun_rate": 65,       # +15% gun fire rate
            "gun_damage": 60,     # +15% gun damage
            "lightning_range": 55,# +20% lightning range
            "jets": 50            # -25% jet drain
        }

    def _update_memory(self, obs: mf.Observation):
        t = obs["t"]
        # Track combat events
        for ev in obs.get("events", []):
            ev_type = ev.get("type")
            if ev_type == "damaged":
                self.last_damaged_time = t
            elif ev_type == "hit":
                self.last_attack_time = t

        # Track enemies (persisting for 5s)
        for m in obs.get("mechs", []):
            if m.get("kind") == "mech":
                self.seen_enemies[int(m["id"])] = {
                    "data": m,
                    "seen_at": t
                }

        # Expire stale enemies (> 5.0s)
        expired = [eid for eid, info in self.seen_enemies.items() if t - info["seen_at"] > 5.0]
        for eid in expired:
            del self.seen_enemies[eid]

    def _choose_upgrade(self, offer: mf.Offer) -> int:
        best_idx = 0
        best_score = -1
        for idx, choice in enumerate(offer.get("choices", [])):
            cid = choice.get("id", "")
            score = self.upgrade_ranks.get(cid, 10)
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def _handle_loot_swaps(self, me: Dict[str, Any], pickups: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Evaluates ground loot within reach and picks up the best weapons."""
        pickup_radius = self.config.get("loot", {}).get("pickup_radius", 4.0)
        x, y, _ = me["pos"]
        w0 = me["weapons"][0]
        w1 = me["weapons"][1]

        # Find pickups in reach
        near_pickups = [
            p for p in pickups
            if math.hypot(p["pos"][0] - x, p["pos"][1] - y) <= pickup_radius
        ]
        if not near_pickups:
            return None

        # Priority: Laser > Lightning > Mortar
        loot_rank = {"laser": 3, "lightning": 2, "mortar": 1, "gun": 0}
        near_pickups.sort(key=lambda p: loot_rank.get(p.get("kind", ""), 0), reverse=True)
        best_pickup = near_pickups[0]
        p_kind = best_pickup.get("kind")
        p_id = best_pickup.get("id")

        held_kinds = [w["kind"] for w in (w0, w1) if w is not None]

        # Case 1: Empty slot available
        if w1 is None and p_kind not in held_kinds:
            return {"take": p_id}

        # Case 2: Both full, but we hold Gun and want Laser/Lightning
        if w0 and w0["kind"] == "gun" and p_kind in ("laser", "lightning") and p_kind not in held_kinds:
            return {"take": p_id, "drop": 0}

        return None

    def act(self, obs: mf.Observation) -> mf.Action:
        me = obs["self"]
        x, y, z = me["pos"]
        my_hp = me["hp"]
        tick = int(obs["tick"])
        t = obs["t"]

        self._update_memory(obs)
        action: mf.Action = {"tick": tick}

        # 1. Handle upgrade offer
        if obs.get("offer"):
            action["pick"] = self._choose_upgrade(obs["offer"])

        # 2. Handle weapon pickups
        loot_action = self._handle_loot_swaps(me, obs.get("pickups", []))
        if loot_action:
            action.update(loot_action)

        # 3. Assess threats & combat status
        time_since_combat = t - max(self.last_damaged_time, self.last_attack_time)
        in_combat_reset = time_since_combat >= 5.0  # 45 HP/s active healing!

        visible_enemies = [m for m in obs.get("mechs", []) if m.get("kind") == "mech"]
        visible_enemies.sort(key=lambda m: math.hypot(m["pos"][0] - x, m["pos"][1] - y))

        closest_enemy = visible_enemies[0] if visible_enemies else None
        enemy_dist = math.hypot(closest_enemy["pos"][0] - x, closest_enemy["pos"][1] - y) if closest_enemy else 1e9

        # 4. High-Level Decision Engine (Disciplined Opportunist)
        zone = obs["zone"]
        zone_center = zone.get("next_center", zone.get("center", [0, 0]))
        zone_radius = zone.get("next_radius", zone.get("radius", 500))
        dist_to_zone_center = math.hypot(x - zone_center[0], y - zone_center[1])

        # Default movement & aim targets
        nav_target: Tuple[float, float] = (zone_center[0], zone_center[1])
        torso_aim: Tuple[float, float] = (0.0, 0.0)
        should_boost = False
        should_fire = [False, False]

        # --- Priority 1: Zone Safety ---
        # Outside zone or in danger of collapse
        if not me.get("in_zone", True) or dist_to_zone_center > zone_radius - 20:
            self.active_state = "ZONE_ROTATION"
            nav_target = (zone_center[0], zone_center[1])
            should_boost = not me.get("in_zone", True)

        # --- Priority 2: Combat Reset / Cover Retreat ---
        elif closest_enemy and (my_hp < 600 or (not in_combat_reset and my_hp < 900)):
            should_engage, _ = self.combat.evaluate_fight(my_hp, me["weapons"], me.get("cubes", 0), closest_enemy)
            if not should_engage:
                self.active_state = "COMBAT_RESET"
                # Find cover spot breaking LOS to enemy
                cover = self.nav.find_cover_spot((x, y), (closest_enemy["pos"][0], closest_enemy["pos"][1]))
                if cover:
                    nav_target = cover
                should_boost = enemy_dist < 40.0

        # --- Priority 3: Supply Drop Snatch / Ambush ---
        elif obs.get("drops") and t > 65.0:
            drop = obs["drops"][0]
            d_pos = drop["pos"]
            d_dist = math.hypot(d_pos[0] - x, d_pos[1] - y)
            d_lands_in = drop.get("lands_in", 0.0)
            d_holder = drop.get("holder")

            if d_lands_in <= 10.0 and d_dist < 100.0:
                if d_holder is None and (not visible_enemies or enemy_dist > 60.0):
                    # Uncontested! Rush and capture for instant full 1400 HP + 3 cubes!
                    self.active_state = "DROP_CAPTURE"
                    nav_target = (d_pos[0], d_pos[1])
                    should_boost = d_dist > 15.0
                else:
                    # Ambush from 50-70m away with cover
                    self.active_state = "DROP_AMBUSH"
                    ambush_pos = (d_pos[0] - 50.0, d_pos[1] - 20.0)
                    nav_target = ambush_pos

        # --- Priority 4: Third-Party / Tactical Combat ---
        elif closest_enemy:
            should_engage, _ = self.combat.evaluate_fight(my_hp, me["weapons"], me.get("cubes", 0), closest_enemy)
            if should_engage:
                self.active_state = "ENGAGE"
                # Keep optimal 50m distance
                if enemy_dist < 40.0:
                    kmove = self.combat.compute_kiting_move([x, y], closest_enemy["pos"], nav_target, tick)
                    nav_target = (x + kmove[0] * 20.0, y + kmove[1] * 20.0)
                else:
                    nav_target = (closest_enemy["pos"][0], closest_enemy["pos"][1])
            else:
                self.active_state = "STALKING"

        # --- Priority 5: Safe Farm (Crates / NPC Drones) ---
        else:
            self.active_state = "SAFE_FARM"
            crates = obs.get("crates", [])
            if crates:
                crates.sort(key=lambda c: math.hypot(c["pos"][0] - x, c["pos"][1] - y))
                nav_target = (crates[0]["pos"][0], crates[0]["pos"][1])

        # 5. Weapon Aiming & Firing Execution
        has_enemy_los = False
        if closest_enemy:
            has_enemy_los = self.nav.has_los(x, y, closest_enemy["pos"][0], closest_enemy["pos"][1])
            if has_enemy_los:
                # Solve predictive lead aim
                yaw, pitch, _ = self.ballistics.compute_lead_aim(
                    [x, y, z],
                    closest_enemy["pos"],
                    closest_enemy.get("vel", [0, 0, 0]),
                    self.ballistics.gun_speed
                )
                torso_aim = (yaw, pitch)
                
                # Check weapon firing triggers
                should_fire = self.ballistics.decide_weapon_triggers(
                    me["weapons"],
                    closest_enemy,
                    enemy_dist,
                    has_enemy_los
                )

        # If retreating, aim toward enemy while moving away (rear armor protection)
        if self.active_state == "COMBAT_RESET" and closest_enemy:
            dx = closest_enemy["pos"][0] - x
            dy = closest_enemy["pos"][1] - y
            torso_aim = (math.atan2(dy, dx), 0.0)
            # Stop firing so 45 HP/s out-of-combat heal activates!
            should_fire = [False, False]

        # 6. Motion Planning (Waypoints via A*)
        path = self.nav.plan_path((x, y), nav_target)
        if path:
            wp = path[0]
            # Advance waypoint if close
            if len(path) > 1 and math.hypot(wp[0] - x, wp[1] - y) < 4.0:
                wp = path[1]
            move_yaw = math.atan2(wp[1] - y, wp[0] - x)
            move_vec = [math.cos(move_yaw), math.sin(move_yaw)]
        else:
            move_yaw = math.atan2(nav_target[1] - y, nav_target[0] - x)
            move_vec = [math.cos(move_yaw), math.sin(move_yaw)]

        # If not actively aiming at enemy, look in direction of movement
        if not closest_enemy or not has_enemy_los:
            torso_aim = (move_yaw, 0.0)

        action.update(
            move=move_vec,
            aim=[torso_aim[0], torso_aim[1]],
            boost=should_boost,
            fire=should_fire
        )

        # Debug Visualizations
        mf.draw.label(f"{self.active_state} | HP {my_hp:.0f} | {'REGEN' if in_combat_reset else 'COMBAT'}")
        if closest_enemy and has_enemy_los:
            mf.draw.line([x, y, z + 7], [closest_enemy["pos"][0], closest_enemy["pos"][1], closest_enemy["pos"][2] + 4.5], "#ff3333")
        mf.draw.point([nav_target[0], nav_target[1], z], "#33ff88", 6)

        return action

if __name__ == "__main__":
    mf.run(Bot)
