"""
Disciplined Opportunist: Tournament-winning MechRoyale Bot.
Implements the high-level brain, out-of-combat 45 HP/s resets, rear-armor protection,
ballistic lead prediction, and supply-drop ambushes.
"""
import atexit
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple, Any

import mechfight as mf
from nav import NavMesh
from ballistics import Ballistics
from combat import CombatSystem
from acoustics import AcousticRadar

class Bot:
    name = "opportunist-v1"

    def __init__(self, hello: mf.Hello):
        self.config = hello.get("config", {})
        map_dir = hello.get("map", {}).get("dir", ".")
        
        # Core engines
        self.nav = NavMesh(map_dir)
        self.ballistics = Ballistics(self.config)
        self.combat = CombatSystem(self.config)
        self.radar = AcousticRadar()

        # Tactical memory
        self.seen_enemies: Dict[int, Dict[str, Any]] = {}
        self.last_damaged_time = -100.0
        self.last_attack_time = -100.0
        self.current_waypoints: List[Tuple[float, float]] = []
        self.active_state = "SPAWN"

        # Complete upgrade priority ranking (all 22 upgrades from config)
        self.upgrade_ranks = {
            "nanites": 100,             # +60% out-of-combat regen, starts 1.5s sooner (powers combat resets!)
            "armor": 92,                # -8% damage taken (multiplicative effective HP)
            "plating": 88,              # +15% max HP
            "servos": 85,               # +8% speed (boosts transverse strafe dodging vs naive bots)
            "gun_mag": 80,              # +8 rounds, -20% reload time (less reload vulnerability)
            "gun_rate": 76,             # +15% arm gun fire rate
            "gun_damage": 72,           # +15% arm gun damage
            "gun_velocity": 70,         # +25% arm gun projectile speed (even easier to lead)
            "laser_charge": 68,         # -25% laser charge time
            "laser_damage": 65,         # +18% laser damage
            "laser_pierce": 63,         # laser passes through one more target
            "lightning_range": 60,      # +20% lightning range and cone
            "lightning_efficiency": 58, # -35% lightning energy cost, +10% damage
            "lightning_chain": 55,      # +1 lightning chain jump
            "reactor": 50,              # +30% energy regen
            "sensors": 45,              # +15m proximity sensors, +25m vision
            "jets": 40,                 # -25% boost & jet drain
            "capacitor": 35,            # +30 max energy
            "gyros": 30,                # +30% aim turn rate
            "mortar_reload": 20,        # -18% mortar cooldown
            "mortar_velocity": 15,      # +20% shell speed
            "mortar_splash": 10         # +30% mortar splash radius
        }

        # Match Telemetry Tracker
        self.telemetry = {
            "ticks": 0,
            "damage_dealt": 0.0,
            "damage_taken": 0.0,
            "kills": 0,
            "cubes": 0,
            "drops_captured": 0,
            "upgrades": [],
            "last_place": 16
        }
        atexit.register(self.log_telemetry)

    def log_telemetry(self):
        try:
            summary = json.dumps(self.telemetry)
            sys.stderr.write(f"[MATCH_TELEMETRY] {summary}\n")
            sys.stderr.flush()
            with open("/tmp/match_telemetry.jsonl", "a") as f:
                f.write(summary + "\n")
        except Exception:
            pass

    def _update_memory(self, obs: mf.Observation):
        t = obs["t"]
        self.telemetry["ticks"] += 1
        self.telemetry["cubes"] = obs["self"].get("cubes", 0)
        self.telemetry["last_place"] = obs.get("alive", 16)

        # Track combat events
        for ev in obs.get("events", []):
            ev_type = ev.get("type")
            if ev_type == "damaged":
                self.last_damaged_time = t
                self.telemetry["damage_taken"] += ev.get("amount", 0)
            elif ev_type == "hit":
                self.last_attack_time = t
                self.telemetry["damage_dealt"] += ev.get("amount", 0)
            elif ev_type == "kill":
                self.telemetry["kills"] += 1
            elif ev_type == "drop":
                self.telemetry["drops_captured"] += 1
            elif ev_type == "upgrade":
                self.telemetry["upgrades"].append(ev.get("id"))

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
        self.radar.update(obs.get("sounds", []), [x, y], t)
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
        next_radius = zone.get("next_radius", zone.get("radius", 380))
        dist_to_next_center = math.hypot(x - zone_center[0], y - zone_center[1])
        shrink_start = zone.get("shrink_start", 60.0)

        # Default movement & aim targets
        nav_target: Tuple[float, float] = (zone_center[0], zone_center[1])
        torso_aim: Tuple[float, float] = (0.0, 0.0)
        should_boost = False
        should_fire = [False, False]

        # Only rotate for the zone when actually outside, or when shrink is imminent (< 15s) and we are far
        is_outside = not me.get("in_zone", True)
        shrink_imminent = t >= (shrink_start - 15.0) and dist_to_next_center > (next_radius - 25.0)
        needs_zone_rotation = is_outside or shrink_imminent

        # --- Priority 1: Zone Safety ---
        if needs_zone_rotation:
            self.active_state = "ZONE_ROTATION"
            nav_target = (zone_center[0], zone_center[1])
            should_boost = is_outside

        # --- Priority 2: Combat Reset / Cover Retreat ---
        elif closest_enemy and my_hp < 500:
            should_engage, _ = self.combat.evaluate_fight(my_hp, me["weapons"], me.get("cubes", 0), closest_enemy)
            if not should_engage:
                self.active_state = "COMBAT_RESET"
                cover = self.nav.find_cover_spot((x, y), (closest_enemy["pos"][0], closest_enemy["pos"][1]))
                if cover:
                    nav_target = cover
                should_boost = enemy_dist < 30.0

        # --- Priority 3: Supply Drop Snatch / Ambush ---
        elif obs.get("drops") and t > 65.0:
            drop = obs["drops"][0]
            d_pos = drop["pos"]
            d_dist = math.hypot(d_pos[0] - x, d_pos[1] - y)
            d_lands_in = drop.get("lands_in", 0.0)
            d_holder = drop.get("holder")

            if d_dist < 250.0:
                if d_holder is None and (not visible_enemies or enemy_dist > 50.0):
                    self.active_state = "DROP_CAPTURE"
                    nav_target = (d_pos[0], d_pos[1])
                    should_boost = d_dist > 20.0 and d_lands_in <= 5.0
                else:
                    self.active_state = "DROP_AMBUSH"
                    ambush_pos = (d_pos[0] - 50.0, d_pos[1] - 20.0)
                    nav_target = ambush_pos

        # --- Priority 4: Tactical Combat (Always defend or press advantage) ---
        elif closest_enemy:
            should_engage, _ = self.combat.evaluate_fight(my_hp, me["weapons"], me.get("cubes", 0), closest_enemy)
            if should_engage or enemy_dist < 45.0:
                self.active_state = "ENGAGE"
                # Use transverse combat strafing: orbits at 50-80m to dodge unled naive bot shots
                combat_move = self.combat.compute_combat_move([x, y], closest_enemy["pos"], tick)
                nav_target = (x + combat_move[0] * 20.0, y + combat_move[1] * 20.0)
            else:
                self.active_state = "STALKING"

        # --- Priority 5: Safe Farm (Shoot crates & harvest cubes) ---
        else:
            self.active_state = "SAFE_FARM"
            death_pos = self.radar.find_recent_death_event()
            if death_pos and math.hypot(death_pos[0] - x, death_pos[1] - y) < 120.0:
                self.active_state = "THIRD_PARTY_AMBUSH"
                nav_target = (death_pos[0], death_pos[1])
            else:
                crates = obs.get("crates", [])
                if crates:
                    crates.sort(key=lambda c: math.hypot(c["pos"][0] - x, c["pos"][1] - y))
                    target_crate = crates[0]
                    nav_target = (target_crate["pos"][0], target_crate["pos"][1])
                    crate_dist = math.hypot(target_crate["pos"][0] - x, target_crate["pos"][1] - y)
                    if crate_dist < 40.0 and self.nav.has_los(x, y, target_crate["pos"][0], target_crate["pos"][1]):
                        cdx = target_crate["pos"][0] - x
                        cdy = target_crate["pos"][1] - y
                        cdz = (target_crate.get("pos", [0, 0, 0])[2] + 1.2) - (z + 7.0)
                        torso_aim = (math.atan2(cdy, cdx), math.atan2(cdz, max(crate_dist, 0.01)))
                        w0 = me["weapons"][0]
                        if w0 and w0.get("ammo", 0) > 0 and w0.get("reload", 0) == 0:
                            should_fire[0] = True

        # 5. Weapon Aiming & Firing Execution
        has_enemy_los = False
        if closest_enemy:
            has_enemy_los = self.nav.has_los(x, y, closest_enemy["pos"][0], closest_enemy["pos"][1])
            if has_enemy_los:
                yaw, pitch, _ = self.ballistics.compute_lead_aim(
                    [x, y, z],
                    closest_enemy["pos"],
                    closest_enemy.get("vel", [0, 0, 0]),
                    self.ballistics.gun_speed
                )
                torso_aim = (yaw, pitch)
                
                should_fire = self.ballistics.decide_weapon_triggers(
                    me["weapons"],
                    closest_enemy,
                    enemy_dist,
                    has_enemy_los
                )

        # If in combat reset, aim at enemy to protect rear armor, and ONLY cease fire once LOS is broken!
        if self.active_state == "COMBAT_RESET" and closest_enemy:
            dx = closest_enemy["pos"][0] - x
            dy = closest_enemy["pos"][1] - y
            torso_aim = (math.atan2(dy, dx), 0.0)
            if not has_enemy_los:
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

        # Steer away from active sound clusters during exploration
        if self.active_state in ("SAFE_FARM", "ZONE_ROTATION") and not closest_enemy:
            rep_x, rep_y = self.radar.get_danger_vector([x, y])
            if (rep_x, rep_y) != (0.0, 0.0):
                move_vec = [move_vec[0] * 0.75 + rep_x * 0.25, move_vec[1] * 0.75 + rep_y * 0.25]
                mlen = math.hypot(move_vec[0], move_vec[1])
                if mlen > 0.01:
                    move_vec = [move_vec[0] / mlen, move_vec[1] / mlen]

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
