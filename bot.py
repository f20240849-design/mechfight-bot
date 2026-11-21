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
from artillery import ArtillerySystem
from gatekeeper import ZoneGatekeeper, Chokepoint
from rooftops import RooftopManager, RooftopEscapeTarget, is_on_rooftop

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
        self.artillery = ArtillerySystem(self.config)
        self.gatekeeper = ZoneGatekeeper(map_dir)
        self.rooftops = RooftopManager(map_dir)

        # Tactical memory
        self.seen_enemies: Dict[int, Dict[str, Any]] = {}
        self.last_damaged_time = -100.0
        self.last_attack_time = -100.0
        self.current_waypoints: List[Tuple[float, float]] = []
        self.cached_nav_target: Optional[Tuple[float, float]] = None
        self.active_state = "SPAWN"
        self.gatekeep_aim_yaw = 0.0
        self.active_escape: Optional[RooftopEscapeTarget] = None

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

    def _handle_loot_swaps(self, me: Dict[str, Any], pickups: List[Dict[str, Any]], enemies: Optional[List[Dict[str, Any]]] = None, t: float = 0.0) -> Optional[Dict[str, Any]]:
        """Delegates to SmartWeaponSwapManager from artillery system."""
        return self.artillery.swap_manager.evaluate_swap(me, pickups, enemies, t)

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
        visible_enemies = [m for m in obs.get("mechs", []) if m.get("kind") == "mech"]
        visible_enemies.sort(key=lambda m: math.hypot(m["pos"][0] - x, m["pos"][1] - y))

        loot_action = self._handle_loot_swaps(me, obs.get("pickups", []), visible_enemies, t)
        if loot_action:
            action.update(loot_action)

        # 3. Assess threats & combat status
        time_since_combat = t - max(self.last_damaged_time, self.last_attack_time)
        in_combat_reset = time_since_combat >= 5.0  # 45 HP/s active healing!

        closest_enemy = visible_enemies[0] if visible_enemies else None
        enemy_dist = math.hypot(closest_enemy["pos"][0] - x, closest_enemy["pos"][1] - y) if closest_enemy else 1e9

        # 4. High-Level Decision Engine (Disciplined Opportunist & Zone Gatekeeper)
        zone = obs["zone"]
        zone_center = zone.get("next_center", zone.get("center", [0, 0]))
        next_radius = zone.get("next_radius", zone.get("radius", 380))
        dist_to_next_center = math.hypot(x - zone_center[0], y - zone_center[1])
        shrink_start = zone.get("shrink_start", 60.0)
        alive_count = obs.get("alive", 16)

        # Default movement & aim targets
        nav_target: Tuple[float, float] = (zone_center[0], zone_center[1])
        torso_aim: Tuple[float, float] = (0.0, 0.0)
        should_boost = False
        should_fire = [False, False]

        # 0. Check Zone Urgency & Gas Status
        is_outside = not me.get("in_zone", True)
        needs_zone_rotation, eta_seconds = self.gatekeeper.evaluate_rotation_urgency(
            (x, y), zone, my_speed=11.7, current_time=t
        )
        shrink_imminent = (eta_seconds > max(1.0, shrink_start - t)) or (shrink_start - t < 15.0 and dist_to_next_center > next_radius - 20.0)

        # 0. Check Rooftop Verticality Status
        current_roof = self.rooftops.is_on_rooftop([x, y, z])
        energy = float(me.get("energy", 100.0))

        # --- Priority 0: Active Rooftop Control & Safe Descent ---
        if current_roof:
            should_descend, descend_reason = self.rooftops.should_descend(
                [x, y, z], zone, my_hp, me.get("max_hp", 1400.0), time_since_combat,
                is_outside=is_outside, shrink_imminent=shrink_imminent
            )
            # If outside zone or needs urgent rotation, FORCE immediate safe descent!
            if should_descend or is_outside or needs_zone_rotation:
                # Initiate safe zero-fall-damage descent into ground cover toward zone center
                exit_info = self.rooftops.get_safe_exit_vector(
                    current_roof, (zone_center[0], zone_center[1]), closest_enemy["pos"] if closest_enemy else None
                )
                if exit_info:
                    self.active_state = "ROOFTOP_EXIT"
                    nav_target = (exit_info["ground_pos"][0], exit_info["ground_pos"][1])
                    should_boost = is_outside or eta_seconds > max(1.0, shrink_start - t)
            else:
                # Hold safe 3m setback perch point: out-of-combat 45 HP/s active healing!
                self.active_state = "ROOFTOP_PERCH"
                perch_pt = self.rooftops.get_safe_perch_point(
                    current_roof, closest_enemy["pos"] if closest_enemy else None
                )
                nav_target = (perch_pt[0], perch_pt[1])
                # Cease fire to allow full HP regeneration
                should_fire = [False, False]

        # --- Priority 1: Endgame Survival Engine (< 5 alive) ---
        elif self.gatekeeper.is_endgame(alive_count, next_radius):
            self.active_state = "ENDGAME_SURVIVAL"
            endgame_pos = self.gatekeeper.get_endgame_position((x, y), zone, alive_count)
            nav_target = endgame_pos

            # If enemy visible, check whether to avoid optional fight
            if closest_enemy:
                avoid_fight = self.gatekeeper.should_avoid_combat(alive_count, my_hp, closest_enemy, enemy_dist)
                if avoid_fight:
                    # Break LOS and stay in deep cover; do not shoot unless directly attacked
                    cover = self.nav.find_cover_spot((x, y), (closest_enemy["pos"][0], closest_enemy["pos"][1]))
                    if cover:
                        nav_target = cover
                    should_boost = enemy_dist < 30.0

        # --- Priority 2: Zone Safety & Gatekeeper Ambush Doctrine ---
        elif needs_zone_rotation:
            chokes = self.gatekeeper.find_chokepoints(
                zone.get("radius", 560.0),
                (zone_center[0], zone_center[1]),
                next_radius
            )
            gatekeep_target = self.gatekeeper.select_gatekeep_position((x, y), chokes)

            # Check for Ohio Stadium pinch
            is_pinched, bypass_wp = self.gatekeeper.is_stadium_pinch((x, y), gatekeep_target)
            if is_pinched and bypass_wp:
                nav_target = bypass_wp
            else:
                nav_target = gatekeep_target

            dist_to_gatekeep = math.hypot(gatekeep_target[0] - x, gatekeep_target[1] - y)
            is_in_safe_zone = dist_to_next_center < (next_radius - 12.0)

            # If arrived early at the gatekeep ambush spot inside the safe zone:
            if dist_to_gatekeep < 6.0 and is_in_safe_zone and not is_outside:
                self.active_state = "GATEKEEP_AMBUSH"
                if chokes:
                    holding_yaw, holding_pitch = self.gatekeeper.get_ambush_aim((x, y), chokes[0])
                    self.gatekeep_aim_yaw = holding_yaw
                    torso_aim = (holding_yaw, holding_pitch)
            else:
                self.active_state = "ZONE_ROTATION"
                should_boost = is_outside or eta_seconds > max(1.0, shrink_start - t)

        # --- Priority 2: Cover Retreat / Combat Reset ---
        # Retreat to cover if:
        # 1. Under 800 HP
        # 2. Or fight evaluation is unfavorable (ratio < 0.85)
        # 3. Or enemy holds Laser while we only have starter gun and are in open
        # 4. Or enemy is currently charging a Laser!
        elif closest_enemy and (
            my_hp < 800
            or not self.combat.evaluate_fight(my_hp, me["weapons"], me.get("cubes", 0), closest_enemy)[0]
            or ("laser" in closest_enemy.get("weapons", []) and me["weapons"][1] is None and enemy_dist > 45.0)
            or closest_enemy.get("charging", False)
        ):
            # Check for immediate jump-jet rooftop escape: ground bots cannot follow!
            roof_escape = self.rooftops.find_nearest_rooftop_escape([x, y, z], closest_enemy["pos"], energy)
            if roof_escape and roof_escape.dist_to_approach < 55.0:
                self.active_state = "ROOFTOP_ESCAPE"
                climb_act = self.rooftops.compute_climb_action([x, y, z], roof_escape, me.get("grounded", True), energy)
                if climb_act.get("jump"):
                    action["jump"] = True
                should_boost = climb_act.get("boost", False)
                if climb_act["stage"] == "APPROACHING_WALL":
                    nav_target = (roof_escape.ground_approach[0], roof_escape.ground_approach[1])
                elif climb_act["stage"] in ("JUMPING", "JETTING_UP"):
                    nav_target = (roof_escape.climb_edge[0], roof_escape.climb_edge[1])
                else:
                    nav_target = (roof_escape.perch_point[0], roof_escape.perch_point[1])
                if climb_act["stage"] == "PERCHED_REGEN":
                    should_fire = [False, False]
            else:
                self.active_state = "COMBAT_RESET"
                cover = self.nav.find_cover_spot((x, y), (closest_enemy["pos"][0], closest_enemy["pos"][1]))
                if cover:
                    nav_target = cover
                else:
                    kmove = self.combat.compute_kiting_move([x, y], closest_enemy["pos"], None, tick)
                    nav_target = (x + kmove[0] * 25.0, y + kmove[1] * 25.0)
                should_boost = enemy_dist < 40.0 or closest_enemy.get("charging", False)

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

        # --- Priority 4: Tactical Combat (Engage with Transverse Strafing) ---
        elif closest_enemy:
            self.active_state = "ENGAGE"
            combat_move = self.combat.compute_combat_move([x, y], closest_enemy["pos"], tick)
            nav_target = (x + combat_move[0] * 20.0, y + combat_move[1] * 20.0)

        # --- Priority 5: Safe Farm (Harvest Loose Cubes, Floor Weapons, Crates) ---
        else:
            self.active_state = "SAFE_FARM"
            death_pos = self.radar.find_recent_death_event()
            if death_pos and math.hypot(death_pos[0] - x, death_pos[1] - y) < 120.0:
                self.active_state = "THIRD_PARTY_AMBUSH"
                nav_target = (death_pos[0], death_pos[1])
            else:
                cubes = obs.get("cubes", [])
                pickups = [p for p in obs.get("pickups", []) if p.get("kind") in ("laser", "lightning", "mortar")]
                crates = obs.get("crates", [])

                # 1. Loose power cubes (free +4% HP & dmg, 0 ammo needed)
                if cubes:
                    cubes.sort(key=lambda c: math.hypot(c["pos"][0] - x, c["pos"][1] - y))
                    nav_target = (cubes[0]["pos"][0], cubes[0]["pos"][1])
                    should_boost = math.hypot(cubes[0]["pos"][0] - x, cubes[0]["pos"][1] - y) > 20.0
                # 2. Floor weapons (Laser / Lightning / Mortar)
                elif pickups and (me["weapons"][1] is None or me["weapons"][0]["kind"] == "gun"):
                    pickups.sort(key=lambda p: math.hypot(p["pos"][0] - x, p["pos"][1] - y))
                    nav_target = (pickups[0]["pos"][0], pickups[0]["pos"][1])
                    should_boost = math.hypot(pickups[0]["pos"][0] - x, pickups[0]["pos"][1] - y) > 25.0
                # 3. Unopened crates
                elif crates:
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
                w1 = me["weapons"][1]
                w0 = me["weapons"][0]
                holds_laser = (w1 and w1.get("kind") == "laser") or (w0 and w0.get("kind") == "laser")
                if holds_laser:
                    yaw, pitch = self.ballistics.compute_laser_aim([x, y, z], closest_enemy["pos"])
                else:
                    yaw, pitch, _ = self.ballistics.compute_lead_aim(
                        [x, y, z],
                        closest_enemy["pos"],
                        closest_enemy.get("vel", [0, 0, 0]),
                        self.ballistics.gun_speed
                    )
                torso_aim = (yaw, pitch)

        # Unified Artillery & Weapon evaluation (Laser peek, Lightning cone, Mortar siege, Gun burst)
        art_res = self.artillery.evaluate_combat_tick(
            me,
            closest_enemy,
            has_enemy_los,
            obs.get("pickups", []),
            obs.get("drops"),
            visible_enemies,
            not has_enemy_los,
            t
        )
        should_fire = art_res.get("fire", [False, False])
        if art_res.get("tactical_move") == "SIEGE_DROP":
            torso_aim = (art_res["aim"][0], art_res["aim"][1])
        elif art_res.get("aim") and closest_enemy and (me["weapons"][1] and me["weapons"][1].get("kind") in ("laser", "mortar")):
            torso_aim = (art_res["aim"][0], art_res["aim"][1])

        # If in combat reset or rooftop escape/perch, aim at enemy to protect rear armor, and cease fire once LOS is broken!
        if self.active_state in ("COMBAT_RESET", "ROOFTOP_ESCAPE", "ROOFTOP_PERCH") and closest_enemy:
            dx = closest_enemy["pos"][0] - x
            dy = closest_enemy["pos"][1] - y
            torso_aim = (math.atan2(dy, dx), 0.0)
            if not has_enemy_los and not (me["weapons"][1] and me["weapons"][1].get("kind") == "mortar"):
                should_fire = [False, False]

        # 6. Motion Planning (Waypoints via A* with caching)
        target_dist_from_cached = (
            math.hypot(nav_target[0] - self.cached_nav_target[0], nav_target[1] - self.cached_nav_target[1])
            if hasattr(self, "cached_nav_target") and self.cached_nav_target
            else 999.0
        )
        # Advance current waypoints if we are close to the next one
        while self.current_waypoints and math.hypot(self.current_waypoints[0][0] - x, self.current_waypoints[0][1] - y) < 4.0:
            self.current_waypoints.pop(0)

        # Re-plan if target changed significantly (>6m), waypoints empty, or lost line-of-sight to next waypoint
        needs_replan = (
            not self.current_waypoints
            or target_dist_from_cached > 6.0
            or not self.nav.has_los(x, y, self.current_waypoints[0][0], self.current_waypoints[0][1])
        )

        if needs_replan:
            self.current_waypoints = self.nav.plan_path((x, y), nav_target)
            self.cached_nav_target = nav_target

        if self.current_waypoints:
            wp = self.current_waypoints[0]
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

        # If in GATEKEEP_AMBUSH, hold aim down the alley; otherwise look in direction of movement
        if self.active_state == "GATEKEEP_AMBUSH" and (not closest_enemy or not has_enemy_los):
            torso_aim = (self.gatekeep_aim_yaw, 0.0)
        elif (not closest_enemy or not has_enemy_los) and not (should_fire[0] or should_fire[1]):
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
