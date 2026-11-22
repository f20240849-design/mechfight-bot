"""
Disciplined Opportunist: Tournament-winning MechRoyale Bot.
Implements the high-level brain, evasion & projectile dodging, out-of-combat 45 HP/s resets,
decoupled +/-70 deg rear-armor protection, radar vision sweeping, ballistic lead prediction,
and ghost phase weapon rush.
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
from flank_orbit import FlankOrbitEngine
from evasion import EvasionSystem

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
        self.flank_orbit = FlankOrbitEngine(map_dir)
        self.evasion = EvasionSystem()

        # Tactical memory
        self.seen_enemies: Dict[int, Dict[str, Any]] = {}
        self.last_damaged_time = -100.0
        self.last_damaged_bearing: Optional[float] = None
        self.last_damaged_rear = False
        self.last_attack_time = -100.0
        self.current_waypoints: List[Tuple[float, float]] = []
        self.cached_nav_target: Optional[Tuple[float, float]] = None
        self.active_state = "SPAWN"
        self.gatekeep_aim_yaw = 0.0
        self.active_escape: Optional[RooftopEscapeTarget] = None

        # Vision Radar Sweep State (Decoupled 360-deg FOV Refresh)
        self.last_sweep_time = 0.0
        self.sweep_stage = 0  # 0: normal, 1: flick +120 deg, 2: flick -120 deg
        self.sweep_ticks = 0

        # Complete upgrade priority ranking (all 22 upgrades from config)
        self.upgrade_ranks = {
            "nanites": 100,             # +60% out-of-combat regen, starts 1.5s sooner (powers combat resets!)
            "armor": 95,                # -8% damage taken (multiplicative effective HP)
            "plating": 90,              # +15% max HP
            "servos": 85,               # +8% speed (boosts transverse strafe dodging vs naive bots)
            "gun_mag": 80,              # +8 rounds, -20% reload time (less reload vulnerability)
            "laser_damage": 78,         # +18% laser damage
            "laser_charge": 76,         # -25% laser charge time
            "gun_rate": 72,             # +15% arm gun fire rate
            "gun_damage": 70,           # +15% arm gun damage
            "laser_pierce": 66,         # laser passes through one more target
            "lightning_efficiency": 62, # -35% lightning energy cost, +10% damage
            "gun_velocity": 60,         # +25% arm gun projectile speed
            "lightning_range": 58,      # +20% lightning range and cone
            "lightning_chain": 54,      # +1 lightning chain jump
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
                if ev.get("bearing") is not None:
                    self.last_damaged_bearing = ev["bearing"]
                self.last_damaged_rear = ev.get("rear", False)
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

    def compute_defensive_facing(self, threat_bearing: float, move_yaw: float) -> float:
        """
        The Claude +/-70 deg Defensive Facing Rule:
        b = bearing to primary threat
        v = travel direction
        aim = b + clamp(wrap(v - b), -70 deg, +70 deg)
        Ensures threat is ALWAYS inside front 180 deg arc (at most 70 deg off center),
        guaranteeing ZERO rear-armor multiplier, while keeping travel path directly
        inside the 120 deg vision wedge!
        """
        diff = math.atan2(math.sin(move_yaw - threat_bearing), math.cos(move_yaw - threat_bearing))
        clamped = max(-1.2217, min(1.2217, diff))  # 70 degrees = 1.2217 rad
        return threat_bearing + clamped

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

        # 2. Evaluate incoming projectile & beam evasion
        evasion_res = self.evasion.evaluate(me, obs)
        is_threatened = evasion_res["threat_level"] > 0.20
        should_jump = evasion_res["should_jump"] and me.get("grounded", False)
        evasion_boost = evasion_res["should_boost"]

        # 3. Handle weapon pickups & inventory
        visible_enemies = [m for m in obs.get("mechs", []) if m.get("kind") == "mech"]
        visible_enemies.sort(key=lambda m: math.hypot(m["pos"][0] - x, m["pos"][1] - y))

        loot_action = self._handle_loot_swaps(me, obs.get("pickups", []), visible_enemies, t)
        if loot_action:
            action.update(loot_action)

        # 4. Assess threats & combat status
        time_since_combat = t - max(self.last_damaged_time, self.last_attack_time)
        in_combat_reset = time_since_combat >= 5.0  # 45 HP/s active healing!

        # Target selection: BALLIS Dynamic Target Scorer
        optimal_target = self.artillery.select_optimal_target([x, y, z], obs.get("mechs", []), t)
        closest_enemy = optimal_target if optimal_target else (visible_enemies[0] if visible_enemies else None)
        
        # LOS-prioritization guard: If current target has no LOS, check if another visible enemy player has LOS!
        if visible_enemies:
            los_enemies = [e for e in visible_enemies if self.nav.has_los(x, y, e["pos"][0], e["pos"][1])]
            if los_enemies:
                if not closest_enemy or not self.nav.has_los(x, y, closest_enemy["pos"][0], closest_enemy["pos"][1]):
                    closest_enemy = min(los_enemies, key=lambda e: (e["pos"][0] - x)**2 + (e["pos"][1] - y)**2)

        enemy_dist = math.hypot(closest_enemy["pos"][0] - x, closest_enemy["pos"][1] - y) if closest_enemy else 1e9

        # High-Level Decision Engine & Macro Navigation
        zone = obs["zone"]
        zone_center = zone.get("next_center", zone.get("center", [0, 0]))
        next_radius = zone.get("next_radius", zone.get("radius", 380))
        dist_to_next_center = math.hypot(x - zone_center[0], y - zone_center[1])
        shrink_start = zone.get("shrink_start", 60.0)
        alive_count = obs.get("alive", 16)

        # Default movement & aim targets
        nav_target: Tuple[float, float] = (zone_center[0], zone_center[1])
        torso_aim: Tuple[float, float] = (0.0, 0.0)
        should_boost = evasion_boost
        should_fire = [False, False]
        should_reload = False

        # Inventory check
        w1 = me["weapons"][1]
        w0 = me["weapons"][0]
        holds_laser = (w1 and w1.get("kind") == "laser") or (w0 and w0.get("kind") == "laser")
        holds_lightning = (w1 and w1.get("kind") == "lightning") or (w0 and w0.get("kind") == "lightning")
        holds_mortar = (w1 and w1.get("kind") == "mortar") or (w0 and w0.get("kind") == "mortar")
        holds_secondary = w1 is not None

        floor_weapons = [p for p in obs.get("pickups", []) if p.get("kind") in ("laser", "lightning", "mortar")]
        crates = obs.get("crates", [])
        cubes = obs.get("cubes", [])

        has_enemy_los = False
        enemy_is_charging = False
        enemy_holds_laser = False
        enemy_holds_mortar = False
        enemy_holds_lightning = False
        if closest_enemy:
            has_enemy_los = self.nav.has_los(x, y, closest_enemy["pos"][0], closest_enemy["pos"][1])
            enemy_is_charging = closest_enemy.get("charging", False)
            enemy_weapons = closest_enemy.get("weapons", [])
            enemy_holds_laser = "laser" in enemy_weapons
            enemy_holds_mortar = "mortar" in enemy_weapons
            enemy_holds_lightning = "lightning" in enemy_weapons

        # Zone urgency
        is_outside = not me.get("in_zone", True)
        needs_zone_rotation, eta_seconds = self.gatekeeper.evaluate_rotation_urgency(
            (x, y), zone, my_speed=11.7, current_time=t
        )
        shrink_imminent = (eta_seconds > max(1.0, shrink_start - t)) or (shrink_start - t < 15.0 and dist_to_next_center > next_radius - 20.0)

        # Rooftop status
        current_roof = self.rooftops.is_on_rooftop([x, y, z])
        energy = float(me.get("energy", 100.0))

        # Check for reload opportunity: if gun has < 5 ammo and no immediate close combat, reload now!
        if w0 and w0.get("ammo", 12) <= 4 and w0.get("reload", 0) == 0:
            if not closest_enemy or not has_enemy_los or enemy_dist > 35.0:
                should_reload = True

        # =========================================================================
        # UNIFIED STRATEGIC HIERARCHY (Strictly Mutually Exclusive Priorities)
        # =========================================================================

        # Priority 0: Zone Survival (Storm / Gas Escape)
        if is_outside or needs_zone_rotation or shrink_imminent:
            if current_roof:
                exit_info = self.rooftops.get_safe_exit_vector(
                    current_roof, (zone_center[0], zone_center[1]), closest_enemy["pos"] if closest_enemy else None
                )
                if exit_info:
                    self.active_state = "ROOFTOP_EXIT"
                    nav_target = (exit_info["ground_pos"][0], exit_info["ground_pos"][1])
                    should_boost = is_outside or eta_seconds > max(1.0, shrink_start - t)
            else:
                self.active_state = "ZONE_ROTATION"
                chokes = self.gatekeeper.find_chokepoints(zone.get("radius", 560.0), (zone_center[0], zone_center[1]), next_radius)
                gatekeep_target = self.gatekeeper.select_gatekeep_position((x, y), chokes)
                is_pinched, bypass_wp = self.gatekeeper.is_stadium_pinch((x, y), gatekeep_target)
                nav_target = bypass_wp if (is_pinched and bypass_wp) else gatekeep_target
                should_boost = is_outside or eta_seconds > max(1.0, shrink_start - t)

        # Priority 1: Critical Combat Reset / Cover Retreat
        # Triggered if:
        # a) Low HP (< 550)
        # b) Enemy is charging Laser with direct LOS and we don't have Laser
        # c) Enemy holds Laser at range (> 50m) while we only have Gun
        # d) Ghost Phase (t < 30s) encounter at > 60m without Laser: NEVER duel in the open!
        # e) TTK inequality test indicates negative expected value
        elif (closest_enemy and (
            my_hp < 550.0
            or (enemy_is_charging and has_enemy_los and not holds_laser)
            or (enemy_holds_laser and enemy_dist > 50.0 and not holds_laser and has_enemy_los)
            or (t < 30.0 and enemy_dist > 50.0 and not holds_laser)
        )) or (t - self.last_damaged_time < 1.5 and my_hp < 650.0):
            self.active_state = "COMBAT_RESET"
            threat_pos = (closest_enemy["pos"][0], closest_enemy["pos"][1]) if closest_enemy else (
                (x - math.cos(self.last_damaged_bearing or 0.0) * 50.0, y - math.sin(self.last_damaged_bearing or 0.0) * 50.0)
            )
            cover = self.nav.find_cover_spot((x, y), threat_pos)
            if cover:
                nav_target = cover
            else:
                kmove = self.combat.compute_kiting_move([x, y], list(threat_pos), None, tick, nav=self.nav)
                nav_target = (x + kmove[0] * 25.0, y + kmove[1] * 25.0)
            should_boost = True

        # Priority 2: Ghost Phase & Weapon Rush (Opening 0-30s or Empty Slot 1)
        # In early game, securing a Laser/Lightning weapon is the single highest expected value move
        # RULE: Walk at 11.7 m/s, NEVER boost to travel (avoids 90m acoustic alert)
        elif (t < 30.0 or not holds_secondary) and floor_weapons:
            # Prefer Laser > Lightning > Mortar
            def wp_priority(p):
                k = p.get("kind")
                d = math.hypot(p["pos"][0] - x, p["pos"][1] - y)
                tier = 0 if k == "laser" else (1 if k == "lightning" else 2)
                return (tier, d)

            best_wp = min(floor_weapons, key=wp_priority)
            wp_dist = math.hypot(best_wp["pos"][0] - x, best_wp["pos"][1] - y)
            if wp_dist < 160.0:
                self.active_state = "WEAPON_RUSH"
                nav_target = (best_wp["pos"][0], best_wp["pos"][1])
                # Walk at 11.7 m/s to stay quiet (footstep 40m vs boost 90m sound!)
                should_boost = False

        # Priority 3: Rooftop Perch Healing (if on roof and safe)
        elif current_roof:
            self.active_state = "ROOFTOP_PERCH"
            perch_pt = self.rooftops.get_safe_perch_point(current_roof, closest_enemy["pos"] if closest_enemy else None)
            nav_target = (perch_pt[0], perch_pt[1])
            should_fire = [False, False]

        # Priority 4: Endgame Survival Engine (< 5 alive)
        elif self.gatekeeper.is_endgame(alive_count, next_radius):
            self.active_state = "ENDGAME_SURVIVAL"
            endgame_pos = self.gatekeeper.get_endgame_position((x, y), zone, alive_count)
            nav_target = endgame_pos
            if closest_enemy and self.gatekeeper.should_avoid_combat(alive_count, my_hp, closest_enemy, enemy_dist):
                cover = self.nav.find_cover_spot((x, y), (closest_enemy["pos"][0], closest_enemy["pos"][1]))
                if cover:
                    nav_target = cover

        # Priority 5: Tactical Combat Engagement (within viable combat range)
        # Apply Claude TTK Disengage vs Commit Inequality:
        # Commit only if: e / D_us < 0.75 * h / D_them OR e <= 315 (One-shot laser kill!)
        elif closest_enemy and (
            (holds_laser and enemy_dist <= 160.0)
            or (holds_lightning and enemy_dist <= 45.0)
            or (enemy_dist <= 65.0 and not enemy_holds_laser)
            or (enemy_dist <= 35.0)
        ):
            # Calculate TTK
            d_us = 131.0 if holds_laser else (240.0 if holds_lightning else 102.0)
            if holds_laser and enemy_holds_mortar:
                d_us *= 1.6  # Cookoff
            d_them = 131.0 if enemy_holds_laser else (240.0 if enemy_holds_lightning else 102.0)
            enemy_e_hp = closest_enemy.get("hp", 1000.0)
            
            ttk_them = enemy_e_hp / max(d_us, 1.0)
            ttk_us = my_hp / max(d_them, 1.0)

            # One-shot rule: If we hold laser and e <= 315 with LOS, ALWAYS COMMIT!
            is_one_shot = holds_laser and enemy_e_hp <= 315.0 and has_enemy_los

            if is_one_shot or (ttk_them <= 0.75 * ttk_us) or (enemy_dist < 25.0):
                self.active_state = "ENGAGE"
                combat_move = self.combat.compute_combat_move([x, y], closest_enemy["pos"], tick, nav=self.nav)
                c_target = (x + combat_move[0] * 25.0, y + combat_move[1] * 25.0)
                if not self.nav.is_walkable(c_target[0], c_target[1]):
                    combat_move_alt = [-combat_move[1], combat_move[0]]
                    alt_target = (x + combat_move_alt[0] * 25.0, y + combat_move_alt[1] * 25.0)
                    if self.nav.is_walkable(alt_target[0], alt_target[1]):
                        c_target = alt_target
                nav_target = c_target
                should_boost = enemy_dist < 30.0 or holds_lightning
            else:
                # Disadvantageous duel: disengage into cover!
                self.active_state = "COMBAT_RESET"
                cover = self.nav.find_cover_spot((x, y), (closest_enemy["pos"][0], closest_enemy["pos"][1]))
                if cover:
                    nav_target = cover
                else:
                    kmove = self.combat.compute_kiting_move([x, y], closest_enemy["pos"], None, tick, nav=self.nav)
                    nav_target = (x + kmove[0] * 25.0, y + kmove[1] * 25.0)
                should_boost = True

        # Priority 6: Supply Drop Vulture Strategy
        # Drop 1 & 2 (t < 180s): Approach to 80-120m in cover; don't step into the ring killzone!
        # Drop 3 (t >= 180s): Aggressively contest or snatch
        elif obs.get("drops") and (
            drop_target := self.gatekeeper.evaluate_drop_objective((x, y), obs.get("drops", []), visible_enemies, t)
        ) is not None:
            if t < 180.0 and visible_enemies and math.hypot(drop_target[0] - x, drop_target[1] - y) < 40.0:
                # Vulture hold: observe from cover at 70m
                cover = self.nav.find_cover_spot((x, y), drop_target, min_dist=50.0, max_dist=80.0)
                nav_target = cover if cover else drop_target
                self.active_state = "DROP_VULTURE"
            else:
                self.active_state = "DROP_CAPTURE"
                nav_target = drop_target
                should_boost = math.hypot(drop_target[0] - x, drop_target[1] - y) > 15.0

        # Priority 7: Safe Farm & Flank Orbit (Loot, Crates, Perimeter)
        else:
            self.active_state = "SAFE_FARM"
            death_pos = self.radar.find_recent_death_event()

            # 1. Floor weapons if slot 1 is empty
            if not holds_secondary and floor_weapons:
                floor_weapons.sort(key=lambda p: math.hypot(p["pos"][0] - x, p["pos"][1] - y))
                nav_target = (floor_weapons[0]["pos"][0], floor_weapons[0]["pos"][1])
                should_boost = False
            # 2. Loose power cubes within 50m
            elif cubes and math.hypot(cubes[0]["pos"][0] - x, cubes[0]["pos"][1] - y) < 50.0:
                cubes.sort(key=lambda c: math.hypot(c["pos"][0] - x, c["pos"][1] - y))
                nav_target = (cubes[0]["pos"][0], cubes[0]["pos"][1])
                should_boost = False
            # 3. Third-party recent kill
            elif death_pos and math.hypot(death_pos[0] - x, death_pos[1] - y) < 80.0:
                self.active_state = "THIRD_PARTY_AMBUSH"
                nav_target = (death_pos[0], death_pos[1])
            # 4. Nearby crates within 35m
            elif crates and math.hypot(crates[0]["pos"][0] - x, crates[0]["pos"][1] - y) < 35.0:
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
            # 5. Otherwise, FLANK ORBIT along the 0.75*R perimeter in deep cover!
            else:
                self.active_state = "FLANK_ORBIT"
                flank_wp = self.flank_orbit.get_flank_waypoint((x, y), zone, obs.get("sounds", []), alive_count)
                nav_target = flank_wp

        # 5. Weapon Aiming & Firing Execution
        if closest_enemy:
            if holds_laser and has_enemy_los:
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
        art_fire = art_res.get("fire", [False, False])
        should_fire = [should_fire[0] or art_fire[0], should_fire[1] or art_fire[1]]
        if art_res.get("tactical_move") == "SIEGE_DROP":
            torso_aim = (art_res["aim"][0], art_res["aim"][1])
        elif art_res.get("aim") and closest_enemy:
            torso_aim = (art_res["aim"][0], art_res["aim"][1])

        # Gun trigger discipline: only fire at enemy if aim yaw is reasonably aligned (< 0.20 rad)!
        if closest_enemy:
            curr_yaw = me.get("aim", [0.0, 0.0])[0]
            yaw_err = abs(math.atan2(math.sin(torso_aim[0] - curr_yaw), math.cos(torso_aim[0] - curr_yaw)))
            if yaw_err > 0.20 and not (me["weapons"][1] and me["weapons"][1].get("kind") == "mortar"):
                should_fire[0] = False

        # 6. Motion Planning (Waypoints via A* with caching)
        target_dist_from_cached = (
            math.hypot(nav_target[0] - self.cached_nav_target[0], nav_target[1] - self.cached_nav_target[1])
            if hasattr(self, "cached_nav_target") and self.cached_nav_target
            else 999.0
        )
        while self.current_waypoints and math.hypot(self.current_waypoints[0][0] - x, self.current_waypoints[0][1] - y) < 4.0:
            self.current_waypoints.pop(0)

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

        # Apply ATLAS Wall-Slide Vector to glide smoothly around building corners & concave pockets
        if hasattr(self.nav, "compute_wall_slide_vector"):
            slide_vx, slide_vy = self.nav.compute_wall_slide_vector(x, y, move_vec[0], move_vec[1])
            move_vec = [slide_vx, slide_vy]

        # 7. Dynamic Evasion Vector Integration
        if is_threatened and evasion_res["dodge_vec"]:
            d_vec = evasion_res["dodge_vec"]
            if self.nav.is_walkable(x + d_vec[0] * 3.0, y + d_vec[1] * 3.0):
                move_vec = [d_vec[0] * 0.85 + move_vec[0] * 0.15, d_vec[1] * 0.85 + move_vec[1] * 0.15]
                mlen = math.hypot(move_vec[0], move_vec[1])
                if mlen > 0.01:
                    move_vec = [move_vec[0] / mlen, move_vec[1] / mlen]
                should_boost = True

        # Steer away from active sound clusters during exploration
        if self.active_state in ("SAFE_FARM", "ZONE_ROTATION") and not closest_enemy and not is_threatened:
            rep_x, rep_y = self.radar.get_danger_vector([x, y])
            if (rep_x, rep_y) != (0.0, 0.0):
                move_vec = [move_vec[0] * 0.75 + rep_x * 0.25, move_vec[1] * 0.75 + rep_y * 0.25]
                mlen = math.hypot(move_vec[0], move_vec[1])
                if mlen > 0.01:
                    move_vec = [move_vec[0] / mlen, move_vec[1] / mlen]

        # =========================================================================
        # 8. Decoupled Facing & Vision Sweep Policy (Sections 2 & 6)
        # =========================================================================
        # If in retreat or kiting, execute the Claude +/-70 deg defensive facing rule:
        # Protects front-arc 180 deg armor while keeping travel path in 120 deg vision wedge!
        if self.active_state in ("COMBAT_RESET", "ROOFTOP_EXIT", "ROOFTOP_ESCAPE") and closest_enemy:
            threat_yaw = math.atan2(closest_enemy["pos"][1] - y, closest_enemy["pos"][0] - x)
            defensive_aim_yaw = self.compute_defensive_facing(threat_yaw, move_yaw)
            torso_aim = (defensive_aim_yaw, 0.0)
            if not has_enemy_los and not (me["weapons"][1] and me["weapons"][1].get("kind") == "mortar"):
                should_fire = [False, False]

        elif (not closest_enemy or not has_enemy_los) and t - self.last_damaged_time < 1.5:
            # Turn to face recent unseen damage bearing
            if self.last_damaged_bearing is not None:
                defensive_aim_yaw = self.compute_defensive_facing(self.last_damaged_bearing, move_yaw)
                torso_aim = (defensive_aim_yaw, 0.0)

        elif self.active_state == "GATEKEEP_AMBUSH" and (not closest_enemy or not has_enemy_los):
            torso_aim = (self.gatekeep_aim_yaw, 0.0)

        elif (not closest_enemy or not has_enemy_los) and not (should_fire[0] or should_fire[1]):
            # Vision Radar Sweep (Section 6):
            # Every 3.5s, flick +120 deg (2 ticks) then -120 deg (2 ticks) to refresh
            # all enemies within 200m inside the 5.0s tracking memory!
            if t - self.last_sweep_time > 3.5:
                self.sweep_stage = 1
                self.sweep_ticks = 0
                self.last_sweep_time = t

            if self.sweep_stage == 1:
                # Flick right (+120 deg)
                torso_aim = (move_yaw + 2.0944, 0.0)
                self.sweep_ticks += 1
                if self.sweep_ticks >= 2:
                    self.sweep_stage = 2
                    self.sweep_ticks = 0
            elif self.sweep_stage == 2:
                # Flick left (-120 deg)
                torso_aim = (move_yaw - 2.0944, 0.0)
                self.sweep_ticks += 1
                if self.sweep_ticks >= 2:
                    self.sweep_stage = 0
                    self.sweep_ticks = 0
            else:
                torso_aim = (move_yaw, 0.0)

        # Energy Reserve Invariant Guard:
        # Energy must never drop below 15.0 unless actively evading mortar splash!
        is_dodging_mortar = is_threatened and (evasion_res.get("source") == "mortar")
        if not is_dodging_mortar:
            if energy < 16.7:
                should_boost = False
            if energy < 15.0:
                should_jump = False
            if holds_lightning and energy < 18.0:
                if me["weapons"][1] and me["weapons"][1].get("kind") == "lightning":
                    should_fire[1] = False
                elif me["weapons"][0] and me["weapons"][0].get("kind") == "lightning":
                    should_fire[0] = False

        action.update(
            move=move_vec,
            aim=[torso_aim[0], torso_aim[1]],
            boost=should_boost,
            jump=should_jump,
            fire=should_fire,
            reload=should_reload
        )

        # Debug Visualizations
        mf.draw.label(f"{self.active_state} | HP {my_hp:.0f} | {'DODGE' if is_threatened else ('REGEN' if in_combat_reset else 'COMBAT')}")
        if closest_enemy and has_enemy_los:
            mf.draw.line([x, y, z + 7], [closest_enemy["pos"][0], closest_enemy["pos"][1], closest_enemy["pos"][2] + 4.5], "#ff3333")
        mf.draw.point([nav_target[0], nav_target[1], z], "#33ff88", 6)

        return action

if __name__ == "__main__":
    mf.run(Bot)
