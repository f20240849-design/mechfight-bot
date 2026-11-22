#!/usr/bin/env python3
"""
Fast Local 16-Mech Battle Royale Simulator for MechFight/1.
Simulates 1 OpportunistBot vs 15 Naive Bots across the full 480-second match.
Used for offline strategy benchmarking, win-rate calculation, and rapid iteration.
"""

import math
import random
import time
import json
import sys
import os
from typing import Dict, List, Tuple, Any, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from bot import Bot


class MechSim:
    def __init__(self, mech_id: int, is_hero: bool, spawn_pos: Tuple[float, float]):
        self.id = mech_id
        self.is_hero = is_hero
        self.pos = [spawn_pos[0], spawn_pos[1], 0.0]
        self.vel = [0.0, 0.0, 0.0]
        self.aim = [0.0, 0.0]  # [yaw, pitch]
        self.hp = 1400.0
        self.max_hp = 1400.0
        self.energy = 100.0
        self.cubes = 0
        self.alive = True
        self.weapons = [
            {"kind": "gun", "ammo": 12, "reload": 0.0, "cooldown": 0.0},
            None
        ]
        self.last_damaged_time = -100.0
        self.last_attack_time = -100.0
        self.kills = 0
        self.damage_dealt = 0.0
        self.damage_taken = 0.0
        self.placement = 16


class MatchSim:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.t = 0.0
        self.tick = 0
        self.dt = 0.1  # 10 decisions per second (tick_hz 20, decide_every 2)
        self.duration_s = 480.0

        # Zone configuration (from arena config)
        self.start_center = [41.3, -72.8]
        self.start_radius = 560.0
        self.zone_phases = [
            {"hold_s": 60, "shrink_s": 50, "radius": 380, "dps": 20},
            {"hold_s": 45, "shrink_s": 40, "radius": 240, "dps": 35},
            {"hold_s": 35, "shrink_s": 35, "radius": 140, "dps": 60},
            {"hold_s": 30, "shrink_s": 30, "radius": 70, "dps": 100},
            {"hold_s": 25, "shrink_s": 25, "radius": 25, "dps": 160},
            {"hold_s": 15, "shrink_s": 25, "radius": 0, "dps": 260}
        ]

        # Drops config: spawn at 70, 140, 210, fall for 25s
        self.drop_schedules = [70.0, 140.0, 210.0]
        self.active_drops = []

        # Floor loot & Crates & NPCs
        self.pickups = []
        self._spawn_initial_loot()

        # Crates
        self.crates = []
        for cid in range(25):
            cr_angle = random.uniform(0, 2 * math.pi)
            cr_dist = random.uniform(40, 350)
            self.crates.append({
                "id": cid + 100,
                "pos": [self.start_center[0] + math.cos(cr_angle) * cr_dist,
                        self.start_center[1] + math.sin(cr_angle) * cr_dist, 0.0],
                "hp": 300.0
            })

        # NPCs (12 roaming NPCs with 350 HP, dropping 2 cubes)
        self.npcs = []
        for nid in range(12):
            n_angle = random.uniform(0, 2 * math.pi)
            n_dist = random.uniform(80, 420)
            self.npcs.append({
                "id": nid + 500,
                "kind": "npc",
                "pos": [self.start_center[0] + math.cos(n_angle) * n_dist,
                        self.start_center[1] + math.sin(n_angle) * n_dist, 0.0],
                "vel": [0.0, 0.0, 0.0],
                "hp": 350.0,
                "max_hp": 350.0
            })

        # Spawn 16 Mechs
        self.mechs: List[MechSim] = []
        angles = [i * (2 * math.pi / 16) for i in range(16)]
        random.shuffle(angles)
        for i in range(16):
            r_spawn = random.uniform(280, 420)
            sx = self.start_center[0] + math.cos(angles[i]) * r_spawn
            sy = self.start_center[1] + math.sin(angles[i]) * r_spawn
            self.mechs.append(MechSim(i, is_hero=(i == 0), spawn_pos=(sx, sy)))

        # Initialize Hero Bot
        hello_msg = {
            "type": "hello",
            "protocol": "mechfight/1",
            "slot": 0,
            "seed": seed,
            "config": {
                "weapons": {
                    "gun": {"damage": 45, "cooldown": 0.24, "speed": 252, "range": 230, "spread_deg": 0.6, "mag": 12, "reload": 2.4},
                    "laser": {"damage": 315, "charge": 0.75, "min_charge": 0.15, "min_frac": 0.35, "hold_s": 2, "cooldown": 1.65, "range": 280, "cookoff": 0.6},
                    "lightning": {"damage": 60, "cooldown": 0.25, "range": 45, "cone_deg": 30, "chain": 2, "energy": 3, "short_s": 2},
                    "mortar": {"damage": 200, "splash": 160, "splash_radius": 7, "cooldown": 2.6, "speed": 72, "gravity": 22}
                },
                "drops": {"times": [70, 140, 210], "fall_s": 25, "radius": 8, "capture_s": 4, "cubes": 3, "repair": 1},
                "cubes": {"pickup_radius": 3.5, "hp_per_cube": 0.04, "damage_per_cube": 0.04, "stat_cap": 10},
                "mech": {"radius": 2.5, "walk_speed": 11.7, "boost_speed": 19.5, "hp": 1400, "energy": 100}
            },
            "map": {"dir": os.path.join(ROOT, "arena")}
        }
        self.hero_bot = Bot(hello_msg)
        self.nav = self.hero_bot.nav

        # Ensure all spawn positions are walkable
        for m in self.mechs:
            if not self.nav.is_walkable(m.pos[0], m.pos[1]):
                c_2m = int((m.pos[0] - self.nav.min_x) / 2.0)
                r_2m = int((m.pos[1] - self.nav.min_y) / 2.0)
                nc, nr = self.nav.find_nearest_walkable_2m(c_2m, r_2m)
                m.pos[0] = self.nav.min_x + nc * 2.0 + 1.0
                m.pos[1] = self.nav.min_y + nr * 2.0 + 1.0

    def _spawn_initial_loot(self):
        # 30 floor weapons (lasers, lightnings, mortars)
        kinds = ["laser", "lightning", "mortar", "laser", "lightning"]
        for pid in range(30):
            p_angle = random.uniform(0, 2 * math.pi)
            p_dist = random.uniform(60, 440)
            self.pickups.append({
                "id": pid + 1,
                "kind": kinds[pid % len(kinds)],
                "pos": [self.start_center[0] + math.cos(p_angle) * p_dist,
                        self.start_center[1] + math.sin(p_angle) * p_dist, 0.0]
            })

    def get_zone_status(self) -> Dict[str, Any]:
        """Calculates current zone radius, center, and shrink times."""
        elapsed = self.t
        curr_radius = self.start_radius
        curr_center = self.start_center
        next_radius = self.start_radius
        next_center = self.start_center
        shrink_start = 60.0
        shrink_end = 110.0
        phase_dps = 20.0

        accum = 0.0
        for ph in self.zone_phases:
            h_s = ph["hold_s"]
            s_s = ph["shrink_s"]
            p_rad = ph["radius"]
            dps = ph["dps"]

            t_hold_end = accum + h_s
            t_shrink_end = t_hold_end + s_s

            if elapsed < t_hold_end:
                curr_radius = curr_radius
                next_radius = p_rad
                shrink_start = t_hold_end
                shrink_end = t_shrink_end
                phase_dps = dps
                break
            elif elapsed < t_shrink_end:
                frac = (elapsed - t_hold_end) / max(0.1, s_s)
                curr_radius = curr_radius - frac * (curr_radius - p_rad)
                next_radius = p_rad
                shrink_start = t_hold_end
                shrink_end = t_shrink_end
                phase_dps = dps
                break
            else:
                curr_radius = p_rad
                accum = t_shrink_end

        return {
            "center": curr_center,
            "radius": curr_radius,
            "next_center": next_center,
            "next_radius": next_radius,
            "shrink_start": shrink_start,
            "shrink_end": shrink_end,
            "dps": phase_dps
        }

    def step(self) -> bool:
        """Simulates one 0.1s tick. Returns False when match ends."""
        self.t += self.dt
        self.tick += 1

        alive_mechs = [m for m in self.mechs if m.alive]
        if len(alive_mechs) <= 1 or self.t >= self.duration_s:
            # Match Over
            for m in alive_mechs:
                m.placement = len(alive_mechs)
            return False

        zone = self.get_zone_status()
        z_center = zone["center"]
        z_radius = zone["radius"]
        z_dps = zone["dps"]

        # 1. Update Supply Drops
        for d_time in self.drop_schedules:
            if abs(self.t - d_time) < self.dt:
                d_angle = random.uniform(0, 2 * math.pi)
                d_dist = random.uniform(20, max(30.0, zone["next_radius"] * 0.6))
                self.active_drops.append({
                    "pos": [z_center[0] + math.cos(d_angle) * d_dist,
                            z_center[1] + math.sin(d_angle) * d_dist, 0.0],
                    "lands_in": 25.0,
                    "holder": None,
                    "capture_progress": {}  # mech_id: seconds inside
                })

        for drop in self.active_drops:
            if drop["lands_in"] > 0:
                drop["lands_in"] = max(0.0, drop["lands_in"] - self.dt)

        # 2. Hero Bot Step
        hero = self.mechs[0]
        if hero.alive:
            obs = self._build_obs_for_hero(zone)
            try:
                act = self.hero_bot.act(obs)
            except Exception as e:
                act = {}
            self._apply_hero_action(hero, act)

        # 3. Naive Bots Step (Bots 1 to 15)
        for m in self.mechs[1:]:
            if not m.alive:
                continue
            self._step_naive_bot(m, zone)

        # 4. Resolve Combat, Bullets, Lasers, Drops, and Zone Damage
        self._resolve_tick(zone)

        return True

    def _build_obs_for_hero(self, zone: Dict[str, Any]) -> Dict[str, Any]:
        hero = self.mechs[0]
        alive_count = sum(1 for m in self.mechs if m.alive)

        # Enemies in vision (within 160m)
        visible_mechs = []
        for m in self.mechs:
            if m.id == hero.id or not m.alive:
                continue
            dist = math.hypot(m.pos[0] - hero.pos[0], m.pos[1] - hero.pos[1])
            if dist <= 160.0:
                visible_mechs.append({
                    "id": m.id,
                    "kind": "mech",
                    "pos": list(m.pos),
                    "vel": list(m.vel),
                    "aim": list(m.aim),
                    "hp": m.hp,
                    "max_hp": m.max_hp,
                    "cubes": m.cubes,
                    "weapons": [w.get("kind") for w in m.weapons if w],
                    "charging": False
                })

        # Add living NPCs within 120m
        for npc in self.npcs:
            if npc["hp"] <= 0:
                continue
            dist = math.hypot(npc["pos"][0] - hero.pos[0], npc["pos"][1] - hero.pos[1])
            if dist <= 120.0:
                visible_mechs.append({
                    "id": npc["id"],
                    "kind": "npc",
                    "pos": list(npc["pos"]),
                    "vel": list(npc["vel"]),
                    "hp": npc["hp"],
                    "max_hp": npc["max_hp"]
                })

        # Pickups within 100m
        near_pickups = [
            p for p in self.pickups
            if math.hypot(p["pos"][0] - hero.pos[0], p["pos"][1] - hero.pos[1]) < 100.0
        ]

        # Drops
        drop_obs = []
        for d in self.active_drops:
            drop_obs.append({
                "pos": list(d["pos"]),
                "lands_in": d["lands_in"],
                "holder": d["holder"]
            })

        dist_from_zone_center = math.hypot(hero.pos[0] - zone["center"][0], hero.pos[1] - zone["center"][1])
        in_zone = dist_from_zone_center <= zone["radius"]

        return {
            "type": "obs",
            "tick": self.tick,
            "t": round(self.t, 2),
            "alive": alive_count,
            "self": {
                "id": hero.id,
                "pos": list(hero.pos),
                "vel": list(hero.vel),
                "aim": list(hero.aim),
                "hp": hero.hp,
                "max_hp": hero.max_hp,
                "energy": hero.energy,
                "cubes": hero.cubes,
                "in_zone": in_zone,
                "grounded": True,
                "weapons": hero.weapons
            },
            "mechs": visible_mechs,
            "pickups": near_pickups,
            "drops": drop_obs,
            "crates": [c for c in self.crates if c["hp"] > 0 and math.hypot(c["pos"][0] - hero.pos[0], c["pos"][1] - hero.pos[1]) < 80.0],
            "zone": zone,
            "events": []
        }

    def _apply_hero_action(self, hero: MechSim, act: Dict[str, Any]):
        # Aim slewing
        if "aim" in act and act["aim"]:
            hero.aim[0] = act["aim"][0]
            hero.aim[1] = act["aim"][1]

        # Movement
        move_vec = act.get("move", [0.0, 0.0])
        speed = 19.5 if (act.get("boost") and hero.energy > 5.0) else 11.7
        if act.get("boost") and hero.energy > 5.0:
            hero.energy = max(0.0, hero.energy - 17.0 * self.dt)
        else:
            hero.energy = min(100.0, hero.energy + 18.0 * self.dt)

        mlen = math.hypot(move_vec[0], move_vec[1])
        if mlen > 0.01:
            hero.vel[0] = (move_vec[0] / mlen) * speed
            hero.vel[1] = (move_vec[1] / mlen) * speed
        else:
            hero.vel = [0.0, 0.0, 0.0]

        # Walkable check with sliding
        nx = hero.pos[0] + hero.vel[0] * self.dt
        ny = hero.pos[1] + hero.vel[1] * self.dt
        if self.nav.is_walkable(nx, ny):
            hero.pos[0] = nx
            hero.pos[1] = ny
        elif self.nav.is_walkable(nx, hero.pos[1]):
            hero.pos[0] = nx
        elif self.nav.is_walkable(hero.pos[0], ny):
            hero.pos[1] = ny

        # Firing
        fire = act.get("fire", [False, False])
        self._execute_mech_firing(hero, fire)

        # Loot action (take/drop)
        take_id = act.get("take")
        if take_id is not None:
            for p in list(self.pickups):
                if p["id"] == take_id:
                    pdist = math.hypot(p["pos"][0] - hero.pos[0], p["pos"][1] - hero.pos[1])
                    if pdist <= 4.0:
                        hero.weapons[1] = {"kind": p["kind"], "cooldown": 0.0}
                        self.pickups.remove(p)
                        break

    def _step_naive_bot(self, m: MechSim, zone: Dict[str, Any]):
        """Naive bot: walks directly to zone.next_center, aims at nearest enemy with unled aim, holds fire."""
        nx, ny = zone["next_center"]
        dx = nx - m.pos[0]
        dy = ny - m.pos[1]
        dist_to_center = math.hypot(dx, dy)

        # Movement: straight to center
        if dist_to_center > 10.0:
            m.vel[0] = (dx / dist_to_center) * 11.7
            m.vel[1] = (dy / dist_to_center) * 11.7
        else:
            m.vel = [0.0, 0.0, 0.0]
        # Walkable check with sliding
        nx_pos = m.pos[0] + m.vel[0] * self.dt
        ny_pos = m.pos[1] + m.vel[1] * self.dt
        if self.nav.is_walkable(nx_pos, ny_pos):
            m.pos[0] = nx_pos
            m.pos[1] = ny_pos
        elif self.nav.is_walkable(nx_pos, m.pos[1]):
            m.pos[0] = nx_pos
        elif self.nav.is_walkable(m.pos[0], ny_pos):
            m.pos[1] = ny_pos

        # Find closest living target (hero or other naive bot)
        closest_target = None
        closest_dist = 1e9
        for other in self.mechs:
            if other.id == m.id or not other.alive:
                continue
            odist = math.hypot(other.pos[0] - m.pos[0], other.pos[1] - m.pos[1])
            if odist < closest_dist and odist <= 120.0:
                closest_dist = odist
                closest_target = other

        # Naive aim: direct aim without leading
        if closest_target:
            tdx = closest_target.pos[0] - m.pos[0]
            tdy = closest_target.pos[1] - m.pos[1]
            m.aim[0] = math.atan2(tdy, tdx)
            # Hold trigger
            self._execute_mech_firing(m, [True, False])

    def _execute_mech_firing(self, m: MechSim, fire: List[bool]):
        for slot, should_f in enumerate(fire):
            if not should_f or slot >= len(m.weapons) or not m.weapons[slot]:
                continue
            w = m.weapons[slot]
            kind = w.get("kind")
            cd = w.get("cooldown", 0.0)

            if kind == "gun":
                ammo = w.get("ammo", 12)
                reload_t = w.get("reload", 0.0)
                if reload_t > 0:
                    w["reload"] = max(0.0, reload_t - self.dt)
                elif cd <= 0 and ammo > 0:
                    w["ammo"] = ammo - 1
                    w["cooldown"] = 0.24
                    self._cast_gun_bullet(m)
                    if w["ammo"] <= 0:
                        w["reload"] = 2.4
                        w["ammo"] = 12

            elif kind == "laser":
                if cd <= 0:
                    w["cooldown"] = 1.65 + 0.75  # charge + cooldown
                    self._cast_laser_beam(m)

            elif kind == "lightning":
                if cd <= 0 and m.energy >= 3.0:
                    w["cooldown"] = 0.25
                    m.energy -= 3.0
                    self._cast_lightning(m)

    def _cast_gun_bullet(self, shooter: MechSim):
        """Fires 252 m/s bullet along shooter.aim[0]. Checks hits on other mechs/NPCs."""
        yaw = shooter.aim[0]
        # Bullet travels along yaw
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        cube_mult = 1.0 + 0.04 * min(shooter.cubes, 10)
        base_dmg = 45.0 * cube_mult

        for target in self.mechs:
            if target.id == shooter.id or not target.alive:
                continue
            # Ray-circle intersection
            tx = target.pos[0] - shooter.pos[0]
            ty = target.pos[1] - shooter.pos[1]
            t_dist = math.hypot(tx, ty)
            if t_dist > 230.0:
                continue
            if not self.nav.has_los(shooter.pos[0], shooter.pos[1], target.pos[0], target.pos[1]):
                continue

            # Project target onto bullet ray
            proj = tx * cos_y + ty * sin_y
            if proj > 0:
                perp_dist = abs(tx * sin_y - ty * cos_y)
                # Mech hit radius is 2.5m
                if perp_dist <= 2.5:
                    # Check rear armor multiplier
                    dmg = base_dmg
                    t_facing = target.aim[0]
                    angle_diff = abs(math.atan2(math.sin(yaw - t_facing), math.cos(yaw - t_facing)))
                    if angle_diff < math.radians(90.0):
                        dmg *= 1.5  # Rear hit!

                    target.hp -= dmg
                    target.last_damaged_time = self.t
                    shooter.damage_dealt += dmg
                    shooter.last_attack_time = self.t
                    if target.hp <= 0:
                        target.alive = False
                        shooter.kills += 1
                        shooter.cubes += target.cubes + 2
                    break

    def _cast_laser_beam(self, shooter: MechSim):
        yaw = shooter.aim[0]
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        cube_mult = 1.0 + 0.04 * min(shooter.cubes, 10)
        base_dmg = 315.0 * cube_mult

        for target in self.mechs:
            if target.id == shooter.id or not target.alive:
                continue
            tx = target.pos[0] - shooter.pos[0]
            ty = target.pos[1] - shooter.pos[1]
            if math.hypot(tx, ty) > 280.0:
                continue
            if not self.nav.has_los(shooter.pos[0], shooter.pos[1], target.pos[0], target.pos[1]):
                continue
            proj = tx * cos_y + ty * sin_y
            if proj > 0:
                perp = abs(tx * sin_y - ty * cos_y)
                if perp <= 2.5:
                    dmg = base_dmg
                    angle_diff = abs(math.atan2(math.sin(yaw - target.aim[0]), math.cos(yaw - target.aim[0])))
                    if angle_diff < math.radians(90.0):
                        dmg *= 1.5
                    target.hp -= dmg
                    target.last_damaged_time = self.t
                    shooter.damage_dealt += dmg
                    if target.hp <= 0:
                        target.alive = False
                        shooter.kills += 1
                        shooter.cubes += target.cubes + 2
                    break

    def _cast_lightning(self, shooter: MechSim):
        cube_mult = 1.0 + 0.04 * min(shooter.cubes, 10)
        base_dmg = 60.0 * cube_mult
        for target in self.mechs:
            if target.id == shooter.id or not target.alive:
                continue
            dx = target.pos[0] - shooter.pos[0]
            dy = target.pos[1] - shooter.pos[1]
            dist = math.hypot(dx, dy)
            if dist <= 40.0:
                if not self.nav.has_los(shooter.pos[0], shooter.pos[1], target.pos[0], target.pos[1]):
                    continue
                bearing = math.atan2(dy, dx)
                diff = abs(math.atan2(math.sin(bearing - shooter.aim[0]), math.cos(bearing - shooter.aim[0])))
                if diff <= math.radians(15.0):
                    target.hp -= base_dmg
                    target.last_damaged_time = self.t
                    shooter.damage_dealt += base_dmg
                    if target.hp <= 0:
                        target.alive = False
                        shooter.kills += 1
                        shooter.cubes += target.cubes + 2

    def _resolve_tick(self, zone: Dict[str, Any]):
        z_center = zone["center"]
        z_rad = zone["radius"]
        z_dps = zone["dps"]

        # Cooldown & Regen tick
        for m in self.mechs:
            if not m.alive:
                continue
            for w in m.weapons:
                if w and w.get("cooldown", 0) > 0:
                    w["cooldown"] = max(0.0, w["cooldown"] - self.dt)

            # Zone gas damage
            dist_to_zc = math.hypot(m.pos[0] - z_center[0], m.pos[1] - z_center[1])
            if dist_to_zc > z_rad:
                m.hp -= z_dps * self.dt
                m.last_damaged_time = self.t
                if m.hp <= 0:
                    m.alive = False

            # Out of combat active regen: 45 HP/s after 5.0s
            if self.t - max(m.last_damaged_time, m.last_attack_time) >= 5.0:
                if m.hp < m.max_hp:
                    m.hp = min(m.max_hp, m.hp + 45.0 * self.dt)

        # Supply drop capture resolution
        for drop in list(self.active_drops):
            if drop["lands_in"] > 0:
                continue
            dx, dy = drop["pos"][0], drop["pos"][1]
            for m in self.mechs:
                if not m.alive:
                    continue
                if math.hypot(m.pos[0] - dx, m.pos[1] - dy) <= 8.0:
                    prog = drop["capture_progress"].get(m.id, 0.0) + self.dt
                    drop["capture_progress"][m.id] = prog
                    if prog >= 4.0:  # Captured!
                        m.hp = m.max_hp  # 100% full repair!
                        m.cubes += 3
                        self.active_drops.remove(drop)
                        break


def run_benchmark(num_matches: int = 20) -> Dict[str, Any]:
    print(f"[*] Running {num_matches} Headless Simulated Practice Matches vs 15 Naive Bots...")
    t0 = time.time()
    placements = []
    kills_list = []
    damage_list = []
    wins = 0
    top3 = 0

    for m_idx in range(num_matches):
        sim = MatchSim(seed=1000 + m_idx)
        step_count = 0
        while sim.step():
            step_count += 1
            if step_count > 4800:
                break

        hero = sim.mechs[0]
        # Calculate final placement
        alive_at_end = [m for m in sim.mechs if m.alive]
        if hero.alive:
            # Sorted by HP
            alive_at_end.sort(key=lambda m: -m.hp)
            place = alive_at_end.index(hero) + 1
        else:
            # Dead mechs ranked by death order
            place = len(alive_at_end) + 1

        placements.append(place)
        kills_list.append(hero.kills)
        damage_list.append(hero.damage_dealt)
        if place == 1:
            wins += 1
        if place <= 3:
            top3 += 1

    dt = time.time() - t0
    avg_place = sum(placements) / len(placements)
    avg_kills = sum(kills_list) / len(kills_list)
    avg_dmg = sum(damage_list) / len(damage_list)

    # Tournament score estimate (16 - avg_place + avg_kills * 0.5)
    est_score = max(0.0, 16.0 - avg_place + avg_kills * 0.5)

    print(f"[+] Completed {num_matches} matches in {dt:.2f}s ({dt/num_matches*1000:.1f} ms/match)")
    print(f"    Avg Placement: {avg_place:.2f} / 16  (1st place wins: {wins}/{num_matches}, Top 3: {top3}/{num_matches})")
    print(f"    Avg Kills:     {avg_kills:.2f}")
    print(f"    Avg Damage:    {avg_dmg:.1f}")
    print(f"    Estimated Hackathon Score: {est_score:.2f} / 15.0")
    return {
        "avg_placement": avg_place,
        "avg_kills": avg_kills,
        "avg_dmg": avg_dmg,
        "est_score": est_score,
        "wins": wins,
        "top3": top3
    }

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    run_benchmark(n)
