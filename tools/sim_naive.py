"""
Lightweight Monte Carlo Scenario Simulator for MechRoyale Tournament.
Simulates 15 Naive Built-in Bots vs 1 Disciplined Opportunist Bot.

Naive Bot Behavior (Reverse Engineered from protocol observations):
- Paths directly toward zone.next_center (center brawl).
- Aims directly at closest enemy with 0 lead.
- Holds trigger until 12-round mag is empty, reloads for 2.4s.
- Never retreats, never seeks cover, never resets HP.
- Ignores supply drops unless wandering within 8m.

Opportunist Bot:
- Avoids center rush; loots edges and takes supply drops.
- Strafes perpendicularly at 60-80m (completely dodges unled naive shots).
- Leads arm gun fire (lands 80%+ of shots).
- Decouples torso to protect rear armor.
- Takes uncontested supply drops for +3 cubes and full HP reset.
"""
import math
import random
from typing import List, Dict, Any

class NaiveBot:
    def __init__(self, bot_id: int, pos: List[float]):
        self.id = bot_id
        self.pos = list(pos)
        self.hp = 1400.0
        self.max_hp = 1400.0
        self.ammo = 12
        self.reload_timer = 0.0
        self.cooldown = 0.0
        self.cubes = 0
        self.alive = True
        self.death_time = -1.0

    def step(self, dt: float, target_center: List[float], all_bots: List['NaiveBot']):
        if not self.alive:
            return

        # 1. Walk directly to zone center
        dx = target_center[0] - self.pos[0]
        dy = target_center[1] - self.pos[1]
        dist = math.hypot(dx, dy)
        speed = 11.7
        if dist > 2.0:
            self.pos[0] += (dx / dist) * speed * dt
            self.pos[1] += (dy / dist) * speed * dt

        # 2. Fire at closest living enemy with NO lead
        if self.reload_timer > 0:
            self.reload_timer -= dt
            if self.reload_timer <= 0:
                self.ammo = 12
            return

        if self.cooldown > 0:
            self.cooldown -= dt
            return

        # Find closest
        closest = None
        c_dist = 1e9
        for b in all_bots:
            if b.id != self.id and b.alive:
                d = math.hypot(b.pos[0] - self.pos[0], b.pos[1] - self.pos[1])
                if d < c_dist:
                    c_dist = d
                    closest = b

        if closest and c_dist < 150.0:
            # Fire 1 round
            self.ammo -= 1
            self.cooldown = 0.24
            if self.ammo <= 0:
                self.reload_timer = 2.4

            # Bullet travels at 252 m/s
            t_flight = c_dist / 252.0
            # Naive aiming: aims at where enemy IS right now.
            # If enemy has transverse velocity (e.g. strafing), enemy displaces by v_perp * t_flight.
            # If displacement > 2.5m, shot misses!
            is_opportunist = getattr(closest, 'is_opportunist', False)
            if is_opportunist and closest.is_strafing and c_dist > 45.0:
                # Transverse velocity = 11.7 m/s => displacement at 60m is 11.7 * (60/252) = 2.78m > 2.5m hit radius!
                hit = False  # Dodged!
            else:
                hit = random.random() < 0.75  # Normal hit rate against non-strafing targets

            if hit:
                dmg = 45.0 * (1.0 + 0.04 * min(self.cubes, 10))
                closest.take_damage(dmg)

    def take_damage(self, amount: float):
        self.hp -= amount
        if self.hp <= 0:
            self.alive = False


class OpportunistSimBot(NaiveBot):
    def __init__(self, bot_id: int, pos: List[float]):
        super().__init__(bot_id, pos)
        self.is_opportunist = True
        self.is_strafing = False
        self.cubes = 2  # Farms 2-4 early crates
        self.regen_timer = 0.0
        self.drops_taken = 0

    def step(self, dt: float, target_center: List[float], all_bots: List[NaiveBot], drops: List[Dict[str, Any]], zone_r: float, t: float):
        if not self.alive:
            return

        # Regen check: 45 HP/s after 5s of peace
        self.regen_timer += dt
        if self.regen_timer >= 5.0 and self.hp < self.max_hp:
            self.hp = min(self.max_hp, self.hp + 45.0 * dt)

        # 1. Check supply drops (lands at 95, 165, 235)
        active_drop = None
        for d in drops:
            if d["active"] and not d["captured"]:
                active_drop = d
                break

        target_pos = target_center
        should_capture = False

        if active_drop:
            d_dist = math.hypot(active_drop["pos"][0] - self.pos[0], active_drop["pos"][1] - self.pos[1])
            if d_dist < 200.0:
                target_pos = active_drop["pos"]
                if d_dist < 8.0:
                    should_capture = True
                    active_drop["progress"] += dt
                    if active_drop["progress"] >= 4.0:
                        active_drop["captured"] = True
                        active_drop["active"] = False
                        self.cubes += 3
                        self.hp = self.max_hp
                        self.drops_taken += 1

        # 2. Movement logic
        # Find closest living threat
        closest_enemy = None
        c_dist = 1e9
        for b in all_bots:
            if b.id != self.id and b.alive:
                d = math.hypot(b.pos[0] - self.pos[0], b.pos[1] - self.pos[1])
                if d < c_dist:
                    c_dist = d
                    closest_enemy = b

        speed = 11.7
        if closest_enemy and c_dist < 100.0 and not should_capture:
            # Tactical standoff strafing at 60m
            self.is_strafing = True
            dx = closest_enemy.pos[0] - self.pos[0]
            dy = closest_enemy.pos[1] - self.pos[1]
            dist = math.hypot(dx, dy)
            to_x, to_y = dx / dist, dy / dist
            perp_x, perp_y = -to_y, to_x  # Perpendicular orbit

            if c_dist < 45.0:
                # Backpedal
                mv_x = -to_x * 0.7 + perp_x * 0.5
                mv_y = -to_y * 0.7 + perp_y * 0.5
            elif c_dist > 80.0:
                mv_x = to_x * 0.5 + perp_x * 0.6
                mv_y = to_y * 0.5 + perp_y * 0.6
            else:
                mv_x = perp_x
                mv_y = perp_y

            mlen = math.hypot(mv_x, mv_y)
            self.pos[0] += (mv_x / mlen) * speed * dt
            self.pos[1] += (mv_y / mlen) * speed * dt
        else:
            self.is_strafing = False
            # Move towards drop or zone flank (stay near radius - 40m, not center brawl)
            flank_r = max(20.0, zone_r - 40.0)
            cur_r = math.hypot(self.pos[0] - target_center[0], self.pos[1] - target_center[1])
            if active_drop and d_dist > 8.0:
                dx = target_pos[0] - self.pos[0]
                dy = target_pos[1] - self.pos[1]
                dist = max(0.1, math.hypot(dx, dy))
                self.pos[0] += (dx / dist) * speed * dt
                self.pos[1] += (dy / dist) * speed * dt
            elif cur_r > flank_r + 20.0:
                dx = target_center[0] - self.pos[0]
                dy = target_center[1] - self.pos[1]
                dist = max(0.1, math.hypot(dx, dy))
                self.pos[0] += (dx / dist) * speed * dt
                self.pos[1] += (dy / dist) * speed * dt

        # 3. Opportunist Shooting (With Predictive Lead)
        if self.reload_timer > 0:
            self.reload_timer -= dt
            if self.reload_timer <= 0:
                self.ammo = 12
            return

        if self.cooldown > 0:
            self.cooldown -= dt
            return

        if closest_enemy and c_dist < 120.0:
            self.ammo -= 1
            self.cooldown = 0.24
            if self.ammo <= 0:
                self.reload_timer = 2.4

            # Opportunist leads targets and punishes slow naive bots (85% accuracy)
            hit = random.random() < 0.85
            if hit:
                self.regen_timer = 0.0  # Reset combat timer on hit
                dmg = 45.0 * (1.0 + 0.04 * min(self.cubes, 10))
                closest_enemy.take_damage(dmg)

    def take_damage(self, amount: float):
        super().take_damage(amount)
        self.regen_timer = 0.0


def run_match():
    # Spawns around campus circle
    bots = []
    for i in range(15):
        ang = random.uniform(0, 2 * math.pi)
        r = random.uniform(180, 280)
        bots.append(NaiveBot(i, [r * math.cos(ang), r * math.sin(ang)]))

    # Bot 15 is our Opportunist
    ang = random.uniform(0, 2 * math.pi)
    r = random.uniform(220, 280)
    opportunist = OpportunistSimBot(15, [r * math.cos(ang), r * math.sin(ang)])
    bots.append(opportunist)

    # 3 supply drops
    drops = [
        {"time": 95.0, "pos": [random.uniform(-60, 60), random.uniform(-60, 60)], "active": False, "progress": 0.0, "captured": False},
        {"time": 165.0, "pos": [random.uniform(-40, 40), random.uniform(-40, 40)], "active": False, "progress": 0.0, "captured": False},
        {"time": 235.0, "pos": [random.uniform(-20, 20), random.uniform(-20, 20)], "active": False, "progress": 0.0, "captured": False},
    ]

    # Sim parameters
    t = 0.0
    dt = 0.1
    zone_center = [0.0, 0.0]
    zone_r = 560.0

    death_order = []

    while t < 360.0:
        # Zone shrinking
        if t < 60.0:
            zone_r = 560.0
        elif t < 110.0:
            zone_r = 560.0 - (560.0 - 380.0) * ((t - 60.0) / 50.0)
        elif t < 160.0:
            zone_r = 380.0
        elif t < 210.0:
            zone_r = 380.0 - (380.0 - 240.0) * ((t - 160.0) / 50.0)
        elif t < 260.0:
            zone_r = 240.0
        elif t < 310.0:
            zone_r = 240.0 - (240.0 - 100.0) * ((t - 260.0) / 50.0)
        else:
            zone_r = 100.0

        zone_dps = 20.0 if t < 150.0 else (50.0 if t < 250.0 else 120.0)

        # Activate drops
        for d in drops:
            if t >= d["time"] and not d["captured"]:
                d["active"] = True

        # Step bots
        living_count = 0
        for b in bots:
            if b.alive:
                living_count += 1
                # Zone damage
                dist_z = math.hypot(b.pos[0] - zone_center[0], b.pos[1] - zone_center[1])
                if dist_z > zone_r:
                    b.take_damage(zone_dps * dt)

                if b.id == 15:
                    b.step(dt, zone_center, bots, drops, zone_r, t)
                else:
                    b.step(dt, zone_center, bots)

        # Record newly dead bots
        for b in bots:
            if not b.alive and b.id not in death_order:
                b.death_time = t
                death_order.append(b.id)

        if living_count <= 1:
            break

        t += dt

    # Final standings
    survivors = [b for b in bots if b.alive]
    survivors.sort(key=lambda b: b.hp, reverse=True)
    for b in survivors:
        death_order.append(b.id)

    # Placement: last in death_order is 1st
    final_placements = {bot_id: 16 - idx for idx, bot_id in enumerate(death_order)}
    return final_placements[15], opportunist.cubes, opportunist.drops_taken


def main():
    print("=" * 65)
    print("RUNNING 48-MATCH PRACTICE SIMULATION (15 NAIVE BOTS vs 1 OPPORTUNIST)")
    print("=" * 65)

    placements = []
    cubes_list = []
    drops_list = []

    for match_idx in range(48):
        place, cubes, drops = run_match()
        placements.append(place)
        cubes_list.append(cubes)
        drops_list.append(drops)

    avg_place = sum(placements) / len(placements)
    avg_score = 16.0 - avg_place
    wins = sum(1 for p in placements if p == 1)
    top3 = sum(1 for p in placements if p <= 3)
    top5 = sum(1 for p in placements if p <= 5)
    avg_cubes = sum(cubes_list) / len(cubes_list)
    avg_drops = sum(drops_list) / len(drops_list)

    print(f"Total Matches: 48")
    print(f"Average Placement: {avg_place:.2f} (Tournament Score: {avg_score:.2f} / 15.0)")
    print(f"1st Place Wins:    {wins} / 48 ({wins/48*100:.1f}%)")
    print(f"Top 3 Finishes:    {top3} / 48 ({top3/48*100:.1f}%)")
    print(f"Top 5 Finishes:    {top5} / 48 ({top5/48*100:.1f}%)")
    print(f"Avg Drops Claimed: {avg_drops:.2f} / 3.0")
    print(f"Avg Power Cubes:   {avg_cubes:.1f}")
    print("=" * 65)

if __name__ == "__main__":
    main()
