"""
Standalone Test Suite for BALLIS Artillery & Weapon Triangle Engine (artillery.py).
Tests:
1. Indirect Parabolic Mortar Solver (High-arc solution, trajectory verification, supply drop disruption, camper flushing).
2. Hitscan Laser Charge/Discharge Controller (Cover tactics, peek-and-fire, cookoff damage).
3. Lightning Cone Trigger (40m trigger, 30° arc, energy checks, laser-carrier priority).
4. Smart Weapon Swap Manager (Arm gun preservation invariant, triangle optimization).
5. Performance & Computational Budget Benchmark (< 1.5 ms / tick).
"""

import math
import sys
import time
from typing import Dict, List, Any

from artillery import (
    IndirectParabolicMortarSolver,
    MortarTrajectory,
    HitscanLaserController,
    LightningConeTrigger,
    SmartWeaponSwapManager,
    ArtillerySystem,
)


def test_mortar_high_arc_solver():
    print("=== TEST 1: Parabolic Mortar Solver (High-Arc & Accuracy) ===")
    solver = IndirectParabolicMortarSolver(
        speed=70.0,
        gravity=18.0,
        blast_radius=14.0,
        direct_damage=200.0,
        cooldown=2.6,
        tube_height_offset=9.5,
        min_range=25.0
    )

    my_pos = (100.0, 100.0, 0.0)

    # Test across tactical bombardment distances (70m to 120m) with varying elevations
    test_cases = [
        {"desc": "70m Flat Ground", "target": (170.0, 100.0, 0.0)},
        {"desc": "85m Low Target", "target": (160.0, 160.0, -5.0)},
        {"desc": "100m Uphill Target", "target": (100.0, 200.0, 12.0)},
        {"desc": "120m Campus Long Shot", "target": (220.0, 100.0, 3.0)},
    ]

    for tc in test_cases:
        target = tc["target"]
        traj = solver.solve_high_arc(my_pos, target)
        assert traj is not None, f"Failed to solve high arc for {tc['desc']}"
        assert traj.is_high_arc, "Expected high-arc trajectory"
        assert math.degrees(traj.pitch) > 45.0, f"High arc pitch must be > 45°, got {math.degrees(traj.pitch):.1f}°"

        # Trajectory Physics Verification: z(t_flight) must hit target z
        lz = my_pos[2] + solver.tube_height_offset
        vz = solver.speed * math.sin(traj.pitch)
        t_f = traj.flight_time
        z_impact = lz + vz * t_f - 0.5 * solver.gravity * t_f * t_f

        err = abs(z_impact - target[2])
        assert err < 1e-4, f"Impact z error too high: {err:.6f}m (target {target[2]}, got {z_impact})"

        # Apex verification: High-arc shells must reach high altitudes (> 50m) to clear buildings
        apex = traj.apex_height
        assert apex > 50.0, f"Apex too low ({apex:.1f}m) to clear buildings"

        # Test sampling
        points = traj.sample_points(num_points=10, gravity=solver.gravity, speed=solver.speed)
        assert len(points) == 10
        assert abs(points[0][0] - my_pos[0]) < 1e-3
        assert abs(points[-1][2] - target[2]) < 1e-3

        print(f"  [PASS] {tc['desc']}: dist={traj.horizontal_dist:.1f}m, pitch={math.degrees(traj.pitch):.1f}°, "
              f"flight_time={traj.flight_time:.2f}s, apex={apex:.1f}m (Error: {err*1000:.2f} mm)")

    # Test point-blank safety (< 25m)
    pb_target = (115.0, 100.0, 0.0)  # 15m away
    pb_traj = solver.solve_high_arc(my_pos, pb_target)
    assert pb_traj is None, "Point-blank safety check failed: solver allowed shot < 25m!"
    print("  [PASS] Point-blank self-damage mitigation verified (target < 25m rejected).")


def test_supply_drop_and_camper_flushing():
    print("\n=== TEST 2: Supply Drop Capture Interruption & Camper Flushing ===")
    solver = IndirectParabolicMortarSolver(
        speed=70.0,
        gravity=18.0,
        blast_radius=14.0,
        cooldown=2.6
    )

    my_pos = (50.0, 50.0, 0.0)
    drop_pos = (130.0, 50.0, 0.0)  # 80m away

    # Scenario A: Enemy capturing supply drop
    drop_data = {
        "id": 20000,
        "pos": drop_pos,
        "radius": 8.0,
        "lands_in": 0.0,
        "progress": 2.1,
        "holder": 5
    }
    interruption = solver.interrupt_supply_drop(my_pos, drop_data)
    assert interruption is not None, "Failed to interrupt active supply drop capture!"
    (aim_yaw, aim_pitch), t_flight, status = interruption
    assert status == "INTERRUPT_ACTIVE_CAPTURE"
    assert math.degrees(aim_pitch) > 60.0
    print(f"  [PASS] Active Supply Drop Interrupted: pitch={math.degrees(aim_pitch):.1f}°, t_flight={t_flight:.2f}s, status={status}")

    # Scenario B: Pre-emptive strike on landing drop
    drop_landing = {
        "id": 20001,
        "pos": drop_pos,
        "radius": 8.0,
        "lands_in": 7.0,  # Lands in 7.0s, close to flight time ~7.5s
        "progress": 0.0,
        "holder": None
    }
    preemptive = solver.interrupt_supply_drop(my_pos, drop_landing)
    assert preemptive is not None, "Failed to identify pre-emptive drop bombardment!"
    _, _, p_status = preemptive
    assert p_status == "PREEMPTIVE_DROP_SIEGE"
    print(f"  [PASS] Preemptive Drop Siege triggered: status={p_status}")

    # Scenario C: Flushing a camper behind cover 90m away
    camper_pos = (50.0, 140.0, 0.0)
    flush_res = solver.flush_camper(my_pos, camper_pos)
    assert flush_res is not None, "Failed to compute camper flush arc"
    (c_yaw, c_pitch), c_time = flush_res
    assert math.degrees(c_pitch) > 70.0
    print(f"  [PASS] Camper Flushed behind building: 90m high-arc pitch={math.degrees(c_pitch):.1f}°, t_flight={c_time:.2f}s")


def test_laser_charge_discharge_controller():
    print("\n=== TEST 3: Hitscan Laser Charge/Discharge Controller ===")
    laser = HitscanLaserController(
        charge_time=0.75,
        cooldown=1.65,
        range_m=280.0,
        damage=315.0
    )

    my_pos = [0.0, 0.0, 0.0]
    enemy_pos = [80.0, 60.0, 0.0]  # 100m away

    # Test cookoff damage multiplier
    assert laser.estimate_laser_damage(["gun", "lightning"]) == 315.0
    assert laser.estimate_laser_damage(["gun", "mortar"]) == 504.0, "Cookoff damage must be 504.0 (+60%)"
    print(f"  [PASS] Cookoff damage against Mortar carrier verified: 504.0 dmg (+60%)")

    # Aim calculation: direct hitscan aim
    yaw, pitch = laser.compute_aim(my_pos, enemy_pos)
    expected_yaw = math.atan2(60.0, 80.0)
    assert abs(yaw - expected_yaw) < 1e-4
    print(f"  [PASS] Hitscan aim aligned: yaw={math.degrees(yaw):.1f}°, pitch={math.degrees(pitch):.1f}°")

    # Cover cycle simulation:
    # 1. Start behind cover, no LOS -> Charge trigger held, intent = HOLD_COVER
    weapon_w = {"kind": "laser", "cooldown": 0.0, "charge": 0.0}
    hold, move = laser.update_cycle(weapon_w, has_los=False, target_dist=100.0, in_cover=True)
    assert hold is True and move == "HOLD_COVER", f"Expected hold=True, HOLD_COVER, got {hold}, {move}"

    # 2. Reached full charge behind cover -> Ready to peek!
    weapon_w["charge"] = 0.75
    hold, move = laser.update_cycle(weapon_w, has_los=False, target_dist=100.0, in_cover=True)
    assert hold is True and move == "PEEK_OUT", f"Expected hold=True, PEEK_OUT, got {hold}, {move}"

    # 3. Peeks out with LOS -> RELEASE TRIGGER TO DISCHARGE BEAM!
    hold, move = laser.update_cycle(weapon_w, has_los=True, target_dist=100.0, in_cover=True)
    assert hold is False and move == "RETREAT_TO_COVER", f"Discharge failed! Expected hold=False, RETREAT_TO_COVER, got {hold}, {move}"

    # 4. Weapon enters cooldown (1.65s) -> Must retreat and hold trigger False
    weapon_w["cooldown"] = 1.65
    weapon_w["charge"] = 0.0
    hold, move = laser.update_cycle(weapon_w, has_los=False, target_dist=100.0, in_cover=True)
    assert hold is False and move == "RETREAT_TO_COVER"
    print("  [PASS] Peek-and-shoot cover state machine fully verified (Charge in cover -> Peek -> Discharge -> Duck back).")


def test_lightning_cone_trigger():
    print("\n=== TEST 4: Lightning 45m Cone Trigger ===")
    lightning = LightningConeTrigger(
        max_range=45.0,
        trigger_range=40.0,
        cone_deg=30.0,
        damage_per_shot=60.0,
        cooldown=0.25,
        energy_cost=3.0
    )

    my_pos = [0.0, 0.0]
    aim_yaw = 0.0  # Aiming East along +X

    # Target 1: 30m East (Dead Center of 30° cone) -> Should fire
    t1 = [30.0, 0.0]
    assert lightning.should_fire(my_pos, aim_yaw, t1, current_energy=20.0, weapon_cooldown=0.0) is True
    print("  [PASS] Target dead-center in cone (30m) triggers lightning.")

    # Target 2: 30m East, offset by 10° (< 15° half-cone) -> Should fire
    t2 = [30.0 * math.cos(math.radians(10)), 30.0 * math.sin(math.radians(10))]
    assert lightning.should_fire(my_pos, aim_yaw, t2, current_energy=20.0, weapon_cooldown=0.0) is True
    print("  [PASS] Target at 10° offset (inside 30° arc) triggers lightning.")

    # Target 3: 30m East, offset by 22° (> 15° half-cone) -> Outside cone, must not fire
    t3 = [30.0 * math.cos(math.radians(22)), 30.0 * math.sin(math.radians(22))]
    assert lightning.should_fire(my_pos, aim_yaw, t3, current_energy=20.0, weapon_cooldown=0.0) is False
    print("  [PASS] Target at 22° offset (outside 30° arc) does not trigger.")

    # Target 4: 42m away (exceeds conservative 40m trigger range) -> Must not fire
    t4 = [42.0, 0.0]
    assert lightning.should_fire(my_pos, aim_yaw, t4, current_energy=20.0, weapon_cooldown=0.0) is False
    print("  [PASS] Target at 42m (> 40m trigger threshold) does not trigger.")

    # Energy gate: low energy (< 3.0) -> Must not fire
    assert lightning.should_fire(my_pos, aim_yaw, t1, current_energy=2.5, weapon_cooldown=0.0) is False
    print("  [PASS] Energy conservation gate verified (< 3 energy prevents firing).")

    # Target prioritization: Laser carrier must be prioritized for 2.0s short circuit!
    candidates = [
        {"id": 1, "pos": [25.0, 2.0, 0.0], "hp": 400.0, "weapons": ["gun"]},
        {"id": 2, "pos": [30.0, 0.0, 0.0], "hp": 800.0, "weapons": ["gun", "laser"]},  # Holds laser!
    ]
    best_t = lightning.prioritize_target(my_pos, aim_yaw, candidates)
    assert best_t is not None and best_t["id"] == 2, "Failed to prioritize laser carrier!"
    print("  [PASS] Weapon Triangle priority verified: Laser carrier selected for 2.0s short lock.")


def test_smart_weapon_swap_manager():
    print("\n=== TEST 5: Smart Weapon Swap Manager & Arm Gun Invariant ===")
    mgr = SmartWeaponSwapManager(pickup_radius=4.0)

    # Initial state: Slot 0 = Arm Gun, Slot 1 = empty
    me = {
        "pos": [0.0, 0.0, 0.0],
        "weapons": [
            {"kind": "gun", "ammo": 12, "cooldown": 0.0},
            None
        ]
    }

    # Case 1: First loot weapon found (Laser) -> Takes into empty Slot 1, DO NOT drop Gun
    pickups = [{"id": 8001, "kind": "laser", "pos": [2.0, 1.0, 0.0]}]
    act = mgr.evaluate_swap(me, pickups)
    assert act == {"take": 8001}, f"Expected take without drop, got {act}"
    print("  [PASS] Empty Slot 1: Takes Laser, keeps Arm Gun.")

    # Case 2: Holds Gun (slot 0) + Mortar (slot 1), finds Laser -> Swap Slot 1, DO NOT drop Gun
    me["weapons"][1] = {"kind": "mortar", "cooldown": 0.0}
    act = mgr.evaluate_swap(me, pickups)
    assert act == {"take": 8001, "drop": 1}, f"Expected swap slot 1 (mortar), got {act}"
    print("  [PASS] Gun + Mortar: Replaces Mortar with Laser in Slot 1 (Arm Gun preserved).")

    # Case 3: Holds Gun (slot 0) + Laser (slot 1), finds Mortar -> Do NOT drop Gun or Laser
    mortar_pickup = [{"id": 8002, "kind": "mortar", "pos": [1.0, 2.0, 0.0]}]
    me["weapons"][1] = {"kind": "laser", "cooldown": 0.0, "charge": 0.0}
    act = mgr.evaluate_swap(me, mortar_pickup)
    assert act is None, f"Gun + Laser must NOT swap for Mortar! Got {act}"
    print("  [PASS] Gun + Laser: Rejects inferior Mortar pickup.")

    # Case 4: GOLDEN INVARIANT RULE:
    # "Never drop the Arm Gun unless you hold Laser AND find Lightning (or vice versa)."
    # We hold Gun (slot 0) + Laser (slot 1), and find Lightning!
    lightning_pickup = [{"id": 8003, "kind": "lightning", "pos": [2.0, 0.0, 0.0]}]
    act = mgr.evaluate_swap(me, lightning_pickup)
    assert act == {"take": 8003, "drop": 0}, f"Expected drop slot 0 (Gun) for Lightning! Got {act}"
    print("  [PASS] Golden Rule Verified: Holding Laser and finding Lightning drops Arm Gun (Slot 0) to hold Laser + Lightning!")

    # Vice-versa: Hold Gun (slot 0) + Lightning (slot 1), find Laser -> Drop Gun (slot 0)
    me["weapons"][1] = {"kind": "lightning", "cooldown": 0.0}
    act = mgr.evaluate_swap(me, pickups)  # pickups has Laser
    assert act == {"take": 8001, "drop": 0}, f"Expected drop slot 0 (Gun) for Laser! Got {act}"
    print("  [PASS] Vice-Versa Verified: Holding Lightning and finding Laser drops Arm Gun (Slot 0) to hold Lightning + Laser!")

    # Case 5: Holds Laser + Lightning (no Gun). Finds Mortar -> Never swap
    me["weapons"][0] = {"kind": "lightning", "cooldown": 0.0}
    me["weapons"][1] = {"kind": "laser", "cooldown": 0.0}
    act = mgr.evaluate_swap(me, mortar_pickup)
    assert act is None, f"Ultimate loadout (Laser + Lightning) must not swap! Got {act}"
    print("  [PASS] Ultimate loadout (Laser + Lightning) locks against lower-tier weapons.")


def test_performance_budget():
    print("\n=== TEST 6: Performance Benchmark (< 1.5 ms execution budget) ===")
    artillery = ArtillerySystem()

    me = {
        "pos": [120.0, -45.0, 2.0],
        "aim": [0.45, 0.1],
        "energy": 65.0,
        "weapons": [
            {"kind": "gun", "ammo": 8, "reload": 0.0, "cooldown": 0.0},
            {"kind": "laser", "cooldown": 0.0, "charge": 0.5}
        ]
    }
    target = {
        "pos": [180.0, 20.0, 4.0],
        "vel": [5.0, -2.0, 0.0],
        "weapons": ["gun", "mortar"]
    }
    pickups = [
        {"id": 8005, "kind": "lightning", "pos": [121.0, -44.0, 2.0]},
        {"id": 8006, "kind": "mortar", "pos": [118.0, -46.0, 2.0]}
    ]
    drops = [
        {"id": 20000, "pos": [210.0, 10.0, 0.0], "radius": 8.0, "lands_in": 0.0, "holder": 3, "progress": 1.5}
    ]
    enemies = [target]

    # Benchmark 10,000 full combat tick evaluations
    num_iterations = 10000
    times = []
    t_start_total = time.perf_counter()

    for _ in range(num_iterations):
        t0 = time.perf_counter()
        res = artillery.evaluate_combat_tick(
            me=me,
            target=target,
            has_los=True,
            pickups=pickups,
            drops=drops,
            all_enemies=enemies,
            in_cover=False,
            current_time=12.0
        )
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    total_wall_s = time.perf_counter() - t_start_total
    times.sort()
    p50 = times[int(num_iterations * 0.50)]
    p90 = times[int(num_iterations * 0.90)]
    p99 = times[int(num_iterations * 0.99)]
    max_t = times[-1]
    mean_t = sum(times) / len(times)

    print(f"  Benchmark ({num_iterations:,} ticks completed in {total_wall_s:.2f}s):")
    print(f"    Mean execution time: {mean_t:.4f} ms")
    print(f"    p50 execution time:  {p50:.4f} ms")
    print(f"    p90 execution time:  {p90:.4f} ms")
    print(f"    p99 execution time:  {p99:.4f} ms")
    print(f"    Max execution time:  {max_t:.4f} ms")

    # Budget assertion: < 1.5 ms per tick
    budget_limit = 1.5
    assert p99 < budget_limit, f"p99 execution time {p99:.3f} ms exceeded 1.5 ms budget!"
    assert mean_t < 0.2, f"Mean execution time {mean_t:.3f} ms is unexpectedly high!"
    print(f"  [PASS] STRICT BUDGET COMPLIED: p99 ({p99:.4f} ms) is ~{budget_limit / p99:.0f}x faster than 1.5 ms cap!")


if __name__ == "__main__":
    print("==================================================================")
    print("RUNNING BALLIS ARTILLERY & WEAPON TRIANGLE COMPREHENSIVE TEST SUITE")
    print("==================================================================")
    test_mortar_high_arc_solver()
    test_supply_drop_and_camper_flushing()
    test_laser_charge_discharge_controller()
    test_lightning_cone_trigger()
    test_smart_weapon_swap_manager()
    test_performance_budget()
    print("\n==================================================================")
    print("ALL TESTS PASSED WITH 100% SUCCESS!")
    print("==================================================================")
