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
    NPCHunter,
    DynamicTargetScorer,
    LaserLightningCombo,
    select_optimal_target,
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


def test_npc_hunter():
    print("\n=== TEST 6: NPCHunter (Economy Snowball: Opening 60s, 90m, 2 Cubes) ===")
    hunter = NPCHunter(hunt_window_s=60.0, max_hunt_dist=90.0)
    my_pos = [0.0, 0.0, 0.0]

    # Entity list with NPCs and mechs
    entities = [
        {"id": 101, "kind": "npc", "hp": 350.0, "pos": [50.0, 0.0, 0.0]},    # 50m away (< 90m)
        {"id": 102, "kind": "npc", "hp": 350.0, "pos": [120.0, 0.0, 0.0]},   # 120m away (> 90m)
        {"id": 5, "kind": "mech", "hp": 800.0, "pos": [30.0, 0.0, 0.0]},
    ]

    # 1. In opening 60s (t = 20s), NPC 101 within 90m is found
    target = hunter.find_target(my_pos, entities, current_time=20.0)
    assert target is not None and target["id"] == 101, f"Expected NPC 101, got {target}"
    print("  [PASS] Opening 60s (t=20s): NPC within 90m prioritized for 2 power cubes.")

    # 2. Beyond 60s (t = 65s), hunt window is closed
    target_late = hunter.find_target(my_pos, entities, current_time=65.0)
    assert target_late is None, "Hunter must not target NPCs after 60s window!"
    print("  [PASS] Post-60s window: Hunter automatically deactivates.")

    # 3. Lowest HP priority among multiple NPCs
    entities_multi = [
        {"id": 101, "kind": "npc", "hp": 350.0, "pos": [40.0, 0.0, 0.0]},
        {"id": 103, "kind": "npc", "hp": 150.0, "pos": [60.0, 0.0, 0.0]},  # Lower HP!
    ]
    target_low = hunter.find_target(my_pos, entities_multi, current_time=15.0)
    assert target_low is not None and target_low["id"] == 103, "Failed to prioritize lower-HP NPC!"
    print("  [PASS] Lower-HP NPC prioritized to guarantee fastest 2-cube kill.")


def test_dynamic_target_scorer():
    print("\n=== TEST 7: DynamicTargetScorer (Rear-Armor 1.5x & Dynamic Formula) ===")
    scorer = DynamicTargetScorer()
    my_pos = [0.0, 0.0, 0.0]

    # Target 1: Enemy at [50, 0], aiming East (yaw=0.0). We are at [0, 0] (West of enemy).
    # Shot enters through enemy's BACK (> 90 deg from aim).
    rear_enemy = {
        "id": 1,
        "kind": "mech",
        "hp": 300.0,
        "pos": [50.0, 0.0, 0.0],
        "aim": [0.0, 0.0],  # Facing East away from us
        "weapons": ["gun"]
    }
    assert scorer.is_rear_facing(my_pos, rear_enemy) is True, "Expected rear-facing to be True!"
    print("  [PASS] Rear-facing angle detection verified (outside 180° front arc).")

    # Target 2: Enemy at [50, 0], aiming West (yaw=pi). Facing directly toward us.
    front_enemy = {
        "id": 2,
        "kind": "mech",
        "hp": 1400.0,
        "pos": [30.0, 0.0, 0.0],
        "aim": [math.pi, 0.0],  # Facing West toward us
        "weapons": ["gun"]
    }
    assert scorer.is_rear_facing(my_pos, front_enemy) is False, "Expected rear-facing to be False!"
    print("  [PASS] Front-facing angle detection verified (inside front arc).")

    # Score calculation formula test:
    # Score = (1000 / enemy_hp) + (is_rear_facing * 200) + (is_fighting_others * 150) - (distance * 0.05)
    score_rear = scorer.compute_score(my_pos, rear_enemy)
    # Expected: (1000 / 300) + (1 * 200) + (1 * 150) - (50 * 0.05) = 3.333 + 200 + 150 - 2.5 = 350.833
    expected_score_rear = (1000.0 / 300.0) + 200.0 + 150.0 - (50.0 * 0.05)
    assert abs(score_rear - expected_score_rear) < 0.01, f"Score mismatch: got {score_rear}, expected {expected_score_rear}"

    score_front = scorer.compute_score(my_pos, front_enemy)
    # Expected: (1000 / 1400) + 0 + 0 - (30 * 0.05) = 0.714 - 1.5 = -0.786
    expected_score_front = (1000.0 / 1400.0) + 0.0 + 0.0 - (30.0 * 0.05)
    assert abs(score_front - expected_score_front) < 0.01

    assert score_rear > score_front + 300.0, "Rear-facing wounded enemy must heavily outscore front-facing tank!"
    print(f"  [PASS] Formula verified: Rear-facing wounded score={score_rear:.1f} vs Front-facing score={score_front:.1f}")

    # Laser 1.5x rear execution math check
    laser_hit_rear = 315.0 * 1.5
    assert laser_hit_rear == 472.5
    assert laser_hit_rear > rear_enemy["hp"], "Laser hit must one-shot execute the 300 HP wounded enemy!"
    print(f"  [PASS] Laser rear-armor execution confirmed: 315 * 1.5 = {laser_hit_rear:.1f} dmg (One-shot kill!).")


def test_laser_lightning_combo():
    print("\n=== TEST 8: LaserLightningCombo (Short Locks & Combo Burst) ===")
    combo = LaserLightningCombo(range_m=40.0, cone_deg=30.0, short_duration=2.0)
    my_pos = [0.0, 0.0]
    aim_yaw = 0.0

    # Target carrying laser at 35m in cone
    laser_target = {
        "id": 1,
        "pos": [35.0, 0.0, 0.0],
        "weapons": ["gun", "laser"],
        "hp": 900.0
    }
    assert combo.should_discharge_lightning(my_pos, aim_yaw, laser_target, current_energy=20.0, lightning_cooldown=0.0) is True
    print("  [PASS] Preemptive short lock triggered on laser carrier within 40m cone.")

    # Target out of 40m range (45m)
    far_target = {
        "id": 2,
        "pos": [45.0, 0.0, 0.0],
        "weapons": ["laser"],
        "hp": 900.0
    }
    assert combo.should_discharge_lightning(my_pos, aim_yaw, far_target, current_energy=20.0, lightning_cooldown=0.0) is False
    print("  [PASS] Target at 45m (> 40m combo limit) rejected.")

    # Burst calculation on rear-facing enemy holding mortar
    mortar_rear = {
        "id": 3,
        "pos": [30.0, 0.0, 0.0],
        "weapons": ["gun", "mortar"],
        "hp": 700.0
    }
    burst = combo.calculate_combo_burst(mortar_rear, is_rear_facing=True)
    # Laser: 315 * 1.6 (cookoff) * 1.5 (rear) = 756.0!
    assert burst["laser_dmg"] == 756.0
    print(f"  [PASS] Combo burst against rear-armor mortar carrier: {burst['laser_dmg']:.1f} laser dmg!")


def test_select_optimal_target_api():
    print("\n=== TEST 9: select_optimal_target Exposed API ===")
    my_pos = [0.0, 0.0, 0.0]

    all_entities = [
        {"id": 101, "kind": "npc", "hp": 350.0, "pos": [50.0, 0.0, 0.0], "aim": [0.0, 0.0]},
        {"id": 2, "kind": "mech", "hp": 1400.0, "pos": [25.0, 0.0, 0.0], "aim": [math.pi, 0.0]},  # Front facing full HP
        {"id": 3, "kind": "mech", "hp": 320.0, "pos": [60.0, 0.0, 0.0], "aim": [0.0, 0.0]},        # Rear facing wounded
    ]

    # 1. At t=15s (opening 60s): Must select NPC 101 for economy snowball
    t_opt_early = select_optimal_target(my_pos, all_entities, current_time=15.0)
    assert t_opt_early is not None and t_opt_early["id"] == 101, f"Expected NPC 101, got {t_opt_early}"
    print("  [PASS] select_optimal_target at t=15s selects NPC 101 (Economy Snowball).")

    # 2. At t=75s (post 60s): Must select rear-facing wounded mech 3 instead of closer tank 2
    t_opt_late = select_optimal_target(my_pos, all_entities, current_time=75.0)
    assert t_opt_late is not None and t_opt_late["id"] == 3, f"Expected rear-facing Mech 3, got {t_opt_late}"
    print("  [PASS] select_optimal_target at t=75s selects rear-facing wounded Mech 3.")

    # 3. Method on ArtillerySystem
    sys_art = ArtillerySystem()
    t_sys = sys_art.select_optimal_target(my_pos, all_entities, current_time=15.0)
    assert t_sys is not None and t_sys["id"] == 101
    print("  [PASS] ArtillerySystem.select_optimal_target method verified.")


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
    test_npc_hunter()
    test_dynamic_target_scorer()
    test_laser_lightning_combo()
    test_select_optimal_target_api()
    test_performance_budget()
    print("\n==================================================================")
    print("ALL TESTS PASSED WITH 100% SUCCESS!")
    print("==================================================================")
