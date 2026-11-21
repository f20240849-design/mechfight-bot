#!/usr/bin/env python3
"""
Comprehensive Test and Benchmark Suite for ATLAS (3D Verticality and Rooftop Specialist).
Tests:
1. Startup Rooftop Parser & Landability Filter (10m - 25m, flat, unobstructed).
2. Jump-Jet Climb Energy Solver (~35 energy for 15m roof, 2.5s regen).
3. Jump-Jet Escape Controller (find_nearest_rooftop_escape with enemy shielding).
4. The 3-Meter Roof-Edge Setback Rule & Ground Muzzle LOS Occlusion.
5. Safe Drop-Down & Exit Vectors (Zero Fall Damage descent into walkable ground).
6. Multi-stage Climb State Machine (APPROACH -> JUMP -> JET -> PERCH).
7. Performance Latency Benchmark (< 1.5 ms execution budget).
"""

import math
import sys
import time
import numpy as np

from rooftops import (
    RooftopManager,
    Rooftop,
    RooftopEscapeTarget,
    DropDownPoint,
    find_nearest_rooftop_escape,
    get_safe_perch_point,
    get_safe_exit_vector,
    is_on_rooftop,
    point_segment_distance,
    point_in_polygon,
    ROOF_MIN_HEIGHT,
    ROOF_MAX_HEIGHT,
    ROOF_MIN_SETBACK,
    ENERGY_REGEN,
    REGEN_PER_S,
    MECH_HEIGHT,
    MUZZLE_HEIGHT,
    EYE_HEIGHT
)


def run_tests():
    print("=" * 72)
    print("RUNNING ATLAS 3D VERTICALITY & ROOFTOP SPECIALIST TEST SUITE")
    print("=" * 72)

    # -------------------------------------------------------------
    # TEST 1: Startup Rooftop Parser & Landability Verification
    # -------------------------------------------------------------
    print("\n=== TEST 1: Startup Rooftop Parser & Landability Filter ===")
    t0 = time.perf_counter()
    mgr = RooftopManager(map_dir="arena")
    t1 = time.perf_counter()
    init_ms = (t1 - t0) * 1000.0
    print(f"  [PASS] RooftopManager initialized in {init_ms:.2f} ms")
    print(f"  [PASS] Identified {len(mgr.rooftop_list)} flat, landable rooftops between 10m and 25m")

    assert len(mgr.rooftop_list) >= 80, f"Expected >= 80 rooftops, got {len(mgr.rooftop_list)}"

    for r in mgr.rooftop_list:
        assert ROOF_MIN_HEIGHT <= r.height <= ROOF_MAX_HEIGHT, f"Roof {r.id} height {r.height:.1f} out of bounds"
        assert r.max_setback >= ROOF_MIN_SETBACK, f"Roof {r.id} max setback {r.max_setback:.2f} < 3.0m"
        assert len(r.drop_points) > 0, f"Roof {r.id} has no walkable drop points"

    print("  [PASS] All rooftops strictly verified within 10m-25m height and >= 3.0m setback capacity.")

    # -------------------------------------------------------------
    # TEST 2: Jump-Jet Climb Energy Physics
    # -------------------------------------------------------------
    print("\n=== TEST 2: Jump-Jet Climb Energy Physics ===")
    # 15m roof: (15 - 1.2) / 10 = 1.38s -> 15 + 1.38 * 15 = 35.7 energy
    e10 = RooftopManager.calculate_climb_energy(10.0)
    e15 = RooftopManager.calculate_climb_energy(15.0)
    e20 = RooftopManager.calculate_climb_energy(20.0)
    e25 = RooftopManager.calculate_climb_energy(25.0)

    print(f"  [PASS] 10m Climb: {e10:.1f} energy (Regens in {e10/ENERGY_REGEN:.2f}s)")
    print(f"  [PASS] 15m Climb: {e15:.1f} energy (Regens in {e15/ENERGY_REGEN:.2f}s) -> Matches ~35 energy spec")
    print(f"  [PASS] 20m Climb: {e20:.1f} energy (Regens in {e20/ENERGY_REGEN:.2f}s)")
    print(f"  [PASS] 25m Climb: {e25:.1f} energy (Regens in {e25/ENERGY_REGEN:.2f}s)")

    assert 34.0 <= e15 <= 37.0, f"Expected ~35 energy for 15m roof, got {e15:.1f}"
    assert e15 / ENERGY_REGEN <= 2.5, f"Expected regen in <= 2.5s, got {e15/ENERGY_REGEN:.2f}s"

    # -------------------------------------------------------------
    # TEST 3: Jump-Jet Escape Controller & Enemy Shielding
    # -------------------------------------------------------------
    print("\n=== TEST 3: Jump-Jet Escape Controller (find_nearest_rooftop_escape) ===")
    my_pos = [150.0, -300.0, 0.0]
    enemy_pos = [130.0, -290.0, 0.0]
    escape = mgr.find_nearest_rooftop_escape(my_pos, enemy_pos, energy=75.0)

    assert escape is not None, "Failed to find rooftop escape target"
    print(f"  [PASS] Escape Roof Selected: {escape.roof_name} (ID: {escape.roof_id})")
    print(f"         Height: {escape.roof_height:.1f}m, Top: {escape.roof_top:.1f}m")
    print(f"         Energy Cost: {escape.energy_cost:.1f} <= 75.0 available")
    print(f"         Ground Approach: ({escape.ground_approach[0]:.1f}, {escape.ground_approach[1]:.1f}, {escape.ground_approach[2]:.1f})")
    print(f"         Climb Edge: ({escape.climb_edge[0]:.1f}, {escape.climb_edge[1]:.1f}, {escape.climb_edge[2]:.1f})")
    print(f"         Safe 3m Perch: ({escape.perch_point[0]:.1f}, {escape.perch_point[1]:.1f}, {escape.perch_point[2]:.1f})")

    # Energy starvation defense
    starved_escape = mgr.find_nearest_rooftop_escape(my_pos, enemy_pos, energy=10.0)
    print(f"  [PASS] Energy Starvation Fall Protection: Escape with 10 energy correctly rejected: {starved_escape is None}")
    assert starved_escape is None, "Bot must not attempt climbs with insufficient energy"

    # -------------------------------------------------------------
    # TEST 4: The 3-Meter Roof-Edge Setback Rule & Ground LOS Occlusion
    # -------------------------------------------------------------
    print("\n=== TEST 4: The 3-Meter Roof-Edge Setback Rule ===")
    # Pomerene Hall (ID 21, Top 16.1m, Height 16.6m)
    pomerene = mgr.rooftops[21]
    enemy_ground = [pomerene.center[0] - 25.0, pomerene.center[1], 0.0]
    perch = mgr.get_safe_perch_point(pomerene, enemy_ground)

    # Compute exact continuous distance to all edges
    min_edge_dist = min(point_segment_distance(perch[0], perch[1], e[0][0], e[0][1], e[1][0], e[1][1])[0] for e in pomerene.edges)
    print(f"  [PASS] Rooftop [{pomerene.name}] Perch: ({perch[0]:.1f}, {perch[1]:.1f}, {perch[2]:.1f})")
    print(f"         Continuous setback from all edges: {min_edge_dist:.2f}m >= 3.0m rule")
    assert min_edge_dist >= 3.0, f"Violated 3m setback rule: distance {min_edge_dist:.2f}m < 3.0m"

    # Line of sight occlusion test
    los_occluded = mgr.check_ground_los_occluded(pomerene, perch, enemy_ground)
    print(f"  [PASS] Ground muzzle LOS (z={MUZZLE_HEIGHT}m) occluded by 16.1m roof edge: {los_occluded}")
    assert los_occluded is True, "Roof edge must mathematically block line of sight from ground muzzle!"

    # -------------------------------------------------------------
    # TEST 5: Safe Drop-Down & Exit Vectors (Zero Fall Damage)
    # -------------------------------------------------------------
    print("\n=== TEST 5: Safe Drop-Down & Exit Vectors (Zero Fall Damage) ===")
    zone_center = [0.0, 0.0]
    exit_vector = mgr.get_safe_exit_vector(pomerene, zone_center, enemy_ground)

    assert exit_vector is not None, "Failed to compute exit vector"
    print(f"  [PASS] Drop Edge Position: ({exit_vector['edge_pos'][0]:.1f}, {exit_vector['edge_pos'][1]:.1f}, {exit_vector['edge_pos'][2]:.1f})")
    print(f"         Exit Vector Normal: ({exit_vector['exit_vec'][0]:.2f}, {exit_vector['exit_vec'][1]:.2f})")
    print(f"         Ground Landing Spot: ({exit_vector['ground_pos'][0]:.1f}, {exit_vector['ground_pos'][1]:.1f}, {exit_vector['ground_pos'][2]:.1f})")

    landing_walkable = mgr.is_ground_walkable(exit_vector["ground_pos"][0], exit_vector["ground_pos"][1])
    print(f"  [PASS] Ground Landing Spot Verified Walkable: {landing_walkable} (Clearance >= 2.5m)")
    assert landing_walkable is True, "Landing position must be open walkable ground"

    # -------------------------------------------------------------
    # TEST 6: Multi-Stage Climb Controller State Machine
    # -------------------------------------------------------------
    print("\n=== TEST 6: Multi-Stage Climb Controller State Machine ===")
    # Stage 1: Approach wall
    act1 = mgr.compute_climb_action([150.0, -300.0, 0.0], escape, is_grounded=True, current_energy=80.0)
    print(f"  [PASS] Stage 1 (Far from wall):   stage={act1['stage']:18s} | jump={act1['jump']} | boost={act1['boost']}")
    assert act1["stage"] == "APPROACHING_WALL" and not act1["jump"]

    # Stage 2: Ground jump at wall base
    act2 = mgr.compute_climb_action(escape.ground_approach, escape, is_grounded=True, current_energy=80.0)
    print(f"  [PASS] Stage 2 (At wall base):    stage={act2['stage']:18s} | jump={act2['jump']} | boost={act2['boost']}")
    assert act2["stage"] == "JUMPING" and act2["jump"] is True

    # Stage 3: In air jetting upward
    air_pos = [escape.climb_edge[0], escape.climb_edge[1], escape.roof_top - 5.0]
    act3 = mgr.compute_climb_action(air_pos, escape, is_grounded=False, current_energy=60.0)
    print(f"  [PASS] Stage 3 (In air jetting):  stage={act3['stage']:18s} | jump={act3['jump']} | boost={act3['boost']}")
    assert act3["stage"] == "JETTING_UP" and act3["boost"] is True

    # Stage 4: Clearing edge
    edge_pos = [escape.climb_edge[0], escape.climb_edge[1], escape.roof_top - 0.1]
    act4 = mgr.compute_climb_action(edge_pos, escape, is_grounded=False, current_energy=50.0)
    print(f"  [PASS] Stage 4 (Clearing edge):   stage={act4['stage']:18s} | jump={act4['jump']} | boost={act4['boost']}")
    assert act4["stage"] == "CLEARING_EDGE"

    # Stage 5: Perched on top (45 HP/s healing active)
    perch_pos = escape.perch_point
    act5 = mgr.compute_climb_action(perch_pos, escape, is_grounded=True, current_energy=100.0)
    print(f"  [PASS] Stage 5 (Safe perch):      stage={act5['stage']:18s} | move={act5['move']} (45 HP/s healing active!)")
    assert act5["stage"] == "PERCHED_REGEN" and act5["move"] == [0.0, 0.0]

    # -------------------------------------------------------------
    # TEST 7: Descent Urgency & Should Descend Evaluation
    # -------------------------------------------------------------
    print("\n=== TEST 7: Rooftop Descent Decision Engine ===")
    zone_safe = {"center": [pomerene.center[0], pomerene.center[1]], "radius": 380.0, "next_center": [pomerene.center[0], pomerene.center[1]], "next_radius": 300.0}
    descend, reason = mgr.should_descend(perch_pos, zone_safe, current_hp=500.0, max_hp=1400.0, time_since_combat=2.0)
    print(f"  [PASS] Wounded in safe zone: descend={descend}, reason={reason}")
    assert not descend, "Must hold perch while wounded to regenerate HP"

    # Healed to full
    descend_healed, reason_healed = mgr.should_descend(perch_pos, zone_safe, current_hp=1380.0, max_hp=1400.0, time_since_combat=10.0)
    print(f"  [PASS] Full HP restored: descend={descend_healed}, reason={reason_healed}")
    assert descend_healed and reason_healed == "RESET_COMPLETE_DESCENT"

    # Zone closing over building
    zone_far = {"center": [0.0, 0.0], "radius": 140.0, "next_center": [-300.0, 0.0], "next_radius": 50.0}
    descend_zone, reason_zone = mgr.should_descend(perch_pos, zone_far, current_hp=500.0, max_hp=1400.0, time_since_combat=2.0)
    print(f"  [PASS] Safe zone collapsing: descend={descend_zone}, reason={reason_zone}")
    assert descend_zone and reason_zone == "ZONE_CLOSING_DESCENT"

    # -------------------------------------------------------------
    # TEST 8: Rigorous Latency Benchmark (< 1.5 ms constraint)
    # -------------------------------------------------------------
    print("\n=== TEST 8: Rigorous Latency Benchmark (< 1.5 ms execution budget) ===")
    N = 1000
    np.random.seed(42)
    test_positions = np.random.uniform(-250.0, 250.0, size=(N, 2))
    test_enemies = np.random.uniform(-250.0, 250.0, size=(N, 2))

    times_escape = []
    for i in range(N):
        mpos = [float(test_positions[i, 0]), float(test_positions[i, 1]), 0.0]
        epos = [float(test_enemies[i, 0]), float(test_enemies[i, 1]), 0.0]
        t_start = time.perf_counter()
        mgr.find_nearest_rooftop_escape(mpos, epos, energy=80.0)
        times_escape.append((time.perf_counter() - t_start) * 1000.0)

    times_perch = []
    for i in range(N):
        r_idx = i % len(mgr.rooftop_list)
        epos = [float(test_enemies[i, 0]), float(test_enemies[i, 1]), 0.0]
        t_start = time.perf_counter()
        mgr.get_safe_perch_point(mgr.rooftop_list[r_idx], epos)
        times_perch.append((time.perf_counter() - t_start) * 1000.0)

    times_exit = []
    for i in range(N):
        r_idx = i % len(mgr.rooftop_list)
        epos = [float(test_enemies[i, 0]), float(test_enemies[i, 1]), 0.0]
        t_start = time.perf_counter()
        mgr.get_safe_exit_vector(mgr.rooftop_list[r_idx], [0.0, 0.0], epos)
        times_exit.append((time.perf_counter() - t_start) * 1000.0)

    times_on_roof = []
    for i in range(N):
        mpos = [float(test_positions[i, 0]), float(test_positions[i, 1]), 15.0]
        t_start = time.perf_counter()
        mgr.is_on_rooftop(mpos)
        times_on_roof.append((time.perf_counter() - t_start) * 1000.0)

    benchmarks = [
        ("find_nearest_rooftop_escape", times_escape),
        ("get_safe_perch_point", times_perch),
        ("get_safe_exit_vector", times_exit),
        ("is_on_rooftop", times_on_roof)
    ]

    print(f"\n{'Operation':30s} | {'Mean (ms)':10s} | {'Median (ms)':12s} | {'95th % (ms)':12s} | {'Max (ms)':10s} | {'Status':8s}")
    print("-" * 92)
    for name, t_list in benchmarks:
        arr = np.array(t_list)
        mean_v = float(np.mean(arr))
        med_v = float(np.median(arr))
        p95_v = float(np.percentile(arr, 95))
        max_v = float(np.max(arr))
        passed = p95_v < 1.5
        status_str = "PASS" if passed else "FAIL"
        print(f"{name:30s} | {mean_v:10.4f} | {med_v:12.4f} | {p95_v:12.4f} | {max_v:10.4f} | {status_str:8s}")
        assert passed, f"Latency cap exceeded for {name}: {p95_v:.4f} ms >= 1.5 ms"

    print("-" * 92)
    print("ALL TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 72)


if __name__ == "__main__":
    run_tests()
