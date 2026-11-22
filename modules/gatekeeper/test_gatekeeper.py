#!/usr/bin/env python3
"""
Standalone Test Suite for CHOKE (Zone Gatekeeper, Macro Strategist, Supply Drop Snatcher & Sound Cluster Deflector).
Standard library + numpy only.
Tests:
1. Chokepoint & Funnel Precomputation across all zone phases.
2. Rational Rotation Urgency (NO premature 30s panic rotation; triggers when time_to_shrink <= eta + 4.0s).
3. Supply Drop Ambush Timing (25m ambush with LOS, let bots brawl in 8m ring, clean up survivor).
4. Sound Cluster Deflection (Triangulate death 600m & explosion 420m, wait for bot clusters to eliminate each other).
5. Ohio Stadium Pinch Avoidance & Safe Bypass Routes.
6. The Gatekeeper Ambush Doctrine (early rotation, cover positioning, pre-aim).
7. Endgame Survival Engine (< 5 alive: HP preservation, fight avoidance, high ground / deep cover).
8. Execution Time Constraint (< 2.0 ms).
"""

import math
import sys
import time
from typing import Dict, List, Any
from gatekeeper import (
    ZoneGatekeeper,
    Chokepoint,
    evaluate_drop_objective,
    update_acoustics,
    should_wait_for_cluster,
    get_deflected_rotation_target
)


def test_chokepoints_detection(gatekeeper: ZoneGatekeeper):
    print("\n=== TEST 1: Chokepoint & Funnel Precomputation ===")
    center = (41.3, -72.8)
    for r in [380.0, 240.0, 140.0, 60.0]:
        t0 = time.perf_counter()
        chokes = gatekeeper.find_chokepoints(r + 100.0, center, r)
        dt = (time.perf_counter() - t0) * 1000

        assert len(chokes) > 0, f"Expected chokepoints at radius {r}m"
        top_choke = chokes[0]
        assert 3.5 <= top_choke.width <= 38.0
        assert top_choke.traffic_score > 0
        assert gatekeeper.is_walkable(top_choke.ambush_pos[0], top_choke.ambush_pos[1])
        dist_ambush = math.hypot(top_choke.ambush_pos[0] - center[0], top_choke.ambush_pos[1] - center[1])
        assert dist_ambush <= r + 2.0
        print(f"  Radius {r:5.1f}m: {len(chokes):2d} chokepoints in {dt:.2f} ms | Top funnel: {top_choke}")


def test_rational_rotation_urgency(gatekeeper: ZoneGatekeeper):
    print("\n=== TEST 2: Gatekeeper Rational Rotation Urgency Doctrine ===")
    zone_obs = {
        "center": [41.3, -72.8],
        "radius": 560.0,
        "next_center": [41.3, -72.8],
        "next_radius": 380.0,
        "shrink_start": 60.0,
        "shrink_end": 110.0,
        "t": 30.0
    }

    # Bot placed at 480m (100m outside the safe circle 380m)
    # Traversal ETA ~ 12.8s. Time to shrink = 30s.
    # Rule: Only rotate if time_to_shrink <= traversal_eta + 4.0s (30 > 12.8 + 4.0 = 16.8) -> DO NOT PANIC!
    bot_pos = (41.3 + 480.0, -72.8)
    needs_rot_30s, eta = gatekeeper.evaluate_rotation_urgency(bot_pos, zone_obs, my_speed=11.7, current_time=30.0)
    assert needs_rot_30s is False, "Gatekeeper must NOT panic rotate 30s prematurely when farming is critical!"
    print(f"  Outer bot (dist=480m, safe=380m) at t=30.0s: ETA={eta:.1f}s -> No premature panic rotation: needs_rot={needs_rot_30s} (PASS)")

    # At t=45.0s (15s before shrink, time_to_shrink = 15.0s <= eta + 4.0s = 16.8s) -> MUST ROTATE NOW!
    needs_rot_45s, _ = gatekeeper.evaluate_rotation_urgency(bot_pos, zone_obs, my_speed=11.7, current_time=45.0)
    assert needs_rot_45s is True, "Gatekeeper must trigger rotation on schedule when time_left <= eta + 4.0s!"
    print(f"  Same bot at t=45.0s (15s before shrink): time_left=15s <= ETA+4.0s -> Timely rotation: needs_rot={needs_rot_45s} (PASS)")

    # Bot already safely inside safe circle (at 200m)
    inner_bot = (41.3 + 200.0, -72.8)
    needs_rot_safe, _ = gatekeeper.evaluate_rotation_urgency(inner_bot, zone_obs, my_speed=11.7, current_time=45.0)
    assert needs_rot_safe is False, "Bot already safe inside circle should not panic rotate"
    print(f"  Inner bot (dist=200m, safe=380m) at t=45.0s: needs_rot={needs_rot_safe} (PASS)")


def test_supply_drop_ambush_timing(gatekeeper: ZoneGatekeeper):
    print("\n=== TEST 3: Supply Drop Ambush Timing (25m Ambush with LOS, Clean Up Survivor) ===")
    drop_test = {
        "id": 20000,
        "pos": [100.0, 100.0, 0.0],
        "radius": 8.0,
        "lands_in": 25.0,
        "progress": 0,
        "holder": None
    }
    my_bot = (125.0, 100.0)  # 25m from drop

    # Sub-case A: Beacon just appeared at t=70s (lands_in = 25.0s). Bot is already close (25m).
    # Expected: None (do not wait idle 25s early, continue farming).
    targ_a = gatekeeper.evaluate_drop_objective(my_bot, [drop_test], [], 70.0)
    print(f"  Sub-case A: lands_in=25.0s, dist=25m -> target: {targ_a} (Expected: None - continue farming)")
    assert targ_a is None, "Should not intercept 25s early when close"

    # Sub-case B: Predictive Ambush Setup:
    # Bot is at 100m away (200, 100). Ambush spot at (125, 100) (25m from drop).
    # Distance to ambush spot = 75m. Traversal ETA = (75 / 11.7) * 1.15 = 7.37s.
    # When lands_in = 9.0s, lands_in - 2.0 = 7.0s <= traversal_eta (7.37s).
    # Expected: path toward 25m ambush spot (125.0, 100.0) to set up with direct LOS!
    far_bot_100m = (200.0, 100.0)
    drop_test["lands_in"] = 9.0
    targ_b = gatekeeper.evaluate_drop_objective(far_bot_100m, [drop_test], [], 86.0)
    d_ambush = math.hypot(targ_b[0] - 100.0, targ_b[1] - 100.0)
    print(f"  Sub-case B (Predictive Ambush Setup): lands_in=9.0s, dist=100m -> target: {targ_b}, dist_to_drop={d_ambush:.1f}m")
    assert targ_b != (100.0, 100.0), "Must NOT rush to the 8m ring center early!"
    assert 22.0 <= d_ambush <= 28.0, f"Must establish ambush position ~25m away (actual: {d_ambush:.1f}m)!"
    assert gatekeeper.has_los(targ_b[0], targ_b[1], 100.0, 100.0), "Ambush spot must have direct line of sight to drop!"

    # Sub-case C: Drop landed (lands_in = 0.0s), other bots brawling inside 8m ring!
    # Enemy in 8m ring: let other bots fight inside 8m ring, shoot and clean up survivor!
    drop_test["lands_in"] = 0.0
    enemy_in_ring = {"id": 99, "pos": [102.0, 100.0, 0.0], "hp": 480.0}
    targ_c = gatekeeper.evaluate_drop_objective(my_bot, [drop_test], [enemy_in_ring], 96.0)
    dist_c = math.hypot(targ_c[0] - 100.0, targ_c[1] - 100.0)
    print(f"  Sub-case C (Ring Contested): Enemy in ring -> target: {targ_c}, dist_to_drop={dist_c:.1f}m")
    assert targ_c != (100.0, 100.0), "Must NOT step into the contested 8m ring killzone!"
    assert 22.0 <= dist_c <= 28.0, "Must hold 25m ambush position while ring is contested!"

    # Defensive aim locks onto the survivor/enemy inside the ring
    aim_yaw, _ = gatekeeper.get_drop_defensive_aim(targ_c, drop_test, [enemy_in_ring])
    enemy_yaw = math.atan2(enemy_in_ring["pos"][1] - targ_c[1], enemy_in_ring["pos"][0] - targ_c[0])
    dev = abs(math.atan2(math.sin(aim_yaw - enemy_yaw), math.cos(aim_yaw - enemy_yaw)))
    print(f"    Defensive aim locks on survivor in ring: dev={math.degrees(dev):.2f}° (PASS)")
    assert math.degrees(dev) < 0.1, "Aim must lock directly onto enemy in ring to clean up survivor!"

    # Sub-case D: Drop landed (lands_in = 0.0s), ring is clear of enemies!
    # Survivor eliminated! Now step in to capture:
    targ_d = gatekeeper.evaluate_drop_objective(my_bot, [drop_test], [], 97.0)
    print(f"  Sub-case D (Ring Clear): No enemies -> target: {targ_d} (Navigate into 8m ring to capture!)")
    assert targ_d == (100.0, 100.0), "Should navigate into 8m capture ring once clear of enemies!"

    # Sub-case E: Bot inside 8m ring (4m away) -> holds position for 4.0s capture!
    inside_bot = (104.0, 100.0)
    targ_e = gatekeeper.evaluate_drop_objective(inside_bot, [drop_test], [], 98.0)
    print(f"  Sub-case E: Inside 8m ring -> target: {targ_e} (Hold ring for 4.0s capture!)")
    assert targ_e == inside_bot, "Should hold within 8.0m capture radius for 4.0s!"


def test_sound_cluster_deflection(gatekeeper: ZoneGatekeeper):
    print("\n=== TEST 4: Sound Cluster Deflection (Triangulate Death 600m & Explosion 420m) ===")
    gatekeeper.sound_deflector.clusters.clear()

    # Bot at (0, 0). Sound 1: death sound (600m carry distance) from East at mid distance (110m)
    # Sound 2: explosion sound (420m carry distance) from East at mid distance (110m)
    sounds_sim = [
        {"kind": "death", "bearing": 0.0, "dist": "mid", "age": 0.1},
        {"kind": "explosion", "bearing": 0.05, "dist": "mid", "age": 0.2}
    ]
    gatekeeper.update_acoustics(sounds_sim, (0.0, 0.0), current_time=30.0)
    active_clusters = gatekeeper.get_active_sound_clusters(current_time=30.0)
    print(f"  Ingested sounds -> Found {len(active_clusters)} active conflict cluster(s):")
    assert len(active_clusters) >= 1, "Must triangulate and identify active sound cluster!"
    cluster = active_clusters[0]
    cl_center = cluster["center"]
    print(f"    Cluster Center: {cl_center}, Deaths: {cluster['death_count']}, Explosions: {cluster['explosion_count']}, Weight: {cluster['weight']}")
    assert cluster["death_count"] >= 1, "Must record death sound in cluster"
    assert cluster["explosion_count"] >= 1, "Must record explosion sound in cluster"
    assert math.hypot(cl_center[0] - 110.0, cl_center[1] - 0.0) < 25.0, "Cluster must center near triangulated coordinates ~110m East"

    # Sub-case 4A: Target inside sound cluster, time left=30s -> should_wait: True
    zone_sound_test = {
        "center": [110.0, 0.0],
        "radius": 560.0,
        "next_center": [110.0, 0.0],
        "next_radius": 380.0,
        "shrink_start": 60.0,
        "t": 30.0
    }
    should_wait, staging_pos = gatekeeper.should_wait_for_cluster((0.0, 0.0), (110.0, 0.0), zone_sound_test, current_time=30.0)
    print(f"  Sub-case 4A: Target inside sound cluster, time left=30s -> should_wait: {should_wait}, staging_pos: {staging_pos}")
    assert should_wait is True, "Must wait for bot clusters to eliminate each other before rotating in!"
    assert staging_pos is not None
    d_to_cl = math.hypot(staging_pos[0] - cl_center[0], staging_pos[1] - cl_center[1])
    print(f"    Staging setback distance from cluster: {d_to_cl:.1f}m (Target: ~85m)")
    assert 70.0 <= d_to_cl <= 100.0, f"Staging position must maintain safe setback buffer (~85m) from cluster (actual: {d_to_cl:.1f}m)!"

    # Sub-case 4B: Zone urgency forces rotation (time_left = 3s <= eta + 4s):
    zone_sound_urgent = dict(zone_sound_test, shrink_start=33.0)  # time_to_shrink = 3.0s <= 4.0s
    should_wait_urg, _ = gatekeeper.should_wait_for_cluster((0.0, 0.0), (110.0, 0.0), zone_sound_urgent, current_time=30.0)
    assert should_wait_urg is False, "Cannot wait when gas is encroaching"
    deflected_tgt = gatekeeper.get_deflected_rotation_target((0.0, 0.0), (110.0, 0.0), zone_sound_urgent, current_time=30.0)
    d_deflect_to_cl = math.hypot(deflected_tgt[0] - cl_center[0], deflected_tgt[1] - cl_center[1])
    print(f"  Sub-case 4B: Urgent rotation -> Deflected Waypoint: {deflected_tgt}, buffer from cluster: {d_deflect_to_cl:.1f}m")
    assert d_deflect_to_cl >= 65.0, "Deflected waypoint must route around the cluster deathball!"


def test_stadium_pinch_bypass(gatekeeper: ZoneGatekeeper):
    print("\n=== TEST 5: Ohio Stadium Pinch Avoidance & Safe Bypass ===")
    west_bot = (-360.0, 100.0)
    south_goal = (41.3, -72.8)
    north_goal = (41.3, 200.0)

    # South goal -> selects South bypass
    pinched_s, bypass_s = gatekeeper.is_stadium_pinch(west_bot, south_goal)
    assert pinched_s is True, "Must detect stadium dead end"
    assert bypass_s == gatekeeper.stadium_south_bypass
    print(f"  West bot to South goal: Pinch={pinched_s}, Selected Waypoint={bypass_s} (PASS)")

    # North goal -> selects North bypass
    pinched_n, bypass_n = gatekeeper.is_stadium_pinch(west_bot, north_goal)
    assert pinched_n is True, "Must detect stadium dead end"
    assert bypass_n == gatekeeper.stadium_north_bypass
    print(f"  West bot to North goal: Pinch={pinched_n}, Selected Waypoint={bypass_n} (PASS)")

    # Verify traversal ETA with stadium detour is elevated
    zone_obs = {
        "center": [41.3, -72.8],
        "radius": 560.0,
        "next_center": [41.3, -72.8],
        "next_radius": 380.0,
        "shrink_start": 60.0,
        "shrink_end": 110.0,
        "t": 40.0
    }
    needs_rot, eta = gatekeeper.evaluate_rotation_urgency(west_bot, zone_obs, my_speed=11.7, current_time=40.0)
    assert needs_rot is True, "Detoured traversal must trigger timely rotation before gas encroachment!"
    assert eta > 15.0
    print(f"  Pinch traversal ETA: {eta:.1f}s at t=40.0s -> Timely rotation triggered: {needs_rot} (PASS)")


def test_gatekeeper_ambush_doctrine(gatekeeper: ZoneGatekeeper):
    print("\n=== TEST 6: Gatekeeper Ambush Doctrine Simulation ===")
    center = (41.3, -72.8)
    chokes = gatekeeper.find_chokepoints(380.0, center, 240.0)
    assert len(chokes) > 0
    top_choke = chokes[0]

    ambush_pos = gatekeeper.select_gatekeep_position((41.3, -72.8), top_choke)
    yaw, pitch = gatekeeper.get_ambush_aim(ambush_pos, top_choke)

    assert gatekeeper.is_walkable(ambush_pos[0], ambush_pos[1])
    assert pitch == 0.0

    # Incoming enemy at throat of chokepoint
    enemy_pos = top_choke.pos
    enemy_yaw = math.atan2(enemy_pos[1] - ambush_pos[1], enemy_pos[0] - ambush_pos[0])
    angle_error = abs(math.atan2(math.sin(yaw - enemy_yaw), math.cos(yaw - enemy_yaw)))
    print(f"  Holding ambush at: {ambush_pos}, pre-aim yaw: {math.degrees(yaw):.1f}°")
    print(f"  Enemy approaching alley: {enemy_pos}, angle error: {math.degrees(angle_error):.2f}°")
    assert angle_error < 0.1, "Pre-aim error must be < 0.1 rad for instant zero-slew kill"
    print("  Zero-slew ambush trigger: CONFIRMED (PASS)")


def test_endgame_survival_engine(gatekeeper: ZoneGatekeeper):
    print("\n=== TEST 7: Endgame Survival Engine (< 5 alive) ===")
    # State identification
    assert gatekeeper.is_endgame(alive_count=4, zone_radius=60.0) is True
    assert gatekeeper.is_endgame(alive_count=12, zone_radius=200.0) is False

    # Avoid optional fights
    enemy_far = {"id": 5, "hp": 1000.0, "weapons": ["gun"]}
    assert gatekeeper.should_avoid_combat(alive_count=3, my_hp=1400.0, enemy=enemy_far, enemy_dist=75.0) is True
    assert gatekeeper.should_avoid_combat(alive_count=3, my_hp=1400.0, enemy=enemy_far, enemy_dist=12.0) is False
    print("  Fight avoidance logic (alive <= 5): CONFIRMED (PASS)")

    # High ground / deep cover selection in final 60m circle
    final_zone = {"center": [41.3, -72.8], "radius": 60.0, "next_radius": 60.0}
    endgame_pos = gatekeeper.get_endgame_position((41.3, -72.8), final_zone, alive_count=3)

    assert gatekeeper.is_walkable(endgame_pos[0], endgame_pos[1])
    d_center = math.hypot(endgame_pos[0] - 41.3, endgame_pos[1] - (-72.8))
    assert d_center <= 52.0
    assert d_center >= 10.0
    print(f"  Selected final defensive position: {endgame_pos}, dist to center={d_center:.1f}m (PASS)")


def test_execution_time_constraint(gatekeeper: ZoneGatekeeper):
    print("\n=== TEST 8: Execution Time Constraint Benchmark (< 2.0 ms) ===")
    zone_obs = {
        "center": [41.3, -72.8],
        "radius": 240.0,
        "next_center": [41.3, -72.8],
        "next_radius": 140.0,
        "shrink_start": 160.0,
        "shrink_end": 210.0,
        "t": 130.0
    }
    center = (41.3, -72.8)
    enemy = {"id": 1, "hp": 500.0, "weapons": ["gun"]}
    drop_test = {"id": 1, "pos": [100.0, 100.0, 0.0], "radius": 8.0, "lands_in": 10.0}

    # Warm up cache
    gatekeeper.find_chokepoints(240.0, center, 140.0)
    gatekeeper.get_endgame_position(center, zone_obs, 3)

    iterations = 500
    t0 = time.perf_counter()
    for i in range(iterations):
        my_pos = (41.3 + (i % 50), -72.8)
        gatekeeper.evaluate_rotation_urgency(my_pos, zone_obs, 11.7, 130.0)
        chks = gatekeeper.find_chokepoints(240.0, center, 140.0)
        gatekeeper.select_gatekeep_position(my_pos, chks, current_time=130.0)
        gatekeeper.should_wait_for_cluster(my_pos, (110.0, 0.0), zone_obs, 130.0)
        evaluate_drop_objective(my_pos, [drop_test], [], 130.0)
        gatekeeper.should_avoid_combat(3, 1400.0, enemy, 50.0)
        gatekeeper.get_endgame_position(my_pos, zone_obs, 3)
    elapsed = (time.perf_counter() - t0) * 1000
    avg_per_tick = elapsed / iterations

    print(f"  Executed {iterations} tick cycles in {elapsed:.2f} ms")
    print(f"  Average execution time per tick: {avg_per_tick:.4f} ms")
    assert avg_per_tick < 2.0, f"Violation of 2.0 ms constraint: {avg_per_tick:.4f} ms"
    print(f"  >> Performance Constraint PASSED ({avg_per_tick:.4f} ms << 2.0 ms, {(2.0 / avg_per_tick):.1f}x headroom)")


def run_all():
    print("=" * 70)
    print("CHOKE (Zone Gatekeeper & Macro Strategist) Test Suite")
    print("=" * 70)
    gk = ZoneGatekeeper()
    test_chokepoints_detection(gk)
    test_rational_rotation_urgency(gk)
    test_supply_drop_ambush_timing(gk)
    test_sound_cluster_deflection(gk)
    test_stadium_pinch_bypass(gk)
    test_gatekeeper_ambush_doctrine(gk)
    test_endgame_survival_engine(gk)
    test_execution_time_constraint(gk)
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all()
