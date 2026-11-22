#!/usr/bin/env python3
"""
Standalone Test Suite for CHOKE (Zone Gatekeeper, Macro Strategist & Supply Drop Snatcher).
Standard library + numpy only.
Tests:
1. Chokepoint & Funnel Precomputation across all zone phases.
2. Rational Rotation Urgency (NO premature 30s panic rotation; triggers when time_to_shrink <= eta + 4.0s).
3. Ohio Stadium Pinch Avoidance & Safe Bypass Routes.
4. Supply Drop Interception & Kinematics (Predictive Touchdown Arrival, Contested Ring Standoff, Landed Capture).
5. The Gatekeeper Ambush Doctrine (early rotation, cover positioning, pre-aim).
6. Endgame Survival Engine (< 5 alive: HP preservation, fight avoidance, high ground / deep cover).
7. Execution Time Constraint (< 2.0 ms).
"""

import math
import sys
import time
from typing import Dict, List, Any
from gatekeeper import ZoneGatekeeper, Chokepoint, evaluate_drop_objective


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

    # Bot far outside safe circle (at 480m)
    # Traversal ETA ~ 12.8s. Time to shrink = 30s.
    # Rule: Only rotate if time_to_shrink <= traversal_eta + 4.0s (30 > 12.8 + 4.0 = 16.8) -> DO NOT PANIC!
    outer_bot = (41.3 + 480.0, -72.8)
    needs_rot_30s, eta = gatekeeper.evaluate_rotation_urgency(outer_bot, zone_obs, my_speed=11.7, current_time=30.0)
    assert needs_rot_30s is False, "Gatekeeper must NOT panic rotate 30s prematurely when farming is critical!"
    print(f"  Outer bot (dist=480m, safe=380m) at t=30.0s: ETA={eta:.1f}s -> No premature panic rotation: needs_rot={needs_rot_30s} (PASS)")

    # At t=45.0s (15s before shrink, time_to_shrink = 15.0s <= eta + 4.0s = 16.8s) -> MUST ROTATE NOW!
    needs_rot_45s, _ = gatekeeper.evaluate_rotation_urgency(outer_bot, zone_obs, my_speed=11.7, current_time=45.0)
    assert needs_rot_45s is True, "Gatekeeper must trigger rotation on schedule when time_left <= eta + 4.0s!"
    print(f"  Same bot at t=45.0s (15s before shrink): time_left=15s <= ETA+4.0s -> Timely rotation: needs_rot={needs_rot_45s} (PASS)")

    # Bot already safely inside safe circle (at 200m)
    inner_bot = (41.3 + 200.0, -72.8)
    needs_rot_safe, _ = gatekeeper.evaluate_rotation_urgency(inner_bot, zone_obs, my_speed=11.7, current_time=45.0)
    assert needs_rot_safe is False, "Bot already safe inside circle should not panic rotate"
    print(f"  Inner bot (dist=200m, safe=380m) at t=45.0s: needs_rot={needs_rot_safe} (PASS)")


def test_stadium_pinch_bypass(gatekeeper: ZoneGatekeeper):
    print("\n=== TEST 3: Ohio Stadium Pinch Avoidance & Safe Bypass ===")
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


def test_supply_drop_interception(gatekeeper: ZoneGatekeeper):
    print("\n=== TEST 4: Supply Drop Interceptor & Kinematics ===")
    drop_test = {
        "id": 20000,
        "pos": [100.0, 100.0, 0.0],
        "radius": 8.0,
        "lands_in": 25.0,
        "progress": 0,
        "holder": None
    }
    my_bot = (120.0, 100.0)  # 20m from drop, traversal_eta = (20/11.7)*1.15 = 1.97s

    # Sub-case A: Beacon just appeared at t=70s (lands_in = 25.0s). Bot is close (20m).
    # Rule: traversal_eta >= (lands_in - 2.0) -> 1.97 >= 23.0 is False!
    # Expected: None (do not wait idle 25s early, continue farming).
    targ_a = gatekeeper.evaluate_drop_objective(my_bot, [drop_test], [], 70.0)
    print(f"  Sub-case A: lands_in=25.0s, dist=20m, ETA=1.97s -> target: {targ_a} (Expected: None)")
    assert targ_a is None, "Should not intercept 25s early when close"

    # Sub-case B: Predictive Touchdown Arrival:
    # Bot is at 100m away, traversal_eta = (100 / 11.7) * 1.15 = 9.83s.
    # When lands_in = 11.0s, lands_in - 2.0 = 9.0s <= traversal_eta (9.83s).
    # Expected: start pathing toward drop beacon to arrive ~2s before touchdown!
    far_bot_100m = (200.0, 100.0)
    drop_test["lands_in"] = 11.0
    targ_b = gatekeeper.evaluate_drop_objective(far_bot_100m, [drop_test], [], 84.0)
    print(f"  Sub-case B (Predictive Touchdown): lands_in=11.0s, dist=100m, ETA=9.83s >= lands_in-2.0 (9.0s) -> target: {targ_b}")
    assert targ_b == (100.0, 100.0), "Should path toward landing coordinate when traversal_eta >= lands_in - 2.0!"

    # Sub-case C: Drop has landed (lands_in = 0.0s), bot is outside the 8m ring (15m away).
    drop_test["lands_in"] = 0.0
    targ_c = gatekeeper.evaluate_drop_objective((115.0, 100.0), [drop_test], [], 95.0)
    print(f"  Sub-case C: lands_in=0.0s, dist=15m -> target: {targ_c} (Expected: (100.0, 100.0))")
    assert targ_c == (100.0, 100.0), "Should navigate into 8m capture ring"

    # Sub-case D: Drop has landed (lands_in = 0.0s), bot is inside 8m ring (4m away).
    # Expected: hold within 8.0m capture radius for 4.0s!
    inside_bot = (104.0, 100.0)
    targ_d = gatekeeper.evaluate_drop_objective(inside_bot, [drop_test], [], 96.0)
    print(f"  Sub-case D: lands_in=0.0s, inside ring (dist=4m) -> target: {targ_d} (Hold ring)")
    assert targ_d == inside_bot, "Should hold within 8.0m capture radius for 4.0s!"

    # Sub-case E: Contested Ring Standoff (Vulcan Failure Mode 2 Guard):
    # If an enemy mech is ALREADY inside the 8m capture ring:
    # DO NOT step into the ring (avoid the sitting duck killzone).
    # Instead, establish a 45m standoff position with line-of-sight and shoot the stationary capturing enemy!
    enemy_in_ring = {"id": 99, "pos": [102.0, 100.0, 0.0], "hp": 1400.0}
    targ_standoff = gatekeeper.evaluate_drop_objective(my_bot, [drop_test], [enemy_in_ring], 96.0)
    dist_standoff = math.hypot(targ_standoff[0] - 100.0, targ_standoff[1] - 100.0)
    print(f"  Sub-case E (Contested Ring Standoff): Enemy in 8m ring -> target: {targ_standoff}, dist_to_drop={dist_standoff:.1f}m")
    assert targ_standoff != (100.0, 100.0), "Must NOT step into the contested 8m ring!"
    assert dist_standoff > 8.0, "Standoff position must be outside the 8.0m killzone!"
    assert 40.0 <= dist_standoff <= 50.0, f"Standoff position must establish ~45m distance (actual: {dist_standoff:.1f}m)!"
    assert gatekeeper.has_los(targ_standoff[0], targ_standoff[1], 100.0, 100.0), "Standoff position must have clear line-of-sight to drop!"

    # Defensive aim while at standoff aiming at enemy in ring
    aim_yaw, _ = gatekeeper.get_drop_defensive_aim(targ_standoff, drop_test, [enemy_in_ring])
    enemy_yaw = math.atan2(enemy_in_ring["pos"][1] - targ_standoff[1], enemy_in_ring["pos"][0] - targ_standoff[0])
    dev = abs(math.atan2(math.sin(aim_yaw - enemy_yaw), math.cos(aim_yaw - enemy_yaw)))
    print(f"    Defensive aim directly at capturing enemy: dev={math.degrees(dev):.2f}° (PASS)")
    assert math.degrees(dev) < 0.1, "Defensive aim must lock onto capturing enemy in the ring!"


def test_gatekeeper_ambush_doctrine(gatekeeper: ZoneGatekeeper):
    print("\n=== TEST 5: Gatekeeper Ambush Doctrine Simulation ===")
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
    print("\n=== TEST 6: Endgame Survival Engine (< 5 alive) ===")
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
    print("\n=== TEST 7: Execution Time Constraint Benchmark (< 2.0 ms) ===")
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

    iterations = 1000
    t0 = time.perf_counter()
    for i in range(iterations):
        my_pos = (41.3 + (i % 50), -72.8)
        gatekeeper.evaluate_rotation_urgency(my_pos, zone_obs, 11.7, 130.0)
        chks = gatekeeper.find_chokepoints(240.0, center, 140.0)
        gatekeeper.select_gatekeep_position(my_pos, chks)
        gatekeeper.evaluate_drop_objective(my_pos, [drop_test], [], 130.0)
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
    test_stadium_pinch_bypass(gk)
    test_supply_drop_interception(gk)
    test_gatekeeper_ambush_doctrine(gk)
    test_endgame_survival_engine(gk)
    test_execution_time_constraint(gk)
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all()
