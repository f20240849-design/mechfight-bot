# VULCAN ADVERSARIAL RED TEAM AUDIT REPORT
**Target System**: MechRoyale Autonomous Tournament Bot (`opportunist-v1`)  
**Workspace**: `/Users/mittals/Downloads/mechfight-kit`  
**Repository**: `https://github.com/f20240849-design/mechfight-bot`  
**Auditor**: VULCAN (Adversarial Red Team & Breakage Specialist)  
**Date**: 2026-11-22  
**Status**: 🚨 **CRITICAL VULNERABILITIES IDENTIFIED — IMMEDIATE REMEDIATION MANDATED**

---

## 1. Executive Summary & Threat Matrix

A rigorous, end-to-end adversarial audit of the fully integrated codebase—incorporating ATLAS (`rooftops.py`), BALLIS (`artillery.py`), CHOKE (`gatekeeper.py`), and the core brain (`bot.py`, `nav.py`, `combat.py`, `ballistics.py`)—has been conducted.

While the baseline conformance harness (`python3 check/check.py --map-dir ./arena -- python3 -u bot.py`) returns a superficial `PASS`, **telemetry from the 300-tick real match stream proves the bot is dying in 13th place with 0 kills and 1665.0 damage taken.**

Crucially, profiling under the constraints of the tournament container (1 CPU, 512 MB RAM, strict 50 ms wall-clock decision budget) reveals multiple **fatal cross-module collisions, priority inversions, geometry blind spots, and latent latency spikes (p90 = 62.01 ms, max = 91.64 ms on host) that guarantee CFS container throttling, dropped actions, and disqualification.**

| Threat Vector | Severity | Affected Modules | Impact Summary |
| :--- | :---: | :--- | :--- |
| **Zone Perch Gas Trap** | **CRITICAL (P0)** | `bot.py`, `rooftops.py`, `gatekeeper.py` | Priority inversion: rooftop perch suppresses zone rotation. Bot sits in gas taking 60–100 DPS waiting for reset. |
| **Setback Muzzle Occlusion** | **CRITICAL (P0)** | `rooftops.py`, `artillery.py`, `ballistics.py` | 3m setback causes Lasers to clip roof lip on close targets; Mortar obstacle checker falsely detects roof floor as a wall. |
| **Energy Starvation Plunge** | **HIGH (P1)** | `rooftops.py`, `bot.py` | Initiating climbs without +30 energy reserve stalls mechs at 14m, dropping them to ground with 0 energy. |
| **1-CPU Container Latency Breach** | **HIGH (P1)** | `nav.py`, `bot.py` | p90 tick latency of 62.01 ms on host (150–280 ms on 1 CPU) due to un-cached A* and $O(W^2)$ numpy allocations in string-pulling. |
| **Protocol Schema Landmine** | **HIGH (P1)** | `ballistics.py:112` | Returns 2-tuple `(fire, default_aim)` instead of `List[bool]` when target/LOS is absent, violating `mechfight.schema.json`. |
| **Rooftop Escape Oscillation** | **MEDIUM (P2)** | `bot.py`, `rooftops.py` | Bot wasted 113 ticks in `APPROACHING_WALL` running against obstacles under laser sniper fire without ascending. |

---

## 2. Cross-Module Collision Analysis

### 2.1 ATLAS vs. CHOKE: The Rooftop Perch Gas Trap
#### The Interrogation:
> *"What happens if a rooftop escape conflicts with an urgent zone shrink? Does the bot safely abort the roof perch before the gas hits?"*

#### The Vulnerability & Failure Mechanics:
1. **FSM Priority Inversion**:
   In `bot.py` (lines 196–245), `if current_roof:` is evaluated at **Priority 0**. CHOKE's `needs_zone_rotation` is placed at **Priority 2** (`elif needs_zone_rotation:`). Consequently, whenever the bot is on a rooftop, CHOKE's zone rotation engine is **completely bypassed and silenced**.
2. **Defective Descent Evaluation (`rooftops.py:1049-1082`)**:
   ```python
   next_center = zone_info.get("next_center", zone_info.get("center", [0.0, 0.0]))
   next_radius = zone_info.get("next_radius", zone_info.get("radius", 380.0))
   dist_to_next_center = math.hypot(current_roof.center[0] - next_center[0], current_roof.center[1] - next_center[1])

   if dist_to_next_center > (next_radius - 20.0):
       return True, "ZONE_CLOSING_DESCENT"

   if current_hp >= max_hp * 0.92 and time_since_combat >= REGEN_DELAY:
       return True, "RESET_COMPLETE_DESCENT"

   return False, "HOLDING_PERCH_HEALING"
   ```
3. **The Deadlock Condition**:
   - `dist_to_next_center > (next_radius - 20.0)` only checks if the building center is outside the *future* zone radius. It completely ignores:
     - The *current* collapsing border (`zone.radius`, `zone.shrink_start`, `zone.shrink_end`).
     - Whether the bot is *currently outside the safe zone* (`not me.get("in_zone", True)`).
     - The travel time required to drop down and path to the boundary.
   - If the rooftop center is within `next_radius - 20`, but the bot took damage (e.g. `current_hp = 600`), `should_descend()` returns `False, "HOLDING_PERCH_HEALING"`.
   - **The Gas Trap Trigger**: As the zone contracts over the rooftop, zone damage inflicts 20 to 100 HP/s. In MechRoyale, taking damage resets `time_since_combat` to 0.0!
   - Because `time_since_combat` is continuously reset to 0, `time_since_combat >= REGEN_DELAY` **never evaluates to True**. The bot remains locked in `ROOFTOP_PERCH`, executing `should_fire = [False, False]` and standing motionless in the lethal gas until death.

---

### 2.2 ATLAS vs. BALLIS: Setback Lip Occlusion & 2D Grid False Positives
#### The Interrogation:
> *"From a 3m roof setback perch, does shooting a Laser or Mortar clear the roof lip, or does it hit the building?"*

#### The Geometry & Physics Breakdown:
* Mech height: $9.0\text{ m}$. Muzzle height from feet: $z_0 = z_{\text{roof}} + 7.0\text{ m}$.
* Setback distance from roof edge: $d_{\text{setback}} = 3.0\text{ m}$.
* Elevation of roof edge: $z_{\text{roof}}$.

```
   [MECH] (z_roof + 7.0m)
     | \
     |  \
     |   \
   [ROOF]|====\================[ROOF LIP] (z_roof)
     |         \                 |
     |          \                |
     |           \               |
     |            \              |  BUILDING WALL (Height H)
     |             \             |
     |              \            |
     |               \           |
     |                \          |
   [GROUND]            \========[BLIND ZONE]======[ENEMY]
```

1. **Laser Line-of-Sight Occlusion**:
   - The line from the muzzle $(x=0, z=z_{\text{roof}} + 7.0)$ through the roof edge $(x=3.0, z=z_{\text{roof}})$ has a downward slope:
     $$\frac{\Delta z}{\Delta x} = -\frac{7.0}{3.0} = -2.333 \quad (\text{elevation angle } -66.8^\circ)$$
   - The mech's mechanical pitch limit is `min_pitch = -0.7` radians ($-40.1^\circ$, slope $-0.842$).
   - Over the 3m setback, maximum pitch only lowers the beam by $3.0 \times 0.842 = 2.53\text{ m}$, passing $4.47\text{ m}$ above the lip.
   - When firing at an enemy on the ground $(z=z_{\text{ground}})$, the ray clips the roof lip whenever the enemy is closer to the building wall than:
     $$d_{\text{ground}} < \frac{d_{\text{setback}}}{z_{\text{muzzle}} - z_{\text{roof}}} \cdot (z_{\text{roof}} - z_{\text{ground}}) = \frac{3.0}{7.0} H$$
   - **For a standard 14m roof ($H=14\text{ m}$)**: Any enemy within **$6.0\text{ m}$** of the wall is in the blind spot.
   - **For a 21m roof ($H=21\text{ m}$)** with a 5m deep interior perch: Any enemy within **$15.0\text{ m}$** of the building base is in the blind spot.
   - **Failure Mode**: When BALLIS triggers Laser discharge at an enemy within this blind zone, the hitscan beam **strikes the building roof lip**, absorbs 315 damage harmlessly into the terrain, and triggers a full 1.65 s cooldown penalty.

2. **Mortar Obstacle Clearance False Positive**:
   - In `artillery.py` line 291–298:
     ```python
     if nav_mesh and hasattr(nav_mesh, "world_to_grid") and hasattr(nav_mesh, "grid"):
         check_x = lx + min_wall_clearance * math.cos(yaw)
         check_y = ly + min_wall_clearance * math.sin(yaw)
         col, row = nav_mesh.world_to_grid(check_x, check_y)
         if nav_mesh.grid[row, col] == 1:
             return False # Wall directly in front of muzzle!
     ```
   - **The Bug**: `nav_mesh.grid` is a **2D ground projection** (where all building footprints are mapped as `1 = solid wall`).
   - When the mech is standing on a rooftop, 4m in front of the muzzle along `yaw` is still on the roof surface. However, `nav_mesh.grid[row, col]` evaluates to `1`!
   - `can_clear_obstacles()` assumes a wall is blocking the muzzle and **refuses to fire mortar shells from rooftops**, disabling artillery fire entirely while perched.

3. **Low-Arc Mortar Self-Detonation**:
   - If `IndirectParabolicMortarSolver.solve_low_arc()` is commanded against a target below the building, downward pitch trajectories passing within 3m of the roof edge risk detonating on rooftop parapets or antennae.
   - Detonation within the 7m splash radius inflicts **80 self-damage**, **16 m/s knockback** (launching the mech off the building), and **45 energy concussion drain**.

---

### 2.3 Energy Starvation: The Jump-Jet Plunge
#### The Interrogation:
> *"What happens if jump-jetting leaves us at 0 energy when cornered?"*

#### The Mechanics of Failure:
1. **Missing Reserve Check in `rooftops.py:668`**:
   - `TACTICAL_SPECS.md` line 54 mandates: `Only attempt climbs if energy >= E_required + 30`.
   - In `rooftops.py` line 668, ATLAS implemented:
     ```python
     energy_cost = self.calculate_climb_energy(h_diff)
     if energy < energy_cost:
         continue
     ```
   - For a 15m building, `energy_cost` is $\approx 35.7$ energy ($15.0$ jump + $20.7$ jet).
2. **The Stalling Sequence**:
   - If the bot initiates an escape with 36.0 energy, it consumes 15 energy on jump initiation (`jump: True`), leaving 21.0 energy.
   - Jet acceleration rises at $10\text{ m/s}$ max rise speed. Over $1.4\text{ s}$, jet drain ($15\text{ energy/s}$) consumes the remaining 21.0 energy.
   - Energy drops to **0.0** when the mech's feet are at $z = z_{\text{roof}} - 0.5\text{ m}$.
   - Upward thrust cuts out instantly. Gravity ($22\text{ m/s}^2$) halts upward momentum and plunges the mech back down 14.5m to the street.
3. **The Ground Ambush Pinata**:
   - The bot lands at the base of the wall with **0 energy**.
   - `energy_regen_delay` imposes a 0.6 s lockout before regeneration begins.
   - Boost sprint is disabled (`boost_drain: 17.0 energy/s`).
   - Jump/jetting is disabled (`jump_cost: 15.0`).
   - Ground speed is hard-clamped to walk speed ($11.7\text{ m/s}$).
   - Transverse combat strafing cannot displace the mech fast enough without boost. The cornered bot is annihilated in $< 2.0\text{ s}$.

---

## 3. 1-CPU Container Sandbox Profiling

```
================================================================================
            CONTAINER PERFORMANCE AUDIT (Target: 1 CPU, 512 MB RAM)
================================================================================
METRIC / SUBSYSTEM              HOST (Bare-Metal)       1-CPU CONTAINER (EST.)
--------------------------------------------------------------------------------
Hello Handshake Initialization  8,180.2 ms              18,500 ms (Near 20s Cap)
Memory Profile (Active Heap)    17.64 MB                ~22.0 MB (Cap: 512 MB)
Memory Profile (Peak Heap)      18.80 MB                ~24.5 MB (Cap: 512 MB)
Resident Process RSS            97.0 MB                 ~110.0 MB
Tick Latency (p50)              4.47 ms                 12.5 ms
Tick Latency (p90)              62.01 ms [FAIL]         155.0 ms [BREACH]
Tick Latency (p99)              82.01 ms [FAIL]         210.0 ms [BREACH]
Tick Latency (Max Observed)     91.64 ms [FAIL]         275.0 ms [BREACH]
================================================================================
```

### 3.1 Root Causes of Latency Budget Breaches
1. **$O(W^2)$ String-Pulling with Numpy Allocation in `nav.py:179-189`**:
   - In `NavMesh.smooth_path()`, every waypoint iterates backwards through all forward nodes, calling `has_los()`.
   - Each `has_los()` call allocates two dynamic numpy arrays via `np.linspace(..., dtype=int)` and executes boolean slice indexing `self.grid[rows, cols] == 1`.
   - A single 50-waypoint path generates over **2,400 heap-allocated numpy array objects per tick**, triggering Python's cyclic GC and causing 30–50 ms execution spikes.
2. **Brute-Force Cover Spot Search in `nav.py:192-216`**:
   - `find_cover_spot()` evaluates a $41 \times 41$ grid ($1,681$ iterations) with radial distance checks and line-of-sight raycasts. Benchmarked at **17.7 ms to 38.8 ms** standalone on host.
3. **Uncached A* Recomputation**:
   - `bot.py:413` re-runs `self.nav.plan_path((x, y), nav_target)` every single tick (10 times per second), even when moving toward a stationary waypoint.
4. **The `ROOFTOP_ESCAPE` Oscillation Loop**:
   - In the 300-tick match stream, the bot triggered `ROOFTOP_ESCAPE` for **129 ticks** (43% of the game).
   - In **113 of those ticks**, it was stuck in `stage == "APPROACHING_WALL"`. It never climbed because line-of-sight pathing around obstacles kept redirecting the ground approach vector while an enemy sniper engaged from range.

---

## 4. Protocol Conformance & Schema Landmine

1. **Schema Violation Bug in `ballistics.py:112`**:
   - `decide_weapon_triggers()` returns `fire, default_aim` (a 2-tuple) at line 112 when `target is None` or `not has_los`.
   - At line 158, it returns `fire` (a list).
   - If line 112 is reached and passed to `action["fire"]`, `mechfight.schema.json` validation fails immediately (`fire` requires `{"type": "array", "items": {"type": "boolean"}, "minItems": 2, "maxItems": 2}`).
2. **File System Conformance**:
   - Full audit of file operations confirms: `open()` is only invoked on `/tmp/match_telemetry.jsonl` (writable 64MB tmpfs). No unauthorized writes outside `/tmp` occur.

---

## 5. Mandatory Actionable Remediation Plan

To prevent immediate disqualification and tournament elimination, the following patches must be applied immediately:

### P0 Fix 1: Priority Resolution in `bot.py`
* Elevate `needs_zone_rotation` above `current_roof`. If zone shrink is imminent ($< 15\text{ s}$) or bot is outside the zone, force immediate rooftop descent via `get_safe_exit_vector()` regardless of HP or combat reset status.

### P0 Fix 2: Patch `should_descend()` in `rooftops.py`
* Add explicit checks for `zone_info.get("shrink_start")`, `zone_info.get("shrink_end")`, `current_time`, and `me.get("in_zone")`.
* If taking gas damage (`is_outside == True`), immediately descend toward the safe zone center.

### P0 Fix 3: 3D Roof Clearance in `artillery.py`
* Modify `can_clear_obstacles()` to ignore `nav_mesh.grid` if the launch height $z > 10.0\text{ m}$ (on rooftop).
* Calculate geometric clearance over the roof edge: ensure target line-of-sight pitch is steeper than $-\arctan(7.0 / d_{\text{setback}})$ before discharging weapons from setbacks.

### P1 Fix 4: Energy Reserve Guard in `rooftops.py`
* In `find_nearest_rooftop_escape()` and `compute_climb_action()`, require `energy >= energy_cost + 30.0`. If energy is below reserve, abort rooftop climbing and fallback to ground kiting/cover.

### P1 Fix 5: Replace Numpy Raycasting & Add Waypoint Caching in `nav.py`
* Replace `np.linspace` in `has_los()` with a zero-allocation integer Bresenham / DDA algorithm.
* Cache `self.current_waypoints` in `bot.py`; only replan A* if the target coordinates deviate by $> 8.0\text{ m}$.

### P1 Fix 6: Resolve `ballistics.py` Return Type
* Change line 112 in `ballistics.py` to `return fire` (`[False, False]`).

---
*VULCAN Audit Complete. All failure modes and cross-module collisions cataloged. Awaiting engineering remediations before deployment.*
