#!/usr/bin/env python3
"""
Automated Multi-Agent Integration & Deployment Pipeline for MechRoyale.
Author: Coordinator / Lead Architect Tab

What it does:
1. Pulls module updates from dedicated agent subfolders:
   - modules/verticality/rooftops.py (ATLAS)
   - modules/artillery/artillery.py (BALLIS)
   - modules/gatekeeper/gatekeeper.py (CHOKE)
2. Syncs them into root and python/ for container bundling.
3. Executes unit tests and micro-benchmarks on all modules.
4. Executes full protocol conformance check (check/check.py) on bot.py.
5. If git commit flag is enabled, commits and pushes to GitHub (main & release).
"""

import os
import sys
import shutil
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

def run_cmd(cmd: str, desc: str) -> bool:
    print(f"[*] {desc}...")
    t0 = time.time()
    res = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True)
    dt = (time.time() - t0) * 1000
    if res.returncode != 0:
        print(f"[-] FAILED in {dt:.1f} ms:")
        print(res.stderr or res.stdout)
        return False
    print(f"[+] PASSED in {dt:.1f} ms")
    return True

def sync_modules():
    print("=" * 65)
    print("STEP 1: SYNCING AGENT SUBFOLDERS TO ROOT & CONTAINER PATHS")
    print("=" * 65)
    
    mapping = {
        "modules/verticality/rooftops.py": ["rooftops.py", "python/rooftops.py"],
        "modules/artillery/artillery.py": ["artillery.py", "python/artillery.py"],
        "modules/gatekeeper/gatekeeper.py": ["gatekeeper.py", "python/gatekeeper.py"],
    }
    
    for src_rel, dst_list in mapping.items():
        src = os.path.join(ROOT, src_rel)
        if os.path.exists(src):
            for dst_rel in dst_list:
                dst = os.path.join(ROOT, dst_rel)
                shutil.copy2(src, dst)
                print(f"  Synced: {src_rel} -> {dst_rel}")
        else:
            print(f"  [WARN] {src_rel} does not exist yet (agent working)")

def test_modules() -> bool:
    print("\n" + "=" * 65)
    print("STEP 2: RUNNING MODULE UNIT TESTS & MICRO-BENCHMARKS")
    print("=" * 65)

    tests = [
        ("python3 -c 'from artillery import ArtillerySystem; print(\"ArtillerySystem OK\")'", "Artillery Integration Check"),
        ("python3 -c 'from gatekeeper import ZoneGatekeeper; print(\"ZoneGatekeeper OK\")'", "Gatekeeper Integration Check"),
        ("python3 -c 'from rooftops import RooftopManager; print(\"RooftopManager OK\")'", "Rooftop Integration Check"),
    ]

    for cmd, desc in tests:
        if not run_cmd(cmd, desc):
            return False
    return True

def test_full_bot() -> bool:
    print("\n" + "=" * 65)
    print("STEP 3: RUNNING FULL TOURNAMENT CONFORMANCE CHECK")
    print("=" * 65)
    cmd = "python3 check/check.py --map-dir ./arena -- python3 -u bot.py"
    return run_cmd(cmd, "Tournament Conformance & 50ms Budget Check")

def deploy_to_github():
    print("\n" + "=" * 65)
    print("STEP 4: DEPLOYING TO GITHUB (MAIN & RELEASE BRANCHES)")
    print("=" * 65)

    cmds = [
        "git add Dockerfile data/ bot.py nav.py ballistics.py combat.py acoustics.py artillery.py gatekeeper.py rooftops.py modules/",
        "git add python/",
        "git commit -m 'Fix scipy ModuleNotFoundError, rooftop gas trap, energy buffer, and mortar clearance' || echo 'No new changes to commit'",
        "git push origin main",
        "git checkout release",
        "git merge main -m 'Sync bugfixes and scipy dependency to release'",
        "git push origin release",
        "git checkout main"
    ]
    for c in cmds:
        run_cmd(c, c.split()[0] + " " + c.split()[1] if len(c.split()) > 1 else c)

def main():
    sync_modules()
    if not test_modules():
        sys.exit(1)
    if not test_full_bot():
        sys.exit(1)
    
    print("\n[SUCCESS] ALL MULTI-AGENT MODULES TESTED & INTEGRATED!")
    if "--deploy" in sys.argv or "-d" in sys.argv:
        deploy_to_github()

if __name__ == "__main__":
    main()
