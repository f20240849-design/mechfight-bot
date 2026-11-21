#!/usr/bin/env python3
"""
Precompute a 2D occupancy & walkability grid from arena/v2/world.json and arena/v2/terrain.i16.

Rules implemented (from protocol.md §7 & §11):
1. Terrain slope: max_slope_deg = 50. Cells steeper than 50 degrees are non-traversable.
2. Solid prisms: All buildings and stadium walls occupy space from ground to top.
3. Elevated bridges: Bridges with vertical clearance >= 9.0m (mech height) above ground
   can be walked under (not treated as ground obstacles).
4. C-Space inflation: Mech radius is 2.5m. Center cannot be within 2.5m of any wall.

Outputs:
- arena/v2/occupancy.bin: 724 x 1070 raw uint8 array (0=walkable, 1=solid obstacle, 2=near wall buffer)
- arena/v2/occupancy_meta.json: metadata and coordinate mapping formulas
- arena/v2/occupancy_map.png: high-resolution visual occupancy map
"""
import json
import os
import sys
import numpy as np
import scipy.ndimage as ndi
from PIL import Image, ImageDraw

def build_grid(kit_dir: str):
    world_path = os.path.join(kit_dir, "arena", "v2", "world.json")
    terrain_path = os.path.join(kit_dir, "arena", "v2", "terrain.i16")
    output_bin = os.path.join(kit_dir, "arena", "v2", "occupancy.bin")
    output_meta = os.path.join(kit_dir, "arena", "v2", "occupancy_meta.json")
    output_png = os.path.join(kit_dir, "arena", "v2", "occupancy_map.png")

    print(f"Reading world geometry: {world_path}")
    with open(world_path, "r") as f:
        world = json.load(f)

    tinfo = world["terrain"]
    min_x = float(tinfo["minX"])
    min_y = float(tinfo["minY"])
    step = float(tinfo["step"])
    nx = int(tinfo["nx"])
    ny = int(tinfo["ny"])

    print(f"Grid dimensions: {nx} x {ny} (step={step}m, minX={min_x}, minY={min_y})")

    # 1. Load terrain elevations
    print(f"Reading terrain elevations: {terrain_path}")
    with open(terrain_path, "rb") as f:
        terrain = np.frombuffer(f.read(), dtype="<i2").reshape((ny, nx)).astype(np.float32) * 0.01

    # 2. Compute terrain slopes (slopes > 50 deg cannot be walked up)
    print("Computing terrain gradient & slope...")
    gy, gx = np.gradient(terrain, step, step)
    grad_mag = np.hypot(gx, gy)
    slope_deg = np.degrees(np.arctan(grad_mag))
    steep_mask = slope_deg > 50.0

    # 3. Rasterize prisms
    print(f"Rasterizing {len(world['prisms'])} prisms...")
    img = Image.new("L", (nx, ny), 0)
    draw = ImageDraw.Draw(img)

    walkable_bridges = 0
    blocked_bridges = 0

    for idx, p in enumerate(world["prisms"]):
        kind = p["kind"]
        base = p["base"]
        ring = p["ring"]
        holes = p.get("holes", [])

        # Check bridge clearance
        if kind == "bridge":
            xs = ring[0::2]
            ys = ring[1::2]
            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
            col = int(round((cx - min_x) / step))
            row = int(round((cy - min_y) / step))
            if 0 <= col < nx and 0 <= row < ny:
                clearance = base - terrain[row, col]
                if clearance >= 9.0:
                    walkable_bridges += 1
                    continue  # Mechs walk under it!
            blocked_bridges += 1

        poly = [((ring[i] - min_x) / step, (ring[i+1] - min_y) / step) for i in range(0, len(ring), 2)]
        draw.polygon(poly, fill=255)
        for h in holes:
            hpoly = [((h[i] - min_x) / step, (h[i+1] - min_y) / step) for i in range(0, len(h), 2)]
            draw.polygon(hpoly, fill=0)

    prism_mask = np.array(img, dtype=bool)

    # Combined solid obstacles
    solid_obstacles = prism_mask | steep_mask

    # 4. Euclidean Distance Transform (C-Space clearance)
    print("Computing Euclidean distance transform (C-Space inflation)...")
    dist_to_wall = ndi.distance_transform_edt(~solid_obstacles) * step

    # 5. Build labeled occupancy grid (uint8)
    # 0 = Walkable center (clearance >= 2.5m)
    # 1 = Solid obstacle (prism or steep slope)
    # 2 = Near-wall buffer zone (0 < clearance < 2.5m)
    grid = np.zeros((ny, nx), dtype=np.uint8)
    grid[dist_to_wall < 2.5] = 2  # Buffer zone
    grid[solid_obstacles] = 1     # Solid obstacle

    walkable_count = int(np.sum(grid == 0))
    buffer_count = int(np.sum(grid == 2))
    solid_count = int(np.sum(grid == 1))
    total_cells = nx * ny

    print(f"Results:")
    print(f"  Solid obstacle cells: {solid_count:,} ({solid_count / total_cells * 100:.1f}%)")
    print(f"  Buffer (clearance < 2.5m): {buffer_count:,} ({buffer_count / total_cells * 100:.1f}%)")
    print(f"  Walkable mech center cells: {walkable_count:,} ({walkable_count / total_cells * 100:.1f}%)")
    print(f"  Bridges: {walkable_bridges} clear (walk-under), {blocked_bridges} low (obstacle)")

    # 6. Save binary grid
    print(f"Saving occupancy grid: {output_bin}")
    with open(output_bin, "wb") as f:
        f.write(grid.tobytes())

    # Also save distance transform for path planning cost functions
    dist_bin = os.path.join(kit_dir, "arena", "v2", "clearance.f32")
    print(f"Saving clearance map: {dist_bin}")
    with open(dist_bin, "wb") as f:
        f.write(dist_to_wall.astype(np.float32).tobytes())

    # 7. Save metadata
    meta = {
        "version": 1,
        "nx": nx,
        "ny": ny,
        "step": step,
        "minX": min_x,
        "maxX": min_x + (nx - 1) * step,
        "minY": min_y,
        "maxY": min_y + (ny - 1) * step,
        "mech_radius": 2.5,
        "mech_height": 9.0,
        "legend": {
            0: "walkable",
            1: "solid_obstacle",
            2: "wall_buffer"
        },
        "stats": {
            "total_cells": total_cells,
            "walkable_cells": walkable_count,
            "buffer_cells": buffer_count,
            "solid_cells": solid_count,
            "walkable_bridges": walkable_bridges,
            "blocked_bridges": blocked_bridges
        },
        "formulas": {
            "world_to_grid": "col = round((x - minX) / step); row = round((y - minY) / step)",
            "grid_to_world": "x = minX + col * step; y = minY + row * step",
            "index": "index = row * nx + col"
        }
    }
    with open(output_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {output_meta}")

    # 8. Render visualization
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        print(f"Generating visual map: {output_png}...")
        # Create RGB color map: 0=light grey/white, 1=dark brown, 2=gold/amber
        rgb = np.ones((ny, nx, 3), dtype=np.uint8) * 245
        rgb[grid == 2] = [230, 190, 80]   # Buffer zone
        rgb[grid == 1] = [40, 40, 55]     # Solid buildings/obstacles

        # Flip vertically for display so +y (North) is up
        rgb_disp = np.flipud(rgb)

        plt.figure(figsize=(15, 10), dpi=150)
        extent = [min_x, min_x + nx * step, min_y, min_y + ny * step]
        plt.imshow(rgb_disp, extent=extent)
        plt.title("Mech Royale 2D Occupancy Grid (Ohio State Campus: Stadium to Oval)", fontsize=14, fontweight="bold")
        plt.xlabel("East-West (m)")
        plt.ylabel("North-South (m)")

        # Annotate landmarks
        plt.text(-460, -50, "Ohio Stadium\n(The Horseshoe)", color="cyan", fontsize=10, fontweight="bold", ha="center")
        plt.text(350, -240, "The Oval\n(Library)", color="lime", fontsize=10, fontweight="bold", ha="center")
        plt.text(41.3, -72.8, "Zone Origin (Spawn Center)", color="magenta", fontsize=9, ha="center")
        plt.scatter([41.3], [-72.8], color="magenta", s=50, marker="x")

        plt.tight_layout()
        plt.savefig(output_png, bbox_inches="tight")
        plt.close()
        print(f"Map visualization successfully saved to {output_png}")
    except Exception as e:
        print(f"Warning: could not render plot: {e}")

if __name__ == "__main__":
    kit_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    build_grid(kit_dir)
