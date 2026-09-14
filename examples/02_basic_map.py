"""Example: parameter-focused ``plot_map`` test harness.

Run:
    python examples/02_basic_map.py

Every map in this example uses ``add_coastlines=True`` and a title that documents
exactly which parameters are exercised (grid_sampling, max_distance, markers,
plot_robust, extent). Several configs are combined per plot to keep the set
compact.

Requires the ``coastlines`` extra (cartopy) for the coastlines:
    pip install "plotting_joseph[coastlines]"
"""

from pathlib import Path

import numpy as np
import pandas as pd

from plotting_joseph import plot_map

GLOBAL = (-180, 180, -60, 85)
EUROPE = (-12, 40, 35, 70)

# ---------------------------------------------------------------------------
# 1. Build a small master lookup (location_id -> lat/lon/tile_id)
# ---------------------------------------------------------------------------
Path("lookup_tables").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)
rng = np.random.RandomState(42)
n = 400
master = pd.DataFrame(
    {
        "location_id": np.arange(n),
        "lat": rng.uniform(-60, 85, n),
        "lon": rng.uniform(-180, 180, n),
        "tile_id": [f"t{i % 4}" for i in range(n)],
    }
)
MASTER = "lookup_tables/location_id_to_tile_id.parquet"
master.to_parquet(MASTER, index=False)

# ---------------------------------------------------------------------------
# 2. Map data: one value per location
# ---------------------------------------------------------------------------
data = pd.DataFrame(
    {
        "location_id": master["location_id"],
        "backscatter40": rng.normal(size=n),
    }
)


def save(fig, name: str) -> None:
    fig.savefig(f"figures/{name}", dpi=150, bbox_inches="tight")
    print(f"  Saved figures/{name}")


def in_extent(latitude: float, longitude: float, extent) -> bool:
    lon_min, lon_max, lat_min, lat_max = extent
    return lat_min <= latitude <= lat_max and lon_min <= longitude <= lon_max


def pick_markers(extent, count: int = 3, styles=("o", "x", "^")):
    """Pick source locations inside ``extent`` to use as markers."""
    inside = master[
        master.apply(
            lambda r: in_extent(r["lat"], r["lon"], extent), axis=1
        )
    ]
    picked = inside.sample(n=min(count, len(inside)), random_state=0)
    return [(styles[i % len(styles)], int(loc)) for i, loc in enumerate(picked["location_id"])]


GLOBAL_MARKERS = pick_markers(GLOBAL)


def make_title(**params) -> str:
    parts = []
    if "grid_sampling" in params:
        parts.append(f"grid_sampling={params['grid_sampling']:g}")
    if "max_distance_km" in params:
        parts.append(f"max_dist={params['max_distance_km']:g}km")
    if params.get("plot_robust"):
        low, high = params["plot_robust"]
        parts.append(f"robust({low:g},{high:g})")
    if params.get("add_marker"):
        parts.append("markers")
    parts.append(params.get("extent_name", "map"))
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# 3a. grid_sampling sweep (global, everything else fixed)
# ---------------------------------------------------------------------------
for gs in (0.25, 0.5, 1.0):
    fig = plot_map(
        data=data,
        var="backscatter40",
        master_lookup=MASTER,
        extent=GLOBAL,
        grid_sampling=gs,
        max_distance_km=100.0,
        plot_robust=(2, 98),
        add_marker=GLOBAL_MARKERS,
        add_coastlines=True,
        title=make_title(grid_sampling=gs, max_distance_km=100.0,
                         plot_robust=(2, 98), add_marker=True, extent_name="global"),
    )
    save(fig, f"grid_sampling_{gs:g}.png")

# ---------------------------------------------------------------------------
# 3b. max_distance sweep (global, minimal vs heavy proximity fill)
# ---------------------------------------------------------------------------
for dist in (15.0, 400.0):
    fig = plot_map(
        data=data,
        var="backscatter40",
        master_lookup=MASTER,
        extent=GLOBAL,
        grid_sampling=0.5,
        max_distance_km=dist,
        plot_robust=(2, 98),
        add_marker=GLOBAL_MARKERS,
        add_coastlines=True,
        title=make_title(grid_sampling=0.5, max_distance_km=dist,
                         plot_robust=(2, 98), add_marker=True, extent_name="global"),
    )
    save(fig, f"max_distance_{dist:g}.png")

# ---------------------------------------------------------------------------
# 3c. plot_robust demo (Europe, coarse grid so few pixels, visible color scale)
# ---------------------------------------------------------------------------
EUROPE_MARKERS = pick_markers(EUROPE, count=2)
fig = plot_map(
    data=data,
    var="backscatter40",
    master_lookup=MASTER,
    extent=EUROPE,
    grid_sampling=0.25,
    max_distance_km=50.0,
    plot_robust=(2, 98),
    add_marker=EUROPE_MARKERS,
    add_coastlines=True,
    title=make_title(grid_sampling=0.25, max_distance_km=50.0,
                     plot_robust=(2, 98), add_marker=True, extent_name="Europe"),
)
save(fig, "europe_robust.png")

# ---------------------------------------------------------------------------
# 3d. add_marker demo (Europe, several marker styles listed in title)
# ---------------------------------------------------------------------------
fig = plot_map(
    data=data,
    var="backscatter40",
    master_lookup=MASTER,
    extent=EUROPE,
    grid_sampling=0.25,
    max_distance_km=50.0,
    add_marker=EUROPE_MARKERS,
    add_coastlines=True,
    title=make_title(grid_sampling=0.25, max_distance_km=50.0,
                     add_marker=True, extent_name="Europe"),
)
save(fig, "europe_markers.png")

# ---------------------------------------------------------------------------
# 3e. extent/zoom demo (same-ish params, global vs Europe)
# ---------------------------------------------------------------------------
for extent, label in ((GLOBAL, "global"), (EUROPE, "Europe")):
    markers = pick_markers(extent, count=2)
    fig = plot_map(
        data=data,
        var="backscatter40",
        master_lookup=MASTER,
        extent=extent,
        grid_sampling=0.5,
        max_distance_km=100.0,
        plot_robust=(2, 98),
        add_marker=markers,
        add_coastlines=True,
        title=make_title(grid_sampling=0.5, max_distance_km=100.0,
                         plot_robust=(2, 98), add_marker=True, extent_name=label),
    )
    save(fig, f"extent_{label}.png")

print("\nAll example maps written to ./figures")
