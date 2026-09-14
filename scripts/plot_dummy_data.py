#!/usr/bin/env python3
"""Plot the generated dummy dataset interactively.

Generates the dummy data automatically if it is missing, then shows each plot
full-screen. Close a window (click the X) and the next plot is rendered.

Run:
    python scripts/plot_dummy_data.py                 # auto-generates + shows
    python scripts/plot_dummy_data.py -d /path/to/data
    python scripts/plot_dummy_data.py --generate      # run the parameter test suite
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plotting_joseph import Timeseries, plot_map

EU_WIDE = (-10, 20, 35, 60)
EU_ZOOM = (-2, 10, 42, 52)


def load_split(root: Path, split: str, var: str = "backscatter40") -> pd.DataFrame:
    """Read one metrics file -> DataFrame for plot_map."""
    p = root / split / "metrics_global_plot" / f"{var}.parquet"
    return pd.read_parquet(p)


def load_timeseries(root: Path, split: str, location_id: int) -> pd.DataFrame:
    """Gather the timeseries rows for a single location across tiles."""
    frames = []
    for tile in (root / split / "timeseries").glob("*.parquet"):
        df = pd.read_parquet(tile)
        frames.append(df[df["location_id"] == location_id])
    return pd.concat(frames, ignore_index=True)


def show_and_wait(fig) -> None:
    """Block until the user closes the window, then move on."""
    plt.show()
    plt.close(fig)


def build_title(**params) -> str:
    """Build a self-describing title from the tested parameters."""
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


def pick_markers(master: pd.DataFrame, extent, count: int = 3, styles=("o", "x", "^")):
    """Pick ``count`` location ids from ``master`` that fall inside ``extent``.

    Markers must lie within the plotted extent or ``plot_map`` skips them with a
    warning, so this filters by lat/lon rather than taking the first N ids.
    """
    lon_min, lon_max, lat_min, lat_max = extent
    inside = master[
        (master["lon"] >= lon_min) & (master["lon"] <= lon_max)
        & (master["lat"] >= lat_min) & (master["lat"] <= lat_max)
    ]
    if inside.empty:
        return []
    picked = inside["location_id"].to_numpy()[:count]
    return [(styles[i % len(styles)], int(loc)) for i, loc in enumerate(picked)]


def show_generate_suite(
    data: pd.DataFrame,
    master_lookup: Path,
    root: Path,
    split: str,
    var: str,
    location: int,
) -> None:
    """Show a compact parameter test suite: 3 maps + 2 timeseries.

    Maps always use ``add_coastlines=True`` and a self-describing title. The two
    timeseries exercise different ``var_specs`` / neighbor options.
    """
    master_df = pd.read_parquet(master_lookup)

    # --- 3 map plots ---------------------------------------------------
    markers_wide = pick_markers(master_df, EU_WIDE, count=3)
    markers_zoom = pick_markers(master_df, EU_ZOOM, count=2)

    # Map 1: grid_sampling
    fig = plot_map(
        data=data, var=var, master_lookup=master_lookup, extent=EU_WIDE,
        grid_sampling=0.25, max_distance_km=100.0, plot_robust=(2, 98),
        add_marker=markers_wide, add_coastlines=True,
        title=build_title(grid_sampling=0.25, max_distance_km=100.0,
                          plot_robust=(2, 98), add_marker=True, extent_name="EU-wide"),
        show_plot=False,
    )
    show_and_wait(fig)

    # Map 2: max_distance (minimal vs heavy fill across two sparse extents)
    fig = plot_map(
        data=data, var=var, master_lookup=master_lookup, extent=EU_WIDE,
        grid_sampling=0.5, max_distance_km=15.0, plot_robust=(2, 98),
        add_marker=markers_wide, add_coastlines=True,
        title=build_title(grid_sampling=0.5, max_distance_km=15.0,
                          plot_robust=(2, 98), add_marker=True, extent_name="EU-wide"),
        show_plot=False,
    )
    show_and_wait(fig)

    # Map 3: max_distance + extent-zoom
    fig = plot_map(
        data=data, var=var, master_lookup=master_lookup, extent=EU_ZOOM,
        grid_sampling=0.5, max_distance_km=300.0, plot_robust=(2, 98),
        add_marker=markers_zoom, add_coastlines=True,
        title=build_title(grid_sampling=0.5, max_distance_km=300.0,
                          plot_robust=(2, 98), add_marker=True, extent_name="EU-zoom"),
        show_plot=False,
    )
    show_and_wait(fig)

    # --- 2 timeseries plots --------------------------------------------
    print(f"  timeseries location {location}")
    ts = load_timeseries(root, split, location)

    # TS 1: multi-panel variables, seasons overlay, rolling_mean + shading/points.
    ts1 = Timeseries.plot_time_series(
        data=ts,
        location_ids=[location],
        var_specs=[
            {"name": "backscatter40", "color": "royalblue", "show_seasons": True},
            {"name": "lai", "color": "green",
             "transforms": [{"type": "rolling_mean", "window": 15}]},
            {"name": "swvl1", "color": "darkorange", "plotstyle": "points",
             "lower_percentile": (10, "lightblue")},
        ],
        master_lookup=master_lookup,
        show_plot=False,
    )
    for f in ts1:
        show_and_wait(f)

    # TS 2: one variable, line+points, NaN interpolation, nearest-neighbour
    # background (add_closest_points) and a rolling-mean overlay on a 2nd axis.
    ts2 = Timeseries.plot_time_series(
        data=ts,
        location_ids=[location],
        var_specs=[
            {"name": "backscatter40", "color": "crimson", "plotstyle": "both",
             "interpolate": True},
            {"name": "backscatter40", "color": "black", "line_width": 2.5,
             "transforms": [{"type": "rolling_mean", "window": 30}],
             "add_to": "backscatter40", "add_second_axis": True},
        ],
        add_closest_points=(3, 50.0),
        master_lookup=master_lookup,
        show_plot=False,
    )
    for f in ts2:
        show_and_wait(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the dummy dataset interactively")
    parser.add_argument("--data", "-d", type=Path, default=Path("/tmp/plotting_joseph_dummy"))
    parser.add_argument("--split", type=str, default="split_2020_2022")
    parser.add_argument("--var", type=str, default="backscatter40")
    parser.add_argument("--location", type=int, default=0)
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Run the compact parameter test suite (3 maps + 2 timeseries)",
    )
    args = parser.parse_args()

    root: Path = args.data.expanduser().resolve()
    master_lookup = root / "ers_tile_id_location_id.parquet"
    if not master_lookup.exists():
        if args.generate:
            from generate_dummy_data import generate_dummy_data

            print(f"Generating dummy data in {root}")
            generate_dummy_data(root=root)
        else:
            raise SystemExit(
                f"Master lookup not found at {master_lookup}. "
                f"Generate it first: python scripts/generate_dummy_data.py -o {root}, "
                f"or pass --generate to create it automatically."
            )

    data = load_split(root, args.split, args.var)

    print("Close each window (click X) to render the next plot.\n")

    if args.generate:
        print("=== generate suite: 3 maps + 2 timeseries ===")
        show_generate_suite(
            data=data,
            master_lookup=master_lookup,
            root=root,
            split=args.split,
            var=args.var,
            location=args.location,
        )
    else:
        # Simple default: one map + one timeseries.
        fig = plot_map(
            data=data,
            var=args.var,
            master_lookup=master_lookup,
            extent=EU_WIDE,
            grid_sampling=0.5,
            max_distance_km=50.0,
            add_coastlines=True,
            title=f"Dummy {args.var} — max_distance_km=50",
            show_plot=False,
        )
        show_and_wait(fig)

        print(f"  timeseries for location {args.location}")
        ts = load_timeseries(root, args.split, args.location)
        figs = Timeseries.plot_time_series(
            data=ts,
            location_ids=[args.location],
            var_specs=[{"name": "backscatter40", "color": "royalblue"}],
            master_lookup=master_lookup,
            show_plot=False,
        )
        for f in figs:
            show_and_wait(f)

    print("\nDone.")


if __name__ == "__main__":
    main()
