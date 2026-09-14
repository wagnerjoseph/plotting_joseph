#!/usr/bin/env python3
"""Generate a small dummy dataset for quick testing of plotting_joseph.

Mirrors the dummy-data generator used by ``dataviewer_joseph`` (see
``src/dataviewer_joseph/data.py::generate_dummy_data``) so the on-disk layout and
the master lookup are identical in spirit:

    <root>/
        ers_tile_id_location_id.parquet    # master lookup: location_id/lat/lon/tile_id
        <split>/
            metrics_global_plot/<var>.parquet   # one value per location (for plot_map)
            timeseries/<tile>.parquet           # time/location_id/backscatter40/lai/swvl1

Run:
    python scripts/generate_dummy_data.py            # -> /tmp/plotting_joseph_dummy
    python scripts/generate_dummy_data.py -o data/dummy -n 200 -s split_2020_2022
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_dummy_data(
    root: Path | str,
    n_locations: int = 100,
    n_tiles: int = 4,
    splits: list[str] | None = None,
    variables: list[str] | None = None,
    seed: int = 42,
) -> Path:
    """Generate dummy data and return the root directory."""
    rng = np.random.default_rng(seed)
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    if splits is None:
        splits = ["split_2020_2022", "split_2023_2024"]
    if variables is None:
        variables = ["backscatter40", "rmse", "pearson"]

    # --- Locations on a compact geographic footprint (Europe-ish) ------------
    location_ids = np.arange(n_locations)
    tile_ids = np.arange(n_tiles)
    lats = rng.uniform(40, 50, n_locations)
    lons = rng.uniform(-5, 15, n_locations)
    location_tile_ids = rng.choice(tile_ids, n_locations)

    lookup_df = pd.DataFrame(
        {
            "location_id": location_ids,
            "lat": lats,
            "lon": lons,
            "tile_id": location_tile_ids,
        }
    )
    lookup_df.to_parquet(root / "ers_tile_id_location_id.parquet", index=False)
    print(f"  Master lookup: {root / 'ers_tile_id_location_id.parquet'} ({n_locations} locations)")

    date_range = pd.date_range(start="2020-01-01", end="2022-12-31", freq="D")

    for split in splits:
        split_dir = root / split

        # One value per location per variable -> direct input to plot_map.
        metrics_dir = split_dir / "metrics_global_plot"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        for var in variables:
            var_data = pd.DataFrame(
                {
                    "location_id": location_ids,
                    var: rng.normal(0.5, 0.2, n_locations),
                }
            )
            var_data.to_parquet(metrics_dir / f"{var}.parquet", index=False)

        # Per-tile timeseries with a seasonal signal -> input to plot_time_series.
        ts_dir = split_dir / "timeseries"
        ts_dir.mkdir(parents=True, exist_ok=True)
        for tile_id in tile_ids:
            tile_locations = location_ids[location_tile_ids == tile_id]
            rows = []
            for loc_id in tile_locations:
                for date in date_range:
                    seasonal = np.sin(2 * np.pi * date.dayofyear / 365)
                    rows.append(
                        {
                            "location_id": loc_id,
                            "time": date,
                            "backscatter40": seasonal + rng.normal(0, 0.5),
                            "lai": 2 + seasonal + rng.normal(0, 0.3),
                            "swvl1": 0.3 + seasonal * 0.1 + rng.normal(0, 0.05),
                        }
                    )
            ts_df = pd.DataFrame(rows)
            ts_df.to_parquet(ts_dir / f"{tile_id}.parquet", index=False)

        print(f"  Split '{split}': metrics + {n_tiles} timeseries tiles")

    return root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate dummy data for plotting_joseph development/testing"
    )
    parser.add_argument("--output", "-o", type=Path, default=Path("/tmp/plotting_joseph_dummy"))
    parser.add_argument("--locations", "-n", type=int, default=100)
    parser.add_argument("--tiles", "-t", type=int, default=4)
    parser.add_argument("--splits", "-s", type=str, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Generating dummy data in {args.output}")
    root = generate_dummy_data(
        root=args.output,
        n_locations=args.locations,
        n_tiles=args.tiles,
        splits=args.splits,
        seed=args.seed,
    )
    print(f"\nDone. Use 'python scripts/plot_dummy_data.py -d {root}' to plot it.")


if __name__ == "__main__":
    main()
