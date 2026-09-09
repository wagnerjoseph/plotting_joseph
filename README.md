# plotting_joseph

Reusable visualization tools for spatiotemporal Earth observation data: flexible
**multi-panel time series** plotting and **global maps**. Designed to be data-format
agnostic and easy to drop into any project.

## Features

- **Time series** (`plot_time_series`) — multi-panel plots with:
  - overlays on a shared or secondary y-axis
  - Pearson / Spearman correlation annotations
  - threshold shading (per-panel or across all panels)
  - season markers, interpolation, rolling transforms
  - nearest-neighbor background series
- **Global maps** (`plot_map`) — gridded worldwide maps with histogram colorbar,
  robust color ranges, markers and optional coastlines.
- **Flexible data loading** (`DataLoader`) — parquet, CSV, or any in-memory
  DataFrame with automatic column mapping/validation.
- **Auto-generated lookup tables** — point the plotting functions at a single
  master lookup (`location_id` → `tile_id` with lat/lon) and the derived lookups
  they need (countries, grid/map, neighbors) are created **on demand**, cached,
  and reused for identical parameters.

## Installation

### Install with uv (recommended)

```bash
uv add "plotting_joseph[all] @ git+https://github.com/wagnerjoseph/plotting_joseph.git"
```

### Install from local repository

```bash
pip install -e .
# with optional extras
pip install -e ".[all]"
```

Extras:

| Extra        | Provides                                        |
|--------------|--------------------------------------------------|
| `netcdf`     | xarray / netCDF loading (load then pass DataFrame) |
| `coastlines` | cartopy natural-earth coastlines on maps         |
| `all`        | netcdf + coastlines                              |
| `dev`        | pytest + ruff                                    |

> **Country names are a built-in, optional feature.** When you pass a
> `master_lookup` (which maps `location_id -> lat/lon`) to
> `plot_time_series`, the country for each plotted point is resolved
> automatically via `reverse_geocoder` (a core dependency) — no separate
> country file required. The first country lookup downloads a small GeoNames
> snapshot and caches it, so internet is only needed once. Locations with no
> resolvable country simply omit the `(country)` tag from the title.

## Quick Start

### Load data and plot time series

```python
from plotting_joseph import DataLoader, plot_time_series

# Normalizes + validates columns (rename yours via column_map)
df = DataLoader().load("monthly_mean/0009.parquet")

plot_time_series(
    data=df,
    location_ids=[2156788],
    var_specs=[
        {"name": "backscatter40", "color": "royalblue"},
        {"name": "lai", "color": "forestgreen", "add_to": "backscatter40",
         "add_second_axis": True, "compute_corr": True},
    ],
    master_lookup="lookup_tables/location_id_to_tile_id.parquet",
    add_closest_points=(4, 100.0),  # auto-generates neighbor lookup for background series
    save_dir="figures",
)
```

Pass a `master_lookup` to auto-generate country titles (via reverse geocoding) and
neighbor lookups (when `add_closest_points` is set). The basics are `data`,
`var_specs`, and `save_dir`; `master_lookup` and `add_closest_points` are optional.

### Plot a global map

```python
from plotting_joseph import plot_map

plot_map(
    data=df,                      # DataFrame with location_id + a value column
    var="backscatter40",
    master_lookup="lookup_tables/location_id_to_tile_id.parquet",  # required
    extent=(-180, 180, -60, 85),
    grid_sampling=0.5,            # optional, defaults to 0.5
    title="Global Backscatter",
    save_path="figures/global_map.png",
)
```

`plot_map` builds the grid/map lookup automatically from the master lookup. The
lookup is created once per combination of geometric parameters (`grid_sampling`,
`extent`, `k`) and reused afterwards.

## Auto-generated lookups

Pass a **master lookup** (`location_id_to_tile_id.parquet` with `location_id`,
`lat`, `lon` and optional `tile_id`) to the plotting functions and the derived
lookups are **created on demand**, only when they don't already exist:

* **countries** (for time-series titles) — resolved online via reverse
  geocoding (downloaded and cached on first use)
* **grid/map lookup** (for `plot_map`) — built from the lat/lon coordinates,
  keyed by `grid_sampling` + `extent` + `k`
* **neighbors** (for `add_closest_points`) — per-tile nearest neighbors, keyed
  by `k` + `max_distance_km`

Generated lookups are saved automatically next to the master lookup — the
map grid lookups into a `map_lookups/` folder and the neighbors into a
`neighbor_lookups/` folder (derived lookups like `location_ids.parquet` and
`countries.pkl` go next to the master file itself). Lookups are **reused
whenever the same parameters are passed again**; the geometric parameters are
encoded in the lookup filename, so different grids/extents produce separate
cached files.

The `ensure_*` helpers (`ensure_location_ids`, `ensure_country_lookup`,
`ensure_grid_lookup`, `ensure_neighbor_lookup`) expose the same generation for
manual use.

## Documentation

- [**API reference (full)**](docs/api/reference.md) — every parameter of `plot_time_series` & `plot_map`
- [Lookup tables](docs/lookup_tables.md) — how to create & use the geographic lookups
- [Data formats](docs/data_formats.md) — supported formats and column mapping
- [API: timeseries](docs/api/timeseries.md)
- [API: maps](docs/api/maps.md)

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src
```

## License

MIT — see [LICENSE](LICENSE). Maintained by Joseph Wagner
(joseph.wagner@geo.tuwien.ac.at).
