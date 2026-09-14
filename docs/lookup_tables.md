# Lookup Tables

Lookup tables connect your data's location identifiers to geographic
information, so the plotting functions can place points on a map, label
titles with countries, and draw nearest-neighbor backgrounds.

`plotting_joseph` does **not** ship lookup tables. Instead it provides
tools to **create them from your own data**. Below is the exact format each
table must follow, and how to generate them.

## Auto-generated from a master lookup (recommended)

The easiest workflow is to point the plotting functions at a single **master
lookup** — a `location_id_to_tile_id.parquet` with `location_id`, `lat`, `lon`
and (optionally) `tile_id`. Everything else is derived from it **on demand**:

* `plot_map(..., master_lookup=..., extent=..., grid_sampling=..., k=..., max_distance_km=...)`
  builds the grid/map lookup when none is passed.
* `Timeseries.plot_time_series(..., master_lookup=..., ...)` builds the country
  lookup (from the web) and, with `add_closest_points`, the neighbors.

Generated lookups are saved automatically next to the master lookup — the map
grid lookups into a `map_lookups/` folder and the neighbors into a
`neighbor_lookups/` folder; `location_ids.parquet` and `countries.pkl` go next
to the master file itself. Lookups are **reused whenever the same parameters
are passed again**; the geometric parameters are encoded in the filename/dir
name, so different grids/neighbor requests produce separate cached files:

```
map_lookups/gridSampling_0.5_extent_-180_180_-60_85_k1_maxDistKm_15.parquet
neighbor_lookups/neighbors_k8_maxd100/location_id_to_tile_id_0009.parquet
```

When `max_distance_km > 0` the grid lookup filename also encodes the maximum
proximity-fill distance (see below); passing `max_distance_km=0` produces the
older filename without the `_maxDistKm_` tag.

The same generation is available as reusable helpers:

```python
from plotting_joseph import (
    ensure_location_ids, ensure_country_lookup,
    ensure_grid_lookup, ensure_neighbor_lookup,
)

ensure_location_ids("lookup_tables/location_id_to_tile_id.parquet")
ensure_country_lookup("lookup_tables/location_id_to_tile_id.parquet")
ensure_grid_lookup("lookup_tables/location_id_to_tile_id.parquet",
                   grid_sampling=0.5, extent=(-180, 180, -60, 85), k=1,
                   max_distance_km=15.0)
ensure_neighbor_lookup("lookup_tables/location_id_to_tile_id.parquet",
                       k_neighbors=8, max_distance_km=100.0)
```

> The neighbor search distance is capped at **100 km** so the generated lookup
> files stay small.

The manual workflows below describe the file formats in case you want to build
the lookups yourself.

## Required: `location_ids.parquet`

Used by `plot_map` (and optionally by `Timeseries.plot_time_series` for
neighbor lookups).

### Schema

| Column        | Type    | Description                             |
|---------------|---------|-----------------------------------------|
| `location_id` | int64   | Unique location identifier              |
| `lat`         | float64 | Latitude in degrees (-90 to 90)         |
| `lon`         | float64 | Longitude in degrees (-180 to 180)      |
| `tile_id`     | str     | *(optional)* tile/region grouping       |

### Create from a DataFrame / parquet / CSV

```python
from plotting_joseph.data import LookupTableCreator

LookupTableCreator.from_parquet(
    input_path="tiles/0009.parquet",
    location_id_column="location_id",
    lat_column="lat",
    lon_column="lon",
    output_path="lookup_tables/location_ids.parquet",
    tile_id_column="tile_id",
)

# Equivalent from CSV
LookupTableCreator.from_csv(
    csv_path="locations.csv",
    location_id_column="location_id",
    lat_column="latitude",
    lon_column="longitude",
    output_path="lookup_tables/location_ids.parquet",
)
```

### Create on a regular grid (no source coordinates)

```python
LookupTableCreator.from_grid(
    output_path="lookup_tables/location_ids.parquet",
    resolution_km=12.5,
    lat_min=-60, lat_max=85, lon_min=-180, lon_max=180,
)
```

## Required for maps: the grid-sampling lookup

`plot_map` builds this lookup automatically from the master lookup, mapping
`location_id` -> `pixel_id` on the regular grid you want to render. The
auto-generated filename follows the pattern
`gridSampling_<res>_extent_<lon0>_<lon1>_<lat0>_<lat1>_kN[_maxDistKm_<dist>].parquet`
where `N` is the number of aggregated neighbors (`1` = direct 1:1 mapping),
`<res>` is the grid resolution in degrees, and `<dist>` is the proximity-fill
distance in km when `max_distance_km > 0`.

### Two ways to map locations to pixels

By default (`max_distance_km > 0`) an **inverted, proximity-filled** lookup is
built: every grid pixel is assigned the value of its **nearest** source
location, but only if that location lies within `max_distance_km` from the
pixel. This fills in the pixels around every measurement, so nearby areas are
colored instead of white; only pixels farther than `max_distance_km` from every
location stay blank. The default is `15.0` km.

Pass `max_distance_km=0` (or a negative value) for the exact **per-location
snap**: each location only colors the single grid cell it falls into, and any
location outside `extent` is dropped (its surrounding pixels stay white).

### Schema (`k1`)

| Column        | Type  | Description                    |
|---------------|-------|--------------------------------|
| `location_id` | int64 | Location identifier            |
| `pixel_id`    | int64 | Flat index into the output grid (`row * n_lon + col`) |

For the inverted lookup each `pixel_id` appears once, mapped to its nearest
location (a single location therefore appears in many rows — one per pixel it
fills).

### Schema (`kN`, `N > 1`)

| Column        | Type  | Description                          |
|---------------|-------|--------------------------------------|
| `pixel_id`    | int64 | Pixel index                          |
| `location_ids`| list  | Nearest locations mapped to the pixel (within `max_distance_km` when filled) |

The grid dimensions are derived from the `extent` and the sampling embedded in
the filename: `n_lat = (lat_max - lat_min) / grid_sampling`,
`n_lon = (lon_max - lon_min) / grid_sampling`.

> **Note:** because the pixel/grid geometry is coupled to your specific grid,
> this table is normally generated by your own gridding pipeline (or with
> `ensure_grid_lookup`, see above). The package validates it and renders
> whatever regular grid you provide.

## Optional: `countries.pkl`

Used to show a country name in time-series titles.

### Schema

A Python pickle of a `dict`: `{location_id: country_name}`.

### Generate

Reverse geocoding is a core feature (no extra install); internet is required
only on first use (it downloads a GeoNames snapshot and caches it):

```python
LookupTableCreator.generate_country_lookup(
    location_ids_path="lookup_tables/location_ids.parquet",
    output_path="lookup_tables/countries.pkl",
)
```

## Optional: `neighbors_dir`

Enables nearest-neighbor background series in `Timeseries.plot_time_series`.
One parquet file per tile, named `{tile_id}.parquet`.

### Schema (per file)

| Column                | Type    | Description                 |
|-----------------------|---------|-----------------------------|
| `location_id`         | int64   | Center location             |
| `neighbor_location_id`| int64   | Neighbor location           |
| `distance_km`         | float64 | Distance in km              |
| `rank`                | int64   | 1 = closest, 2 = 2nd, ...   |

### Generate

```python
LookupTableCreator.generate_neighbor_lookup(
    location_ids_path="lookup_tables/location_ids.parquet",
    output_dir="lookup_tables/closest_location_ids/",
    k_neighbors=10,
    max_distance_km=100.0,
)
```

## Validation

```python
from plotting_joseph import validate_lookup_tables
from plotting_joseph import LookupTables

errors = validate_lookup_tables(
    LookupTables(
        location_ids="lookup_tables/location_ids.parquet",
        countries="lookup_tables/countries.pkl",
        neighbors_dir="lookup_tables/closest_location_ids/",
    )
)
```

Returns a list of error strings (empty when everything is valid).
