# Timeseries API

> 📚 For the **complete parameter reference**, see [API reference](reference.md).

See `src/plotting_joseph/plotting/timeseries.py` for the full docstring.

## `plot_time_series(data, ...)`

Creates one multi-panel figure per selected location.

**Recommended usage:** import the module-level function for a clean API that mirrors `plot_map`:

```python
from plotting_joseph import plot_time_series

plot_time_series(data=df, location_ids=[...], var_specs=[...])
```

### Key parameters

| Parameter            | Description                                                            |
|----------------------|------------------------------------------------------------------------|
| `data`               | pandas (or dask) DataFrame with `location_id`, `time`, and variables.  |
| `var_specs`          | list of var-spec dicts (see below). `None` infers numeric columns.     |
| `location_ids`       | locations to plot; `None` selects random ones via `random_points`.     |
| `add_closest_points` | `(k, max_km)` for nearest-neighbor background series.                  |
| `lookup_tables`      | a `LookupTables` config for `countries` / `neighbors_dir`.             |
| `master_lookup`      | master lookup (`location_id` -> tile with `lat`/`lon`); auto-generates `countries` and (when `add_closest_points` is used) the neighbor lookup. |
| `save_dir`           | save each location as `{save_dir}/{location_id}.png`.                  |
| `show_plot`          | display figures interactively.                                         |

Returns a list of matplotlib figures (one per location).

> When `master_lookup` is given the country lookup is auto-generated from the
> web when it doesn't exist yet, and the neighbor lookup is generated (and
> reused) when `add_closest_points` is set. Neighbors go into a
> `neighbor_lookups/` folder next to the master lookup; `countries.pkl` and
> `location_ids.parquet` are written next to the master file itself. If
> `lookup_tables` is provided it takes precedence over `master_lookup`.
>
> The neighbor search distance is capped at **100 km** to keep the generated
> lookup files small — a larger `max_km` is silently clamped to 100.

### var-spec options

| Key                    | Type                  | Effect                                   |
|------------------------|-----------------------|------------------------------------------|
| `name`                 | str *(required)*      | column name                              |
| `label`                | str                   | legend / y-axis label                    |
| `color`                | str                   | line color                               |
| `line_width`, `alpha`  | float                 | line styling                             |
| `plotstyle`            | line/points/both      | plot style                               |
| `show_seasons`         | bool                  | overlay JJA/DJF markers                  |
| `interpolate`          | bool                  | interpolate NaNs                         |
| `transforms`           | list[dict]            | e.g. `{"type": "rolling_mean", "window": 12}` |
| `add_to`               | str                   | overlay onto parent panel (must define `add_to`) |
| `add_second_axis`      | bool                  | put overlay on secondary y-axis          |
| `align_zero`           | bool                  | align zero points of both axes           |
| `compute_corr`         | bool                  | show Pearson+Spearman vs parent (2-line panels only) |
| `lower_treshold`       | (value, color)        | shade where values below value           |
| `upper_treshold`       | (value, color)        | shade where values above value           |
| `lower_percentile`     | (pct, color)          | shade where values below var's `pct`-th percentile |
| `upper_percentile`     | (pct, color)          | shade where values above var's `pct`-th percentile |
| `apply_shading_to_all` | bool                  | extend threshold shading to all panels   |

### Example

```python
from plotting_joseph import plot_time_series

plot_time_series(
    data=df,
    location_ids=[2156788, 2156790],
    var_specs=[
        {"name": "backscatter40", "color": "royalblue"},
        {"name": "lai", "color": "forestgreen", "add_to": "backscatter40",
         "add_second_axis": True, "compute_corr": True},
    ],
    master_lookup="lookup_tables/location_id_to_tile_id.parquet",
    add_closest_points=(4, 100.0),
    save_dir="figures",
)
```

Pass a `master_lookup` to auto-generate country titles and neighbor lookups.
The basics are `data`, `var_specs`, and `save_dir`; `master_lookup` and
`add_closest_points` are optional.

**Alternative (backward-compatible):** `Timeseries.plot_time_series(...)` also works.

## `Timeseries.compute_correlation(df, var1, var2, methods=("pearson", "spearman"))`

Returns a dict of correlations (NaN if < 3 valid pairs). Useful for quick,
plot-free analysis.
