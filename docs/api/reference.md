# API Reference

This page documents every parameter of the two main plotting functions,
`plot_time_series` and `plot_map`, together with the `var_specs` schema that
drives the time-series panels. It is the authoritative reference — the
docstrings in `src/plotting_joseph/plotting/` mirror the same information.

Both functions are exposed at the top level of the package:

```python
from plotting_joseph import plot_time_series, plot_map
```

---

## `plot_time_series(data, ...)`

Creates **one multi-panel figure per selected location**. Each variable gets
its own panel; overlays (via `add_to`) can share a panel, optionally on a
right-hand secondary y-axis.

```python
plot_time_series(
    data,
    var_specs=None,
    location_ids=None,
    random_points=(2, 123),
    add_closest_points=(0, 0),
    lookup_tables=None,
    master_lookup=None,
    save_dir=None,
    figsize=(10, 5),
    font_scale=1.0,
    show_plot=False,
) -> list[Figure]
```

`data` is the only required argument. Everything else has a sensible default.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `DataFrame` / `dask.DataFrame` | *(required)* | Data with `location_id`, `time` and the numeric variables to plot. |
| `var_specs` | `list[dict]` | `None` | Which variables to plot and how (see [var-spec schema](#var-spec-schema)). If `None`, all numeric columns are inferred automatically. |
| `location_ids` | `Iterable[int]` | `None` | Locations to plot. If `None`, `random_points` locations are chosen. |
| `random_points` | `tuple[int, int]` | `(2, 123)` | `(n_points, seed)` for random location selection. Only used when `location_ids` is `None`. |
| `add_closest_points` | `tuple[int, float]` | `(0, 0)` | `(k_closest, max_distance_km)` to draw nearest-neighbor background series per panel. `(0, 0)` disables. `max_distance_km` is capped at 100 km. |
| `lookup_tables` | `LookupTables` | `None` | Explicit paths for the `location_ids` table, `countries` pickle and `neighbors_dir`. Takes precedence over `master_lookup`. |
| `master_lookup` | `str` / `Path` | `None` | Master lookup parquet (`location_id` → tile with `lat`/`lon`). Auto-generates the country lookup and (with `add_closest_points`) the neighbor lookup, caching them for reuse. |
| `save_dir` | `str` / `Path` | `None` | Save each location figure as `{save_dir}/{location_id}.png`. |
| `figsize` | `tuple[int, int]` | `(10, 5)` | Base figure size; height scales with the number of panels. |
| `font_scale` | `float` | `1.0` | Font-size scale factor for labels, ticks and legend. |
| `show_plot` | `bool` | `False` | If `True`, display figures interactively. |

**Returns:** `list[Figure]` — one matplotlib `Figure` per location.

> **Country names.** When you pass `master_lookup`, the country for each
> plotted point is resolved automatically via `reverse_geocoder` and shown in
> the title, e.g. `location_id = 5 (Poland)`. Reverse geocoding needs internet
> only on first use (a GeoNames snapshot is cached); locations with no
> resolvable country simply omit the `(country)` tag.

---

### var-spec schema

Each element of `var_specs` is a dict describing one line to draw. How it is
interpreted depends on the presence of `add_to`:

- **No `add_to`** → the variable becomes its **own panel**.
- **With `add_to`** → the variable is drawn as an **overlay** on the panel of
  the named parent variable.

| Key | Type | Default | Effect |
|-----|------|---------|--------|
| `name` | `str` | *(required)* | Column name in `data`. |
| `label` | `str` | `name` | Label used in the legend and as the y-axis label. |
| `color` | `str` | `"tab:blue"` | Line/marker color. |
| `line_width` | `float` | `1.5` | Line width. |
| `alpha` | `float` | `1` | Line/marker opacity. |
| `plotstyle` | `str` | `"line"` | `"line"`, `"points"` or `"both"`. |
| `point_size` | `float` | `10` | Marker size when plotting points. |
| `point_linewidth` | `float` | `0.5` | Marker edge width. |
| `show_seasons` | `bool` | `False` | Overlay JJA (red) and DJF (black) seasonal markers. |
| `interpolate` | `bool` | `False` | Interpolate over NaN values in the series. |
| `transforms` | `list[dict]` | `[]` | Derived series, e.g. `{"type": "rolling_mean", "window": 12}`. |
| `add_to` | `str` | *absent* | Parent variable name to overlay onto (turning this spec into an overlay). |
| `add_second_axis` | `bool` | `False` | Draw the overlay on a right-hand secondary y-axis. |
| `align_zero` | `bool` | `False` | Align the zero points of the two axes. |
| `compute_corr` | `bool` | `False` | Annotate Pearson + Spearman correlation with the parent. Requires exactly two lines (1 parent + 1 overlay) on the panel. |
| `lower_treshold` | `tuple` | *absent* | `(value, color)` to shade vertical bands where values fall below `value`. |
| `upper_treshold` | `tuple` | *absent* | `(value, color)` to shade vertical bands where values rise above `value`. |
| `lower_percentile` | `tuple` | *absent* | `(percentile, color)` to shade vertical bands where values fall below the variable's `percentile`-th percentile. |
| `upper_percentile` | `tuple` | *absent* | `(percentile, color)` to shade vertical bands where values rise above the variable's `percentile`-th percentile. |
| `apply_shading_to_all` | `bool` | `False` | Extend the threshold shading to all panels. |

### Example

```python
from plotting_joseph import plot_time_series

figs = plot_time_series(
    data=df,
    location_ids=[5, 17],
    var_specs=[
        {"name": "backscatter40", "color": "royalblue"},
        {"name": "lai", "label": "Leaf Area Index", "color": "forestgreen",
         "add_to": "backscatter40", "add_second_axis": True, "compute_corr": True},
    ],
    master_lookup="lookup_tables/location_id_to_tile_id.parquet",
    add_closest_points=(4, 100.0),
    save_dir="figures",
)
```

---

## `plot_map(data, var, master_lookup, ...)`

Plots a **gridded global map** of a single variable with a histogram colorbar,
robust color ranges and optional coastlines and markers.

```python
plot_map(
    data,
    var,
    master_lookup=None,
    month=None,
    stat="median",
    title=None,
    cbar_label=None,
    cmap="viridis",
    center_at_zero=False,
    extent=(-180, 180, -60, 85),
    grid_sampling=0.5,
    k=1,
    value_range=None,
    save_path=None,
    plot_robust=None,
    add_coastlines=False,
    add_marker=None,
    dpi=300,
    figsize=None,
    font_scale=1.0,
    show_plot=False,
) -> Figure
```

`data`, `var` and `master_lookup` are required.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `DataFrame` | *(required)* | Data with `location_id` (and optional `time`) plus the value column to plot. |
| `var` | `str` | *(required)* | Column name of the value to plot. |
| `master_lookup` | `str` / `Path` | *(required)* | Master lookup parquet (`location_id` → tile with `lat`/`lon`). The grid lookup is built from it and reused. |
| `month` | `str` | `None` | Filter to a single month (e.g. `"2020-01"`) using the `time` column. `None` uses all data. |
| `stat` | `str` | `"median"` | Aggregation when `k > 1`: `"min"`, `"max"`, `"mean"`, `"median"`. |
| `title` | `str` | `None` | Plot title. `None` auto-generates one. |
| `cbar_label` | `str` | `None` | Colorbar label. `None` auto-generates one. |
| `cmap` | `str` | `"viridis"` | Matplotlib colormap name. |
| `center_at_zero` | `bool` | `False` | Center the color scale at 0 (symmetric range). |
| `extent` | `tuple` | `(-180, 180, -60, 85)` | Bounding box `(lon_min, lon_max, lat_min, lat_max)` of the map. |
| `grid_sampling` | `float` | `0.5` | Grid resolution in degrees used to build the grid lookup. |
| `k` | `int` | `1` | Number of aggregated neighbors per pixel (`1` = direct 1:1 mapping). |
| `value_range` | `tuple[float, float]` | `None` | Fixed color range `(vmin, vmax)`; values outside are clipped. `None` uses the data min/max. |
| `save_path` | `str` / `Path` | `None` | Where to save the figure. `None` does not save. |
| `plot_robust` | `tuple[float, float]` | `None` | Robust color range as percentiles `(low, high)`, e.g. `(2, 98)`. |
| `add_coastlines` | `bool` | `False` | Overlay natural-earth coastlines (requires the `coastlines` extra / `cartopy`). |
| `add_marker` | `tuple` / `list[tuple]` | `None` | `(style, location_id)` marker(s) placed on the map. |
| `dpi` | `int` | `300` | Resolution for the saved figure. |
| `figsize` | `tuple[int, int]` | `None` | Figure size `(width, height)`. `None` derives it proportionally from `extent`. |
| `font_scale` | `float` | `1.0` | Font-size scale factor. |
| `show_plot` | `bool` | `False` | If `True`, display the figure interactively. |

**Returns:** the matplotlib `Figure`.

> **Grid lookup reuse.** The map grid lookup is auto-built from
> `master_lookup` and cached. Its filename encodes `grid_sampling`, `extent`
> and `k`, so identical parameter combinations reuse the same file. Lookups are
> stored in a `map_lookups/` folder next to the master lookup.

### Example

```python
from plotting_joseph import plot_map

fig = plot_map(
    data=df,                        # location_id + a value column
    var="backscatter40",
    master_lookup="lookup_tables/location_id_to_tile_id.parquet",
    extent=(-180, 180, -60, 85),
    grid_sampling=0.5,
    plot_robust=(2, 98),
    cmap="viridis",
    center_at_zero=True,
    title="Global Backscatter (2-98%)",
    save_path="figures/global_backscatter.png",
)
```

---

## Supporting helpers

Top-level utilities referenced throughout the docs:

| Helper | Purpose |
|--------|---------|
| `DataLoader(column_map=...)` | Load & normalise parquet / CSV / DataFrame to the canonical schema. |
| `LookupTables.from_dict(...)` | Container for `location_ids`, `countries`, `neighbors_dir` paths. |
| `validate_lookup_tables(...)` | Return a list of errors (empty if the lookups are valid). |
| `ensure_location_ids(...)` | Build or reuse `location_ids.parquet` from a master lookup. |
| `ensure_country_lookup(...)` | Build or reuse `countries.pkl` (reverse geocoding, cached). |
| `ensure_grid_lookup(...)` | Build or reuse the map grid lookup for given geometry. |
| `ensure_neighbor_lookup(...)` | Build or reuse the per-tile neighbor lookup. |
| `country_name_from_code(...)` | Translate a 2-letter country code to a full English name. |
| `Timeseries.compute_correlation(df, var1, var2)` | Plot-free Pearson/Spearman correlation. |

For lookup-table file formats and how to build them yourself, see
[Lookup tables](../lookup_tables.md). For how the data must be shaped and how
to load/rename columns, see [Data formats](../data_formats.md).

## Related

- [Time series API](timeseries.md)
- [Maps API](maps.md)
- [Lookup tables](../lookup_tables.md)
- [Data formats](../data_formats.md)
