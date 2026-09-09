# Maps API

> 📚 For the **complete parameter reference**, see [API reference](reference.md).

See `src/plotting_joseph/plotting/maps.py` for the full docstring.

## `plot_map(data, var, master_lookup, ...)`

Plots a gridded global map of a variable with a histogram colorbar.

### Key parameters

| Parameter         | Description                                                          |
|-------------------|----------------------------------------------------------------------|
| `data`            | DataFrame with `location_id` (and optional `time`) + the value column. |
| `var`             | column name of the value to plot.                                    |
| `master_lookup`   | *(required)* master lookup (`location_id` -> tile with `lat`/`lon`); the grid lookup is built from it. |
| `grid_sampling`   | grid resolution (°) used to build the grid lookup (default `0.5`).   |
| `extent`          | `(lon_min, lon_max, lat_min, lat_max)`.                             |
| `k`               | number of aggregated neighbors per pixel (`1` = 1:1 mapping).       |
| `month`           | filter to a month (e.g. `"2020-01"`) using the `time` column.         |
| `stat`            | aggregation when `k > 1`: min/max/mean/median.                       |
| `title`, `cbar_label` | plot labels.                                                     |
| `cmap`            | matplotlib colormap.                                                 |
| `center_at_zero`  | symmetric color scale centered at 0.                                 |
| `value_range`     | fixed `(vmin, vmax)` color range.                                    |
| `plot_robust`     | `(low, high)` percentiles for robust color range.                    |
| `add_coastlines`  | overlay natural-earth coastlines (needs `coastlines` extra).         |
| `add_marker`      | `(style, location_id)` or list of them, placed on the map.           |
| `save_path`       | where to save the figure.                                            |
| `show_plot`       | display the figure interactively.                                    |

Returns the matplotlib figure.

> The grid lookup is auto-built from `master_lookup` and **reused for identical
> calls** — the lookup filename encodes `grid_sampling`, `extent` and `k`, so
> different parameter combinations produce separate cached files. Map lookups
> are saved automatically to a `map_lookups/` folder next to the master lookup.

### Example

```python
from plotting_joseph import plot_map

fig = plot_map(
    data=df,                        # location_id + a value column
    var="backscatter40",
    master_lookup="lookup_tables/location_id_to_tile_id.parquet",
    extent=(-180, 180, -60, 85),
    grid_sampling=0.5,
    k=1,
    plot_robust=(2, 98),
    title="Global Backscatter (2-98%)",
    save_path="figures/global_backscatter.png",
)
```

See [Lookup tables](../lookup_tables.md) for the grid lookup format.
