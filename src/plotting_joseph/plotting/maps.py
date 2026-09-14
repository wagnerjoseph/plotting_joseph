"""Global map plotting (renamed from ``plot_worldwide``).

This module plots a gridded map from per-location values and a lookup table
that maps ``location_id`` -> pixel on a regular grid.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:  # cartopy is used only for optional coastlines
    import cartopy.io.shapereader as shpreader  # type: ignore
    _HAS_CARTOPY = True
except Exception:  # noqa: BLE001 - optional dependency guard  # pragma: no cover
    _HAS_CARTOPY = False


def _build_pixel_mapping(lut: pd.DataFrame, k: int) -> pd.Series:
    """Build a ``location_id`` -> ``pixel_id`` mapping from a grid lookup table.

    The lookup table is expected to contain ``location_id`` and ``pixel_id``
    columns. For ``k == 1`` a direct 1:1 mapping is used. For ``k > 1`` each
    row is expected to have a ``location_ids`` column listing the original
    locations aggregated into the pixel and a ``pixel_id`` column (either a
    scalar or a list aligned with ``location_ids``).
    """
    import ast

    if k == 1:
        return lut.set_index("location_id")["pixel_id"]

    def safe_eval(x):
        return ast.literal_eval(x) if isinstance(x, str) else x

    pixel_series = (
        lut[["pixel_id", "location_ids"]]
        .assign(
            pixel_id=lut["pixel_id"].apply(safe_eval),
            location_ids=lut["location_ids"].apply(safe_eval),
        )
        .explode("location_ids")
        .rename(columns={"location_ids": "location_id"})
        .explode("pixel_id")
        .dropna(subset=["location_id", "pixel_id"])
    )
    pixel_series["pixel_id"] = pixel_series["pixel_id"].astype(int)
    pixel_series["location_id"] = pixel_series["location_id"].astype(int)
    return pixel_series.set_index("location_id")["pixel_id"]


def _aggregate_pixels(
    pixel_ids: np.ndarray, values: np.ndarray, n_pixels: int, stat: str = "median"
) -> np.ndarray:
    """Aggregate multiple values per pixel using the specified statistic."""
    stat_funcs = {
        "min": np.min,
        "max": np.max,
        "mean": np.mean,
        "median": np.median,
    }
    if stat not in stat_funcs:
        raise ValueError(f"stat must be one of {list(stat_funcs)}, got '{stat}'")
    func = stat_funcs[stat]

    image_flat = np.full(n_pixels, np.nan, dtype=np.float64)
    pixel_bins = {}
    for p, v in zip(pixel_ids, values):
        if 0 <= p < n_pixels:
            pixel_bins.setdefault(p, []).append(v)
    for p, vals in pixel_bins.items():
        image_flat[p] = func(vals)
    return image_flat


def _fill_image_from_lookup(
    lut: pd.DataFrame,
    data_sub: pd.DataFrame,
    var: str,
    n_pixels: int,
    k: int,
    stat: str,
) -> np.ndarray:
    """Build the map image directly, per pixel, from the inverted lookup.

    For ``k == 1`` ``lut`` holds every ``pixel_id`` -> ``location_id`` (``-1``
    for pixels whose closest location is beyond ``max_distance_km``). Each pixel
    shows the value of its closest location and stays ``NaN`` (white) when the
    location is missing/too far. For ``k > 1`` ``lut`` holds
    ``pixel_id -> location_ids`` lists, aggregated with ``stat``.
    """
    value_map = (
        data_sub[["location_id", var]]
        .drop_duplicates("location_id")
        .set_index("location_id")[var]
    )
    loc_to_value = value_map.to_dict()

    if k == 1:
        pids = lut["pixel_id"].to_numpy(dtype=np.int64)
        vals = lut["location_id"].map(value_map).to_numpy(dtype=np.float64)
        image_flat = np.full(n_pixels, np.nan, dtype=np.float64)
        valid = (pids >= 0) & (pids < n_pixels)
        image_flat[pids[valid]] = vals[valid]
        return image_flat

    # k > 1: aggregate the values of the closest locations per pixel.
    row_pixels: list[int] = []
    row_values: list[float] = []
    for pid, locs in zip(lut["pixel_id"], lut["location_ids"]):
        if locs is None:
            continue
        for lid in locs:
            v = loc_to_value.get(int(lid), np.nan)
            if not np.isnan(v):
                row_pixels.append(int(pid))
                row_values.append(float(v))
    return _aggregate_pixels(np.asarray(row_pixels), np.asarray(row_values), n_pixels, stat=stat)


def _get_color_norm(
    image: np.ndarray,
    center_at_zero: bool = False,
    value_range: tuple[float, float] | None = None,
):
    if value_range is None:
        value_range = (np.nanmin(image), np.nanmax(image))

    vmin, vmax = value_range
    if center_at_zero:
        vmax_symmetric = max(abs(vmin), abs(vmax))
        return TwoSlopeNorm(vmin=-vmax_symmetric, vcenter=0, vmax=vmax_symmetric)
    return plt.Normalize(vmin=vmin, vmax=vmax)


def _pixel_center(pixel: int, extent, grid_sampling: float, n_lon: int):
    """Return the ``(lon, lat)`` centre of a pixel, always from ``extent``.

    This is the single source of truth for pixel -> longitude/latitude so that
    every plotted feature (markers included) is positioned from the same
    ``extent`` parameter that defines the grid.
    """
    lon_min, _lon_max, _lat_min, lat_max = extent
    col = pixel % n_lon
    row = pixel // n_lon
    return lon_min + (col + 0.5) * grid_sampling, lat_max - (row + 0.5) * grid_sampling


def _location_home_pixel(
    master,
    location_id,
    extent,
    grid_sampling: float,
    n_lon: int,
):
    """Return ``(pixel_id, lon, lat)`` of the grid cell a location falls in.

    This resolves the *home* cell (the pixel the location's lat/lon lands in),
    independent of how the lookup was built (direct snap or inverted fill), so
    markers are always placed reliably. Returns ``None`` if the location is not
    in ``master`` or falls outside ``extent``.
    """
    m = master[master["location_id"] == location_id]
    if m.empty:
        return None
    lat = float(m["lat"].iloc[0])
    lon = float(m["lon"].iloc[0])
    lon_min, lon_max, lat_min, lat_max = extent
    n_lat = int(np.round((lat_max - lat_min) / grid_sampling))
    col = int(np.floor((lon - lon_min) / grid_sampling))
    row = int(np.floor((lat_max - lat) / grid_sampling))
    if not (0 <= row < n_lat and 0 <= col < n_lon):
        return None
    pixel = row * n_lon + col
    return pixel, *_pixel_center(pixel, extent, grid_sampling, n_lon)


def _make_title(var: str, stat: str, month: str | None = None, k: int = 1) -> str:
    """Construct a descriptive title for the plot."""
    title = var.capitalize() + (f" — {month}" if month else "")
    if k > 1:
        title += f" ({k} neighbors combined using {stat})"
    return title


def plot_map(
    data: pd.DataFrame,
    var: str,
    master_lookup: str | Path | None = None,
    month: str | None = None,
    stat: str = "median",
    title: str | None = None,
    cbar_label: str | None = None,
    cmap: str = "viridis",
    center_at_zero: bool = False,
    extent: tuple[float, float, float, float] = (-180, 180, -60, 85),
    grid_sampling: float = 0.5,
    k: int = 1,
    max_distance_km: float = 15.0,
    value_range: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    plot_robust: tuple[float, float] | None = None,
    add_coastlines: bool = False,
    add_marker: object | None = None,
    dpi: int = 300,
    figsize: tuple[int, int] | None = None,
    font_scale: float = 1.0,
    show_plot: bool = False,
):
    """Plot a global map of the given variable.

    Parameters
    ----------
    data : pd.DataFrame
        Data with ``location_id`` (and optional ``time`` for month filtering)
        columns and the variable to plot.
    var : str
        Column name of the variable to plot.
    master_lookup : str or Path, (required)
        Master lookup parquet (``location_id`` -> tile with ``lat``/``lon``).
        The grid/map lookup is built from it and reused for identical
        ``grid_sampling`` / ``extent`` / ``k``. Required; raises a clear error
        if missing.
    month : str, optional
        Filter to a specific month (e.g. ``"2020-01"``) using the ``time``
        column. If None, uses all data.
    stat : str, default="median"
        Aggregation statistic when ``k > 1``: "min", "max", "mean", "median".
    title : str, optional
        Custom title. If None, auto-generated.
    cbar_label : str, optional
        Colorbar label. If None, auto-generated.
    cmap : str, default="viridis"
        Matplotlib colormap name.
    center_at_zero : bool, default=False
        If True, center the color scale at 0 (symmetric range).
    extent : tuple, default=(-180, 180, -60, 85)
        Bounding box ``(lon_min, lon_max, lat_min, lat_max)``.
    grid_sampling : float, default=0.5
        Grid resolution in degrees used to build the grid lookup.
    k : int, default=1
        Number of aggregated neighbors per pixel (``1`` = direct 1:1 mapping).
    max_distance_km : float, default=15.0
        When ``> 0``, each grid pixel is filled with the value of its nearest
        source location, but only if that location lies within
        ``max_distance_km``. Pixels farther than this from every location are
        left blank, so proximity-filled pixels around measurements are kept
        while remote empty cells stay white. Pass ``0`` (or negative) to fall
        back to the exact 1:1 floor-snap per location (each location only fills
        the single pixel it falls in).
    value_range : tuple, optional
        Fixed color range ``(vmin, vmax)``; values outside are clipped. If
        None, uses the data min/max.
    save_path : str or Path, optional
        Where to save the figure. If None, the figure is not saved.
    plot_robust : tuple, optional
        Robust color range as percentiles ``(low, high)``, e.g. ``(2, 98)``.
    add_coastlines : bool, default=False
        If True, overlay natural-earth coastlines (requires ``cartopy``).
    add_marker : tuple or list of tuples, optional
        ``(style, location_id)`` marker(s) placed on the map.
    dpi : int, default=300
        Resolution for the saved figure.
    figsize : tuple, optional
        Figure size ``(width, height)``. If None (default), automatically
        derived from the ``extent`` so the map is not distorted (width
        proportional to lon span, height proportional to lat span).
    font_scale : float, default=1.0
        Font-size scale factor.
    show_plot : bool, default=False
        If True, display the figure interactively (otherwise it is closed).
    """

    df = data.copy() if hasattr(data, "copy") else data

    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("data must be a pandas.DataFrame")

    if month is not None:
        if "time" not in df.columns:
            raise ValueError("month filtering requires a 'time' column in data")
        df = df[df["time"] == pd.to_datetime(f"{month}-01")]

    if {"location_id", var} - set(df.columns):
        raise ValueError(
            f"data must contain columns 'location_id' and '{var}'. "
            f"Got: {list(df.columns)}. Rename columns via DataLoader column_map."
        )

    data_sub = df[["location_id", var]]

    # The map grid lookup is always derived from the master lookup.
    if master_lookup is None:
        raise ValueError(
            "plot_map requires 'master_lookup' (a location_id -> tile_id "
            "parquet with lat/lon) to build the map grid lookup."
        )

    from ..data import ensure_grid_lookup

    lookuptable_path = ensure_grid_lookup(
        master_lookup,
        grid_sampling=grid_sampling,
        extent=extent,
        k=k,
        max_distance_km=max_distance_km,
    )

    lut = pd.read_parquet(lookuptable_path)

    lon_min, lon_max, lat_min, lat_max = extent
    n_lat = int(np.round((lat_max - lat_min) / grid_sampling))
    n_lon = int(np.round((lon_max - lon_min) / grid_sampling))
    n_pixels = n_lat * n_lon

    if max_distance_km > 0:
        # Direct per-pixel nearest-location fill: every pixel shows its closest
        # location's value, and is blank (white) when that location is farther
        # than ``max_distance_km``.
        image_flat = _fill_image_from_lookup(lut, data_sub, var, n_pixels, k, stat)
    else:
        location_to_pixel = _build_pixel_mapping(lut, k)

        data_with_pixel_id = data_sub.merge(
            location_to_pixel.to_frame("pixel_id"),
            left_on="location_id",
            right_index=True,
            how="inner",
        )

        pixel_id = data_with_pixel_id["pixel_id"].to_numpy()
        values = data_with_pixel_id[var].to_numpy()

        if k == 1:
            image_flat = np.full(n_pixels, np.nan, dtype=np.float64)
            valid = (pixel_id >= 0) & (pixel_id < n_pixels)
            image_flat[pixel_id[valid]] = values[valid]
        else:
            image_flat = _aggregate_pixels(pixel_id, values, n_pixels, stat=stat)

    image = image_flat.reshape(n_lat, n_lon)

    # Guard against plotting no data: either no location_id in ``data`` fell
    # inside ``extent`` (inner merge yielded no pixels), or all plotted values
    # were NaN/missing. Fail with a clear message instead of a cryptic
    # "zero-size array" error from numpy min/max.
    if not np.any(~np.isnan(image_flat)):
        raise ValueError(
            f"No plottable data for '{var}' within extent {extent}. "
            "This usually means none of the data's location_ids fall inside "
            "the requested extent, or the variable values are all NaN. "
            "Check that 'extent' covers your locations (e.g. print "
            "data[['location_id', var]] and the master lookup lat/lon) and "
            "that grid_sampling is reasonable for the extent."
        )

    # Robust color range
    if plot_robust is not None:
        flat_data = image.ravel()
        flat_data = flat_data[~np.isnan(flat_data)]
        if len(flat_data) > 0:
            q_low, q_high = plot_robust
            value_range = (
                np.quantile(flat_data, q_low / 100.0),
                np.quantile(flat_data, q_high / 100.0),
            )

    norm = _get_color_norm(image, center_at_zero, value_range)

    if cbar_label is None:
        cbar_label = f"{stat}({var})" if k > 1 else var

    if title is None:
        title = _make_title(var, stat, month, k) + (
            " (robust)" if plot_robust is not None else ""
        )

    flat_data = image.ravel()
    flat_data = flat_data[~np.isnan(flat_data)]

    cmap_obj = plt.get_cmap(cmap)

    # Auto-derive figsize from extent if not specified, so the map is not distorted
    if figsize is None:
        base_height = 6.0
        lon_span = lon_max - lon_min
        lat_span = lat_max - lat_min
        geo_aspect = lon_span / lat_span if lat_span > 0 else 1.0
        # The right-hand colorbar/histogram strip takes ~15% of the figure
        # width; widen the figure so the map axes keep the true lon:lat aspect.
        map_width_fraction = 0.85
        figsize = (base_height * geo_aspect / map_width_fraction, base_height)

    fig, ax = plt.subplots(figsize=figsize)

    if add_coastlines:
        if not _HAS_CARTOPY:
            raise ImportError(
                "add_coastlines requires the optional 'coastlines' extra: "
                "pip install 'plotting_joseph[coastlines]'"
            )
        coast_shp = shpreader.natural_earth(
            resolution="110m", category="physical", name="coastline"
        )
        for record in shpreader.Reader(coast_shp).geometries():
            ax.plot(*record.xy, color="black", linewidth=1)

    im = ax.imshow(
        image,
        origin="upper",
        extent=[lon_min, lon_max, lat_min, lat_max],
        cmap=cmap,
        norm=norm,
        aspect="auto",
    )

    # Pin the display to the requested extent and stop autoscale. Coastlines are
    # global Line2Ds that expand the axes data limits, so later marker ``plot``
    # calls would re-autoscale the viewport out to the whole globe. Fixing the
    # limits here keeps the shown region equal to ``extent`` regardless of any
    # additional artists (coastlines, markers).
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.autoscale(False)

    marker_value = None
    if add_marker is not None:
        markers_to_plot = add_marker if isinstance(add_marker, list) else [add_marker]
        # Markers always resolve via the location's home cell from the master
        # coordinates, independent of the lookup mode.
        master_df = pd.read_parquet(master_lookup)
        marker_pixels = []
        for marker_style, marker_location_id in markers_to_plot:
            m = master_df[master_df["location_id"] == marker_location_id]
            if m.empty:
                print(
                    f"  Warning: location_id={marker_location_id} not found in the "
                    "master lookup; marker skipped."
                )
                continue
            lat = float(m["lat"].iloc[0])
            lon = float(m["lon"].iloc[0])
            if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
                print(
                    f"  Warning: marker location_id={marker_location_id} at "
                    f"({lat:.2f}, {lon:.2f}) lies outside the requested extent "
                    f"(lon {lon_min:.2f}..{lon_max:.2f}, lat {lat_min:.2f}.."
                    f"{lat_max:.2f}); marker skipped."
                )
                continue
            home = _location_home_pixel(
                master_df, marker_location_id, extent, grid_sampling, n_lon
            )
            if home is None:
                print(
                    f"  Warning: marker location_id={marker_location_id} falls "
                    "outside the requested extent; marker skipped."
                )
                continue
            marker_pixel, lon, lat = home
            marker_pixels.append((marker_style, marker_pixel, lon, lat))
            row = marker_pixel // n_lon
            col = marker_pixel % n_lon
            ax.plot(
                lon,
                lat,
                marker_style,
                color="red",
                markersize=10,
                markeredgewidth=2,
            )
            current_marker_value = image[row, col]
            if marker_value is None:
                marker_value = current_marker_value
            if value_range is not None:
                vmin, vmax = value_range
                current_marker_value = np.clip(
                    current_marker_value,
                    vmin + 0.005 * (vmax - vmin),
                    vmax - 0.005 * (vmax - vmin),
                )

    divider = make_axes_locatable(ax)
    hist_ax = divider.append_axes("right", size="10%", pad=0.07, sharey=None)

    if value_range is not None:
        vmin, vmax = value_range
    elif plot_robust is not None:
        q_low, q_high = plot_robust
        vmin = np.quantile(flat_data, q_low / 100.0)
        vmax = np.quantile(flat_data, q_high / 100.0)
    else:
        vmin, vmax = flat_data.min(), flat_data.max()

    flat_data_clipped = np.clip(flat_data, vmin, vmax)
    unique_vals = np.unique(flat_data)

    if len(unique_vals) == 1:
        val = unique_vals[0]
        bins = [val - 0.5, val + 0.5]
        counts = np.array([len(flat_data)])
        hist_ax.barh(
            y=val,
            width=counts[0],
            height=bins[1] - bins[0],
            color=cmap_obj(im.norm(val)),
            edgecolor="k",
        )
    else:
        number_of_bins = 100
        counts, bins, patches = hist_ax.hist(
            flat_data_clipped,
            bins=number_of_bins,
            range=(vmin, vmax),
            orientation="horizontal",
        )
        for patch, y0, y1 in zip(patches, bins[:-1], bins[1:]):
            y_center = 0.5 * (y0 + y1)
            patch.set_facecolor(cmap_obj(im.norm(y_center)))

    hist_ax.xaxis.set_visible(False)
    hist_ax.yaxis.set_visible(False)
    hist_ax.invert_xaxis()

    if add_marker is not None:
        marker_values = []
        for _style, pixel, _lon, _lat in marker_pixels:
            row = pixel // n_lon
            col = pixel % n_lon
            marker_val = image[row, col]
            if not np.isnan(marker_val):
                marker_values.append(marker_val)
        for marker_val in marker_values:
            hist_ax.axhline(marker_val, color="red", linewidth=1, alpha=0.7)

    if center_at_zero and vmin <= 0 <= vmax:
        hist_ax.axhline(0, color="black", linewidth=0.4)

    BASE_TITLE_SIZE = 17
    BASE_CBAR_LABEL_SIZE = 12
    BASE_TICK_SIZE = 10

    title_size = BASE_TITLE_SIZE * font_scale
    cbar_label_size = BASE_CBAR_LABEL_SIZE * font_scale
    tick_size = BASE_TICK_SIZE * font_scale

    cax = divider.append_axes("right", size="5%", pad=0.07, sharey=hist_ax)
    cbar = fig.colorbar(im, cax=cax, label=cbar_label)
    cbar.ax.set_ylabel(cbar_label, fontsize=cbar_label_size)
    cbar.ax.tick_params(labelsize=tick_size)

    fig.suptitle(title, fontsize=title_size, y=0.94)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show_plot:
        plt.show()

    return fig
