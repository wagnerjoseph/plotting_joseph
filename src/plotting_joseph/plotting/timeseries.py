"""Multi-panel time series plotting.

This module provides :func:`plot_time_series`, a flexible multi-panel time
series plotter supporting overlays, secondary axes, correlation annotations,
season markers, threshold shading (per-panel or across all panels) and
nearest-neighbor background series.
"""

from __future__ import annotations

import pickle
import warnings
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.dates import YearLocator, num2date
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import ScaledTranslation

try:
    import dask.dataframe as dd  # type: ignore
    _HAS_DASK = True
except Exception:  # noqa: BLE001 - optional dependency guard  # pragma: no cover
    dd = None
    _HAS_DASK = False

from ..config import LookupTables


def _as_pandas(data):
    """Return a pandas DataFrame, computing from a dask DataFrame if needed."""
    if _HAS_DASK and isinstance(data, dd.DataFrame):
        return data.compute()
    return data


def _align_axes_zero(ax1, ax2, data1_min, data1_max, data2_min, data2_max):
    """Align zero points of two y-axes based on combined data characteristics.

    Parameters
    ----------
    ax1, ax2 : matplotlib axes
        Primary and secondary y-axes.
    data1_min, data1_max : float
        Data range for the primary axis (all variables combined).
    data2_min, data2_max : float
        Data range for the secondary axis (all variables combined).
    """
    range1 = data1_max - data1_min
    range2 = data2_max - data2_min

    if range1 > 0:
        p1 = abs(data1_min) / range1 if data1_min < 0 < data1_max else None
    else:
        p1 = None

    if range2 > 0:
        p2 = abs(data2_min) / range2 if data2_min < 0 < data2_max else None
    else:
        p2 = None

    if p1 is not None and p2 is not None:
        p = (p1 + p2) / 2
    elif p1 is not None:
        p = p1
    elif p2 is not None:
        p = p2
    else:
        if (data1_min > 0 and data2_min > 0) or (data1_max < 0 and data2_max < 0):
            p = 0.0 if data1_min > 0 and data2_min > 0 else 1.0
        else:
            p = 0.5

    if p < 0.01:
        p = 0.01
    elif p > 0.99:
        p = 0.99

    range1_needed = max(
        abs(data1_min) / p if data1_min < 0 else 0,
        data1_max / (1 - p) if data1_max > 0 else 0,
    )
    range1_final = range1_needed * 1.02

    range2_needed = max(
        abs(data2_min) / p if data2_min < 0 else 0,
        data2_max / (1 - p) if data2_max > 0 else 0,
    )
    range2_final = range2_needed * 1.02

    ax1.set_ylim(-p * range1_final, (1 - p) * range1_final)
    ax2.set_ylim(-p * range2_final, (1 - p) * range2_final)


class Timeseries:
    """Static helpers and plotter for time series data."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def compute_correlation(
        df: pd.DataFrame,
        var1: str,
        var2: str,
        methods: tuple[str, ...] = ("pearson", "spearman"),
    ) -> dict:
        """Compute Pearson and/or Spearman correlation between two variables.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the variables.
        var1, var2 : str
            Column names to correlate.
        methods : tuple of str, default=("pearson", "spearman")
            Correlation method(s) to compute.

        Returns
        -------
        dict
            Mapping of method name -> correlation coefficient. A value is NaN
            if fewer than 3 valid (non-NaN) pairs are available.
        """
        clean = df[[var1, var2]].dropna()

        result = {}
        for method in methods:
            if len(clean) < 3:
                result[method] = np.nan
                continue
            corr_matrix = clean.corr(method=method)
            result[method] = corr_matrix.loc[var1, var2]
        return result

    @staticmethod
    def apply_transform(b: pd.DataFrame, var: str, transform: dict) -> pd.Series:
        """Apply a named transform to a variable column."""
        if transform["type"] == "rolling_mean":
            return b[var].rolling(window=transform["window"], center=True).mean()
        # Add more transformations here
        raise ValueError(f"Unknown transform type: {transform.get('type')}")

    @staticmethod
    def _add_threshold_bands(
        ax, time, values, threshold_value, color, alpha=0.15, below=True
    ):
        """Shade vertical bands on a single axis where values exceed a threshold."""
        time_vals = time.values
        y_vals = values.values

        if below:
            mask = ~np.isnan(y_vals) & (y_vals < threshold_value)
        else:
            mask = ~np.isnan(y_vals) & (y_vals > threshold_value)

        i = 0
        n = len(mask)
        while i < n:
            if mask[i]:
                start_idx = i
                while i < n and mask[i]:
                    i += 1
                end_idx = i - 1
                ax.axvspan(
                    time_vals[start_idx],
                    time_vals[end_idx],
                    color=color,
                    alpha=alpha,
                    linewidth=0,
                )
            else:
                i += 1

    @staticmethod
    def _draw_global_threshold_bands(
        axes: np.ndarray,
        time: pd.Series,
        values: pd.Series,
        threshold_value: float,
        color: str,
        alpha: float = 0.15,
        below: bool = True,
    ):
        """Draw vertical threshold bands across ALL panels."""
        time_vals = time.values
        y_vals = values.values

        if below:
            mask = ~np.isnan(y_vals) & (y_vals < threshold_value)
        else:
            mask = ~np.isnan(y_vals) & (y_vals > threshold_value)

        i = 0
        n = len(mask)
        while i < n:
            if mask[i]:
                start_idx = i
                while i < n and mask[i]:
                    i += 1
                end_idx = i - 1
                for ax in axes:
                    ax.axvspan(
                        time_vals[start_idx],
                        time_vals[end_idx],
                        color=color,
                        alpha=alpha,
                        linewidth=0,
                        zorder=0,
                    )
            else:
                i += 1

    @staticmethod
    def _plot_panel(
        ax,
        b,
        var_spec,
        neighbors_data=None,
        font_scale: float = 1.0,
        sec_ax=None,
        align_zero=False,
    ):

        var = var_spec["name"]
        color = var_spec.get("color", "tab:blue")
        label = var_spec.get("label", var)
        transforms = var_spec.get("transforms", [])
        show_seasons = var_spec.get("show_seasons", False)
        interpolate_nan = var_spec.get("interpolate", False)
        line_width = var_spec.get("line_width", 1.5)
        alpha = var_spec.get("alpha", 1)
        plotstyle = var_spec.get("plotstyle", "line")
        point_size = var_spec.get("point_size", 10)
        point_linewidth = var_spec.get("point_linewidth", 0.5)
        add_second_axis = var_spec.get("add_second_axis", False)
        lower_treshold = var_spec.get("lower_treshold")
        upper_treshold = var_spec.get("upper_treshold")

        BASE_LABEL_SIZE = 12
        BASE_LEGEND_SIZE = 10
        BASE_TICK_SIZE = 10

        label_size = BASE_LABEL_SIZE * font_scale
        legend_size = BASE_LEGEND_SIZE * font_scale
        tick_size = BASE_TICK_SIZE * font_scale

        plot_ax = sec_ax if (add_second_axis and sec_ax is not None) else ax

        if neighbors_data is not None:
            for ndf in neighbors_data.values():
                plot_ax.plot(ndf["time"], ndf[var], color=color, lw=0.5, alpha=0.2)

        y_main = b[var].interpolate() if interpolate_nan else b[var]

        if plotstyle == "line":
            plot_ax.plot(
                b["time"], y_main, color=color, lw=line_width, label=label, alpha=alpha
            )
        elif plotstyle == "points":
            plot_ax.scatter(
                b["time"],
                y_main,
                s=point_size,
                c=color,
                alpha=alpha,
                linewidth=point_linewidth,
                label=label,
            )
        elif plotstyle == "both":
            plot_ax.plot(
                b["time"], y_main, color=color, lw=line_width, label=label, alpha=alpha
            )
            plot_ax.scatter(
                b["time"],
                y_main,
                s=point_size,
                c=color,
                alpha=0.5,
                linewidth=point_linewidth,
            )

        if lower_treshold is not None:
            lower_val, lower_color = lower_treshold
            Timeseries._add_threshold_bands(
                plot_ax, b["time"], y_main, lower_val, lower_color, alpha=0.15, below=True
            )
        if upper_treshold is not None:
            upper_val, upper_color = upper_treshold
            Timeseries._add_threshold_bands(
                plot_ax, b["time"], y_main, upper_val, upper_color, alpha=0.15, below=False
            )

        for tr in transforms:
            y = Timeseries.apply_transform(b, var, tr)
            tr_plotstyle = tr.get("plotstyle", "line")
            tr_point_size = tr.get("point_size", point_size)

            if tr_plotstyle == "line":
                style = tr.get("style", {})
                plot_ax.plot(
                    b["time"], y, color=color, label=tr.get("label", tr["type"]), **style
                )
            elif tr_plotstyle == "points":
                plot_ax.scatter(
                    b["time"],
                    y,
                    s=tr_point_size,
                    c=color,
                    alpha=alpha,
                    label=tr.get("label", tr["type"]),
                    linewidth=point_linewidth,
                )
            elif tr_plotstyle == "both":
                style = tr.get("style", {})
                plot_ax.plot(
                    b["time"], y, color=color, label=tr.get("label", tr["type"]), **style
                )
                plot_ax.scatter(
                    b["time"],
                    y,
                    s=tr_point_size,
                    c=color,
                    alpha=0.5,
                    linewidth=point_linewidth,
                )

        if show_seasons:
            jja = b["time"].dt.month.isin([6, 7, 8])
            djf = b["time"].dt.month.isin([12, 1, 2])
            plot_ax.scatter(
                b["time"][jja], b[var][jja], s=10, c="red", alpha=0.6, label="JJA"
            )
            plot_ax.scatter(
                b["time"][djf], b[var][djf], s=10, c="black", alpha=0.6, label="DJF"
            )

        if not plot_ax.get_ylabel():
            plot_ax.set_ylabel(label, fontsize=label_size)

        if sec_ax is not None and plot_ax == sec_ax:
            plot_ax.tick_params(axis="y", labelsize=tick_size)
            plot_ax.yaxis.set_label_position("right")
            plot_ax.set_ylabel(label, fontsize=label_size)

        ax.xaxis.set_major_locator(YearLocator(1))
        ax.xaxis.set_major_formatter(
            FuncFormatter(
                lambda x, pos: f"{int(num2date(x).year)}"
                if int(num2date(x).year) % 2 == 0
                else ""
            )
        )
        ax.grid(True)

        plot_ax.legend(
            loc="upper left" if plot_ax == ax else "upper right", fontsize=legend_size
        )

        ax.tick_params(axis="x", labelsize=tick_size, rotation=45)
        ax.tick_params(axis="y", labelsize=tick_size)
        offset = ScaledTranslation(4 / 72, 0, ax.figure.dpi_scale_trans)
        for label in ax.get_xticklabels():
            label.set_transform(label.get_transform() + offset)

    @staticmethod
    def get_neighborhood(
        loc_id: int,
        neighbors_df: pd.DataFrame,
        data: pd.DataFrame,
        k: int,
        max_distance_km: float,
    ):
        """Return up to ``k`` nearest neighbor time series for ``loc_id``.

        Parameters
        ----------
        loc_id : int
            Center location id.
        neighbors_df : pd.DataFrame
            Neighbor lookup with columns ``location_id``, ``neighbor_location_id``,
            ``distance_km``, ``rank``.
        data : pd.DataFrame
            Full time series data.
        k : int
            Maximum number of neighbors to return.
        max_distance_km : float
            Maximum neighbor distance; 0/negative disables filtering.
        """
        loc_neighbors = neighbors_df[
            neighbors_df["location_id"] == loc_id
        ].copy()

        from ..data import MAX_NEIGHBOR_DISTANCE_KM

        if max_distance_km > 0:
            max_distance_km = min(max_distance_km, MAX_NEIGHBOR_DISTANCE_KM)
            loc_neighbors = loc_neighbors[
                loc_neighbors["distance_km"] <= max_distance_km
            ]

        loc_neighbors = loc_neighbors.sort_values("rank").head(k)

        if loc_neighbors.empty:
            return None, 0

        neighbor_ids = loc_neighbors["neighbor_location_id"].values
        neighbor_data_df = data[data["location_id"].isin(neighbor_ids)]

        neighbors_data = {
            nid: neighbor_data_df[neighbor_data_df["location_id"] == nid].sort_values(
                "time"
            )
            for nid in neighbor_ids
        }
        return neighbors_data, len(neighbor_ids)

    # ------------------------------------------------------------------
    # Main plotter
    # ------------------------------------------------------------------
    @staticmethod
    def plot_time_series(
        data,
        var_specs: list[dict] | None = None,
        location_ids: Iterable[int] | None = None,
        random_points: tuple[int, int] = (2, 123),
        add_closest_points: tuple[int, float] = (0, 0),
        lookup_tables: LookupTables | None = None,
        master_lookup: str | Path | None = None,
        save_dir: str | Path | None = None,
        figsize: tuple[int, int] = (10, 5),
        font_scale: float = 1.0,
        show_plot: bool = False,
    ) -> list:
        """Plot multi-panel time series for selected locations.

        Creates one multi-panel figure per selected location. Each variable is
        drawn on its own panel; overlays (via ``add_to``) can be placed on an
        existing panel, optionally on a secondary y-axis.

        Parameters
        ----------
        data : pandas.DataFrame or dask.DataFrame
            Time series data with at least ``location_id`` and ``time`` columns
            plus the numeric variables to plot.
        var_specs : list of dict, optional
            Variable specifications (see notes below). If None, numeric
            columns are inferred automatically.
        location_ids : iterable of int, optional
            Locations to plot. If None, ``random_points`` locations are chosen.
        random_points : tuple, default=(2, 123)
            ``(n_points, seed)`` for random location selection (only when
            ``location_ids`` is None).
        add_closest_points : tuple, default=(0, 0)
            ``(k_closest, max_distance_km)`` enabling nearest-neighbor
            background series. Set to ``(0, 0)`` to disable.
        lookup_tables : LookupTables, optional
            Provides the ``location_ids`` table (for the location->tile_id
            mapping), ``countries`` pickle, and ``neighbors_dir`` directory.
        master_lookup : str or Path, optional
            Master lookup parquet (``location_id`` -> tile with ``lat``/``lon``).
            Alternative to ``lookup_tables``. The countries lookup is automatically
            resolved online (reverse geocoding) and cached for reuse; country
            names fall back to "Unknown" only if generation fails (e.g. no
            internet on first use). When ``add_closest_points`` is used, the
            neighbor lookup is also generated on demand and cached.
        save_dir : str or Path, optional
            Save each location figure as ``{save_dir}/{location_id}.png``.
        figsize : tuple, default=(10, 5)
            Base figure size; height scales with the number of panels.
        font_scale : float, default=1.0
            Font-size scale factor.
        show_plot : bool, default=False
            If True, display figures interactively.

        Returns
        -------
        list
            The matplotlib figure objects (one per location).

        Notes on ``var_specs``
        ----------------------
        Each spec dict may contain:
            - 'name' (required): variable column name.
            - 'label': legend / y-axis label.
            - 'color': line color.
            - 'line_width', 'alpha': line styling.
            - 'plotstyle': 'line', 'points' or 'both'.
            - 'show_seasons': overlay JJA/DJF markers.
            - 'interpolate': interpolate NaNs.
            - 'transforms': list of dicts (e.g. {'type': 'rolling_mean', 'window': 12}).
            - 'add_to': parent variable to overlay onto.
            - 'add_second_axis': put overlay on a right-hand secondary y-axis.
            - 'align_zero': align the zero points of both axes.
            - 'compute_corr': show Pearson+Spearman correlation with the parent.
            - 'lower_treshold': (value, color) shade where values below value.
            - 'upper_treshold': (value, color) shade where values above value.
            - 'apply_shading_to_all': extend threshold shading to all panels.
        """
        import matplotlib.pyplot as plt

        data = _as_pandas(data)

        # --- Prepare neighbor request ---
        k_closest, max_distance_km = (0, 0)
        if add_closest_points is not None:
            k_closest, max_distance_km = add_closest_points

        # Cap the neighbor search distance so lookups stay small.
        from ..data import MAX_NEIGHBOR_DISTANCE_KM

        if max_distance_km > 0:
            max_distance_km = min(max_distance_km, MAX_NEIGHBOR_DISTANCE_KM)

        # --- Generate lookup tables from the master lookup (on demand) ---
        if lookup_tables is None and master_lookup is not None:
            from ..data import (
                ensure_country_lookup,
                ensure_location_ids,
                ensure_neighbor_lookup,
            )

            location_ids_path = ensure_location_ids(master_lookup)
            # Countries are included by default; fall back to "Unknown" only if
            # generation fails (e.g. no internet on the first country lookup).
            try:
                countries = ensure_country_lookup(master_lookup)
            except (ImportError, OSError) as e:
                warnings.warn(
                    "Could not generate country lookup; country names will be "
                    f"'Unknown'. Reverse geocoding downloads a GeoNames snapshot "
                    f"on first use, so internet access is required: {e}"
                )
                countries = None
            neighbors_dir = (
                ensure_neighbor_lookup(master_lookup, k_closest, max_distance_km)
                if k_closest > 0
                else None
            )
            lookup_tables = LookupTables(
                location_ids=location_ids_path,
                countries=countries,
                neighbors_dir=neighbors_dir,
            )
        lookup_tables = lookup_tables or LookupTables()

        # --- Select locations ---
        if location_ids is None:
            n_points, seed = random_points
            rng = np.random.default_rng(seed)
            all_ids = data["location_id"].unique()
            location_ids = rng.choice(all_ids, size=n_points, replace=False)

        # --- Filter main data ---
        data_sel = data[data["location_id"].isin(location_ids)].sort_values(
            ["location_id", "time"]
        )

        locid_to_tile = {}
        neighbors_by_tile = {}

        if k_closest > 0:
            if lookup_tables.location_ids is None:
                raise ValueError(
                    "add_closest_points requires a 'location_ids' lookup table "
                    "with a 'tile_id' column."
                )
            tile_lookup_table = pd.read_parquet(lookup_tables.location_ids)
            if "tile_id" not in tile_lookup_table.columns:
                tile_lookup_table["tile_id"] = "tile0"
            filtered_lookup = tile_lookup_table[
                tile_lookup_table["location_id"].isin(location_ids)
            ]
            locid_to_tile = dict(
                zip(filtered_lookup["location_id"], filtered_lookup["tile_id"])
            )
            if lookup_tables.neighbors_dir is None:
                raise ValueError(
                    "add_closest_points requires a 'neighbors_dir' lookup table."
                )
            for tile_id in filtered_lookup["tile_id"].unique():
                neighbor_file = lookup_tables.neighbors_dir / f"{tile_id}.parquet"
                neighbors_by_tile[tile_id] = (
                    pd.read_parquet(neighbor_file) if neighbor_file.exists() else None
                )

        # --- Build variable specifications ---
        if var_specs is None:
            exclude = {"time", "location_id"}
            inferred_vars = [
                c
                for c in data.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(data[c])
            ]
            var_specs = [
                {
                    "name": v,
                    "label": v.replace("_", " ").upper(),
                    "color": "royalblue",
                    "show_seasons": False,
                    "transforms": [],
                    "interpolate": False,
                    "plotting_styles": "",
                }
                for v in inferred_vars
            ]

        # --- Split panel specs and overlays ---
        panel_specs = []
        overlays_by_parent = {}
        for spec in var_specs:
            parent = spec.get("add_to")
            if parent is None:
                panel_specs.append(spec)
            else:
                overlays_by_parent.setdefault(parent, []).append(spec)

        # --- Validate correlation requests ---
        for parent_name, overlay_list in overlays_by_parent.items():
            corr_requests = [s for s in overlay_list if s.get("compute_corr", False)]
            if len(corr_requests) > 1:
                raise ValueError(
                    f"Cannot compute correlation for panel '{parent_name}': multiple "
                    f"overlays request correlation. Requires exactly 2 lines "
                    f"(1 parent + 1 overlay)."
                )
            if len(corr_requests) == 1 and len(overlay_list) > 1:
                raise ValueError(
                    f"Cannot compute correlation for panel '{parent_name}': requires "
                    f"exactly 2 lines total (1 parent + 1 overlay), but "
                    f"{len(overlay_list)} overlays were added."
                )

        # --- Collect global thresholds ---
        global_thresholds = []
        for spec in var_specs:
            if not spec.get("apply_shading_to_all", False):
                continue
            var = spec.get("name")
            if spec.get("lower_treshold") is not None:
                val, color = spec["lower_treshold"]
                global_thresholds.append(
                    {"var": var, "value": val, "color": color, "below": True, "alpha": 0.15}
                )
            if spec.get("upper_treshold") is not None:
                val, color = spec["upper_treshold"]
                global_thresholds.append(
                    {"var": var, "value": val, "color": color, "below": False, "alpha": 0.15}
                )

        # --- Load country lookup (optional) ---
        country_lookup = {}
        if lookup_tables.countries is not None and lookup_tables.countries.exists():
            with open(lookup_tables.countries, "rb") as f:
                country_lookup = pickle.load(f)

        figures = []

        # --- Loop over locations ---
        for loc_id in location_ids:
            b = data_sel[data_sel["location_id"] == loc_id].sort_values("time")

            # --- Load neighbors ---
            neighbors_data, num_neighbors = None, 0
            if k_closest > 0:
                tile_id = locid_to_tile.get(loc_id)
                if tile_id is not None and neighbors_by_tile.get(tile_id) is not None:
                    neighbors_data, num_neighbors = Timeseries.get_neighborhood(
                        loc_id=loc_id,
                        neighbors_df=neighbors_by_tile[tile_id],
                        data=data,
                        k=k_closest,
                        max_distance_km=max_distance_km,
                    )

            # --- Create figure dynamically ---
            n_panels = len(panel_specs)
            fig, axes = plt.subplots(
                nrows=n_panels,
                ncols=1,
                figsize=(figsize[0], figsize[1] * max(1, n_panels / 3)),
                sharex=True,
            )
            axes = np.atleast_1d(axes)

            sec_axes = {}
            sec_data_ranges = {}
            primary_data_ranges = {}
            panels_with_alignment = set()
            correlations_to_show = {}

            for ax_idx, (ax, panel_spec) in enumerate(zip(axes, panel_specs)):
                primary_data_ranges[ax_idx] = {
                    "min": b[panel_spec["name"]].min(),
                    "max": b[panel_spec["name"]].max(),
                }

                Timeseries._plot_panel(
                    ax=ax,
                    b=b,
                    var_spec=panel_spec,
                    neighbors_data=neighbors_data,
                    font_scale=font_scale,
                )

                for overlay_spec in overlays_by_parent.get(panel_spec["name"], []):
                    if overlay_spec.get("compute_corr", False):
                        corr_value = Timeseries.compute_correlation(
                            b, panel_spec["name"], overlay_spec["name"]
                        )
                        correlations_to_show[ax_idx] = corr_value

                    if overlay_spec.get("add_second_axis", False):
                        if ax_idx not in sec_axes:
                            sec_ax = ax.twinx()
                            sec_axes[ax_idx] = sec_ax
                            sec_ax.grid(True, alpha=0.3)
                            sec_data_ranges[ax_idx] = {
                                "min": b[overlay_spec["name"]].min(),
                                "max": b[overlay_spec["name"]].max(),
                            }
                        else:
                            sec_ax = sec_axes[ax_idx]
                            sec_data_ranges[ax_idx]["min"] = min(
                                sec_data_ranges[ax_idx]["min"],
                                b[overlay_spec["name"]].min(),
                            )
                            sec_data_ranges[ax_idx]["max"] = max(
                                sec_data_ranges[ax_idx]["max"],
                                b[overlay_spec["name"]].max(),
                            )

                        if overlay_spec.get("align_zero", False):
                            panels_with_alignment.add(ax_idx)

                        Timeseries._plot_panel(
                            ax=ax,
                            b=b,
                            var_spec=overlay_spec,
                            sec_ax=sec_ax,
                            align_zero=False,
                            font_scale=font_scale,
                        )
                    else:
                        Timeseries._plot_panel(
                            ax=ax,
                            b=b,
                            var_spec=overlay_spec,
                            neighbors_data=None,
                            font_scale=font_scale,
                        )

            axes[-1].set_xlabel("Time", fontsize=12 * font_scale)

            # Apply zero alignment after all variables are plotted
            for ax_idx in panels_with_alignment:
                if (
                    ax_idx in sec_axes
                    and ax_idx in sec_data_ranges
                    and ax_idx in primary_data_ranges
                ):
                    data1_min = primary_data_ranges[ax_idx]["min"]
                    data1_max = primary_data_ranges[ax_idx]["max"]
                    data2_min = sec_data_ranges[ax_idx]["min"]
                    data2_max = sec_data_ranges[ax_idx]["max"]
                    _align_axes_zero(
                        axes[ax_idx],
                        sec_axes[ax_idx],
                        data1_min,
                        data1_max,
                        data2_min,
                        data2_max,
                    )
                    axes[ax_idx].axhline(
                        y=0,
                        color="black",
                        linestyle="-",
                        linewidth=1.5,
                        alpha=0.8,
                        zorder=10,
                    )

            # Correlation annotations
            if correlations_to_show:
                for panel_idx, corr_value in correlations_to_show.items():
                    valid = {
                        method: value
                        for method, value in corr_value.items()
                        if not np.isnan(value)
                    }
                    if not valid:
                        continue
                    corr_text = "\n".join(
                        f"{method}: {value:.3f}" for method, value in valid.items()
                    )
                    ax = axes[panel_idx]
                    sec_ax = sec_axes.get(panel_idx)
                    legend = None
                    if sec_ax is not None:
                        legend = sec_ax.get_legend()
                    if legend is None:
                        legend = ax.get_legend()
                    if legend is not None:
                        legend_bbox = legend.get_window_extent(fig.canvas.get_renderer())
                        legend_y_center = legend_bbox.y0 + legend_bbox.height / 2
                        inv = ax.transAxes.inverted()
                        _, y_norm = inv.transform((0, legend_y_center))
                        y_pos = y_norm
                    else:
                        y_pos = 0.9
                    ax.text(
                        0.5,
                        y_pos,
                        corr_text,
                        transform=ax.transAxes,
                        fontsize=10 * font_scale,
                        verticalalignment="center",
                        horizontalalignment="center",
                        bbox={
                            "boxstyle": "round",
                            "facecolor": "white",
                            "edgecolor": "0.8",
                            "alpha": 0.8,
                        },
                    )

            plt.tight_layout()

            # Global threshold bands across all panels
            if global_thresholds:
                for thresh in global_thresholds:
                    if thresh["var"] in b.columns:
                        Timeseries._draw_global_threshold_bands(
                            axes=axes,
                            time=b["time"],
                            values=b[thresh["var"]],
                            threshold_value=thresh["value"],
                            color=thresh["color"],
                            alpha=thresh["alpha"],
                            below=thresh["below"],
                        )

            # Country for title
            country = country_lookup.get(int(loc_id), "Unknown")

            # Main figure title
            BASE_TITLE_SIZE = 16
            title_size = BASE_TITLE_SIZE * font_scale

            if k_closest > 0 and max_distance_km > 0:
                neighbor_text = (
                    f"with {num_neighbors} nearest points closer than {max_distance_km}km"
                )
                fig.suptitle(
                    f"location_id = {loc_id} ({country}) - {neighbor_text}",
                    fontsize=title_size,
                    y=1.02,
                )
            elif k_closest > 0 and max_distance_km == 0:
                neighbor_text = f"with {k_closest} closest points"
                fig.suptitle(
                    f"location_id = {loc_id} ({country}) — {neighbor_text}",
                    fontsize=title_size,
                    y=1.02,
                )
            else:
                fig.suptitle(
                    f"location_id = {loc_id} ({country})", fontsize=title_size, y=1.02
                )

            # Save or show
            if save_dir is not None:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_dir / f"{loc_id}.png", dpi=400, bbox_inches="tight")

            if show_plot:
                plt.show()

            figures.append(fig)

        return figures


# -----------------------------------------------------------------------------
# Module-level public API (mirrors plot_map style)
# -----------------------------------------------------------------------------
def plot_time_series(
    data,
    var_specs: list[dict] | None = None,
    location_ids: Iterable[int] | None = None,
    random_points: tuple[int, int] = (2, 123),
    add_closest_points: tuple[int, float] = (0, 0),
    lookup_tables: LookupTables | None = None,
    master_lookup: str | Path | None = None,
    save_dir: str | Path | None = None,
    figsize: tuple[int, int] = (10, 5),
    font_scale: float = 1.0,
    show_plot: bool = False,
) -> list:
    """Plot multi-panel time series for selected locations.

    Creates one multi-panel figure per selected location. Each variable is
    drawn on its own panel; overlays (via ``add_to``) can be placed on an
    existing panel, optionally on a secondary y-axis.

    Parameters
    ----------
    data : pandas.DataFrame or dask.DataFrame
        Time series data with at least ``location_id`` and ``time`` columns
        plus the numeric variables to plot.
    var_specs : list of dict, optional
        Variable specifications (see notes below). If None, numeric
        columns are inferred automatically.
    location_ids : iterable of int, optional
        Locations to plot. If None, ``random_points`` locations are chosen.
    random_points : tuple, default=(2, 123)
        ``(n_points, seed)`` for random location selection (only when
        ``location_ids`` is None).
    add_closest_points : tuple, default=(0, 0)
        ``(k_closest, max_distance_km)`` enabling nearest-neighbor
        background series. Set to ``(0, 0)`` to disable.
    lookup_tables : LookupTables, optional
        Provides the ``location_ids`` table (for the location->tile_id
        mapping), ``countries`` pickle, and ``neighbors_dir`` directory.
    master_lookup : str or Path, optional
        Master lookup parquet (``location_id`` -> tile with ``lat``/``lon``).
        If provided and ``lookup_tables`` is None, the country lookup is
        auto-generated from the web (reverse geocoding) and cached for reuse.
        Country names fall back to "Unknown" only if generation fails (e.g. no
        internet on first use).
    save_dir : str or Path, optional
        Save each location figure as ``{save_dir}/{location_id}.png``.
    figsize : tuple, default=(10, 5)
        Base figure size; height scales with the number of panels.
    font_scale : float, default=1.0
        Font-size scale factor.
    show_plot : bool, default=False
        If True, display figures interactively.

    Returns
    -------
    list
        The matplotlib figure objects (one per location).

    Notes on ``var_specs``
    ----------------------
    Each spec dict may contain:
        - 'name' (required): variable column name.
        - 'label': legend / y-axis label.
        - 'color': line color.
        - 'line_width', 'alpha': line styling.
        - 'plotstyle': 'line', 'points' or 'both'.
        - 'show_seasons': overlay JJA/DJF markers.
        - 'interpolate': interpolate NaNs.
        - 'transforms': list of dicts (e.g. {'type': 'rolling_mean', 'window': 12}).
        - 'add_to': parent variable to overlay onto.
        - 'add_second_axis': put overlay on a right-hand secondary y-axis.
        - 'align_zero': align the zero points of both axes.
        - 'compute_corr': show Pearson+Spearman correlation with the parent.
        - 'lower_treshold': (value, color) shade where values below value.
        - 'upper_treshold': (value, color) shade where values above value.
        - 'apply_shading_to_all': extend threshold shading to all panels.
    """
    return Timeseries.plot_time_series(
        data=data,
        var_specs=var_specs,
        location_ids=location_ids,
        random_points=random_points,
        add_closest_points=add_closest_points,
        lookup_tables=lookup_tables,
        master_lookup=master_lookup,
        save_dir=save_dir,
        figsize=figsize,
        font_scale=font_scale,
        show_plot=show_plot,
    )
