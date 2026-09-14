"""Tests for plot_map."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from plotting_joseph import plot_map


def _make_master(tmp_path, n_lat=10, n_lon=20, grid_sampling=1.0):
    """Build a master location_id -> tile_id (with lat/lon) lookup."""
    extent = (-180.0, 180.0, -60.0, 85.0)
    lon_min, lon_max, lat_min, lat_max = extent
    # Build a few locations spread on the grid, with the given sampling.
    n_lon_cells = int(round((lon_max - lon_min) / grid_sampling))
    n_lat_cells = int(round((lat_max - lat_min) / grid_sampling))
    rows = []
    for i in range(0, n_lat_cells, max(1, n_lat_cells // n_lat)):
        for j in range(0, n_lon_cells, max(1, n_lon_cells // n_lon)):
            rows.append(
                {
                    "location_id": int(i * n_lon_cells + j),
                    "lat": lat_max - (i + 0.5) * grid_sampling,
                    "lon": lon_min + (j + 0.5) * grid_sampling,
                    "tile_id": "tile0",
                }
            )
    df = pd.DataFrame(rows)
    path = tmp_path / "location_id_to_tile_id.parquet"
    df.to_parquet(path, index=False)
    return path, df


def test_plot_map_basic(tmp_path):
    master, master_df = _make_master(tmp_path)
    data = pd.DataFrame(
        {
            "location_id": master_df["location_id"],
            "backscatter40": np.random.RandomState(0).normal(size=len(master_df)),
        }
    )

    fig = plot_map(
        data=data,
        var="backscatter40",
        master_lookup=master,
        grid_sampling=1.0,
        show_plot=False,
    )
    assert fig is not None
    out = tmp_path / "out"
    plot_map(
        data=data,
        var="backscatter40",
        master_lookup=master,
        grid_sampling=1.0,
        save_path=out / "map.png",
        title="Test map",
    )
    assert (out / "map.png").exists()

    import matplotlib.pyplot as plt

    plt.close("all")


def test_plot_map_auto_figsize(tmp_path):
    """Verify that figsize=None auto-derives from extent so the map is not distorted."""
    import matplotlib.pyplot as plt

    master, master_df = _make_master(tmp_path)
    data = pd.DataFrame(
        {
            "location_id": master_df["location_id"],
            "backscatter40": np.random.RandomState(0).normal(size=len(master_df)),
        }
    )

    # Default extent (-180, 180, -60, 85) => lon_span=360, lat_span=145, ratio ~2.48
    fig = plot_map(
        data=data,
        var="backscatter40",
        master_lookup=master,
        grid_sampling=1.0,
        figsize=None,  # auto-derive
        show_plot=False,
    )
    w, h = fig.get_size_inches()
    # The auto-derived figsize should have width > height (world is wider than tall)
    assert w > h
    # Ratio should be roughly proportional to lon:lat span (~2.48), accounting for
    # the ~15% colorbar compensation factor
    expected_ratio = (360.0 / 145.0) / 0.85
    actual_ratio = w / h
    assert abs(actual_ratio - expected_ratio) < 0.5, f"Expected ratio ~{expected_ratio}, got {actual_ratio}"

    plt.close("all")


def test_plot_map_fill_produces_more_data_than_snap(tmp_path):
    """Fill-by-proximity colors more pixels than the exact 1:1 snap."""
    import matplotlib.pyplot as plt

    master, master_df = _make_master(tmp_path)
    data = pd.DataFrame(
        {
            "location_id": master_df["location_id"],
            "backscatter40": np.random.RandomState(0).normal(size=len(master_df)),
        }
    )

    def filled_count(max_distance_km):
        fig = plot_map(
            data=data,
            var="backscatter40",
            master_lookup=master,
            grid_sampling=1.0,
            max_distance_km=max_distance_km,
            show_plot=False,
        )
        ax = fig.axes[0]
        im = ax.get_images()[0]
        arr = im.get_array()
        plt.close(fig)
        return int(np.count_nonzero(~np.isnan(arr)))

    large = filled_count(500.0)
    tiny = filled_count(0.01)
    assert large > tiny, f"Expected fill to cover more pixels, got {large} vs {tiny}"


def test_plot_map_marker_with_fill(tmp_path):
    """Markers still resolve their home cell when 1-to-many fill is used."""
    import matplotlib.pyplot as plt

    master, master_df = _make_master(tmp_path)
    data = pd.DataFrame(
        {
            "location_id": master_df["location_id"],
            "backscatter40": np.random.RandomState(0).normal(size=len(master_df)),
        }
    )
    marker_loc = master_df["location_id"].iloc[0]
    fig = plot_map(
        data=data,
        var="backscatter40",
        master_lookup=master,
        grid_sampling=1.0,
        max_distance_km=500.0,
        add_marker=("o", int(marker_loc)),
        show_plot=False,
    )
    assert fig is not None
    plt.close(fig)


def test_plot_map_marker_respects_extent(tmp_path):
    """The marker is placed from the passed extent, never a default (worldwide) one."""
    import matplotlib.pyplot as plt

    master, master_df = _make_master(tmp_path)
    data = pd.DataFrame(
        {
            "location_id": master_df["location_id"],
            "backscatter40": np.random.RandomState(0).normal(size=len(master_df)),
        }
    )
    extent = (-20.0, 20.0, -30.0, 30.0)
    grid_sampling = 1.0

    # Pick a real master location that falls inside the regional extent.
    inside = master_df[
        (master_df["lon"] >= extent[0]) & (master_df["lon"] <= extent[1])
        & (master_df["lat"] >= extent[2]) & (master_df["lat"] <= extent[3])
    ]
    assert not inside.empty
    row_m = inside.iloc[0]
    marker_loc = int(row_m["location_id"])
    loc_lon, loc_lat = float(row_m["lon"]), float(row_m["lat"])

    # Expected home cell centre, derived from the passed extent.
    col = int(np.floor((loc_lon - extent[0]) / grid_sampling))
    row = int(np.floor((extent[3] - loc_lat) / grid_sampling))
    expected_lon = extent[0] + (col + 0.5) * grid_sampling
    expected_lat = extent[3] - (row + 0.5) * grid_sampling

    fig = plot_map(
        data=data,
        var="backscatter40",
        master_lookup=master,
        grid_sampling=grid_sampling,
        extent=extent,
        max_distance_km=500.0,
        add_marker=("o", marker_loc),
        show_plot=False,
    )
    ax = fig.axes[0]

    # The axes must reflect the passed extent, not the world default.
    assert list(ax.get_xlim()) == pytest.approx([-20.0, 20.0])
    assert list(ax.get_ylim()) == pytest.approx([-30.0, 30.0])

    # The single marker line must sit at the home cell centre from `extent`.
    marker_lines = [ln for ln in ax.lines if ln.get_marker()]
    assert len(marker_lines) == 1
    marker_x, marker_y = marker_lines[0].get_data()
    assert list(marker_x) == pytest.approx([expected_lon])
    assert list(marker_y) == pytest.approx([expected_lat])

    plt.close(fig)
