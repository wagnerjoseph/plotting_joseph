"""Tests for config-less, master-lookup driven auto-generation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plotting_joseph import (
    Timeseries,
    ensure_country_lookup,
    ensure_grid_lookup,
    ensure_location_ids,
    ensure_neighbor_lookup,
    plot_map,
)


@pytest.fixture
def master(tmp_path):
    """A small master lookup (location_id -> tile_id with lat/lon)."""
    rng = np.random.default_rng(0)
    n = 300
    lat = rng.uniform(-10, 10, n)
    lon = rng.uniform(-20, 20, n)
    loc_id = np.arange(2000, 2000 + n)
    tile = [f"t{i % 4}" for i in range(n)]
    df = pd.DataFrame(
        {"location_id": loc_id, "lat": lat, "lon": lon, "tile_id": tile}
    )
    path = tmp_path / "location_id_to_tile_id.parquet"
    df.to_parquet(path, index=False)
    return tmp_path / "cache", path


def test_ensure_location_ids(master):
    cache, master_path = master
    out = ensure_location_ids(master_path, cache)
    df = pd.read_parquet(out)
    assert {"location_id", "lat", "lon", "tile_id"} <= set(df.columns)
    assert df["location_id"].is_unique


def test_output_layout(master):
    """location_ids live next to the master; map/neighbors in their folders."""
    cache, master_path = master
    loc = ensure_location_ids(master_path, cache)
    assert loc.parent == Path(master_path).resolve().parent or loc.parent == Path(cache).resolve()

    grid = ensure_grid_lookup(master_path, grid_sampling=0.5, extent=(-25, 25, -15, 15), cache_dir=cache)
    assert grid.parent.name == "map_lookups"

    nbr = ensure_neighbor_lookup(master_path, k_neighbors=4, max_distance_km=500.0, cache_dir=cache)
    assert nbr.parent.name == "neighbor_lookups"
    assert nbr.name.startswith("neighbors_k4_maxd100")


def test_neighbor_distance_capped_at_100(master):
    """max_distance_km > 100 is clamped to 100 km in the generated lookup."""
    cache, master_path = master
    nbr = ensure_neighbor_lookup(master_path, k_neighbors=4, max_distance_km=500.0, cache_dir=cache)
    assert nbr.name.startswith("neighbors_k4_maxd100")
    for f in nbr.glob("*.parquet"):
        df = pd.read_parquet(f)
        assert (df["distance_km"] <= 100.0).all()


def test_ensure_grid_lookup(master):
    cache, master_path = master
    out = ensure_grid_lookup(master_path, grid_sampling=0.5, extent=(-25, 25, -15, 15), cache_dir=cache)
    assert out.exists()
    df = pd.read_parquet(out)
    assert {"location_id", "pixel_id"} <= set(df.columns)


def test_grid_lookup_reused_on_same_params(master):
    cache, master_path = master
    a = ensure_grid_lookup(master_path, grid_sampling=0.5, extent=(-25, 25, -15, 15), cache_dir=cache)
    b = ensure_grid_lookup(master_path, grid_sampling=0.5, extent=(-25, 25, -15, 15), cache_dir=cache)
    assert a == b
    assert a.exists()


def test_grid_lookup_differs_on_sampling(master):
    cache, master_path = master
    a = ensure_grid_lookup(master_path, grid_sampling=0.5, extent=(-25, 25, -15, 15), cache_dir=cache)
    b = ensure_grid_lookup(master_path, grid_sampling=0.25, extent=(-25, 25, -15, 15), cache_dir=cache)
    assert a != b
    assert pd.read_parquet(b)["pixel_id"].nunique() > pd.read_parquet(a)["pixel_id"].nunique()


def test_neighbor_lookup(master):
    cache, master_path = master
    out = ensure_neighbor_lookup(master_path, k_neighbors=4, max_distance_km=500.0, cache_dir=cache)
    files = list(out.glob("*.parquet"))
    assert len(files) > 0
    df = pd.read_parquet(files[0])
    assert {
        "location_id",
        "neighbor_location_id",
        "distance_km",
        "rank",
    } <= set(df.columns)


def test_grid_lookup_requires_sampling(master):
    cache, master_path = master
    with pytest.raises(ValueError, match="grid_sampling"):
        ensure_grid_lookup(master_path, grid_sampling=None, extent=(-25, 25, -15, 15), cache_dir=cache)


def test_grid_lookup_encodes_max_distance_in_name(master):
    """Enabled fill encodes max_distance_km in the filename; disabled does not."""
    cache, master_path = master
    filled = ensure_grid_lookup(
        master_path, grid_sampling=0.5, extent=(-25, 25, -15, 15),
        max_distance_km=15.0, cache_dir=cache,
    )
    assert "_maxDistKm_15" in filled.name

    snapped = ensure_grid_lookup(
        master_path, grid_sampling=0.5, extent=(-25, 25, -15, 15),
        max_distance_km=0.0, cache_dir=cache,
    )
    assert "_maxDistKm" not in snapped.name
    assert snapped != filled


def test_grid_lookup_inverted_fills_near_locations(master):
    """With a large max_distance every pixel gets a value (no white holes)."""
    cache, master_path = master
    out = ensure_grid_lookup(
        master_path, grid_sampling=0.5, extent=(-25, 25, -15, 15),
        max_distance_km=2000.0, cache_dir=cache,
    )
    df = pd.read_parquet(out)
    assert {"location_id", "pixel_id"} <= set(df.columns)
    assert (df["pixel_id"] >= 0).all()
    n_pixels = int(round((25 - -25) / 0.5)) * int(round((15 - -15) / 0.5))
    assert df["pixel_id"].nunique() == n_pixels


def test_grid_lookup_inverted_leaves_distant_pixels_blank(master):
    """A tiny max_distance leaves most pixels unmapped (dropped -> white)."""
    cache, master_path = master
    out = ensure_grid_lookup(
        master_path, grid_sampling=0.5, extent=(-25, 25, -15, 15),
        max_distance_km=0.01, cache_dir=cache,
    )
    df = pd.read_parquet(out)
    assert df["pixel_id"].nunique() < 200


def test_grid_lookup_inverted_k_gt_1(master):
    """k>1 returns pixel_id -> location_ids lists within max_distance."""
    cache, master_path = master
    out = ensure_grid_lookup(
        master_path, grid_sampling=0.5, extent=(-25, 25, -15, 15),
        max_distance_km=2000.0, k=3, cache_dir=cache,
    )
    df = pd.read_parquet(out)
    assert {"pixel_id", "location_ids"} <= set(df.columns)
    assert df["location_ids"].map(len).max() >= 1


def test_plot_map_out_of_extent_still_fills(tmp_path, master):
    """Locations just outside the extent still color nearby pixels when filled."""
    _, master_path = master
    master_df = pd.read_parquet(master_path)
    data = pd.DataFrame(
        {
            "location_id": master_df["location_id"],
            "backscatter40": np.random.RandomState(1).normal(-12, 3, len(master_df)),
        }
    )
    # Extent adjacent to the master locations (lon -20..20) but not containing
    # any: with fill-by-proximity the nearby locations still color pixels.
    fig = plot_map(
        data=data,
        var="backscatter40",
        master_lookup=master_path,
        extent=(30, 40, -10, 10),
        grid_sampling=0.5,
        max_distance_km=1500.0,
        show_plot=False,
    )
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_map_out_of_extent_blank_when_disabled(master):
    """Without fill (max_distance=0), locations outside extent fail loudly."""
    _, master_path = master
    master_df = pd.read_parquet(master_path)
    data = pd.DataFrame(
        {
            "location_id": master_df["location_id"],
            "backscatter40": np.random.RandomState(1).normal(-12, 3, len(master_df)),
        }
    )
    with pytest.raises(ValueError, match="No plottable data"):
        plot_map(
            data=data,
            var="backscatter40",
            master_lookup=master_path,
            extent=(30, 40, -10, 10),
            grid_sampling=0.5,
            max_distance_km=0.0,
            show_plot=False,
        )


def test_plot_map_via_master(master):
    _, master_path = master
    master_df = pd.read_parquet(master_path)
    data = pd.DataFrame(
        {
            "location_id": master_df["location_id"],
            "backscatter40": np.random.RandomState(1).normal(-12, 3, len(master_df)),
        }
    )
    fig = plot_map(
        data=data,
        var="backscatter40",
        master_lookup=master_path,
        extent=(-25, 25, -15, 15),
        grid_sampling=0.5,
        show_plot=False,
    )
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_map_requires_master(tmp_path):
    data = pd.DataFrame({"location_id": [2000], "backscatter40": [-12.0]})
    with pytest.raises(ValueError, match="master_lookup"):
        plot_map(data=data, var="backscatter40", show_plot=False)


def test_plot_map_default_grid_sampling(tmp_path, master):
    _, master_path = master
    master_df = pd.read_parquet(master_path)
    data = pd.DataFrame(
        {
            "location_id": master_df["location_id"],
            "backscatter40": np.random.RandomState(1).normal(-12, 3, len(master_df)),
        }
    )
    # plot without grid_sampling -> defaults to 0.5 and builds a cached lookup
    fig = plot_map(
        data=data,
        var="backscatter40",
        master_lookup=master_path,
        extent=(-25, 25, -15, 15),
        show_plot=False,
    )
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_timeseries_via_master(master):
    cache, master_path = master
    master_df = pd.read_parquet(master_path)
    loc_ids = master_df["location_id"].head(20).tolist()

    times = pd.date_range("2000-01-01", periods=24, freq="MS")
    frames = []
    for lid in loc_ids:
        frames.append(
            pd.DataFrame(
                {
                    "location_id": lid,
                    "time": times,
                    "backscatter40": np.full(24, -12.0),
                    "lai": np.full(24, 1.0),
                }
            )
        )
    ts = pd.concat(frames, ignore_index=True)

    # Pre-seed countries.pkl to keep test offline-deterministic
    countries_path = Path(master_path).parent / "countries.pkl"
    import pickle

    with open(countries_path, "wb") as f:
        pickle.dump({loc_ids[0]: "TestCountry"}, f)

    figs = Timeseries.plot_time_series(
        data=ts,
        location_ids=[loc_ids[0]],
        var_specs=[{"name": "backscatter40", "color": "royalblue"}],
        add_closest_points=(3, 500.0),
        master_lookup=master_path,
        show_plot=False,
    )
    assert len(figs) == 1
    import matplotlib.pyplot as plt

    for f in figs:
        plt.close(f)


def test_country_lookup_auto(tmp_path, master):
    """Country generation needs internet; tolerate failures."""
    cache, master_path = master
    tiny = tmp_path / "master_tiny.parquet"
    pd.read_parquet(master_path).head(3).to_parquet(tiny, index=False)
    try:
        out = ensure_country_lookup(tiny, cache)
        assert out.exists()
    except (ImportError, OSError):
        pytest.skip("geocoding extra or internet unavailable")
