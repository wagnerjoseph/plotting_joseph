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
    """With a large max_distance every pixel gets a real location (no white holes)."""
    cache, master_path = master
    out = ensure_grid_lookup(
        master_path, grid_sampling=0.5, extent=(-25, 25, -15, 15),
        max_distance_km=2000.0, cache_dir=cache,
    )
    df = pd.read_parquet(out)
    assert {"location_id", "pixel_id"} <= set(df.columns)
    n_pixels = int(round((25 - -25) / 0.5)) * int(round((15 - -15) / 0.5))
    # All grid pixels are kept, and all are mapped to a real location.
    assert df["pixel_id"].nunique() == n_pixels
    assert (df["location_id"] >= 0).all()


def test_grid_lookup_inverted_leaves_distant_pixels_blank(master):
    """A tiny max_distance keeps all pixels but marks most as blank (-1)."""
    cache, master_path = master
    out = ensure_grid_lookup(
        master_path, grid_sampling=0.5, extent=(-25, 25, -15, 15),
        max_distance_km=0.01, cache_dir=cache,
    )
    df = pd.read_parquet(out)
    n_pixels = int(round((25 - -25) / 0.5)) * int(round((15 - -15) / 0.5))
    assert df["pixel_id"].nunique() == n_pixels
    blank = int((df["location_id"] == -1).sum())
    assert blank > 0.9 * n_pixels


def test_grid_lookup_inverted_k_gt_1(master):
    """k>1 returns pixel_id -> location_ids lists (all pixels, empty allowed)."""
    cache, master_path = master
    out = ensure_grid_lookup(
        master_path, grid_sampling=0.5, extent=(-25, 25, -15, 15),
        max_distance_km=2000.0, k=3, cache_dir=cache,
    )
    df = pd.read_parquet(out)
    n_pixels = int(round((25 - -25) / 0.5)) * int(round((15 - -15) / 0.5))
    assert {"pixel_id", "location_ids"} <= set(df.columns)
    assert df["pixel_id"].nunique() == n_pixels
    assert df["location_ids"].map(len).max() >= 1


def test_grid_lookup_uses_great_circle_distance(tmp_path):
    """Fill threshold uses true great-circle km, not a planar lat/lon distance."""
    from plotting_joseph.data import ensure_grid_lookup

    cache = tmp_path / "cache"
    # Two source locations: (lat=0, lon=0) and (lat=40, lon=0).
    df = pd.DataFrame(
        {"location_id": [1, 2], "lat": [0.0, 40.0], "lon": [0.0, 0.0], "tile_id": ["t", "t"]}
    )
    master = tmp_path / "m.parquet"
    df.to_parquet(master, index=False)

    extent = (0.0, 0.5, 39.0, 40.5)  # a small box around the lat=40 location
    # The box spans up to ~102 km of great-circle distance from (lat=40, lon=0).
    # A radius of 110 km reaches every pixel; 20 km reaches none.
    out = ensure_grid_lookup(
        master, grid_sampling=0.25, extent=extent,
        max_distance_km=110.0, k=1, cache_dir=cache,
    )
    lut = pd.read_parquet(out)
    # The box is within ~110 km of (lat=40, lon=0) -> all pixels map to loc 2.
    assert set(lut["location_id"].unique()) == {2}

    out_far = ensure_grid_lookup(
        master, grid_sampling=0.25, extent=extent,
        max_distance_km=10.0, k=1, cache_dir=cache,
    )
    lut_far = pd.read_parquet(out_far)
    # 10 km < nearest pixel (~17.5 km) -> no location reaches the box, all blank.
    assert set(lut_far["location_id"].unique()) == {-1}


def test_grid_lookup_inverted_pixel_position_correct(tmp_path):
    """Each location maps to its actual home cell, not a transposed neighbour.

    Regression for a meshgrid-ordering bug: on a non-square grid the pixel
    centres were generated transposed relative to ``pixel_id = row*n_lon+col``,
    scrambling the fill into horizontal/vertical bands.
    """
    from plotting_joseph.data import ensure_grid_lookup

    cache = tmp_path / "cache"
    grid_sampling = 1.0
    extent = (0.0, 4.0, 0.0, 2.0)  # n_lon=4, n_lat=2 (non-square)
    lon_min, _lon_max, _lat_min, lat_max = extent
    n_lon = int(round((extent[1] - extent[0]) / grid_sampling))

    master = pd.DataFrame(
        {"location_id": [7, 8], "lat": [0.5, 1.5], "lon": [0.5, 3.5], "tile_id": ["t", "t"]}
    )
    master_path = tmp_path / "m.parquet"
    master.to_parquet(master_path, index=False)

    out = ensure_grid_lookup(
        master_path, grid_sampling=grid_sampling, extent=extent,
        max_distance_km=1.0, k=1, cache_dir=cache,
    )
    lut = pd.read_parquet(out).set_index("location_id")["pixel_id"]

    def pixel_for(lat, lon):
        col = int(np.floor((lon - lon_min) / grid_sampling))
        row = int(np.floor((lat_max - lat) / grid_sampling))
        return row * n_lon + col

    expected_7 = pixel_for(0.5, 0.5)
    expected_8 = pixel_for(1.5, 3.5)
    assert lut[7] == expected_7, f"loc7 -> {lut[7]}, expected {expected_7}"
    assert lut[8] == expected_8, f"loc8 -> {lut[8]}, expected {expected_8}"
    # Distinguished: must not be transposed (swapped row/col index).
    assert lut[7] != pixel_for(0.5, 1.5)
    assert lut[8] != pixel_for(1.5, 0.5)


def test_grid_lookup_coordinate_order_matches_great_circle(tmp_path):
    """Nearest-location fill uses (lat, lon) haversine, matching brute force.

    Regression for a scikit-learn BallTree gotcha: the haversine metric expects
    points as (latitude, longitude). Passing (lon, lat) silently computes wrong
    great-circle distances and scrambles which location fills each pixel.
    Compare the inverted lookup against a brute-force great-circle nearest-
    neighbour on off-axis data, where a lat/lon swap would disagree.
    """
    from plotting_joseph.data import _build_inverted_grid_lookup

    rng = np.random.default_rng(7)
    n = 12
    master = pd.DataFrame(
        {
            "location_id": np.arange(n),
            "lat": rng.uniform(-30, 60, n),
            "lon": rng.uniform(-60, 60, n),
        }
    )

    extent = (-70, 70, -40, 70)
    grid_sampling = 3.0
    max_distance_km = 10000.0
    lut = _build_inverted_grid_lookup(
        master, grid_sampling, extent, 1, max_distance_km
    )

    lon_min, lon_max, lat_min, lat_max = extent
    n_lon = int(round((lon_max - lon_min) / grid_sampling))
    n_lat = int(round((lat_max - lat_min) / grid_sampling))
    got = np.full(n_lat * n_lon, -1, dtype=np.int64)
    got[lut["pixel_id"].to_numpy(np.int64)] = lut["location_id"].to_numpy(np.int64)

    def great_circle(lat1, lon1, lat2, lon2):
        p1, p2 = np.radians([lat1, lon1]), np.radians([lat2, lon2])
        dlat, dlon = p2[0] - p1[0], p2[1] - p1[1]
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(p1[0]) * np.cos(p2[0]) * np.sin(dlon / 2) ** 2
        )
        return 2 * 6371.0 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    src_lat = master["lat"].to_numpy(float)
    src_lon = master["lon"].to_numpy(float)
    src_id = master["location_id"].to_numpy(np.int64)

    got = got.reshape(n_lat, n_lon)
    ref = np.full_like(got, -1)
    for i in range(n_lat):
        lat = lat_max - (i + 0.5) * grid_sampling  # row 0 == north (origin="upper")
        for j in range(n_lon):
            lon = lon_min + (j + 0.5) * grid_sampling
            dists = great_circle(lat, lon, src_lat, src_lon)
            best = int(np.argmin(dists))
            if dists[best] <= max_distance_km:
                ref[i, j] = src_id[best]

    n_mismatch = int(np.sum(got != ref))
    assert n_mismatch == 0, (
        f"{n_mismatch} pixels disagree with brute-force great-circle nearest "
        "neighbour (coordinate order wrong)"
    )


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


def test_plot_map_fill_whites_beyond_max_distance(tmp_path):
    """Direct per-pixel fill: close pixels get the location's value, far ones are white."""
    import matplotlib.pyplot as plt

    grid_sampling = 1.0
    extent = (0.0, 4.0, 0.0, 2.0)  # n_lon=4, n_lat=2 -> 8 pixels
    master = pd.DataFrame(
        {"location_id": [7], "lat": [0.5], "lon": [0.5], "tile_id": ["t"]}
    )
    master_path = tmp_path / "m.parquet"
    master.to_parquet(master_path, index=False)

    data = pd.DataFrame({"location_id": [7], "backscatter40": [10.0]})

    def count_filled(max_distance_km):
        fig = plot_map(
            data=data, var="backscatter40", master_lookup=master_path,
            extent=extent, grid_sampling=grid_sampling,
            max_distance_km=max_distance_km, show_plot=False,
        )
        arr = fig.axes[0].get_images()[0].get_array()
        plt.close(fig)
        return int(np.count_nonzero(~np.isnan(arr)))

    # 500 km reaches every pixel in the small box -> all filled with the value.
    assert count_filled(500.0) == 8
    # 40 km only reaches the location's own cell -> a single filled pixel.
    assert count_filled(40.0) == 1


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
