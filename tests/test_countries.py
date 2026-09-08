"""Tests for country-name resolution and lookup generation."""

import pickle

import pandas as pd
import reverse_geocoder

from plotting_joseph import (
    LookupTableCreator,
    Timeseries,
    country_name_from_code,
    derive_countries_for_locations,
)
from plotting_joseph.countries import COUNTRY_NAMES


def test_country_names_are_complete_uppercase():
    assert len(COUNTRY_NAMES) >= 240
    for code, name in COUNTRY_NAMES.items():
        assert code == code.upper() and len(code) == 2
        assert isinstance(name, str) and name


def test_country_name_from_code_case_insensitive():
    assert country_name_from_code("AT") == "Austria"
    assert country_name_from_code("at") == "Austria"
    assert country_name_from_code("RU") == "Russia"
    assert country_name_from_code("US") == "United States"


def test_country_name_from_code_unknown():
    assert country_name_from_code(None) == "Unknown"
    assert country_name_from_code("") == "Unknown"
    assert country_name_from_code("ZZ") == "Unknown"


def test_generate_country_lookup_uses_full_names(tmp_path, monkeypatch):
    loc = tmp_path / "locs.parquet"
    pd.DataFrame(
        {
            "location_id": [1, 2, 3],
            "lat": [47.0, 41.0, 0.0],
            "lon": [13.0, -74.0, 0.0],
        }
    ).to_parquet(loc, index=False)

    fake_results = [
        {"lat": 47.0, "lon": 13.0, "name": "Klagenfurt", "admin1": "Carinthia",
         "admin2": "", "cc": "AT"},
        {"lat": 41.0, "lon": -74.0, "name": "x", "admin1": "x", "admin2": "", "cc": "us"},
        {"lat": 0.0, "lon": 0.0, "name": "mid-ocean", "admin1": "", "admin2": "", "cc": ""},
    ]
    monkeypatch.setattr(reverse_geocoder, "search", lambda coords: fake_results)

    out = tmp_path / "countries.pkl"
    LookupTableCreator.generate_country_lookup(loc, out)

    with open(out, "rb") as f:
        records = pickle.load(f)
    assert records[1] == "Austria"
    assert records[2] == "United States"  # normalized case from "us"
    assert records[3] == "Unknown"  # empty cc -> Unknown, not a raw repr


def test_generate_country_lookup_handles_namedtuple_like(tmp_path, monkeypatch):
    loc = tmp_path / "locs.parquet"
    pd.DataFrame(
        {"location_id": [9], "lat": [52.0], "lon": [13.0]}
    ).to_parquet(loc, index=False)

    monkeypatch.setattr(
        reverse_geocoder, "search", lambda coords: ["not-a-dict-result"]
    )

    out = tmp_path / "countries.pkl"
    LookupTableCreator.generate_country_lookup(loc, out)
    with open(out, "rb") as f:
        records = pickle.load(f)
    assert records[9] == "Unknown"


def _make_master_lookup(path):
    pd.DataFrame(
        {
            "location_id": [1, 2, 3],
            "lat": [47.0, 41.0, 0.0],
            "lon": [13.0, -74.0, 0.0],
            "tile_id": ["t0", "t0", "t0"],
        }
    ).to_parquet(path, index=False)


def test_derive_countries_for_locations_subset(tmp_path, monkeypatch):
    master = tmp_path / "master.parquet"
    _make_master_lookup(master)

    fake = {
        47.0: {"lat": 47.0, "lon": 13.0, "name": "x", "admin1": "x", "admin2": "", "cc": "AT"},
        41.0: {"lat": 41.0, "lon": -74.0, "name": "x", "admin1": "x", "admin2": "", "cc": "US"},
    }
    calls = {}

    def fake_search(coords):
        calls["coords"] = list(coords)
        return [fake[float(c[0])] for c in coords]

    monkeypatch.setattr(reverse_geocoder, "search", fake_search)

    result = derive_countries_for_locations(master, location_ids=[1, 2])
    assert result == {1: "Austria", 2: "United States"}
    # Only the requested points were geocoded.
    requested_lats = {c[0]: None for c in calls["coords"]}
    assert set(requested_lats) == {47.0, 41.0}


def test_timeseries_title_includes_country(sample_timeseries_data, tmp_path, monkeypatch):
    master = tmp_path / "master.parquet"
    pd.DataFrame(
        {
            "location_id": [1, 2],
            "lat": [47.0, 41.0],
            "lon": [13.0, -74.0],
            "tile_id": ["t0", "t0"],
        }
    ).to_parquet(master, index=False)

    cc = {
        47.0: {"lat": 47.0, "lon": 13.0, "name": "x", "admin1": "x", "admin2": "", "cc": "AT"},
        41.0: {"lat": 41.0, "lon": -74.0, "name": "x", "admin1": "x", "admin2": "", "cc": "US"},
    }
    monkeypatch.setattr(
        reverse_geocoder,
        "search",
        lambda coords: [cc[float(c[0])] for c in coords],
    )

    figs = Timeseries.plot_time_series(
        data=sample_timeseries_data,
        var_specs=[{"name": "backscatter40"}],
        location_ids=[1, 2],
        master_lookup=master,
        show_plot=False,
    )
    titles = {f._suptitle.get_text() for f in figs}
    assert "location_id = 1 (Austria)" in titles
    assert "location_id = 2 (United States)" in titles


def test_timeseries_title_omits_unknown_country(tmp_path, monkeypatch, sample_timeseries_data):
    master = tmp_path / "master.parquet"
    pd.DataFrame(
        {
            "location_id": [1],
            "lat": [0.0],
            "lon": [0.0],
            "tile_id": ["t0"],
        }
    ).to_parquet(master, index=False)

    monkeypatch.setattr(
        reverse_geocoder,
        "search",
        lambda coords: [{"lat": 0.0, "lon": 0.0, "name": "sea", "admin1": "", "admin2": "", "cc": ""}],
    )

    figs = Timeseries.plot_time_series(
        data=sample_timeseries_data,
        var_specs=[{"name": "backscatter40"}],
        location_ids=[1],
        master_lookup=master,
        show_plot=False,
    )
    title = figs[0]._suptitle.get_text()
    assert title == "location_id = 1"
    assert "Unknown" not in title
    assert "(" not in title


def test_timeseries_no_lookup_shows_no_country(sample_timeseries_data):
    figs = Timeseries.plot_time_series(
        data=sample_timeseries_data,
        var_specs=[{"name": "backscatter40"}],
        location_ids=[1],
        show_plot=False,
    )
    assert figs[0]._suptitle.get_text() == "location_id = 1"
