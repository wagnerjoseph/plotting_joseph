"""Tests for country-name resolution and lookup generation."""

import pickle

import pandas as pd
import reverse_geocoder

from plotting_joseph import LookupTableCreator, country_name_from_code
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
