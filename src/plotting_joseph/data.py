"""Data loading and lookup-table creation.

This module implements the ``DataLoader`` and ``LookupTableCreator`` helpers
referenced by the package ``__init__`` and documented in ``docs/``. It
normalises user data (parquet / CSV / in-memory DataFrame) into the canonical
schema expected by the plotting functions and generates the geographic lookup
tables they rely on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Canonical name -> list of acceptable source column names (auto-detected).
DEFAULT_ALIASES: dict[str, list[str]] = {
    "time": ["time", "date", "datetime", "timestamp"],
    "location_id": ["location_id", "loc_id", "loc", "gpi"],
    "backscatter40": ["backscatter40", "bs40", "sig0"],
    "lai": ["lai"],
    "swvl1": ["swvl1", "swvl", "swvl1_normalized"],
}

# Canonical columns that are mandatory regardless of use-case.
_REQUIRED = ["time", "location_id"]

# Upper bound (km) for nearest-neighbor lookups, so generated files stay small.
MAX_NEIGHBOR_DISTANCE_KM = 100.0


class DataLoader:
    """Load and normalise data from common storage formats.

    Parameters
    ----------
    column_map : dict, optional
        Mapping of ``canonical name -> your column name``, e.g.
        ``{"time": "date", "location_id": "loc_id"}``. Can also remap
        arbitrary variable columns (e.g. ``{"backscatter40": "bs40"}``).
    """

    def __init__(self, column_map: dict[str, str] | None = None) -> None:
        self.column_map = column_map or {}

    # -- loading ------------------------------------------------------------
    def load(self, source) -> pd.DataFrame:
        """Load and normalise ``source`` into a canonical DataFrame.

        ``source`` may be a path to a parquet / CSV file, or an in-memory
        pandas.DataFrame.
        """
        if isinstance(source, (str, Path)):
            df = self._load_path(Path(source))
        elif isinstance(source, pd.DataFrame):
            df = source.copy()
        else:
            raise TypeError(
                f"source must be a path (str/Path) or pandas.DataFrame, got {type(source)!r}"
            )

        df = self._rename_known(df)
        df = self._coerce_dtypes(df)
        df = self._validate(df)
        return df

    def _load_path(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix in {".csv", ".txt"}:
            # time columns are parsed lazily on a best-effort basis below
            df = pd.read_csv(path)
            if "time" in df.columns or "date" in df.columns:
                df = self._coerce_dtypes(df)
            return df
        raise ValueError(f"Unsupported file format: {suffix}")

    # -- normalisation ------------------------------------------------------
    def _rename_known(self, df: pd.DataFrame) -> pd.DataFrame:
        mapping: dict[str, str] = {}
        for canonical, source in self.column_map.items():
            if source in df.columns:
                mapping[source] = canonical

        used_names = set(df.columns)
        for canonical, aliases in DEFAULT_ALIASES.items():
            if canonical in mapping.values():
                continue
            for alias in aliases:
                if alias in used_names:
                    mapping[alias] = canonical
                    break
        return df.rename(columns=mapping)

    def _coerce_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        if "time" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"])
        if "location_id" in df.columns:
            df["location_id"] = pd.to_numeric(df["location_id"], errors="coerce").astype(
                "Int64"
            )
        return df

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in _REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(
                "Missing required column(s) after mapping: "
                f"{missing}. Available columns: {list(df.columns)}. "
                "Add the correct entry to `column_map` to fix it."
            )
        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
            raise ValueError("'time' column must be datetime-like.")
        return df


def _latlon_to_flat(lat: np.ndarray, lon: np.ndarray, grid_sampling: float) -> np.ndarray:
    """Compute flat ``pixel_id = row * n_lon + col`` for the default grid."""
    lat_min, lat_max = -60.0, 85.0
    lon_min, lon_max = -180.0, 180.0
    n_lon = int(np.round((lon_max - lon_min) / grid_sampling))
    col = np.floor((lon - lon_min) / grid_sampling).astype(int)
    row = np.floor((lat_max - lat) / grid_sampling).astype(int)
    n_lat = int(np.round((lat_max - lat_min) / grid_sampling))
    flat = row * n_lon + col
    flat[(row < 0) | (row >= n_lat) | (col < 0) | (col >= n_lon)] = -1
    return flat


class LookupTableCreator:
    """Create geographic lookup tables from user data or a regular grid."""

    # -- location_ids -------------------------------------------------------
    @staticmethod
    def _write_location_ids(
        df: pd.DataFrame,
        location_id_column: str,
        lat_column: str,
        lon_column: str,
        output_path,
        tile_id_column: str | None,
    ) -> Path:
        if {location_id_column, lat_column, lon_column} - set(df.columns):
            raise ValueError(
                f"Input must have columns {location_id_column!r}, {lat_column!r}, "
                f"{lon_column!r}. Got: {list(df.columns)}"
            )
        out = df[[location_id_column, lat_column, lon_column]].copy()
        out = out.rename(
            columns={
                location_id_column: "location_id",
                lat_column: "lat",
                lon_column: "lon",
            }
        )
        if tile_id_column is not None:
            if tile_id_column not in df.columns:
                raise ValueError(f"tile_id_column {tile_id_column!r} not found.")
            out["tile_id"] = df[tile_id_column].astype(str)
        out["location_id"] = out["location_id"].astype(np.int64)
        out = out.drop_duplicates("location_id").reset_index(drop=True)
        out.to_parquet(output_path, index=False)
        return Path(output_path)

    @classmethod
    def from_dataframe(
        cls,
        df,
        location_id_column: str,
        lat_column: str,
        lon_column: str,
        output_path,
        tile_id_column: str | None = None,
    ) -> Path:
        """Create ``location_ids.parquet`` from an in-memory DataFrame."""
        return cls._write_location_ids(
            df, location_id_column, lat_column, lon_column, output_path, tile_id_column
        )

    @classmethod
    def from_parquet(
        cls,
        input_path,
        location_id_column: str,
        lat_column: str,
        lon_column: str,
        output_path,
        tile_id_column: str | None = None,
    ) -> Path:
        """Create ``location_ids.parquet`` from a parquet file."""
        df = pd.read_parquet(input_path)
        return cls._write_location_ids(
            df, location_id_column, lat_column, lon_column, output_path, tile_id_column
        )

    @classmethod
    def from_csv(
        cls,
        csv_path,
        location_id_column: str,
        lat_column: str,
        lon_column: str,
        output_path,
        tile_id_column: str | None = None,
    ) -> Path:
        """Create ``location_ids.parquet`` from a CSV file."""
        df = pd.read_csv(csv_path)
        return cls._write_location_ids(
            df, location_id_column, lat_column, lon_column, output_path, tile_id_column
        )

    @classmethod
    def from_grid(
        cls,
        output_path,
        resolution_km: float,
        lat_min: float = -60.0,
        lat_max: float = 85.0,
        lon_min: float = -180.0,
        lon_max: float = 180.0,
    ) -> Path:
        """Create ``location_ids.parquet`` on a regular lat/lon grid.

        Longitude sampling is adjusted by ``cos(lat)`` so cells stay roughly
        square on the ground.
        """
        km_per_deg = 111.0
        res_deg_lat = resolution_km / km_per_deg
        lat = np.arange(lat_min, lat_max, res_deg_lat)

        # location_id must be unique; assign a running counter.
        frames = []
        counter = 0
        for la in lat:
            coslat = max(np.cos(np.radians(float(la))), 0.01)
            res_deg_lon = resolution_km / km_per_deg / coslat
            lon = np.arange(lon_min, lon_max, res_deg_lon)
            n = len(lon)
            ids = np.arange(counter, counter + n, dtype=np.int64)
            counter += n
            frames.append(
                pd.DataFrame(
                    {
                        "location_id": ids,
                        "lat": np.full(n, float(la)),
                        "lon": lon,
                        "tile_id": np.full(n, "tile0", dtype=object),
                    }
                )
            )
        df = pd.concat(frames, ignore_index=True)
        df["location_id"] = df["location_id"].astype(np.int64)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        return Path(output_path)

    # -- countries ----------------------------------------------------------
    @classmethod
    def generate_country_lookup(cls, location_ids_path, output_path) -> Path:
        """Build a ``{location_id: country_name}`` pickle.

        Uses ``reverse_geocoder`` (a core dependency) to resolve each point's
        nearest populated place and its 2-letter country code, then translates
        the code to a full English name via the bundled ``COUNTRY_NAMES`` map.
        Requires internet access on first use (``reverse_geocoder`` downloads
        a GeoNames snapshot); lookups are fast and fully offline afterwards.
        Points with no resolvable country are stored as ``"Unknown"``.
        """
        import reverse_geocoder as rg

        from .countries import country_name_from_code

        loc = pd.read_parquet(location_ids_path)
        coords = list(zip(loc["lat"].to_numpy(), loc["lon"].to_numpy()))
        results = rg.search(coords)
        records = {}
        for loc_id, res in zip(loc["location_id"], results):
            cc = res.get("cc") if isinstance(res, dict) else None
            records[int(loc_id)] = country_name_from_code(cc)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        import pickle

        with open(output_path, "wb") as f:
            pickle.dump(records, f)
        return output_path

    # -- neighbors ----------------------------------------------------------
    @classmethod
    def generate_neighbor_lookup(
        cls,
        location_ids_path,
        output_dir,
        k_neighbors: int = 10,
        max_distance_km: float = 100.0,
        workers: int = 1,
    ) -> Path:
        """Create per-tile nearest-neighbor lookup parquet files.

        One file per tile in ``output_dir`` (named ``{tile_id}.parquet``),
        with columns ``location_id``, ``neighbor_location_id``, ``distance_km``
        and ``rank``.
        """
        try:
            from scipy.spatial import cKDTree
        except ImportError as e:  # pragma: no cover
            raise ImportError("generate_neighbor_lookup requires scipy") from e

        loc = pd.read_parquet(location_ids_path)
        if "tile_id" not in loc.columns:
            loc["tile_id"] = "tile0"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for tile_id, tile in loc.groupby("tile_id"):
            coords_rad = np.radians(tile[["lat", "lon"]].to_numpy(dtype=float))
            tree = cKDTree(coords_rad)

            rows = []
            for i, (loc_id, lat, lon) in enumerate(
                zip(tile["location_id"], tile["lat"], tile["lon"])
            ):
                point_rad = np.radians([[float(lat), float(lon)]])
                dists, idxs = tree.query(point_rad, k=min(k_neighbors + 1, len(tile)))
                dists = np.ravel(dists)
                idxs = np.ravel(idxs).astype(int)
                for d, j in zip(dists[1:], idxs[1:]):
                    dist_km = d * 6371.0
                    if max_distance_km > 0 and dist_km > max_distance_km:
                        continue
                    rows.append(
                        {
                            "location_id": int(loc_id),
                            "neighbor_location_id": int(tile["location_id"].iloc[int(j)]),
                            "distance_km": float(dist_km),
                        }
                    )
            # assign ranks per location after collecting candidates
            df_rows = pd.DataFrame(rows)
            if not df_rows.empty:
                df_rows["rank"] = (
                    df_rows.groupby("location_id")["distance_km"]
                    .rank(ascending=True)
                    .astype(int)
                )
                df_rows.to_parquet(output_dir / f"{tile_id}.parquet", index=False)
        return output_dir


def _format_number(x: float) -> str:
    """Format a number consistently for use inside a lookup filename."""
    return f"{x:.6f}".rstrip("0").rstrip(".")


def _grid_lookup_name(grid_sampling: float, extent, k: int) -> str:
    """Unique, human-readable filename encoding every geometric parameter."""
    lon_min, lon_max, lat_min, lat_max = extent
    ext = "_".join(_format_number(v) for v in (lon_min, lon_max, lat_min, lat_max))
    return f"gridSampling_{grid_sampling}_extent_{ext}_k{k}.parquet"


def _lookup_cache_dir(master_lookup, cache_dir=None) -> Path:
    """Return (and create) the base directory for generated lookups.

    Defaults to the folder that contains the master lookup. May be overridden
    with an explicit ``cache_dir`` (primarily for the public ``ensure_*``
    helpers).
    """
    if cache_dir is not None:
        d = Path(cache_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d
    return Path(master_lookup).resolve().parent


def _map_lookup_dir(master_lookup, cache_dir=None) -> Path:
    """The folder for map grid lookups (``map_lookups/`` next to the master)."""
    d = _lookup_cache_dir(master_lookup, cache_dir) / "map_lookups"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _neighbor_lookup_dir(master_lookup, k_neighbors, max_distance_km, cache_dir=None) -> Path:
    """The per-parameter neighbor lookup folder (``neighbor_lookups/``)."""
    d = (
        _lookup_cache_dir(master_lookup, cache_dir)
        / "neighbor_lookups"
        / f"neighbors_k{k_neighbors}_maxd{max_distance_km:g}"
    )
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_master(master_lookup) -> pd.DataFrame:
    """Read the master lookup and expose canonical ``location_id/lat/lon/tile_id``."""
    if master_lookup is None:
        raise ValueError(
            "No lookup tables configured. Pass 'master_lookup' (a "
            "location_id -> tile_id parquet with lat/lon) to the plotting call."
        )
    master = pd.read_parquet(master_lookup)
    if not {"location_id", "lat", "lon"} <= set(master.columns):
        raise ValueError(
            "master_lookup must contain 'location_id', 'lat' and 'lon' columns. "
            f"Got: {list(master.columns)}"
        )
    if "tile_id" not in master.columns:
        master["tile_id"] = "tile0"
    return master


def ensure_location_ids(master_lookup, cache_dir=None) -> Path:
    """Build (or reuse) a canonical ``location_ids.parquet`` from the master."""
    master = _read_master(master_lookup)
    d = _lookup_cache_dir(master_lookup, cache_dir)
    out = d / "location_ids.parquet"
    if not out.exists():
        master[["location_id", "lat", "lon", "tile_id"]].to_parquet(out, index=False)
    return out


def ensure_country_lookup(
    master_lookup, cache_dir=None, force: bool = False
) -> Path:
    """Return a ``{location_id: country}`` pickle, generating it if missing.

    The lookup is resolved online (reverse geocoding) when it does not exist
    yet, unless ``force=False`` and it is already cached. Needs internet access
    on first use.
    """
    master = _read_master(master_lookup)
    d = _lookup_cache_dir(master_lookup, cache_dir)
    out = d / "countries.pkl"
    if out.exists() and not force:
        return out

    tmp_loc = d / "location_ids_for_countries.parquet"
    master[["location_id", "lat", "lon"]].to_parquet(tmp_loc, index=False)
    LookupTableCreator.generate_country_lookup(tmp_loc, out)
    return out


def ensure_grid_lookup(
    master_lookup,
    grid_sampling: float,
    extent: tuple[float, float, float, float] = (-180.0, 180.0, -60.0, 85.0),
    k: int = 1,
    cache_dir=None,
) -> Path:
    """Build (or reuse) the map lookup for the given geometric parameters.

    The lookup maps ``location_id`` -> ``pixel_id`` on a grid of resolution
    ``grid_sampling`` over ``extent``. For ``k > 1`` each pixel stores the list
    of locations aggregated into it.

    The filename encodes ``grid_sampling``, ``extent`` and ``k`` so that a
    lookup built for one set of parameters is reused for identical calls.
    """
    if grid_sampling is None or grid_sampling <= 0:
        raise ValueError(
            "plot_map needs a positive grid_sampling (degrees) to build a grid lookup."
        )
    master = _read_master(master_lookup)
    d = _map_lookup_dir(master_lookup, cache_dir)
    out = d / _grid_lookup_name(grid_sampling, extent, k)
    if out.exists():
        return out

    lon_min, lon_max, lat_min, lat_max = extent
    n_lon = int(round((lon_max - lon_min) / grid_sampling))
    n_lat = int(round((lat_max - lat_min) / grid_sampling))

    lon = master["lon"].to_numpy(dtype=float)
    lat = master["lat"].to_numpy(dtype=float)
    col = np.floor((lon - lon_min) / grid_sampling).astype(int)
    row = np.floor((lat_max - lat) / grid_sampling).astype(int)
    valid = (row >= 0) & (row < n_lat) & (col >= 0) & (col < n_lon)
    pixel = np.where(valid, row * n_lon + col, -1)

    if k == 1:
        grid_lut = pd.DataFrame(
            {"location_id": master["location_id"].to_numpy(), "pixel_id": pixel}
        )
    else:
        df = pd.DataFrame(
            {
                "location_id": master["location_id"].to_numpy(),
                "pixel_id": pixel,
            }
        ).query("pixel_id >= 0")
        grid_lut = df.groupby("pixel_id")["location_id"].apply(list).reset_index()
        grid_lut = grid_lut.rename(columns={"location_id": "location_ids"})
        grid_lut["pixel_id"] = grid_lut["pixel_id"].astype(int)

    grid_lut.to_parquet(out, index=False)
    return out


def ensure_neighbor_lookup(
    master_lookup,
    k_neighbors: int,
    max_distance_km: float,
    cache_dir=None,
) -> Path:
    """Build (or reuse) the per-tile neighbor lookup directory.

    The directory name encodes ``k_neighbors`` and ``max_distance_km`` so that
    identical neighbor requests reuse the same generated files. The distance is
    capped at :data:`MAX_NEIGHBOR_DISTANCE_KM` (100 km) to keep the generated
    lookups reasonably small.
    """
    max_distance_km = min(max_distance_km, MAX_NEIGHBOR_DISTANCE_KM)
    master = _read_master(master_lookup)
    neighbors_dir = _neighbor_lookup_dir(master_lookup, k_neighbors, max_distance_km, cache_dir)
    if any(neighbors_dir.glob("*.parquet")):
        return neighbors_dir

    tmp_loc = _lookup_cache_dir(master_lookup, cache_dir) / "location_ids_neighbors.parquet"
    master[["location_id", "lat", "lon", "tile_id"]].to_parquet(tmp_loc, index=False)
    LookupTableCreator.generate_neighbor_lookup(
        location_ids_path=tmp_loc,
        output_dir=neighbors_dir,
        k_neighbors=k_neighbors,
        max_distance_km=max_distance_km,
    )
    return neighbors_dir



def validate_lookup_tables(lookup_tables) -> list[str]:
    """Validate the lookup tables referenced by a ``LookupTables`` config.

    Returns a list of human-readable error strings (empty when valid).
    """
    from .config import LookupTables

    if not isinstance(lookup_tables, LookupTables):
        lookup_tables = LookupTables(
            location_ids=lookup_tables.get("location_ids"),
            countries=lookup_tables.get("countries"),
            neighbors_dir=lookup_tables.get("neighbors_dir"),
        )

    errors: list[str] = []

    if lookup_tables.location_ids is not None:
        p = Path(lookup_tables.location_ids)
        if not p.exists():
            errors.append(f"location_ids table not found: {p}")
        else:
            df = pd.read_parquet(p)
            if "location_id" not in df.columns:
                errors.append(f"location_ids table missing 'location_id': {p}")
            for c in ("lat", "lon"):
                if c not in df.columns:
                    errors.append(f"location_ids table missing '{c}': {p}")

    if lookup_tables.countries is not None:
        p = Path(lookup_tables.countries)
        if not p.exists():
            errors.append(f"countries table not found: {p}")

    if lookup_tables.neighbors_dir is not None:
        d = Path(lookup_tables.neighbors_dir)
        if not d.exists():
            errors.append(f"neighbors_dir not found: {d}")
        elif not any(d.glob("*.parquet")):
            errors.append(f"neighbors_dir contains no parquet files: {d}")

    return errors
