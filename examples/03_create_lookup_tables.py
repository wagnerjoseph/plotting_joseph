"""Example: create lookup tables from your own data.

Run:
    python examples/03_create_lookup_tables.py
"""

from pathlib import Path

from plotting_joseph import LookupTableCreator, LookupTables, validate_lookup_tables

OUT_DIR = Path("lookup_tables")

# ---------------------------------------------------------------------------
# 1. location_ids.parquet — from your own coordinate data
# ---------------------------------------------------------------------------
# Replace with a real path to your data (parquet / csv / DataFrame).
# Below we generate a small synthetic coordinate frame to demonstrate.
import numpy as np
import pandas as pd

n = 200
locations = pd.DataFrame(
    {
        "location_id": list(range(n)),
        "latitude": np.linspace(40, 60, n),
        "longitude": np.linspace(-10, 15, n),
        "tile_id": [f"t{i % 4}" for i in range(n)],
    }
)

LookupTableCreator.from_dataframe(
    locations,
    location_id_column="location_id",
    lat_column="latitude",
    lon_column="longitude",
    output_path=OUT_DIR / "location_ids.parquet",
    tile_id_column="tile_id",
)
print("Created location_ids.parquet")

# Or, without source coordinates, on a regular grid:
# LookupTableCreator.from_grid(
#     output_path=OUT_DIR / "location_ids.parquet",
#     resolution_km=12.5,
# )

# ---------------------------------------------------------------------------
# 2. countries.pkl — auto country names (reverse geocoding, built-in)
# ---------------------------------------------------------------------------
# LookupTableCreator.generate_country_lookup(
#     location_ids_path=OUT_DIR / "location_ids.parquet",
#     output_path=OUT_DIR / "countries.pkl",
# )

# ---------------------------------------------------------------------------
# 3. nearest-neighbor lookup (for time-series backgrounds)
# ---------------------------------------------------------------------------
LookupTableCreator.generate_neighbor_lookup(
    location_ids_path=OUT_DIR / "location_ids.parquet",
    output_dir=OUT_DIR / "closest_location_ids/",
    k_neighbors=5,
    max_distance_km=100.0,
)
print("Created closest_location_ids/")

# ---------------------------------------------------------------------------
# 4. Validate
# ---------------------------------------------------------------------------
errors = validate_lookup_tables(
    LookupTables(
        location_ids=OUT_DIR / "location_ids.parquet",
        neighbors_dir=OUT_DIR / "closest_location_ids/",
    )
)
print("Validation errors:", errors or "none")
