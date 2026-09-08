"""plotting_joseph - visualization tools for spatiotemporal Earth observation data.

Provides multi-panel time series plotting and global map plotting with
flexible data formats and user-generated lookup tables.
"""

from .config import Config, LookupTables
from .countries import COUNTRY_NAMES, country_name_from_code
from .data import (
    DataLoader,
    LookupTableCreator,
    derive_countries_for_locations,
    ensure_country_lookup,
    ensure_grid_lookup,
    ensure_location_ids,
    ensure_neighbor_lookup,
    validate_lookup_tables,
)
from .plotting import Timeseries, plot_map, plot_time_series

__version__ = "0.1.0"

__all__ = [
    "COUNTRY_NAMES",
    "Config",
    "DataLoader",
    "LookupTableCreator",
    "LookupTables",
    "Timeseries",
    "country_name_from_code",
    "derive_countries_for_locations",
    "ensure_country_lookup",
    "ensure_grid_lookup",
    "ensure_location_ids",
    "ensure_neighbor_lookup",
    "plot_map",
    "plot_time_series",
    "validate_lookup_tables",
]
