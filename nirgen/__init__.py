"""NIRGEN: discrete-ion neural computation."""

from .inn import INN, INNConfig, init_inn
from .inn.mechanics import ste_clamp, ste_floor
from .inn.registry import CAL_MODES, DEVICE_TYPES, SPECIES, TYPE_PRESETS

__all__ = [
    "INN",
    "INNConfig",
    "init_inn",
    "ste_clamp",
    "ste_floor",
    "CAL_MODES",
    "DEVICE_TYPES",
    "SPECIES",
    "TYPE_PRESETS",
]
__version__ = "0.1.0"

