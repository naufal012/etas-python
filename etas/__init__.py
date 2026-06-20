"""
ETAS Python package.

A pure-Python implementation of the Epidemic-Type Aftershock Sequence (ETAS)
model for seismicity, translating the core functionality of the R ETAS package.
"""

from .R.catalog import catalog, Catalog, print_catalog
from .R.etas import etas, ETASResult, print_etas
from .R.rates import rates, probs
from .R.simulate import simetas
from .R.residuals import residuals
from .R.search_isc import search_isc

__all__ = [
    'catalog',
    'Catalog',
    'print_catalog',
    'etas',
    'ETASResult',
    'print_etas',
    'rates',
    'probs',
    'simetas',
    'residuals',
    'search_isc'
]

__version__ = '0.1.0'
