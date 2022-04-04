from pkg_resources import DistributionNotFound, get_distribution

from .base import SchemaError  # noqa: F401
from .components import (  # noqa: F401
    ArrayTypeSchema,
    ChunksSchema,
    DimsSchema,
    DTypeSchema,
    NameSchema,
    ShapeSchema,
)
from .dataarray import CoordsSchema, DataArraySchema  # noqa: F401
from .dataset import DatasetSchema  # noqa: F401

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:  # noqa: F401; pragma: no cover
    # package is not installed
    pass
