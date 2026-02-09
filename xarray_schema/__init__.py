import importlib.metadata

from .base import SchemaError  # noqa: F401
from .components import (  # noqa: F401
    ArrayTypeSchema,
    AttrSchema,
    AttrsSchema,
    ChunksSchema,
    DimsSchema,
    DTypeSchema,
    NameSchema,
    ShapeSchema,
)
from .dataarray import CoordsSchema, DataArraySchema  # noqa: F401
from .dataset import DatasetSchema  # noqa: F401

__version__ = importlib.metadata.version('xarray-schema')
