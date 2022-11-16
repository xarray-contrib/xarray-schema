#!/usr/bin/env python3
# flake8: noqa

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError, version as _version

from .base import SchemaError
from .components import (
    ArrayTypeSchema,
    AttrSchema,
    AttrsSchema,
    ChunksSchema,
    DimsSchema,
    DTypeSchema,
    NameSchema,
    ShapeSchema,
)
from .dataarray import CoordsSchema, DataArraySchema
from .dataset import DatasetSchema

try:
    __version__ = _version(__name__)
except _PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"  # pragma: no cover
