from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import xarray as xr

from .base import BaseSchema, SchemaError
from .components import (
    ArrayTypeSchema,
    AttrsSchema,
    ChunksSchema,
    DimsSchema,
    DTypeSchema,
    NameSchema,
    ShapeSchema,
)
from .types import ChunksT, DimsT, DTypeLike, ShapeT


class DataArraySchema(BaseSchema):
    '''A light-weight xarray.DataArray validator

    Parameters
    ----------
    dtype : DTypeLike or DTypeSchema, optional
        Datatype of the the variable. If a string is specified it must be a valid NumPy data type value, by default None
    shape : ShapeT or ShapeSchema, optional
        Shape of the DataArray. `None` may be used as a wildcard value. By default None
    dims : DimsT or DimsSchema, optional
        Dimensions of the DataArray.  `None` may be used as a wildcard value. By default None
    chunks : Union[bool, Dict[str, Union[int, None]]], optional
        If bool, specifies whether DataArray is chunked or not, agnostic to chunk sizes.
        If dict, includes the expected chunks for the DataArray, by default None
    name : str, optional
        Name of the DataArray, by default None
    array_type : Any, optional
        Type of the underlying data in a DataArray (e.g. `numpy.ndarray`), by default None
    checks : List[Callable], optional
        List of callables that take and return a DataArray, by default None
    '''

    _json_schema = {'type': 'object'}
    _schema_slots = ['dtype', 'dims', 'shape', 'coords', 'name', 'chunks', 'attrs', 'array_type']

    _dtype: Union[DTypeSchema, None]
    _shape: Union[ShapeSchema, None]
    _dims: Union[DimsSchema, None]
    _name: Union[NameSchema, None]
    _coords: Union[Any, None]
    _chunks: Union[ChunksSchema, None]
    _attrs: Union[AttrsSchema, None]
    _array_type: Union[ArrayTypeSchema, None]

    def __init__(
        self,
        dtype: Union[DTypeLike, DTypeSchema] = None,
        shape: Union[ShapeT, ShapeSchema] = None,
        dims: Union[DimsT, DimsSchema] = None,
        name: Union[str, NameSchema] = None,
        coords: Dict[str, Any] = None,
        chunks: Union[ChunksT, ChunksSchema] = None,
        array_type: Any = None,
        attrs: Mapping[str, Any] = None,
        checks: List[Callable] = None,
    ) -> None:
        # see https://github.com/python/mypy/issues/3004
        self.dtype = dtype  # type: ignore
        self.shape = shape  # type: ignore
        self.dims = dims  # type: ignore
        self.name = name  # type: ignore
        self.coords = coords  # type: ignore
        self.chunks = chunks  # type: ignore
        self.attrs = attrs  # type: ignore
        self.array_type = array_type  # type: ignore
        self.checks = checks  # type: ignore

    @property
    def dtype(self) -> Union[DTypeSchema, None]:
        return self._dtype

    @dtype.setter
    def dtype(self, value: Union[DTypeSchema, DTypeLike, None]):
        if value is None or isinstance(value, DTypeSchema):
            self._dtype = value
        else:
            self._dtype = DTypeSchema(value)

    @property
    def dims(self) -> Union[DimsSchema, None]:
        return self._dims

    @dims.setter
    def dims(self, value):
        if value is None or isinstance(value, DimsSchema):
            self._dims = value
        else:
            self._dims = DimsSchema(value)

    @property
    def shape(self) -> Optional[ShapeSchema]:
        return self._shape

    @shape.setter
    def shape(self, value: Union[ShapeSchema, ShapeT, None]):
        if value is None or isinstance(value, ShapeSchema):
            self._shape = value
        else:
            self._shape = ShapeSchema(value)

    @property
    def chunks(self) -> Optional[ChunksSchema]:
        return self._chunks

    @chunks.setter
    def chunks(self, value):
        if value is None or isinstance(value, ChunksSchema):
            self._chunks = value
        else:
            self._chunks = ChunksSchema(value)

    @property
    def name(self) -> Optional[NameSchema]:
        return self._name

    @name.setter
    def name(self, value):
        if value is None or isinstance(value, NameSchema):
            self._name = value
        else:
            self._name = NameSchema(value)

    @property
    def array_type(self) -> Optional[ArrayTypeSchema]:
        return self._array_type

    @array_type.setter
    def array_type(self, value):
        if value is None or isinstance(value, ArrayTypeSchema):
            self._array_type = value
        else:
            self._array_type = ArrayTypeSchema(value)

    @property
    def attrs(self) -> Optional[AttrsSchema]:
        return self._attrs

    @attrs.setter
    def attrs(self, value):
        if value is None or isinstance(value, AttrsSchema):
            self._attrs = value
        else:
            self._attrs = AttrsSchema(value)

    @property
    def coords(self) -> Optional[CoordsSchema]:
        return self._coords

    @coords.setter
    def coords(self, value):
        if value is None or isinstance(value, CoordsSchema):
            self._coords = value
        else:
            self._coords = CoordsSchema(value)

    @property
    def checks(self) -> List[Callable]:
        return self._checks

    @checks.setter
    def checks(self, value):
        if value is not None:
            if not all([callable(f) for f in value]):
                raise ValueError('All checks must be callables')
            self._checks = value
        else:
            self._checks = []

    def validate(self, da: xr.DataArray) -> None:
        '''Check if the DataArray complies with the Schema.

        Parameters
        ----------
        da : xr.DataArray
            DataArray to be validated

        Returns
        -------
        xr.DataArray
            Validated DataArray

        Raises
        ------
        SchemaError
        '''
        if not isinstance(da, xr.DataArray):
            raise ValueError('Input must be a xarray.DataArray')

        if self.dtype is not None:
            self.dtype.validate(da.dtype)

        if self.name is not None:
            self.name.validate(da.name)

        if self.dims is not None:
            self.dims.validate(da.dims)

        if self.shape is not None:
            self.shape.validate(da.shape)

        if self.coords is not None:
            self.coords.validate(da.coords)

        if self.chunks is not None:
            self.chunks.validate(da.chunks, da.dims, da.shape)

        if self.attrs:
            self.attrs.validate(da.attrs)

        if self.array_type is not None:
            self.array_type.validate(da.data)

        for check in self.checks:
            check(da)

    @property
    def json(self) -> dict:
        obj = {}
        for slot in self._schema_slots:
            try:
                obj[slot] = getattr(self, slot).json
            except AttributeError:
                pass
        return obj

    @classmethod
    def from_json(cls, obj: dict):
        kwargs = {}

        if 'dtype' in obj:
            kwargs['dtype'] = DTypeSchema.from_json(obj['dtype'])
        if 'shape' in obj:
            kwargs['shape'] = ShapeSchema.from_json(obj['shape'])
        if 'dims' in obj:
            kwargs['dims'] = DimsSchema.from_json(obj['dims'])
        if 'name' in obj:
            kwargs['name'] = NameSchema.from_json(obj['name'])
        if 'coords' in obj:
            kwargs['coords'] = CoordsSchema.from_json(obj['coords'])
        if 'chunks' in obj:
            kwargs['chunks'] = ChunksSchema.from_json(obj['chunks'])
        if 'array_type' in obj:
            kwargs['array_type'] = ArrayTypeSchema.from_json(obj['array_type'])
        if 'attrs' in obj:
            kwargs['attrs'] = AttrsSchema.from_json(obj['attrs'])

        return cls(**kwargs)


class CoordsSchema(BaseSchema):
    '''Schema container for Coordinates

    Parameters
    ----------
    coords : dict
        Dict of coordinate keys and ``DataArraySchema`` objects
    require_all_keys : bool
        Whether require to all coordinates included in ``coords``
    allow_extra_keys : bool
        Whether to allow coordinates not included in ``coords`` dict

    Raises
    ------
    SchemaError
    '''

    _json_schema = {
        'type': 'object',
        'properties': {
            'require_all_keys': {
                'type': 'boolean'
            },  # Question: is this the same as JSON's additionalProperties?
            'allow_extra_keys': {'type': 'boolean'},
            'coords': {'type': 'object'},
        },
    }

    def __init__(
        self,
        coords: Mapping[str, DataArraySchema],
        require_all_keys: bool = True,
        allow_extra_keys: bool = True,
    ) -> None:
        self.coords = coords
        self.require_all_keys = require_all_keys
        self.allow_extra_keys = allow_extra_keys

    @classmethod
    def from_json(cls, obj: dict):
        coords = obj.pop('coords', {})
        coords = {k: DataArraySchema(**v) for k, v in coords.items()}
        return cls(coords, **obj)

    def validate(self, coords: Any) -> None:
        '''Validate coords

        Parameters
        ----------
        coords : dict_like
            coords of the DataArray. `None` may be used as a wildcard value.
        '''

        if self.require_all_keys:
            missing_keys = set(self.coords) - set(coords)
            if missing_keys:
                raise SchemaError(f'coords has missing keys: {missing_keys}')

        if not self.allow_extra_keys:
            extra_keys = set(coords) - set(self.coords)
            if extra_keys:
                raise SchemaError(f'coords has extra keys: {extra_keys}')

        for key, da_schema in self.coords.items():
            if key not in coords:
                raise SchemaError(f'key {key} not in coords')
            else:
                da_schema.validate(coords[key])

    @property
    def json(self) -> dict:
        obj = {
            'require_all_keys': self.require_all_keys,
            'allow_extra_keys': self.allow_extra_keys,
            'coords': {k: v.json for k, v in self.coords.items()},
        }
        return obj
