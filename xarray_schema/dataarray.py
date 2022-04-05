from typing import Any, Callable, Dict, List, Mapping, Union

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

        self.dtype = dtype
        self.shape = shape
        self.dims = dims
        self.name = name
        self.coords = coords
        self.chunks = chunks
        self.attrs = attrs
        self.array_type = array_type
        self.checks = checks

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
    def shape(self) -> ShapeSchema:
        return self._shape

    @shape.setter
    def shape(self, value: Union[ShapeSchema, ShapeT, None]):
        if value is None or isinstance(value, ShapeSchema):
            self._shape = value
        else:
            self._shape = ShapeSchema(value)

    @property
    def chunks(self) -> ChunksSchema:
        return self._chunks

    @chunks.setter
    def chunks(self, value):
        if value is None or isinstance(value, ChunksSchema):
            self._chunks = value
        else:
            self._chunks = ChunksSchema(value)

    @property
    def name(self) -> NameSchema:
        return self._name

    @name.setter
    def name(self, value):
        if value is None or isinstance(value, NameSchema):
            self._name = value
        else:
            self._name = NameSchema(value)

    @property
    def array_type(self) -> ArrayTypeSchema:
        return self._array_type

    @array_type.setter
    def array_type(self, value):
        if value is None or isinstance(value, ArrayTypeSchema):
            self._array_type = value
        else:
            self._array_type = ArrayTypeSchema(value)

    @property
    def attrs(self) -> AttrsSchema:
        return self._attrs

    @attrs.setter
    def attrs(self, value):
        if value is None or isinstance(value, AttrsSchema):
            self._attrs = value
        else:
            self._attrs = AttrsSchema(value)

    @property
    def coords(self) -> Mapping:
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


class CoordsSchema(BaseSchema):

    _json_schema = {'type': 'string'}

    def __init__(
        self,
        coords: Mapping[str, Any],
        require_all_keys: bool = True,
        allow_extra_keys: bool = True,
    ) -> None:
        self.coords = coords
        self.require_all_keys = require_all_keys
        self.allow_extra_keys = allow_extra_keys

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
        return {k: v.json for k, v in self.attrs.items()}
