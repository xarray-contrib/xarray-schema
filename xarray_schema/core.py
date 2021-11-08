from typing import Any, Callable, Dict, Hashable, Iterable, Tuple, Union

import numpy as np
import xarray as xr

# Leaving off here:
# Need to:
# - write tests
# - typing
# - document missing features (mostly class based args)
# - api grouping, should the constructors look similar to the DataArray/Dataset constructors


class SchemaError(Exception):
    pass


class DataArraySchema:
    def __init__(
        self,
        dtype: Any = None,
        shape: Tuple[Union[int, None]] = None,
        dims: Tuple[Union[Hashable, None]] = None,
        coords: Dict[Hashable, Any] = None,
        chunks: Dict[Hashable, Union[int, None]] = None,
        name: str = None,
        checks: Iterable[Callable] = None,
        array_type: Any = None,
        attrs: Dict[Hashable, Any] = None,
    ) -> None:

        self.dtype = dtype
        self.shape = shape
        self.dims = dims
        self.coords = coords
        self.name = name
        self.chunks = chunks
        self.attrs = attrs
        self.array_type = array_type

        self.checks = checks if checks is not None else []

    def validate(self, da: xr.DataArray) -> xr.DataArray:

        if self.dtype is not None and not np.issubdtype(da.dtype, self.dtype):
            raise SchemaError(f'dtype {da.dtype} != {self.dtype}')

        if self.name is not None and self.name != da.name:
            raise SchemaError(f'name {da.name} != {self.name}')

        if self.shape is not None:
            if len(self.shape) != da.ndim:
                raise SchemaError(
                    f'number of dimensions in shape ({da.ndim}) != da.ndim ({len(self.shape)})'
                )

            for i, (actual, expected) in enumerate(zip(da.shape, self.shape)):
                if expected is not None and actual != expected:
                    raise SchemaError(f'shape mismatch in axis {i}: {actual} != {expected}')

        if self.dims is not None:
            if len(self.dims) != len(da.dims):
                raise SchemaError(
                    f'length of dims does not match: {len(da.dims)} != {len(self.dims)}'
                )

            for i, (actual, expected) in enumerate(zip(da.dims, self.dims)):
                if expected is not None and actual != expected:
                    raise SchemaError(f'dim mismatch in axis {i}: {actual} != {expected}')

        if self.coords is not None:
            raise NotImplementedError('coords schema not implemented yet')

        if self.chunks:
            raise NotImplementedError('chunk schema not implemented yet')

        if self.attrs:
            raise NotImplementedError('attrs schema not implemented yet')

        if self.array_type and not isinstance(da.data, self.array_type):
            raise SchemaError(f'array_type {type(da.data)} != {self.array_type}')

        if self.checks:
            for check in self.checks:
                da = check(da)

        return da


class DatasetSchema:
    def __init__(
        self,
        data_vars: Dict[Hashable, Union[DataArraySchema, None]] = None,
        coords: Dict[Hashable, Any] = None,
        attrs: Dict[Hashable, Any] = None,
        checks: Iterable[Callable] = None,
    ) -> None:

        self.data_vars = data_vars
        self.coords = coords
        self.attrs = attrs
        self.checks = checks

    def validate(self, ds: xr.Dataset) -> xr.Dataset:

        if self.data_vars is not None:
            for key, da_schema in self.data_vars.items():
                if da_schema is not None:
                    da_schema.validate(ds.data_vars[key])
                else:
                    if key not in ds.data_vars:
                        raise SchemaError(f'data variable {key} not in ds')

        if self.coords is not None:
            raise NotImplementedError('coords schema not implemented yet')

        if self.attrs:
            raise NotImplementedError('attrs schema not implemented yet')

        if self.checks:
            for check in self.checks:
                ds = check(ds)

        return ds
