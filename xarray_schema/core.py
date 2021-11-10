from typing import Any, Callable, Dict, Hashable, Iterable, Tuple, Union

import numpy as np
import xarray as xr

# TODOs:
# - api grouping, should the constructors look similar to the DataArray/Dataset constructors


class SchemaError(Exception):
    '''Custom Schema Error'''

    pass


class DataArraySchema:
    '''A light-weight xarray.DataArray validator

    Parameters
    ----------
    dtype : Any, optional
        Datatype of the the variable. If a string is specified it must be a valid NumPy data type value, by default None
    shape : Tuple[Union[int, None]], optional
        Shape of the DataArray. `None` may be used as a wildcard value. By default None
    dims : Tuple[Union[Hashable, None]], optional
        Dimensions of the DataArray.  `None` may be used as a wildcard value. By default None
    name : str, optional
        Name of the DataArray, by default None
    array_type : Any, optional
        Type of the underlying data in a DataArray (e.g. `numpy.ndarray`), by default None
    checks : Iterable[Callable], optional
        List of callables that take and return a DataArray, by default None'''

    def __init__(
        self,
        dtype: Any = None,
        shape: Tuple[Union[int, None]] = None,
        dims: Tuple[Union[Hashable, None]] = None,
        coords: Dict[Hashable, Any] = None,
        chunks: Dict[Hashable, Union[int, None]] = None,
        name: str = None,
        array_type: Any = None,
        attrs: Dict[Hashable, Any] = None,
        checks: Iterable[Callable] = None,
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
        assert isinstance(da, xr.core.dataarray.DataArray),'Input is not an xarray DataArray and schema for chunks are not yet implemented'

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
            dim_chunks = dict(zip(da.dims, da.chunks))
            for key, ec in self.chunks.items():
                if isinstance(ec, int):
                    # handles case of expected chunksize is shorthand of -1 which translates to the full length of dimension
                    if ec==-1:
                        ec = len(da[key])
                        # grab the first entry in da's tuple of chunks to be representative (as it should be assuming they're regular)
                    ac = dim_chunks[key][0]
                    if ac != ec:
                        raise SchemaError(f'{key} chunks did not match: {ac} != {ec}')

                else:  # assumes ec is an iterable
                    ac = dim_chunks[key]
                    if tuple(ac) != tuple(ec):
                        raise SchemaError(f'{key} chunks did not match: {ac} != {ec}')

        if self.attrs:
            raise NotImplementedError('attrs schema not implemented yet')

        if self.array_type and not isinstance(da.data, self.array_type):
            raise SchemaError(f'array_type {type(da.data)} != {self.array_type}')

        if self.checks:
            for check in self.checks:
                da = check(da)


class DatasetSchema:
    '''A light-weight xarray.Dataset validator

    Parameters
    ----------
    data_vars : mapping of variable names and DataArraySchemas, optional
        Per-variable DataArraySchema's, by default None
    checks : Iterable[Callable], optional
        Dataset wide checks, by default None
    '''

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
        '''Check if the Dataset complies with the Schema.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset to be validated

        Returns
        -------
        xr.Dataset
            Validated Dataset

        Raises
        ------
        SchemaError
        '''

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
