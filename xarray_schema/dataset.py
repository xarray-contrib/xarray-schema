from typing import Any, Callable, Dict, Hashable, Iterable, Union

import xarray as xr

from .base import BaseSchema, SchemaError
from .dataarray import DataArraySchema


class DatasetSchema(BaseSchema):
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

    def validate(self, ds: xr.Dataset) -> None:
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
                    if key not in ds.data_vars:
                        raise SchemaError(f'data variable {key} not in ds')
                    else:
                        da_schema.validate(ds.data_vars[key])

        if self.coords is not None:  # pragma: no cover
            raise NotImplementedError('coords schema not implemented yet')

        if self.attrs:
            self.attrs.validate(ds.attrs)

        if self.checks:
            for check in self.checks:
                check(ds)

    @property
    def json(self):
        obj = {'data_vars': {}, 'attrs': self.attrs.json if self.attrs is not None else None}
        if self.data_vars:
            for key, var in self.data_vars.items():
                obj['data_vars'][key] = var.json
        return obj
