from typing import Any, Callable, Dict, Hashable, Iterable, Optional, Union

import xarray as xr

from .base import BaseSchema, SchemaError
from .components import AttrSchema, AttrsSchema
from .dataarray import CoordsSchema, DataArraySchema


class DatasetSchema(BaseSchema):
    '''A light-weight xarray.Dataset validator

    Parameters
    ----------
    data_vars : mapping of variable names and DataArraySchemas, optional
        Per-variable DataArraySchema's, by default None
    checks : Iterable[Callable], optional
        Dataset wide checks, by default None
    '''

    _json_schema = {
        'type': 'object',
        'properties': {
            'data_vars': {'type': 'object'},
            'coords': {'type': 'object'},
            'attrs': {'type': 'object'},
        },
    }

    def __init__(
        self,
        data_vars: Optional[Dict[Hashable, Optional[DataArraySchema]]] = None,
        coords: Union[CoordsSchema, Dict[Hashable, DataArraySchema], None] = None,
        attrs: Union[AttrsSchema, Dict[Hashable, AttrSchema], None] = None,
        checks: Iterable[Callable] = None,
    ) -> None:
        self.data_vars = data_vars  # type: ignore
        self.coords = coords  # type: ignore
        self.attrs = attrs  # type: ignore
        self.checks = checks

    @classmethod
    def from_json(cls, obj: dict):
        kwargs = {}
        if 'data_vars' in obj:
            kwargs['data_vars'] = {
                k: DataArraySchema.from_json(v) for k, v in obj['data_vars'].items()
            }
        if 'coords' in obj:
            kwargs['coords'] = {k: CoordsSchema.from_json(v) for k, v in obj['coords'].items()}
        if 'attrs' in obj:
            kwargs['attrs'] = {k: AttrsSchema.from_json(v) for k, v in obj['attrs'].items()}

        return cls(**kwargs)

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
    def attrs(self) -> Union[AttrsSchema, None]:
        return self._attrs

    @attrs.setter
    def attrs(self, value: Union[AttrsSchema, Dict[Hashable, Any], None]):
        if value is None or isinstance(value, AttrsSchema):
            self._attrs = value
        else:
            self._attrs = AttrsSchema(value)

    @property
    def data_vars(self) -> Optional[Dict[Hashable, Optional[DataArraySchema]]]:
        return self._data_vars  # type: ignore

    @data_vars.setter
    def data_vars(self, value: Optional[Dict[Hashable, Optional[DataArraySchema]]]):
        if isinstance(value, dict):
            self._data_vars = {}
            for k, v in value.items():
                if isinstance(v, DataArraySchema):
                    self._data_vars[k] = v
                else:
                    self._data_vars[k] = DataArraySchema(**v)  # type: ignore
        elif value is None:
            self._data_vars = None  # type: ignore
        else:
            raise ValueError('must set data_vars with a dict')

    @property
    def coords(self) -> Optional[CoordsSchema]:
        return self._coords  # type: ignore

    @coords.setter
    def coords(self, value: Optional[Union[CoordsSchema, Dict[Hashable, DataArraySchema]]]):
        if value is None or isinstance(value, CoordsSchema):
            self._coords = value
        else:
            self._coords = CoordsSchema(value)

    @property
    def json(self):
        obj = {'data_vars': {}, 'attrs': self.attrs.json if self.attrs is not None else {}}
        if self.data_vars:
            for key, var in self.data_vars.items():
                obj['data_vars'][key] = var.json
        if self.coords:
            obj['coords'] = self.coords.json
        return obj
