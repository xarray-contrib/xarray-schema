import numpy as np
import pytest
import xarray as xr

from xarray_schema import DataArraySchema, DatasetSchema
from xarray_schema.base import SchemaError
from xarray_schema.components import (
    ArrayTypeSchema,
    ChunksSchema,
    DimsSchema,
    DTypeSchema,
    NameSchema,
    ShapeSchema,
)


@pytest.mark.parametrize(
    'component, schema_args, validate, json',
    [
        (DTypeSchema, np.integer, ['i4', 'int', np.int32], 'integer'),
        (DTypeSchema, np.int64, ['i8', np.int64], '<i8'),
        (DTypeSchema, '<i8', ['i8', np.int64], '<i8'),
        (DimsSchema, ('foo', None), [('foo', 'bar'), ('foo', 'baz')], ['foo', None]),
        (DimsSchema, ('foo', 'bar'), [('foo', 'bar')], ['foo', 'bar']),
        (ShapeSchema, (1, 2, None), [(1, 2, 3), (1, 2, 5)], [1, 2, None]),
        (ShapeSchema, (1, 2, 3), [(1, 2, 3)], [1, 2, 3]),
        (NameSchema, 'foo', ['foo'], 'foo'),
        (ArrayTypeSchema, np.ndarray, [np.array([1, 2, 3])], "<class 'numpy.ndarray'>"),
        (ChunksSchema, True, [(((1, 1),), ('x',), (2,))], True),
        (ChunksSchema, {'x': 2}, [(((2, 2),), ('x',), (4,))], {'x': 2}),
        (ChunksSchema, {'x': (2, 2)}, [(((2, 2),), ('x',), (4,))], {'x': [2, 2]}),
        (ChunksSchema, {'x': [2, 2]}, [(((2, 2),), ('x',), (4,))], {'x': [2, 2]}),
        (ChunksSchema, {'x': 4}, [(((4,),), ('x',), (4,))], {'x': 4}),
        (ChunksSchema, {'x': -1}, [(((4,),), ('x',), (4,))], {'x': -1}),
        (ChunksSchema, {'x': (1, 2, 1)}, [(((1, 2, 1),), ('x',), (4,))], {'x': [1, 2, 1]}),
    ],
)
def test_component_schema(component, schema_args, validate, json):
    schema = component(schema_args)
    for v in validate:
        if component in [ChunksSchema]:  # special case construction
            schema.validate(*v)
        else:
            schema.validate(v)
    assert schema.json == json


@pytest.mark.parametrize(
    'component, schema_args, validate, match',
    [
        (DTypeSchema, np.integer, np.float32, r'.*float.*'),
        (DimsSchema, ('foo', 'bar'), ('foo',), r'.*length.*'),
        (DimsSchema, ('foo', 'bar'), ('foo', 'baz'), r'.*mismatch.*'),
        (ShapeSchema, (1, 2, None), (1, 2), r'.*number of dimensions.*'),
        (ShapeSchema, (1, 4, 4), (1, 3, 4), r'.*mismatch.*'),
        (NameSchema, 'foo', 'bar', r'.*name bar != foo.*'),
        (ArrayTypeSchema, np.ndarray, 'bar', r'.*array_type.*'),
        (ChunksSchema, {'x': 3}, (((2, 2),), ('x',), (4,)), r'.*(3).*'),
        (ChunksSchema, {'x': (2, 1)}, (((2, 2),), ('x',), (4,)), r'.*(2, 1).*'),
        (ChunksSchema, True, (None, ('x',), (4,)), r'.*expected array to be chunked.*'),
        (
            ChunksSchema,
            False,
            (((2, 2),), ('x',), (4,)),
            r'.*expected unchunked array but it is chunked*',
        ),
        (ChunksSchema, {'x': -1}, (((1, 2, 1),), ('x',), (4,)), r'.*did not match.*'),
    ],
)
def test_component_raises_schema_error(component, schema_args, validate, match):
    schema = component(schema_args)
    with pytest.raises(SchemaError, match=match):
        if component in [ChunksSchema]:  # special case construction
            schema.validate(*validate)
        else:
            schema.validate(validate)


def test_dataarray_empty_constructor():

    da = xr.DataArray(np.ones(4, dtype='i4'))
    da_schema = DataArraySchema()
    assert hasattr(da_schema, 'validate')
    assert da_schema.json == {}
    da_schema.validate(da)


def test_dataarray_validate_dtype():

    da = xr.DataArray(np.ones(4, dtype='i4'))
    schema = DataArraySchema(dtype='i4')
    schema.validate(da)

    component = schema.dtype
    assert isinstance(component, DTypeSchema)


def test_dataset_empty_constructor():
    ds_schema = DatasetSchema()
    assert hasattr(ds_schema, 'validate')
    ds_schema.json == {}


def test_dataset_example():

    ds = xr.Dataset(
        {
            'x': xr.DataArray(np.arange(4) - 2, dims='x'),
            'foo': xr.DataArray(np.ones(4, dtype='i4'), dims='x'),
            'bar': xr.DataArray(np.arange(8, dtype=np.float32).reshape(4, 2), dims=('x', 'y')),
        }
    )

    ds_schema = DatasetSchema(
        {
            'foo': DataArraySchema(name='foo', dtype=np.int32, dims=['x']),
            'bar': DataArraySchema(name='bar', dtype=np.floating, dims=['x', 'y']),
        }
    )
    ds_schema.validate(ds)

    assert list(ds_schema.json['data_vars'].keys()) == ['foo', 'bar']
