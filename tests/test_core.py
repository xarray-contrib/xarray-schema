import numpy as np
import pytest
import xarray as xr

from xarray_schema import DataArraySchema, DatasetSchema
from xarray_schema.base import SchemaError
from xarray_schema.components import (
    ArrayTypeSchema,
    AttrSchema,
    AttrsSchema,
    ChunksSchema,
    DimsSchema,
    DTypeSchema,
    NameSchema,
    ShapeSchema,
)


@pytest.fixture
def ds():
    ds = xr.Dataset(
        {
            'x': xr.DataArray(np.arange(4) - 2, dims='x'),
            'foo': xr.DataArray(np.ones(4, dtype='i4'), dims='x'),
            'bar': xr.DataArray(np.arange(8, dtype=np.float32).reshape(4, 2), dims=('x', 'y')),
        }
    )
    return ds


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
        # schema_args for ChunksSchema include [chunks, dims, shape]
        (ChunksSchema, True, [(((1, 1),), ('x',), (2,))], True),
        (ChunksSchema, {'x': 2}, [(((2, 2),), ('x',), (4,))], {'x': 2}),
        (ChunksSchema, {'x': (2, 2)}, [(((2, 2),), ('x',), (4,))], {'x': [2, 2]}),
        (ChunksSchema, {'x': [2, 2]}, [(((2, 2),), ('x',), (4,))], {'x': [2, 2]}),
        (ChunksSchema, {'x': 4}, [(((4,),), ('x',), (4,))], {'x': 4}),
        (ChunksSchema, {'x': -1}, [(((4,),), ('x',), (4,))], {'x': -1}),
        (ChunksSchema, {'x': (1, 2, 1)}, [(((1, 2, 1),), ('x',), (4,))], {'x': [1, 2, 1]}),
        (
            ChunksSchema,
            {'x': 2, 'y': -1},
            [(((2, 2), (10,)), ('x', 'y'), (4, 10))],
            {'x': 2, 'y': -1},
        ),
        (
            AttrsSchema,
            {'foo': AttrSchema(value='bar')},
            [{'foo': 'bar'}],
            {'foo': {'type': None, 'value': 'bar'}},
        ),
        (
            AttrsSchema,
            {'foo': AttrSchema(value=1)},
            [{'foo': 1}],
            {'foo': {'type': None, 'value': 1}},
        ),
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
    assert isinstance(schema.to_json(), str)


@pytest.mark.parametrize(
    'type, value, validate, json',
    [
        (str, None, 'foo', {'type': str, 'value': None}),
        (None, 'foo', 'foo', {'type': None, 'value': 'foo'}),
        (str, 'foo', 'foo', {'type': str, 'value': 'foo'}),
    ],
)
def test_attr_schema(type, value, validate, json):
    schema = AttrSchema(type=type, value=value)
    schema.validate(validate)
    assert schema.json == json
    # assert isinstance(schema.to_json(), str)


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
        # schema_args for ChunksSchema include [chunks, dims, shape]
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
        (ChunksSchema, {'x': 2}, (((2, 3, 2),), ('x',), (7,)), r'.*did not match.*'),
        (ChunksSchema, {'x': 2}, (((2, 2, 3),), ('x',), (7,)), r'.*did not match.*'),
        (ChunksSchema, {'x': 2, 'y': -1}, (((2, 2), (5, 5)), ('x', 'y'), (4, 10)), r'.*(5).*'),
    ],
)
def test_component_raises_schema_error(component, schema_args, validate, match):
    schema = component(schema_args)
    with pytest.raises(SchemaError, match=match):
        if component in [ChunksSchema]:  # special case construction
            schema.validate(*validate)
        else:
            schema.validate(validate)


def test_chunks_schema_raises_for_invalid_chunks():
    with pytest.raises(ValueError, match=r'.*int.*'):
        schema = ChunksSchema(chunks=2)
        schema.validate(((2, 2),), ('x',), (4,))


def test_dataarray_empty_constructor():

    da = xr.DataArray(np.ones(4, dtype='i4'))
    da_schema = DataArraySchema()
    assert hasattr(da_schema, 'validate')
    assert da_schema.json == {}
    da_schema.validate(da)


@pytest.mark.parametrize(
    'kind, component, schema_args',
    [
        ('dtype', DTypeSchema, 'i4'),
        ('dims', DimsSchema, ('x', None)),
        ('shape', ShapeSchema, (2, None)),
        ('name', NameSchema, 'foo'),
        ('array_type', ArrayTypeSchema, np.ndarray),
        ('chunks', ChunksSchema, False),
    ],
)
def test_dataarray_component_constructors(kind, component, schema_args):
    da = xr.DataArray(np.zeros((2, 4), dtype='i4'), dims=('x', 'y'), name='foo')
    comp_schema = component(schema_args)
    schema = DataArraySchema(**{kind: schema_args})
    assert comp_schema.json == getattr(schema, kind).json
    assert isinstance(getattr(schema, kind), component)

    schema.validate(da)


def test_dataarray_schema_validate_raises_for_invalid_input_type():
    ds = xr.Dataset()
    schema = DataArraySchema()
    with pytest.raises(ValueError, match='Input must be a xarray.DataArray'):
        schema.validate(ds)


def test_dataset_empty_constructor():
    ds_schema = DatasetSchema()
    assert hasattr(ds_schema, 'validate')
    ds_schema.json == {}


def test_dataset_example(ds):

    ds_schema = DatasetSchema(
        {
            'foo': DataArraySchema(name='foo', dtype=np.int32, dims=['x']),
            'bar': DataArraySchema(name='bar', dtype=np.floating, dims=['x', 'y']),
        }
    )
    assert list(ds_schema.json['data_vars'].keys()) == ['foo', 'bar']
    ds_schema.validate(ds)

    ds['foo'] = ds.foo.astype('float32')
    with pytest.raises(SchemaError, match='dtype'):
        ds_schema.validate(ds)

    ds = ds.drop_vars('foo')
    with pytest.raises(SchemaError, match='variable foo'):
        ds_schema.validate(ds)


def test_checks_ds(ds):
    def check_foo(ds):
        assert 'foo' in ds

    ds_schema = DatasetSchema(checks=[check_foo])
    ds_schema.validate(ds)

    ds = ds.drop_vars('foo')
    with pytest.raises(AssertionError):
        ds_schema.validate(ds)

    ds_schema = DatasetSchema(checks=[])
    ds_schema.validate(ds)

    # TODO
    # with pytest.raises(ValueError):
    #     DatasetSchema(checks=[2])


def test_checks_da(ds):
    da = ds['foo']

    def check_foo(da):
        assert da.name == 'foo'

    def check_bar(da):
        assert da.name == 'bar'

    schema = DataArraySchema(checks=[check_foo])
    schema.validate(da)

    schema = DataArraySchema(checks=[check_bar])
    with pytest.raises(AssertionError):
        schema.validate(da)

    schema = DataArraySchema(checks=[])
    schema.validate(da)

    with pytest.raises(ValueError):
        DataArraySchema(checks=[2])
