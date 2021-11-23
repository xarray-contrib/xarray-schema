import numpy as np
import pytest
import xarray as xr

from xarray_schema import DataArraySchema, DatasetSchema
from xarray_schema.core import SchemaError


def test_dataarray_empty_constructor():

    da_schema = DataArraySchema()
    assert hasattr(da_schema, 'validate')


def test_dataarray_validate_dtype():

    da = xr.DataArray(np.ones(4, dtype='i4'))
    schema = DataArraySchema(dtype='i4')
    schema.validate(da)

    schema = DataArraySchema(dtype=np.int32)
    schema.validate(da)

    schema = DataArraySchema(dtype=np.integer)
    schema.validate(da)

    schema = DataArraySchema(dtype=np.floating)
    with pytest.raises(SchemaError, match=r'.*floating.*'):
        schema.validate(da)


def test_dataarray_validate_name():

    da = xr.DataArray(np.ones(4), name='foo')
    schema = DataArraySchema(name='foo')
    schema.validate(da)

    schema = DataArraySchema(name='bar')
    with pytest.raises(SchemaError, match=r'.*foo.*'):
        schema.validate(da)


def test_dataarray_validate_shape():

    da = xr.DataArray(np.ones(4))
    schema = DataArraySchema(shape=(4,))
    schema.validate(da)

    schema = DataArraySchema(shape=(4, 2))
    with pytest.raises(SchemaError, match=r'.*ndim.*'):
        schema.validate(da)

    schema = DataArraySchema(shape=(3,))
    with pytest.raises(SchemaError, match=r'.*(4).*'):
        schema.validate(da)


def test_dataarray_validate_dims():

    da = xr.DataArray(np.ones(4), dims=['x'])
    schema = DataArraySchema(dims=['x'])
    schema.validate(da)

    schema = DataArraySchema(dims=(['x', 'y']))
    with pytest.raises(SchemaError, match=r'.*length of dims.*'):
        schema.validate(da)

    schema = DataArraySchema(dims=['y'])
    with pytest.raises(SchemaError, match=r'.*(y).*'):
        schema.validate(da)


def test_dataarray_validate_array_type():

    da = xr.DataArray(np.ones(4), dims=['x'])
    schema = DataArraySchema(array_type=np.ndarray)
    schema.validate(da)

    schema = DataArraySchema(array_type=float)
    with pytest.raises(SchemaError, match=r'.*(float).*'):
        schema.validate(da)


def test_dataarray_validate_chunks():
    pytest.importorskip('dask')

    da = xr.DataArray(np.ones(4), dims=['x']).chunk({'x': 2})
    schema = DataArraySchema(chunks={'x': 2})
    schema.validate(da)

    schema = DataArraySchema(chunks={'x': (2, 2)})
    schema.validate(da)

    schema = DataArraySchema(chunks={'x': [2, 2]})
    schema.validate(da)

    schema = DataArraySchema(chunks={'x': 3})
    with pytest.raises(SchemaError, match=r'.*(3).*'):
        schema.validate(da)

    schema = DataArraySchema(chunks={'x': (2, 1)})
    with pytest.raises(SchemaError, match=r'.*(2, 1).*'):
        schema.validate(da)

    # check that when expected chunk == -1 it fails
    schema = DataArraySchema(chunks={'x': -1})
    with pytest.raises(SchemaError, match=r'.*(4).*'):
        schema.validate(da)

    # check that when chunking schema is -1 it also works
    # both when chunking is specified as -1 and as 4
    schema = DataArraySchema(chunks={'x': 4})
    da = xr.DataArray(np.ones(4), dims=['x']).chunk({'x': -1})
    schema.validate(da)

    schema = DataArraySchema(chunks={'x': -1})
    da = xr.DataArray(np.ones(4), dims=['x']).chunk({'x': 4})
    schema.validate(da)

    schema = DataArraySchema(chunks={'x': -1})
    da = xr.DataArray(np.ones(4), dims=['x']).chunk({'x': -1})
    schema.validate(da)

    # test for agnostic chunks
    schema = DataArraySchema(chunks=True)
    da = xr.DataArray(np.ones(4), dims=['x'])
    with pytest.raises(SchemaError, match='Schema expected DataArray to be chunked but it is not'):
        schema.validate(da)

    # now try passing an irregularly chunked data array
    da = xr.DataArray(np.ones(4), dims=['x']).chunk({'x': (1, 2, 1)})
    schema.validate(da)

    # test the check for regular chunk sizes
    schema = DataArraySchema(chunks={'x': -1})
    with pytest.raises(AssertionError, match=r'.*(gracious).*'):
        schema.validate(da)


def test_dataset_empty_constructor():
    ds_schema = DatasetSchema()
    assert hasattr(ds_schema, 'validate')


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


def test_validate():
    da = xr.DataArray(np.ones(4), dims=['x']).chunk({'x': (1, 2, 1)})
    schema = DataArraySchema(chunks=False)
    # check that da is unchunked
    with pytest.raises(SchemaError, match='Schema expected unchunked DataArray but it is chunked!'):
        schema.validate(da)
    da = xr.DataArray(np.ones(4), dims=['x'])
    schema.validate(da)
