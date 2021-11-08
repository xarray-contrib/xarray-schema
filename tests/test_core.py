from xarray_schema import DataArraySchema, DatasetSchema


def test_empyt_constructors():

    da_schema = DataArraySchema()
    assert hasattr(da_schema, 'validate')
    ds_schema = DatasetSchema()
    assert hasattr(ds_schema, 'validate')
