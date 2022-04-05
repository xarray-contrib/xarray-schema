<p align="left">
  <a href="https://carbonplan.org/#gh-light-mode-only">
    <img src="https://carbonplan-assets.s3.amazonaws.com/monogram/dark-small.png" height="48px" />
  </a>
  <a href="https://carbonplan.org/#gh-dark-mode-only">
    <img src="https://carbonplan-assets.s3.amazonaws.com/monogram/light-small.png" height="48px" />
  </a>
</p>

# xarray-schema

Schema validation for Xarray

[![CI](https://github.com/carbonplan/ndpyramid/actions/workflows/main.yaml/badge.svg)](https://github.com/carbonplan/xarray-schema/actions/workflows/main.yaml)
[![codecov](https://codecov.io/gh/carbonplan/xarray-schema/branch/main/graph/badge.svg?token=EI729ZRFK0)](https://codecov.io/gh/carbonplan/xarray-schema)
![MIT License](https://badgen.net/badge/license/MIT/blue)

## installation

Install xarray-schema from PyPI:

```shell
pip install xarray-schema
```

Conda:

```shell
conda install -c conda-forge xarray-schema
```

Or install it from source:

```shell
pip install git+https://github.com/carbonplan/xarray-schema
```

## usage

Xarray-schema's API is modeled after [Pandera](https://pandera.readthedocs.io/en/stable/). The `DataArraySchema` and `DatasetSchema` objects both have `.validate()` methods.

The basic usage is as follows:

```python
import numpy as np
import xarray as xr
from xarray_schema import DataArraySchema, DatasetSchema, CoordsSchema

da = xr.DataArray(np.ones(4, dtype='i4'), dims=['x'], name='foo')

schema = DataArraySchema(dtype=np.integer, name='foo', shape=(4, ), dims=['x'])

schema.validate(da)
```

You can also use it to validate a `Dataset` like so:

```
schema_ds = DatasetSchema({'foo': schema})

schema_ds.validate(da.to_dataset())
```

Each component of the Xarray data model is implemented as a stand alone class:

```python
from xarray_schema.components import (
    DTypeSchema,
    DimsSchema,
    ShapeSchema,
    NameSchema,
    ChunksSchema,
    ArrayTypeSchema,
    AttrSchema,
    AttrsSchema
)

# example constructions
dtype_schema = DTypeSchema('i4')
dims_schema = DimsSchema(('x', 'y', None))  # None is used as a wildcard
shape_schema = ShapeSchema((5, 10, None))  # None is used as a wildcard
name_schema = NameSchema('foo')
chunk_schema = ChunkSchema({'x': None, 'y': -1})  # None is used as a wildcard, -1 is used as
ArrayTypeSchema = ArrayTypeSchema(np.ndarray)

# Example usage
dtype_schama.validate(da.dtype)

# Each object schema can be exported to JSON format
dtype_json = dtype_schama.to_json()
```

## roadmap

This is a very early prototype of a library. Some key things are missing:

1. Validation of `coords` and `attrs`. These are not implemented yet.
1. Exceptions: Pandera accumulates schema exceptions and reports them all at once. Currently, we are a eagerly raising `SchemaErrors` when the are found.
1. Roundtrip schemas to/from JSON and/or YAML format.

## license

All the code in this repository is [MIT](https://choosealicense.com/licenses/mit/) licensed, but we request that you please provide attribution if reusing any of our digital content (graphics, logo, articles, etc.).

## about us

CarbonPlan is a non-profit organization that uses data and science for climate action. We aim to improve the transparency and scientific integrity of climate solutions through open data and tools. Find out more at [carbonplan.org](https://carbonplan.org/) or get in touch by [opening an issue](https://github.com/carbonplan/xarray-schema/issues/new) or [sending us an email](mailto:hello@carbonplan.org).
