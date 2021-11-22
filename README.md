<img
  src='https://carbonplan-assets.s3.amazonaws.com/monogram/dark-small.png'
  height='48'
/>

# xarray-schema

Schema validation for Xarray

[![CI](https://github.com/carbonplan/ndpyramid/actions/workflows/main.yaml/badge.svg)](https://github.com/carbonplan/xarray-schema/actions/workflows/main.yaml)
![MIT License](https://badgen.net/badge/license/MIT/blue)

# installation

This package is in the early stages of development. Install it from source:

```shell
pip install git+git://github.com/carbonplan/xarray-schema
```

# usage

Xarray-schema's API is modeled after [Pandera](https://pandera.readthedocs.io/en/stable/). The `DataArraySchema` and `DatasetSchema` objects both have `.validate()` methods.

The basic usage is as follows:

```python
import numpy as np
import xarray as xr
from xarray_schema import DataArraySchema, DatasetSchema

da = xr.DataArray(np.ones(4, dtype='i4'), dims=['x'], name='foo')

schema = DataArraySchema(dtype=np.integer, name='foo', shape=(4, ), dims=['x'])

schema.validate(da)
```

You can also use it to validate a Dataset like so:
```
schema_ds = DatasetSchema({'foo': schema})

schema_ds.validate(da.to_dataset())

# roadmap

This is a very early prototype of a library. Some key things are missing:

1. Validation of `coords`, `chunks`, and `attrs`. None of these are implemented yet.
1. Class-based schema's for parts of the Xarray data model. Most validations are currently made as direct comparisons (`da.name == self.name`) but a more robust approach is possible that leverages classes for each component of the data model. We're already handling some special cases using `None` as a sentinel value to allow for wildcard-like behavior in places (i.e. `dims` and `shape`)
1. Exceptions: Pandera accumulates schema exceptions and reports them all at once. Currently, we are a eagerly raising `SchemaErrors` when the are found.

## license

All the code in this repository is [MIT](https://choosealicense.com/licenses/mit/) licensed, but we request that you please provide attribution if reusing any of our digital content (graphics, logo, articles, etc.).

## about us

CarbonPlan is a non-profit organization working on the science and data of carbon removal. We aim to improve the transparency and scientific integrity of carbon removal and climate solutions through open data and tools. Find out more at [carbonplan.org](https://carbonplan.org/) or get in touch by [opening an issue](https://github.com/carbonplan/xarray-schema/issues/new) or [sending us an email](mailto:hello@carbonplan.org).
