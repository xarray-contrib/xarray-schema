from invoke import task

SRC_DIR = 'xarray_schema'
TEST_DIR = 'tests'


@task
def mypy(c):  # pragma: no cover
    c.run(f'mypy {SRC_DIR} {TEST_DIR}')
