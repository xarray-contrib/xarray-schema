from invoke import task  # pragma: no cover

SRC_DIR = 'xarray_schema'  # pragma: no cover
TEST_DIR = 'tests'  # pragma: no cover


@task
def mypy(c):  # pragma: no cover
    c.run(f'mypy {SRC_DIR} {TEST_DIR}')
