#!/usr/bin/env python

"""The setup script."""

from os.path import exists

from setuptools import find_packages, setup

with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

if exists('README.md'):
    with open('README.md') as f:
        long_description = f.read()
else:
    long_description = ''

CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Topic :: Scientific/Engineering',
]

setup(
    name='xarray-schema',
    description='Schema validation for Xarray objects',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.8',
    maintainer='Joe Hamman',
    maintainer_email='joe@carbonplan.org',
    classifiers=CLASSIFIERS,
    url='https://github.com/carbonplan/xarray-schema',
    packages=find_packages(exclude=('tests',)),
    package_dir={'xarray_schema': 'xarray_schema'},
    include_package_data=True,
    install_requires=install_requires,
    license='MIT',
    zip_safe=False,
    keywords=['xarray', 'schema'],
    use_scm_version={'version_scheme': 'post-release', 'local_scheme': 'dirty-tag'},
    setup_requires=['setuptools_scm', 'setuptools>=30.3.0'],
)
