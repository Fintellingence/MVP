from setuptools import setup, find_packages
setup(
    name = 'classes',
    # We insert the next line to include datafiles
    # https://setuptools.readthedocs.io/en/latest/userguide/datafiles.html
    include_package_data=True,
    packages = find_packages(),
)