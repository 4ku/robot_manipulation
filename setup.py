from setuptools import setup, find_packages

setup(
    name="offline_sac",
    version="0.1",
    packages=find_packages(where='sac_rnd'),
    package_dir={'': 'sac_rnd'},
)