from setuptools import setup

setup(
    name='geomatt',
    version='0.2',
    description='geometric attention',
    packages=['geomatt'],  # same as name
    install_requires=['torch', 'torchvision', 'numpy', 'sklearn', 'skorch', 'scipy', 'matplotlib'],# external packages as dependencies
)