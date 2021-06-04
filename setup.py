from setuptools import setup

setup(
    name='geometric_attention',
    version='0.2',
    description='geometric attention',
    packages=['geometric_attention'],  # same as name
    install_requires=['torch', 'torchvision', 'numpy', 'sklearn', 'skorch', 'scipy', 'matplotlib'],# external packages as dependencies
)