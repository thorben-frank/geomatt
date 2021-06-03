from setuptools import setup

setup(
    name='geometric_attention',
    version='0.2',
    description='geometric attention',
    packages=['geometric_attention'],  # same as name
    install_requires=['torch>=1.6.0','torchvision>=0.7.0', 'numpy==1.19.1', 'sklearn', 'skorch', 'scipy', 'matplotlib'],# external packages as dependencies
)