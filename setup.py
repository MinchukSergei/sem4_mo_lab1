from setuptools import setup

setup(
    name='lab1',
    version='1.0',
    description='MO Prepare notMnist set',
    author='Sergey Minchuk',
    packages=['lab1'],  # same as name
    install_requires=['opencv-python', 'numpy', 'scikit-learn', 'tqdm'],  # external packages as dependencies
)
