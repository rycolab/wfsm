from setuptools import setup, find_packages

setup(name='wfsm',
      version='1.0',
      description='Derivatives for WFSM',
      author='Ran Zmigrod',
      url='https://github.com/rycolab/wfsm',
      install_requires=[
            'numpy', 'torch', 'numba'
      ],
      packages=find_packages(),
      )
