from setuptools import setup

setup(name='gym_multigrid',
      version='0.0.1',
        packages=['gym_multigrid', 'gym_multigrid.envs'],
        install_requires=[
        'gym>=0.9.6',
        'numpy>=1.15.0'
        ]
)