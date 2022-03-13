from setuptools import find_packages, setup

setup(
    name='gym_battleship',
    version='0.0.3',
    install_requires=['gym', 'numpy', 'pandas', 'ipython'],
    # extras_require pip install -e .[test]
    extras_require={'test': ['pytest']},
    packages=find_packages()
)
