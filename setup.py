from setuptools import setup, find_packages
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "The MSAgent is designed to work with Python 3.6 and greater."

setup(
    name='msagent',
    py_modules=['msagent'],
    version='0.1',
    scripts=['msagent/msagent'],
    packages=find_packages(),
    install_requires=[
        'gym[atari]~=0.15.3',
        'atari-py==0.2.5',
        'pyzmq',
        'pickle5',
        'pyarrow',
        'python-consul',
        'ipython',
        'matplotlib==3.1.1',
        'numpy',
        'psutil',
        'tqdm',
        'redis'
    ],
    description="Distributed framework of deep RL.",
    author="Junge Zhang, Bailin Wang, Kaiqi Huang",
)
