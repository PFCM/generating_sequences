"""So that we can setup links on the python path."""
from setuptools import setup
import os


with open(os.path.abspath('./README.md')) as f:
    descr = f.read()

setup(
    name='models',
    version='0.0.0',
    description='some rnn models for generating sequences',
    long_description=descr,
    url='https://github.com/PFCM/generating_sequences',
    author='pfcm',
    license='MIT',
    packages=['models']
)
