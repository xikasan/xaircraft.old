# coding: utf-8

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='xaircraft',
    version='0.2.1',
    description="aircraft model and env set",
    long_description=readme,
    author='xikasan',
    # author_email='',
    url='https://github.com/xikasan/xaircraft',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
