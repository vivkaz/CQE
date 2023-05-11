from setuptools import setup

setup(
    name='CQE',
    version='1.0.1',
    packages=['CQE','CQE.unit_classifier'],
    package_data={'': ['unit.json'],'': ['unit_models.zip']},# both has to be empty
    url='',
    license='',
    author='satyaalmasian and vivian kazakova',
    author_email='satya.almasian@gmail.com',
    description='quantity extractor',
    install_requires=['fuzzywuzzy', 'spacy==3.0.9', 'greek'],
)
