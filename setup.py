from setuptools import setup
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='CQE',
    version='1.0.2',
    packages=['CQE','CQE.unit_classifier'],
    package_data={'': ['unit.json'],'': ['unit_models.zip']},# both has to be empty
    url='https://github.com/vivkaz/CQE',
    license='',
    long_description_content_type="text/markdown",
    long_description=long_description,
    author='satyaalmasian and vivian kazakova',
    author_email='satya.almasian@gmail.com',
    description='quantity extractor',
    install_requires=['fuzzywuzzy', 'spacy==3.0.9', 'greek'],
)
