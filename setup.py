from setuptools import setup
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='CQE',
    version='1.0.6',
    packages=['CQE','CQE.unit_classifier'],
    package_data={'': ['unit.json'],'': ['unit_models.zip']},# both has to be empty
    url='https://github.com/vivkaz/CQE',
    license='',
    long_description_content_type="text/markdown",
    long_description=long_description,
    author='satyaalmasian and vivian kazakova',
    author_email='satya.almasian@gmail.com',
    description='quantity extractor',
    install_requires=['fuzzywuzzy==0.18.0', 'more_itertools','ordered-set','python-Levenshtein',
                      'spacy==3.0.9', 'greek','spacy-legacy','requests',
                      'regex','emoji==1.7','torch==2.0.0','spacy-transformers==1.0.4','protobuf==3.20.1','inflect==5.4.0',
                      'spacy_download'],
)






