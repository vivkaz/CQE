# Extension of Numerical Information Extractor
The goal of this project is to extract numerical information from unstructured textual data such as news and financial articles in the English language and transform it into a structured representation.
The model detects changes, values, units and referred concepts from sentences, and normalize them. See [DOCUMENTATION.md](priavte_modules/DOCUMENTATION.md) for more details.

>Advanced Software Practical, Summer 2022
>
>Author: Vivian Kazakova ([Email](mailto:vivian.kazakova@stud.uni-heidelberg.de))
>
>Supervisor: Satya Almasian

## Getting Started
Download or clone the repository to your local machine.

### Prerequisites
Make sure you have Python 3.8 and spaCy 3.0 installed. You may also need to install some python packages. Run
```
pip install -r requirements.txt
```
### Pip
you can also install the package using on the root directory of the package.
```
pip install .
```
### Usage
Create a `NumParser` and parse some text or sentence.

```python

from numparser import NumParser

parser = NumParser()
text = "The sp 500 was down 2.1% and nasdaq fell 2.5%."
result = parser.parse(text)
print(result)

>> > [(=, 2.1, [%], percentage, [sp, 500]), (=, 2.5, [%], percentage, [nasdaq])]
```
See the example in [numparser/example.py](numparser/example.py) as well. Run
```python
python3 numparser/example.py
```
Use the overload option for additional functionality. The NumParser will compute the span indices of the Quantity, the normalized input sentence, the long and the simplified scientific notation of the Value, whether the unit is scientific or noun based and the unit surface forms.
```python
parser = NumParser(overload=True)
text = "The sp 500 was down 2.1% and nasdaq fell 2.5%."
result = parser.parse(text)

for res in result:
    print(f"""
	Quantity: {res}
	=====
	indices                         =   {res.get_char_indices()}
	normalized text                 =   {res.get_normalized_text()}
	pre processed text              =   {res.get_preprocessed_text()}
	scientific notation             =   {res.value.scientific_notation}
	simplified scientific notation  =   {res.value.simplified_scientific_notation}
	scientific unit                 =   {res.unit.scientific}
	unit surfaces forms             =   {res.unit.unit_surfaces_forms}""")

>>> Quantity: (down,2.1,[%],percentage,{0: [sp, 500]})
=====
indices                         =   [5, 6]
normalized text                 =   The sp 500 was down 2.1 percentage and nasdaq fell 2.5 percentage .
pre processed text              =   The sp 500 was down 2.1% and nasdaq fell 2.5% .
scientific notation             =   2.100000e+00
simplified scientific notation  =   2.1e+00
scientific unit                 =   True
unit surfaces forms             =   ['percentage', 'percent', 'pc', '%', 'pct', 'pct.']


Quantity: (down,2.5,[%],percentage,{0: [nasdaq]})
=====
indices                         =   [10, 11]
normalized text                 =   The sp 500 was down 2.1 percentage and nasdaq fell 2.5 percentage .
pre processed text              =   The sp 500 was down 2.1% and nasdaq fell 2.5% .
scientific notation             =   2.500000e+00
simplified scientific notation  =   2.5e+00
scientific unit                 =   True
unit surfaces forms             =   ['percentage', 'percent', 'pc', '%', 'pct', 'pct.']
```
## Overview

### Files
| File                                                                                                                                                           | Description |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------| ---         |
| [numparser/NumberNormalizer.py](numparser/NumberNormalizer.py)                                                                                                 | Change, Value and Unit normalization script |
| [numparser/NumberParser.py](NumberParser.py)                                                                                                                   | Quantity extraction script |
| [numparser/rules.py](numparser/rules.py)                                                                                                                       | Rules for [DependencyMatcher](https://spacy.io/usage/rule-based-matching#dependencymatcher) and [Matcher](https://spacy.io/usage/rule-based-matching)|
| [numparser/unit.json](data/unit.json)                                                                                                                          | 529 units used for the Unit Normalization |
| [numparser/classes.py](numparser/classes.py)                                                                                                                   | Definition of the Change, Range, Value, Unit, Concept and Quanitity classes |
| [numparser/number_lookup.py](numparser/number_lookup.py)                                                                                                       | Number-word to number mappings, bounds, prefixes and suffixes |
| [numparser/example.py](numparser/example.py)                                                                                                                   | Usage example |
| [evaluation-scripts](evaluation-scripts/)                                                                                                                      | Folder containing the evaluation script ([evaluation-scripts/evaluate_models.py](evaluate_models.py)) and additional scripts used for the evaluation of the models |
| [data/formatted_test_set](data/formatted_test_set/)                                                                                                            | Folder with the test dataset - in total there are 6 json files: the Microsoft.Recognizers.Text test dataset (Age-, Currency-, Dimension- and TemperatureModel), NewsQuant (own dataset in 2 versions) |
| [data/evaluation_output](data/evaluation_output/)                                                                                                              | Folder containing the evaluation results on the different json test dataset files |
| [data/gpt_3_output](data/gpt_3_output)                                                                                                                         | Folder containing the predictions made by ChatGPT3 on the different json test datasets |
| [private_modules](priavte_modules/)                                                                                                                            | Folder containing old and extra files |
| [evaluation-scripts/gpt-3/gpt3-tag.py](evaluation-scripts/gpt-3/gpt3-tag.py)                                                                                   | Uses open ai API to tag the test set.                                                                                       |
| [evaluation-scripts/tagger.py](evaluation-scripts/tagger.py)                                                                                                   | Creates a unified representation for different models, by adding normalization and text cleaning to prepare for evaluation. |
| [evaluation-scripts/evaluate_models.py](evaluation-scripts/evaluate_models.py)                                                                                 | Evaluation script for CQE and other baselines                                                                               |
| [evaluation-scripts/test_unit_classifier.py](evaluation-scripts/test_unit_classifier.py)                                                                       | Evaluation code for unit disambiguator                                                                                      |                                        |
| [evaluation-scripts/significance_test/significance_test.py](evaluation-scripts/evaluation/significance_test.py)                                                | Computing P values based on the F1 scores for different systems.                                                            |
| [evaluation-scripts/significance_test/permutation_significance_test.py](sevaluation-scripts/ignificance_test/permutation_significance_test.py)                 | Code for permutation based significance testing for the specific output of CQE                                              |                       |
| [evaluation-scripts/significance_test/significance_testing_for_arrary_input.py](evaluation-scripts/significance_test/significance_testing_for_arrary_input.py) | Code for permutation based significance testing for normal  array input, used for the classification of unit disambiguator  |                                        |

### Data
The following datasets are used for the evaluation of the model.
|Dataset | #Sentences | Source |
|----------------|----------------|----------------
| [NewsQuant.json](data/NewsQuant.json)           | 590        | 906         | tagged by the authors                                                                                                                                       |
| [NewsQuant_version2.json](priavte_modules/data/NewsQuant_version2.json) | 527 | enlarged own dataset |
| [age-model.json](data/formatted_test_set//age-model.json) | 19 | [Microsoft.Recognizers.Text Test Cases Specs](https://github.com/microsoft/Recognizers-Text/blob/master/Specs/NumberWithUnit/English/AgeModel.json) |
| [currency-model.json](data/formatted_test_set/currency-model.json) | 180 | [Microsoft.Recognizers.Text Test Cases Specs](https://github.com/microsoft/Recognizers-Text/blob/master/Specs/NumberWithUnit/English/CurrencyModel.json)|
| [dimension-model.json](data/formatted_test_set/dimension-model.json) | 93 | [Microsoft.Recognizers.Text Test Cases Specs](https://github.com/microsoft/Recognizers-Text/blob/master/Specs/NumberWithUnit/English/DimensionModel.json) |
| [temperature-model.json](data/formatted_test_set/temperature-model.json) | 36 | [Microsoft.Recognizers.Text Test Cases Specs](https://github.com/microsoft/Recognizers-Text/blob/master/Specs/NumberWithUnit/English/TemperatureModel.json) |
The predictions from GPT-3 for the test datasets above can be found in  `/data/gpt_3_ouput`.

To train and test the unit classifier we generated data using ChatGPT, for the prompts used refer to the paper.
The generated sentences and their classes are under `/data/units/train` (1,827 samples) and `/data/units/test` (180 samples).



Run
```
python3 evaluate_models.py
```

### Units
The information needed for unit normalization of the extracted quantities is stored in the [unit.json](data/unit.json) file. Each of the 531 units includes surfaces, symbols, prefixes, entity, URI, dimensions, and currency_code.

To create this file, a combination of units from [quantulum3](https://github.com/nielstron/quantulum3/blob/dev/quantulum3/units.json), [units](https://en.wikipedia.org/wiki/Template:Convert/list_of_units) from Wikipedia, surfaces from [Microsoft.Recognizers.Text](https://github.com/microsoft/Recognizers-Text/blob/master/Patterns/English/English-NumbersWithUnit.yaml), and the [UCUM units and surfaces](https://github.com/lhncbc/ucum-lhc/blob/master/data/ucumDefs.json) were utilized. Modifications, adaptations, and additions were made as necessary to create the final list of units.

Example:
```json
"euro": {
    "surfaces": [
        "Euro",
        "Euros",
        "euro",
        "euros"
    ],
    "entity": "currency",
    "URI": "Euro",
    "dimensions": [],
    "symbols": [
        "EUR",
        "eur",
        "\u20ac"
    ],
    "currency_code": "EUR"
},
...
"metre": {
    "surfaces": [
        "meter",
        "meter-long",
        "meters",
        "metre",
        "metre-long",
        "metres"
    ],
    "entity": "length",
    "URI": "Metre",
    "dimensions": [],
    "symbols": [
        "m"
    ],
    "prefixes": [
        "q",
        "r",
        "y",
        "z",
        "a",
        "f",
        "p",
        "n",
        "\u03bc",
        "\u00b5",
        "m",
        "c",
        "d",
        "h",
        "k",
        "M",
        "G",
        "T",
        "P",
        "E",
        "Z",
        "Y",
        "R",
        "Q"
    ]
},
...
"light-year": {
    "surfaces": [
        "light year",
        "light years",
        "light-year"
    ],
    "entity": "length",
    "URI": "Light-year",
    "dimensions": [],
    "symbols": [
        "ly"
    ]
},
```
### Rules
There are more than 60 rules for [DependencyMatcher](https://spacy.io/usage/rule-based-matching#dependencymatcher) and [Matcher](https://spacy.io/usage/rule-based-matching) defined in the [rules.py](CQE/rules.py). We use the [en core web sm](https://spacy.io/models/en) spaCy-model to create a [Doc object](https://spacy.io/api/doc) with various [linguistic annotations](https://spacy.io/usage/linguistic-featuress). It's important to note that these rules are not solely based on simple pattern matching using individual words in the sentence. Instead, they leverage the linguistic annotations and sentence structure to perform their matching.

Existing rules can be modified or new ones can be added by editing the file. However, please ensure that you pay attention to the syntax of the DependencyMatcher when making any changes.

Example:
```json
"num_symbol" : [
{
"RIGHT_ID": "number",
"RIGHT_ATTRS": {"POS": "NUM"}
},
{
"LEFT_ID": "number",
"REL_OP": ">",
"RIGHT_ID": "symbol",
"RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "nmod"]}, "POS": "SYM", "ORTH": {"NOT_IN": ["#"]}}
},
],

...

"noun_num" : [
{
"RIGHT_ID": "noun",
"RIGHT_ATTRS": {"POS": {"IN": ["PROPN", "NOUN", "INTJ"]}, "ORTH": {"NOT_IN": ["#", "ers"]}}
},
{
"LEFT_ID": "noun",
"REL_OP": ">",
"RIGHT_ID": "number",
"RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "nmod", "nummod", "compound", "amod", "nsubj"]}, "POS": "NUM"}
},
],

...

"phone_number_pattern_1" : [
{"ORTH": "("},
{"SHAPE": "d"},
{"ORTH": ")"},
{"SHAPE": "dd"},
{"ORTH": "-", "OP": "?"},
{"SHAPE": "ddd"},
{"ORTH": "-", "OP": "?"},
{"SHAPE": "dddd"}
],
```
```
Input: "The September crude contract was up 19 cents at US $58.24 per barrel and the September natural gas contract was up 10.4 cents to US $2.24 per mmBTU."

Matches:
NUM_SYMBOL [58.24, US$]
NUM_SYMBOL [2.24, US$]
NOUN_NUM [cents, 19]
NOUN_NUM [cents, 10.4]
QUANTMOD_DIRECT_NUM [19, up]
QUANTMOD_DIRECT_NUM [10.4, up]
NOUN_NOUN [contract, gas, natural]
UNIT_FRAC [58.24, per, barrel]
UNIT_FRAC [2.24, per, mmBTU]
LONELY_NUM [19]
LONELY_NUM [58.24]
LONELY_NUM [10.4]
LONELY_NUM [2.24]

Candidates: [[US$, 58.24, per, barrel, 10], [US$, 2.24, per, mmBTU, 25], [up, 19, cents, 6], [up, 10.4, cents, 21]]
Quadruples: [([], [58.24], [US$, per, barrel], 10), ([], [2.24], [US$, per, mmBTU], 25), ([up], [19], [cents], 6), ([up], [10.4], [cents], 21)]

Output: [(up,19.0,[cents],cent,{0: [September, crude, contract]}), (=,58.24,[US$, per, barrel],dollar per barrel,{0: [September, crude, contract]}), (up,10.4,[cents],cent,{0: [September, natural, gas, contract]}), (=,2.24,[US$, per, mmBTU],dollar per mega british thermal unit,{0: [September, natural, gas, contract]})]
```
_Please note that in each candidate and 4-tuple, the numbers 6, 10, 21, and 25 indicate the position of the quantity within the text._
