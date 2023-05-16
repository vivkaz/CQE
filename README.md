# CQE
## Evaultion and data in [CQE_Evaluation](https://github.com/satya77/CQE_Evaluation).

A Framework for Comprehensive Quantity Extraction. This repository contains code for the paper:

[CQE: A Framework for Comprehensive Quantity Extraction
](https://arxiv.org/pdf/2305.08853v1.pdf)

Satya Almasian*, Vivian Kazakova*, Philipp Göldner, Michael Gertz  
Institute of Computer Science, Heidelberg University  
(`*` indicates equal contribution)

If you found this useful, consider citing us: 
```
@misc{almasian2023cqe,
      title={CQE: A Comprehensive Quantity Extractor}, 
      author={Satya Almasian and Vivian Kazakova and Philip Göldner and Michael Gertz},
      year={2023},
      eprint={2305.08853},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
### Prerequisites
you can also install the package using on the root directory of the package.
```
pip install .
```
### Usage
Create a `NumParser` and parse some text or sentence.
```python
from CQE import CQE

parser = CQE.NumParser()
text = "The sp 500 was down 2.1% and nasdaq fell 2.5%."
result = parser.parse(text)
print(result)

>>> [(=,2.1,[%],percentage,[sp, 500]), (=,2.5,[%],percentage,[nasdaq])]
```
Use the overload option for additional functionality. The NumParser will compute the span indices of the Quantity, the normalized input sentence, the long and the simplified scientific notation of the Value, whether the unit is scientific or noun based and the unit surface forms.
```python
parser = CQE.NumParser(overload=True)
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
See the example in [CQE/example.py](example.py) as well. Run
```python
python3 CQE/example.py
```
### Evaluation and Data
For replicating the results on the paper and comparing against other system, make sure CQE is installed and use
the [CQE_Evaluation](https://github.com/satya77/CQE_Evaluation) repo.
The evaluation script and data used for evaluation and training unit disambiguators are in this repository.

### File and folder structure
Main files for CQE are under CQE package, where `unit_classifer` contains code for unit disambiguation based on BERT classifier trained using spacy-transformers. `units.json` file is used for normalization of units and `unit_models.zip`
contains the trained models for the disambiguation which will be unziped on the first run of `NumParser`class.

| File                                                                                                       | Description                                                                                                                 |
|------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| [CQE/NumberNormalizer.py](CQE/NumberNormalizer.py)                                                         | Bound, Number and Unit Normalization script                                                                                 |
| [CQE/NumberParser.py](CQE/NumberParser.py)                                                                 | Quantity Extraction script                                                                                                  |
| [CQE/rules.py](CQE/rules.py)                                                                               | Rules for [DependencyMatcher](https://spacy.io/usage/rule-based-matching#dependencymatcher)                                 |
| [CQE/unit.json](CQE/unit.json)                                                                             | 531 units used for the Unit Normalization                                                                                   |
| [CQE/classes.py](CQE/classes.py)                                                                           | Definition of the Bound, Range, Number, Unit, Noun and Quanitity classes                                                    |
| [CQE/number_lookup.py](CQE/number_lookup.py)                                                               | Number-word to number mappings                                                                                              |
| [CQE/example.py](example.py)                                                                               | Usage example                                                                                                               |
| [CQE/unit_classifer/unit_disambiguator.py](CQE/unit_classifer/unit_disambiguator.py)                       | Class for unit disambiguator based on the bert based classifiers.                                                           |
| [CQE/unit_classifer/train_classifier_bert.py](CQE/unit_classifer/train_classifier_bert.py)                 | Script for generating spacy based training data and training commands to create classifiers for disambiguation.             |
| [CQE/unit_classifer/sample_usage.py](CQE/unit_classifer/sample_usage.py)                                   | Usage example for disambiguation class.                                                                                     |



### Units
The units used for normalization of the unit of an extracted quantity are stored in the [unit.json](data/unit.json) . Each of the 531 units has surfaces, symbols, prefixes, entity, URI, dimensions and currency_code. For composing the file, the list of units from [quantulum3](https://github.com/nielstron/quantulum3/blob/dev/quantulum3/units.json), the list of units from [Wikipedia](https://en.wikipedia.org/wiki/Template:Convert/list_of_units), the surfaces from [Microsoft.Recognizers.Text](https://github.com/microsoft/Recognizers-Text/blob/master/Patterns/English/English-NumbersWithUnit.yaml) ,the [UCUM](https://github.com/lhncbc/ucum-lhc/blob/master/data/ucumDefs.json) units and surfaces and
wikipedia page of [units] (https://en.wikipedia.org/wiki/Template:Convert/list_of_units)
were used.

Example:
```json
"light-year":  {
	"surfaces":  [
		"light-year",
		"light year",
		"light years"
	],
	"entity":  "length",
	"URI":  "Light-year",
	"dimensions":  [],
	"symbols":  [
		"ly",
		"[ly]"
	]
}
```
### Rules
There are more than 50 rules for [DependencyMatcher](https://spacy.io/usage/rule-based-matching#dependencymatcher) defined in the [rules.py](rules.py). We use the spaCy-model [en core web sm](https://spacy.io/models/en) to create a [Doc object](https://spacy.io/api/doc) with [linguistic annotations](https://spacy.io/usage/linguistic-featuress). The key point is that the rules are not simple pattern matching based on the single words in the sentence, but on those annotations and exploit the structure of the sentence.

Existing rules can be changed and new ones can be added by editing the file. Pay attention to the DependencyMatcher syntax.

Example:
```json
"num_symbol"  :  [
	{
		"RIGHT_ID":  "number",
		"RIGHT_ATTRS":  {"POS":  "NUM"}
	},
	{
		"LEFT_ID":  "number",
		"REL_OP":  ">",
		"RIGHT_ID":  "symbol",
		"RIGHT_ATTRS":  {"DEP":  {"IN":  ["quantmod",  "nmod"]},  "POS":  "SYM"}
	},
]
```
```
Input: "The September crude contract was up 19 cents at US $58.24 per barrel and the September natural gas contract was up 10.4 cents to US $2.24 per mmBTU."

Matches:
NUM_SYMBOL [58.24, US$]
NUM_SYMBOL [2.24, US$]
NOUN_NUM [cents, 19]
NOUN_NUM [cents, 10.4]
NUM_RIGHT_NOUN [58.24, barrel]
NUM_RIGHT_NOUN [2.24, mmBTU]
NOUN_NOUN [contract, gas, natural]
UNIT_FRAC [58.24, per, barrel]
UNIT_FRAC [58.24, per, gas]
UNIT_FRAC [58.24, per, contract]
UNIT_FRAC [58.24, per, cents]
UNIT_FRAC [58.24, per, mmBTU]
UNIT_FRAC [2.24, per, mmBTU]
UNIT_FRAC_2 [58.24, per, gas, natural]
LONELY_NUM [19]
LONELY_NUM [58.24]
LONELY_NUM [10.4]
LONELY_NUM [2.24]

Candidates: [[US$, 58.24, per, barrel, 10], [US$, 2.24, per, mmBTU, 25], [19, cents, 6], [10.4, cents, 21]]
Quadruples: [([], [58.24], [US$, per, barrel], 10), ([], [2.24], [US$, per, mmBTU], 25), ([], [19], [cents], 6), ([], [10.4], [cents], 21)]

Output: [(=,58.24,[US$, per, barrel],united states dollar / barrel,[September, crude, contract]), (=,2.24,[US$, per, mmBTU],united states dollar / mmBTU,[September, natural, gas, contract]), (=,19.0,[cents],cent,[September, crude, contract]), (=,10.4,[cents],cent,[September, natural, gas, contract])]
```
_Note that the numbers 6, 10, 21 and 25 indicate the position of the quantity in the text._
