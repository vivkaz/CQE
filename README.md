# CQE
A Framework for Comprehensive Quantity Extraction. This repository contains code for the paper:

[CQE: A Framework for Comprehensive Quantity Extraction
]()
 
Satya Almasian*, Vivian Kazakova*, Philipp Göldner, Michael Gertz  
Institute of Computer Science, Heidelberg University  
(`*` indicates equal contribution)

### Prerequisites
Make sure you have Python 3.9 and spaCy 3.0.9 installed. You may also need to install some python packages. Run
```
pip install -r requirements.txt
```
If you do not want to evaluate the system you can remove `recognizers-text, 
recognizers-text-suite, cython ,ccg_nlpy, quantulum3,scikit-learn` from the requirements for a more lightweight installation. 
### Usage
Create a `NumParser` and parse some text or sentence.
```python
import NumParser

parser = NumParser()
text = "The sp 500 was down 2.1% and nasdaq fell 2.5%."
result = parser.parse(text)
print(result)

>>> [(=,2.1,[%],percentage,[sp, 500]), (=,2.5,[%],percentage,[nasdaq])]
```
See the example in [CQE/example.py](example.py) as well. Run
```python
python3 example.py
```
you can also install the package using on the root directory of the package. 
```
python setup.py install

```
### File and folder structure 
Main files for CQE are under CQE package, where `unit_classifer` contains code for unit disambiguation based on BERT classifier trained using spacy-transformers. `units.json` file is used for normalization of units and `unit_models.zip`
contains the trained models for the disambiguation which will be unziped on the first run of `NumParser`class.
To replicated the results from the paper and signifcant testing, refer to `evaluation` package. The computed results are also present under `data/evaluation_output`.

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
| [evaluation/gpt-3/gpt3-tag.py](evaluation/gpt-3/gpt3-tag.py)                                               | Uses open ai API to tag the test set.                                                                                       |
| [evaluation/tagger.py](evaluation/tagger.py)                                                               | Creates a unified representation for different models, by adding normalization and text cleaning to prepare for evaluation. |
| [evaluation/evaluate_models.py](evaluation/evaluate_models.py)                                             | Evaluation script for CQE and other baselines                                                                               |
| [evaluation/significance_test.py](evaluation/significance_test.py)                                         | Computing P values based on the F1 scores for different systems.                                                            |
| [evaluation/permutation_significance_test.py](evaluation/permutation_significance_test.py)                 | Code for permutation based significance testing for the specific output of CQE                                              |                       |
| [evaluation/significance_testing_for_arrary_input.py](evaluation/significance_testing_for_arrary_input.py) | Code for permutation based significance testing for normal  array input, used for the classification of unit disambiguator  |                                        |
| [evaluation/test_unit_classifier.py](evaluation/test_unit_classifier.py)                                   | Evaluation code for unit disambiguator                                                                                      |                                        |


### Data
The evaluation data can be found in `/data/formatted_test_set` and consists of 5 evaluation sets. 

| Dataset                                         | #Sentences | #Quantities | Source                                                                                                                                                      |
|-------------------------------------------------|------------|-------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------
| [NewsQuant.json](data/NewsQuant.json)           | 590        | 906         | tagged by the authors                                                                                                                                       |
| [age-model.json](data/age-model.json)           | 19         | 22          | [Microsoft.Recognizers.Text Test Cases Specs](https://github.com/microsoft/Recognizers-Text/blob/master/Specs/NumberWithUnit/English/AgeModel.json)         |
| [currency-model.json](data/currency-model.json) | 180        | 255         | [Microsoft.Recognizers.Text Test Cases Specs](https://github.com/microsoft/Recognizers-Text/blob/master/Specs/NumberWithUnit/English/CurrencyModel.json)    |
| [dimension-model.json](data/dimension-model.json) | 93         | 121         | [Microsoft.Recognizers.Text Test Cases Specs](https://github.com/microsoft/Recognizers-Text/blob/master/Specs/NumberWithUnit/English/DimensionModel.json)   |
| [temperature-model.json](data/recognizers-text/temperature-model.json) | 36         | 34          | [Microsoft.Recognizers.Text Test Cases Specs](https://github.com/microsoft/Recognizers-Text/blob/master/Specs/NumberWithUnit/English/TemperatureModel.json) |

The predictions from GPT-3 for the test datasets above can be found in  `/data/gpt_3_ouput`.

To train and test the unit classifier we generated data using ChatGPT, for the prompts used refer to the paper.
The generated sentences and their classes are under `/data/units/train` (1,827 samples) and `/data/units/test` (180 samples).


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
There are 50 rules for [DependencyMatcher](https://spacy.io/usage/rule-based-matching#dependencymatcher) defined in the [rules.py](rules.py). We use the spaCy-model [en core web sm](https://spacy.io/models/en) to create a [Doc object](https://spacy.io/api/doc) with [linguistic annotations](https://spacy.io/usage/linguistic-featuress). The key point is that the rules are not simple pattern matching based on the single words in the sentence, but on those annotations and exploit the structure of the sentence.

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