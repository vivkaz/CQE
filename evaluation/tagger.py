import abc
import json
import re
from collections import OrderedDict
from typing import List, Dict

from ccg_nlpy import remote_pipeline
from quantulum3 import parser as quantparsesr
from recognizers_number import NumberRecognizer
from recognizers_number_with_unit import NumberWithUnitRecognizer
from recognizers_text import Culture

from numparser.NumParser import NumParser
from numparser.classes import Range

"""
Generate unified outputs for all the models we are evaluating against, to make the evaluation easier. 
The output is a list of dictionary. 
Example: 
[{'change': '>', 'value': '12.0', 'unit': [people], 'normalised_unit': 'people', 'referred_concept': [[buliding]]}, {'change': '~', 'value': '2.0', 'unit': [$], 'normalised_unit': 'dollar', 'referred_concept': [[buliding]]}, {'change': '=', 'value': '2', 'unit': [meter], 'normalised_unit': 'metre', 'referred_concept': [[buliding]]}]

"""


################################################### Abstract tagger class ###################################################

class Tagger(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def tag(self):
        pass


################################################### NumParser Tagger ###################################################

@Tagger.register
class NumParserTagger():
    def __init__(self):
        self.parser = NumParser()
        self.name = "CQE-Numparser"

    def tag(self, text):
        results = self.parser.parse(text)
        new_results = []
        for res in results:
            if isinstance(res.value, Range):
                value = (res.value.lower, res.value.upper)
            else:
                value = res.value.value
            unit = res.unit.norm_unit

            change = res.change.change
            referred_concept = res.referred_concepts.get_nouns()
            flat_list = [item.text for sublist in referred_concept for item in sublist]
            if len(flat_list) < 1:
                flat_list = ['-']
            new_results.append({"value": value, "normalized_unit": unit.strip(), "change": change.strip(),
                                "referred_concepts": flat_list})
        return new_results


################################################### GPT3 Tagger ###################################################

@Tagger.register
class GPT3Tagger():
    def __init__(self, input_file):
        self.parser = NumParser()
        self.name = "GPT3"
        self.input_file = input_file
        with open(input_file, "r", encoding="utf8") as file:
            self.data = json.load(file)
        self.dictionary_unit = {
            "celsiu": "celsius",
            "kilometer": "kilometre",
            "meter": "metre",
            "square meter": "square metre",
            "kilovolt per centimeter": "kilovolt per centimetre",
            "micrometer": "micrometre",
            "kilopounds per square inch": "kilopound per square inch",
            "milliliter": "millilitre",
            "degrees celsiu": "celsius",
            "degree celsiu": "celsius",
            "us dollar": "dollar",
            "percent": "percentage",
            "miles per hour": "mile per hour",
            "metres / second": "metre per second",
            "parts per billion": "parts-per-billion",
            "deputie": "deputy",
            "miles / hour": "mile per hour",
            "inche": "inch",
            "feet": "foot",
            "decimeter": "decimetre",
            "microgrammes / gramme": "microgram per gram",
            "square kilometer": "square kilometre",
            "millimeter": "millimetre",
            "mah": "milliampere-hour",
            "ppb": "parts-per-billion",
            "pound-feet": "pound-foot",
            "g": "gram",
            "min": "minute",
            "gigatonne": "gigaton",
            "time": "count",
            "ghz": "gigahertz",
            "hz": "hertz",
            "amperes per cm2": "ampere per square centimetre",
            "degrees fahrenheit": "fahrenheit",
            "nanometer": "nanometre",
            "yuan": "chinese yuan",
            "us$": "dollar",
            "kilometers / hour": "kilometre per hour",
            "rad": "radian",
            "celciu": "celsius",
            "centigrate": "celsius",
            "deg": "degree",
            "usd": "dollar",
            "btc": "bitcoin",
            "€": "euro",
            "元": "chinese yuan",
            "franc": "swiss franc",
            "cent-a-share": "cent per share",
            "sq m":"square metre",
            "degrees Fahrenheit":"Fahrenheit",
            "parts per million":"part per million",
            "barrel-a-day":"barrel per day",
            "points":"point",
            "miles per gallon":"mile per gallon",
            "contracts":"contract",
            "bathrooms":"bathroom",
            "bedrooms":"bedroom",
            "megabits per second":"megabit per second",
            "liter":"litre",
            "barrels / day":"barrel per day",
            "lari":"georgian lari",
            "ரூ":"sri lankan rupee",
            "lira":"turkish lira",
            "yen":"japanese yen",
            "rand":"south african rand",
            "kuna":"croatian kuna",
            "touchdown":"touch down",
            "cents per share":"cent per share",
            "wins above replacement":"win above replacement",
            "rebounds per game":"rebound per game",
            "points per game":"point per game",
            "us pint":"pint",
            "us quart":"quart",
            "us gallon":"gallon",
            "us fluid ounce":"ounce",
            "ksi":"kilopound per square inch",
            "kilovolts per centimeter":"kilovolt per centimetre",
            "commercial fishermen":"commercial fisherman",
            "tonne":"metric ton",
            "kilowatt-hour":"kilowatt hour",
            "newton meter":"newton metre",
            "ringgit":"malaysian ringgit",
            "assists per game":"assist per game",
            "gigawatt-hour":"gigawatt hour",
            "milliampere hour":"milliampere-hour"


        }

    def set_dataset_text_recongizer(self):
        self.dictionary_unit["year"] = "year of age"
        self.dictionary_unit["day"] = "day of age"
        self.dictionary_unit["week"] = "week of age"
        self.dictionary_unit["month"] = "month of age"

    def clean_concepts(self,concepts):
        new_list=[]
        for c in concepts:
            if c not in ["of", "the","a","He","total"]:
                new_list.append(c)
        return new_list
    def decode_output(self, list_values):
        all_values = []
        for list_val in list_values:
            if len(list_val) > 0:
                value = list_val[1]
                change = list_val[0]
                unit = list_val[3].lower().strip()
                if unit.endswith("s"):
                    unit = unit[:-1]
                if unit in self.dictionary_unit.keys():
                    unit = self.dictionary_unit[unit]
                concepts = re.split(r'[,\s]+', list_val[4].strip())
                concepts=self.clean_concepts(concepts)
                changes = change.strip()
                if changes == "+":
                    changes = "up"
                all_values.append({"value": value.strip(), "normalized_unit": unit, "change": changes,
                                   "referred_concepts": concepts})
        return all_values

    def tag(self, text):
        for gpt_line in self.data:
            if text == gpt_line["text"]:
                decoded_ouput = self.decode_output(gpt_line["quantities"])
                return decoded_ouput


################################################### Quantulum Tagger ###################################################

@Tagger.register
class QuantulumTagger():
    # tries to find SI units when there is none
    def __init__(self):
        self.parser = quantparsesr
        self.name = "quantulum3"
        self.dictionary_unit = {
            "dimensionless": "-",
            "unk": "-",
            "per cent": "percentage",
            "dollar dollar": "dollar",
            "united states dollar": "dollar",
            "tonne": "metric ton",
            "degree angle": "degree",
            "degree celsius": "celsius",
            "degree fahrenheit": "fahrenheit",
            "percentage drop":"percentage",
            "second of arc":"second",
            "minute of arc":"minute"
            ,"yard yard":"yard",
            "cubic centimetre":"millilitre",
            "degree angle degree fahrenheit":"fahrenheit",
            "pound sterling year":"pound sterling per year",
            "per year per newton per day":"year"

        }

    def set_dataset_text_recongizer(self):
        self.dictionary_unit["year"] = "year of age"
        self.dictionary_unit["day"] = "day of age"
        self.dictionary_unit["week"] = "week of age"
        self.dictionary_unit["month"] = "month of age"

    def tag(self, text: str):

        results = self.parser.parse(text)
        new_results = []
        for res in results:
            # some terminology needs to be adjusted here, the list might be longer @TODO
            value = res.value
            unit = res.unit.name
            if unit.lower() in self.dictionary_unit.keys():
                unit = self.dictionary_unit[unit.lower()]
            new_results.append({"value": value, "normalized_unit": unit.strip()})
        return new_results


################################################### Text recognizer Tagger ###################################################

@Tagger.register
class TextRecognizerTagger():
    def __init__(self):
        self.parser_name = None
        recognizer = NumberWithUnitRecognizer(Culture.English)
        age_model = recognizer.get_age_model()
        currency_model = recognizer.get_currency_model()
        dimension_model = recognizer.get_dimension_model()
        temperature_model = recognizer.get_temperature_model()
        percentage_recognizer = NumberRecognizer(Culture.English)
        percentage_model = percentage_recognizer.get_percentage_model()
        only_value = percentage_recognizer.get_number_model()
        self.parser = OrderedDict({"currency": currency_model, "dimension": dimension_model,
                                   "temperature": temperature_model, "precentage": percentage_model, "age": age_model,
                                   "value": only_value})
        self.name = "text-recognizer"

        self.dictionary_unit = {"liter": "litre",
                                "kilometer": "kilometre",
                                "milliliter": "millilitre",
                                "millimeter": "millimetre",
                                "centimeter": "centimetre",
                                "square meter": "square metre",
                                "meter": "metre",
                                "c": "celsius",
                                "british pound": "pound sterling",
                                "kilometer per hour": "kilometre per hour",
                                "decimeter": "decimetre",
                                "picometer": "picometre",
                                "micrometer": "micrometre",
                                "square kilometer": "square kilometre",
                                "f": "fahrenheit",
                                "franc": "swiss franc",
                                "united states dollar": "dollar"}

    def set_dataset_text_recongizer(self):
        self.dictionary_unit["year"] = "year of age"
        self.dictionary_unit["day"] = "day of age"
        self.dictionary_unit["week"] = "week of age"
        self.dictionary_unit["month"] = "month of age"

    def tag(self, text: str):
        results = []
        parsers_to_consdier = OrderedDict({})
        if self.parser_name is not None:
            parsers_to_consdier[self.parser_name] = self.parser[self.parser_name]
            for key, value in self.parser:
                if key not in parsers_to_consdier:
                    parsers_to_consdier[key] = value
        else:
            parsers_to_consdier = self.parser
        for name, p in parsers_to_consdier.items():
            if name == "precentage":
                percent_results = p.parse(text)
                for percent_result in percent_results:
                    percent_result.resolution['value'] = percent_result.resolution['value'].rstrip("%").replace(',', '')
                    percent_result.resolution['unit'] = "percentage"
                    results.append({"value": percent_result.resolution['value'],
                                    "normalized_unit": percent_result.resolution['unit']})
            elif name == "value":
                value_results = p.parse(text)
                before_values = [v["value"] for v in results]
                for v in value_results:  # only add a simple value if none of the other models were able to detect anything
                    if v.resolution['value'] not in before_values:
                        results.append({"value": v.resolution['value'].replace(',', ''), "normalized_unit": "-"})

            else:
                res = p.parse(text)

                for r in res:
                    # we need to adjust the terminology of the output to the ground truth, the text recognizer has the american spelling and we have the british one
                    if r.resolution['value'] is not None:
                        if r.resolution['unit'].strip().lower() in self.dictionary_unit.keys():
                            unit_norm = self.dictionary_unit[r.resolution['unit'].strip().lower()]

                        else:
                            unit_norm = r.resolution['unit'].strip().lower()
                        results.append({"value": r.resolution['value'], "normalized_unit": unit_norm})

        return results


################################################### Quantifier Tagger ###################################################

@Tagger.register
class CCG_NILPTagger():
    def __init__(self):
        self.parser = remote_pipeline.RemotePipeline()
        self.name = "ccg_nlp"
        self.dictionary_unit = {
            "US$": "dollar",
            'percent': "percentage",
            "pc": "percentage",
            "degf": "fahrenheit",
            'c': "celsius",
            "celsiu": "celsius",
            "km": "kilometre",
            "ksi": "kilopound per square inch",
            "m2": "square metre",
            "mpa": "megapascal",
            "ml": "millilitre",
            "p": "point",
            "yuan": "chinese yuan",
            "meter": "metre",
            "miles per hour": "mile per hour",
            "mph": "mile per hour",
            "month-old": "month of age",
            "year-old": "year of age",
            "kph": "kilometre per hour",
            "kg": "kilogram",
            "yen": "japanese yen",
            "barrels a day": "barrel per day",
            "kilometer": "kilometre",
            "sq": "square metre",
            "smaller tenancie": "small tenancy",
            "cm": "centimetre",
            "inche": "inch",
            "gb": "gigabyte",
            "kwh": "kilowatt hour",
            "kw": "kilowatt",
            "hp": "horsepower",
            "mbp": "megabit per second",
            "bhp": "brake horsepower",
            "viruse": "virus",
            "pound-feet": "pound-foot",
            "rpm": "revolutions per minute",
            "microgrammes per gramme": "microgram per gram",
            "lb": "pound-mass",
            "feet": "foot",
            "picometer": "picometre",
            "mm": "millimetre",
            "miles an hour": "mile per hour",
            "square meter": "square metre",
            "day old": "day of age",
            "week of age": "weeks old",
            "months old": "month of age",
            "years old": "year of age",
            "kilometers per hour": "kilometer per hour",
            "° celsiu": "celsius",
            "g": "gram",
            "min": "minute",
            "gigatonne": "gigaton",
            "time": "count",
            "ghz": "gigahertz",
            "hz": "hertz",
            "largest citie": "largest city",
            "year - old": "year of age",
            "dm": "decimetre",
            "m": "metre",
            "kilometer per hour": "kilometre per hour",
            "kip": "lao kip",
            "colone": "costa rican colón",
            "losse": "loss",
            "usd": "dollar",
            "btc": "bitcoin",
            "lb-ft":"pound-force",
            "nm":"newton metre",
            "year-old site":"year of age",
            "ev":"exavolt",
            "commercial fishermen":"commercial fisherman",
            "μm":"micrometre",
            "children":"child",
            "cubic feet":"cubic foot",
            "companie":"company",
            "gwh":"gigawatt hour",
            "ppg":"point per game",
            "rpg":"rebound per game",
            "war":"win above replacement",
            "franc":"swiss franc",
            "cent-a-share":"cent per share",
            "model 3 sedan":"model 3 sedans",
            "launche":"launch",
            "previous year":"year",
            "μg":"microgram",
            "rad":"radian",
            "european countrie":"european country",
            "different bosse":"different boss"


        }

    def set_dataset_text_recongizer(self):
        self.dictionary_unit["year"] = "year of age"
        self.dictionary_unit["day"] = "day of age"
        self.dictionary_unit["week"] = "week of age"
        self.dictionary_unit["month"] = "month of age"

    def tag(self, text: str):
        results = []
        try:
            ccg_out = self.parser.doc(text).get_quantities
        except:
            return []
        if ccg_out is None:
            return []
        if ccg_out.cons_list is None:
            return []
        # output is a text that is seperated with space, it has the change, value and unit in order
        ccg_list = [surface["label"] for surface in ccg_out]
        for ccg in ccg_list:
            ccg = ccg.replace("]", "").replace("[", "")
            ccg = ' '.join(ccg.split())
            ccg = ccg.replace("per cent", "percent")
            splitted_values = ccg.split(" ")
            if not splitted_values[1].startswith("Date") and "E-" not in splitted_values[
                1]:  # we disregard the dates since there is no groundtruth for it

                if len(splitted_values) > 2:
                    unit_norm = ' '.join(splitted_values[2:]).strip()
                    if unit_norm.endswith(" /"):
                        unit_norm = unit_norm[:-2]
                    if unit_norm.startswith("-"):
                        unit_norm = unit_norm[1:]
                    if unit_norm.endswith("s"):
                        unit_norm = unit_norm[:-1]
                    if unit_norm in self.dictionary_unit.keys():
                        unit_norm = self.dictionary_unit[unit_norm]
                else:
                    unit_norm = '-'
                changes = splitted_values[0].strip()
                if changes == "+":
                    changes = "up"
                results.append({"value": splitted_values[1].replace(',', '').strip(), "normalized_unit": unit_norm,
                                "change": changes})
        return results


################################################### Tag an entire file with a tagger ###################################################

def tag_entire_file(file_lines: List[Dict], parser: Tagger):
    """
    Goes over an entire file from the test set and tags it based on the defined parser
    """
    result = []
    for line in file_lines:
        output = parser.tag(line["text"])
        result.append({"text": line["text"], "gt": line["quantities"], "tags": output})
    return result


def tag_entire_file_as_dictionary(file_lines: List[Dict], parser: Tagger):
    """
    Goes over an entire file from the test set and tags it based on the defined parser
    """
    result = {}
    for line in file_lines:
        output = parser.tag(line["text"])
        result[line["text"]] = {"gt": line["quantities"], "tags": output}
    return result


def tag_entire_file_with_index(file_lines: List[Dict], parser: Tagger):
    """
    Goes over an entire file from the test set and tags it based on the defined parser
    """
    result = []
    for index, line in enumerate(file_lines):
        output = parser.tag(line["text"])
        result.append({"text": line["text"], "gt": line["quantities"], "tags": output})
    return result


################################################### gpt3 decoder ###################################################

parser = QuantulumTagger()







# tags=parser.tag( "The mass of the sun is approximately 1.99×10−30 kilograms.")
# for tag in tags:
#     print(tag)

