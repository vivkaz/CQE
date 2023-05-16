import itertools
import logging
import locale
import re
from typing import List
import json
import os
from ordered_set import OrderedSet

from spacy.lang.lex_attrs import is_digit, like_num
from spacy.tokens import Token

from .number_lookup import maps, prefixes
from .unit_classifier.unit_disambiguator import unit_disambiguator

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

from pathlib import Path
def get_project_root() -> Path:
    return Path(__file__).parent

root_path=str(get_project_root())
if not os.path.exists(os.path.join(root_path+"/unit_models")):
    import zipfile
    with zipfile.ZipFile(os.path.join(root_path+"/unit_models.zip"), 'r') as zip_ref:
        zip_ref.extractall(os.path.join(root_path))
disambiguator = unit_disambiguator()

AMBIGUOUS_FORMS = ["c", "¥", "kn", "p", "R", "b", "'", "′", '"', '″', "C", "F", "kt", "B", "P", "dram", "pound", "pounds", "a"]
AMBIGUOUS_UNITS = ["celsius", "pound sterling", "japanese yen", "pixel", "poise", "barn", "bel", "dram", "minute", "south african rand",
                   "bit", "cent", "knot", "penny", "chinese yuan", "croatian kuna", "inch", "kilobyte", "acre", "second", "armenian dram",
                   "year", "kiloton", "kibibyte", "coulomb", "farad", "roentgen", "byte", "fahrenheit", "foot", "pound-mass", "point"]

def get_project_root() -> Path:
    return Path(__file__).parent

class NormalizeException(Exception):
    pass


def is_num(s):
    try:
        locale.atoi(s)
    except ValueError:
        try:
            locale.atof(s)
        except ValueError:
            return False
    return True


def preprocess_number(text: List):
    remove_words = ["a"]
    for word in remove_words:
        if word in text:
            text.remove(word)
    res = []
    # a fraction can't be a scale if its at the first position. e.g. halve a million
    if len(text) > 1:
        res.append(str(maps["fractions"].get(text[0], text[0])))
        res.extend(text[1:])

    elif text[0] in maps["fractions"].keys(): # e.g. half turn (the fractions from number_lookup.py)
        res.append("".join(str(maps["fractions"][text[0]])))

    elif re.findall(r"([0-9,.]*[0-9]+)×?([e|E])([+|−]?)(\d*)\b", text[0]): # e.g. 6.022E20, 6.022E+20, 6.022E−20 etc.
        res.extend([re.sub(r"([0-9,.]*[0-9]+)×?([e|E])([+|−]?)(\d*)\b", r"\1×e\3\4", text[0])])

    elif re.findall(r"([0-9,.]*[0-9]+)×(10)([+|−]+)(\d*)\b", text[0]): # e.g. 1.06×10−10
        res.extend([re.sub(r"([0-9,.]*[0-9]+)×(10)([+|−]+)(\d*)\b", r"\1×e\3\4", text[0])])
    else:
        res.extend(text)
    return res


def normalize_number(text: List):
    text = preprocess_number(text)

    if "×" in text[0]: # scientific notation e.g. 5.067×10−4
        number_1 = locale.atof(text[0][:text[0].index("×")])
        if text[0][text[0].index("×")+1:] in maps["scientific_notation"]:
            number_2 = maps["scientific_notation"][text[0][text[0].index("×")+1:]]
        elif "e−" in text[0][text[0].index("×")+1:]: # 6.022×e-20
            number_2 = 10**locale.atof("-"+text[0][text[0].index("×")+3:])
        elif "e" in text[0][text[0].index("×")+1:]:  # 6.022×e+20
            number_2 = 10**locale.atof(text[0][text[0].index("×")+2:])
        elif "^−" in text[0][text[0].index("×")+1:]:  # 1.06×10^−10
            number_2 = 10**locale.atof("-"+text[0][text[0].index("−")+1:])
        elif "^" in text[0]: # 1.06×10^10
            number_2 = 10**locale.atof(text[0][text[0].index("^")+1:])
        else: # 1.06×10
            number_2 = locale.atof(text[0][text[0].index("×")+1:])
        return number_1*number_2

    if len(text) == 1 and is_num(text[0]):
        return locale.atof(text[0])

    if len(text) == 1 and not is_digit(text[0]):
        if text[0] in maps["string_num_map"]:
            # return maps["string_num_map"][text[0]]
            return locale.atof(str(maps["string_num_map"][text[0]]))
        if any(char in ('⁄', '/') for char in text[0]): # 1/16 of a US pint, 1⁄32 of a US quart, and 1⁄128 of a US gallon
            if '⁄' in text[0]:
                return int(text[0][:text[0].index("⁄")])/int(text[0][text[0].index("⁄")+1:])
            return int(text[0][:text[0].index("/")])/int(text[0][text[0].index("/")+1:])
        return locale.atof(str(maps["scales"][text[0]]))

    if len(text) == 2 and like_num(text[0]) and not like_num(text[1]):
        if text[1] == "m":
            return locale.atof(text[0]) * maps["scales"]["million"]
        return locale.atof(text[0]) * maps["scales"][text[1]]

    if len(text) == 2 and not is_digit(text[0]) and like_num(text[1]): # five, 20
        if text[0] in maps["string_num_map"]:
            return [locale.atof(str(maps["string_num_map"][text[0]])), locale.atof(text[1])]

    if len(text) == 2 and like_num(text[0]) and like_num(text[1]): # e.g. [66.5, 107] from '66.5 x 107 mm'
        return [locale.atof(text[0]), locale.atof(text[1])]

    current = 0.0
    result = 0.0
    for i, word in enumerate(text):
        if is_digit(word) and i != 0:
            raise NormalizeException("Mixing words and numbers (e.g. four 100 thousand) is not supported")
        if word not in (maps["scales"].keys() | maps["string_num_map"].keys() | maps["fractions"].keys()) and not is_digit(word.replace(".","")):
            raise NormalizeException(f"Number lookup for {word} failed.")
        if word in maps["scales"].keys() | maps["fractions"].keys():
            if word in maps["scales"]:
                scale = maps["scales"][word]
            else:
                scale = maps["fractions"][word]
            incr = 0
        else:
            scale = 1
            if word in maps["string_num_map"]:
                incr = maps["string_num_map"][word]
            elif is_digit(word.replace(".","")):
                incr = locale.atof(word)
        current = incr + current * scale
        if scale > 100:
            result += current
            current = 0
    return result + current


def normalize_number_token(tokens: List[Token]):
    lemmatized_text = [token.lemma_ for token in tokens]
    if any(token._.one_of_number and token in maps["string_num_map"] for token in tokens):
        lemmatized_text = [tokens[0].lemma_]
    try:
        normalized = normalize_number(lemmatized_text)
        return normalized
    except:
        logging.warning('\033[93m' + f"Cant normalize {lemmatized_text}" + '\033[0m')
        return None


def normalize_bound_token(tokens: List[Token]):
    # if not specified assume equality
    bound = "="
    if len(tokens) > 1 and " ".join(token.lemma_.lower() for token in tokens) in maps["bounds"]:
        bound = maps["bounds"][" ".join(token.lemma_ for token in tokens)] # e.g. "up to"
    else:
        for token in tokens:
            if token.lemma_.lower() in maps["bounds"]:
                bound = maps["bounds"][token.lemma_.lower()]
    return bound

def match_text_to_unit(unit_text, units_dict):
    if any(unit_text==key for key in units_dict):
        return unit_text, [unit_text]
    
    for key in units_dict.keys():
        if units_dict[key].get("surfaces") is not None and unit_text in units_dict[key].get("surfaces"):
            return key, [key]
        if units_dict[key].get("symbols") is not None and unit_text in units_dict[key].get("symbols"):
            return key, [key]
        if units_dict[key].get("prefixes") is not None:
            for symbol in units_dict[key].get("symbols"):
                if unit_text in [(prefix+symbol) for prefix in units_dict[key].get("prefixes")]:
                    index_prefix = [(unit_text in (prefix+symbol) ) for prefix in units_dict[key].get("prefixes")].index(True)
                    return prefixes[units_dict[key].get("prefixes")[index_prefix]]+key, [key]

    return unit_text, []


def check_ambiguous_unit(text, norm_unit, unit):
    """"use BERT classifier to diambiguate unit if it is ambiguous"""
    if unit=="pounds": # for pounds and pound, use the surface form "pound" in both cases to do disambiguation
        unit="pound"
    if norm_unit in AMBIGUOUS_UNITS and unit in AMBIGUOUS_FORMS:
        return disambiguator.disambiguate(text, unit)
    return norm_unit


def normalize_unit_token(tokens: List[Token], text, recursive=False): # return normalized unit + index of the token + whether in the unit.json (i.e scientific) + list of keys
    if not tokens or (len(tokens)==1 and tokens[0].text=="of"): # unit-less quantity
        return "-", None, False, [], False
    
    if len(set(token.text for token in tokens)) != 1 and len(tokens) > 1:
        tokens = [token for i,token in enumerate(tokens) if token.text != tokens[(i+1)%len(tokens)].text]

    if any(token._.one_of_noun for token in tokens):
        i = [token._.one_of_noun for token in tokens].index(True)
        def accept_token(token):
            return not token.is_punct and not (token.text in ["this","that","which","what","who","whose","whom"]) or token.text in ["/", "-"]
        return " ".join([t.lemma_ for t in list(itertools.takewhile(accept_token, list(tokens[i].subtree))) if not t.is_stop]), -1, False, [], False

    norm_units = ""
    index = None
    in_unit_list = True
    path = get_project_root()
    file_name = os.path.join(path, "unit.json")
    with open(file_name, "r", encoding="utf8") as f:
        units_dict = json.load(f)

    if len(tokens) > 1: # first look if the tokens as whole are present in the units_dict
        index = -1 # whole list is unit
        token_text = " ".join(list(OrderedSet(token.lemma_.upper() if token.text.isupper() else token.lemma_ for token in tokens))) # if not token.is_punct)))
    else:
        index = 0 # single token
        token_text = tokens[0].text # use text because of e.g. 'A/cm2 (a/cm2)

    if "/" in token_text: # e.g. 'kV/cm'
        connector = token_text.index("/")
        first_unit, first_key = match_text_to_unit(token_text[:connector], units_dict)
        second_unit, second_key = match_text_to_unit(token_text[connector+1:], units_dict)

        return first_unit+" per "+second_unit, index, in_unit_list, [first_key,second_key], False

    if any(token_text==key for key in units_dict):
        return token_text, index, in_unit_list, [token_text], False # lemmatized unit is already normalized, e.g. 'mile' -> 'mile'

    for key in units_dict:
        if units_dict[key].get("surfaces") is not None and token_text in units_dict[key].get("surfaces"):
            return check_ambiguous_unit(text, key, token_text), index, in_unit_list, [key], False # e.g. 'light year' -> 'light-year'
        if units_dict[key].get("symbols") is not None and token_text in units_dict[key].get("symbols"):
            return check_ambiguous_unit(text, key, token_text), index, in_unit_list, [key], False # e.g. '%' -> 'percentage'

    if any(token.text in ["per"] for token in tokens): # '$ per square foot' -> 'dollar per square foot'
        token_text = [token.text for token in tokens]
        i = token_text.index("per")
        token_text[:i] = [normalize_unit_token(tokens[:i], text)[0]] # C$ -> canadian dollar
        first_key = normalize_unit_token(tokens[:i], text)[-2]
        j = token_text.index("per")
        _, index_slice, _, _, slice_unit = normalize_unit_token(tokens[i+1:], text, True)
        token_text[j+1:] = [normalize_unit_token(tokens[i+1:], text, True)[0]]
        second_key = normalize_unit_token(tokens[i+1:], text, True)[-2]
        
        if slice_unit:
            return " ".join(token_text), index_slice+j+1, in_unit_list, [first_key,second_key], True
        if index_slice is not None:
            if index_slice+j+1==len(tokens):
                return " ".join(token_text), index, in_unit_list, [first_key,second_key], False
            return " ".join(token_text), -1, in_unit_list, [first_key,second_key], False
        return " ".join(token_text[:j]), -1, in_unit_list, [first_key,second_key], False

    if any(token.text in ["a", "an"] for token in tokens): # 'barrels a day', '£ an hour'
        token_text = [token.text for token in tokens]
        i = token_text.index("a") if "a" in token_text else token_text.index("an")
        token_text[:i] = [normalize_unit_token(tokens[:i], text)[0]] # C$ -> canadian dollar
        first_key = normalize_unit_token(tokens[:i], text)[-2]
        j = token_text.index("a") if "a" in token_text else token_text.index("an")
        token_text[j] = "per"
        _, index_slice, _, _, slice_unit = normalize_unit_token(tokens[i+1:], text, True) # [$, a, month, mortgage]
        token_text[j+1:] = [normalize_unit_token(tokens[i+1:], text, True)[0]]
        second_key = normalize_unit_token(tokens[i+1:], text, True)[-2]

        if slice_unit:
            return " ".join(token_text), index_slice+j+1, in_unit_list, [first_key,second_key], True
        if index_slice is not None:
            if index_slice+j+1==len(tokens):
                return " ".join(token_text), index, in_unit_list, [first_key,second_key], False
            return " ".join(token_text), -1, in_unit_list, [first_key,second_key], False
        return " ".join(token_text[:j]), -1, in_unit_list, [first_key,second_key], False

    if len(tokens) > 2:
        index = 1
        token_text = " ".join([token.lemma_ for token in tokens[:2] if token.text not in ["-"]]) # e.g. [US, $, contract] -> US $

        if token_text in units_dict.keys():
            return token_text, index, in_unit_list, [token_text], True

        for key in units_dict.keys():
            if units_dict[key].get("surfaces") is not None and token_text in units_dict[key].get("surfaces"):
                return check_ambiguous_unit(text, key, token_text), index, in_unit_list, [key], True
            if units_dict[key].get("symbols") is not None and token_text in units_dict[key].get("symbols"):
                return check_ambiguous_unit(text, key, token_text), index, in_unit_list, [key], True

    for key in units_dict.keys():
        if units_dict[key].get("symbols") is not None:
            if any(token.text in units_dict[key].get("symbols") for token in tokens):
                index = [token.text in units_dict[key].get("symbols") for token in tokens].index(True)
                return check_ambiguous_unit(text, key, tokens[index].text), index, in_unit_list, [key], False # e.g. [G, data, month] -> generation wireless
            if units_dict[key].get("prefixes") is not None:
                for i, token in enumerate(tokens):
                    for symbol in units_dict[key].get("symbols")+units_dict[key].get("surfaces"):
                        if token.text in [(prefix+symbol) for prefix in units_dict[key].get("prefixes")]:
                            index_prefix = [(token.text in (prefix+symbol)) for prefix in units_dict[key].get("prefixes")].index(True)
                            return prefixes[units_dict[key].get("prefixes")[index_prefix]]+key, i, in_unit_list, [key], False # e.g. GHz -> gigahertz
                        if token.text in [(prefix+symbol) for prefix in prefixes.values()]:
                            index_prefix = [(token.text in (prefix+symbol)) for prefix in prefixes.values()].index(True)
                            return list(prefixes.values())[index_prefix]+key, i, in_unit_list, [key], False # e.g. gigawatts -> gigawatt
                        if token.lemma_ in [(prefix+symbol) for prefix in prefixes.values()]:
                            index_prefix = [(token.lemma_ in (prefix+symbol)) for prefix in prefixes.values()].index(True)
                            return list(prefixes.values())[index_prefix]+key, i, in_unit_list, [key], False # e.g. gigawatt -> gigawatt

    # cases like '40 yard line' where both 'yard' and 'line' are in unit.json
    index_loop = len(tokens)
    key_token = None
    for key in units_dict.keys():
        if units_dict[key].get("surfaces") is not None and any(token.lemma_ in units_dict[key].get("surfaces") for token in tokens):
            index_new = [token.lemma_ in units_dict[key].get("surfaces") for token in tokens].index(True)
            if index_new < index_loop:
                index_loop = index_new
                key_token = key
    if key_token:
        return key_token, index_loop, in_unit_list, [key_token], bool(len(tokens)!=index_loop+1)

    # check if some of the tokens is present as a key in the unit.json file
    norm_units = [token.lemma_.upper() if token.text.isupper() else token.lemma_ for token in tokens if token.lemma_ in units_dict.keys()]

    if not norm_units:
        in_unit_list = False

    if not norm_units and any(token.pos_ in ["NOUN", "PROPN"] or token._.unit for token in tokens): # skip e.g. "cheaper", extract parts per million
        if not recursive:
            norm_units = [token.lemma_.upper() if token.text.isupper() else token.lemma_ for token in tokens if token.pos_ in ["NOUN", "PROPN", "ADJ"]]
            if norm_units: # e.g. 'students' -> 'student'
                if len(tokens) == 1:
                    index = 0
                else:
                    index = -1 # whole list is unit
        else: # e.g. $ 70 - a - share price
            norm_units = [token.lemma_.upper() if token.text.isupper() else token.lemma_ for token in tokens if token.pos_ in ["NOUN", "PROPN"] or token._.unit]
            if norm_units:
                norm_units = [norm_units[0]]
                index = 0
                return " ".join(norm_units) if norm_units else "-", index, in_unit_list, [], False

    return " ".join(norm_units) if norm_units else "-", index, in_unit_list, [], False
