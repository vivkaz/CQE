import logging
import locale
from typing import List
import json
import os

from spacy.lang.lex_attrs import is_digit, like_num
from spacy.tokens import Token

from rule_set.number_lookup import maps, prefixes

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

from pathlib import Path

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
    elif text[0] in maps["fractions"].keys(): # e.g. half turn (fractions from number_lookup.py)
        res.append("".join(str(maps["fractions"][text[0]])))
    else:
        res.extend(text)
    return res


def normalize_number(text: List):
    text = preprocess_number(text)
    if len(text) == 1 and is_num(text[0]):
        return locale.atof(text[0])
    if len(text) == 1 and not is_digit(text[0]):
        if text[0] in maps["string_num_map"]:
            #print(locale.atof(str(maps["string_num_map"][text[0]])))
            return maps["string_num_map"][text[0]]
        return maps["scales"][text[0]]
    if len(text) == 2 and like_num(text[0]) and not like_num(text[1]):
        if text[1] == "m":
            return locale.atof(text[0]) * maps["scales"]["million"]
        return locale.atof(text[0]) * maps["scales"][text[1]]
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
        return unit_text
    
    for key in units_dict.keys():
        if units_dict[key].get("surfaces") is not None and unit_text in units_dict[key].get("surfaces"):
            return key
        if units_dict[key].get("symbols") is not None and unit_text in units_dict[key].get("symbols"):
            return key
        if units_dict[key].get("prefixes") is not None:
            for symbol in units_dict[key].get("symbols"):
                if unit_text in [(prefix+symbol) for prefix in units_dict[key].get("prefixes")]:
                    index_prefix = [(unit_text in (prefix+symbol) ) for prefix in units_dict[key].get("prefixes")].index(True)
                    return prefixes[units_dict[key].get("prefixes")[index_prefix]]+key

    return unit_text


def normalize_unit_token(tokens: List[Token]): # return normalized unit + index of the token + whether in the unit.json
    if not tokens or len(tokens)==1 and tokens[0].text=="of": # unit-less quantity
        return "-", None, False

    if any(token._.one_of_noun for token in tokens):
        i = [token._.one_of_noun for token in tokens].index(True)
        return " ".join([t.lemma_ for t in list(tokens[i].subtree) if not t.is_stop]), -1, False

    norm_units = ""
    index = None
    in_unit_list = True
    path = get_project_root()
    file_name = os.path.join(path,"data/unit.json")
    with open(file_name, "r", encoding="utf8") as f:
        units_dict = json.load(f)

    if len(tokens) > 1: # first look if the tokens as whole are present in the units_dict
        index = -1 # whole list is unit
        token_text = " ".join([token.lemma_.upper() if token.text.isupper() else token.lemma_ for token in tokens if not token.is_punct])
    else:
        index = 0 # single token
        token_text = tokens[0].text # use text because of e.g. 'A/cm2 (a/cm2)

    if "/" in token_text: # e.g. 'kV/cm'
        connector = token_text.index("/")
        first_unit = match_text_to_unit(token_text[:connector], units_dict)
        second_unit = match_text_to_unit(token_text[connector+1:], units_dict)

        return first_unit+" per "+second_unit, index, in_unit_list

    if any(token_text==key for key in units_dict):
        return token_text, index, in_unit_list # lemmatized unit is already normalized, e.g. 'mile' -> 'mile'

    for key in units_dict:
        if units_dict[key].get("surfaces") is not None and token_text in units_dict[key].get("surfaces"):
            return key, index, in_unit_list # e.g. 'light year' -> 'light-year'
        if units_dict[key].get("symbols") is not None and token_text in units_dict[key].get("symbols"):
            return key, index, in_unit_list # e.g. '%' -> 'percentage'

    if any(token.text in ["per"] for token in tokens): # '$ per square foot' -> 'dollar per square foot'
        token_text = [token.text for token in tokens]
        i = token_text.index("per")
        token_text[:i] = [normalize_unit_token(tokens[:i])[0]] # C$ -> canadian dollar
        j = token_text.index("per")
        token_text[j+1:] = [normalize_unit_token(tokens[i+1:])[0]]
        return " ".join(token_text), index, in_unit_list

    if any(token.text in ["a"] for token in tokens): # 'barrels a day' -> 'barrels per day'
        token_text = [token.text for token in tokens]
        i = token_text.index("a")
        token_text[:i] = [normalize_unit_token(tokens[:i])[0]] # C$ -> canadian dollar
        j = token_text.index("a")
        token_text[j] = "per"
        token_text[j+1:] = [normalize_unit_token(tokens[i+1:])[0]]
        return " ".join(token_text), index, in_unit_list

    if len(tokens) > 2:
        index = 1
        token_text = " ".join([token.lemma_ for token in tokens[:2] if token.text not in ["-"]]) # e.g. [US, $, contract] -> US $

        if token_text in units_dict.keys():
            return token_text, index, in_unit_list

        for key in units_dict.keys():
            if units_dict[key].get("surfaces") is not None and token_text in units_dict[key].get("surfaces"):
                return key, index, in_unit_list
            if units_dict[key].get("symbols") is not None and token_text in units_dict[key].get("symbols"):
                return key, index, in_unit_list

    for key in units_dict.keys():
        if units_dict[key].get("symbols") is not None:
            if any(token.text in units_dict[key].get("symbols") for token in tokens):
                index = [token.text in units_dict[key].get("symbols") for token in tokens].index(True)
                return key, index, in_unit_list # e.g. [G, data, month] -> generation wireless
            if units_dict[key].get("prefixes") is not None:
                for i, token in enumerate(tokens):
                    for symbol in units_dict[key].get("symbols")+units_dict[key].get("surfaces"):
                        if token.text in [(prefix+symbol) for prefix in units_dict[key].get("prefixes")]:
                            index_prefix = [(token.text in (prefix+symbol)) for prefix in units_dict[key].get("prefixes")].index(True)
                            return prefixes[units_dict[key].get("prefixes")[index_prefix]]+key, i, in_unit_list # e.g. GHz -> gigahertz
                        if token.text in [(prefix+symbol) for prefix in prefixes.values()]:
                            index_prefix = [(token.text in (prefix+symbol)) for prefix in prefixes.values()].index(True)
                            return list(prefixes.values())[index_prefix]+key, i, in_unit_list # e.g. gigawatts -> gigawatt
                        if token.lemma_ in [(prefix+symbol) for prefix in prefixes.values()]:
                            index_prefix = [(token.lemma_ in (prefix+symbol)) for prefix in prefixes.values()].index(True)
                            return list(prefixes.values())[index_prefix]+key, i, in_unit_list # e.g. gigawatt -> gigawatt


    for key in units_dict.keys():
        if units_dict[key].get("surfaces") is not None and any(token.lemma_ in units_dict[key].get("surfaces") for token in tokens):
            index = [token.lemma_ in units_dict[key].get("surfaces") for token in tokens].index(True)
            return key, index, in_unit_list
    
    # check if some of the tokens is present as a key in the unit.json file
    norm_units = [token.lemma_.upper() if token.text.isupper() else token.lemma_ for token in tokens if token.lemma_ in units_dict.keys()]

    in_unit_list = False

    if len(tokens) == 1: # token is symbol not present in the units_dict
        norm_units = [token.text for token in tokens if token.pos_ == "SYM" and token.text not in [".", ",", "-", ";"]]

    if not norm_units: # e.g. 'students' -> 'student'
        norm_units = [token.lemma_.upper() if token.text.isupper() else token.lemma_ for token in tokens if token.pos_ in ["NOUN", "ADJ"]]#[:1]
        if norm_units: # unit as compound e.g. one of four separate mission concepts
            index = -1 # whole list is unit

    if not norm_units: # proper noun
        norm_units = [token.text for token in tokens if token.pos_ == "PROPN"][:1]

    if not norm_units: # default case
        norm_units = [token.lemma_.upper() if token.text.isupper() else token.lemma_ for token in tokens][:1] # first lemmatized token

    if len(norm_units) == 1:
        index = [(token.lemma_.upper() if token.text.isupper() else token.lemma_) in norm_units for token in tokens].index(True)

    return " ".join(norm_units), index, in_unit_list
