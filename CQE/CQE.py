import itertools
import json
import re
import argparse
import logging
import spacy
import os
import numpy
import copy
from spacy.matcher import DependencyMatcher, Matcher
from spacy.tokens import Token
from spacy.lang.lex_attrs import is_digit
from spacy.lang.lex_attrs import like_num as sp_like_num
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, HYPHENS
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
from spacy_download import load_spacy

from fuzzywuzzy import fuzz

from . import NumberNormalizer
from .rules import rules
from pathlib import Path
from .number_lookup import maps, suffixes,circled_digits
from .classes import Change, Value, Range, Unit, Quantity
from .unit_conversion import unit_conversion

def get_project_root() -> Path:
    return Path(__file__).parent

# Add a debug option to the numparser, so that we can disable the warnings
parser = argparse.ArgumentParser()
parser.add_argument('--loglevel',
                    default='',
                    choices=logging._nameToLevel.keys(),
                    help='Provide logging level parameter in order to set what level of log messages you want to record.' )

args = parser.parse_args()

if not args.loglevel: # default is to disable warnings
    logging.getLogger().disabled = True
else:
    logging.basicConfig(format='%(levelname)s:%(message)s', level=args.loglevel) # change default level


#SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
#OBJECTS = ["dobj", "dative", "attr", "oprd"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December", "Jan.", "Feb.", "Aug.", "Sept.", "Oct", "Nov.", "Dec."]
BLACK_LIST_UNITS = {"low", "high", "am"} # black list of units that should not be a unit
MONEY_SYMBOL = {"¥", "$", "₿", "¢", "₡", "₫", "₾", "₵", "₹", "円", "圓", "₭", "₥", "ரூ", "₩"}


def lk_num(text):
    if sp_like_num(text):
        return True
    return lk_scale(text)

def lk_scale(text):
    if text in maps["scales"]:
        return True
    return False

def is_float(string):
    try:
        float(string)
        return float(string) not in [float('inf'), float('-inf')] and float(string) == float(string) # NaN
    except ValueError:
        return False

def lk_digit(text):
    if text[0] == '-': # negative number
        text = text[1:]
    if text.isnumeric():
        return True
    if is_float(text):
        return True
    if text.replace(",","").isnumeric(): # e.g. 117,000
        return True
    return is_float(text.replace(",","")) # e.g. 64,863.10

def clean_up(unit, value, nouns):
    for noun in nouns.copy():
        if noun in unit+value:
            nouns.remove(noun) # (=,400000.0,[barrels, a, day],barrel / day,[decrease, barrels, day]) -> (=,400000.0,[barrels, a, day],barrel / day,[decrease])
    return nouns

def remove_match(func):
    # noinspection PyProtectedMember
    def remove(matcher, doc, i, matches):
        # TODO make this a set and order it later
        remove_ids = []
        for j, match in enumerate(matches):
            if not (all(doc[token_id]._.in_bracket for token_id in match[1]) or
                    all(not doc[token_id]._.in_bracket for token_id in match[1])):
                remove_ids.append(j)
            if any(doc[token_id]._.ignore for token_id in match[1]):
                if j not in remove_ids:
                    remove_ids.append(j)
            if any(doc[token_id]._.consider_date for token_id in match[1]) and not all(doc[token_id]._.consider_date for token_id in match[1]):
                if j not in remove_ids:
                    remove_ids.append(j)
            if any(doc[token_id].text in [",", "or"] for token_id in range(min(match[1]), max(match[1])+1)) and match[0] != 18149999624860298012: # Group Unite, down; $29 million or 33 cents per share
                if j not in remove_ids:
                    remove_ids.append(j)
            if match[0] == 18149999624860298012: # COMPLEX_EXPR
                index = [doc[token].text for token in sorted(match[1])].index("and") if any(doc[token].text == "and" for token in sorted(match[1])) else [doc[token].text for token in sorted(match[1])].index(",")
                if doc[sorted(match[1])[index-1]].lemma_ == doc[sorted(match[1])[-1]].lemma_: # 6.1 inches and 6.7 inches
                    if j not in remove_ids:
                        remove_ids.append(j)
                nouns = ["pound-mass" if doc[t].lemma_ == "pound" else doc[t].lemma_ for t in sorted(match[1]) if doc[t].pos_ in ["PROPN", "NOUN"]]
                if nouns[0] in unit_conversion and (not nouns[1] in unit_conversion[nouns[0]] or (nouns[1] in unit_conversion[nouns[0]] and unit_conversion[nouns[0]][nouns[1]] < 1)): # seven days and 400 miles; 4 months and 3 years; 10 cents, $1
                    if j not in remove_ids:
                        remove_ids.append(j)
                numbers = [doc[t].lemma_ for t in sorted(match[1]) if doc[t].pos_ in ["NUM"]]
                if not all(lk_digit(num) for num in numbers) or not all(num in maps["string_num_map"] for num in numbers): # one-on-one 3 minutes and 47 seconds
                    if j not in remove_ids:
                        remove_ids.append(j)
        for j in sorted(remove_ids, reverse=True):
            matches.pop(j)
            if j < i:
                i = i - 1
            elif j == i:
                return
        func(matcher, doc, i, matches)
    return remove


# noinspection PyProtectedMember
@remove_match
def num_quantmod(matcher, doc, i, matches):
    match_id, tokens = matches[i]
    for token in tokens[1:]:
        if doc[token].dep_ in ["quantmod", "nummod"] and doc[token].pos_ in ["NUM"]:
            doc[token]._.set("number", True)
        elif doc[token].lemma_ in (maps["scales"].keys() | maps["fractions"].keys() | maps["string_num_map"].keys()):
            doc[token]._.set("number", True)
        else:
            doc[token]._.set("bound", True)

# noinspection PyProtectedMember
@remove_match
def lonely_num(matcher, doc, i, matches):
    match_id, tokens = matches[i]
    if not doc[tokens[0]]._.unit:
        doc[tokens[0]]._.set("number", True)
        if doc[tokens[0]].lemma_ in maps["scales"]:
            doc[tokens[0]]._.set("scale", True)
        if doc[tokens[0]].lemma_ in maps["fractions"]:
            doc[tokens[0]]._.set("fraction", True)

# noinspection PyProtectedMember
@remove_match
def default_callback(matcher, doc, i, matches):
    match_id, tokens = matches[i]
    number_index = tokens[[doc[token].pos_=="NUM" for token in tokens].index(True)] if any(doc[token].pos_=="NUM" for token in tokens) else -1
    for token in tokens:
        if doc[token].lemma_ in maps["scales"]:
            doc[token]._.set("scale", True)
        if doc[token].lemma_ in (maps["scales"].keys() | maps["fractions"].keys() | maps["string_num_map"].keys()) or lk_digit(doc[token].lemma_):
            doc[token]._.set("number", True)
            number_index = token
        if doc[token].lemma_ in maps["fractions"]:
            doc[token]._.set("fraction", True)
        if doc[token].text in maps["bounds"] and number_index != -1 and number_index > token: # increased 5% vs. 75 percent increase in e-cigarette usage
            doc[token]._.set("bound", True)
    return

# noinspection PyProtectedMember
@remove_match
def unit_fraction(matcher, doc, i, matches):
    match_id, tokens = matches[i]
    number_index = -1
    tokens = sorted(tokens)
    per_index = tokens.index([token for token in tokens if doc[token].text in ["per", "a", "an"]][0])
    for token in tokens[:per_index]:
        if doc[token].lemma_ in maps["scales"]:
            doc[token]._.set("scale", True)
        if doc[token].lemma_ in (maps["scales"].keys() | maps["fractions"].keys() | maps["string_num_map"].keys()):
            doc[token]._.set("number", True)
            number_index = token
        if doc[token].lemma_ in maps["fractions"]:
            doc[token]._.set("fraction", True)
        if doc[token].text in maps["bounds"] and number_index != -1 and number_index > token:
            doc[token]._.set("bound", True)
    for token in tokens[per_index:]:
        if doc[token].lemma_ in (maps["scales"].keys() | maps["fractions"].keys() | maps["string_num_map"].keys()):
            doc[token]._.set("unit", True)
            doc[token]._.set("number", False) # e.g. 64.2 cents a dozen
        if doc[token].lemma_ in maps["fractions"]:
            doc[token]._.set("fraction", True)
        if doc[token].text in maps["bounds"] and number_index != -1 and number_index > token:
            doc[token]._.set("bound", True)
        else:
            doc[token]._.set("bound", False)

# noinspection PyProtectedMember
@remove_match
def range_callback(matcher, doc, i, matches):
    match_id, tokens = matches[i]
    if len(tokens) == len(set(tokens)) > 3:
        j = len(tokens)
        for k, token in enumerate(sorted(tokens)):
            if doc[token].pos_ in ["NUM"]:
                doc[token]._.set("number", True)
                if not doc[token].lemma_ in maps["scales"]:
                    if j > k:
                        doc[token]._.set("range_l", True)
                        j = k
                    else:
                        doc[token]._.set("range_r", True)
            if doc[token].pos_ in ["ADP", "CCONJ", "PART"]:
                doc[token]._.set("bound", True)
    elif len(tokens) == 3 and len([token for token in tokens if not doc[token].lemma_ in maps["scales"] and doc[token].pos_ == "NUM"]) == 2 :
        consider_sport_unit = bool("-" in [doc[token].text for token in tokens]) # num_to_num_dig
        tokens = sorted(list(set(tokens)))
        if doc[tokens[0]].pos_ in ["NUM"] and doc[tokens[-1]].pos_ in ["NUM"]:
            doc[tokens[0]]._.set("number", True)
            if not doc[tokens[0]].lemma_ in maps["scales"]:
                doc[tokens[0]]._.set("range_l", True)
            doc[tokens[2]]._.set("number", True)
            if not doc[tokens[2]].lemma_ in maps["scales"]:
                doc[tokens[2]]._.set("range_r", True)
            doc[tokens[1]]._.set("bound", True)
            if consider_sport_unit is True: # extract <number>-<number> quantities, keep them only if they have proper unit
                doc[tokens[0]]._.set("consider", True)
                doc[tokens[2]]._.set("consider", True)

# noinspection PyProtectedMember
@remove_match
def between_range_callback(matcher, doc, i, matches): # add for quantities like 'between 100 and 300'
    match_id, tokens = matches[i]
    range_l_r = []
    bound_indices = []
    if max(tokens) - min(tokens) <= 10:
        for token in tokens:
            if doc[token].pos_ in ["NUM"]:
                doc[token]._.set("number", True)
                if not doc[token].lemma_ in maps["scales"]:
                    range_l_r.append(token)
            if doc[token].pos_ in ["ADP", "CCONJ", "PART"]:
                doc[token]._.set("bound", True)
                bound_indices.append(token)
        bound_index_l = min(bound_indices)
        bound_index_r = max(bound_indices)
        if bound_index_r-bound_index_l <= 5:
            for j, num_index in enumerate(sorted(range_l_r)):
                if j == 0 and bound_index_l < num_index < bound_index_r:
                    if not doc[num_index]._.range_r and not doc[num_index]._.range_l: # check if already set
                        doc[num_index]._.set("range_l", True)
                    elif j < len(range_l_r) and doc[num_index]._.range_l: # left already set
                        continue
                    else:
                        matches[i][1].remove(num_index)
                        break
                elif bound_index_r < num_index and not doc[num_index]._.range_l: # e.g. from 12.3 inches to between 12.7 and 13 # j == 1?
                    doc[num_index]._.set("range_r", True)

# noinspection PyProtectedMember
@remove_match
def broad_range_single(matcher, doc, i, matches):
    match_id, tokens = matches[i]
    if lk_num(doc[tokens[0]].lemma_) and not is_digit(doc[tokens[0]].lemma_):
        doc[tokens[0]]._.set("number", True)
        doc[tokens[0]]._.set("broad_range", True)

# noinspection PyProtectedMember
@remove_match
def broad_range_double(matcher, doc, i, matches):
    match_id, tokens = matches[i]
    for token in tokens:
        if lk_num(doc[token].lemma_) and not is_digit(doc[token].lemma_):
            doc[tokens[0]]._.set("number", True)
            doc[tokens[0]]._.set("broad_range", True)

# noinspection PyProtectedMember
@remove_match
def frac_callback(matcher, doc, i, matches):
    match_id, tokens = matches[i]
    doc[tokens[0]]._.set("denominator", True)
    doc[tokens[-1]]._.set("numerator", True)

# noinspection PyProtectedMember
@remove_match
def one_of_callback(matcher, doc, i, matches):
    match_id, tokens = matches[i]
    appropriate_flag = False # appropriate match: One of the students vs. million of the 1.5 billion
    for j,token in enumerate(tokens):
        if doc[token].pos_ == "NUM" and doc[token].text.lower() in maps["string_num_map"]:
            appropriate_flag = True
            doc[token]._.set("number", True)
            doc[token]._.set("one_of_number", True)
        if doc[token].pos_ == "NOUN" and appropriate_flag:
            if any(token.pos_=="NUM" and is_digit(token.text) for token in list(doc[token].children)): # one of those 40 percent
                for token in tokens[:j]:
                    doc[token]._.set("ignore", True)
            else:
                doc[token]._.set("one_of_noun", True)

# noinspection PyProtectedMember
@remove_match
def compound_num(matcher, doc, i, matches):
    match_id, tokens = matches[i]
    for token in tokens:
        doc[token]._.set("number", True)

@remove_match
def complex_expr_callback(matcher, doc, i, matches):
    match_id, tokens = matches[i]
    if not (any(doc[t]._.range_l for t in tokens) and any(doc[t]._.range_r for t in tokens)): # 5 hours and 30 minutes
        index = [doc[token].text for token in sorted(tokens)].index("and") if any(doc[t].text == "and" for t in tokens) else [doc[token].text for token in sorted(tokens)].index(",")
        for token in sorted(tokens)[:index]: # 50 hours
            if not doc[token]._.second_compl_num:
                doc[token]._.set("first_compl_num", True)
        for token in sorted(tokens)[index+1:]: # 30 minutes
            doc[token]._.set("second_compl_num", True)

# noinspection PyProtectedMember
def phone_number(matcher, doc, i, matches):
    match_id, start, end = matches[i]
    span = doc[start:end]
    if span:
        for token in span:
            token._.set("ignore", True)

def zip_code(matcher, doc, i, matches):
    match_id, start, end = matches[i]
    span = doc[start:end]
    if span:
        for token in span:
            token._.set("ignore", True)
path = get_project_root()
file_name = os.path.join(path, "unit.json")
with open(file_name, "r", encoding="utf8") as file:
    units_dict = json.load(file)
class CQE:
    def __init__(self, overload=False, ignore_for=False, spacy_model: str = "en_core_web_sm"):
        self.nlp = load_spacy(spacy_model)
        self.overload = overload
        self.ignore_some_rules = ignore_for # of, for, off after the quantity
        self._modify_defaults_stopwords()
        prefixes = list(self.nlp.Defaults.prefixes)

        prefixes.append("~")
        prefixes.append(">")
        prefixes.append("<")
        prefix_regex = spacy.util.compile_prefix_regex(prefixes)
        self.nlp.tokenizer.prefix_search = prefix_regex.search

        # Modify tokenizer infix patterns

        infixes = (
                LIST_ELLIPSES
                + LIST_ICONS
                + [
                    r"(?<=[0-9])[+\-\*^](?=[0-9-])", # result: +,-,*,^ (positive lookahead)
                    r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                        al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
                    ),
                    r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
                    r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
                    r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
                    r"[a-zA-Z]+[+\-\*^][a-zA-Z]+", #.format(a=ALPHA),  # Newly added not form spacy
                ]
        )


        infix_re = compile_infix_regex(infixes)
        self.nlp.tokenizer.infix_finditer = infix_re.finditer
        self.matcher = DependencyMatcher(self.nlp.vocab) # match subtrees within a dependency parse

        # ranges (added first because of the check for filtering out)
        self.matcher.add("NUM_TO_NUM", [rules["num_to_num"], rules["num_to_num_2"], rules["num_to_num_3"], rules["num_to_num_4"], rules["num_to_num_5"], rules["num_to_num_dig"]], on_match=range_callback)
        self.matcher.add("ADP_NUM_CCONJ_NUM", [rules["adp_num_cconj_num"], rules["adp_num_cconj_num_2"], rules["adp_num_cconj_num_3"]], on_match=between_range_callback)
        #self.matcher.add("ADP_NUM_CCONJ_NUM_WITH_SCALE", [rules["adp_num_cconj_num_with_scale"]], on_match=between_range_callback) #
        self.matcher.add("ADP_NUM_CCONJ_NUM_WITH_UNIT", [rules["adp_num_cconj_num_with_unit"], rules["adp_num_cconj_num_with_unit_2"], rules["adp_num_cconj_num_with_unit_3"], rules["adp_num_cconj_num_with_unit_4"], rules["adp_num_cconj_num_with_unit_5"], rules["adp_num_cconj_num_with_unit_6"], rules["adp_num_cconj_num_with_unit_7"]], on_match=between_range_callback)
        # other ranges
        self.matcher.add("RANGE_SINGLE", [rules["range_single"]], on_match=broad_range_single)
        self.matcher.add("RANGE_DOUBLE", [rules["range_double"]], on_match=broad_range_double)

        # fractions
        self.matcher.add("FRAC", [rules["frac"], rules["frac_2"]], on_match=frac_callback)

        self.matcher.add("COMPOUND_NUM", [rules["compound_num"], rules["compound_num_2"]], on_match=compound_num)
        self.matcher.add("NUM_NUM", [rules["num_num"]], on_match=default_callback)
        self.matcher.add("NUM_SYMBOL", [rules["num_symbol"]], on_match=default_callback)
        self.matcher.add("SYMBOL_NUM", [rules["symbol_num"]], on_match=default_callback)

        # rules including check for filtering out (e.g. iphone 11, FTSE 100 etc.)
        self.matcher.add("COMPLEX_EXPR", [rules["complex_num_expression"]], on_match=complex_expr_callback)
        self.matcher.add("NOUN_NUM", [rules["noun_num"]], on_match=default_callback)
        self.matcher.add("NUM_QUANTMOD", [rules["num_quantmod"], rules["num_quantmod_chain"]], on_match=num_quantmod)
        self.matcher.add("NUM_DIRECT_PROPN", [rules["num_direct_propn"]], on_match=default_callback)

        # known rules
        self.matcher.add("MINUS_NUM", [rules["minus_num"], rules["minus_num_2"]], on_match=num_quantmod)
        self.matcher.add("QUANTMOD_DIRECT_NUM", [rules["quantmod_direct_num"]], on_match=num_quantmod)
        self.matcher.add("VERB_QUANTMOD_NUM", [rules["verb_quantmod_num"]], on_match=num_quantmod)
        self.matcher.add("VERB_QUANTMOD2_NUM", [rules["verb_quantmod2_num"]], on_match=num_quantmod)
        self.matcher.add("NUM_DIRECT_NOUN", [rules["num_direct_noun"]], on_match=default_callback)
        self.matcher.add("NOUN_NUM_ADP_RIGHT_NOUN", [rules["noun_num_adp_right_noun"]], on_match=default_callback)
        self.matcher.add("NOUN_NUM_QUANT", [rules["noun_num_quant"], rules["noun_num_quant_2"], rules["noun_num_quant_3"], rules["noun_num_quant_4"]], on_match=default_callback)
        self.matcher.add("NOUN_COMPOUND_NUM", [rules["noun_compound_num"]], on_match=default_callback)
        self.matcher.add("NOUN_ADJ_NUM", [rules["noun_adj_num"]], on_match=default_callback)
        self.matcher.add("ADJ_NUM", [rules["adj_num"]], on_match=default_callback)
        self.matcher.add("ADJ_NOUN_NUM", [rules["adj_noun_num"]], on_match=default_callback)
        self.matcher.add("NOUN_NOUN", [rules["noun_noun"], rules["noun_noun_2"]], on_match=default_callback)
        self.matcher.add("UNIT_FRAC", [rules["unit_frac"], rules["unit_frac_2"], rules["unit_frac_3"], rules["unit_frac_4"], rules["unit_frac_5"], rules["unit_frac_6"]], on_match=unit_fraction)
        self.matcher.add("NUM_NOUN", [rules["num_noun"]], on_match=default_callback)
        self.matcher.add("DIM2", [rules["dimensions_2"]], on_match=default_callback)
        self.matcher.add("NOUN_QUANT_NOUN_NOUN", [rules["noun_quant_noun_noun"]], on_match=default_callback)

        # other (unknown)
        self.matcher.add("NOUN_NUM_RIGHT_NOUN", [rules["noun_num_right_noun"]], on_match=default_callback)
        self.matcher.add("NUM_NUM_ADP_RIGHT_NOUN", [rules["num_num_adp_right_noun"]], on_match=default_callback)
        self.matcher.add("NUM_TO_NUM_NUM", [rules["num_to_num_num"], rules["num_to_num_num_dig"]], on_match=default_callback) #
        self.matcher.add("NUM_RIGHT_NOUN", [rules["num_right_noun"]], on_match=default_callback) #

        self.matcher.add("LONELY_NUM", [rules["lonely_num"]], on_match=lonely_num)
        self.matcher.add("ONE_OF", [rules["one_of"]], on_match=one_of_callback)

        self.pattern_matcher = Matcher(self.nlp.vocab) # match sequences of tokens, based on pattern rules
        self.pattern_matcher.add("PHONE_NUMBER",
                                 [rules["phone_number_pattern_1"], rules["phone_number_pattern_2"], rules["phone_number_pattern_3"], rules["phone_number_pattern_4"], rules["phone_number_pattern_5"]],
                                 on_match=phone_number)
        self.pattern_matcher.add("ZIP_CODE", [rules["zip_number_pattern"]], on_match=zip_code)

        if not Token.has_extension("bound"):
            Token.set_extension("bound", default=None)
        if not Token.has_extension("in_bracket"):
            Token.set_extension("in_bracket", default=None)
        if not Token.has_extension("in_comma"):
            Token.set_extension("in_comma", default=None)
        if not Token.has_extension("number"):
            Token.set_extension("number", default=None)
        if not Token.has_extension("range_r"):
            Token.set_extension("range_r", default=None)
        if not Token.has_extension("range_l"):
            Token.set_extension("range_l", default=None)
        if not Token.has_extension("broad_range"):
            Token.set_extension("broad_range", default=None)
        if not Token.has_extension("consider"):
            Token.set_extension("consider", default=None)
        if not Token.has_extension("consider_date"):
            Token.set_extension("consider_date", default=None)
        if not Token.has_extension("ignore"):
            Token.set_extension("ignore", default=None)
        if not Token.has_extension("ignore_noun"):
            Token.set_extension("ignore_noun", default=None)
        if not Token.has_extension("unit"):
            Token.set_extension("unit", default=None)
        if not Token.has_extension("one_of_number"):
            Token.set_extension("one_of_number", default=None)
        if not Token.has_extension("one_of_noun"):
            Token.set_extension("one_of_noun", default=None)
        if not Token.has_extension("scale"):
            Token.set_extension("scale", default=None)
        if not Token.has_extension("like_scale"):
            Token.set_extension("like_scale", getter=lambda token: lk_scale(token.lemma_))
        if not Token.has_extension("numerator"):
            Token.set_extension("numerator", default=None)
        if not Token.has_extension("denominator"):
            Token.set_extension("denominator", default=None)
        if not Token.has_extension("like_number"):
            Token.set_extension("like_number", getter=lambda token: lk_num(token.lemma_))
        if not Token.has_extension("like_numeric_value"):
            Token.set_extension("like_numeric_value", getter=lambda token: lk_digit(token.lemma_))
        if not Token.has_extension("fraction"):
            Token.set_extension("fraction", getter=lambda token: token.lemma_ in maps["fractions"]) #.keys())
        if not Token.has_extension("quantity_part"):
            Token.set_extension("quantity_part", default=None)
        if not Token.has_extension("first_compl_num"):
            Token.set_extension("first_compl_num", default=None)
        if not Token.has_extension("second_compl_num"):
            Token.set_extension("second_compl_num", default=None)

    @staticmethod
    def get_unit_hierarchy():
        """returns unit hierarchy based on the entity attribute"""


        unit_hierarchy = {}
        for entry in units_dict:
            if units_dict[entry]["entity"] not in unit_hierarchy:
                unit_hierarchy[units_dict[entry]["entity"]] = {}
            unit_hierarchy[units_dict[entry]["entity"]][entry] = {"surfaces": units_dict[entry]["surfaces"], "symbols": units_dict[entry]["symbols"]}
        return unit_hierarchy

    @staticmethod
    def get_all_units():
        """returns a list of all normalized forms of the units we have"""

        return list(units_dict.keys())

    @staticmethod
    def get_all_unit_surface_forms():
        """returns a dictionary with all units, their surface forms and symbols"""

        return { key: { "surfaces": units_dict[key].get("surfaces"), "symbols": units_dict[key].get("symbols") } for key in units_dict.keys() }

    @staticmethod
    def get_specific_unit_surface_forms(normalized_unit):
        """given a normalized form of a unit, returns all the surface forms and symbols"""

        splitted_normalized_unit = normalized_unit.split()

        if "per" in splitted_normalized_unit: # e.g. 'dollar per square foot', 'cent per share per year'
            keys = normalized_unit.split(" per ")
            return { normalized_unit: { "surfaces": [' per '.join(element) for element in itertools.product(*[units_dict[key].get("surfaces") if key in units_dict else [key] for key in keys])], \
                                        "symbols": [' per '.join(element) for element in itertools.product(*[units_dict[key].get("symbols") if key in units_dict else [key] for key in keys])]
                                        }
                     } | { key: { "surfaces": units_dict[key].get("surfaces"), "symbols": units_dict[key].get("symbols") } if key in units_dict else {} for key in keys }

        if "a" in splitted_normalized_unit: # e.g. 'dollar a month'
            keys = normalized_unit.split(" a ")
            return { normalized_unit: { "surfaces": [' a '.join(element) for element in itertools.product(*[units_dict[key].get("surfaces") if key in units_dict else [key] for key in keys])], \
                                        "symbols": [' a '.join(element) for element in itertools.product(*[units_dict[key].get("symbols") if key in units_dict else [key] for key in keys])]
                                        }
                     } | { key: { "surfaces": units_dict[key].get("surfaces"), "symbols": units_dict[key].get("symbols") } if key in units_dict else {} for key in keys }

        key_unit_list, prefix_symbol, prefix_text = NumberNormalizer.find_prefix_and_unit(normalized_unit)
        if key_unit_list:
            if len(key_unit_list) == 2: # e.g. km/h
                if prefix_symbol or prefix_text:
                    first_key_prefixes = prefix_symbol
                    second_key_prefixes = prefix_text
                    if first_key_prefixes and second_key_prefixes:
                        return { normalized_unit: { "surfaces": [' per '.join(element) for element in itertools.product(*[[first_key_prefixes[0]+surface if key_unit_list[0] in units_dict else [first_key_prefixes[0]+key_unit_list[0]] for surface in units_dict[key_unit_list[0]].get("surfaces")], [second_key_prefixes[0]+surface if key_unit_list[1] in units_dict else [second_key_prefixes[0]+key_unit_list[1]] for surface in units_dict[key_unit_list[1]].get("surfaces")]])], \
                                                    "symbols": [' per '.join(element) for element in itertools.product(*[[first_key_prefixes[1]+symbol if key_unit_list[0] in units_dict else [first_key_prefixes[1]+key_unit_list[0]] for symbol in units_dict[key_unit_list[0]].get("symbols")], [second_key_prefixes[1]+symbol if key_unit_list[1] in units_dict else [second_key_prefixes[1]+key_unit_list[1]] for symbol in units_dict[key_unit_list[1]].get("symbols")]])]
                                                    }
                                 } | { first_key_prefixes[0]+key_unit_list[0]: { "surfaces": [ first_key_prefixes[0]+surface if key_unit_list[0] in units_dict else [] for surface in units_dict[key_unit_list[0]].get("surfaces") ] } | { "symbols": [ first_key_prefixes[1]+symbol if key_unit_list[0] in units_dict else [] for symbol in units_dict[key_unit_list[0]].get("symbols") ] }, \
                                       second_key_prefixes[0]+key_unit_list[1]: { "surfaces": [ second_key_prefixes[0]+surface if key_unit_list[1] in units_dict else [] for surface in units_dict[key_unit_list[1]].get("surfaces") ] } | { "symbols": [ second_key_prefixes[1]+symbol if key_unit_list[1] in units_dict else [] for symbol in units_dict[key_unit_list[1]].get("symbols") ] }
                                       }
                    if first_key_prefixes:
                        return { normalized_unit: { "surfaces": [' per '.join(element) for element in itertools.product(*[[first_key_prefixes[0]+surface if key_unit_list[0] in units_dict else [first_key_prefixes[0]+key_unit_list[0]] for surface in units_dict[key_unit_list[0]].get("surfaces")], units_dict[key_unit_list[1]].get("surfaces") if key_unit_list[1] in units_dict else [key_unit_list[1]]])], \
                                                    "symbols": [' per '.join(element) for element in itertools.product(*[[first_key_prefixes[1]+symbol if key_unit_list[0] in units_dict else [first_key_prefixes[1]+key_unit_list[0]] for symbol in units_dict[key_unit_list[0]].get("symbols")], units_dict[key_unit_list[1]].get("symbols") if key_unit_list[1] in units_dict else [key_unit_list[1]]])]
                                                    }
                                 } | { first_key_prefixes[0]+key_unit_list[0]: { "surfaces": [ first_key_prefixes[0]+surface if key_unit_list[0] in units_dict else [] for surface in units_dict[key_unit_list[0]].get("surfaces") ] } | { "symbols": [ first_key_prefixes[1]+symbol if key_unit_list[0] in units_dict else [] for symbol in units_dict[key_unit_list[0]].get("symbols") ] }, \
                                       key_unit_list[1]: { "surfaces": units_dict[key_unit_list[1]].get("surfaces"), "symbols": units_dict[key_unit_list[1]].get("symbols") if key_unit_list[1] in units_dict else {} }
                                       }
                    if second_key_prefixes:
                        return { normalized_unit: { "surfaces": [' per '.join(element) for element in itertools.product(*[[second_key_prefixes[0]+surface if key_unit_list[1] in units_dict else [second_key_prefixes[0]+key_unit_list[1]] for surface in units_dict[key_unit_list[1]].get("surfaces") ], units_dict[key_unit_list[0]].get("surfaces") if key_unit_list[0] in units_dict else [key_unit_list[0]]])], \
                                                    "symbols": [' per '.join(element) for element in itertools.product(*[[second_key_prefixes[1]+symbol if key_unit_list[1] in units_dict else [second_key_prefixes[1]+key_unit_list[1]] for symbol in units_dict[key_unit_list[1]].get("symbols")], units_dict[key_unit_list[0]].get("symbols") if key_unit_list[0] in units_dict else [key_unit_list[0]]])]
                                                    }
                                 } | { key_unit_list[0]: { "surfaces": units_dict[key_unit_list[0]].get("surfaces"), "symbols": units_dict[key_unit_list[0]].get("symbols") if key_unit_list[0] in units_dict else {} }, \
                                       second_key_prefixes[0]+key_unit_list[1]: { "surfaces": [ second_key_prefixes[0]+surface if key_unit_list[1] in units_dict else [] for surface in units_dict[key_unit_list[1]].get("surfaces") ] } | { "symbols": [ second_key_prefixes[1]+symbol if key_unit_list[1] in units_dict else [] for symbol in units_dict[key_unit_list[1]].get("symbols") ] }
                                       }

                return { normalized_unit: { "surfaces": [' per '.join(element) for element in itertools.product(*[units_dict[key].get("surfaces") if key in units_dict else [key] for key in key_unit_list])], \
                                            "symbols": [' per '.join(element) for element in itertools.product(*[units_dict[key].get("symbols") if key in units_dict else [key] for key in key_unit_list])]
                                            }
                         } | { key: { "surfaces": units_dict[key].get("surfaces"), "symbols": units_dict[key].get("symbols") } if key in units_dict else {} for key in key_unit_list }

            if prefix_symbol and prefix_text: # e.g. millimeter
                return { prefix_text+key: { "surfaces": [ prefix_text+surface for surface in units_dict[key].get("surfaces") ], "symbols": [ prefix_symbol+symbol for symbol in units_dict[key].get("symbols") ] } if key in units_dict else {} for key in key_unit_list}
            return { key: { "surfaces": units_dict[key].get("surfaces"), "symbols": units_dict[key].get("symbols") } if key in units_dict else {} for key in key_unit_list}

        return {}

    def parse(self, text):
        # multiple sentences as input
        """if len([sent for sent in self.nlp(text).sents]) > 1:
            result = []
            for sent in list(self.nlp(text).sents):
                result.extend(self.parse(str(sent)))
            return result"""

        # single sentence as input
        original_text = text
        text,old_to_new_suffixes = self._preprocess_text(text)
        doc = self.nlp(text)
        doc = self._preprocess_doc(doc)
        candidates = self._extract_candidates(doc)
        sent_boundaries = self._sentence_boundaries(doc)
        three_tuples = self._match_tuples(candidates, doc, sent_boundaries)
        referred_nouns = self._extract_global_referred_nouns(doc)
        self._postprocess_nouns(three_tuples, referred_nouns)
        normalized_tuples = self._normalize_tuples(three_tuples, referred_nouns, text, doc)

        if self.overload:
            Quantity.original_text = original_text
            normalized_text = self._normalize_text(normalized_tuples, doc)
            normalized_text_values_only = self._normalize_text_values_only(normalized_tuples, doc)
            normalized_text_units_only = self._normalize_text_units_only(normalized_tuples, doc)
            quantity=None
            for quantity in normalized_tuples:
                quantity.set_preprocessed_text(text)
                quantity.set_normalized_text(normalized_text)
                quantity.set_normalized_text_values_only(normalized_text_values_only)
                quantity.set_normalized_text_units_only(normalized_text_units_only,old_to_new_suffixes)
                quantity.update_indices(old_to_new_suffixes)

            if quantity:
                normalized_scientific_text_values_only,scientific_notations= self._normalize_text_scientific_notation_values_only(normalized_tuples, quantity.get_preprocessed_text())
                normalized_scientific_text = self._normalize_text_scientific_notation(normalized_tuples, normalized_scientific_text_values_only)

                if scientific_notations is not None: # digits, not written words
                    quantity.set_scientific_notations(scientific_notations)
                for quantity in normalized_tuples:
                    quantity.set_normalized_scientific_text(normalized_scientific_text)
                    quantity.set_normalized_scientific_text_with_values_only(normalized_scientific_text_values_only)

        return normalized_tuples

    def _print_tokens(self, doc):
        #print([(token.text, token.dep_, token.pos_, list(token.children)) for token in doc if token.dep_=="ROOT"])
        #print([(token.text, token.dep_, token.pos_, list(token.children)) for token in doc if token.dep_ in ["nsubj", "nsubjpass"]])
        #print([(token.text, token.dep_, token.pos_, list(token.children)) for token in doc if token.pos_ == "PROPN"])
        #print([spacy.explain(token.tag_) for token in doc])
        #print([(token.text, list(token.subtree)) for token in doc])
        #print([(token.text, list(token.ancestors)) for token in doc])
        print([(token.text, token.dep_, token.pos_, list(token.children), token.i) for token in doc])

    def _print_temp_results(self, boundaries, candidates, tuples, nouns):
        print(f"boundaries: {boundaries}")
        print(f"candidates: {candidates}")
        print(f"tuples: {tuples}")
        print(f"nouns: {nouns}")

    def _print_sentences(self, doc, boundaries): # print clauses of the sentence
        for bound in boundaries:
            print(doc[bound[0]:bound[1]+1])

    def _modify_defaults_stopwords(self):
        stopwords_to_be_removed = []
        for stopword in self.nlp.Defaults.stop_words:
            if "NUM" in [token.pos_ for token in self.nlp(stopword)]: # numbers
                stopwords_to_be_removed.append(stopword)
        self.nlp.Defaults.stop_words -= set(stopwords_to_be_removed) # remove
        self.nlp.Defaults.stop_words -= {"top","us","up","amount","never"} # additional that should not be considered stop words
        self.nlp.Defaults.stop_words |= {"total","instead","maybe", "≈", "minus", "s"} # add

    def _remove_stopwords_from_noun(self, nouns): # remove stop words like "a, of" etc. from referred_noun
        def accept_token(token): # accept until punct (assumption: new subsentence, exception e.g. Australias S&P/ASX 200), Relative or demonstrative pronoun
            return not token.pos_ in ["PUNCT"] and not (token.text in ["that","which","what","who","whose","whom"]) or token.text in ["/", "-"] # token.pos_ == "DET" and this, that
        filtered_nouns = list(itertools.takewhile(accept_token, nouns))
        return [token for token in filtered_nouns if not token.text.lower() in self.nlp.Defaults.stop_words and not token.pos_ in ["PUNCT", "SYM"] and not (token.pos_ in ["VERB"] and token.dep_ in ["xcomp"]) and not token._.bound]

    def _sentence_boundaries(self, doc):
        boundaries = []
        for sent in doc.sents:
            last = sent[0].i

            # list of the indices of comma separated parts in the sentence
            # e.g. while energy, last years worst performing sector, fell 0.94% as concerns about an economic slowdown also hit oil prices.
            comma_separated_parts = []
            j=-1 # last already considered index
            for i,token in enumerate(doc):
                if token._.in_comma and i>j:
                    comma_separated_parts.append([token.i-1]) # first comma
                    comma_separated_parts[-1].extend([t.i for t in list(itertools.takewhile(lambda x:x._.in_comma, doc[token.i:]))]) # tagged tokens
                    j = comma_separated_parts[-1][-1]
                    comma_separated_parts[-1].extend([j+1]) # second comma

            for tok in sent:
                # not child of the previous token, e.g. of the total issued and outstanding shares
                # not part of range, e.g. between ... and ...
                # coordinating conjunction (e.g. and,or,but) or marker (=word marking a clause as subordinate to another clause), but not modifier of quantifier
                # not followed by a verb
                # not ( ) = or -
                if (tok not in list(doc[tok.i-1].children) and not tok._.bound and (tok.pos_ in ["CCONJ"] or tok.dep_ in ["mark"]) and tok.dep_ not in ["quantmod"] and (tok.i+1 < len(doc) and doc[tok.i+1].pos_ not in ["VERB"])) or (tok.pos_ in ["PUNCT"] and tok.text not in ["(", ")", "=", "-"]):
                    if tok._.in_comma:
                        continue
                    if tok.i in [part[-1] for part in comma_separated_parts]:
                        if boundaries:
                            boundaries[-1][-1]=tok.i
                        else:
                            boundaries.append([0, tok.i]) # when no boundaries yet, add first boundary

                    elif last != tok.i:
                        if tok.i - last >= 3:
                            if (boundaries and boundaries[-1][-1]-boundaries[-1][0]<=3):
                                boundaries[-1][-1]=tok.i
                            else:
                                boundaries.append([last, tok.i])
                        elif boundaries:
                            boundaries[-1][-1]=tok.i
                        else:
                            boundaries.append([last, tok.i])
                    elif tok.dep_ in ["mark"]: # e.g. while
                        boundaries.append([tok.i, tok.i])
                    elif boundaries:
                        boundaries[-1][-1]=tok.i
                    last = tok.i + 1

            # should be removed, since temp fixes!
            if boundaries and boundaries[0][0] != 0:
                boundaries[0][0] = 0 # fix the first boundary
            if last != sent[-1].i and sent[-1].i-last > 3: # omit boundaries like [[0, 14], [15, 25], [26, 26]] and [[0, 17], [18, 24], [25, 24]]
                boundaries.append([last, sent[-1].i])
            elif boundaries:
                boundaries[-1][-1]=sent[-1].i # -> [[0, 14], [15, 26]]

        return boundaries

    def _preprocess_doc(self, doc):
        doc = self._retokenize(doc)
        doc = self._tag_brackets(doc)
        doc = self._tag_commas(doc)
        doc = self._tag_ignore(doc)
        return doc

    def _preprocess_text(self, text: str):

        for k, v in circled_digits.items():
            text = re.sub(rf"{k}", f"{v}", text) # remove circled numbers

        for k, v in maps["vulgar_fractions"].items():
            text = re.sub(rf"(\d+)\s?{k}", rf"\1.{str(v).split('.')[1]}", text) # 7½ -> 7.5
            text = re.sub(rf"{k}", rf"{v}", text) # ½ -> 0.5

        text = re.sub(r"(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b", "", text) # remove urls

        # put a space between number and unit so spacy detects them as separate tokens
        text = re.sub(r"(\w)(\([a-zA-Z]+)", r"\1 \2", text)
        text = re.sub(r"(^\w+)\s?(—)\s?", r"\1 ", text) # Walmart — Shares of the retail giant
        text = re.sub(r"(\s+[^\s]+)\s?(—)", r"\1 ", text) # Peabody Energy — Shares of the major coal producer
        text = re.sub(r"[\s]{2,}"," ",text) # remove multiple spaces
        text = text.replace(" | "," ,  ")
        #text = text.replace("\"",". \"")
        text = text.replace(" c $"," C$") # the c $ 300 million charge -> the C$ 300 million charge
        text = re.sub(r"\b(C)\s+(\$)", r"\1\2", text) # C $6 -> C$6
        text = re.sub(r"\b(S)\s(\$)(\s*)", r"\1\2 ", text) # S $ -> S$
        text = text.replace(" EUR "," eur ") # EUR 125 -> eur 125
        text = text.replace(" U.S. "," US ") # $75 U.S. per barrel -> $75 US per barrel
        text = re.sub(r"\bper cent\b", "percent", text)
        text = re.sub(r"\b(percent|%)-([a-z]+)", r"\1 \2", text) # 31%-owned

        text = re.sub(r"\[([a-z]+|[%]+)\]", r"\1", text) # [kg] or [%] -> kg and %
        text = re.sub(r"\<([a-z]+)\>", r"\1", text) # <kg> -> kg
        text = re.sub(r"\(([a-z]{2,3})\)", r"\1", text) # (kg) -> kg

        text = re.sub(r"\b(\d+)(\s*yo)\b", r"\1 years old", text) # 87 yo -> 87 years old
        text = re.sub(r"([\d]+)-[\s+|\.]", r"\1", text) # 18- to 34 year-old -> 18 to 34 year-old

        text = re.sub(r"\b([a-z])\.(?!$)", r"\1", text) # e.g. m.p.h -> mph

        text = text.replace("™","")
        amount_unit = re.findall(r'(?<![A-Z\d,.])[0-9,.]*[0-9]+(?!E\+|E\−|E\d+|e\+|e\−|e-|e\d+|x|×)[a-zA-Z]{1,3}[.,]{0,1}', text)
        for v in amount_unit:
            index_first_alpha = v.find(next(filter(str.isalpha, v)))
            digit_part = []
            for i, char in enumerate(v):
                if is_digit(char) or (char=="." and i < index_first_alpha):
                    digit_part.append(char)
            digit_part = "".join(digit_part) # e.g. 0.1m2, 5.4m.
            char_part = "".join(v[index_first_alpha:])
            if "." in char_part: # and len(text) != text.index(char_part)+len(char_part): # . separates two sentences
                char_part = char_part[:char_part.index(".")]+" "+char_part[char_part.index("."):] # e.g. m.-> m . (e.g. from 140m.)
            text = text.replace(v, digit_part+" "+char_part) # e.g. 2m -> 2 m

        text = re.sub(r"(inr|jpy|cad|sen|usd|aud|eur|rmb|gbp)\s?([0-9,.]*[0-9]+)\s?(m(?![a-z\d]))[., \n]?", r"\1 \2 M", text, flags=re.IGNORECASE) # e.g. USD 10m. -> USD 10 M.
        text = re.sub(r"\bPS([0-9,.]*[0-9]+)", r"£\1", text) # e.g. PS2.7 -> £2.7
        text = re.sub(r"([$|€|£]\s?[0-9,.]*[0-9]+\s?)(m(?![a-z\d]))([., \n]?)", r"\1M\3", text) # e.g. $5.4m. -> $5.4M.
        text = re.sub(r"\b(RM)\s{0,1}([0-9,.]*[0-9]+)(million|mil|m|M)+\b", r"\2 M \1", text) # RM3.16mil -> 3.16 mil RM
        text = re.sub(r"\b(RM)\s{0,1}([0-9,.]*[0-9]+)(k|K|bn|b|B|tn)+\b", r"\2 \3 \1", text) # RM31.20k -> 31.20 k RM
        text = re.sub(r"\b(RM)([0-9,.]*[0-9]+)\b", r"\1 \2", text) # RM23.50 -> RM 23.50

        amount_dash_unit=re.findall(r'[0-9\.]+-[a-zA-z]+', text)
        for v in amount_dash_unit:
            text = text.replace(v, v.replace("-"," ")) # 100-km -> 100 km
        old_to_new_suffixes={}
        for k, v in suffixes.items():
            pattern = re.compile(rf"(?<![A-Z\d,.])([\d,.]+)\s?{k}([\s\n.,()]+|$)")
            for suf_text in re.findall(pattern, text):
                old_suffix_text=suf_text[0]+k+suf_text[1]
                new_text=suf_text[0]+" "+v+suf_text[1]
                old_to_new_suffixes[new_text]=old_suffix_text
            text = re.sub(rf"(?<![A-Z\d,.])([\d,.]+)\s?{k}([\s\n.,()]+|$)", rf"\1 {v}\2", text) # 3K -> 3 thousand
            text = re.sub(rf"([\d,.]+)\s?{v}([-])", rf"\1 {v} ", text) # 400 million-year-old -> 400 million year-old

        text = re.sub(r"([\d+,.]+)\s*(sq)\s(m)\b", r"\1\2\3", text) # 200sq m -> 200 sqm
        text = re.sub(r"([\d+,.]+)\s*(sqm)\b", r"\1 m2", text) # sqm -> m2

        text = re.sub(r"([\d]+)\s+(\d)/(\d)[\s+|\.]", lambda x: str(int(x.group(1)) + int(x.group(2))/int(x.group(3))) if not all(c in ["0",".",","] for c in x.group(3)) else x.group(0), text) # 10 1/2 miles -> 10.5 miles
        text = re.sub(r"\b(\d+[\.|,]?\d*)\/(\d+[\.|,]?[\d,]*)(?![\.]?\d)", lambda x: str(float(x.group(1).replace(",", ""))/float(x.group(2).replace(",", ""))) if not all(c in ["0",".",","] for c in x.group(2)) else x.group(0), text) # 1/2 -> 0.5, 8/0 -> 8/0, 1/10,000 -> 1/10,000

        text = re.sub(r"([\d+,.]+)\s+deg\b", r"\1 degree", text) # 50 deg -> 50 degree
        text = re.sub(r"(degree|degrees)\s+([f\s+|f\.|F\s+|F\.])\b", "degf", text) # 25 degree f -> 25 degf
        text = re.sub(r"\s+(to)(-)([\d,.,,]+)", r"\1 \3", text) # e.g. 1.5 to-2 degree range -> 1.5 to 2 degree range

        text = re.sub(r"(\s)(\$)(US)", r"\1\3\2", text) # $US 113 million -> US$ 113 million
        text = re.sub(r"\b(u\.s|US|hk|HK)(\s*)(\$)(\s*)", lambda x: x.group(1).upper() + x.group(3) + " ", text) # u.s $100,000,000 -> U.S$ 100,000,000
        text = re.sub(r"([0-9,.|million]+\s+)(us\s)", r"\1US ", text) # 15,000 us dollar -> 15,000 US dollar
        text = re.sub(r"(us)\s*(\$)", r"US \2", text) # us $ 150 million -> US $ 150 million
        text = re.sub(r"(aud|usd)(\$)", lambda x: x.group(1).upper() + x.group(2) ,text) # aud$ 15 -> AUD$ 15
        text = re.sub(r"([0-9,.]*[0-9]+)\s*([$|€])", r"\2\1", text) # 10$ -> $10
        text = re.sub(r"(?<![a-zA-Z])(M)(inr|jpy|cad|sen|usd|aud|eur|rmb|gbp)\b",r"\1 \2" , text, flags=re.IGNORECASE) # MUSD -> M USD, but not MSRP -> M SRP

        for k, v in suffixes.items():
            text = re.sub(rf"([0-9,.]*[0-9]+)\s*(inr|jpy|cad|sen|usd|aud|eur|rmb|gbp|₡|₫|₵|¥|₥)(\s*{v})", r"\1\3 \2", text, flags=re.IGNORECASE) # 75 USD million -> 75 million USD
            text = re.sub(rf"(?<![a-zA-Z])(M)*(inr|jpy|cad|sen|usd|aud|eur|rmb|gbp|₡|₫|₵|¥|₥)\s*([0-9,.]*[0-9]+)(\s*{v})*", r"\3\4 \1\2", text, flags=re.IGNORECASE) # usd 20 -> 20 usd

        text = re.sub(r"([0-9,.]*[0-9]+\s*)(bitcoins|bitcoin)", r"\1btc", text) # 10000 bitcoins -> 10000 btc
        text = re.sub(r"\b(us)(?=[\d,.]+)", r"\1 ", text, flags=re.IGNORECASE) # US71.37 -> US 71.37
        text = re.sub(r"(?<![A-Z\d,.])([0-9,.]+)(₿|¢|₡|₫|₾|₵|₹|元|円|圓|₭|₥|ரூ|₩|′|″|”|Å|Ω|Ω)", r"\1 \2", text) # 71\u00a2 -> 71 \u00a2
        text = re.sub(r"([₡|₫|₵|¥|₥|$|₿|¢|₡|₫|₾|₵|₹|元|円|圓|₭|₥|ரூ|₩]+[0-9,.]*[0-9]+)(-)([₡|₫|₵|¥|₥|$|₿|¢|₡|₫|₾|₵|₹|元|円|圓|₭|₥|ரூ|₩]+[0-9,.]*[0-9]+)", r"\1 \2 \3", text) # $102-$110 -> $102 - $110

        text = re.sub(r"([0-9,.]*[0-9]+\s*)(mB|gB|kB|tB)\b", lambda x: x.group(1) + x.group(2).upper(), text) # 10 mB -> 10 MB
        text = re.sub(r"([0-9,.]*[0-9]+\s*)(mb|Mb|gb|Gb|kb|Kb|tb|Tb)\b", lambda x: x.group(1) + x.group(2).upper()[:1]+"bit", text) # 10 mb -> 10 Mbit

        text = re.sub(r"([a-z]+)(-)(a)(-)([a-z]+)", r"\1 \3 \5", text) # 12-cent-a-share -> 12-cent a share

        text = re.sub(r"(?<![\^|\d|a-zA-Z])\+(?!\s?\d{2}\s?[(]?\d{4}[)]?\s?\d{4,7}\b)", r"up ", text) # +0.2% -> up 0.2%
        text = re.sub(r"minus-([\d,.,,]+)", r"-\1", text) # minus-130 -> -130
        text = re.sub(r"[\s]-([\d,.,,]+)", r" minus \1", text) # -5 -> minus 5
        text = re.sub(r"^-([\d,.,,]+)", r"minus \1", text) # when at the beginning of the line
        text = re.sub(r"sub-([\d,.,,]+)", r"under \1", text) # sub-500 sqm -> under 500 sqm

        # note: (?<![A-Z\d,.]) is needed to exclude things like PS88.3m
        text = re.sub(r"(?<![A-Z\d,.])([0-9,.]*[0-9]+)([a-zA-Z,\/]{2,})", r"\1 \2", text) # 14days -> 14 days
        text = re.sub(r"(\sHz|\shz)\b", lambda x: x.group(1).upper(), text) # 300 Hz -> 300 HZ

        text = re.sub(r"([a-z,A-Z])(=)([0-9,.]*[0-9]+)", r"\3 \1", text) # e.g. Y=28,030 -> 28,030 Y

        # add dot, question mark or exclamation mark and additional space before the end of the sentence if not there
        if text.endswith(' .'):
            return text,old_to_new_suffixes
        if text.endswith('?'):
            return text[:-1]+' ?',old_to_new_suffixes
        if text.endswith('!'):
            return text[:-1]+' !',old_to_new_suffixes
        if text.endswith(' .\n'):
            return text[:-1],old_to_new_suffixes
        if text.endswith('.\n'):
            return text[:-2]+' .',old_to_new_suffixes
        if text.endswith('?\n'):
            return text[:-2]+' ?',old_to_new_suffixes
        if text.endswith('!\n'):
            return text[:-2]+' !',old_to_new_suffixes
        if text.endswith(' \n'):
            return text[:-1]+'.',old_to_new_suffixes
        if text.endswith('\n'):
            return text[:-1]+' .',old_to_new_suffixes
        return (text+' .',old_to_new_suffixes) if not text.endswith('.') else (text[:-1]+' .',old_to_new_suffixes)

    def _postprocess_nouns(self, tuples, nouns):
        nouns_copy = nouns.copy()
        for noun in nouns_copy:
            # remove single character nouns like: '[m]' (e.g. from '... between 60 and 90 m subscribers')
            if len(noun[:-1]) == 1 and len(noun[:-1][0].text) == 1:
                try:
                    nouns.remove(noun)
                except ValueError:
                    continue
            # remove nouns consisting mainly of NUMs
            if len([token for token in noun[:-1] if token.pos_ ==  "NUM"]) > len(noun[:-1])/2:
                try:
                    nouns.remove(noun)
                except ValueError:
                    continue
            if all(token._.ignore for token in noun[:-1]):
                try:
                    nouns.remove(noun) # e.g. nov 6 (DATE, TIME entities)
                except ValueError:
                    continue

        # remove nouns like: '[536, hp, a, combined, torque, output, of, lb-ft, 20]', '[which, 1]'
        for tuple, noun in itertools.product(tuples, nouns):
            if (tuple[1] and tuple[1][0] in noun[:-1]) or (all(token.pos_ in ["DET", "PRON"] for token in noun[:-1])):
                try:
                    nouns.remove(noun)
                except ValueError:
                    continue

            if any(quantity in noun[:-1] for quantity in tuple[1]): # '[a, state, incentive, grant, worth, million]'
                if len(noun[:-1])-1 <= len(noun[:-1])/2:
                    try:
                        nouns.remove(noun)
                    except ValueError:
                        continue
                else:
                    noun.remove(next(i for i in noun[:-1] if i in tuple[1]))

        # remove (single) nouns that are contained in some of the others (=less informative)
        for noun in nouns_copy:
            for _, noun_2 in itertools.product(noun, nouns_copy[:nouns_copy.index(noun):] + nouns_copy[nouns_copy.index(noun)+1:]):
                if set(t.i for t in noun[:-1]).intersection(set(t.i for t in noun_2[:-1])): # looking at all neighbours
                    if len(noun) < len(noun_2):
                        try:
                            nouns.remove(noun)
                        except ValueError:
                            continue
                    else:
                        try:
                            nouns.remove(noun_2)
                        except ValueError:
                            continue

        # remove nouns that consist only of stopwords
        for noun in nouns_copy:
            if len(self._remove_stopwords_from_noun(noun[:-1])) == 0:
                try:
                    nouns.remove(noun)
                except ValueError:
                    continue

        # change the index
        for noun in nouns:
            noun[-1] = int((noun[0].i + noun[-2].i)/2) # take the median

    # noinspection PyProtectedMember
    def _tag_brackets(self, doc): # e.g. "It drops to just 87 mph (140 km/h)..."
        # This does not work for complex cases like nested brackets, but is fine for the majority of sentences
        expression = r"\(([^\(]*)\)"
        for match in re.finditer(expression, doc.text):
            start, end = match.span()
            span = doc.char_span(start, end)
            if span:
                for token in span:
                    if token.text not in ["(", ")"]:
                        token._.set("in_bracket", True)
        return doc

    # noinspection PyProtectedMember
    def _tag_commas(self, doc): # e.g. "It slid 88.6 points, or 1.6 per cent, to 5557.8 while ..."
        expression = r",([^\,]*),"
        for match in re.finditer(expression, doc.text):
            start, end = match.span()
            span = doc.char_span(start, end)
            if span:
                for token in span:
                    if token.text not in [","]:
                        token._.set("in_comma", True)
        return doc

    # noinspection PyProtectedMember
    def _tag_ignore(self, doc):
        phone_number_like = r"^[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}$"
        sequence_number_like = r"-?(?:\d+\-){2,}\d+" # 5-4-10-12-1, 18-11-9-5, 5-2-2
        full_time_like = r"\d{1,2}[\s]?:[\s]?\d{0,2}[:.][\s]?\d{0,2}" # 15:06:40, 2:45.5
        time_like = r"\d{1,2}[\s]?:[\s]?\d{1,2}" # 2:00
        date_like = r"(Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sept|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?).{0,1}\s{0,1}(\d{1,2}-{0,1}\s{0,1}\d{0,2})"
        full_date_like = r"(\d{4})-(\d{2})-(\d{2})" # 2019-09-05
        name_like = r"[a-zA-Z]+\d+[.|,]*\d+[a-zA-Z]*" # PS46.6m
        ordinal = r"\d+\s*(?:st|nd|rd|th)"
        pattern = [phone_number_like, sequence_number_like, full_time_like, time_like, date_like, full_date_like, name_like, ordinal]
        for expr in pattern:
            for match in re.finditer(expr, doc.text):
                start, end = match.span()
                span = doc.char_span(start, end)
                if span is not None:
                    for token in span:
                        token._.set("ignore", True)
        for ent in doc.ents:
            if ent.label_ in ["DATE", "TIME"]:
                span = doc.char_span(ent.start_char, ent.end_char)
                if span is not None:
                    if any(month in span.text for month in MONTHS):
                        for token in span:
                            token._.set("ignore_noun", True) # ignore months as nouns, e.g November but mid-November as well
                    if {"year", "month", "week", "day", "second", "minute", "hour", "daily"} & {sp.lemma_ for sp in span} and not {"this", "that"} & {sp.lemma_ for sp in span}: # ignore "2011 this year"
                        continue
                    if len(span)==1 and span[0].text.isnumeric() and len(span[0].text)>4: # if the number has more than 4 digits (e.g. 2483805) it is not a date.
                        continue
                    if ent.label_=="DATE" and any(t.text.isdigit() or t.pos_ == "NUM" for t in span):
                        for token in span:
                            token._.set("consider_date", True) # extract DATE entities, keep them only if they have proper unit
                    else:
                        for token in span:
                            token._.set("ignore", True)

        self.pattern_matcher(doc) # phone numbers, zip code

        expression = r"\+\s?\d{2}\s?[(]?\d{4}[)]?\s?\d{4,7}\b" # + AA (AAAA) BBBBBBB (with or without space and brackets)
        # [\+0{2}]\s?[(-\[]?\d{2,3}[-)\]]?\s?[(]?\d{3,4}[)]?[\s-\/]?[\d\s]{1,11}
        # e.g. + AAA BBBB, + AAA AAA AAA AAA AAA, 00AA-AAAABBBBBBB, +([AA)]AAAABBBBBBB, +AAAAAAAA/BBBBBBB etc.
        for match in re.finditer(expression, doc.text):
            start, end = match.span()
            span = doc.char_span(start, end)
            if span is not None:
                for token in span:
                    token._.set("ignore", True)

        return doc

    def _retokenize(self, doc): # 'km', '/', 'h' (from km/h) -> 'km/h'
        with doc.retokenize() as retokenizer:

            # retokenize the fractions from number_lookup.py
            for k, v in maps["fractions"].items(): # e.g. half, third
                expression = rf"\b{k}\b"
                for match in re.finditer(expression, doc.text, re.IGNORECASE):
                    start, end = match.span()
                    span = doc.char_span(start, end)
                    if span:
                        retokenizer.merge(span, attrs={"POS": "NUM"})

            # retokenize the number strings from number_lookup.py
            for k, v in maps["string_num_map"].items(): # e.g. twenty-six
                expression = rf"\b{k}\b"
                for match in re.finditer(expression, doc.text, re.IGNORECASE):
                    start, end = match.span()
                    span = doc.char_span(start, end)
                    if span:
                        retokenizer.merge(span, attrs={"POS": "NUM"})

            # retokenize the scientific notation format
            expression = r"[0-9,.]*[0-9]+[×|x]+10\^?[−|-]?\d+" # e.g. 1.38×10^10
            for match in re.finditer(expression, doc.text):
                start, end = match.span()
                span = doc.char_span(start, end)
                if span:
                    try:
                        retokenizer.merge(span, attrs={"POS": "NUM"})
                    except ValueError:
                        logging.warning('\033[93m' + f"Can't merge non-disjoint spans. '{span}' is already part of tokens to merge." + '\033[0m')

            # retokenize the comma-separated format for numbers (decimal and thousands separator)
            expression = r"\b\d{1,3}(,\d{3,})+(\.\d+)?\b" # e.g. 34,567.89, 64,863.10
            for match in re.finditer(expression, doc.text):
                start, end = match.span()
                span = doc.char_span(start, end)
                if span:
                    try:
                        retokenizer.merge(span, attrs={"POS": "NUM"})
                    except ValueError:
                        logging.warning('\033[93m' + f"Can't merge non-disjoint spans. '{span}' is already part of tokens to merge." + '\033[0m')

            # retokenize fractions
            expression = r"\b\d{1}[/|⁄]+\d+\b" # e.g. 1/16 of a US pint, 1⁄32 of a US quart, and 1⁄128 of a US gallon
            for match in re.finditer(expression, doc.text):
                start, end = match.span()
                span = doc.char_span(start, end)
                if span:
                    try:
                        retokenizer.merge(span, attrs={"POS": "NUM"})
                    except ValueError:
                        logging.warning('\033[93m' + f"Can't merge non-disjoint spans. '{span}' is already part of tokens to merge." + '\033[0m')

            expression = r"\w+\/\w+"
            for match in re.finditer(expression, doc.text):
                start, end = match.span()
                span = doc.char_span(start, end)
                if span:
                    try:
                        retokenizer.merge(span, attrs={"POS": "NOUN"})
                    except ValueError:
                        logging.warning('\033[93m' + f"Can't merge non-disjoint spans. '{span}' is already part of tokens to merge." + '\033[0m')

            expression = r"\bsq\s(m|ft)\b" # e.g. sq m, sq ft
            for match in re.finditer(expression, doc.text):
                start, end = match.span()
                span = doc.char_span(start, end)
                if span:
                    try:
                        retokenizer.merge(span, attrs={"POS": "NOUN"})
                    except ValueError:
                        logging.warning('\033[93m' + f"Can't merge non-disjoint spans. '{span}' is already part of tokens to merge." + '\033[0m')

            expression = r"degrees ([C|c]+el[s|c]+ius|[F|f]a(h)?renheit)+\b"
            for match in re.finditer(expression, doc.text):
                start, end = match.span()
                span = doc.char_span(start, end)
                if span:
                    try:
                        retokenizer.merge(span, attrs={"POS": "NOUN"})
                    except ValueError:
                        logging.warning('\033[93m' + f"Can't merge non-disjoint spans. '{span}' is already part of tokens to merge." + '\033[0m')

            expression = r"°[C|c]+(elsius|elcius|entigrade)?\b"
            for match in re.finditer(expression, doc.text):
                start, end = match.span()
                span = doc.char_span(start, end)
                if span:
                    try:
                        retokenizer.merge(span, attrs={"POS": "NOUN"})
                    except ValueError:
                        logging.warning('\033[93m' + f"Can't merge non-disjoint spans. '{span}' is already part of tokens to merge." + '\033[0m')

            expression = r"¥"
            for match in re.finditer(expression, doc.text):
                start, end = match.span()
                span = doc.char_span(start, end)
                if span:
                    try:
                        retokenizer.merge(span, attrs={"POS": "SYM"})
                    except ValueError:
                        logging.warning('\033[93m' + f"Can't merge non-disjoint spans. '{span}' is already part of tokens to merge." + '\033[0m')

            # retokenize negative number
            expression = r"[\b|\s](-)([\d,.,,]+)\b" # e.g. '-5'
            for match in re.finditer(expression, doc.text):
                start, end = match.span()
                span = doc.char_span(start, end)
                if span:
                    retokenizer.merge(span, attrs={"POS": "NUM"})

        return doc

    def _number_in_unit(self, input):
        return any(is_digit(char) for char in input)

    # check for shared units and assign missing unit
    def _shared_unit_check(self, doc, boundaries, unitless, triples):
        if not boundaries:
            return unitless

        bound_1 = [bound for bound in boundaries if bound[0]<=unitless[-1]<=bound[1]][0]
        bounds = [bound for bound in boundaries for i in [triple[-1] for triple in triples] if bound[0]<=i<=bound[1]]
        # reverse so that the more near subsentences to the one consisting the unitless quantity are considered first
        bounds.reverse()
        triples.reverse()

        # quantities that are close to each other may also have the same unit, e.g., "It costs about 8 or $9."
        for triple in triples:
            # same subsentence, distance and order
            # because of sentences like: "Australias S&P/ASX 200 costs 0.7 dollar or 6,327.80." or "Pluristem rose 8.15% to 3.65 after reporting a positive meeting"
            if bound_1[0]<=triple[-1]<=bound_1[1] and 1 < (min(triple[1][0].i, triple[2][0].i)-unitless[1][0].i) <= 3:
                # check for punct or verb pos_ tag
                # e.g. It has an MSRP of 2,435,000 YEN ($22,480) for the 5 speed manual transmission version and 2,380,000 ($21,972) for the CVT.
                # e.g. 125 million australian dollars of zero - coupon eurobonds due dec. 12, 1994, priced at 50.9375 to yield 15.06 % less fees via hambros bank ltd.
                if not {"VERB","PUNCT"} & set(token.pos_ for token in doc[unitless[1][0].i+1:min(triple[1][0].i, triple[2][0].i)]):
                    logging.info('\033[93m' + f"Shared unit found in \"{doc}\" for {unitless} and {triple}" + '\033[0m\n')
                    unitless[2] = triple[2] # assign unit
                    return tuple(unitless)

        # similar clause structure
        for i, bound_2 in enumerate(bounds):
            if bound_1 != bound_2 and (bound_1[-1]+1 == bound_2[0] or bound_2[-1]+1 == bound_1[0]): # look only at different but adjacent subsenetences
                if doc[bound_1[0]].pos_ in ["SCONJ"]:
                    doc_1 = doc[bound_1[0]+1:bound_1[1]]
                else:
                    doc_1 = doc[bound_1[0]:bound_1[1]]
                doc_2 = doc[bound_2[0]:bound_2[1]]
                pos_1 = [tok.pos_ for tok in doc_1 if tok.pos_ not in ["PUNCT"]]
                pos_2 = [tok.pos_ for tok in doc_2 if tok.pos_!="PUNCT"]
                ratio = numpy.average([fuzz.token_sort_ratio(" ".join(pos_1), " ".join(pos_2)), fuzz.partial_ratio(" ".join(pos_1), " ".join(pos_2)), fuzz.ratio(" ".join(pos_1), " ".join(pos_2))])
                # consider ratio, token like 'while', 'whilst' or 'whereas' and the position of the quantity in the sentence
                if ratio == 100 or (ratio >= 58 and (set(t.text for t in doc) & set(["while", "whilst", "whereas", "but", "and"]))) and (abs((bound_1[-1]-unitless[1][0].i) - (bound_2[-1]-triples[-i][1][0].i)) < 3):
                    logging.info('\033[93m' + f"Shared unit found in \"{doc}\" for {unitless} and {triples[-i]}" + '\033[0m\n')
                    unitless[2] = triples[-i][2] # assign unit
                    return tuple(unitless)

        return unitless # no shared unit found


    # noinspection PyProtectedMember
    def _match_tuples(self, candidates, doc, boundaries):
        triples = []
        bound = []
        quantity = []
        unit = []
        candidates_proned=[]

        for i, candidate in enumerate(candidates):
            if ", x," not in str(candidate[:-1]):
                candidates_proned.append(candidate)
            else:
                #for dimensions add them each seperately
                units=[element for element in candidate[:-1] if not is_digit(re.sub('[.,,]', '', element.text)) and element.text!="x"]
                for element in candidate[:-1]:
                    if is_digit(re.sub('[.,,]', '', element.text)):
                        if [element]+units not in candidates_proned:
                            candidates_proned.append([element]+units+[i])

        for candidate in candidates_proned:
            b_list = []
            q_list = []
            u_list = []
            broad_range = False
            for token in candidate[:-1]:
                if token._.bound:
                    b_list.append(token)
                elif token._.number and not broad_range:
                    q_list.append(token)
                    if token._.broad_range:
                        broad_range = True
                else:
                    if token.text not in ('x', '-'): # add != "-" because of ranges like 0-60mph -> [0, 60], [mph]
                        u_list.append(token)
                if len(q_list)>1 and len(b_list)>0:
                    for q_element in q_list.copy():
                        if q_element.i + 1 == b_list[0].i and not any(prep == b.text for prep in ["to", "out", "of", "-"] for b in b_list): # distinguish ranges and fractions
                            q_list.remove(q_element)

            if len(u_list) > 2 and len([t_unit for t_unit in u_list if t_unit.pos_ == "PROPN"]) >= len(u_list)/2 and not any(t_unit for t_unit in u_list if t_unit.pos_ == "SYM" or len(t_unit.text)<=3):
                # e.g. 575 Wilbraham Road, but 'US $', '30 Mbps plan' or '2.4 Ghz' should be extracted
                #q_list = [] # no quantity => no extraction
                #continue
                for token in q_list:
                    token._.set("consider", True)

            possible_compound_value = False

            if not all(q._.first_compl_num or q._.second_compl_num for q in q_list):

                if len([q for q in q_list if q.text in (maps["scales"].keys() | maps["string_num_map"].keys())]) >= 3:
                    possible_compound = [q for q in q_list if q.text in (maps["scales"].keys() | maps["string_num_map"].keys())]
                    if sorted([token.i for token in possible_compound]) == list(range(min(token.i for token in possible_compound), max(token.i for token in possible_compound)+1)):
                        possible_compound_value = True # e.g. twenty-eight thousand six hundred forty-two

                if not possible_compound_value and (len(q_list) == 3 and all(is_digit(q.text) for q in q_list) or len([q for q in q_list if is_digit(q.text) or q.text.replace(".", "").replace(",","").isdigit() or q.text in (maps["fractions"].keys() | maps["string_num_map"].keys())])==3): # e.g. [160, 180, 200] or [3, 4, 5]
                    range_l = [q for q in q_list if q._.range_l]
                    range_r = [q for q in q_list if q._.range_r]
                    if range_l and range_r:
                        if len(range_l) > len(range_r): # 180-200 and 160-200
                            new_candidate = candidate.copy() # [from, 160, to, $, 180, 200, 20]
                            new_candidate.remove(range_l[1]) # [from, 160, to, $, 200, 20]
                            candidate.remove(range_l[0]) # [from, to, $, 180, 200, 20]
                            candidates_proned.append(new_candidate) # [[from, to, $, 180, 200, 20], [$, 220, 38], [$, 300, 41]] -> [[from, to, $, 180, 200, 20], [$, 220, 38], [$, 300, 41], [from, 160, to, $, 200, 20]]
                            q_list.remove(range_l[0]) # [160, 180, 200] -> [180, 200]

                        elif len(range_l) < len(range_r): # 180-160 and 180-200
                            new_candidate = candidate.copy()
                            new_candidate.remove(range_r[1])
                            candidate.remove(range_r[0])
                            candidates_proned.append(new_candidate)
                            q_list.remove(range_r[0])

                        elif len(range_l)==len(range_r)==2: # e.g. [3, 4] [4, 5]
                            new_candidate = candidate.copy() # [3, to, 4, 5, percent, pressure, 3]
                            new_candidate.remove(range_r[1]) # [3, to, 4, percent, pressure, 3]
                            candidate.remove(range_l[0]) # [to, 4, 5, percent, pressure, 3]
                            candidates_proned.append(new_candidate)
                            q_list.remove(range_l[0])

                        """elif len(range_l)==len(range_r)==1:
                            print(candidate, candidates_proned)
                            min_index = min(token.i for token in range_l+range_r)
                            min_index_token = [token for token in range_l+range_r if token.i == min_index][0]
                            candidate = candidate[candidate.index(min_index_token)]
                            print(candidate, candidates_proned)"""

                    elif not range_l and not range_r: # e.g. we have to maintain this for 40, 50, 60 years
                        new_candidate = candidate.copy() # [40, 50, 60, years, 17]
                        new_candidate_2 = candidate.copy()
                        candidate.remove(q_list[1])
                        candidate.remove(q_list[2])
                        new_candidate.remove(q_list[2])
                        new_candidate.remove(q_list[0])
                        new_candidate_2.remove(q_list[0])
                        new_candidate_2.remove(q_list[1])
                        candidates_proned.append(new_candidate)
                        candidates_proned.append(new_candidate_2) # [([], [40], [years], 17), ([], [50], [years], 17), ([], [60], [years], 17)]
                        q_list = q_list[:1]

                elif len(q_list) >= 4 and len([is_digit(q.text) for q in q_list if is_digit(q.text) or q.text.replace(".", "").replace(",", "").isdigit()]) == 4:
                    values = sorted([q for q in q_list if is_digit(q.text) or q.text.replace(".", "").replace(",", "").isdigit()], key=lambda t: t.i) # [2,500, 250, 50, 10]
                    scales = sorted([q for q in q_list if q not in values]) # [million, million, million, million]
                    new_candidate = candidate.copy()
                    if len(values) == len(scales):
                        [new_candidate.remove(value) for value in values[:2]+scales[:2]] # [to, $, 20, million, to, $, 24, million, 34]
                        [candidate.remove(value) for value in values[2:]+scales[2:]] # [46, million, to, $, 50, million, to, $, 34]
                        candidates_proned.append(new_candidate) # [[46, million, to, $, 50, million, to, $, 34], [to, $, 20, million, to, $, 24, million, 34]]
                        [q_list.remove(value) for value in values[2:]+scales[2:]] # [46, million, 50, million]
                    else:

                        [new_candidate.remove(value) for value in values[:2]] # [from, to, 50, no, more, 10, 17]
                        [candidate.remove(value) for value in values[2:]] # [from, 2,500, to, 250, no, more, 17]
                        candidates_proned.append(new_candidate) # [[from, 2,500, to, 250, no, more, 17], [from, to, 50, no, more, 10, 17]]
                        [q_list.remove(value) for value in values[2:]] # [2,500, 250]

            if len(u_list) == 0 and q_list and not any(q._.consider or q._.consider_date for q in q_list) and [triple[2] for triple in triples if triple[2] and triple[1]]: # unitless and potential shared unit
                triple = self._shared_unit_check(doc, boundaries, [b_list, q_list, u_list, candidate[-1]], [triple for triple in triples if triple[2] and triple[1]])
                triples.append((triple[0], triple[1], triple[2], triple[3]))
            else:
                triples.append((b_list, q_list, u_list, candidate[-1])) # 4.element = index of value in doc in order to check for referred_noun later

            if b_list:
                bound.append(b_list)
            if q_list:
                quantity.append(q_list)
            if u_list:
                unit.append(u_list)

        logging.info("Bounds:")
        logging.info(bound)
        logging.info("Quantities:")
        logging.info(quantity)
        logging.info("Unit")
        logging.info(unit)
        return triples

    def _extract_global_referred_nouns(self, doc): # look for noun referred by the value
        referred_nouns = []
        referred_nouns = [[token, token.i] for token in doc if not token._.ignore_noun and token.dep_=="ROOT" and token.pos_ in ["NOUN", "PROPN"]]
        root_flag = len(referred_nouns) # appropriate noun (root of the sentence) found

        referred_nouns = self._extend_root(referred_nouns)

        referred_nouns.extend([[token, token.i] for token in doc if not token._.ignore_noun and \
                               (token.dep_ in ["nsubj", "nsubjpass", "dobj"] and token.pos_ in ["NOUN", "PROPN", "PRON", "ADJ", "DET", "ADV"]) #or\
                               #(token.dep_ in ["pobj"] and token.pos_ in ["NOUN", "PROPN"])
                               ])

        if referred_nouns:
            if root_flag:
                referred_nouns[root_flag:] = self._extend_nouns(referred_nouns[root_flag:]) # do not traverse the ROOT children, since they are usually many
            else:
                referred_nouns = self._extend_nouns(referred_nouns)

        return referred_nouns

    def _extend_root(self, root_tokens):
        def accept_root_children(token):
            return token.pos_ in ["NOUN","PROPN", "DET", "ADJ"]
        childrens = list(sorted(list(token.children)+[token], key=lambda t: t.i)+[tokens[-1]] for tokens in root_tokens for token in tokens[:-1])
        return list(list(itertools.takewhile(accept_root_children, child[:-1]))+[child[-1]] for child in childrens)

    # extend referred_nouns by additional tokens
    def _extend_nouns(self, referred_nouns):
        if referred_nouns is None:
            return referred_nouns

        def traverse_children(extended_list, children, level): # traverse all children
            for child in children:
                if list(child.children) and level < 4:
                    level += 1
                    extended_list.extend(list(c for c in child.children if c.pos_ not in ["NUM", "PUNCT", "SYM", "VERB"] and c.dep_ != "quantmod" or (c.pos_ in ["VERB"] and c.dep_ in ["amod"]))) # ignore digits, punctuations, verbs and quantmod
                    traverse_children(extended_list, child.children, level)
                else:
                    continue

        for i, referred_noun in enumerate(referred_nouns):
            level = 0
            for ref_noun in referred_noun[:-1]:
                if list(ref_noun.children): # find children to make noun more informative
                    level += 1
                    children = list(ref_noun.children)
                    filtered_children = [child for child in children if child.pos_ not in ["PUNCT", "SYM", "VERB"] or (child.pos_ in ["VERB"] and child.dep_ in ["amod"])] # if child.dep_ in ["det", "compound", "amod", "nmod", "conj", "prep"] or child.pos_ in ["NUM","NOUN","ADJ"]] # prep, nummod ?
                    extended_children = []
                    extended = False
                    for child in filtered_children: # e.g. [Group, Johnson] -> [UnitedHealth, Group, Inc, Johnson, &, Johnson]
                        extended_children.append(child)
                        if list(child.children):
                            level += 1
                            extended_children.extend(list(c for c in child.children if c.pos_ not in ["NUM", "PUNCT", "SYM", "VERB"] and c.dep_ != "quantmod" or (c.pos_ in ["VERB"] and c.dep_ in ["amod"]))) # ignore digits, punctuations, verbs and quantmod
                            extended = True
                            traverse_children(extended_children, child.children, level)
                    if extended:
                        extended_children.append(ref_noun) #, referred_noun[-1]])
                        extended_children =  list(dict.fromkeys(extended_children)) # eliminate duplicates
                        referred_nouns[i][:-1] += extended_children
                    else:
                        filtered_children.append(ref_noun) #, referred_noun[-1]])
                        filtered_children =  list(dict.fromkeys(filtered_children)) # eliminate duplicates
                        referred_nouns[i][:-1] += filtered_children

            referred_nouns[i][:-1] = sorted(list(dict.fromkeys(referred_nouns[i][:-1])), key=lambda t: t.i) # sort tokens index

        return referred_nouns

    def _extract_quantifiers(self, doc, tuple):
        related_nouns = []
        flag = False # to check if there was 'of' in the tuples

        if tuple[1]:
            index = 0
            if tuple[1][0].i + 1 < len(doc) and doc[tuple[1][0].i + 1].text == "(": # e.g. '2,380,000 ($21,972) for the CVT'
                index = 2
                while tuple[1][0].i + index < len(doc) and doc[tuple[1][0].i + index]._.in_bracket:
                    index += 1
            if index == 0:
                quantity_index = max([t.i for t in tuple[1]]+[t.i for t in tuple[2]]) # e.g. 30 million shares up for sale
            else:
                quantity_index = max([t.i for t in tuple[1]]+[t.i for t in tuple[2][:index+1]])
            distances = [(token.i - (quantity_index + index)) for token in doc if token.text in ["for", "of", "off"]] # e.g. '$52,000 for CN'

            if distances and not self.ignore_some_rules:
                j = min(abs(dist) for dist in distances) if {1,2} & set(distances) else 0
                if j and doc[quantity_index + index + j] not in tuple[2]:
                    if [child for child in doc[quantity_index + index + j].children if (child.pos_ not in ["NUM"] and not child._.ignore_noun and not child in tuple[2] and not child.text.lower() in self.nlp.Defaults.stop_words)]: # only if there are children, otherwise ignore
                        related_nouns.append(list(doc[quantity_index + index + j].children))
                        related_nouns[-1].append(quantity_index + index + j)
                        related_nouns[-1][:-1] = sorted(related_nouns[-1][:-1], key=lambda t: t.i)
                        flag = True

            if flag:
                related_nouns = self._extend_nouns(related_nouns)
                #return related_nouns # commented out because we continue looking for a second related quantifier

            distances = [(tuple[1][0].i - token.i) for token in doc if token.text in ["of"]] # e.g. 'an increase of 5 percent'
            if distances:
                flag = False
                j = min(abs(dist) for dist in distances) if {1,2,3} & set(distances) else 0
                if j and doc[tuple[1][0].i-j-1].pos_ in ["NOUN", "PROPN"] and doc[tuple[1][0].i-j-1].text not in self.nlp.Defaults.stop_words and list(doc[tuple[1][0].i-j-1].subtree):
                    related_nouns.append([])
                    flag = True
                    for child in doc[tuple[1][0].i-j-1].subtree:
                        if child.i <= tuple[1][0].i:
                            if child._.ignore: # e.g. DATE
                                related_nouns = []
                                flag = False
                                break
                                #return [] # e.g. ... growth rate since the mid 1970s of 1.7 per cent
                            related_nouns[-1].append(child)
                if related_nouns and flag:
                    related_nouns[-1].append(tuple[1][0].i-j-1)
                    related_nouns[-1][:-1] = sorted(related_nouns[-1][:-1], key=lambda t: t.i)
                    flag = True

            # 'for'-pattern: if there exists, 'for <MONEY_SYMBOL>' in the text, then the noun before 'for' is the concept
            if not flag and "for" in list(token.text for token in tuple[1][0].ancestors) and (any(token.start <= tuple[1][0].i <= token.end for token in [ent for ent in doc.ents if ent.label_ == "MONEY"]) or set(t.text for t in tuple[2]) & MONEY_SYMBOL):
                pot_token = [t.head.head.head for t in tuple[2]]
                if pot_token:
                    if pot_token[0].pos_ in ["VERB", "AUX"]: # a seat was sold for down $ 5,000
                        if any(child.pos_ in ["VERB", "AUX"] for child in list(sorted(list(pot_token[0].children)))) and not any(child.pos_ in ["NOUN", "PROPN", "ADJ"] for child in list(sorted(list(pot_token[0].children)))):
                            pot_token = [child for child in list(sorted(list(pot_token[0].children))) if child.pos_ in ["VERB", "AUX"]] # it bought odwalla inc. for $ 186 million
                        token = [token for token in list(sorted(list(pot_token[0].children), key=lambda t: abs(tuple[1][0].i-t.i))) if token.pos_ in ["NOUN","PROPN","ADJ"]]
                        if token:
                            related_nouns.append(list(token[0].subtree))
                            related_nouns[-1].append(token[0].i)
                    else:
                        related_nouns.append(list(itertools.takewhile(lambda t: t.text != "for", list(pot_token[0].subtree))))
                        related_nouns[-1].append(pot_token[0].i)
                    flag = True

            if not flag and {"by"} & set(token.head.text for token in tuple[2]): # Burlington by 2.6%
                j = [token.head.text for token in tuple[2]].index("by")
                if doc[tuple[2][j].head.i-1].pos_ in ["NOUN", "PROPN"]:
                    related_nouns.append(list(doc[tuple[2][j].head.i-1].subtree))
                    related_nouns[-1].append(doc[tuple[2][j].head.i-1])
                    flag = True

            if not flag and {"at"} & set(token.head.text for token in tuple[2]): # with carbon levels at 1200 parts per million
                j = [token.head.text for token in tuple[2]].index("at")
                if doc[tuple[2][j].head.i-1].pos_ in ["NOUN", "PROPN"]:
                    related_nouns.append(list(doc[tuple[2][j].head.i-1].subtree))
                    related_nouns[-1].append(doc[tuple[2][j].head.i-1])
                    flag = True

            if not flag and {"to"} & set(token.head.text for token in tuple[2]): # 19% drop for the MSCI Europe index to 1,350 points
                j = [token.head.text for token in tuple[2]].index("to")
                if doc[tuple[2][j].head.i-1].pos_ in ["NOUN", "PROPN"]:
                    related_nouns.append(list(doc[tuple[2][j].head.i-1].subtree))
                    related_nouns[-1].append(doc[tuple[2][j].head.i-1])
                    flag = True

            if not flag and {"with"} & set(token.head.text for token in tuple[2]): # iPhone 12 Pro Max with 3687 mAh
                j = [token.head.text for token in tuple[2]].index("with")
                if doc[tuple[2][j].head.i-1].pos_ in ["NOUN", "PROPN"]:
                    related_nouns.append(list(doc[tuple[2][j].head.i-1].subtree))
                    related_nouns[-1].append(doc[tuple[2][j].head.i-1])
                    flag = True

        return related_nouns


    def _find_anchor(self, anchors, element):
        for anchor, values in anchors.items():
            if element in values:
                return anchor


    def _extract_candidates(self, doc):
        matches = self.matcher(doc)
        #print([(self.nlp.vocab[match[0]].text, [doc[i] for i in match[1]]) for match in matches])
        anchor = {}
        #for match in matches:
        #    print('---'.join([doc[token_id].text for token_id in match[1]]))

        for match in matches:# self.nlp.vocab[11548777916544960358].text
            #print(self.nlp.vocab[match[0]].text, [doc[i] for i in match[1]])
            #print(match)

            # filter out things like iphone 11, FTSE 100
            if self.nlp.vocab[match[0]].text in ["NOUN_NUM", "NUM_QUANTMOD"]: # num_quantmod because of e.g. "... in the Big 12."
                digit_part= doc[match[1][0]].text if is_digit(re.sub('[.,,]', '', doc[match[1][0]].text)) else doc[match[1][1]].text
                char_part = doc[match[1][0]].text if not is_digit(re.sub('[.,,]', '', doc[match[1][0]].text)) else doc[match[1][1]].text

                # about 1000 (NUM_QUANTMOD) should not be ignored as well as currencies or age 87
                if (char_part+" "+digit_part in doc.text and char_part.lower() not in maps["bounds"] and char_part.lower() not in ["age", "between", "than", "from"]) or char_part in ["GMT", "index", "Max"]:
                    doc[match[1][1]]._.set("ignore", True) # not extract 11 (the number, NUM) from 'iphone 11' as Quantity
                    doc[match[1][0]]._.set("ignore", True) # e.g. Big 12
                    continue

            if self.nlp.vocab[match[0]].text in ["NUM_DIRECT_PROPN"]:
                char_part = doc[match[1][0]].text if not is_digit(re.sub('[.,,]', '', doc[match[1][0]].text)) else doc[match[1][1]].text
                if char_part in ["Max", "Pro", "Mini"]: # to filter out things like iphone 11 Pro
                    doc[match[1][0]]._.set("ignore", True)
                    continue

            # to filter out not real quantities, e.g. 'sp 500', 'BBC One'
            if self.nlp.vocab[match[0]].text == "LONELY_NUM" and match[1][0]-1 >= 0 and match[1][0]+1 < len(doc): # add index check
                if (doc[match[1][0]-1].pos_ == "NOUN" and doc[match[1][0]+1].pos_ in ["VERB", "AUX"]) or doc[match[1][0]-1].pos_ == "PROPN":
                    continue

            if not set(doc[token].text for token in match[1]) & {"between", "from"} and abs(max(match[1]) - min(match[1])) > 5 and not any(doc[token]._.in_bracket for token in range(min(match[1]),max(match[1])+1)): # e.g. 9 billion shekels ($2.408 billion) a year
                continue

            for token_id in match[1]:
                if not doc[token_id]._.ignore:

                    entries = set()
                    for v in anchor.values():
                        entries.update(v)
                    intersection = (set(match[1]) & entries)
                    if doc[token_id]._.number and not doc[token_id]._.ignore and not intersection:
                        anchor[token_id] = set(match[1])
                        break
                    if token_id in anchor:
                        anchor[token_id].update(set(match[1]))
                    elif intersection:
                        id = self._find_anchor(anchor, next(iter(intersection)))
                        anchor[id].update(set(match[1]))

        result = []
        for k, candidate in anchor.items():
            candidate_res = []
            for token_id in sorted(list(candidate)):
                candidate_res.append(doc[token_id])
            candidate_res.append(k) # append index of value in order to check for referred_noun later
            result.append(candidate_res)

        return result


    # noinspection PyProtectedMember
    def _normalize_tuples(self, tuples, nouns, text, doc):
        result = []
        for tuple in tuples:
            # if there is no number there is no point in normalizing
            if tuple[1]:
                fraction = False
                negative_number = False
                scale_m = False

                # 'minus'-approach: when extracted as unit
                if len(tuple[1]) == 1 and any(token.text == "minus" for token in tuple[2]):
                    tuple[2].remove(tuple[2][[token.text == "minus" for token in tuple[2]].index(True)])
                    negative_number = True

                # 'minus'-approach: when extracted as bound
                if len(tuple[1]) == 1 and any(token.text == "minus" for token in tuple[0]):
                    tuple[0].remove(tuple[0][[token.text == "minus" for token in tuple[0]].index(True)])
                    negative_number = True

                # 'm'-approach
                if len(tuple[2]) > 1 and tuple[2][0].text == "m" and any(token.pos_ in ["NOUN", "PROPN"] for token in tuple[2][1:]): # e.g. 1.2m tons, 60 to 90m subscribers
                    tuple[2].remove(tuple[2][0])
                    scale_m = True

                if any(t._.first_compl_num for t in tuple[1]) and any(t._.second_compl_num for t in tuple[1]):
                    left_num = [t for t in tuple[1] if t._.first_compl_num]
                    right_num = [t for t in tuple[1] if t._.second_compl_num]
                    left_unit = [t for t in tuple[2] if t._.first_compl_num]
                    right_unit = [t for t in tuple[2] if t._.second_compl_num]
                    norm_number_l = NumberNormalizer.normalize_number_token(left_num)
                    norm_number_r = [NumberNormalizer.normalize_number_token([r]) for r in right_num]
                    norm_unit_l = NumberNormalizer.normalize_unit_token(left_unit, text)[0]
                    norm_unit_r = [NumberNormalizer.normalize_unit_token([r], text)[0] for r in right_unit]

                    if norm_number_l is None or norm_number_r is None:
                        continue

                    num = norm_number_l
                    for num_r, unit_r in zip(norm_number_r, norm_unit_r):
                        if unit_r in unit_conversion and norm_unit_l in unit_conversion[unit_r]: # because of wrong disambiguatio of pound to pound sterling
                            num += num_r*unit_conversion[unit_r][norm_unit_l]

                    number = Value(num, tuple[1], self.overload)
                    for token in right_unit:
                        tuple[2].remove(token)

                elif any(t._.range_l for t in tuple[1]) and any(t._.range_r for t in tuple[1]):

                    left = [t for t in tuple[1] if t._.range_l]
                    right = [t for t in tuple[1] if t._.range_r]
                    if left and right: # e.g. [24.2, 33] [33]
                        if len(left)>len(right) and any(t in right for t in left):
                            left.remove(left[[t in right for t in left].index(True)])
                        elif len(left)<len(right) and any(t in left for t in right):
                            right.remove(right[[t in left for t in right].index(True)])
                    scales = [t for t in tuple[1] if t._.scale]
                    norm_number_l = NumberNormalizer.normalize_number_token(left)
                    norm_number_r = NumberNormalizer.normalize_number_token(right)
                    number_l = None
                    number_r = None

                    # check for successful normalization
                    if norm_number_l is None and norm_number_r is None:
                        #print(text)
                        continue  # or try something else? same for the other checks below
                    if scales:
                        norm_number_scale = 1
                        if len(scales) == 2: # [160, million, 171, million]
                            norm_number_scale = NumberNormalizer.normalize_number_token([scales[0]])
                            number_l = norm_number_l * norm_number_scale
                            norm_number_scale = NumberNormalizer.normalize_number_token([scales[-1]])
                            number_r = norm_number_r * norm_number_scale
                            scales = []
                        else:
                            for scale in scales:
                                norm_number_scale *= NumberNormalizer.normalize_number_token([scale])
                    else:
                        norm_number_scale = 1

                    # check if number token was successfully normalized
                    if norm_number_l is None:
                        norm_number_l = norm_number_r
                    if norm_number_r is None:
                        norm_number_r = norm_number_l

                    # check if already scaled (e.g. 'from the US $89 to $93 billion')
                    if scales and not any(scale in left for scale in scales) and abs(norm_number_r/norm_number_scale - norm_number_l) <= 1000:
                        number_l = norm_number_l * norm_number_scale #Value(norm_number_l * norm_number_scale, left)
                    elif number_l is None:
                        number_l = norm_number_l
                    if scales and not any(scale in right for scale in scales) and abs(norm_number_r - norm_number_l/norm_number_scale) <= 1000:
                        number_r = norm_number_r * norm_number_scale #Value(norm_number_r * norm_number_scale, right)
                    elif number_r is None:
                        number_r = norm_number_r

                    #if norm_number_l == norm_number_r:
                    #    number = number_l
                    #else:
                    #    number = Range(number_l, number_r)
                    if not isinstance(number_l, list) and not isinstance(number_r,list):
                        number = Range(number_l, number_r, tuple[0]+tuple[1], self.overload)
                    else:
                        continue

                elif any(t._.broad_range for t in tuple[1]):
                    norm_num = NumberNormalizer.normalize_number_token(tuple[1])
                    # check for successful normalization
                    if norm_num is None:
                        continue
                    number = Range(norm_num, norm_num * 10, tuple[0]+tuple[1], self.overload)

                elif any(t._.numerator for t in tuple[1]) and any(t._.denominator for t in tuple[1]) and len(tuple[1])==2:
                    numerator = tuple[1][0]
                    denominator = tuple[1][1]
                    norm_numerator = NumberNormalizer.normalize_number_token([numerator])
                    norm_denominator = NumberNormalizer.normalize_number_token([denominator])
                    # check for successful normalization
                    if norm_numerator is None and norm_denominator is None:
                        continue
                    if norm_denominator is None:
                        number = Value(norm_numerator, [numerator, denominator], self.overload)
                    elif norm_numerator is None:
                        number = Value(norm_denominator, [numerator, denominator], self.overload)
                    elif norm_denominator == 0: # e.g. quarter of zero growth
                        continue
                    else:
                        number = Value(norm_numerator/norm_denominator, [numerator, denominator], self.overload)
                        fraction = True

                else:

                    fraction = any(t._.fraction for t in tuple[1])
                    if len(tuple[1])>1 and any(t._.ignore for t in tuple[1]):
                        tagged_with_ignore = [t for t in tuple[1] if t._.ignore]
                        tuple[1].remove(tagged_with_ignore[0])
                        temp = tuple[2]
                        temp.extend([tagged_with_ignore[0]])
                        temp = sorted(temp, key=lambda t: t.i)
                        tuple[2].clear()
                        tuple[2].extend(temp)
                    norm_number = NumberNormalizer.normalize_number_token(tuple[1])
                    # check for successful normalization
                    if norm_number is None:
                        continue
                    if isinstance(norm_number, list): # e.g. [66.5, 107] from '66.5 x 107 mm'
                        number = Range(norm_number[0], norm_number[-1], tuple[0]+tuple[1], self.overload)
                    else:
                        number = Value(norm_number, tuple[1], self.overload)

                if negative_number and isinstance(number, Value): # 5 -> -5 (-5 degree fahrenheit)
                    number.value = -number.value

                if scale_m: # 1.2m tons -> 1200000.0 tons
                    if isinstance(number, Value):
                        number.value = number.value * maps["scales"]["million"]
                    elif isinstance(number, Range):
                        upper = number.upper
                        lower = number.lower
                        if abs(upper - lower) <= 100: # threshold for ranges: e.g. from 60 to 90m subscribers
                            number.lower = number.lower * maps["scales"]["million"]
                            number.upper = number.upper * maps["scales"]["million"]

                norm_bound = NumberNormalizer.normalize_bound_token(tuple[0])
                norm_unit, index, in_unit_list, keys, slice_unit = NumberNormalizer.normalize_unit_token(tuple[2], text)

                if set(norm_unit.split(" ")) & BLACK_LIST_UNITS or {token.text for token in tuple[2]} & BLACK_LIST_UNITS:
                    norm_unit = "-"

                if (any(token._.consider or token._.consider_date for token in tuple[1]) or set(norm_unit.split(" ")) & set(MONTHS)) and not in_unit_list:
                    for token in tuple[1]:
                        token._.set("ignore",True)
                    continue # skip quantity since not real (e.g. 575 Wilbraham Road; 6 October; DATE entities and <number>-<number> quantities without proper unit)

                # mark tokens that are part of a quantity
                if index not in [None, -1]:
                    if slice_unit:
                        for token in tuple[2][:index+1]:
                            token._.set("quantity_part", True)
                    else:
                        tuple[2][index]._.set("quantity_part", True)
                elif index == -1:
                    for token in tuple[2]:
                        token._.set("quantity_part", True)
                for token in tuple[1]:
                    token._.set("quantity_part", True)

                # skip check
                if f' {norm_unit+" "+tuple[1][0].text} ' in f' {text} ': # e.g. battery-powered 2024
                    continue
                if self._number_in_unit(norm_unit) and not in_unit_list and not any(token._.one_of_number for token in tuple[1]): # when there is a number in the unit and not in the unit.json and not one_of (e.g. 9 AB8)
                    continue

                bound = Change(norm_bound, tuple[0])


                if fraction and not in_unit_list: # e.g. 'One out of three Germans is vaccinated'
                    unit = Unit(tuple[2], "% " + norm_unit, tuple[2][:index+1] if index not in [None, -1] else tuple[2], in_unit_list, keys, self.overload)
                elif any(token._.one_of_noun for token in tuple[2]): # consider subtree

                    k = [token._.one_of_noun for token in tuple[2]].index(True)
                    unit = Unit(tuple[2], norm_unit, doc[list(tuple[2][k].subtree)[0].i:list(tuple[2][k].subtree)[-1].i+1], in_unit_list, keys, self.overload)
                elif index not in [None, -1] and len(tuple[2]) > 1:
                    unit = Unit(tuple[2], norm_unit, [tuple[2][index]] if index==0 or not slice_unit else tuple[2][:index+1], in_unit_list, keys, self.overload)
                else:
                    unit = Unit(tuple[2], norm_unit, tuple[2], in_unit_list, keys, self.overload)

                if tuple[2] != unit.unit:
                    tuple = (tuple[0], tuple[1], unit.unit, tuple[3]) # [of, of, dollars] vs. [of, dollars]

                referred_noun = []
                ref_noun_flag = False # if ref_noun was set

                # approach for quantities in brackets
                if tuple[2] and all(t._.in_bracket for t in tuple[1]+tuple[2]) and unit.norm_unit != "-": # Key players: G Hunter Maldonado (15.3 ppg, 7.4 rpg, 6.9 apg)
                    head_token = [t.head for t in tuple[1]]
                    head_token_old = [tuple[1][0]]
                    while not all(t in head_token_old for t in head_token) and not any(not t._.in_bracket and t.pos_ in ["NOUN", "PROPN"] for t in head_token):
                        head_token_old = head_token
                        head_token = [t.head for t in head_token_old]
                    pot_ref_nouns = [sorted(list(token.subtree), key=lambda t: t.i) for token in head_token if not token._.in_bracket and token.pos_ in ["NOUN", "PROPN"]]
                    if pot_ref_nouns:
                        referred_noun.append(pot_ref_nouns[0])
                        ref_noun_flag = True

                if len(unit.unit) > 1 and index not in [None, -1] and ((any(token.text=="of" for token in unit.unit) and not self.ignore_some_rules) or not any(token.text=="of" for token in unit.unit)): # whole list with units is not the unit
                    if index + 1 < len(unit.unit):
                        pot_ref_nouns = list(list(token.subtree) for token in unit.unit[index+1:] if token.pos_ in ["NOUN","PROPN"] and not token._.ignore_noun) # observe the next tokens and their subtrees
                        pot_ref_nouns = sorted(list(set(token for subtree in pot_ref_nouns for token in subtree)), key=lambda t: t.i)
                    else:
                        pot_ref_nouns = list(unit.unit[(index+1)%len(unit.unit)].subtree) if unit.unit[(index+1)%len(unit.unit)].pos_ in ["NOUN", "PROPN"] and not unit.unit[(index+1)%len(unit.unit)]._.ignore_noun else [] # observe the next token and its children
                        if pot_ref_nouns and unit.unit[(index+1)%len(unit.unit)].dep_ in ["compound"]: # e.g. Rishi Sunaks first 100 days ...
                            pot_ref_nouns += [unit.unit[(index+1)%len(unit.unit)].head]
                    if pot_ref_nouns:
                        pot_ref_nouns = list(filter(lambda x: x not in [unit.unit[index]]+tuple[1], pot_ref_nouns)) # filter number and unit
                        pot_ref_nouns.append(pot_ref_nouns[-1].i)
                        referred_noun.append(pot_ref_nouns[:-1])

                if not ref_noun_flag:
                    if index not in [None, -1] and len(tuple[2]) > 1:# and in_unit_list:
                        temp = [tuple[2][index]] if index==0 or not slice_unit else tuple[2][:index+1]
                        tuple[2].clear()
                        tuple[2].extend(temp)
                    related_quantifiers = self._extract_quantifiers(doc, tuple)
                    if related_quantifiers:
                        for quantifier in related_quantifiers:
                            referred_noun.append(quantifier[:-1])


                if not ref_noun_flag:
                    # follow dependency tree approach
                    # check whether there is VERB in the ancestors of the number
                    # then check if there is nsubj or nsubjpass in the children of the first found VERB
                    # when the nsubj is DET then follow the tree to the actual nsubj
                    # otherwise assign the subtree of the nsubj as concept for that number
                    pot_ref_nouns = [(list(t.subtree), t, t.pos_) for list_t in [t for t in [list(token.children) for token in sorted(list(set().union(*[list(t.ancestors) for t in tuple[1]])), key=lambda t: (tuple[1][0].i-t.i)) if token.pos_ in ["VERB","AUX"]]] for t in sorted(list_t, key= lambda t: (tuple[1][0].i-t.i)) if t.dep_ in ["nsubj","nsubjpass"] and t not in tuple[1]+tuple[2] and t.i<tuple[1][0].i]
                    if pot_ref_nouns:
                        if pot_ref_nouns[0][-1] == "DET": # e.g. which
                            pot_noun = [t for t in list(pot_ref_nouns[0][1].ancestors) if t.dep_ in ["nsubj", "nsubjpass"]]
                            if pot_noun:
                                referred_noun.append(list(pot_noun[0].subtree))
                                ref_noun_flag = True
                        elif pot_ref_nouns[0][-1] in ["NOUN", "PROPN", "ADJ"]:
                            if not (pot_ref_nouns[0][-1] in ["ADJ"] and len(pot_ref_nouns[0][0])==1 and pot_ref_nouns[0][0][0].is_stop):
                                referred_noun.append(pot_ref_nouns[0][0])
                                ref_noun_flag = True

                # follow dependency tree approach
                # check whether there is relcl in the ancestors of the number
                # then check if there is PROPN or NOUN in the ancestors
                # when yes, then the subtree of the first found PROPN or NOUN is assigned as concept for the quantity
                # e.g. Senard, 65, now faces the immediate task of soothing relations with Nissan, which is 43.4 percent-owned by Renault.
                if not ref_noun_flag:
                    pot_ref_nouns = [head for head in [token.head for token in sorted(list(tuple[1][0].ancestors), key=lambda t: abs(tuple[1][0].i-t.i)) if token.pos_ in ["VERB","AUX"] and token.dep_ in ["relcl"]] if head.pos_ in ["PROPN","NOUN"]]
                    if pot_ref_nouns:
                        if pot_ref_nouns[0].dep_ in ["appos"]: # e.g. Ford (F) which has 13000 workers in the ...
                            referred_noun.append(list(pot_ref_nouns[0].head.subtree))
                        else:
                            referred_noun.append(list(pot_ref_nouns[0].subtree))
                        ref_noun_flag = True

                if not ref_noun_flag:
                    # root-nsubj-approach:
                    # check whether there is ROOT in the ancestors of the number
                    # then check if there is nsubj in the children of this ROOT
                    # and finally assign the subtree of the nsubj as concept for that number
                    pot_ref_nouns = [list(t.subtree) for list_t in [t for t in [list(token.children) for token in list(tuple[1][0].ancestors) if token.dep_=="ROOT"]] for t in list_t if t.dep_ in ["nsubj"]]
                    if pot_ref_nouns and not (set(tuple[1]+tuple[2]) & set(pot_ref_nouns[0])):
                        referred_noun.append(pot_ref_nouns[0])
                        ref_noun_flag = True
                        #root_nsubj = True # for the part with global referred concepts that is currently not used

                if not ref_noun_flag:
                    # follow dependency tree approach
                    # check whether there is VERB in the ancestors of the number
                    # then check if there is dobj in the children of the first found VERB
                    # when the dobj is NOUN or PROPN, then its subtree is assigned as concept for the quantity
                    pot_ref_nouns = [(list(t.subtree), t, t.pos_) for list_t in [t for t in [list(token.children) for token in sorted(list(set().union(*[list(t.ancestors) for t in tuple[1]])), key=lambda t: abs(tuple[1][0].i-t.i)) if token.pos_ in ["VERB","AUX"]]] for t in list_t if t.dep_ in ["dobj"] and t not in tuple[1]+tuple[2]]
                    if pot_ref_nouns:
                        if pot_ref_nouns[0][-1] in ["NOUN", "PROPN"] and len(pot_ref_nouns[0][1]) > 1:# this was 0 before
                            referred_noun.append(pot_ref_nouns[0][0])
                            ref_noun_flag = True

                # looking at extracted global referred concepts when no noun was found via the root-nsubj-approach
                """if nouns and not any(bool(self._remove_stopwords_from_noun(list)) for list in referred_noun):# and not ref_noun_flag:# and not root_nsubj: # len(nouns) > 1
                    min_dist = 100
                    ref_noun_to_append = None
                    for sent_bound in boundaries:
                        if sent_bound[0] <= tuple[-1] and tuple[-1] <= sent_bound[1]:
                            for noun in nouns:
                                if sent_bound[0] <= noun[-1] and noun[-1] <= sent_bound[1]:
                                    # if more than one referred_noun found, take the nearest one
                                    # skip noun that contains token(s) which are part of the unit
                                    if abs(tuple[-1]-noun[-1]) < min_dist and (not (set(unit.unit) & set(noun[:-1]))):
                                        if referred_noun and (set(noun[:-1]) & set(referred_noun[-1])):
                                            ref_noun_to_append = None
                                            break # already assigned in a previous step
                                        min_dist = abs(tuple[-1]-noun[-1])
                                        ref_noun_to_append = noun[:-1]
                                        #ref_noun_flag = True
                                    #ref_noun.noun = noun[:-1]
                                    #ref_noun_flag = True
                                    #break # take the first one
                    if ref_noun_to_append is not None:
                        referred_noun.append(ref_noun_to_append)
                        ref_noun_flag = True"""

                if not ref_noun_flag:
                    ans=tuple[1][0].ancestors
                    if tuple[1][0].dep_=="nummod" and tuple[1][0].head.pos_ in ["NOUN","NN","PROPN"]:
                        referred_noun.append([tuple[1][0].head])
                    for an in ans:
                        if an.dep_=="appos" :
                            referred_noun.append([an])



                # default case
                # when there is one input sentence (according to spacy sentenizer)
                # and only one global noun or one quantity, then use the (first) noun
                if not ref_noun_flag and len(list(doc.sents)) == 1 and nouns and (len(nouns)==1 or len(tuples)==1):
                    referred_noun.append(nouns[0][:-1])
                    ref_noun_flag = True


                referred_noun = self.postprocess_referred_noun(ref_noun_flag, referred_noun.copy(), unit, index, slice_unit, tuple)
                referred_noun = self.postprocess_one_of_quantity(referred_noun.copy(), tuple, unit)

                quant = Quantity(number, bound, unit, referred_noun, all(t._.in_bracket for t in tuple[1]+tuple[2]), self.overload)

                # drop if no number found
                if number: # and unit:
                    result.append(quant)

        def accept_token(token):
            if token._.quantity_part:
                return not token.pos_ in ["NUM"]
            return not token._.quantity_part

        # e.g. The Dow Jones and the S&P 500 closed down by 2.5% and 1.8%, respectively.
        if {"respectively"} & set(token.text for token in doc):
            concepts = self.find_respectively_concepts([token for token in doc if token.text=="respectively"])
            if concepts:
                # assign the parts to the quantities
                for i, quantity in enumerate(sorted(result, key=lambda t: t.value.span[0].i)):
                    quantity.referred_concepts = [concepts[i%len(concepts)]]

        for quantity in sorted(result, key=lambda t: t.value.span[0].i):
            # cut noun if it contains tokens that are part of a quantity
            for i, ref_noun_list in enumerate(quantity.referred_concepts):
                if ref_noun_list and all(t._.quantity_part for t in ref_noun_list) and [q for q in result if set(q.value.span+q.unit.unit)&set(ref_noun_list)] and len(quantity.referred_concepts)==1:
                    # use computed char_indices
                    # check if the quantity is in brackets and whether the other quantity is before it in the text
                    if self.overload and quantity.in_bracket and min(list(itertools.chain(*quantity.value.char_indices))) > max(list(itertools.chain(*[q for q in result if set(q.value.span+q.unit.unit)&set(ref_noun_list)][0].value.char_indices))):
                        quantity.referred_concepts = [q for q in result if set(q.value.span+q.unit.unit)&set(ref_noun_list)][0].referred_concepts # take the concept from the other quantity when no other concepts found for this quantity
                    # else compute them and check
                    elif quantity.in_bracket and min(list(itertools.chain(*[(token.idx, token.idx + len(token.text)) for token in quantity.value.span]))) > max(list(itertools.chain(*[(token.idx, token.idx + len(token.text)) for token in [q for q in result if set(q.value.span+q.unit.unit)&set(ref_noun_list)][0].value.span]))):
                        quantity.referred_concepts = [q for q in result if set(q.value.span+q.unit.unit)&set(ref_noun_list)][0].referred_concepts # take the concept from the other quantity when no other concepts found for this quantity
                    else:
                        quantity.referred_concepts = [q for q in result if set(q.value.span+q.unit.unit)&set(ref_noun_list)][0].referred_concepts # take the concept from the other quantity when no other concepts found for this quantity
                    if all(t._.quantity_part for concept in quantity.referred_concepts for t in concept) and set(quantity.value.span+quantity.unit.unit)&set(token for concept in quantity.referred_concepts for token in concept):
                        quantity.referred_concepts = [] # concept points to the quantity itself => empty
                    if quantity.change.change == "=": # shared bound (for quantities in brackets)
                        quantity.change = [q for q in result if set(q.value.span+q.unit.unit)&set(ref_noun_list)][0].change
                    break
                quantity.referred_concepts[i] = list(itertools.takewhile(accept_token, ref_noun_list))
                quantity.referred_concepts[i] = list(noun for noun in quantity.referred_concepts[i] if not noun._.quantity_part) # kw system -> system
            quantity.referred_concepts = list(quantity.referred_concepts for quantity.referred_concepts, _ in itertools.groupby(quantity.referred_concepts))

            # check for overlapping concepts
            # e.g. [men, aged, 18, 34] and [men]
            def donewk(k):
                newk = []
                for i in k:
                    if i not in newk:
                        newk.append(i)
                return newk
            quantity.referred_concepts= donewk(quantity.referred_concepts)

            nouns_copy = quantity.referred_concepts.copy()
            for noun in nouns_copy:
                for _, noun_2 in itertools.product(noun, nouns_copy[:nouns_copy.index(noun):] + nouns_copy[nouns_copy.index(noun)+1:]):
                    if set(t.i for t in noun).intersection(set(t.i for t in noun_2)): # looking at all neighbours
                        if len(noun) < len(noun_2):
                            try:
                                quantity.referred_concepts.remove(noun)
                            except ValueError:
                                continue
                        else:
                            try:
                                quantity.referred_concepts.remove(noun_2)
                            except ValueError:
                                continue

        for quantity in result:
            quantity.transform_concept_to_dict()

        return sorted(result, key=lambda t: t.value.span[0].i)


    def find_respectively_concepts(self, respectively_token):
        # start with the token "respectively"
        # follow the heads til the ROOT
        # then consider the subtree and split it into the parts
        # ruturn the concepts parts
        head_token = respectively_token[0].head
        head_token_old = respectively_token[0]
        while not head_token == head_token_old or not head_token.dep_ in ["ROOT"]:
            head_token_old = head_token
            head_token = head_token_old.head
        pot_nouns = list(itertools.takewhile(lambda t: t != head_token if head_token.pos_ in ["VERB"] else t != respectively_token[0].head, list(head_token.subtree)))
        and_token = [token for token in pot_nouns if token.text in ["and", ","] and pot_nouns.index(token) != 0]
        if and_token:
            concepts = []
            for token in and_token:
                and_index = pot_nouns.index(token)
                concepts.append(self._remove_stopwords_from_noun(pot_nouns[:and_index]))
                pot_nouns = pot_nouns[and_index+1:]
            concepts.append(self._remove_stopwords_from_noun(pot_nouns))
            return list(filter(None, concepts))
        return []


    def postprocess_referred_noun(self, ref_noun_flag, referred_noun, unit, index, slice_unit, tuple):
        if referred_noun:
            for i, ref_noun_list in enumerate(referred_noun):
                ref_noun_list = [] if ref_noun_flag and len(ref_noun_list) == 1 and ref_noun_list[0].pos_ in ["DET", "PRON", "NUM"] else ref_noun_list # digits, this, each, he, I, who, which etc. -> meaning that we did not find one
                ref_noun_list = self._remove_stopwords_from_noun(ref_noun_list)
                if index not in [None, -1] and len(unit.unit) > 1:# and in_unit_list:
                    unit.unit = [unit.unit[index]] if index==0 or not slice_unit else unit.unit[:index+1] # (=,5.0,[percent, penalty],percentage,[a, penalty, of]) -> (=,5.0,[percent],percentage,[a, penalty, of])
                ref_noun_list = clean_up(unit.unit, tuple[1], ref_noun_list) # clean nouns
                referred_noun[i] = ref_noun_list
        return referred_noun

    def postprocess_one_of_quantity(self, referred_noun, tuple, unit):
        if len(tuple[1])==1 and tuple[1][0]._.one_of_number: # e.g. one of the proudest moments in his life.
            if not unit.unit and unit.norm_unit=="-" and referred_noun:
                unit.unit = referred_noun[0] # replace unit with concept
                unit.norm_unit = " ".join([t.lemma_ for t in referred_noun[0] if not t.is_stop]) # replace norm_unit with lemmatized concept + removal of stopwords
                referred_noun = [] # concept becomes empty
            elif referred_noun:
                referred_noun = [] # concept becomes empty
        return referred_noun

    def _replace_indicies_in_text(self,to_be_replaced,text,replaced_checks=None):
        norm_text=""
        end_index=None
        pervious_start_index,pervious_end_index=0,0
        for key,value in to_be_replaced.items():
            start_index,end_index=key
            if pervious_start_index==start_index:
                norm_text=norm_text+text[pervious_start_index:start_index]+" "+value
            else:
                if replaced_checks:
                    if end_index in [k[0] for k in replaced_checks.keys()] :
                        norm_text=norm_text+text[pervious_start_index:start_index]+value+" "
                    elif  start_index in [k[1] for k in replaced_checks.keys()]:
                        norm_text=norm_text+text[pervious_start_index:start_index]+" " +value
                    else:
                        norm_text=norm_text+text[pervious_start_index:start_index]+value
                else:
                    norm_text=norm_text+text[pervious_start_index:start_index]+value

            pervious_start_index=end_index
        if end_index:
            norm_text=norm_text+text[end_index:]
        return norm_text


    def _normalize_text(self, tuples, doc):
        text = str(doc)
        to_be_replaced = {}
        for quantity in tuples:
            char_indices = quantity.get_char_indices()
            value_char_indices = char_indices["value"]
            unit_char_indices = char_indices["unit"]
            for i, idx_tuple in enumerate(value_char_indices):
                to_be_replaced[idx_tuple] = quantity.value.get_str_value(i)
            if unit_char_indices:
                for idx_tuple in unit_char_indices:
                    if quantity.unit.scientific and quantity.unit.norm_unit not in ["-","% -",""] and not quantity.unit.norm_unit.startswith("-"):
                        to_be_replaced[idx_tuple] = quantity.unit.norm_unit
        to_be_replaced = dict(sorted(to_be_replaced.items()))
        norm_text=self._replace_indicies_in_text(to_be_replaced,text)
        return norm_text

    def _normalize_text_values_only(self, tuples, doc):
        text = str(doc)
        to_be_replaced = {}

        for quantity in tuples:
            char_indices = quantity.get_char_indices()
            value_char_indices = char_indices["value"]
            for i, idx_tuple in enumerate(value_char_indices):
                to_be_replaced[idx_tuple] = quantity.value.get_str_value(i)

        to_be_replaced = dict(sorted(to_be_replaced.items()))
        norm_text=self._replace_indicies_in_text(to_be_replaced,text)
        return norm_text

    def _normalize_text_units_only(self, tuples, doc):
        text = str(doc)
        to_be_replaced = {}
        to_be_replaced_vals={}

        for quantity in tuples:
            char_indices = quantity.get_char_indices()
            value_char_indices = char_indices["value"]
            unit_char_indices = char_indices["unit"]
            for i, idx_tuple in enumerate(value_char_indices):
                to_be_replaced_vals[idx_tuple] = quantity.value.get_str_value(i)
            if unit_char_indices:
                for idx_tuple in unit_char_indices:
                    if quantity.unit.scientific and quantity.unit.norm_unit not in ["-","% -",""] and not quantity.unit.norm_unit.startswith("-"):
                        to_be_replaced[idx_tuple] = quantity.unit.norm_unit
        to_be_replaced = dict(sorted(to_be_replaced.items()))
        norm_text=self._replace_indicies_in_text(to_be_replaced,text,to_be_replaced_vals)
        return norm_text

    def _replace_indicies_in_text_with_idx(self,to_be_replaced,text,replaced_checks=None):
        norm_text=""
        end_index=None
        pervious_start_index,pervious_end_index=0,0
        for key,value in to_be_replaced.items():
            start_index,end_index=key
            if pervious_start_index==start_index:
                norm_text=norm_text+text[pervious_start_index:start_index]+" "+value
            else:
                if replaced_checks:
                    if end_index in [k[0] for k in replaced_checks.keys()] :
                        norm_text=norm_text+text[pervious_start_index:start_index]+value+" "
                    elif  start_index in [k[1] for k in replaced_checks.keys()]:
                        norm_text=norm_text+text[pervious_start_index:start_index]+" " +value
                    else:
                        norm_text=norm_text+text[pervious_start_index:start_index]+value
                else:
                    norm_text=norm_text+text[pervious_start_index:start_index]+value

            pervious_start_index=end_index
        if end_index:
            norm_text=norm_text+text[end_index:]
        return norm_text
    def _normalize_text_scientific_notation(self, tuples, text):


        norm_text=""
        end_index=None
        pervious_start_index,pervious_end_index=0,0
        list_of_val_indicies=[]
        list_of_unit_indices=[]
        replace_dictionary_unit={}
        for quantity in tuples:
            char_indices = quantity.get_scientific_char_indices()
            if quantity.unit.scientific and  quantity.unit.norm_unit not in ["-","% -",""] and not quantity.unit.norm_unit.startswith("-") :
                for i in range(len(char_indices["unit"])):
                    to_be_replaced=quantity.unit.norm_unit
                    start_index,end_index=char_indices["unit"][i]
                    if text[start_index-1]!=" ":
                        to_be_replaced=" "+to_be_replaced

                    if end_index<len(text) and text[end_index]!=" ":
                        to_be_replaced=to_be_replaced+" "
                    if i>0:
                        to_be_replaced=" "
                    replace_dictionary_unit[char_indices["unit"][i]]={"replace":to_be_replaced,"original":text[start_index:end_index]}
            list_of_val_indicies.append(copy.deepcopy(char_indices["value"]))
            list_of_unit_indices.append(copy.deepcopy(char_indices["unit"]))
        org_unit_indicies=copy.deepcopy(list_of_unit_indices)
        org_val_indices=copy.deepcopy(list_of_val_indicies)

        for key,value in sorted(replace_dictionary_unit.items()):
            start_index,end_index=key
            norm_text=norm_text+text[pervious_start_index:start_index]+value["replace"]
            pervious_start_index=end_index

            for identifier in range(len(list_of_unit_indices)):
                for index in range(len(list_of_unit_indices[identifier])):
                    if org_unit_indicies[identifier][index][0]==start_index:
                        list_of_unit_indices[identifier][index]=self._adjust_index_end(list_of_unit_indices[identifier][index], len(value["replace"]) - len(value["original"]))
                    elif org_unit_indicies[identifier][index][0]>start_index:
                        list_of_unit_indices[identifier][index]=self._adjust_index_both(list_of_unit_indices[identifier][index], len(value["replace"]) - len(value["original"]))
                    if index<len(org_val_indices[identifier]) and org_val_indices[identifier][index][0]>start_index:
                        list_of_val_indicies[identifier][index]=self._adjust_index_both(list_of_val_indicies[identifier][index],  len(value["replace"]) - len(value["original"]))
        if end_index:
            norm_text=norm_text+text[end_index:]
        if len(replace_dictionary_unit)==0:
            norm_text=text
        for quantity,value_char_indices,unit_char_indices in zip(tuples,list_of_val_indicies,list_of_unit_indices):
            quantity.set_scientific_char_indices_nromalized(value_char_indices,unit_char_indices)

        return norm_text




    def _adjust_index_both(self, old_index, by):
        new_tuple_1=old_index[0]+by
        new_tuple_2=old_index[1]+by
        return (new_tuple_1,new_tuple_2)

    def _adjust_index_end(self, old_index, by):
        new_tuple_1=old_index[0]
        new_tuple_2=old_index[1]+by
        return (new_tuple_1,new_tuple_2)

    def _normalize_text_scientific_notation_values_only(self, tuples, text):
        """returns the preprocessed text, where the values (numbers) have been replace with their short scientific notation"""

        norm_text=""
        end_index=None
        pervious_start_index,pervious_end_index=0,0
        list_of_val_indicies=[]
        list_of_unit_indices=[]
        notations=[]
        replace_dictionary_vals={}
        for quantity in tuples:

            char_indices = quantity.get_char_indices()
            for i in range(len(char_indices["value"])):
                start_index,end_index=char_indices["value"][i]
                to_be_replaced= quantity.value.get_simplified_scientific_notation()
                if isinstance(to_be_replaced,tuple):
                    to_be_replaced=to_be_replaced[i]
                notations.append(to_be_replaced)
                replace_dictionary_vals[char_indices["value"][i]]={"replace":to_be_replaced,"original":text[start_index:end_index]}
            list_of_val_indicies.append(copy.deepcopy(char_indices["value"]))
            list_of_unit_indices.append(copy.deepcopy(char_indices["unit"]))
        org_unit_indicies=copy.deepcopy(list_of_unit_indices)
        org_val_indices=copy.deepcopy(list_of_val_indicies)
        for key,value in sorted(replace_dictionary_vals.items()):
            start_index,end_index=key
            norm_text=norm_text+text[pervious_start_index:start_index]+value["replace"]
            pervious_start_index=end_index
            for identifier in range(len(list_of_val_indicies)):
                for index in range(len(list_of_val_indicies[identifier])):
                    if org_val_indices[identifier][index][0]==start_index:
                        list_of_val_indicies[identifier][index]=self._adjust_index_end(list_of_val_indicies[identifier][index], len(value["replace"]) - len(value["original"]))
                    elif org_val_indices[identifier][index][0]>start_index:
                        list_of_val_indicies[identifier][index]=self._adjust_index_both(list_of_val_indicies[identifier][index], len(value["replace"]) - len(value["original"]))
            for identifier2 in range(len(list_of_unit_indices)):
                for val_index in range(len(list_of_unit_indices[identifier2])):
                    if org_unit_indicies[identifier2][val_index][0]>start_index:
                        list_of_unit_indices[identifier2][val_index]=self._adjust_index_both(list_of_unit_indices[identifier2][val_index],  len(value["replace"]) - len(value["original"]))


        if end_index:
            norm_text=norm_text+text[end_index:]

        for quantity,value_char_indices,unit_char_indices in zip(tuples,list_of_val_indicies,list_of_unit_indices):
            quantity.set_scientific_char_indices(value_char_indices,unit_char_indices)

        if len(norm_text)==0:
            norm_text=text

        return norm_text,notations

