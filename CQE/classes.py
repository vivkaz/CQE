import json
import os
import re
from pathlib import Path
from decimal import Decimal
from typing import Iterable
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

def get_project_root() -> Path:
    return Path(__file__).parent


class Change:
    def __init__(self, change="", span=[]):
        self.span = span
        self.change = change

    def __str__(self):
        return self.change

    def __bool__(self):
        return bool(self.change)


class Range:
    def __init__(self, lower, upper, span=[], overload=False):
        self.span = sorted(span, key=lambda t: t.i)
        self.lower = lower
        self.upper = upper
        self.char_indices = [(token.idx, token.idx + len(token.text)) for token in span if token.pos_ in ["NUM", "NOUN"]] if span and overload else None
        self.scientific_notation_lower = '%e' % Decimal(str(self.lower)) # e.g. 0.000000e+00
        self.scientific_notation_upper = '%e' % Decimal(str(self.upper)) # e.g. 8.000000e+01
        self.scientific_notation = self.scientific_notation_lower + "-" + self.scientific_notation_upper
        self.simplified_scientific_notation_lower, self.simplified_scientific_notation_upper = self.get_simplified_scientific_notation() # e.g. 0e+00, 8e+01
        self.simplified_scientific_notation = self.simplified_scientific_notation_lower + "-" + self.simplified_scientific_notation_upper

    def __str__(self):
        return f"{self.lower}-{self.upper}"

    def __bool__(self):
        return bool(self.lower is not None and self.upper is not None)

    def get_str_value(self, idx_value=0):
        if len(self.char_indices) == 1: # e.g. dozens, millions, thousands etc.
            lower = str(int(self.lower)) if self.lower - int(self.lower) == 0 else str(self.lower)
            upper = str(int(self.upper)) if self.upper - int(self.upper) == 0 else str(self.upper)
            return f"{lower}-{upper}"
        if idx_value == 0:
            return str(int(self.lower)) if self.lower - int(self.lower) == 0 else str(self.lower)
        return str(int(self.upper)) if self.upper - int(self.upper) == 0 else str(self.upper)

    def get_simplified_scientific_notation(self):
        """return a simplified scientific notation of the lower and upper value"""
        notation_lower = '%e' % Decimal(str(self.lower))
        notation_upper = '%e' % Decimal(str(self.upper))
        return notation_lower.split('e')[0].rstrip('0').rstrip('.') + 'e' + notation_lower.split('e')[1], notation_upper.split('e')[0].rstrip('0').rstrip('.') + 'e' + notation_upper.split('e')[1]


class Value:
    def __init__(self, value, span=[], overload=False):
        self.span = sorted(span, key=lambda t: t.i)
        self.value = value
        #self.char_indices = [(token.idx, token.idx + len(token.text)) for token in span] if span and overload else None
        self.char_indices = [(span[0].idx, span[-1].idx + len(span[-1].text))] if span and overload else None
        self.scientific_notation = '%e' % Decimal(self.__str__()) # e.g. 4.000000e-03
        self.simplified_scientific_notation = self.get_simplified_scientific_notation() # e.g. 4e-03

    def __str__(self):
        return str(self.value)

    def __bool__(self):
        return bool(self.value is not None)

    def get_str_value(self, idx_value=0):
        if self.value - int(self.value) == 0:
            return str(int(self.value))
        return str(self.value)

    def get_simplified_scientific_notation(self):
        """return a simplified scientific notation of the value"""
        notation = '%e' % Decimal(self.__str__())
        return notation.split('e')[0].rstrip('0').rstrip('.') + 'e' + notation.split('e')[1]


class Unit:
    def __init__(self, unit=[], norm_unit="-", span=[], scientific=False, keys=[], overload=False):
        self.span = sorted(span, key=lambda t: t.i) if norm_unit != "-" else []
        if len(set(token.text for token in unit)) != 1 and len(unit) > 1:
            self.unit = [token for i,token in enumerate(unit) if token.text != unit[(i+1)%len(unit)].text]
        else:
            self.unit = unit
        self.norm_unit = norm_unit
        if len(span)>1 and sorted([token.i for token in span]) == list(range(min(token.i for token in span), max(token.i for token in span)+1)):
            self.char_indices = [(span[0].idx, span[-1].idx + len(span[-1].text))] if span and overload else None
        else:
            self.char_indices = [(token.idx, token.idx + len(token.text)) for token in span] if span and overload else None
        self.scientific = scientific if overload else None # unit type attribute, which identifies if it is a noun based unit, or a scientific unit (we find it in out dict)
        self.unit_keys = self.set_list_of_keys(keys) if overload else None # list of (lists of) keys from the unit dict used for normalization
        self.unit_surfaces_forms = self.set_unit_surface_forms() if overload else None # all the surface forms of the unit that is detected

    def __str__(self):
        return str(self.unit)

    def __bool__(self):
        return bool(self.unit)
    
    def _flatten(self, items):
        """yield items from nested iterable"""
        for item in items:
            if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
                for sub_item in self._flatten(item):
                    yield sub_item
            else:
                yield item

    def set_list_of_keys(self, keys_list):
        """return a list of the keys used for the normalization of the unit"""
        keys = []
        k_list = list(self._flatten(keys_list))
        for key in k_list:
            keys.append(key)
        return keys

    def set_unit_surface_forms(self):
        """return list or dict of the surface forms for the unit"""
        if self.scientific and self.unit_keys: # in unit dictionary
            path = get_project_root()
            file_name = os.path.join(path, "unit.json")
            with open(file_name, "r", encoding="utf8") as f:
                units_dict = json.load(f)
            if len(self.unit_keys) >= 1: # compound unit e.g. 'dollar per square foot' or single unit e.g. 'percentage'
                return { key: units_dict[key].get("surfaces")+units_dict[key].get("symbols") for key in self.unit_keys }
        return {} # noun based unit => no surface forms


class Concept:
    def __init__(self, noun=[], span=[]):
        self.span = sorted(span, key=lambda t: t.i)
        if noun:
            self.noun = {}
            for tokens_list in noun:
                if tokens_list: # consider only not empty lists
                    self.noun.update({len(self.noun): tokens_list})
            if not self.noun: # empty => "-"
                self.noun = "-"
        else:
            self.noun = "-" # no noun found => "-"

    def __str__(self):
        return str(self.noun)

    def __bool__(self):
        return bool(self.noun)

    def get_nouns(self):
        """return the referred nouns from the dict in a list form"""
        return [list(list_noun) for list_noun in self.noun.values()] if self.noun != "-" else []


class Quantity:
    original_text = None # class variable

    def __init__(self, value, change=Change(), unit=Unit(), referred_concepts=None, in_bracket=False):
        self.change = change
        self.value = value
        self.unit = unit
        self.referred_concepts = [] if not referred_concepts else referred_concepts
        self.__value_char_indices = value.char_indices # characters indices based on the preprocessed sentence
        self.__unit_char_indices = unit.char_indices # characters indices based on the preprocessed sentence
        self.__original_value_char_indices = None # characters indices based on the original sentence
        self.__original_unit_char_indices = None # characters indices based on the original sentence
        self.__preprocessed_text = None # default
        self.__normalized_text = None # default
        self.in_bracket = in_bracket

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"({str(self.change)},{str(self.value)},{str(self.unit.unit)},{str(self.unit.norm_unit)},{str(self.referred_concepts)})"
    
    def set_original_text(self, text):
        Quantity.original_text = text

    def set_preprocessed_text(self, text):
        self.__preprocessed_text = text

    def set_normalized_text(self, text):
        self.__normalized_text = text

    def get_preprocessed_text(self):
        return self.__preprocessed_text
    
    def get_normalized_text(self):
        return self.__normalized_text

    def transform_concept_to_dict(self):
        """create a Concept instance"""
        self.referred_concepts = Concept(self.referred_concepts)

    def get_unit_surface_forms(self):
        """return list or dict of the surface forms for the Unit"""
        return self.unit.unit_surfaces_forms
    
    def get_char_indices(self):
        """return dictionary of the Value and Unit indices in the preprocessed sentence"""
        return { "value": self.__value_char_indices, "unit": self.__unit_char_indices }
    
    def compute_original_char_indices(self):
        """return characters indices based on the the original sentence"""

        def find_most_similar_substring(target_substring, text):
            """find the most similar substring to the given target within the text"""

            def n_grams(tokens, n):
                """generate n-grams from a list of tokens"""
                n_grams = []
                for j in range(1,n+1):
                    n_grams.extend(["".join(tokens[i:i+j]) for i in range(len(tokens)-j+1)])
                return n_grams

            return process.extractOne(target_substring, n_grams(text, len(text)), scorer=fuzz.ratio)

        # value indices
        original_value_indices = []
        value_indices = self.__value_char_indices
        target = self.__preprocessed_text[sorted(value_indices)[0][0]:sorted(value_indices)[-1][1]]
        
        # find matches of the target substring in the original text
        matches = list(re.finditer(re.escape(target), self.original_text))

        if not matches:
            # if no direct matches, find the most similar substring and find matches
            most_similar = find_most_similar_substring(target, self.original_text)[0]
            matches = re.finditer(re.escape(most_similar), self.original_text)

        # iterate through the matches and mark already observed ones
        for match in matches:
            original_value_indices.append((match.start(), match.end()))
            mark = "@"*len(match.group())
            self.set_original_text(re.sub(f"\b{re.escape(match.group())}\b", mark, self.original_text, count=1))
            break

        # unit indices
        original_unit_indices = []
        unit_indices = self.__unit_char_indices if self.__unit_char_indices else []
        target = ""

        # iterate through each span in the sorted list of spans
        for tuple_idx in sorted(unit_indices):
            target = self.__preprocessed_text[tuple_idx[0]:tuple_idx[1]]
            
            # find matches of the target substring in the original text
            matches = list(re.finditer(re.escape(target), self.original_text))

            if not matches:
                # if no direct matches, find the most similar substring and find matches
                most_similar = find_most_similar_substring(target, self.original_text)[0]
                matches = re.finditer(re.escape(most_similar), self.original_text)

            # iterate through the matches and mark already observed ones
            for match in matches:
                original_unit_indices.append((match.start(), match.end()))
                mark = "@"*len(match.group())
                self.set_original_text(re.sub(f"\b{re.escape(match.group())}\b", mark, self.original_text, count=1))
                break

        return original_value_indices, original_unit_indices


    def get_original_char_indices(self):
        """return dictionary of the Value and Unit indices in the original sentence"""
        if self.__original_value_char_indices is None and self.__original_unit_char_indices is None:
            self.__original_value_char_indices, self.__original_unit_char_indices = self.compute_original_char_indices()

        return { "value": self.__original_value_char_indices, "unit": self.__original_unit_char_indices }
