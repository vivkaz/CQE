import json
import os

from pathlib import Path
from decimal import Decimal
from typing import Iterable

from spacy.lang.lex_attrs import like_num

from .number_lookup import maps


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

        self.char_indices = [(token.idx, token.idx + len(token.text)) for token in span if self._is_range(token) ] if span else [] if overload else None
        self.char_indices.sort(key=lambda a: a[0])
        if len(self.char_indices)>2:
            for i in range(len(self.char_indices)-1):

                if i+1< len(self.char_indices) and self.char_indices[i][1]+1==self.char_indices[i+1][0]:#if they overlap them merge them
                    self.char_indices[i]=(self.char_indices[i][0],self.char_indices[i+1][1])
                    self.char_indices.remove(self.char_indices[i+1])

        if len(self.char_indices)>2:
            for i in range(len(self.char_indices)-1):
                if i+1< len(self.char_indices) and abs(self.char_indices[i][1]+1-self.char_indices[i+1][0])<7:#if they overlap them merge them
                    self.char_indices[i]=(self.char_indices[i][0],self.char_indices[i+1][1])
                    self.char_indices.remove(self.char_indices[i+1])

        if len(self.char_indices)==3:
            start_i_1=self.char_indices[0][0]
            start_i_2=self.char_indices[1][0]
            start_i_3=self.char_indices[2][0]
            one_to_two=abs(start_i_1-start_i_2)
            one_to_three=abs(start_i_1-start_i_3)
            two_to_three=abs(start_i_2-start_i_3)
            if one_to_two <one_to_three and one_to_two<two_to_three:
                self.char_indices.remove(self.char_indices[2])
            elif one_to_three<one_to_two and one_to_three<two_to_three:
                self.char_indices.remove(self.char_indices[1])
            else:
                self.char_indices.remove(self.char_indices[0])

        self.scientific_notation_lower = '%e' % Decimal(str(self.lower)) # e.g. 0.000000e+00
        self.scientific_notation_upper = '%e' % Decimal(str(self.upper)) # e.g. 8.000000e+01
        self.scientific_notation = self.scientific_notation_lower + "-" + self.scientific_notation_upper
        self.simplified_scientific_notation_lower, self.simplified_scientific_notation_upper = self.get_simplified_scientific_notation() # e.g. 0e+00, 8e+01
        self.simplified_scientific_notation = self.simplified_scientific_notation_lower + "-" + self.simplified_scientific_notation_upper

    def _is_range(self,token):

        flag=token.pos_ in ["NUM", "NOUN","PROPN"] and (like_num(token.text) or token.text in maps["scales"].keys() or token.text in maps["suffixes"].keys() or token.text in maps["string_num_map"].keys() or token.text in maps["fractions"])
        return flag


    def __str__(self):
        return f"{self.lower}-{self.upper}"

    def __bool__(self):
        return bool(self.lower is not None and self.upper is not None)

    def get_str_value(self, idx_value=0):
        """return the value of the range or lower or upper bound as string"""
        if len(self.char_indices) == 1: # e.g. dozens, millions, thousands etc.
            lower = str(int(self.lower)) if self.lower - int(self.lower) == 0 else str(self.lower)
            upper = str(int(self.upper)) if self.upper - int(self.upper) == 0 else str(self.upper)
            return f"{lower}-{upper}"
        if idx_value == 0: # lower
            return str(int(self.lower)) if self.lower - int(self.lower) == 0 else str(self.lower)
        # else upper
        return str(int(self.upper)) if self.upper - int(self.upper) == 0 else str(self.upper)

    def get_simplified_scientific_notation(self,as_string=False):
        """return a simplified scientific notation of the lower and upper bound"""
        notation_lower = '%e' % Decimal(str(self.lower))
        notation_upper = '%e' % Decimal(str(self.upper))

        return notation_lower.split('e')[0].rstrip('0').rstrip('.') + 'e' + notation_lower.split('e')[1], notation_upper.split('e')[0].rstrip('0').rstrip('.') + 'e' + notation_upper.split('e')[1]


class Value:
    def __init__(self, value, span=[], overload=False):
        self.span = sorted(span, key=lambda t: t.i)
        self.value = value

        self.char_indices = [(span[0].idx, span[-1].idx + len(span[-1].text))] if span else [] if overload else None

        self.scientific_notation = '%e' % Decimal(self.__str__()) # e.g. 4.000000e-03
        self.simplified_scientific_notation = self.get_simplified_scientific_notation() # e.g. 4e-03
        self.scientific_notation_of_compl_expr = None # default

    def __str__(self):
        return str(self.value)

    def __bool__(self):
        return bool(self.value is not None)

    def get_str_value(self, idx_value=0):
        """return the numerical value as string"""
        if self.value - int(self.value) == 0:
            return str(int(self.value))
        return str(self.value)

    def get_simplified_scientific_notation(self,as_string=False):
        """return a simplified scientific notation of the value"""
        notation = '%e' % Decimal(str(self))
        return notation.split('e')[0].rstrip('0').rstrip('.') + 'e' + notation.split('e')[1]


class Unit:
    def __init__(self, unit=[], norm_unit="-", span=[], scientific=False, keys=[], overload=False):
        self.span = sorted(span, key=lambda t: t.i) if norm_unit != "-" else []
        if len(set(token.text for token in unit)) != 1 and len(unit) > 1:
            self.unit = [token for i, token in enumerate(unit) if token.text != unit[(i+1)%len(unit)].text] # skip duplicates
        else:
            self.unit = unit

        self.norm_unit = norm_unit
        set_unit=set()

        if len(span) > 1 and sorted([token.i for token in span]) == list(range(min(token.i for token in span), max(token.i for token in span)+1)):

            self.char_indices = [(span[0].idx, span[-1].idx + len(span[-1].text))] if span else [] if overload else None
        else:
            if overload and not span:
                self.char_indices=[]
            else:
                self.char_indices=[]
                for token in span:
                    if token.text not in set_unit and not self.has_numbers(token.text):
                        set_unit.add(token.text)
                        self.char_indices.append((token.idx, token.idx + len(token.text)))

        self.char_indices.sort(key=lambda a: a[0])
        if self.char_indices and len(self.char_indices)>2:
            for i in range(len(self.char_indices)-1):
                if i+1 <len(self.char_indices) and self.char_indices[i][1]+1==self.char_indices[i+1][0]:#if they overlap them merge them
                    self.char_indices[i]=(self.char_indices[i][0],self.char_indices[i+1][1])
                    self.char_indices.remove(self.char_indices[i+1])

        if self.char_indices and len(self.char_indices)>2:
            for i in range(len(self.char_indices)-1):
                if i+1 <len(self.char_indices) and self.char_indices[i][1]+1==self.char_indices[i+1][0]:#if they overlap them merge them
                    self.char_indices[i]=(self.char_indices[i][0],self.char_indices[i+1][1])
                    self.char_indices.remove(self.char_indices[i+1])

        self.scientific = scientific if overload else None # unit type attribute, which identifies if it is a noun based unit, or a scientific unit (we find it in our dict)
        self.unit_keys = self.get_list_of_keys(keys) if overload else None # list of keys from the unit dict used for normalization
        self.unit_surfaces_forms = self.get_unit_surface_forms() if overload else None # dict of all the surface forms of the unit

    def has_numbers(self,inputString):
        return any(char.isdigit() for char in inputString)
    def __str__(self):
        return str(self.unit)

    def __bool__(self):
        return bool(self.unit)

    def __flatten(self, items):
        """yield items from nested iterable"""
        for item in items:
            if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
                for sub_item in self.__flatten(item):
                    yield sub_item
            else:
                yield item

    def get_list_of_keys(self, keys_list):
        """return a list of the keys used for the normalization of the unit"""
        keys = []
        k_list = list(self.__flatten(keys_list))
        for key in k_list:
            keys.append(key)
        return keys

    def get_unit_surface_forms(self):
        """return a dict of the surface forms for keys that form the unit"""
        if self.scientific and self.unit_keys: # in unit dictionary
            path = get_project_root()
            file_name = os.path.join(path, "unit.json")
            with open(file_name, "r", encoding="utf8") as f:
                units_dict = json.load(f)
            return { key: { "surfaces": units_dict[key].get("surfaces"), "symbols": units_dict[key].get("symbols") } for key in self.unit_keys }
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
    original_text = None # class variable containing the original sentence input
    scientific_text = None # class variable containing the normalized scientific sentence

    def __init__(self, value, change=Change(), unit=Unit(), referred_concepts=None, in_bracket=False, overload=False):
        self.change = change
        self.value = value
        self.unit = unit
        self.referred_concepts = referred_concepts if referred_concepts else []
        self.in_bracket = in_bracket # indicates whether it is a quantity in brackets

        # characters indices based on the preprocessed sentence
        self.__value_char_indices = value.char_indices
        self.__unit_char_indices = unit.char_indices

        # characters indices based on the values normalized sentence
        self.__value_char_indices_val_norm = [] if overload else None
        self.__unit_char_indices_val_norm = [] if overload else None

        # characters indices based on the unit normalized sentence
        self.__value_char_indices_unit_norm = [] if overload else None
        self.__unit_char_indices_unit_norm = [] if overload else None

        # characters indices based on the  normalized sentence
        self.__value_char_indices_norm = [] if overload else None
        self.__unit_char_indices_norm = [] if overload else None


        # characters indices based on the normalized scientific sentence
        self.__scientific_value_char_indices = [] if overload else None
        self.__scientific_unit_char_indices = [] if overload else None

        # characters indices based on the normalized scientific sentence
        self.__scientific_value_char_indices_norm = [] if overload else None
        self.__scientific_unit_char_indices_norm = [] if overload else None

        self.__preprocessed_text = None # default
        self.__normalized_text = None # default
        self.__normalized_scientific_text = None # default
        self.__normalized_text_values_only = None # default
        self.__normalized_text_units_only = None # default
        self.__normalized_scientific_text_values_only = None # default


    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"({str(self.change)},{str(self.value)},{str(self.unit.unit)},{str(self.unit.norm_unit)},{str(self.referred_concepts)})"

    def update_indices(self,old_to_new_suffixes):
        while self.__preprocessed_text.find(" minus ")!=-1:
            index_minus=self.__preprocessed_text.find(" minus ")
            for index in range(len(self.__value_char_indices)):
                quanit_idx=self.__value_char_indices[index][0]
                if quanit_idx>=index_minus:#- comes after a quantity and we shift all quantities after it
                    self.__value_char_indices[index]=self._adjust_index_both(self.__value_char_indices[index], len(" -") - len(" minus "))
                    if index_minus==quanit_idx-len(" minus "):# if the minuse belongs to the quantity
                        self.__value_char_indices[index]=(self.__value_char_indices[index][0]-1,self.__value_char_indices[index][1])
            for val_index in range(len(self.__unit_char_indices)):
                if  self.__unit_char_indices[val_index][0]>=index_minus:
                    self.__unit_char_indices[val_index]=self._adjust_index_both(self.__unit_char_indices[val_index],  len(" -") - len(" minus "))
                # if index<len(self.__unit_char_indices) and self.__unit_char_indices[index][0]>=index_minus:
                #     self.__unit_char_indices[index]=self._adjust_index_both(self.__unit_char_indices[index], len(" -") - len(" minus "))
            self.__preprocessed_text=self.__preprocessed_text[:index_minus]+" -"+self.__preprocessed_text[index_minus+len(" minus "):]

        if self.__preprocessed_text.startswith("minus "):# if the sentence is begining with a minus
            for index in range(len(self.__value_char_indices)):
                index_minus=0
                quanit_idx=self.__value_char_indices[index][0]
                if quanit_idx>=index_minus:#- comes after a quantity and we shift all quantities after it
                    self.__value_char_indices[index]=self._adjust_index_both(self.__value_char_indices[index], len("-") - len("minus "))
                    if index_minus==quanit_idx-len("minus "):# if the minuse belongs to the quantity
                        self.__value_char_indices[index]=(self.__value_char_indices[index][0]-1,self.__value_char_indices[index][1])
                for val_index in range(len(self.__unit_char_indices)):
                    if  self.__unit_char_indices[val_index][0]>=index_minus:
                        self.__unit_char_indices[val_index]=self._adjust_index_both(self.__unit_char_indices[val_index],  len("-") - len("minus "))

                # if index<len(self.__unit_char_indices) and self.__unit_char_indices[index][0]>=index_minus:
                #     self.__unit_char_indices[index]=self._adjust_index_both(self.__unit_char_indices[index], len("-") - len("minus "))
                # if len(self.__value_char_indices)<len(self.__unit_char_indices) and self.__unit_char_indices[index+1][0]>=index_minus:
                #     self.__unit_char_indices[index+1]=self._adjust_index_both(self.__unit_char_indices[index+1], len("-") - len("minus "))
                self.__preprocessed_text=self.__preprocessed_text[:index_minus]+"-"+self.__preprocessed_text[index_minus+len("minus "):]


        for k,v in old_to_new_suffixes.items():
            while self.__preprocessed_text.find(k)!=-1:
                index_minus=self.__preprocessed_text.find(k)
                for index in range(len(self.__value_char_indices)):
                    quanit_idx=self.__value_char_indices[index][0]
                    if self.__value_char_indices[index][0]>=index_minus:
                        if index_minus==quanit_idx:#
                            self.__value_char_indices[index]=self._adjust_index_end(self.__value_char_indices[index], len(v) - len(k))
                        else:
                            self.__value_char_indices[index]=self._adjust_index_both(self.__value_char_indices[index], len(v) - len(k)+1)
                    for val_index in range(len(self.__unit_char_indices)):
                        if index<len(self.__unit_char_indices) and self.__unit_char_indices[val_index][0]>=index_minus:
                            self.__unit_char_indices[val_index]=self._adjust_index_both(self.__unit_char_indices[val_index],  len(v) - len(k)+1)


                self.__preprocessed_text=self.__preprocessed_text[:index_minus]+v+" "+self.__preprocessed_text[index_minus+len(k):]


    def transform_concept_to_dict(self):
        """create a Concept instance"""
        self.referred_concepts = Concept(self.referred_concepts)

    def set_original_text(self, text):
        Quantity.original_text = text

    def set_scientific_text(self, text):
        Quantity.scientific_text = text

    def _adjust_index_both(self, old_index, by):
        new_tuple_1=old_index[0]+by
        new_tuple_2=old_index[1]+by
        return (new_tuple_1,new_tuple_2)

    def _adjust_index_end(self, old_index, by):
        new_tuple_1=old_index[0]
        new_tuple_2=old_index[1]+by
        return (new_tuple_1,new_tuple_2)

    def set_preprocessed_text(self, text):
        self.__preprocessed_text = text

    def set_normalized_text(self, text):
        self.__normalized_text = text.replace(" minus "," ")
        if self.__normalized_text.startswith("minus "):
            self.__normalized_text=self.__normalized_text[len("minus "):]


    def set_normalized_text_values_only(self, text):
        self.__normalized_text_values_only = text.replace(" minus "," ")
        if self.__normalized_text_values_only.startswith("minus "):
            self.__normalized_text_values_only=self.__normalized_text_values_only[len("minus "):]

    def set_normalized_text_units_only(self, text,old_to_new_suffixes):
        for k,v in old_to_new_suffixes.items():
            text=text.replace(k,v)
        self.__normalized_text_units_only = text.replace(" minus "," -")
        if self.__normalized_text_units_only.startswith("minus "):
            self.__normalized_text_units_only=self.__normalized_text_units_only[len("minus "):]


    def set_normalized_scientific_text_with_values_only(self, text):
        self.__normalized_scientific_text_values_only = text

    def set_normalized_scientific_text(self, text):
        self.__normalized_scientific_text = text

    def set_scientific_notations(self, notations):
        if isinstance(self.value, Value) and self.value.scientific_notation_of_compl_expr is None:
            if notations and not {self.value.get_simplified_scientific_notation()} & set(notations): # complex numerical expression
                self.value.scientific_notation_of_compl_expr = notations

    def get_original_text(self):
        return Quantity.original_text

    def get_preprocessed_text(self):
        return self.__preprocessed_text

    def get_normalized_text(self):
        return self.__normalized_text

    def get_normalized_text_values_only(self):
        return self.__normalized_text_values_only

    def get_normalized_text_units_only(self):
        return self.__normalized_text_units_only

    def get_normalized_scientific_text_values_only(self):
        return self.__normalized_scientific_text_values_only

    def get_normalized_scientific_text(self):
        return self.__normalized_scientific_text

    def get_unit_surface_forms(self):
        """return list or dict of the surface forms for the Unit"""
        return self.unit.unit_surfaces_forms


    def get_char_indices(self):
        """return dictionary of the Value and Unit indices in the preprocessed sentence"""
        return { "value": self.__value_char_indices, "unit": self.__unit_char_indices }
    def get_char_indices_val_norm(self):
        """return dictionary of the Value and Unit indices in the value normliazed sentence"""
        return { "value": self.__value_char_indices_val_norm, "unit": self.__unit_char_indices_val_norm }
    def get_char_indices_unit_norm(self):
        """return dictionary of the Value and Unit indices in the unit normalized sentence"""
        return { "value": self.__value_char_indices_unit_norm, "unit": self.__unit_char_indices_unit_norm }

    def get_char_indices_norm(self):
        """return dictionary of the Value and Unit indices in the normliazed sentence"""
        return { "value": self.__value_char_indices_norm, "unit": self.__unit_char_indices_norm }

    def get_scientific_char_indices(self):
        """return dictionary of the Value and Unit indices in the  scientific sentence"""
        return { "value": self.__scientific_value_char_indices, "unit": self.__scientific_unit_char_indices }
    def get_scientific_char_indices_nromalized(self):
        """return dictionary of the Value and Unit indices in the normalized scientific sentence"""
        return { "value": self.__scientific_value_char_indices_norm, "unit": self.__scientific_unit_char_indices_norm }



    def set_scientific_char_indices(self,scientific_value_char_indices,scientific_unit_char_indices):
        """return dictionary of the Value and Unit indices in the normalized scientific sentence"""

        self.__scientific_value_char_indices=scientific_value_char_indices
        self.__scientific_unit_char_indices=scientific_unit_char_indices


    def set_scientific_char_indices_nromalized(self,scientific_value_char_indices,scientific_unit_char_indices):
        """return dictionary of the Value and Unit indices in the normalized scientific sentence"""

        self.__scientific_value_char_indices_norm=scientific_value_char_indices
        self.__scientific_unit_char_indices_norm=scientific_unit_char_indices
