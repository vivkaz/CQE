class Change:
    def __init__(self, change="", span=[]):
        self.span = span
        self.change = change

    def __str__(self):
        return self.change

    def __bool__(self):
        return bool(self.change)


class Range:
    def __init__(self, lower, upper, span=[]):
        self.span = span
        self.lower = lower
        self.upper = upper

    def __str__(self):
        return f"{self.lower}-{self.upper}"

    def __bool__(self):
        return bool(self.lower or self.upper) # 'or' instead of 'and' because of ranges like 0-60mph


class Value:
    def __init__(self, value, span=[]):
        self.span = span
        self.value = value

    def __str__(self):
        return str(self.value)

    def __bool__(self):
        return bool(self.value or self.value == 0) # or value == 0 because of zero-values


class Unit:
    def __init__(self, unit="-", norm_unit="-", span=[]):
        self.span = span
        self.unit = unit
        self.norm_unit = norm_unit

    def __str__(self):
        return str(self.unit)

    def __bool__(self):
        return bool(self.unit != "-")


class Concept:
    def __init__(self, noun=[], span=[]):
        self.span = span
        if noun:
            self.noun = {}
            for tokens_list in noun:
               self.noun.update({len(self.noun): tokens_list})
        else:
            self.noun = "-"

    def __str__(self):
        return str(self.noun)

    def __bool__(self):
        return bool(self.noun)

    def get_nouns(self):
        """return the referred nouns from the dict in a list form"""
        return [list(list_noun) for list_noun in self.noun.values()] if self.noun != "-" else []


class Quantity:
    def __init__(self, value, change=None, unit=None, normalized_unit=None, referred_concepts=None):
        self.change = change
        self.value = value
        self.unit = unit
        self.normalized_unit = normalized_unit
        self.referred_concepts = referred_concepts
        if not change:
            self.change = Change()
        if not unit:
            self.unit = Unit()
        if not referred_concepts:
            self.referred_concepts = [] # empty list instead of Concept()

    def __repr__(self):
        # for common use, eg. example.py, parse-examples.py
        return self.__str__()


    def __str__(self):
        return f"({str(self.change)},{str(self.value)},{str(self.unit)},{str(self.normalized_unit)},{str(self.referred_concepts)})"


    def transform_concept_to_dict(self):
        """create a Concept instance"""
        self.referred_concepts = Concept(self.referred_concepts)
