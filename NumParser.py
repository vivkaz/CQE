import itertools
import logging
import re
import spacy

from spacy.matcher import DependencyMatcher, Matcher
from spacy.tokens import Token
from spacy.lang.lex_attrs import is_digit
from spacy.lang.lex_attrs import like_num as sp_like_num
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, HYPHENS
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex

from fuzzywuzzy import fuzz

import NumberNormalizer
from rule_set.rules import rules
from rule_set.number_lookup import maps, suffixes
from classes import Change, Value, Range, Unit, Quantity

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG) # change default level

SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
OBJECTS = ["dobj", "dative", "attr", "oprd"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December", "Jan.", "Feb.", "Aug.", "Sept.", "Oct", "Nov.", "Dec."]


def lk_num(text):
    if sp_like_num(text):
        return True
    if text in maps["scales"]:
        return True
    return False

def lk_scale(text):
    if text in maps["scales"]:
        return True
    return False

def clean_up(unit, nouns):
    for noun in nouns.copy():
        if noun in unit:
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
        if doc[token].dep_ in ["quantmod"] and doc[token].pos_ in ["NUM"]:
            doc[token]._.set("number", True)
        elif doc[token].lemma_ in (maps["scales"].keys() | maps["fractions"].keys()):
            doc[token]._.set("number", True)
        else:
            doc[token]._.set("bound", True)

# noinspection PyProtectedMember
@remove_match
def lonely_num(matcher, doc, i, matches):
    match_id, tokens = matches[i]
    doc[tokens[0]]._.set("number", True)
    if doc[tokens[0]].lemma_ in maps["scales"]:
        doc[tokens[0]]._.set("scale", True)

# noinspection PyProtectedMember
@remove_match
def default_callback(matcher, doc, i, matches):
    match_id, tokens = matches[i]
    number_index = -1
    for token in tokens:
        if doc[token].lemma_ in maps["scales"]:
            doc[token]._.set("scale", True)
        if doc[token].lemma_ in (maps["scales"].keys() | maps["fractions"].keys() | maps["string_num_map"].keys()):
            doc[token]._.set("number", True)
            number_index = token
        if doc[token].lemma_ in maps["fractions"]:
            doc[token]._.set("fraction", True)
        if doc[token].text in maps["bounds"] and number_index != -1 and number_index > token: # increased 5% vs. 75 percent increase in e-cigarette usage
            doc[token]._.set("bound", True)
    return

# noinspection PyProtectedMember
@remove_match
def range_callback(matcher, doc, i, matches):
    match_id, tokens = matches[i]
    doc[tokens[0]]._.set("number", True)
    if not doc[tokens[0]].lemma_ in maps["scales"]:
        doc[tokens[0]]._.set("range_l", True)
    doc[tokens[2]]._.set("number", True)
    if not doc[tokens[2]].lemma_ in maps["scales"]:
        doc[tokens[2]]._.set("range_r", True)

# noinspection PyProtectedMember
@remove_match
def between_range_callback(matcher, doc, i, matches): # add for quantities like 'between 100 and 300'
    match_id, tokens = matches[i]
    range_l_r = []
    bound_indices = []
    for token in tokens:
        if doc[token].pos_ in ["NUM"]:
            doc[token]._.set("number", True)
            range_l_r.append(token)
        if doc[token].pos_ in ["ADP", "CCONJ", "PART"]:
            doc[token]._.set("bound", True)
            bound_indices.append(token)
    #print([doc[j] for j in bound_indices], [doc[j] for j in range_l_r], bound_indices, range_l_r)
    bound_index_l = min(bound_indices)
    bound_index_r = max(bound_indices)
    range_l_r.sort()
    for j, num_index in enumerate(range_l_r):
        if j == 0 and bound_index_l < num_index < bound_index_r:
            #print(j, doc[num_index])
            if not doc[num_index]._.range_r and not doc[num_index]._.range_l: # check if already set
                doc[num_index]._.set("range_l", True)
                #print("L", doc[num_index])
            else:
                matches[i][1].remove(num_index)
                break
        elif bound_index_r < num_index and not doc[num_index]._.range_l: # e.g. from 12.3 inches to between 12.7 and 13 # j == 1?
            #print(j,doc[num_index])
            doc[num_index]._.set("range_r", True)
            #print("R", doc[num_index])
        #else:
        #    print("no", j, doc[num_index], bound_index_l, num_index, bound_index_r)


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
    for j,token in enumerate(tokens):
        if doc[token].pos_ == "NUM":
            doc[token]._.set("number", True)
            doc[token]._.set("one_of_number", True)
        if doc[token].pos_ == "NOUN":
            doc[token]._.set("one_of_noun", True)

# noinspection PyProtectedMember
def phone_number(matcher, doc, i, matches):
    match_id, start, end = matches[i]
    span = doc[start:end]
    if span:
        for token in span:
            token._.set("ignore", True)

class NumParser:
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.nlp = spacy.load(spacy_model)
        #self.nlp.add_pipe("sentencizer")
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
        self.matcher.add("NUM_TO_NUM", [rules["num_to_num"], rules["num_to_num_2"], rules["num_to_num_dig"]], on_match=range_callback)
        self.matcher.add("ADP_NUM_CCONJ_NUM", [rules["adp_num_cconj_num"], rules["adp_num_cconj_num_2"], rules["adp_num_cconj_num_3"]], on_match=between_range_callback)
        #self.matcher.add("ADP_NUM_CCONJ_NUM_2", [rules["adp_num_cconj_num_2"]], on_match=between_range_callback)
        #self.matcher.add("ADP_NUM_CCONJ_NUM_3", [rules["adp_num_cconj_num_3"]], on_match=between_range_callback)
        #self.matcher.add("ADP_NUM_CCONJ_NUM_WITH_SCALE", [rules["adp_num_cconj_num_with_scale"]], on_match=between_range_callback) #
        self.matcher.add("ADP_NUM_CCONJ_NUM_WITH_UNIT", [rules["adp_num_cconj_num_with_unit"], rules["adp_num_cconj_num_with_unit_2"], rules["adp_num_cconj_num_with_unit_3"], rules["adp_num_cconj_num_with_unit_4"]], on_match=between_range_callback)
        #self.matcher.add("ADP_NUM_CCONJ_NUM_WITH_UNIT_2", [rules["adp_num_cconj_num_with_unit_2"]], on_match=between_range_callback)
        #self.matcher.add("ADP_NUM_CCONJ_NUM_WITH_UNIT_3", [rules["adp_num_cconj_num_with_unit_3"]], on_match=between_range_callback)
        #self.matcher.add("ADP_NUM_CCONJ_NUM_WITH_UNIT_4", [rules["adp_num_cconj_num_with_unit_4"]], on_match=between_range_callback)
        #self.matcher.add("NUM_TO_NUM_NUM", [rules["num_to_num_num"], rules["num_to_num_num_dig"]], on_match=default_callback)
        self.matcher.add("RANGE_SINGLE", [rules["range_single"]], on_match=broad_range_single)
        self.matcher.add("RANGE_DOUBLE", [rules["range_double"]], on_match=broad_range_double)
        self.matcher.add("NOUN_NUM_QUANT", [rules["noun_num_quant"]], on_match=default_callback)
        self.matcher.add("NOUN_NUM_RIGHT_NOUN", [rules["noun_num_right_noun"]], on_match=default_callback)
        self.matcher.add("NOUN_NUM_ADP_RIGHT_NOUN", [rules["noun_num_adp_right_noun"]], on_match=default_callback)
        self.matcher.add("NUM_NUM_ADP_RIGHT_NOUN", [rules["num_num_adp_right_noun"]], on_match=default_callback)
        self.matcher.add("NUM_SYMBOL", [rules["num_symbol"]], on_match=default_callback)
        self.matcher.add("SYMBOL_NUM", [rules["symbol_num"]], on_match=default_callback)
        self.matcher.add("NUM_DIRECT_PROPN", [rules["num_direct_propn"]], on_match=default_callback)
        self.matcher.add("NUM_DIRECT_NOUN", [rules["num_direct_noun"]], on_match=default_callback)
        self.matcher.add("NOUN_NUM", [rules["noun_num"]], on_match=default_callback)
        self.matcher.add("NOUN_COMPOUND_NUM", [rules["noun_compound_num"]], on_match=default_callback)
        self.matcher.add("NOUN_ADJ_NUM", [rules["noun_adj_num"]], on_match=default_callback)
        self.matcher.add("ADJ_NUM", [rules["adj_num"]], on_match=default_callback)
        self.matcher.add("ADJ_NOUN_NUM", [rules["adj_noun_num"]], on_match=default_callback)
        self.matcher.add("NUM_QUANTMOD", [rules["num_quantmod"], rules["num_quantmod_chain"]], on_match=num_quantmod)
        self.matcher.add("QUANTMOD_DIRECT_NUM", [rules["quantmod_direct_num"], rules["quantmod_direct_num"]], on_match=num_quantmod)
        #self.matcher.add("NUM_RIGHT_NOUN", [rules["num_right_noun"]], on_match=default_callback) #
        self.matcher.add("NOUN_NOUN", [rules["noun_noun"], rules["noun_noun2"]], on_match=default_callback)
        self.matcher.add("NUM_NUM", [rules["num_num"]], on_match=default_callback)
        self.matcher.add("FRAC", [rules["frac"], rules["frac_2"]], on_match=frac_callback)
        #self.matcher.add("FRAC_2", [rules["frac_2"]], on_match=frac_callback)
        self.matcher.add("UNIT_FRAC", [rules["unit_frac"], rules["unit_frac_2"]], on_match=default_callback)
        #self.matcher.add("UNIT_FRAC_2", [rules["unit_frac_2"]], on_match=default_callback)
        self.matcher.add("NUM_NOUN", [rules["num_noun"]], on_match=default_callback)
        self.matcher.add("DIM2", [rules["dimensions_2"]], on_match=default_callback)
        self.matcher.add("NOUN_QUANT_NOUN_NOUN", [rules["noun_quant_noun_noun"]], on_match=default_callback)
        self.matcher.add("ONE_OF", [rules["one_of"]], on_match=one_of_callback)
        self.matcher.add("LONELY_NUM", [rules["lonely_num"]], on_match=lonely_num)

        self.pattern_matcher = Matcher(self.nlp.vocab) # match sequences of tokens, based on pattern rules
        self.pattern_matcher.add("PHONE_NUMBER",
                    [rules["phone_number_pattern_1"], rules["phone_number_pattern_2"], rules["phone_number_pattern_3"], rules["phone_number_pattern_4"], rules["phone_number_pattern_5"]],
                    on_match=phone_number)

        if not Token.has_extension("bound"):
            Token.set_extension("bound", default=None)
        if not Token.has_extension("in_bracket"):
            Token.set_extension("in_bracket", default=None)
        if not Token.has_extension("number"):
            Token.set_extension("number", default=None)
        if not Token.has_extension("range_r"):
            Token.set_extension("range_r", default=None)
        if not Token.has_extension("range_l"):
            Token.set_extension("range_l", default=None)
        if not Token.has_extension("broad_range"):
            Token.set_extension("broad_range", default=None)
        if not Token.has_extension("ignore"):
            Token.set_extension("ignore", default=None)
        if not Token.has_extension("ignore_noun"):
            Token.set_extension("ignore_noun", default=None)
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
        if not Token.has_extension("fraction"):
            Token.set_extension("fraction", getter=lambda token: token.lemma_ in maps["fractions"]) #.keys())
        if not Token.has_extension("quantity_part"):
            Token.set_extension("quantity_part", default=None)

    def parse(self, text):
        text = self._preprocess_text(text)
        #print(text)
        doc = self.nlp(text)
        doc = self._preprocess_doc(doc)
        candidates = self._extract_candidates(doc)
        sent_boundaries = self._sentence_boundaries(doc)
        three_tuples = self._match_tuples(candidates, text, doc, sent_boundaries)
        referred_nouns = self._extract_global_referred_nouns(doc)
        self._postprocess_nouns(three_tuples, referred_nouns)
        normalized_tuples = self._normalize_tuples(three_tuples, referred_nouns, sent_boundaries, text, doc)

        return normalized_tuples
        
    def _print_tokens(self, doc):

        print([(token.text, token.dep_, token.pos_, list(token.children),token.i) for token in doc])

    def _print_temp_results(self, boundaries, candidates, tuples, nouns):
        print(f"boundaries: {boundaries}")
        print(f"candidates: {candidates}")
        print(f"tuples: {tuples}")
        print(f"nouns: {nouns}")

    def _modify_defaults_stopwords(self):
        stopwords_to_be_removed = []
        for stopword in self.nlp.Defaults.stop_words:
            if "NUM" in [token.pos_ for token in self.nlp(stopword)]: # numbers
                stopwords_to_be_removed.append(stopword)
            if stopword in ["top","us","up","to","amount","never","'s"]: # additional that should remain
                stopwords_to_be_removed.append(stopword)
        self.nlp.Defaults.stop_words -= set(stopwords_to_be_removed)
        self.nlp.Defaults.stop_words.add("total")
        self.nlp.Defaults.stop_words.add("instead")


    def _remove_stopwords_from_noun(self, nouns): # remove stop words like "a, of" etc. from referred_noun
        def accept_token(token): # accept until comma (assumption: new subsentence)
            return not token.is_punct
        removed_stopords = [token for token in nouns if not token.text.lower() in self.nlp.Defaults.stop_words]
        return list(itertools.takewhile(accept_token, removed_stopords))

    def _sentence_boundaries(self, doc):
        boundaries = []
        for sent in doc.sents:
            last = sent[0].i
            #flag_separate_part = False # e.g. while energy, last years worst performing sector, fell 0.94% as concerns about an economic slowdown also hit oil prices.
            for tok in sent:
                # not child of the previous token, e.g. of the total issued and outstanding shares
                # not part of range, e.g. between ... and ...
                # coordinating conjunction (e.g. and,or,but) or marker (=word marking a clause as subordinate to another clause), but not modifier of quantifier
                # not followed by a verb
                # not ( ) = or -
                if (tok not in list(doc[tok.i-1].children) and not tok._.bound and (tok.pos_ in ["CCONJ"] or tok.dep_ in ["mark"]) and tok.dep_ not in ["quantmod"] and (tok.i+1 < len(doc) and doc[tok.i+1].pos_ not in ["VERB"])) or (tok.pos_ in ["PUNCT"] and tok.text not in ["(", ")", "=", "-"]):
                    if last != tok.i and tok.i - last > 3:
                        if (boundaries and boundaries[-1][-1]-boundaries[-1][0]<=3):# or flag_separate_part is True:
                            #flag_separate_part = not flag_separate_part
                            boundaries[-1][-1]=tok.i
                        else:
                            boundaries.append([last, tok.i])
                    elif tok.dep_ in ["mark"]:
                        boundaries.append([tok.i, tok.i])
                    elif boundaries:
                        boundaries[-1][-1]=tok.i
                    last = tok.i + 1
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
        doc = self._tag_ignore(doc)
        return doc


    def _preprocess_text(self, text: str):
        # put a space between number and unit so spacy detects them as separate tokens
        text = re.sub(r"[\s]{2,}"," ",text) # remove multiple spaces
        text = text.replace(" | "," ,  ")
        text = text.replace(" c $"," C$") # the c $ 300 million charge -> the C$ 300 million charge
        text = re.sub(r"\b(C)\s+(\$)", r"\1\2", text) # C $6 -> C$6
        text = text.replace(" EUR "," eur ") # EUR 125 -> eur 125
        text = text.replace(" U.S. "," US ") # $75 U.S. per barrel -> $75 US per barrel

        text = re.sub(r"\[([a-z]+|[%]+)\]", r"\1", text) # [kg] or [%] -> kg and %
        text = re.sub(r"\<([a-z]+)\>", r"\1", text) # <kg> -> kg
        text = re.sub(r"\(([a-z]{2,3})\)", r"\1", text) # (kg) -> kg

        text = re.sub(r"\b(\d+)(\s*yo)\b", r"\1 years old", text) # 87 yo -> 87 years old
        text = re.sub(r"([\d]+)-[\s+|\.]", r"\1", text) # 18- to 34 year-old -> 18 to 34 year-old
        
        # transform the fractions from number_lookup.py to their corresponding numbers
        # currently different approach is implemented (see _retokenize)
        """for k, v in maps["fractions"].items():
            text = re.sub(rf"\b{k}\b", rf"{v}", text) # half -> 1/2"""

        text = re.sub(r"([a-z])\.(?!$)", r"\1", text) # e.g. m.p.h -> mph

        #text = re.sub(r"(?<![A-Z\d,.])([\d,.]+)([a-zA-Z]{1,3})", r"\1 \2", text) # 1.2mm -> 1.2 mm
        amount_unit = re.findall(r'(?<![A-Z\d,.])[0-9\.,]+[a-zA-Z]{1,3}[.,]{0,1}', text)
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

        text = re.sub(r"([\d+,.]+)\s*(sq)\s(m)\b", r"\1\2\3", text) # 200sq m -> 200 sqm

        text = re.sub(r"([\d]+)\s+(\d)/(\d)[\s+|\.]", lambda x: str(int(x.group(1)) + int(x.group(2))/int(x.group(3))), text) # 10 1/2 miles -> 10.5 miles
        text = re.sub(r"\b(\d)/(\d)[\s+|\.]", lambda x: str(int(x.group(1))/int(x.group(2))), text) # 1/2 -> 0.5

        #text = re.sub(r"\bdegF\b", "degf", text) # degF -> degree-F
        text = re.sub(r"([\d+,.]+)\s+deg\b", r"\1 degree", text) # 50 deg -> 50 degree
        text = re.sub(r"(degree|degrees)\s+([f\s+|f\.|F\s+|F\.])\b", "degf", text) # 25 degree f -> 25 degf
        text = re.sub(r"\s+(to)(-)([\d,.,,]+)", r"\1 \3", text) # e.g. 1.5 to-2 degree range -> 1.5 to 2 degree range

        text = re.sub(r"(\s)(\$)(US)", r"\1\3\2", text) # $US 113 million -> US$ 113 million
        text = re.sub(r"(u.s|US|hk|HK)(\s*)(\$)(\s*)", lambda x: x.group(1).upper() + x.group(3) + " ", text) # u.s $100,000,000 -> U.S$ 100,000,000
        text = re.sub(r"([0-9,.|million]+\s+)(us\s)", r"\1US ", text) # 15,000 us dollar -> 15,000 US dollar
        text = re.sub(r"(us)\s*(\$)", r"US \2", text) # us $ 150 million -> US $ 150 million
        text = re.sub(r"(aud|usd)(\$)", lambda x: x.group(1).upper() + x.group(2) ,text) # aud$ 15 -> AUD$ 15
        text = re.sub(r"([0-9,.]*[0-9]+)\s*([$|€])", r"\2\1", text) # 10$ -> $10
        text = re.sub(r"(?<![a-zA-Z])(M)*(sen|usd|aud|eur|rmb|gbp)\s*([0-9,.]*[0-9]+)(\s*million)*", r"\3\4 \1\2", text, flags=re.IGNORECASE) # usd 20 -> 20 usd
        text = re.sub(r"(?<![a-zA-Z])(M)([A-Z]{3})\b",r"\1 \2" , text) # MUSD -> M USD
        text = re.sub(r"([0-9,.]*[0-9]+)\s*(USD|aud|eur|rmb|gbp)(\s*million)", r"\1\3 \2", text, flags=re.IGNORECASE) # 75 USD million -> 75 million USD
        text = re.sub(r"([0-9,.]*[0-9]+\s*)(bitcoins|bitcoin)", r"\1btc", text) # 10000 bitcoins -> 10000 btc

        text = re.sub(r"([0-9,.]*[0-9]+\s*)(mB|gB|kB|tB)\b", lambda x: x.group(1) + x.group(2).upper(), text) # 10 mB -> 10 MB
        text = re.sub(r"([0-9,.]*[0-9]+\s*)(mb|Mb|gb|Gb|kb|Kb|tb|Tb)\b", lambda x: x.group(1) + x.group(2).upper()[:1]+"bit", text) # 10 mb -> 10 Mbit

        text = re.sub(r"([a-z]+)(-)(a)(-)([a-z]+)", r"\1 \3 \5", text) # 12-cent-a-share -> 12-cent a share

        text = re.sub(r"([$|€]\s?[0-9,.]*[0-9]+\s?)(m)([.| ])", r"\1M\3", text) # e.g. $5.4m. -> $5.4M.

        text = re.sub(r"\+(?!\s?\d{2}\s?[(]?\d{4}[)]?\s?\d{4,7}\b)", r"", text) # +0.2% -> 0.2%
        text = re.sub(r"minus-([\d,.,,]+)", r"-\1", text) # minus-130 -> -130
        text = re.sub(r"[\s]-([\d,.,,]+)", r" minus \1", text) # -5 -> minus 5
        text = re.sub(r"^-([\d,.,,]+)", r"minus \1", text) # when at the beginning of the line
        text = re.sub(r"sub-([\d,.,,]+)", r"under \1", text) # sub-500 sqm -> under 500 sqm

        # note: (?<![A-Z\d,.]) is needed to exclude things like PS88.3m
        text = re.sub(r"(?<![A-Z\d,.])([0-9,.]+)([a-zA-Z,\/]{2,})", r"\1 \2", text) # 14days -> 14 days
        text = re.sub(r"(\sHz|\shz)\b", lambda x: x.group(1).upper(), text) # 300 Hz -> 300 HZ
        text = re.sub(r"per cent", "percent", text)

        amount_dash_unit=re.findall(r'[0-9\.]+-[a-zA-z]+', text)
        for v in amount_dash_unit:
            text = text.replace(v, v.replace("-"," ")) # 100-km -> 100 km

        for k, v in suffixes.items():
            text = re.sub(rf"(?<![A-Z\d,.])([\d,.]+)\s?{k}([\s\n.,()]+|$)", rf"\1 {v}\2", text) # 3K -> 3 thousand
            text = re.sub(rf"([\d,.]+)\s?{v}([-])", rf"\1 {v} ", text) # 400 million-year-old -> 400 million year-old

        text = re.sub(r"([a-z,A-Z])(=)([0-9,,]+)", r"\3 \1", text) # e.g. Y=28,030 -> 28,030 Y

        # add dot and additional space before the end of the sentence if not there
        if text.endswith(' .'):
            return text
        if text.endswith(' .\n'):
            return text[:-1]
        if text.endswith('.\n'):
            return text[:-2]+' .'
        if text.endswith(' \n'):
            return text[:-1]+'.'
        if text.endswith('\n'):
            return text[:-1]+' .'
        return text+' .' if not text.endswith('.') else text[:-1]+' .'

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

            """if any(unit in noun[:-1] for unit in tuple[2]): # '[0.7, percent]'
                if len(noun[:-1])-1 <= len(noun[:-1])/2:
                    try:
                        nouns.remove(noun)
                    except ValueError:
                        continue
                else:
                    noun.remove(next(i for i in noun[:-1] if i in tuple[2]))"""

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
    def _tag_brackets(self, doc):
        # This does not work for complex cases like nested brackets, but is fine for the majority of sentences
        expression = r"\(([^\(]*)\)"
        for match in re.finditer(expression, doc.text):
            start, end = match.span()
            span = doc.char_span(start, end)
            if span:
                for token in span:
                    if token.text not in ["(", ")", "[", "]"]:
                        token._.set("in_bracket", True)
        return doc

    # noinspection PyProtectedMember
    def _tag_ignore(self, doc):
        phone_number_like = r"^[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}$"
        sequence_number_like = r"-?(?:\d+\-){2,}\d+" # 5-4-10-12-1, 5-2-2
        full_time_like = r"\d{1,2}[\s]?:[\s]?\d{0,2}[:.][\s]?\d{0,2}" # 15:06:40, 2:45.5
        time_like = r"\d{1,2}[\s]?:[\s]?\d{1,2}" # 2:00
        date_like = r"(\d{4})-(\d{2})-(\d{2})" # 2019-09-05
        name_like = r"[a-zA-Z]+\d+[.|,]*\d+[a-zA-Z]*" # PS46.6m
        ordinal = r"\d+\s*(?:st|nd|rd|th)"
        pattern = [phone_number_like, sequence_number_like, full_time_like, time_like, date_like, name_like, ordinal]
        for expr in pattern:
            for match in re.finditer(expr, doc.text):
                start, end = match.span()
                span = doc.char_span(start, end)
                if span is not None:
                    for token in span:
                        token._.set("ignore", True)
        for ent in doc.ents:
            #print(ent, ent.label_)
            if ent.label_ in ["DATE", "TIME"]:
                #print(ent, ent.label_)
                span = doc.char_span(ent.start_char, ent.end_char)
                if span is not None:
                    if any(month in span.text for month in MONTHS):
                        span[[s.text in MONTHS for s in span].index(True)]._.set("ignore_noun", True) # ignore months as nouns
                    if {"year", "month", "week", "day", "second", "minute", "hour", "daily"} & {sp.lemma_ for sp in span}:
                        continue
                    for token in span:
                        token._.set("ignore", True)

        self.pattern_matcher(doc) # phone numbers, zipcodes

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

            expression = r"\w+\/\w+"
            for match in re.finditer(expression, doc.text):
                start, end = match.span()
                span = doc.char_span(start, end)
                if span:
                    try:
                        retokenizer.merge(span, attrs={"POS": "NOUN"})
                    except ValueError:
                        logging.warning('\033[93m' + f"Can't merge non-disjoint spans. '{span}' is already part of tokens to merge." + '\033[0m')

            expression = r"[\b|\s](-)([\d,.,,]+)\b" # negative numbers: e.g. '-5'
            for match in re.finditer(expression, doc.text):
                start, end = match.span()
                span = doc.char_span(start, end)
                if span:
                    retokenizer.merge(span, attrs={"POS": "NUM"})

            expression = r"[a-zA-Z]+\d+[.|,]*\d+[a-zA-Z]*" # e.g. PS15m, PS10.5m, PS21,700
            for match in re.finditer(expression, doc.text):
                start, end = match.span()
                span = doc.char_span(start, end)
                if span:
                    try:
                        retokenizer.merge(span, attrs={"POS": "PROPN"})
                    except ValueError:
                        logging.warning('\033[93m' + f"Can't merge non-disjoint spans. '{span}' is already part of tokens to merge." + '\033[0m')

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
        for i, bound_2 in enumerate(bounds):
            if bound_1 != bound_2 and (bound_1[-1]+1==bound_2[0] or bound_2[-1]+1==bound_1[0]): # look only at different but adjacent subsenetences
                if doc[bound_1[0]].pos_ in ["SCONJ"]:
                    doc_1 = doc[bound_1[0]+1:bound_1[1]]
                else:
                    doc_1 = doc[bound_1[0]:bound_1[1]]
                doc_2 = doc[bound_2[0]:bound_2[1]]
                pos_1 = [tok.pos_ for tok in doc_1 if tok.pos_ not in ["PUNCT"]]
                pos_2 = [tok.pos_ for tok in doc_2 if tok.pos_!="PUNCT"]
                ratio = fuzz.partial_ratio(" ".join(pos_1), " ".join(pos_2))

                if ratio == 100 or (ratio >= 60 and (set(t.text for t in doc) & set(["while", "whilst", "whereas", "but"]))) and (abs((bound_1[-1]-unitless[1][0].i) - (bound_2[-1]-triples[-i][1][0].i)) < 3):
                    logging.info('\033[93m' + f"Shared unit found in \"{doc}\" for {unitless} and {triples[-i]}" + '\033[0m\n')
                    unitless[2] = triples[-i][2] # assign unit
                    return tuple(unitless)

        # else: quantities that are close to each other may also have the same unit, e.g., "It costs about 8 or $9."
        for triple in triples:

            # because of sentences like: "Australias S&P/ASX 200 costs 0.7 dollar or 6,327.80." or "Pluristem rose 8.15% to 3.65 after reporting a positive meeting"
            if bound_1[0]<=triple[-1]<=bound_1[1] and 0 < (min(triple[1][0].i, triple[2][0].i)-unitless[1][0].i) <= 4:
                # check for punct pos_ tag
                # e.g. It has an MSRP of 2,435,000 YEN ($22,480) for the 5 speed manual transmission version and 2,380,000 ($21,972) for the CVT.
                if "PUNCT" not in [token.pos_ for token in doc[unitless[1][0].i+1:min(triple[1][0].i, triple[2][0].i)]]:
                    logging.info('\033[93m' + f"Shared unit found in \"{doc}\" for {unitless} and {triple}" + '\033[0m\n')
                    unitless[2] = triple[2] # assign unit
                    return tuple(unitless)

        return unitless # no shared unit found


    # noinspection PyProtectedMember
    def _match_tuples(self, candidates, text, doc, boundaries):
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
                        if q_element.text+" "+b_list[0].text in text and not any(prep == b.text for prep in ["to", "out", "of"] for b in b_list): # distinguish ranges and fractions
                            #print("here remove ", q_element)
                            q_list.remove(q_element)
                #print(token, b_list, q_list, u_list)

            if len(u_list) > 2 and len([t_unit for t_unit in u_list if t_unit.pos_ == "PROPN"]) >= len(u_list)/2 and not any(t_unit for t_unit in u_list if t_unit.pos_ == "SYM" or len(t_unit.text)<=3):
                # e.g. 575 Wilbraham Road, but 'US $', '30 Mbps plan' or '2.4 Ghz' should be extracted
                q_list = [] # no quantity => no extraction
            
            if len(u_list) == 0 and [triple[2] for triple in triples if triple[2]]: # unitless and potential shared unit
                triple = self._shared_unit_check(doc, boundaries, [b_list, q_list, u_list, candidate[-1]], [triple for triple in triples if triple[2]])
                triples.append(triple)
            else:
                triples.append((b_list, q_list, u_list, candidate[-1])) # 4.element = index of value in doc in order to check for referred_noun later

            if len(q_list) == 3 and all(is_digit(q.text) for q in q_list): # e.g. [160, 180, 200]
                range_l = [q for q in q_list if q._.range_l]
                range_r = [q for q in q_list if q._.range_r]
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

            if b_list:
                bound.append(b_list)
            if q_list:
                quantity.append(q_list)
            if u_list:
                unit.append(u_list)

        return triples

    def _extract_global_referred_nouns(self, doc): # look for noun referred by the value
        referred_nouns = []
        referred_nouns = [[token, token.i] for token in doc if not token._.ignore_noun and token.dep_=="ROOT" and token.pos_ in ["NOUN", "PROPN"]]
        root_flag = len(referred_nouns) # appropriate noun (root of the sentence) found

        #print("ROOT",referred_nouns)
        referred_nouns = self._extend_root(referred_nouns)
        #print("ROOT EXT",referred_nouns)

        referred_nouns.extend([[token, token.i] for token in doc if not token._.ignore_noun and (token.dep_ in ["nsubj", "nsubjpass", "dobj"] and token.pos_ in ["NOUN", "PROPN", "PRON", "ADJ", "DET", "ADV"])])
        # or (token.dep_ in ["pobj"] and token.pos_ in ["NOUN", "PROPN"])])
        # pobj, appos?


        if referred_nouns:
            if root_flag:
                referred_nouns[root_flag:] = self._extend_nouns(referred_nouns[root_flag:]) # do not traverse the ROOT children, since they are usually many
            else:
                referred_nouns = self._extend_nouns(referred_nouns)
        #print("EXT",referred_nouns)
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

    def _extract_quantifiers(self, doc, tuple, index_unit):
        related_nouns = []
        flag = False # to ckeck if there was 'of' in the tuples
        for i, element in enumerate(tuple[2]):
            if element.text=="of" and index_unit not in [None, -1] and index_unit < i and not tuple[1][0]._.broad_range and i+1 < len(tuple[2]): # e.g. [%, of, workers] in '60% of the uaw workers who voted approved the contract.'
                # also check via index_unit whether whole list of tokens form the unit
                flag = True
                related_nouns.append([element, tuple[2][i+1], tuple[2][i+1].i])
                for unit in tuple[2][i:]:
                    tuple[2].remove(unit) # e.g. (=,60.0,[%, of, workers],percentage,[of, the, uaw, workers]) -> (=,60.0,[%],percentage,[of, the, uaw, workers])
                #referred_nouns.append([tuple[2][i-1], tuple[2][i-1].i]) # consider more tokens
    
        if flag:
            related_nouns = self._extend_nouns(related_nouns)
            return related_nouns

        if tuple[1]:
            index = 0
            if tuple[1][0].i + 1 < len(doc) and doc[tuple[1][0].i + 1].text == "(": # e.g. '2,380,000 ($21,972) for the CVT'
                index = 1
                while tuple[1][0].i + index < len(doc) and doc[tuple[1][0].i + index].text != ")":
                    index += 1
            if index is None:
                quantity_index = max([t.i for t in tuple[1]]+[t.i for t in tuple[2]]) # e.g. 30 million shares up for sale
            else:
                quantity_index = max([t.i for t in tuple[1]]+[t.i for t in tuple[2][:index+1]]) # e.g. 30 million shares up for sale
            distances = [(token.i - (quantity_index + index)) for token in doc if token.text in ["for", "of", "off"]] # e.g. '$52,000 for CN' # "in" ?

            if distances:
                j = 0
                if 1 in distances:
                    j = 1
                elif 2 in distances:
                    j = 2
                if j and doc[quantity_index + index + j] not in tuple[2]:
                    if [child for child in doc[quantity_index + index + j].children if (child.pos_ not in ["NUM"] and not child in tuple[2] and not child.text.lower() in self.nlp.Defaults.stop_words)]: # only if there are children, otherwise ignore (do not consider just the token 'of')
                        #related_nouns.append([doc[quantity_index + index + j]])
                        #for child in doc[quantity_index + index + j].children:
                        #    related_nouns[-1].append(child)
                        related_nouns.append(list(doc[quantity_index + index + j].children))
                        related_nouns[-1].append(quantity_index + index + j)
                        related_nouns[-1][:-1] = sorted(related_nouns[-1][:-1], key=lambda t: t.i)
                        flag = True

            if flag:
                related_nouns = self._extend_nouns(related_nouns)
                if related_nouns and len(related_nouns[0][:-1]) == 2 and any(noun for noun in related_nouns[0][:-1] if noun.pos_ == "NUM"):
                    return [] # e.g. '5 per cent penalty of $52,000'
                return related_nouns

            distances = [(tuple[1][0].i - token.i) for token in doc if token.text in ["of"]] # e.g. 'an increase of 5 percent'
            if distances:
                j = 0
                if 1 in distances:
                    j = 1
                elif 2 in distances:
                    j = 2
                elif 3 in distances:
                    j = 3
                if j:
                    related_nouns.append([doc[tuple[1][0].i-j-1]])
                    if list(doc[tuple[1][0].i-j-1].children):
                        for child in doc[tuple[1][0].i-j-1].children:
                            if child.i <= tuple[1][0].i:
                                if child._.ignore: # e.g. DATE
                                    return [] # e.g. ... growth rate since the mid 1970s of 1.7 per cent
                                related_nouns[-1].append(child)
                    related_nouns[-1].append(tuple[1][0].i-j-1)
                    related_nouns[-1][:-1] = sorted(related_nouns[-1][:-1], key=lambda t: t.i)

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

            # filter out things like iphone 11
            if self.nlp.vocab[match[0]].text in ["NOUN_NUM", "NUM_QUANTMOD"]:
                digit_part= doc[match[1][0]].text if is_digit(re.sub('[.,,]', '', doc[match[1][0]].text)) else doc[match[1][1]].text
                char_part = doc[match[1][0]].text if not is_digit(re.sub('[.,,]', '', doc[match[1][0]].text)) else doc[match[1][1]].text

                # about 1000 (NUM_QUANTMOD) should not be ignored as well as currencies
                if char_part+" "+digit_part in doc.text and "$" not in char_part and char_part!="of" and char_part.lower() not in maps["bounds"]:
                    doc[match[1][1]]._.set("ignore", True) # not extract 11 (the number, NUM) from 'iphone 11' as Quantity
                    doc[match[1][0]]._.set("ignore", True) # e.g. Big 12
                    #print("here set ignore", doc[match[1][1]], char_part, digit_part)
                    continue

            if self.nlp.vocab[match[0]].text in ["NUM_DIRECT_PROPN"]:
                char_part = doc[match[1][0]].text if not is_digit(re.sub('[.,,]', '', doc[match[1][0]].text)) else doc[match[1][1]].text
                if char_part in ["Max", "Pro", "Mini"]: # to filter out things like iphone 11 Pro
                    doc[match[1][0]]._.set("ignore", True)
                    #print("here set ignore", doc[match[1][0]])
                    continue

            # to filter out not real quantities, e.g. 'sp 500', 'BBC One'
            if self.nlp.vocab[match[0]].text == "LONELY_NUM" and match[1][0]-1 >= 0 and match[1][0]+1 < len(doc): # add index check
                if (doc[match[1][0]-1].pos_ == "NOUN" and doc[match[1][0]+1].pos_ in ["VERB", "AUX"]) or doc[match[1][0]-1].pos_ == "PROPN":
                    continue

            if doc[match[1][0]].text != "between" and abs(match[1][0] - match[1][-1]) > 4:
                continue

            for token_id in match[1]:

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
                if doc[token_id].pos_ == "NUM" or doc[token_id].text not in [cand.text for cand in candidate_res]: # [500,000, barrels, a, day, barrels, 9] -> [500,000, barrels, a, day, 9] aber '3 of 3 passes'
                    candidate_res.append(doc[token_id])
            candidate_res.append(k) # append index of value in order to check for referred_noun later
            result.append(candidate_res)

        return result

    # noinspection PyProtectedMember
    def _normalize_tuples(self, tuples, nouns, boundaries, text, doc):
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
                if len(tuple[2]) > 1 and tuple[2][0].text == "m": # e.g. 1.2m tons, 60 to 90m subscribers
                    tuple[2].remove(tuple[2][0])
                    scale_m = True


                if any(t._.range_l for t in tuple[1]) and any(t._.range_r for t in tuple[1]):
                    left = [t for t in tuple[1] if t._.range_l]
                    right = [t for t in tuple[1] if t._.range_r]
                    scales = [t for t in tuple[1] if t._.scale]
                    norm_number_l = NumberNormalizer.normalize_number_token(left)
                    norm_number_r = NumberNormalizer.normalize_number_token(right)

                    # check for successful normalization
                    if norm_number_l is None and norm_number_r is None:
                        continue  # or try something else? same for the other checks below
                    if scales:
                        norm_number_scale = 1
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
                    if scales and not any(scale in left for scale in scales) and abs(norm_number_r/norm_number_scale - norm_number_l) <= 100:
                        number_l = norm_number_l * norm_number_scale #Value(norm_number_l * norm_number_scale, left)
                    else:
                        number_l = norm_number_l
                    if scales and not any(scale in right for scale in scales) and abs(norm_number_r - norm_number_l/norm_number_scale) <= 100:
                        number_r = norm_number_r * norm_number_scale #Value(norm_number_r * norm_number_scale, right)
                    else:
                        number_r = norm_number_r


                    number = Range(number_l, number_r)

                elif any(t._.broad_range for t in tuple[1]):
                    norm_num = NumberNormalizer.normalize_number_token(tuple[1])
                    # check for successful normalization
                    if norm_num is None:
                        #print(text)
                        continue
                    number = Range(norm_num, norm_num * 10, tuple[1])

                elif any(t._.numerator for t in tuple[1]):
                    numerator = tuple[1][0]
                    denominator = tuple[1][1]
                    norm_numerator = NumberNormalizer.normalize_number_token([numerator])
                    norm_denominator = NumberNormalizer.normalize_number_token([denominator])
                    # check for successful normalization
                    if norm_numerator is None and norm_denominator is None:
                        #print(text)
                        continue
                    if norm_denominator is None:
                        number = Value(norm_numerator, [numerator, denominator])
                    elif norm_numerator is None:
                        number = Value(norm_denominator, [numerator, denominator])
                    else:
                        number = Value(norm_numerator/norm_denominator, [numerator, denominator])
                        fraction = True

                else:
                    fraction = any(t._.fraction for t in tuple[1])
                    norm_number = NumberNormalizer.normalize_number_token(tuple[1])
                    # check for successful normalization
                    if norm_number is None:
                        #print(text)
                        continue
                    if isinstance(norm_number, list): # e.g. [66.5, 107] from '66.5 x 107 mm'
                        number = Range(norm_number[0], norm_number[-1])
                    else:
                        number = Value(norm_number, tuple[1])

                if negative_number: # 5 -> -5 (-5 degree fahrenheit)
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
                norm_unit, index, in_unit_list = NumberNormalizer.normalize_unit_token(tuple[2])

                # mark tokens that are part of a quantity
                if index not in [None, -1]:
                    tuple[2][index]._.set("quantity_part", True)
                for token in tuple[1]:
                    token._.set("quantity_part", True)

                if norm_unit+" "+tuple[1][0].text in text: # for things like iphone 7
                    #print("continue", norm_unit+" "+tuple[1][0].text, tuple)
                    continue
                if self._number_in_unit(norm_unit): #or norm_unit == "_": # if there is a number in the unit do not add them
                    continue
                bound = Change(norm_bound, tuple[0])
                if fraction and not in_unit_list: # e.g. 'One out of three Germans is vaccinated'
                    unit = Unit(tuple[2], "% " + norm_unit, tuple[2])
                else:
                    unit = Unit(tuple[2], norm_unit, tuple[2])

                referred_noun = []
                ref_noun_flag = False # if ref_noun was set
                if len(unit.unit) > 1 and index not in [None, -1]: # whole list with units is not the unit

                    if index + 1 < len(unit.unit) and unit.unit[(index+1)%len(unit.unit)].text not in ["of","to"]: # skip cases like unit.unit = [of, miles] or [$, to]
                        pot_ref_nouns = list(list(token.subtree) for token in unit.unit[index+1:]) # observe the next tokens and their subtrees
                        pot_ref_nouns = sorted(list(set([token for subtree in pot_ref_nouns for token in subtree])), key=lambda t: t.i)
                        #pot_ref_nouns = list(unit.unit[(index+1)%len(unit.unit)].subtree) # observe the next token and its children
                        pot_ref_nouns = list(filter(lambda x: x not in [unit.unit[index]]+tuple[1], pot_ref_nouns)) # filter number and unit
                        pot_ref_nouns.append(pot_ref_nouns[-1].i)
                        referred_noun.append(pot_ref_nouns[:-1])
                        ref_noun_flag= True

                if not ref_noun_flag:
                    related_quantifiers = self._extract_quantifiers(doc, tuple, index)
                    if related_quantifiers:
                        referred_noun.append(related_quantifiers[0][:-1])
                        ref_noun_flag = True


                if nouns: #not ref_noun_flag and nouns: # always look for global extracted nouns in the same clause as the quantity
                    min_dist = 100
                    ref_noun_to_append = None
                    for sent_bound in boundaries:
                        if sent_bound[0] <= tuple[-1] and tuple[-1] <= sent_bound[1]:
                            for noun in nouns:
                                if sent_bound[0] <= noun[-1] and noun[-1] <= sent_bound[1]:
                                    if abs(tuple[-1]-noun[-1]) < min_dist and (not referred_noun or (referred_noun and not (set(noun[:-1]) & set(referred_noun[-1])))): # if more than one referred_noun found, take the nearest one
                                        min_dist = abs(tuple[-1]-noun[-1])
                                        ref_noun_to_append = noun[:-1]
                                        #ref_noun_flag = True
                                    #ref_noun.noun = noun[:-1]
                                    #ref_noun_flag = True
                                    #break # take the first one
                    if ref_noun_to_append is not None:
                        referred_noun.append(ref_noun_to_append)
                        ref_noun_flag = True

                #default
                if not ref_noun_flag and len(list(doc.sents)) == 1 and nouns and (len(nouns)==1 or len(tuples)==1): # when there is one input sentence and only one global noun or one quantity
                    referred_noun.append(nouns[0][:-1])
                    ref_noun_flag = True

                def postprocess(ref_noun_flag, ref_noun, number, tuple):
                    if ref_noun_flag and len(ref_noun) == 1 and ref_noun[0].pos_ in ["DET", "PRON", "NUM"]:
                        ref_noun = []  # digits, this, each, he, I, who, which etc. -> meaning that we did not find one
                    
                    if ref_noun and hasattr(number, 'value') and any(NumberNormalizer.normalize_number_token([noun])==number.value for noun in ref_noun if noun.pos_=="NUM"):
                        i = [NumberNormalizer.normalize_number_token([noun])==number.value for noun in ref_noun if noun.pos_=="NUM"].index(True)
                        ref_noun.remove([noun for noun in ref_noun if noun.pos_=="NUM"][i]) # e.g. (=,14.0,[$, rate],dollar,[the, 14, hourly, rate]) -> (=,14.0,[$, rate],dollar,[the, hourly, rate])
                    return ref_noun

                if referred_noun:
                    for i, ref_noun_list in enumerate(referred_noun):
                        ref_noun_list = postprocess(ref_noun_flag, ref_noun_list, number, tuple)
                        ref_noun_list = self._remove_stopwords_from_noun(ref_noun_list)
                        if index not in [None, -1] and len(unit.unit) > 1:# and in_unit_list:
                            unit.unit = [unit.unit[index]] # (=,5.0,[percent, penalty],percentage,[a, penalty, of]) -> (=,5.0,[percent],percentage,[a, penalty, of])
                        ref_noun_list = clean_up(unit.unit, ref_noun_list) # clean nouns
                        referred_noun[i] = ref_noun_list

                if len(tuple[1])==1 and not unit.unit and unit.norm_unit=="-" and tuple[1][0]._.one_of_number:
                    unit.unit = referred_noun[0] # replace unit with concept
                    unit.norm_unit = " ".join([t.lemma_ for t in referred_noun[0] if not t.is_stop]) # replace norm_unit with lemmatized concept + removal of stopwords
                    referred_noun = [] # concept becomes empty

                quant = Quantity(number, bound, unit.unit, unit.norm_unit, referred_noun)

                # drop if no number found
                if number: # and unit:
                    result.append(quant)

        def accept_token(token):
            return not token._.quantity_part

        for quantity in result: # cut noun if it contains tokens that are part of a quantity
            for i, ref_noun_list in enumerate(quantity.referred_concepts):
                quantity.referred_concepts[i] = list(itertools.takewhile(accept_token, ref_noun_list))
            quantity.referred_concepts = list(quantity.referred_concepts for quantity.referred_concepts, _ in itertools.groupby(quantity.referred_concepts))

            nouns_copy = quantity.referred_concepts.copy()
            for noun in nouns_copy:
                for _, noun_2 in itertools.product(noun, nouns_copy[:nouns_copy.index(noun):] + nouns_copy[nouns_copy.index(noun)+1:]):
                    if set(t.i for t in noun[:-1]).intersection(set(t.i for t in noun_2[:-1])): # looking at all neighbours
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
            quantity.transform_concept_to_dict()
        

        return result
