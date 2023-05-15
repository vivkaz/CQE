rules = {
    "noun_num_quant" : [ # more than four times, about 127 gigatonnes, up to 400 people
        {
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {"POS": {"IN": ["PROPN", "NOUN"]}}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": ">",
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "amod", "compound", "nummod", "nmod", "advmod", "npadvmod", "dobj"]}, "POS": "NUM"}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">>",
            "RIGHT_ID": "adv",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "amod", "compound", "nummod", "nmod", "advmod", "npadvmod"]}, "POS": {"NOT_IN": ["NUM"]}}
        },
    ],

    "noun_num_quant_2" : [ # higher/lower than 8.90 ksi
        {
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {"POS": {"IN": ["PROPN", "NOUN"]}}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": ">",
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "amod", "compound", "nummod", "nmod", "advmod", "npadvmod"]}, "POS": "NUM"}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": "<<",
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {"DEP": {"IN": ["acomp", "amod","punct"]}, "POS": "ADJ"}
        }
    ],

    "noun_num_quant_3" : [ # higher than 130
        {
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "amod", "compound", "nummod", "nmod", "advmod", "npadvmod", "pobj"]}, "POS": "NUM"}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": "<<",
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {"DEP": {"IN": ["acomp", "amod"]}, "POS": "ADJ", "ORTH": {"IN": ["higher", "lower"]}}
        }
    ],

    "noun_num_quant_4" : [ # ~80%
        {
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "amod", "compound", "nummod", "nmod", "advmod", "npadvmod", "pobj"]}, "POS": "NUM"}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ";",
            "RIGHT_ID": "sign",
            "RIGHT_ATTRS": {"DEP": {"IN": ["punct", "attr"]}, "POS": {"IN": ["PUNCT", "X", "SYM"]}, "ORTH": {"IN": ["~","=","<",">", "≈"]}}
        }
    ],

    "noun_num_right_noun" : [
        {
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {"POS": {"IN": ["PROPN", "NOUN"]}}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": ">",
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"DEP": "nummod", "POS": "NUM"}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": ">",
            "RIGHT_ID": "right_noun",
            "RIGHT_ATTRS": {"DEP": {"IN": ["pobj"]}, "POS": {"IN": ["PROPN", "NOUN"]}}
        },
    ],

    "noun_num_adp_right_noun" : [ # 22 miles per hour, 13% of Aeronautics, 127 gigatonnes of mass, 60% of workers
        {
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {"POS": {"IN": ["PROPN", "NOUN"]}}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": ">",
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"DEP": "nummod", "POS": "NUM"}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": ">",
            "RIGHT_ID": "adp",
            "RIGHT_ATTRS": {"DEP": {"IN": ["prep", "appos"]}, "POS": {"IN": ["ADP"]}, "ORTH": {"IN": ["of", "per"]}}
        },
        {
            "LEFT_ID": "adp",
            "REL_OP": ">",
            "RIGHT_ID": "right_noun",
            "RIGHT_ATTRS": {"DEP": {"IN": ["pobj", "appos"]}, "POS": {"IN": ["PROPN", "NOUN", "ADJ"]}}
        },
    ],

    "num_num_adp_right_noun" : [ # One fourth of population
        {
            "RIGHT_ID": "number1",
            "RIGHT_ATTRS": {"POS": {"IN": ["NUM"]}}
        },
        {
            "LEFT_ID": "number1",
            "REL_OP": ">",
            "RIGHT_ID": "number2",
            "RIGHT_ATTRS": {"DEP": "nummod", "POS": "NUM"}
        },
        {
            "LEFT_ID": "number1",
            "REL_OP": ">",
            "RIGHT_ID": "adp",
            "RIGHT_ATTRS": {"DEP": {"IN": ["prep", "appos"]}, "POS": {"IN": ["ADP"]}, "ORTH": {"IN": ["of", "per"]}}
        },
        {
            "LEFT_ID": "adp",
            "REL_OP": ">",
            "RIGHT_ID": "right_noun",
            "RIGHT_ATTRS": {"DEP": {"IN": ["pobj", "appos"]}, "POS": {"IN": ["PROPN", "NOUN"]}}
        },
    ],


    "num_symbol" : [ # $62, $ billion, US$ million
        {
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"POS": "NUM"}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "symbol",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "nmod"]}, "POS": "SYM", "ORTH": {"NOT_IN": ["#"]}}
        },
    ],

    "symbol_num" : [
        {
            "RIGHT_ID": "symbol",
            "RIGHT_ATTRS": {"POS": "SYM"}
        },
        {
            "LEFT_ID": "symbol",
            "REL_OP": ";",
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"POS": "NUM"}
        },
    ],

    "noun_num" : [ # 0.4 percent, 127 gigatonnes, 2.1%, six years, million contract (from $48.2 million army contract)
        {
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {"POS": {"IN": ["PROPN", "NOUN", "INTJ", "ADJ"]}, "ORTH": {"NOT_IN": ["#", "ers"]}}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": ">",
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "nmod", "nummod", "compound", "amod", "nsubj"]}, "POS": "NUM"}
        },
    ],

    "num_noun" : [ # four times, 6.5 times, 20 percent
        {
            "RIGHT_ID": "num",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "nmod", "nummod", "attr"]}, "POS": "NUM"}
        },
        {
            "LEFT_ID": "num",
            "REL_OP": ">",
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {"POS": {"IN": ["PROPN", "NOUN"]}}
        },
    ],

    "num_direct_propn" : [ # 2.4 Ghz, 30 Mbps, 60 GB, 450 Nm, 0.1 m2, 50 mL
        # 575 Willbraham, 8 AB8, 30 TFSI
        {
            "RIGHT_ID": "num",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "nmod", "nummod"]}, "POS": "NUM"}
        },
        {
            "LEFT_ID": "num",
            "REL_OP": ".",
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {"POS": {"IN": ["PROPN"]}}
        },
    ],

    "lonely_num" : [ # just the number: 21, thirteen, million
        {
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"POS": "NUM"}
        }
    ],

    "noun_compound_num" : [ # 15 year-old students
        {
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {"POS": {"IN": ["PROPN", "NOUN"]}}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": ">",
            "RIGHT_ID": "compound",
            "RIGHT_ATTRS": {"DEP": "compound", "POS": {"IN": ["ADJ"]}}
        },
        {
            "LEFT_ID": "compound",
            "REL_OP": ">",
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "amod", "nummod", "nmod"]}, "POS": "NUM"}
        },
    ],

    "noun_adj_num" : [ # double 120 decibels, 600 people worldwide, 6kW electric motor, million Australian dollars, 14 first-team players
        # billion hostile takeover, million year-old bedrock
        {
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {"POS": {"IN": ["PROPN", "NOUN"]}}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": ">",
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "amod", "compound", "nummod", "nmod", "advmod", "npadvmod"]}, "POS": {"IN": ["NUM"]}}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": ">",
            "RIGHT_ID": "adv",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "amod", "compound", "nummod", "nmod", "advmod", "npadvmod"]},
                            "POS": {"IN": ["ADJ", "SCONJ", "ADV", "PART", "ADP", "DET", "SYM"]}}
        },
    ],

    "adj_num" : [ # 300,000 year-old, 5 kW
        {
            "RIGHT_ID": "adv",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "amod", "nummod", "nmod"]}, "POS": {"IN": ["ADJ"]}}
        },
        {
            "LEFT_ID": "adv",
            "REL_OP": ">",
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "amod", "compound", "nummod", "nmod", "advmod", "npadvmod"]}, "POS": "NUM"}

        },
    ],

    "adj_noun_num" : [ # 30 years old, ten months old, 90 day old
        {
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {"DEP": {"IN": ["acomp", "amod"]}, "POS": {"IN": ["ADJ"]}, "ORTH": {"IN": ["old"]}}
        },
        {
            "LEFT_ID": "adj",
            "REL_OP": ">",
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {"DEP": {"IN": ["npadvmod"]}, "POS": {"IN": ["NOUN"]}}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": ">",
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "nmod", "nummod", "compound"]}, "POS": "NUM"}
        }
    ],
    "num_direct_noun" : [ # 87 yo, 50 year-old, 50 mph, 8.5 kW, 0.6 percent/%, one league
        {
            "RIGHT_ID": "num",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "nmod", "nummod"]}, "POS": "NUM"}
        },
        {
            "LEFT_ID": "num",
            "REL_OP": ".",
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {"DEP": {"IN": ["compound"]}, "POS": {"IN": ["NOUN"]}}
        },
    ],

    "num_quantmod" : [ # around/about 600, just/only 799, over/under thousand, more than four times*, up to 400*
        # also between 62 (between $62 and $68)
        # * as many as 14 (14,as;14,many;14,as -> as separate matches)
        {
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"POS": "NUM"}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "quant",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "amod", "compound", "nummod", "nmod", "advmod", "npadvmod"]},
                            "POS": {"IN": ["ADJ", "SCONJ", "ADV", "PART", "ADP", "DET"]}}
        },
    ],

    "minus_num" : [ # minus 0.1
        {
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"POS": "NUM"}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": "<",
            "RIGHT_ID": "quant",
            "RIGHT_ATTRS": {"ORTH": "minus"}
        },
    ],

    "minus_num_2" : [ # minus 1.7%
        {
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {"POS": "NOUN"}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": ">",
            "RIGHT_ID": "quant",
            "RIGHT_ATTRS": {"ORTH": "minus"}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": ">",
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"POS": "NUM"}
        },
    ],

    "quantmod_direct_num" : [ # averaging 15.9, rose/fell 0.6%, lost/gained 286 gigatonnes
        # close one, was five, totalling twenty, plummeted 190, climbed 0.1%
        # down 0.03%
        {
            "RIGHT_ID": "num",
            "RIGHT_ATTRS": {"POS": {"IN": ["NUM"]}}
        },
        {
            "LEFT_ID": "num",
            "REL_OP": ";",
            "RIGHT_ID": "quant",
            "RIGHT_ATTRS": {"DEP": {"IN": ["advcl", "advmod", "amod", "conj", "acl", "prep", "prt", "ROOT", "pcomp", "ccomp", "xcomp"]}, "POS": {"IN": ["VERB", "ADV", "ADP"]}}
        },
    ],
    "num_quantmod_chain" : [ # just under 296, at least 74
        {
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"POS": "NUM"}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "quant",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "amod", "compound", "nummod", "nmod", "advmod", "npadvmod"]},
                            "POS": {"IN": ["ADJ", "SCONJ", "ADV", "PART", "ADP", "DET"]}}
        },
        {
            "LEFT_ID": "quant",
            "REL_OP": ">",
            "RIGHT_ID": "quant2",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "amod", "compound", "nummod", "nmod", "advmod", "npadvmod"]},
                            "POS": {"IN": ["ADJ", "SCONJ", "ADV", "PART", "ADP", "DET"]}} # ADDED DET
        },
    ],

    "verb_quantmod_num" : [ # fell by 25
        {
            "RIGHT_ID": "num",
            "RIGHT_ATTRS": {"POS": "NUM"}
        },
        {
            "LEFT_ID": "num",
            "REL_OP": ";",
            "RIGHT_ID": "quant",
            "RIGHT_ATTRS": {"ORTH": {"IN": ["by"]}}
        },
        {
            "LEFT_ID": "quant",
            "REL_OP": "<",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {"POS": {"IN": ["VERB"]}}
        }
    ],

    "verb_quantmod2_num" : [ # evised upward by Y=40 billion
        {
            "RIGHT_ID": "num",
            "RIGHT_ATTRS": {"POS": "NUM"}
        },
        {
            "LEFT_ID": "num",
            "REL_OP": ";",
            "RIGHT_ID": "quant",
            "RIGHT_ATTRS": {"ORTH": {"IN": ["by"]}}
        },
        {
            "LEFT_ID": "quant",
            "REL_OP": "<",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {"POS": {"IN": ["VERB"]}}
        },
        {
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "quant2",
            "RIGHT_ATTRS": {"DEP": {"IN": ["advmod"]},"POS": {"IN": ["ADV"]}}
        }
    ],

    "num_right_noun" : [ # $62 and $68 per share (62 share; 68 share -> separate matches)
        {
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"POS": "NUM"}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">>",
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {"DEP": {"IN": ["prep", "pobj"]}, "POS": {"IN": ["PROPN", "NOUN"]}}
        },
    ],

    # If it behaves like a number, its probably a number
    "noun_noun" : [ # tens of thousands, 75 kwW system, 68 league games, 50 mph winds
        # but also state and local tax credits, iPhone Mini deal, most cloud-to-ground
        {
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {"POS": {"IN": ["PROPN", "NOUN"]}}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": ">",
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "nummod", "nmod","compound"]}, "POS": {"IN": ["PROPN", "NOUN"]}}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "quantifier",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "amod", "nummod", "nmod", "advmod", "npadvmod"]}}
        },
    ],

    "noun_noun_2" : [ # 40 yard line
        {
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {"POS": {"IN": ["PROPN", "NOUN"]}}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": ">",
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "nummod", "nmod", "compound", "npadvmod"]}, "POS": {"IN": ["PROPN", "NOUN"]}}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": ">",
            "RIGHT_ID": "quantifier",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "nummod", "nmod", "advmod", "npadvmod"]}}
        },
    ],

    "num_num" : [ # 7.2 billion, 13.6 million, 1 trillion
        # 1.0 l, 2.7 m, 7 s
        {
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"POS": "NUM", "_": {"like_scale": True}}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "number2",
            "RIGHT_ATTRS": {"DEP": {"IN": ["compound", "nummod"]}, "POS": {"IN": ["NUM"]}}
        },
    ],

    "num_to_num" : [ # 40 to 50, 100 to 200
        {
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"POS": "NUM"}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ".",
            "RIGHT_ID": "part",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "prep"]}, "POS": {"IN":["PART", "ADP"]}, "ORTH": {"IN": ["to"]}}
        },
        {
            "LEFT_ID": "part",
            "REL_OP": ".",
            "RIGHT_ID": "number2",
            "RIGHT_ATTRS": {"POS": {"IN": ["NUM"]}}
        }
    ],

    "num_to_num_2" : [ # $920 to $1730 a week
        {
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"POS": "NUM", "IS_DIGIT": True}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ".",
            "RIGHT_ID": "part",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "prep"]}, "POS": {"IN":["PART", "ADP"]}, "ORTH": {"IN": ["to"]}}
        },
        {
            "LEFT_ID": "part",
            "REL_OP": ">",
            "RIGHT_ID": "number2",
            "RIGHT_ATTRS": {"POS": {"IN": ["NUM"]}}
        }
    ],

    "num_to_num_3" : [ # 32 million to 33 million, 15 billion to 16 billion euros
        {
            "RIGHT_ID": "number", # 32
            "RIGHT_ATTRS": {"POS": "NUM", "IS_DIGIT": True}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": "<",
            "RIGHT_ID": "number2", # million
            "RIGHT_ATTRS": {"POS": "NUM", "_": {"like_scale": True}}
        },
        {
            "LEFT_ID": "number2",
            "REL_OP": ".",
            "RIGHT_ID": "part", # to
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "prep"]}, "POS": "PART", "ORTH": {"IN": ["to"]}}
        },
        {
            "LEFT_ID": "part",
            "REL_OP": "<",
            "RIGHT_ID": "number3", # million
            "RIGHT_ATTRS": {"POS": {"IN": ["NUM"]}, "_": {"like_scale": True}}
        },
        {
            "LEFT_ID": "number3",
            "REL_OP": ">",
            "RIGHT_ID": "number4", # 33
            "RIGHT_ATTRS": {"POS": "NUM", "IS_DIGIT": True}
        },
    ],

    "num_to_num_4" : [ # 90 billion to 100 billion
        {
            "RIGHT_ID": "number", # billion
            "RIGHT_ATTRS": {"POS": "NUM",  "_": {"like_scale": True}}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "number2", # 90
            "RIGHT_ATTRS": {"POS": "NUM", "IS_DIGIT": True}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "number3", # 100
            "RIGHT_ATTRS": {"POS": "NUM", "IS_DIGIT": True}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "number4", # billion
            "RIGHT_ATTRS": {"POS": "NUM", "_": {"like_scale": True}}
        },
        {
            "LEFT_ID": "number4",
            "REL_OP": ".",
            "RIGHT_ID": "part", # to
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "prep"]}, "POS": "PART", "ORTH": {"IN": ["to"]}}
        }
    ],

    "num_to_num_5" : [ # $48 to $55
        {
            "RIGHT_ID": "number", # 55
            "RIGHT_ATTRS": {"POS": "NUM"}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "part", # to
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "prep"]}, "POS": {"IN":["PART", "ADP"]}, "ORTH": {"IN": ["to"]}}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "number2", # 48
            "RIGHT_ATTRS": {"POS": {"IN": ["NUM"]}}
        }
    ],

    "num_to_num_dig" : [ # 5-10
        {
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"POS": "NUM", "IS_DIGIT": True}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ".",
            "RIGHT_ID": "part",
            "RIGHT_ATTRS": {"ORTH": {"IN": ["-"]}}
        },
        {
            "LEFT_ID": "part",
            "REL_OP": ".",
            "RIGHT_ID": "number2",
            "RIGHT_ATTRS": {"POS": {"IN": ["NUM"]}, "IS_DIGIT": True}
        },

    ],

    "num_to_num_num" : [ # comment out
        {
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"POS": "NUM", "_": {"like_scale": False}}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "part",
            "RIGHT_ATTRS": {"ORTH": {"IN": ["to"]}}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "number2",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod"]}, "POS": {"IN": ["NUM"]}}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "number_compound",
            "RIGHT_ATTRS": {"DEP": {"IN": ["compound"]}, "POS": {"IN": ["NUM"]}}
        },

    ],

    "num_to_num_num_dig" : [ # comment out
        {
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"POS": "NUM", "IS_DIGIT": True}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "part",
            "RIGHT_ATTRS": {"ORTH": {"IN": ["-"]}}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "number2",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod"]}, "POS": {"IN": ["NUM"]}, "IS_DIGIT": True}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "number_compound",
            "RIGHT_ATTRS": {"DEP": {"IN": ["compound"]}, "POS": {"IN": ["NUM"]}}
        },

    ],

    "range_single" : [ # thousands of students/dollars/deals, millions of Americans, dozens of people/vehicles
        {
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"POS": "NOUN", "_": {"like_number": True}}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "of",
            "RIGHT_ATTRS": {"DEP": {"IN": ["prep"]}, "ORTH": {"IN": ["of"]}}
        },
        {
            "LEFT_ID": "of",
            "REL_OP": ">",
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {"DEP": {"IN": ["pobj"]}, "POS": {"IN": ["NOUN", "PROPN"]}}
        },

    ],
    "range_double" : [
        {
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"POS": "NOUN", "_": {"like_number": True}}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "of",
            "RIGHT_ATTRS": {"DEP": {"IN": ["prep"]}, "ORTH": {"IN": ["of"]}}
        },
        {
            "LEFT_ID": "of",
            "REL_OP": ">",
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {"DEP": {"IN": ["pobj"]}, "POS": {"IN": ["NOUN"]}, "_": {"like_number": True}}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "of2",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod"]}, "ORTH": {"IN": ["of"]}}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "number2",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod"]}, "POS": {"IN": ["NOUN"]}}
        },
    ],
    "frac" : [ # one of four
        {
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"POS": "NUM"}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ".",
            "RIGHT_ID": "of",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "prep"]}, "ORTH": {"IN": ["of"]}}
        },
        {
            "LEFT_ID": "of",
            "REL_OP": ".",
            "RIGHT_ID": "number2",
            "RIGHT_ATTRS": {"DEP": {"IN": ["pobj", "appos", "nummod"]}, "POS": {"IN": ["NUM"]}}
        },
    ],

    "frac_2" : [ # three out of five, one out of three
        {
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"POS": "NUM"}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ".",
            "RIGHT_ID": "out",
            "RIGHT_ATTRS": {"ORTH": {"IN": ["out"]}}
        },
        {
            "LEFT_ID": "out",
            "REL_OP": ".",
            "RIGHT_ID": "of",
            "RIGHT_ATTRS": {"ORTH": {"IN": ["of"]}}
        },
        {
            "LEFT_ID": "of",
            "REL_OP": ".",
            "RIGHT_ID": "number_2",
            "RIGHT_ATTRS": {"POS": {"IN": ["NUM"]}}
        }
    ],

    "unit_frac" : [ # 68 per share, miles per hour, million per worker
        {
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"POS": {"IN": ["NUM", "NOUN"]}}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "per",
            "RIGHT_ATTRS": {"ORTH": {"IN": ["per", "a", "an"]}}
        },
        {
            "LEFT_ID": "per",
            "REL_OP": ">",
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN"]}}
        },
    ],

    "unit_frac_2" : [ # 14.95 a month, 1730 a week
        {
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"POS": {"IN": ["NUM"]}}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN"]}}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": ">",
            "RIGHT_ID": "per",
            "RIGHT_ATTRS": {"ORTH": {"IN": ["per", "a", "an"]}}
        },
    ],

    "unit_frac_3" : [ # 21.25 per square foot
        {
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"POS": {"IN": ["NUM"]}}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "per",
            "RIGHT_ATTRS": {"ORTH": {"IN": ["per"]}}
        },
        {
            "LEFT_ID": "per",
            "REL_OP": ">",
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN"]}}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": ">",
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {"POS": {"IN": ["ADJ"]}}
        },
    ],
    "unit_frac_4" : [ # parts per million, tests per day, miles per hour
        {
            "RIGHT_ID": "noun", # parts
            "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN", "SYM"]}}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": ">",
            "RIGHT_ID": "per", # per
            "RIGHT_ATTRS": {"DEP": {"IN": ["prep"]}, "ORTH":{"IN": ["per", "a", "an"]}}
        },
        {
            "LEFT_ID": "per",
            "REL_OP": ">",
            "RIGHT_ID": "number2", # million
            "RIGHT_ATTRS": {"POS": {"IN": ["NUM", "NOUN", "PROPN"]}}
        },
    ],

    "unit_frac_5" : [ # 9 billion shekels ($2.408 billion) a year, RM a month, barrels a day
        {
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN"]}}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": ">",
            "RIGHT_ID": "appos",
            "RIGHT_ATTRS": {"ORTH":{"IN": ["year", "month", "day", "hour", "minute", "second"]}}
        },
        {
            "LEFT_ID": "appos",
            "REL_OP": ">",
            "RIGHT_ID": "per",
            "RIGHT_ATTRS": {"POS": {"IN": ["DET"]}, "ORTH":{"IN": ["per", "a", "an"]}}
        },
    ],

    "unit_frac_6" : [ # 50 hours a week
        {
            "RIGHT_ID": "noun", # hours
            "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN"]}}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": ">",
            "RIGHT_ID": "noun2", # week
            "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN"]}}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": ".",
            "RIGHT_ID": "per", # a
            "RIGHT_ATTRS": {"ORTH":{"IN": ["per", "a", "an"]}}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": ";",
            "RIGHT_ID": "number", # 50
            "RIGHT_ATTRS": {"POS": {"IN": ["NUM"]}}
        }
    ],

    "dimensions_3" : [
        {
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {"POS": {"IN": ["PROPN", "NOUN"]}}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": ">",
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "nmod", "nummod"]}, "POS": "NUM"}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": "$+",
            "RIGHT_ID": "punkt",
            "RIGHT_ATTRS": {"POS": "SYM"}
        },
        {
            "LEFT_ID": "punkt",
            "REL_OP": "$+",
            "RIGHT_ID": "number2",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "nmod", "nummod"]}, "POS": "NUM"}
        },
        {
            "LEFT_ID": "number2",
            "REL_OP": "$+",
            "RIGHT_ID": "punkt2",
            "RIGHT_ATTRS": {"POS": "SYM"}
        },

        {
            "LEFT_ID": "punkt2",
            "REL_OP": "$+",
            "RIGHT_ID": "number3",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "nmod", "nummod"]}, "POS": "NUM"}
        }
    ],

    "dimensions_2" : [
        {
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {"POS": {"IN": ["PROPN", "NOUN"]}}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": ">",
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "nmod", "nummod"]}, "POS": "NUM"}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": "$+",
            "RIGHT_ID": "punkt",
            "RIGHT_ATTRS": {"POS": "SYM"}
        },
        {
            "LEFT_ID": "punkt",
            "REL_OP": "$+",
            "RIGHT_ID": "number2",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod", "nmod", "nummod"]}, "POS": "NUM"}
        }

    ],

    "noun_quant_noun_noun" : [ # 256gb screeen size, 50 foot building lots, ... billion dalkon shield claimants, ... million world bank loan 
        {
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {"POS": {"IN": ["PROPN", "NOUN"]}}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": ">",
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"DEP": "nummod", "POS": "NUM"}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": ">",
            "RIGHT_ID": "noun2",
            "RIGHT_ATTRS": {"DEP":"compound","POS":"NOUN" }
        },
        {
            "LEFT_ID": "noun2",
            "REL_OP": ">",
            "RIGHT_ID": "noun3",
            "RIGHT_ATTRS": {"DEP":"compound","POS":"NOUN" }
        }
    ],

    "adp_num_cconj_num" : [ # between 50 and 70, from 13.5 to 14.0
        {
            "RIGHT_ID": "adp",
            "RIGHT_ATTRS": {"ORTH": {"IN": ["between", "from"]}}
        },
        {
            "LEFT_ID": "adp",
            "REL_OP": ">",
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"POS": {"IN": ["NUM"]}}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "cconj",
            "RIGHT_ATTRS": {"ORTH": {"IN": ["and", "to"]}}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "number2",
            "RIGHT_ATTRS": {"POS": {"IN": ["NUM"]}}
        },
    ],

    "adp_num_cconj_num_2" : [ # from 5.7% to 3.4%, from 2 dollar to 3 dollar
        # from (up to) 252 miles to 287 miles-
        {
            "RIGHT_ID": "adp",
            "RIGHT_ATTRS": {"ORTH": {"IN": ["from"]}}
        },
        {
            "LEFT_ID": "adp",
            "REL_OP": ">",
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "SYM"]}}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": ">",
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"POS": {"IN": ["NUM"]}}
        },
        {
            "LEFT_ID": "adp",
            "REL_OP": ">",
            "RIGHT_ID": "adp2",
            "RIGHT_ATTRS": {"ORTH": {"IN": ["to"]}}
        },
        {
            "LEFT_ID": "adp2",
            "REL_OP": ">",
            "RIGHT_ID": "noun2",
            "RIGHT_ATTRS": {"DEP": {"IN": ["pobj"]}}
        },
        {
            "LEFT_ID": "noun2",
            "REL_OP": ">",
            "RIGHT_ID": "number2",
            "RIGHT_ATTRS": {"POS": {"IN": ["NUM"]}}
        },
    ],

    "adp_num_cconj_num_3" : [ # Fresh fruit prices are expected to increase between 5% and 6%, with dairy prices expected to increase between 4% and 5% and fats and oils between 6% and 7%.
        {
            "RIGHT_ID": "noun2",
            "RIGHT_ATTRS": {"POS": {"IN": ["NOUN"]}}
        },
        {
            "LEFT_ID": "noun2",
            "REL_OP": ">",
            "RIGHT_ID": "number2",
            "RIGHT_ATTRS": {"POS": {"IN": ["NUM"]}}
        },
        {
            "LEFT_ID": "noun2",
            "REL_OP": "<",
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {"POS": {"IN": ["NOUN"]}}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": ">",
            "RIGHT_ID": "adp2",
            "RIGHT_ATTRS": {"ORTH": {"IN": ["and"]}}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": "<",
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"POS": {"IN": ["NUM"]}}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "adp",
            "RIGHT_ATTRS": {"ORTH": {"IN": ["between"]}}
        }
    ],

    "adp_num_cconj_num_with_scale" : [ # between 60m and 90m subscribers
        {
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"POS": {"IN": ["NUM"]}}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "number2",
            "RIGHT_ATTRS": {"POS": "NUM"}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "adp",
            "RIGHT_ATTRS": {"ORTH": {"IN": ["and"]}}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "scale",
            "RIGHT_ATTRS": {"DEP": {"IN": ["quantmod"]}, "POS": {"IN": ["NOUN"]}}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": "<",
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {"POS": {"IN": ["PROPN", "NOUN", "SYM"]}}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": "<",
            "RIGHT_ID": "adp2",
            "RIGHT_ATTRS": {"ORTH": {"IN": ["between"]}}
        }
    ],

    "adp_num_cconj_num_with_unit" : [ # between $62 and $68 per share, between 4 and 5
        {
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"POS": {"IN": ["NUM"]}, "ORTH": {"NOT_IN": [ "thousand", "million", "billion", "trillion"]}, "LIKE_NUM": True}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "number2",
            "RIGHT_ATTRS": {"POS": {"IN": ["NUM"]}, "LIKE_NUM": True}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "adp",
            "RIGHT_ATTRS": {"ORTH": {"IN": ["between"]}}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "adp2",
            "RIGHT_ATTRS": {"ORTH": {"IN": ["and"]}}
        }
    ],

    "adp_num_cconj_num_with_unit_2" : [ # between 20 HZ and 20,000 HZ
        {
            "RIGHT_ID": "number", # 20
            "RIGHT_ATTRS": {"POS": {"IN": ["NUM"]}}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "adp", # between
            "RIGHT_ATTRS": {"ORTH": {"IN": ["between"]}}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": "<",
            "RIGHT_ID": "noun", # HZ
            "RIGHT_ATTRS": {"POS": {"IN": ["PROPN", "NOUN", "SYM"]}}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": ">",
            "RIGHT_ID": "adp2", # and
            "RIGHT_ATTRS": {"ORTH": {"IN": ["and"]}}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": ">",
            "RIGHT_ID": "noun2", # HZ
            "RIGHT_ATTRS": {"POS": {"IN": ["PROPN", "NOUN", "SYM"]}}
        }
    ],

    "adp_num_cconj_num_with_unit_3" : [ # from 2,415 to 2,315, from 160 to 200
        {
            "RIGHT_ID": "number2",
            "RIGHT_ATTRS": {"POS": "NUM", "_": {"like_number": True}}
        },
        {
            "LEFT_ID": "number2",
            "REL_OP": "<",
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"POS": "NUM", "_": {"like_number": True}}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": "<",
            "RIGHT_ID": "adp",
            "RIGHT_ATTRS": {"ORTH": {"IN": ["from"]}}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "adp2",
            "RIGHT_ATTRS": {"ORTH": {"IN": ["to"]}}
        }
    ],

    "adp_num_cconj_num_with_unit_4" : [ # between 15kHz and 17
        {
            "RIGHT_ID": "adp",
            "RIGHT_ATTRS": {"ORTH": {"IN": ["between"]}}
        },
        {
            "LEFT_ID": "adp",
            "REL_OP": ">",
            "RIGHT_ID": "number1",
            "RIGHT_ATTRS": {"POS": {"IN": ["NUM"]}, "_": {"like_number": True}}
        },
        {
            "LEFT_ID": "adp",
            "REL_OP": ">",
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {"POS": {"IN": ["PROPN", "NOUN", "SYM"]}}
        },
        {
            "LEFT_ID": "noun",
            "REL_OP": ">",
            "RIGHT_ID": "adp2",
            "RIGHT_ATTRS": {"ORTH": {"IN": ["and"]}}
        },
        {
            "LEFT_ID": "adp2",
            "REL_OP": ".",
            "RIGHT_ID": "number2",
            "RIGHT_ATTRS": {"POS": {"IN": ["NUM"]}, "_": {"like_number": True}}
        }
    ],

    "adp_num_cconj_num_with_unit_5" : [ #between US$ 160 million and US$ 171 million, between 160 million and 171
        {
            "RIGHT_ID": "number", # million
            "RIGHT_ATTRS": {"POS": {"IN": ["NUM"]}, "_": {"like_scale": True}}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "number2", # 160
            "RIGHT_ATTRS": {"POS": "NUM", "IS_DIGIT": True}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "adp",
            "RIGHT_ATTRS": {"ORTH": {"IN": ["between"]}}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "adp2",
            "RIGHT_ATTRS": {"ORTH": {"IN": ["and"]}}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "number3", # million
            "RIGHT_ATTRS": {"POS": "NUM", "_": {"like_scale": True}}
        },
        {
            "LEFT_ID": "number3",
            "REL_OP": ">",
            "RIGHT_ID": "number4", # 171
            "RIGHT_ATTRS": {"POS": "NUM", "IS_DIGIT": True}
        },
    ],

    "adp_num_cconj_num_with_unit_6" : [ # between ¥ 60 and ¥ 110, between 5% and 6%
        {
            "RIGHT_ID": "adp",
            "RIGHT_ATTRS": {"ORTH": {"IN": ["between"]}}
        },
        {
            "LEFT_ID": "adp",
            "REL_OP": "<",
            "RIGHT_ID": "number",
            "RIGHT_ATTRS": {"POS": {"IN": ["NUM"]}}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "adp2",
            "RIGHT_ATTRS": {"ORTH": {"IN": ["and"]}}
        },
        {
            "LEFT_ID": "number",
            "REL_OP": ">",
            "RIGHT_ID": "sym",
            "RIGHT_ATTRS": {"POS": {"IN": ["SYM", "NOUN"]}}
        },
        {
            "LEFT_ID": "sym",
            "REL_OP": ">",
            "RIGHT_ID": "number2",
            "RIGHT_ATTRS": {"POS": {"IN": ["NUM"]}}
        }
    ],

    "compound_num": [ # five hundred
        {
            "RIGHT_ID": "num",
            "RIGHT_ATTRS": {"POS": {"IN": ["NUM"]}, "IS_DIGIT": False}
        },
        {
            "LEFT_ID": "num",
            "REL_OP": ">",
            "RIGHT_ID": "num2",
            "RIGHT_ATTRS": {"POS": {"IN": ["NUM"], "DEP": {"IN": ["nummod", "compound", "quantmod"]}}, "IS_DIGIT": False}
        },
    ],
    "compound_num_2": [ # [thousand, million, three] from "three million five hundred thousand"
        {
            "RIGHT_ID": "num",
            "RIGHT_ATTRS": {"POS": {"IN": ["NUM"]}, "IS_DIGIT": False}
        },
        {
            "LEFT_ID": "num",
            "REL_OP": ">",
            "RIGHT_ID": "num2",
            "RIGHT_ATTRS": {"POS": {"IN": ["NUM"], "DEP": {"IN": ["nummod", "compound", "quantmod"]}}, "IS_DIGIT": False}
        },
        {
            "LEFT_ID": "num2",
            "REL_OP": ">",
            "RIGHT_ID": "num3",
            "RIGHT_ATTRS": {"POS": {"IN": ["NUM"], "DEP": {"IN": ["nummod", "compound", "quantmod"]}}, "IS_DIGIT": False}
        },
    ],

    "one_of": [ # one of the computers, one of the proudest moments in his life
        {
            "RIGHT_ID": "num",
            "RIGHT_ATTRS": {"POS": "NUM"}
        },
        {
            "LEFT_ID": "num",
            "REL_OP": ">",
            "RIGHT_ID": "of",
            "RIGHT_ATTRS": {"POS": {"IN": ["ADP"]}, "ORTH": {"IN": ["of"]}}
        },
        {
            "LEFT_ID": "of",
            "REL_OP": ">",
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {"POS": {"IN": ["NOUN"]}}
        }
    ],

    "phone_number_pattern_1" : [ # (0) 20 111 2222
        {"ORTH": "("},
        {"SHAPE": "d"},
        {"ORTH": ")"},
        {"SHAPE": "dd"},
        {"ORTH": "-", "OP": "?"},
        {"SHAPE": "ddd"},
        {"ORTH": "-", "OP": "?"},
        {"SHAPE": "dddd"}
    ],

    "phone_number_pattern_2" : [ # (0)20 111 2222
        {"ORTH": "("},
        {"TEXT": {"REGEX": r"[\d]\)[\d]*"}},
        {"ORTH": "-", "OP": "?"},
        {"SHAPE": "ddd"},
        {"ORTH": "-", "OP": "?"},
        {"SHAPE": "dddd"}
    ],

    "phone_number_pattern_3" : [ # (123) 4567 8901, (123) 4567-8901
        {"ORTH": "("},
        {"SHAPE": "ddd"},
        {"ORTH": ")"},
        {"SHAPE": "dddd"},
        {"ORTH": "-", "OP": "?"},
        {"SHAPE": "dddd"}
    ],

    "phone_number_pattern_4" : [ # (123) 456 789
        {"ORTH": "("},
        {"SHAPE": "ddd"},
        {"ORTH": ")"},
        {"SHAPE": "ddd"},
        {"ORTH": "-", "OP": "?"},
        {"SHAPE": "ddd"}
    ],

    "phone_number_pattern_5" : [ # + AA (AAAA) BBBBBBB (international numbers)
        {"ORTH": "+"},
        {"SHAPE": "dd"},
        {"ORTH": "(", "OP": "?"},
        {"SHAPE": "dddd"},
        {"ORTH": ")", "OP": "?"},
        {"TEXT": {"REGEX": r"\d{4,7}"}}
    ],

    "zip_number_pattern" : [ # 90049 ZIP Code
        {"TEXT": {"REGEX": r"\d{5}"}},
        {"ORTH": "ZIP"}
    ],

    #
    #     "noun_quant_punkt" : [#~10tw/h
    #         {
    #             "RIGHT_ID": "adp",
    #             "RIGHT_ATTRS": {"POS":"ADP"},
    #         },
    #
    #         {
    #             "RIGHT_ID": "noun",
    #             "RIGHT_ATTRS": {"DEP": "pobj"},
    #             "LEFT_ID": "adp",
    #             "REL_OP": ";*"}
    # ]
}
