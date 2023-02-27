maps = {
"string_num_map" : {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "twenty-one": 21,
    "twenty-two": 22,
    "twenty-three": 23,
    "twenty-four": 24,
    "twenty-five": 25,
    "twenty-six": 26,
    "twenty-seven": 27,
    "twenty-eight": 28,
    "twenty-nine": 29,
    "thirty-one": 31,
    "thirty-two": 32,
    "thirty-three": 33,
    "thirty-four": 34,
    "thirty-five": 35,
    "thirty-six": 36,
    "thirty-seven": 37,
    "thirty-eight": 38,
    "thirty-nine": 39,
    "fourty-one": 41,
    "fourty-two": 42,
    "fourty-three": 43,
    "fourty-four": 44,
    "fourty-five": 45,
    "fourty-six": 46,
    "fourty-seven": 47,
    "fourty-eight": 48,
    "fourty-nine": 49,
    "fifty-one": 51,
    "fifty-two": 52,
    "fifty-three":53,
    "fifty-four": 54,
    "fifty-five": 55,
    "fifty-six": 56,
    "fifty-seven": 57,
    "fifty-eight": 58,
    "fifty-nine": 59,

},

"scales": {
    "dozen": 12,
    "doz": 12,
    "dozs": 12,
    "dzs": 12,
    "dz": 12,
    "hundred": 100,
    "thousand": 1_000,
    "lakh": 100_000,
    "million": 1_000_000,
    #"m": 1_000_000, # distinguish between meter and million
    "mln": 1_000_000,
    "crore": 10_000_000,
    "billion": 1_000_000_000,
    "bln": 1_000_000_000,
    "trillion": 1_000_000_000_000,
    "tln": 1_000_000_000_000
},

"fractions" : {
    "half": 1/2,
    "halve": 1/2,
    "third": 1/3,
    "two-thirds": 2/3,
    "two-third": 2/3,
    "fourth": 1/4,
    "quarter": 1/4,
    "three-quarters": 3/4,
    "three-quarter": 3/4,
    "fifth": 1/5,
    "two-fifths": 2/5,
    "two-fifth": 2/5,
    "three-fifths": 3/5,
    "three-fifth": 3/5,
    "four-fifths": 4/5,
    "four-fifth": 4/5,
    "sixth": 1/6,
    "five-sixths": 5/6,
    "five-sixth": 5/6,
    "seventh": 1/7,
    "eighth": 1/8,
    "three-eighths": 3/8,
    "three-eighth": 3/8,
    "five-eighths": 5/8,
    "five-eighth": 5/8,
    "seven-eighths": 7/8,
    "seven-eighth": 7/8,
    "ninth": 1/9,
    "two-ninths": 2/9,
    "two-ninth": 2/9,
    "four-ninths": 4/9,
    "four-ninth": 4/9,
    "five-ninths": 5/9,
    "five-ninth": 5/9,
    "seven-ninths": 7/9,
    "seven-ninth": 7/9,
    "eight-ninths": 8/9,
    "eight-ninth": 8/9,
    "tenth": 1/10,
    "twelfth": 1/12,
    "sixteenth": 1/16,
    "one thirty-second": 1/32,
    "hundredth": 1/100
},

"suffixes": {
  "K": 1_000,
  "M": 1_000_000,
  "B": 1_000_000_000,
},

"bounds": {
    # equality
    "exactly": "=",
    "just": "=",
    "equals": "=",
    "equal": "=",
    "totalling": "=",
    # approx
    "barely": "~",
    "about": "~",
    "approximately": "~",
    "average": "~",
    "averaging" : "~",
    "roughly": "~",
    "around": "~",
    "nearly": "~",
    "close": "~",
    "circa": "~",
    "~": "~",
    "~=": "~",

    # greater than
    "more": ">",
    "least": ">",
    "above": ">",
    "over": ">",
    "well over": ">",
    "greater": ">",
    "great": ">",
    "larger": ">",
    "large": ">",
    "exceeding": ">",
    "exceed": ">",
    "higher": ">",
    "high": ">",

    # less than
    "less": "<",
    "fewer": "<",
    "few": "<",
    "under": "<",
    "most": "<",
    "below": "<",
    "up to": "<",
    "smaller": "<",
    "small": "<",
    "beneath": "<",

    # down \u2198
    "fall": "down",
    "drop": "down",
    "plummet": "down",
    "down": "down",
    "lose": "down",
    "plunge": "down",
    "decline": "down",
    "decrease": "down",
    "descend": "down",
    "slide": "down",

    # up \u2197
    "rise": "up",
    "gain": "up",
    "climb": "up",
    "jump": "up",
    "increase": "up",
    "surge": "up",
    "ascend": "up",
    "up": "up",
    "advance": "up",
    "rise under": "up"

    # negation (what was the idea?)
    }
}

suffixes = {
    "B": "billion",
    "M": "million",
    #"m": "million",
    "K": "thousand",
    "k": "thousand",
    "bn": "billion",
    "tn": "trillion",
}


prefixes = {
    "y": "yocto", # 10^-24
    "z": "zepto", # 10^-21
    "a": "atto", # 10^-18
    "f": "femto", # 10^-15
    "p": "pico", # 10^-12
    "n": "nano", # 10^-9
    "μ": "micro", # 10^-6
    "\u00b5": "micro", # 10^-6
    "m": "milli", # 10^-3
    "c": "centi", # 10^-2
    "d": "deci", # 10^-1
    #"da": "deka", # 10^1
    "h": "hecto", # 10^2
    "k": "kilo", # 10^3
    "M": "mega", # 10^6
    "G": "giga", # 10^9
    "T": "tera", # 10^12
    "P": "peta", # 10^15
    "E": "exa", # 10^18
    "Z": "zetta", # 10^21
    "Y": "yotta", # 10^24
    "Gi": "gibi",
    "Ki": "kibi",
    "Mi": "mibi",
    "Ti": "tibi"
}


