unit_conversion = {
    # Time (smaller subdivisions of time)
    "millisecond": {
        "second": 0.001,
        "minute": 0.0166667,
        "hour": 2.77778e-7
    },
    "second": {
        "millisecond": 1000,
        "minute": 1/60,
        "hour": 1/3600
    },
    "minute": {
        "millisecond": 60000,
        "second": 60,
        "hour": 1/60
    },
    "hour": {
        "millisecond": 3.6e+6,
        "second": 3600,
        "minute": 60
    },

    # Time (representing longer durations of time)
    "day": {
        "week": 1/7,
        "month": 1/30.4167,
        "year": 1/365
    },
    "week": {
        "day": 7,
        "month": 1/4.34524,
        "year": 1/52.1429
    },
    "month": {
        "day": 30.4167,
        "week": 4.34524,
        "year": 1/12
    },
    "year": {
        "day": 365,
        "week": 52.1429,
        "month": 12
    },

    # Currency
    "dollar": {
        "cent": 100,
        "euro": 0.93
    },
    "euro": {
        "cent": 100,
        "dollar": 1.08
    },
    "cent": {
        "dollar": 1/100,
        "euro": 1/100
    },

    # Length
    "millimetre": {
        "inch": 1/25.4,
        "centimetre": 0.1,
        "mile": 0.00000062137,
        "kilometre": 0.000001,
        "metre": 0.001,
        "decimetre": 0.01,
        "yard": 0.0010936132983377
    },
    "centimetre": {
        "inch": 1/2.54,
        "millimetre": 10,
        "mile": 0.0000062137,
        "kilometre": 0.00001,
        "metre": 0.01,
        "decimetre": 0.1,
        "yard": 0.010936132983377
    },
    "decimetre": {
        "inch": 3.93701,
        "millimetre": 100,
        "centimetre": 10,
        "mile": 0.0000621371,
        "kilometre": 0.0001,
        "metre": 0.1,
        "yard": 0.10936132983377
    },
    "metre": {
        "inch": 39.3701,
        "millimetre": 1000,
        "centimetre": 100,
        "mile": 0.000621371,
        "kilometre": 0.001,
        "decimetre": 10,
        "yard": 1.0936132983377
    },
    "kilometre": {
        "inch": 39370.1,
        "millimetre": 1000000,
        "centimetre": 100000,
        "mile": 0.621371,
        "metre": 1000,
        "decimetre": 10000,
        "yard": 1093.6132983377
    },
    "mile": {
        "inch": 63360,
        "millimetre": 1609344,
        "centimetre": 160934.4,
        "kilometre": 1.60934,
        "metre": 1609.34,
        "decimetre": 16093.44,
        "yard": 1760
    },
    "inch": {
        "millimetre": 25.4,
        "centimetre": 2.54,
        "mile": 0.000015783,
        "kilometre": 0.0000254,
        "metre": 0.0254,
        "decimetre": 0.254,
        "yard": 0.027777777777778,
        "foot": 0.0833333
    },
    "yard": {
        "mile": 0.00056818181818182,
        "inch": 36,
        "millimetre": 914.4,
        "centimetre": 91.44,
        "kilometre": 0.0009144,
        "metre": 0.9144,
        "decimetre": 9.144,
        "foot": 3
    },
    "foot": {
        "inch": 12,
        "yard": 1/3
    },

    # Weight
    "pound-mass": {
        "ounce": 16,
        "kilogram": 0.453592,
        "gram": 453.592
    },
    "ounce": {
        "pound-mass": 1/16,
        "kilogram": 0.0283495,
        "gram": 28.3495
    },
    "kilogram": {
        "pound-mass": 2.20462,
        "ounce": 35.274,
        "gram": 1000
    },
    "gram": {
        "pound-mass": 0.00220462,
        "ounce": 0.035274,
        "kilogram": 0.001
    },
    "litre": {
        "millilitre": 1000
    },
    "millilitre": {
        "litre": 1/1000
    }
}
