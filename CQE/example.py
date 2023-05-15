from NumParser import NumParser

'''
Note: Use the first return statement in the __repr__ function of the Quantity class in classes.py
'''

if __name__ == '__main__':
    parser = NumParser()

    text = "The sp 500 was down 2.1% and nasdaq fell 2.5%."
    print(text)
    result = parser.parse(text)
    print(result)

    parser2 = NumParser(overload=True)
    text = "The sp 500 was down 2.1% and nasdaq fell 2.5%."
    result = parser2.parse(text)
    for res in result:
        print(f"""
	Quantity: {res}
	=====
	indices                         =   {res.get_char_indices()}
	normalized text                 =   {res.get_normalized_text()}
	pre processed text              =   {res.get_preprocessed_text()}
	scientific notation             =   {res.value.scientific_notation}
	simplified scientific notation  =   {res.value.simplified_scientific_notation}
	scientific unit                 =   {res.unit.scientific}
	unit surfaces forms             =   {res.unit.unit_surfaces_forms}""")
