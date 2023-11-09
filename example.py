from CQE.CQE import CQE

'''
Note: Use the first return statement in the __repr__ function of the Quantity class in classes.py
'''

if __name__ == '__main__':
    parser = CQE()

    text = "The sp 500 was down 2.1% and nasdaq fell 2.5%."
    print(text)
    result = parser.parse(text)
    print(result)

    parser2 = CQE(overload=True) # use the overload option for additional functionality
    result = parser2.parse(text)
    for res in result:
        print(res.get_char_indices())
        if len (res.get_char_indices()['unit'])>0:


            print(f"""
            Quantity: {res}
            =====
            original text                   =   {res.get_original_text()}
            value                           =   {res.value}
            unit                            =   {res.unit.norm_unit}

            preprocessed text               =   {res.get_preprocessed_text()}
            indices on preprocessed text    =   {res.get_char_indices()}
            value indices text              =   {res.get_preprocessed_text()[res.get_char_indices()['value'][0][0]:res.get_char_indices()['value'][0][1]]}
            unit indices text               =   {res.get_preprocessed_text()[res.get_char_indices()['unit'][0][0]:res.get_char_indices()['unit'][0][1]]}
            
            
            normalized text                 =   {res.get_normalized_text()}
            normalized text units only      =   {res.get_normalized_text_units_only()}
            normalized text values only     =   {res.get_normalized_text_values_only()}
           
            scientific notation             =   {res.value.scientific_notation}
            simplified scientific notation  =   {res.value.simplified_scientific_notation}
            norm scientific text            =   {res.get_normalized_scientific_text()}
            sci value indices text          =   {res.get_normalized_scientific_text()[res.get_scientific_char_indices_nromalized()['value'][0][0]:res.get_scientific_char_indices_nromalized()['value'][0][1]]}
            sci unit indices text           =   {res.get_normalized_scientific_text()[res.get_scientific_char_indices_nromalized()['unit'][0][0]:res.get_scientific_char_indices_nromalized()['unit'][0][1]]}
    
            norm scientific values only     =   {res.get_normalized_scientific_text_values_only()}
            sci value indices text          =   {res.get_normalized_scientific_text_values_only()[res.get_scientific_char_indices()['value'][0][0]:res.get_scientific_char_indices()['value'][0][1]]}
            sci unit indices text           =   {res.get_normalized_scientific_text_values_only()[res.get_scientific_char_indices()['unit'][0][0]:res.get_scientific_char_indices()['unit'][0][1]]}
            scientific unit                 =   {res.unit.scientific}
            unit surfaces forms             =   {res.unit.unit_surfaces_forms}""")
        else:

            print(f"""
            Quantity: {res}
            =====
            original text                   =   {res.get_original_text()}
            value                           =   {res.value}
            unit                            =   {res.unit.norm_unit}

            preprocessed text               =   {res.get_preprocessed_text()}
            indices on preprocessed text    =   {res.get_char_indices()}
            value indices text              =   {res.get_preprocessed_text()[res.get_char_indices()['value'][0][0]:res.get_char_indices()['value'][0][1]]}
            
            
            normalized text                 =   {res.get_normalized_text()}
            normalized text units only      =   {res.get_normalized_text_units_only()}
            normalized text values only     =   {res.get_normalized_text_values_only()}
           
            scientific notation             =   {res.value.scientific_notation}
            simplified scientific notation  =   {res.value.simplified_scientific_notation}
            norm scientific text            =   {res.get_normalized_scientific_text()}
            sci value indices text          =   {res.get_normalized_scientific_text()[res.get_scientific_char_indices_nromalized()['value'][0][0]:res.get_scientific_char_indices_nromalized()['value'][0][1]]}
    
            norm scientific values only     =   {res.get_normalized_scientific_text_values_only()}
            sci value indices text          =   {res.get_normalized_scientific_text_values_only()[res.get_scientific_char_indices()['value'][0][0]:res.get_scientific_char_indices()['value'][0][1]]}
            scientific unit                 =   {res.unit.scientific}
            unit surfaces forms             =   {res.unit.unit_surfaces_forms}""")


