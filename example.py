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
