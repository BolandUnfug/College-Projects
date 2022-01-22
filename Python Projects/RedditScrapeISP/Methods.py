"""
Methods is a support class that stores methods used in other classes
Methods:
    simplify(row)
        Removes [, ], \", and \' from a string
    simplifyList(list)
        Removes [, ], \", and \' from a list of strings, then places them back into lists
    get_current_price(symbol)
        Gets the current price of a stock, returning a 2 decimal rounded result
    removeemoji(wordwithemoji)
        Removes emojis from a word by removing any non-letter character.
"""

import yfinance as yf

def simplify(row):
    """Removes [, ], \", and \' from a string, returning the new string """
    row = str(row)
    newstring = ""
    for x in row:
        if(x != "[" and x != "]" and x != "\"" and x != "\'"):
            newstring = newstring + x
    return newstring
    
def simplifyrow(row):
    """Removes [, ], \", and \' from a string, then returns a list of integers"""
    newstring = ""
    for x in row:
        if(x != "[" and x != "]" and x != "\"" and x != "\'"):
            newstring = newstring + x
    newlist = ([int(s) for s in newstring.split(',')])
    return newlist



def simplifylist(list):
    """Removes [, ], \", and \' from a list of strings, then returns a list of lists of integers"""
    newlist = []
    for row in list:
        newstring = ""
        for x in row:
            if(x != "[" and x != "]" and x != "\"" and x != "\'"):
                newstring = newstring + x
        newlist.append([int(s) for s in newstring.split(',')])
    return newlist

def get_current_price(symbol):
    """
    Gets the current price of a stock, returning a 2 decimal rounded result
    Flaw: rounds to a full number, so stocks that are cheap will barely change
    Reason: when appen
    """
    try:
        ticker = yf.Ticker(symbol)
        todays_data = ticker.history(period='1d')
        return round(todays_data['Close'][0],2)
    except:
        return 0

def removeemoji(wordwithemoji):
    """
    Removes emojis from a word by removing any non-letter character.
    This works if UTF-8 encoding is used on the string
    """
    unemojidword = ""
    for x in wordwithemoji:
        if(x.isalpha() == True):
            unemojidword = unemojidword + x
    return unemojidword

