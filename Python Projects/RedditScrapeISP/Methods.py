"""
Methods is a support class that stores methods used in other classes
Methods:
    simplify(row)
        Removes unwanted characters
    get_current_price(symbol)
        Gets the current price of a stock, returning a 2 decimal rounded result
    clean(word)
        Removes all non-alpha characters
"""

import yfinance as yf

def simplify(word, exceptions):
    """
    Removes specific characters from a word.
    Designed for strings, but can simplify numbers as well.

    Parameters
    -----------
    (String) word: any input, I change the type to string.

    Returns
    -------
    String cleanword: word, but with exceptions removed.
    """
    word = str(word) # turns word to a string
    cleanword = ""
    for character in word: 
        if(character not in exceptions):
            cleanword = cleanword + character # appends characters that are not in exceptions
    return cleanword

def get_current_price(symbol):
    """
    Collects the current price of a stock.
    Doubles as a stock checker, if symbol is not found, it will throw an exception.

    Parameters
    -----------
    (String) symbol: any stock, in the form of a string.

    Returns
    -------
    Current stock price, rounded to 2 decimal points.

    Exceptions
    -----------
    Will throw 0 if the stock data is not found.
    """
    try:
        ticker = yf.Ticker(symbol) # creates a new stock object, using stock symbol
        todays_data = ticker.history(period='1d') # collects todays stock prices
        return round(todays_data['Close'][0],2) # rounds the latest stock price to 2 decimals
    except:
        return 0 # if the first condition threw an error, return 0

def clean(word):
    """
    Removes all non alpha characters.

    This works if UTF-8 encoding is used on the string.

    Parameters
    -----------
    (String) word: any valid string

    Returns
    -------
    String cleanword: word, but with non alpha characters removed.
    """
    word = str(word) # turns word to a string
    cleanword = ""
    for x in word:
        if(x.isalpha() == True):
            cleanword = cleanword + x
    return cleanword