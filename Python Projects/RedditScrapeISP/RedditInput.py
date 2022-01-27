"""
Pulls posts from r/wallstreetbets and writes them in Raw.json if a stock is mentioned.

Methods:
--------
    postCheck(submission)
        Checks a submission for a stock.
    checkSub()
        Checks wallstreetbets for new posts.
    reCheckSub()
        Checks the top 1000 posts for posts with stocks.

Production
----------
Version 1.1

Boland Unfug

January 26th, 2022
"""
import praw
import time
import pandas as pd
import Methods as m


reddit = praw.Reddit("bot1") # creates a new reddit instance
wallstreetbets = reddit.subreddit("wallstreetbets") # creates a subreddit instance, of wallstreet bets


def postCheck(submission): 
    """
    Collects the current price of a stock.
    Doubles as a stock checker, if symbol is not found, it will throw an exception.

    Parameters
    -----------
    (any) submission: A reddit submission

    Returns
    -------
    The first valid instance of a stock, otherwise returns false

    Does not get multiple stocks from the same post

    Conditions
    -----------
    Post is not a meme or shitpost

    Word has either capital letters or a dollar sign

    Word is not in stock exeptions (see function)

    Word returns a valid stock price
    """
    stockexceptions = ["A", "YOLO", "AI", "Y", "LMAO", "I"] # Words that correspond to stocks, but usually are just capitalized
    if(submission.link_flair_text != "Meme" and submission.link_flair_text != "Shitpost"): # checks flair for meme and shitpost
        submissiontext = str(submission.title + submission.selftext).split() # splits title and text into a list of words
        for words in submissiontext:
            if(words.isupper() == True or words.find("$") != -1): # checks for uppercase or $
                words = m.clean(words) # removes non alpha characters
                if(words not in stockexceptions): # checks the word for exceptions
                    if(m.get_current_price(words) != 0): # checks the stock
                        return words

    return False

# rows [
# submission id for later retrieval
# upvotes
# stock name
# stock price
# time stamp in minutes since 1970, can be used to calculate exact times
# ]

def checkSub():
    """
    Checks wallstreetbets for new posts that mention stocks in their text.
    Writes the postdata of selected posts to Raw.json.
    """
    print("checking reddit...")

    start = time.time() # starts a timer

    df = pd.DataFrame( # creates a new dataframe, with empty labels
                {
                'ID': [],
                'Upvotes': [],
                'Stock': [],
                'Stockprice': [],
                'Timestamp': [],
                }
            )

    for submission in wallstreetbets.stream.submissions(skip_existing = True): # continuously checks for new posts
        print(str(time.time()-start))
        stock = postCheck(submission) # checks for stocks
        if(stock != False):
            print("adding new post: " + submission.title + " stock " + stock + " at " + str(time.time() - start) + " time.")
            # add a new row with the post data
            df.loc[len(df.index)] = submission.id, submission.score, stock, m.get_current_price(stock), int((time.time()/60))
        if((time.time() - start) > 1800): # if a certain amount of time has passed
            print(df)
            df.to_json("Raw.json") # update Raw.json, overwriting it
        
            
def reCheckSub():
    """
    Checks the top 1000 posts in wallstreetbets for any that mention stocks.
    Writes the post data to Raw.json
    """
    df = pd.DataFrame( # create a new dataframe, with empty labels
                {
                'ID': [],
                'Upvotes': [],
                'Stock': [],
                'Stockprice': [],
                'Timestamp': [],
                }
            )
    for submission in wallstreetbets.new(): # checks 1000 new posts
        stock = postCheck(submission) # checks for stocks
        if(stock != False):
            print("adding new post: " + submission.title + " stock " + stock)
            # add a new row with the post data
            df.loc[len(df.index)] = submission.id, submission.score, stock, m.get_current_price(stock), int((time.time()/60))
    
    print(df)
    df.to_json("Raw.json") # update Raw.json, overwriting it