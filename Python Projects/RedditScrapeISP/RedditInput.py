"""
Reddit Input Pulls posts from r/wallstreetbets and writes them in Data.csv if a stock is mentioned

Methods:
    postCheck(submission)
        checks a submission for a stock
    checkSub()
        checks wallstreetbets for new posts
    reCheckSub()
        checks the top 1000 posts for posts with stocks

Current Flaws:
    Needs to be walked along, is not fully automated
"""
from logging import exception
import praw
import time
import pandas as pd
import Methods as m



reddit = praw.Reddit("bot1")
wallstreetbets = reddit.subreddit("wallstreetbets")

# conditions: not a meme/shitpost
# starts with a dollar sign or is all caps
# check the submission itself as well

def postCheck(submission): 
    """Checks if a post has a valid stock in its text"""
    exceptions = ["A", "YOLO", "AI", "Y", "LMAO"]
    if(submission.link_flair_text != "Meme" and submission.link_flair_text != "Shitpost"):
        submissiontext = str(submission.title + submission.selftext).split()
        for words in submissiontext:
            # a word is upper case, or has a dollar sign and no letters, and is not I
            if(words != "I" and (words.isupper() == True or (words.find("$") != -1 and words.isnumeric == False))):
                words = m.simplify(words)
                #print(words)
                if(words not in exceptions):
                    if(m.get_current_price(words) != 0):
                        #print("Stocks: " + words)
                        return words

    return False

# rows [
# submission id for later retrieval
# stock name
# upvotes
# stock price
# time stamp in minutes since 1970, can be used to calculate exact times
# ]

def checkSub():
    """
    Checks wallstreetbets for posts that mention stocks in their text
    Writes the postdata of selected posts to Data.csv
    
    """
    print("checking reddit...")

    start = time.time()

    df = pd.DataFrame(
                {
                'ID': [],
                'Upvotes': [],
                'Stock': [],
                'Stockprice': [],
                'Timestamp': [],
                }
            )

    for submission in wallstreetbets.stream.submissions(skip_existing = True):
        stock = postCheck(submission)
        if(stock != False):
            print("adding new post: " + submission.title + " stock " + stock)
            df.loc[len(df.index)] = submission.id, submission.score, stock, m.get_current_price(stock), int((time.time()/60))
        # print(df)
        if(time.time() - start > 3600):
            df.to_json("Raw.json")
        
            

def reCheckSub():
    df = pd.DataFrame(
                {
                'ID': [],
                'Upvotes': [],
                'Stock': [],
                'Stockprice': [],
                'Timestamp': [],
                }
            )
    for submission in wallstreetbets.new():
        stock = postCheck(submission)
        if(stock != False):
            print("adding new post: " + submission.title + " stock " + stock)
            df.loc[len(df.index)] = submission.id, submission.score, stock, m.get_current_price(stock), int((time.time()/60))
    print(df)
    df.to_json("Raw.json")
                