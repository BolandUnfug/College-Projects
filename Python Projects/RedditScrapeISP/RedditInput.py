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
import praw
import csv
import time
import Methods as m



reddit = praw.Reddit("bot1")
wallstreetbets = reddit.subreddit("wallstreetbets")

# conditions: not a meme/shitpost
# starts with a dollar sign
# check the submission itself as well

# do I use yfinance to check if it is a stock? would save lines but would the program take much

def postCheck(submission): 
    """Checks if a post has a valid stock in its text"""
    if(submission.link_flair_text != "Meme" and submission.link_flair_text != "Shitpost"):
        submissiontext = str(submission.title + submission.selftext).split()
        for words in submissiontext:
            if(words.startswith("$") == True or words.isupper() == True):
                words = words[1:]
                if(m.get_current_price(words) != 0):
                    if(words != "A" and words != "I"):
                        return words
    
    return False

# Close data.csv somehow
# break the with line into its components

# rows [
# reddit post title
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
    timer = time.time()
    csvfile = open('Raw.csv', 'w+', newline="",encoding="utf8")
    submissionlist = csv.writer(csvfile)
    for submission in wallstreetbets.stream.submissions(skip_existing = True):
        print(submission.title)
    # for submission in wallstreetbets.new():
        stock = postCheck(submission)
        if(stock != False):
            stock = m.removeemoji(stock)
            if(m.get_current_price(stock) != 0):
                print("adding new post: " + submission.title)
                rows = [[submission.title], [submission.id], [stock], [submission.score], [m.get_current_price(stock)], [int(time.time()/60)]]
                submissionlist.writerow(rows)
            if(time.time() - timer > 3600):
                csvfile.close()
                print("The file is closed:" + csvfile.closed())
                return True

def reCheckSub():
    timer = time.time()
    csvfile = open('Raw.csv', 'w+', newline="",encoding="utf8")
    submissionlist = csv.writer(csvfile)
    # header = ['Title,ID,Stock,Upvotes,Stockprice,Timestamp']
    # submissionlist.writerow(header)
    for submission in wallstreetbets.new():
        stock = postCheck(submission)
        if(stock != False):
            stock = m.removeemoji(stock)
            if(m.get_current_price(stock) != 0):
                print("adding new post: " + submission.title)
                rows = [[submission.title], [submission.id], [stock], [submission.score], [m.get_current_price(stock)], [int(time.time()/60)]]
                submissionlist.writerow(rows)
    csvfile.close()
    if(csvfile.closed == True):
        print("closed the file")
        return True
                