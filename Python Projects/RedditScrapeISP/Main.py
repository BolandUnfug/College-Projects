"""
The main function for the other programs, use this for running the code
Process:
    Gather posts from wallstreet bets, compare with stock prices
    Update upvotes and stock prices
    Graph the top 10

TODO
Set system to pandas
Add upvotes
Make a filter for top posts
Figure out how to include decimals for stock prices
    This is important, because wallstreetbets does a lot of trading with <$50 stocks, in which a dollar increase would be large and thus infrequent
    This also generally increases flexibility substantially
Figure out how to speed up the update process
Figure out how to make the csv title properly generate / streamline the process so I can just run it without babying
Create the long term loop, so I can gather data for the next few days
add Docstrings to all methods and files
Figure out how fix Update, so that it can run multiple times without crashing
"""

import RedditInput
import Graphing
import Update
import Methods
import time

start = time.time()

# while((time.time() - start) < 21600):
#     print("time: " + str(int(time.time()- start)))
RedditInput.reCheckSub() #the stream that pulls top posts from wallstreetbets
Update.reCheck()
#RedditInput.checkSub() # the stream that pulls new posts
#Update.reCheck() # update the posts that have already been collected
    #time.sleep(120)
#Graphing.top10() # graphs the new information
Graphing.graphData()