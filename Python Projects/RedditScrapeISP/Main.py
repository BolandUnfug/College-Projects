"""
The main function for the other programs, use this for running the code
Process:
    Gather posts from wallstreet bets, compare with stock prices
    Update upvotes and stock prices
    Graph the top 10

TODO
fix top 10 feature
find a way to get change over time
Figure out how to speed up the update process
Create the long term loop, so I can gather data for the next few days
add Docstrings to all methods and files
why did the timings not add up?
Clean up json file formatting
Clean up code formatting
"""

import RedditInput
import Graphing
import Update
import Methods
import time

start = time.time()

# while((time.time() - start) < 21600):
#     print("time: " + str(int(time.time()- start)))
#RedditInput.reCheckSub() #the stream that pulls top posts from wallstreetbets
#Update.reCheck()
#RedditInput.checkSub() # the stream that pulls new posts
#Update.reCheck() # update the posts that have already been collected
    #time.sleep(120)
#Graphing.top10() # graphs the new information
Graphing.graphData()