"""
Version 1.1
Boland Unfug
January 26th, 2022
The main function for the other programs, use this for running the code
Process:
    Gather posts from wallstreet bets, compare with stock prices
    Update upvotes and stock prices
    Graph the top 10

TODO
find a way to get change over time
Improved Update, still not super fast, but fast enough?
why did the timings not add up?
Clean up json file formatting
2 different duplicates? 1 is TQQQ the other is QQQ but also has TQQQ in it?, now its doing that with GME
differences in graphing
"""

import RedditInput
import Graphing
import Update
import Methods
import time

start = time.time()
RedditInput.reCheckSub() #the stream that pulls top posts from wallstreetbets
Update.update()
# while((time.time() - start) < 21600):
#     print("time: " + str(int(time.time()- start)))
#RedditInput.checkSub() # the stream that pulls new posts
#Update.update() # update the posts that have already been collected
Graphing.graphData()