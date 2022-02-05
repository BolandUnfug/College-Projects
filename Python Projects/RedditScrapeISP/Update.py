"""
Updates collected reddit data.

Methods:
-------
update()
Updates Data.json with new data from collected posts, as well as newly located posts from Raw.json.

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

reddit = praw.Reddit("bot1")

def update():
    """
    Updates Data.json with new data from collected posts, as well as newly located posts from Raw.json.

    Important: Make sure Raw.json is either wiped/updated between updates, or repeating data will occur.
    """

    print("time to update!")

    merge = True

    rawfile = open("Raw.json")
    if rawfile.read(2) != '': # checks to see if there is data to update
        print("updating raw data")
        raw = pd.read_json("Raw.json") # extracting data from Raw.json to a dataframe

        # change types to objects. this allows for lists inside lists.
        raw["Upvotes"] = raw["Upvotes"].astype('object')
        raw["Stockprice"] = raw["Stockprice"].astype('object') 
        raw["Timestamp"] = raw["Timestamp"].astype('object')

        # gets a set of new upvotes, all at once
        submissionids = []
        for row in range(len(raw)):
            submissionids.append(raw.at[row, 'ID'])
        newupvotes = []
        submissions = [i if i.startswith('t3_') else f't3_{i}' for i in submissionids] # idk what this does
        for submission in reddit.info(submissions):
            newupvotes.append(submission.score)
        #

        for row in range(len(raw)): # for each row in raw
            print("updating row " + str(row))

            # collect previous data
            prevupvote = raw.at[row, 'Upvotes']
            prevstockprice = raw.at[row, 'Stockprice']
            prevtime = raw.at[row, 'Timestamp']
            
            # collect new data
            # for some reason new data collection is sometimes long for reddit submissions
            #newupvote = 
            newstockprice = m.get_current_price(raw.at[row, 'Stock'])
            newtime = int((time.time())/60)
            
            # update the current cell with a list of the previous data and the new data
            raw.loc[row, 'Upvotes'] = [[prevupvote], [newupvotes[row]]] 
            raw.loc[row, 'Stockprice'] = [[prevstockprice], [newstockprice]]
            raw.loc[row, 'Timestamp'] = [[prevtime], [newtime]]

        print(raw)
    else: # if there is no data in raw
        merge = False
        print("No data in raw")
    rawfile.close()

    datafile = open("Data.json")
    if datafile.read(2) != '': # checks to see if there is data to update

        print("updating data")
        
        data = pd.read_json("Data.json") # extracting data from Data.json to a dataframe
        
        # don't need to change the types to objects, as they should already be objects

        # gets a set of new upvotes, all at once
        print("starting timer")
        start = time.time()
        submissionids = []
        for row in range(len(data)):
            submissionids.append(data.at[row, 'ID'])
        print("it took " + str(time.time() - start) + "seconds to append IDs")
        start = time.time()
        newupvotes = []
        submissions = [i if i.startswith('t3_') else f't3_{i}' for i in submissionids] # idk what this does
        for submission in reddit.info(submissions):
            newupvotes.append(submission.score)
        print("it took " + str(time.time() - start) + "seconds to gather reddit data")
        start = time.time()
        #

        for row in range(len(data)):

            print("updating row " + str(row))

            # collect the data in the row, then append the new data
            # for some reason data collection for reddit is sometimes long
            upvote = list(data.at[row, 'Upvotes'])
            upvote.append(newupvotes[row])

            stockprice = list(data.at[row, 'Stockprice'])
            stockprice.append(m.get_current_price(data.at[row, 'Stock']))

            times = list(data.at[row, 'Timestamp'])
            times.append(int((time.time())/60))
            
            # update the current cell with a new list of data
            
            data.loc[row, 'Upvotes'] = upvote

            data.loc[row, 'Stockprice'] = stockprice

            data.loc[row, 'Timestamp'] = times

        print(data)
    else: # if there is no data in data
        merge = False
        print("No data in data")
    datafile.close()

    if(merge == True): # if there is data in both files
        result = pd.concat([raw, data], ignore_index=True) # merge files, ignoring index
        result.to_json("Data.json")
    else:
        raw.to_json("Data.json")