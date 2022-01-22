import praw
import time
from tempfile import NamedTemporaryFile
import shutil
import yfinance as yf
import pandas as pd
import Methods as m

reddit = praw.Reddit("bot1")

def reCheck():

    print("time to update!")
    merge = True
    rawfile = open("Raw.json")
    if rawfile.read(2) != '':
        print("updating raw data")
        raw = pd.read_json("Raw.json")

        raw["Upvotes"] = raw["Upvotes"].astype('object')
        raw["Stockprice"] = raw["Stockprice"].astype('object')
        raw["Timestamp"] = raw["Timestamp"].astype('object')

        for row in range(len(raw)):
            
            print("updating row " + str(row))

            prevupvote = raw.at[row, 'Upvotes']
            prevstockprice = raw.at[row, 'Stockprice']
            prevtime = raw.at[row, 'Timestamp']
            
            #for some reason new data collection is hard and long
            newupvote = int(reddit.submission(id = raw.at[row, 'ID']).score)

            newstockprice = m.get_current_price(raw.at[row, 'Stock'])

            newtime = int((time.time())/60)
            
            raw.loc[row, 'Upvotes'] = [[prevupvote], [newupvote]]

            raw.loc[row, 'Stockprice'] = [[prevstockprice], [newstockprice]]

            raw.loc[row, 'Timestamp'] = [[prevtime], [newtime]]

        print(raw)

    else:
        merge = False
        print("No data in raw")
    rawfile.close()

    datafile = open("Data.json")
    if datafile.read(2) != '':

        print("updating data")
        
        data = pd.read_json("Data.json")

        data["Upvotes"] = data["Upvotes"].astype('object')
        data["Stockprice"] = data["Stockprice"].astype('object')
        data["Timestamp"] = data["Timestamp"].astype('object')
        
        for row in range(len(data)):

            print("updating row " + str(row))

            upvote = list(data.at[row, 'Upvotes'])
            upvote.append(int(reddit.submission(id = data.at[row, 'ID']).score))

            stockprice = list(data.at[row, 'Stockprice'])
            stockprice.append(m.get_current_price(data.at[row, 'Stock']))

            times = list(data.at[row, 'Timestamp'])
            times.append(int((time.time())/60))
            
            #for some reason new data collection is hard and long
            
            data.loc[row, 'Upvotes'] = upvote

            data.loc[row, 'Stockprice'] = stockprice

            data.loc[row, 'Timestamp'] = times

        print(data)
    else:
        merge = False
        print("No data in data")
    datafile.close()

    if(merge == True):
        result = pd.concat([raw, data], ignore_index=True)
        result.to_json("Data.json")

    else:
        raw.to_json("Data.json")

    

    #print(result)