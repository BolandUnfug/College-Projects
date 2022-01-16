"""
Graphs data from Data.csv

"""
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import Methods as m

def findDuplicates():
    redditdata = pd.read_csv("Data.csv")
    popularstocks = redditdata["Stock"]
    popularstocks = popularstocks.loc[popularstocks.duplicated()]
    popularstocks = popularstocks.drop_duplicates()

    return redditdata.loc[redditdata["Stock"].isin(popularstocks)]


def top10():
    # get list of duplicates
    # drop duplicates from list
    redditdata = pd.read_csv("Data.csv")


    # Get upvotes from duplicates
    # get the average for upvotes
    # sort highest to lowest / get the top 10
    #
    duplicates = findDuplicates()
    popularposts = duplicates["Upvotes"]
    for posts in popularposts:
        print(m.simplifyrow(posts))
    
    # tierlist = []
    # for posts in popularposts:
    #     tierlist.append([posts, sum(m.simplifyrow(posts))/len(m.simplifyrow(posts))])
    
    # tierlist.sort(key = lambda x: x[1], reverse=True)

    # top10posts = []
    # for x in range(10):
    #     top10posts.append(tierlist[x][0])

    # for rows in duplicates:
    #     print(rows)
    #     for x in top10posts:
    #         print(x)
    #         if(rows["Upvotes"] == x):
                
    # topposts = pd.DataFrame(top10posts)
    # top10duplicates = redditdata.loc[redditdata["Upvotes"].isin(topposts)]

    

    
    


def graphData():
    """
    Graphs the data in Data.csv
    """

    redditdata = pd.read_csv("Data.csv")

    timestamps = redditdata["Timestamp"]
    timestamps = m.simplifylist(timestamps)

    upvotes = redditdata["Upvotes"]
    upvotes = m.simplifylist(upvotes)

    stockprices = redditdata["Stockprice"]
    stockprices = m.simplifylist(stockprices)

    x = timestamps
    y1 = upvotes
    y2 = stockprices

    f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

    counter = 0
    for y in y1:
        ax1.plot(x[counter], y, label=m.simplify(redditdata["Stock"][counter]))  # Plot some data on the axes.
        counter += 1
    counter = 0
    for y in y2:
        ax2.plot(x[counter], y, label=m.simplify(redditdata["Stock"][counter]))  # Plot some data on the axes.
        counter += 1

    ax1.set_xlabel('Time')  # Add an x-label to the axes.
    ax1.set_ylabel('Upvotes')  # Add a y-label to the axes.
    ax2.set_ylabel('Stock Price')  # Add a y-label to the axes.
    ax1.set_title("Reddit Popularity")  # Add a title to the axes.
    ax1.legend();  # Add a legend.
    plt.show()