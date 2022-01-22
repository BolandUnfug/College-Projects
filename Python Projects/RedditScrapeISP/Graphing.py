"""
Graphs data from Data.csv

"""
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import Methods as m


def findDuplicates(redditdata):
    popularstocks = redditdata["Stock"]
    popularstocks = popularstocks.loc[popularstocks.duplicated()]
    popularstocks = popularstocks.drop_duplicates()
    return popularstocks

def findAllDuplicates(redditdata):
    popularstocks = redditdata["Stock"]
    popularstocks = popularstocks.loc[popularstocks.duplicated()]
    popularstocks = popularstocks.drop_duplicates()
    return redditdata.loc[redditdata["Stock"].isin(popularstocks)]

def insertZeros(list):
    row_lengths = []

    for row in list:
        row_lengths.append(len(row))

    max_length = max(row_lengths)

    for row in list:
        for x in range(max_length - len(row)):
            row.insert(0, 0)
    return list
    

def getPopular():

    data = pd.read_json("Data.json")
    graphingdata = data[["Upvotes", "Stock", "Stockprice", "Timestamp"]]

    allduplicates = findAllDuplicates(graphingdata)

    graphabledata = pd.DataFrame(
        {
        'Upvotes': [],
        'Stock': [],
        'Stockprice': [],
        'Timestamp': []
        }
    )
    print(graphabledata)

    for dupe in findDuplicates(graphingdata):
        rowswithdupe = allduplicates['Stock'].str.contains(dupe)
        duplicatelist = allduplicates[rowswithdupe]
        dupeupvotes = insertZeros(list(duplicatelist["Upvotes"]))
        sums = []
        for nums in range(len(dupeupvotes[0])):
            sum = 0
            for lists in dupeupvotes:
                sum = sum + lists[nums]
            sums.append(sum)

        graphabledata.loc[len(graphabledata.index)] = (sums,
        duplicatelist.iloc[0,1], duplicatelist.iloc[len(duplicatelist.index) - 1, 2],
        duplicatelist.iloc[len(duplicatelist.index) - 1, 3])

    return graphabledata


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

    
def graphData():
    """
    Graphs the data in Data.csv
    """

    redditdata = getPopular()

    x = redditdata["Timestamp"]
    y1 = redditdata["Upvotes"]
    y2 = redditdata["Stockprice"]

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
    ax2.set_xlabel('Time')  # Add an x-label to the axes.
    ax1.set_ylabel('Combined Upvotes')  # Add a y-label to the axes.
    ax2.set_ylabel('Stock Price')  # Add a y-label to the axes.
    ax1.set_title("Reddit Popularity")  # Add a title to the axes.
    ax2.set_title("Stock Price")  # Add a title to the axes.
    ax1.legend();  # Add a legend.
    plt.show()