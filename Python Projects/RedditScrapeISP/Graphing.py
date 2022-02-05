"""
Graphs data from Data.csv

Methods
-------
    findDuplicates(data)
        Finds duplicates in data.
    insertZeroes(numlist)
        Inserts zeroes into a list.
    getPopular()
        Creates a Dataframe of popular stocks, and condences them into one row for each stock.
    average(numlist)
        Gets the average of a list of numbers.
    graphData()
        Manipulates the data into a usable form, then graphs the data.

Production
----------
Version 1.1

Boland Unfug

January 26th, 2022
"""
import pandas as pd
import matplotlib.pyplot as plt
import Methods as m


def findDuplicates(data):
    """
    finds duplicates in the data.

    Parameters
    ----------
    (Dataframe) data: a dataframe of data to search through.

    Returns
    -------
    (Dataframe) popularstocks: a list of the stocks with duplicates.
    """
    popularstocks = data["Stock"] # collects the stocks column
    popularstocks = popularstocks.loc[popularstocks.duplicated()] # locates all duplicates
    popularstocks = popularstocks.drop_duplicates() # remove duplicates of duplicates
    return popularstocks

def insertZeros(numlist):
    """
    insert zeroes into a list, starting from position 0.

    Parameters
    ----------
    (list(list)) numlist: a nested list to add zeroes to.

    Returns
    -------
    (list(list)) numlist: the same nested list, but with the zeroes inserted.
    """

    row_lengths = []

    for row in numlist:
        row_lengths.append(len(row)) # create a list of list lengths

    max_length = max(row_lengths) # find the longest list

    for row in numlist:
        # insert a zero until the list has the same length as max
        for x in range(max_length - len(row)): 
            row.insert(0, 0) # insert zeroes at 0
    return numlist
    

def getPopular():
    """
    Creates a Dataframe of popular stocks, and condences them into one row for each stock.

    Returns
    -------
    (Dataframe) graphabledata: a dataframe of condensed data to be graphed.
    
    """
    data = pd.read_json("Data.json")
    graphingdata = data[["Upvotes", "Stock", "Stockprice", "Timestamp"]] # extract graphable data

    # Get a Dataframe of all duplicates
    allduplicates = graphingdata.loc[graphingdata["Stock"].isin(findDuplicates(graphingdata))]

    graphabledata = pd.DataFrame( # create a new dataframe with empty labels
        {
        'Upvotes': [],
        'Stock': [],
        'Stockprice': [],
        'Timestamp': []
        }
    )

    for dupe in findDuplicates(graphingdata):
        # a list of duplicate stock instances
        duplicatelist = allduplicates[allduplicates['Stock'].str.contains(dupe)] 
        # a list of upvotes from duplicatelist, with zeroes inserted
        dupeupvotes = insertZeros(list(duplicatelist["Upvotes"])) 

        # adds the list of upvotes from each duplicate stock instance together
        sums = []
        for nums in range(len(dupeupvotes[0])):
            sum = 0
            for lists in dupeupvotes:
                sum = sum + lists[nums]
            sums.append(sum)
        # sums becomes a list of upvotes, with each position containing a sum of upvotes of instances

        # adds modified data as a new row into graphable data
        graphabledata.loc[len(graphabledata.index)] = (sums,
        duplicatelist.iloc[0,1], duplicatelist.iloc[len(duplicatelist.index) - 1, 2],
        duplicatelist.iloc[len(duplicatelist.index) - 1, 3])

    return graphabledata

def Average(numlist):
    """Averages each list of numbers in a nested list, used for lambda"""
    averages = []
    for values in numlist:
        averages.append(sum(values)/len(values))
    return averages
    
    
def graphData():
    """
    Graphs data from Data.json.
    """

    redditdata = getPopular() # get a summarized dataframe of popular stocks

    # sort dataframe by upvote average, from highest to lowest
    sorteddata = redditdata.sort_values(by='Upvotes',ascending=False, ignore_index=True, key = lambda col: Average(col))

    redditdata = sorteddata.head(10) # cut off any below 10
    
    # simplify variables
    x = redditdata["Timestamp"]
    
    y1 = redditdata["Upvotes"]
    y2 = redditdata["Stockprice"]

    f, (ax1, ax2) = plt.subplots(1, 2, sharex=True) # create a new graph

    counter = 0 # used for x positions
    for y in y1: # graph data in y1
        ax1.plot(x[counter], y, label=m.simplify(redditdata["Stock"][counter], ["[", "]"])) # graph 1 line
        counter += 1
    counter = 0 # reset counter
    for y in y2: # graph data in y2
        ax2.plot(x[counter], y, label=m.simplify(redditdata["Stock"][counter], ["[", "]"])) # graph 1 line
        counter += 1

    # make graph pretty
    ax1.set_xlabel('Time')  
    ax2.set_xlabel('Time')  
    ax1.set_ylabel('Combined Upvotes')  
    ax2.set_ylabel('Stock Price')  
    ax1.set_title("Reddit Popularity")  
    ax2.set_title("Stock Price")
    ax1.legend()
    # show the graph
    plt.show()