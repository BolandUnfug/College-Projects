import praw
import csv
import time
from tempfile import NamedTemporaryFile
import shutil
import yfinance as yf
import Methods as m

reddit = praw.Reddit("bot1")

tempfile = NamedTemporaryFile(mode='w+t', newline="", delete=False, encoding="utf8")

def reCheck(): # updates the previously collected posts
    print("Time to Update!")

    with open('Data.csv', newline="", encoding="utf8") as csvfile,tempfile:
        rawdata = open('Raw.csv', 'r', newline="", encoding="utf8")
        # row 1 only rewrites the title
        # row 2 gets the submission id to be used later
        # row 3 gets the mentioned stock
        # row 4 updates the amount of upvotes a submission has
        # row 5 updates the stock cost
        # row 6 updates the time stamp

        reader = csv.reader(csvfile)
        reader2 = csv.reader(rawdata)
        writer = csv.writer(tempfile)
        
        for row in reader:
            if(row[0] != "Title"):
                submissionid = m.simplify(row[1])
                newscore  = m.simplify(row[3]) + ", " + str(reddit.submission(id = submissionid).score) # updates upvotes
                newprice = m.simplify(row[4]) + ", " + str(m.get_current_price(m.simplify(row[2])))
                newtime = m.simplify(row[5]) + ", " + str(int((time.time())/60)) # updates timestamp
                
                rows = [str(row[0]), str(row[1]), str(row[2]),[newscore], [newprice], [newtime]]
                writer.writerow(rows)
            else:
                writer.writerow(row)
            for rawrow in reader2:
                print(rawrow[1])
                rawsubmissionid = m.simplify(rawrow[1])
                rawnewscore  = m.simplify(rawrow[3]) + ", " + str(reddit.submission(id = rawsubmissionid).score) # updates upvotes
                rawnewprice = m.simplify(rawrow[4]) + ", " + str(m.get_current_price(m.simplify(rawrow[2])))
                rawnewtime = m.simplify(rawrow[5]) + ", " + str(int((time.time())/60)) # updates timestamp
                
                rows = [str(rawrow[0]), str(rawrow[1]), str(rawrow[2]),[rawnewscore], [rawnewprice], [rawnewtime]]
                writer.writerow(rows)

    shutil.move(tempfile.name, 'Data.csv')