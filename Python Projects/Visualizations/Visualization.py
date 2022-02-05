"""
I plan to add 2 - 3 more data columns, since currently my graphs are not particularly useful
- Job
- Number of vacation days?
set up data analysis on a graph by graph basis
heatmap
- convert to numpy array

"""


from matplotlib.colors import Colormap
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import visualizations as vi
import numpy as np

def scatterPlot(data, headers):
    """
    Plots a scatterplot for IT Salary Survey EU 2020.csv, for bonus + stocks, age, and gender.
    """
    data = pd.DataFrame(data[headers])
    data = data.dropna(subset=headers)
    data = data.replace(to_replace="Male", value=0)
    data = data.replace(to_replace="Female", value=2)
    data = data.replace(to_replace="Diverse", value=1)
    pd.to_numeric(data[headers[1]])
    pd.to_numeric(data[headers[2]])
    pd.to_numeric(data[headers[0]])
    ax = data.plot.scatter(x=headers[0], y=headers[1], c=headers[2], cmap="cool",logx=True)

def histogram(data, headers, ranges):
    data = pd.DataFrame(data[headers])
    data = data.dropna(subset=headers)
    pd.to_numeric(data[headers[0]])
    pd.to_numeric(data[headers[1]])
    ax = data.plot.hist(x= headers[1], y=headers[0],range=ranges, bins=20)


data = pd.DataFrame(pd.read_csv("forestfires.csv"))

