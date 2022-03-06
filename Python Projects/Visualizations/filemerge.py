import pandas as pd
import numpy as np

# redwine = pd.read_csv("data/winequality-red.csv", delimiter=";")
# print(redwine)
# redwine.insert(0,"color", 0)
# print(redwine)
# whitewine = pd.read_csv("data/winequality-white.csv", delimiter=";")
# print(redwine)
# whitewine.insert(0,"color", 1)
# print(whitewine )
# wine = pd.DataFrame(pd.concat([redwine, whitewine], ignore_index=True))
# wine.to_csv("winequality.csv")


# test = np.array([[1,2,3],[4,5,6],[7,8,9]])
# print(test[:,0:1])
C = [3,1,1,1,1,2,2,2,3,3,1,7, 22, 1,1,3,4,57,55,6,8,4]

c = [] # each unique class
priors = [] # for now, a count of each class
for n in C: # each element in C
    if n not in c: # if its not a class, add it to c and add 1 to its counter
        c.append(n)
        priors.append(1)
    else: # if it is already a class, add 1 to it
        priors[c.index(n)] += 1

print(c)
print(priors)

for m in set (len(c)): # for each class, divide by the total
    priors[m] = priors[m]/len(C)
