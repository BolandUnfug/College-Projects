import pandas as pd
import numpy as np

# cancer = pd.read_csv("data/cervical_cancer_modified.csv")
# cancer = cancer.dropna()
# print(cancer)
# cancer.to_csv("data/cervical_cancer_modified.csv")

lfw_faces = np.load( "data/lfwcrop.npy" )
print(lfw_faces.shape)