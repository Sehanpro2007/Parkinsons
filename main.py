import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import sample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

ds = pd.read_csv('parkinsons.data')

ds_reorder = ds[
    ['name', 'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP',
     'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ',
     'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE', 'status']]  # rearrange column here
ds_reorder.to_csv('venv/finalDS.csv', index=False)
dsData = pd.read_csv("venv/finalDS.csv")
dsData.drop("name", axis=1, inplace=True)
print(dsData)

# create multiple charts in order to try coming up with model for the program to run (but will not need in final
for label in dsData.columns:
    plt.plot(dsData[dsData['status'] == 1][label], color='blue', label='parkinson', alpha=0.7)
    plt.plot(dsData[dsData['status'] == 0][label], color='red', label='no parkinson', alpha=0.7)
    plt.title(label)
    plt.ylabel("xFeature vector values plotted")
    plt.xlabel(label)
    plt.legend()
    plt.show()

# split my dataframe into 3 separate testing data
train, validate, test = np.split(ds.sample(frac=1), (int(0.6 * len(ds)), int(0.8 * len(ds))))

# scale all numbers in order to keep all datapoint relative to the average

# create scaling function
