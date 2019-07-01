

from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics


processed_data = pd.read_csv("processed_data/processed_gtd1.csv")
print(processed_data.shape)

# null_counts = processed_data.isnull().sum()
# print("Number of null values in each column:\n{}".format(null_counts))

processed_data = processed_data.drop(["nwoundte","nwoundus", "nkillter", "nkillus","latitude", "longitude",
                                      "nperps", "nperpcap"], axis=1)
processed_data = processed_data.dropna()

print(processed_data.shape)
# null_counts = processed_data.isnull().sum()
# print("Number of null values in each column:\n{}".format(null_counts))

print("Data types and their frequency\n{}".format(processed_data.dtypes.value_counts()))

processed_data.info()

processed_data2 = processed_data['gname'].copy()
processed_data_intermediate = processed_data.drop(['gname'], axis=1)
# processed_data_final =processed_data_intermediate
processed_data_final = processed_data_intermediate.join(processed_data2)

null_counts = processed_data.isnull().sum()
print("Number of null values in each column:\n{}".format(null_counts))

# processed_data_final.to_csv("processed_data/cleaned_ordered_gtd.csv", index=False)

# processed_data.to_csv("processed_data/cleaned_gtd.csv", index=False)