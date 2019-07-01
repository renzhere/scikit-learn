

from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics


# skip row 1 so pandas can parse the data properly.
dataset = pd.read_csv('gtd/globalterrorismdb_0718dist.csv', encoding="ISO-8859-1", low_memory=False)
print(dataset.shape)

# Drop any column with more than 50% missing values
half_count = len(dataset) / 2
dataset = dataset.dropna(thresh=half_count,axis=1)

dataset = dataset.drop(['summary','country_txt', 'region_txt', 'provstate', 'city','targsubtype1_txt',
                        'attacktype1_txt', 'targtype1_txt', 'natlty1_txt', 'weaptype1_txt','weapsubtype1_txt',
                        'target1', 'corp1','weapdetail', 'scite1', 'dbsource'], axis=1)

print(dataset.shape)

dataset2 = dataset['gname'].value_counts()
print(dataset2.head(12))

dataset = dataset[(dataset['gname'] == 'Unknown') | (dataset['gname'] =='Taliban') | \
                  (dataset['gname'] == 'Islamic State of Iraq and the Levant (ISIL)') | \
                  (dataset['gname'] == 'Shining Path (SL)') |\
                  (dataset['gname'] == 'Farabundo Marti National Liberation Front (FMLN)')|\
                  (dataset['gname'] == 'Al-Shabaab') | (dataset['gname']=="New People's Army (NPA)") |\
                  (dataset['gname'] == 'Irish Republican Army (IRA)') | \
                  (dataset['gname'] == 'Revolutionary Armed Forces of Colombia (FARC)')|\
                  (dataset['gname'] == 'Boko Haram') | (dataset['gname']== "Kurdistan Workers' Party (PKK)")|\
                  (dataset['gname'] == 'Basque Fatherland and Freedom (ETA)')]

data_mapping_dictionary = {"gname": {"Unknown":0, "Taliban": 1, "Islamic State of Iraq and the Levant (ISIL)":2,
                           "Shining Path (SL)":3, "Farabundo Marti National Liberation Front (FMLN)":4,
                           "Al-Shabaab":5, "New People's Army (NPA)": 6, "Irish Republican Army (IRA)":7,
                           "Revolutionary Armed Forces of Colombia (FARC)":8, "Boko Haram":9,
                            "Kurdistan Workers' Party (PKK)":10,"Basque Fatherland and Freedom (ETA)":11}}

dataset = dataset.replace(data_mapping_dictionary)

# dataset3 = dataset['dbsource'].value_counts()
# print(dataset3)

# dataset.to_csv("processed_data/processed_gtd.csv", index=False)

