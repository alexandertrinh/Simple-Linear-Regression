import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# import data
dataSet = pd.read_csv("Data.csv")
x = dataSet.iloc[:, :-1].values  # country age salary
y = dataSet.iloc[:, 3].values  # purchased or not

# split dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size = 0.20,
                                                    train_size = 0.80,
                                                    random_state = 0)

# feature scaling
"""
standard_scalar_x = StandardScaler()
x_train = standard_scalar_x.fit_transform(x_train)
x_test = standard_scalar_x.transform(x_test)
"""





print("exit(0)")