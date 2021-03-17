import pandas as pd
import numpy as np
import pickle
from joblib import dump, load
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import *
from sklearn.preprocessing import *
from sklearn.model_selection import *
import math

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

maintenance = pd.read_csv('ai4i2020.csv')
maintenance = maintenance.drop(columns=['UDI', 'Product ID', 'Air temperature [K]', 'Process temperature [K]', 'Tool wear [min]', 'Machine failure', 'TWF', 'HDF', 'OSF', 'RNF', 'PWF'])

const = (2*3.14)/60
maintenance['Rotational speed [rad/s]'] = maintenance['Rotational speed [rpm]'] * const

maintenance['Rotational speed [rad/s]'] = np.log(maintenance['Rotational speed [rad/s]'])
maintenance['Torque [Nm]'] = np.log(maintenance['Torque [Nm]'])
maintenance = maintenance.drop(columns=['Rotational speed [rpm]'])
maintenance['POWER [W]'] = maintenance['Torque [Nm]'] + maintenance['Rotational speed [rad/s]']

maintenance[['Type']] = maintenance[['Type']].astype('category')

label = LabelEncoder()
label.fit(maintenance.Type)
maintenance.Type = label.transform(maintenance.Type)

features = maintenance.iloc[:,0:2]
labels = maintenance.iloc[:,-1]

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state=42)

features_train = pd.DataFrame(features_train, columns= features_train.columns)
features_test = pd.DataFrame(features_test, columns= features_test.columns)


Virginia_model = LinearRegression()
Virginia_model.fit(features_train, labels_train)

# #saving model to disk
pickle.dump(Virginia_model, open('model.pkl', 'wb'))

# #loading model to compare the results
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[1, 4.5]]))
#print(Virginia_model.predict([[1, 4.5]]))