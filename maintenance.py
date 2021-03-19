import pandas as pd
import numpy as np
import pickle
from joblib import dump, load
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import *
from sklearn.preprocessing import *
from sklearn.model_selection import *
import math
from sklearn.preprocessing import StandardScaler

maintenance = pd.read_csv('ai4i2020.csv')
osf_maintenance = maintenance.drop(columns=['UDI', 'Product ID', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Machine failure', 'TWF', 'HDF', 'OSF', 'RNF', 'PWF'])
osf_maintenance.drop(osf_maintenance[osf_maintenance['Tool wear [min]'] == 0].index, inplace=True)
osf_maintenance['Tool wear [min]'] = np.log(osf_maintenance['Tool wear [min]'])
osf_maintenance['Torque [Nm]'] = np.log(osf_maintenance['Torque [Nm]'])
osf_maintenance['overstrain [minNm]'] = osf_maintenance['Torque [Nm]'] + osf_maintenance['Tool wear [min]']


osf_maintenance[['Type']] = osf_maintenance[['Type']].astype('category')

label = LabelEncoder()
label.fit(osf_maintenance.Type)
osf_maintenance.Type = label.transform(osf_maintenance.Type)

osf_maintenance[['Torque [Nm]', 'Tool wear [min]']] = StandardScaler().fit_transform(osf_maintenance[['Torque [Nm]', 'Tool wear [min]']])

features = osf_maintenance.iloc[:,1:3]
labels = osf_maintenance.iloc[:,-1]

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state=42)

features_train = pd.DataFrame(features_train, columns= features_train.columns)
features_test = pd.DataFrame(features_test, columns= features_test.columns)


Virginia_model = LinearRegression()
Virginia_model.fit(features_train, labels_train)

# #saving model to disk
pickle.dump(Virginia_model, open('model.pkl', 'wb'))

# #loading model to compare the results
model = pickle.load(open('model.pkl', 'rb'))
print(math.exp(model.predict([[1.924952, 0.915434]])))
#print(Virginia_model.predict([[1, 4.5]]))


#Classification
hdf_maintenance = maintenance.drop(columns=['UDI', 'Product ID', 'Torque [Nm]', 'Tool wear [min]', 'Machine failure', 'TWF', 'OSF', 'RNF'])
hdf_maintenance['Dif temperature [K]'] = hdf_maintenance['Process temperature [K]'] - hdf_maintenance['Air temperature [K]']
hdf_maintenance = hdf_maintenance.drop(columns=['Air temperature [K]', 'Process temperature [K]', 'PWF'])
hdf_maintenance = hdf_maintenance.reindex(columns=['Type', 'Dif temperature [K]', 'Rotational speed [rpm]', 'HDF'])

hdf_maintenance[['Type']] = hdf_maintenance[['Type']].astype('category')
label = LabelEncoder()
label.fit(hdf_maintenance.Type)
hdf_maintenance.Type = label.transform(hdf_maintenance.Type)
hdf_maintenance.dtypes

features_hdf = hdf_maintenance.iloc[:,0:3]
labels_hdf = hdf_maintenance.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features_hdf, labels_hdf, train_size = 0.8, test_size = 0.2, random_state=42)


#Two-class Adaboost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm = "SAMME", n_estimators = 200, learning_rate = 0.1)
bdt.fit(x_train, y_train)

#print(bdt.predict([[1, -8.4, 1363]]))

# #saving model to disk
pickle.dump(bdt, open('model_hdf.pkl', 'wb'))

# #loading model to compare the results
model_hdf = pickle.load(open('model_hdf.pkl', 'rb'))
print(model_hdf.predict([[1, 8.4, 1363]]))
#print(Virginia_model.predict([[1, 4.5]]))
