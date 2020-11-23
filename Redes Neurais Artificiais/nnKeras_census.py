import pandas as pd
import numpy as np

path = '~/ML-AZ/census.csv'
df = pd.read_csv(path)

x_census = df.iloc[:, 0:14].values
y_census = df.iloc[:, 14].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

#OneHotEncoder cria uma coluna dummy para cada atributo nominal único, de
#modo a não criar uma ordem ao atribuir labels do tipo 0, 1, 3, 4...
#ColumnTransformer transforma seletivamente as colunas de um array multi-atributos
onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1,3,5,6,7,8,9,13])],remainder='passthrough')
x_census = onehotencorder.fit_transform(x_census).toarray()
print(x_census.shape)

labelencoder = LabelEncoder()
y_census = labelencoder.fit_transform(y_census)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_census = scaler.fit_transform(x_census)

from sklearn.model_selection import train_test_split
x_train_census, x_test_census, y_train_census, y_test_census = train_test_split(
    x_census, 
    y_census, 
    test_size = 0.15, 
    random_state = 0)

import keras
from keras.models import Sequential
from keras.layers import Dense

clf = Sequential()
clf.add(Dense(units=55, activation='relu', input_dim=108))
clf.add(Dense(units=55, activation='relu'))
clf.add(Dense(units=1, activation='sigmoid'))
clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

clf.fit(x_train_census, y_train_census, batch_size=10, epochs=100)

resultado = clf.predict(x_test_census)
resultado = (resultado > 0.5)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(y_test_census, resultado)
matriz = confusion_matrix(y_test_census, resultado)