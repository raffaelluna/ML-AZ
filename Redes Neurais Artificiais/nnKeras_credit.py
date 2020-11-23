import pandas as pd
import numpy as np

path = '~\ML-AZ\credit_data.csv'
df = pd.read_csv(path)

#df.loc[df['age'] < 0]
# df.drop(df[df.age < 0].index, inplace=True) Deleta as linhas com idade inválida

df['age'][df.age > 0].mean()
df.loc[df.age < 0, 'age'] = 40.92

#df.loc[pd.isnull(df['age'])]

X_credit = df.iloc[:, 1:4].values
Y_credit = df.iloc[:, 4].values

from sklearn.impute import SimpleImputer

# mean imputation df.fillna(df.mean())
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Aprende os parametros dos dados de treino
imputer = imputer.fit(X_credit[:, 1:4])

# Usa estes parâmetros para transformar os dados
X_credit[:, 1:4] = imputer.transform(X_credit[:, 1:4])
X_credit

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(X_credit)

from sklearn.model_selection import train_test_split
X_train_credit, X_test_credit, Y_train_credit, Y_test_credit = train_test_split(
                                                                    X_credit,
                                                                    Y_credit,
                                                                    test_size=0.25,
                                                                    random_state=0)

import keras
from keras.models import Sequential
from keras.layers import Dense

clf = Sequential()

clf.add(Dense(units=2, activation='relu', input_dim=3))
clf.add(Dense(units=2, activation='relu'))
clf.add(Dense(units=1, activation='sigmoid'))
clf.compile(optimizer = 'adam',
            loss = 'binary_crossentropy',
            metrics = ['accuracy'])

clf.fit(X_train_credit, Y_train_credit, batch_size=10, epochs=100)

resultado = clf.predict(X_test_credit)
resultado = (resultado > 0.5)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(Y_test_credit, resultado)
matriz = confusion_matrix(Y_test_credit, resultado)