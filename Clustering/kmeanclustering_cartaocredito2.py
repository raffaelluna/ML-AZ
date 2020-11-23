import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

path = '~/ML-AZ/credit_card_clients.csv'
df = pd.read_csv(path, header=1)

df['BILL_TOTAL'] = df['BILL_AMT1'] + df['BILL_AMT2'] + df['BILL_AMT3'] + df['BILL_AMT4'] + df['BILL_AMT5'] + df['BILL_AMT6']

X = df.iloc[:,[1,2,3,4,5,25]].values
scaler = StandardScaler()
X = scaler.fit_transform(X)

#### ELBOW METHOD PARA DEFINIR O NÚMERO DE CLUSTER
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.xlabel('Número de Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=4, random_state=0)
pred = kmeans.fit_predict(X)

lista_clientes = np.column_stack((df, pred))
lista_clientes = lista_clientes[lista_clientes[:,26].argsort()]