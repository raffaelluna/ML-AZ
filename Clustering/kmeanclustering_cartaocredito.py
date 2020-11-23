import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

path = '~/ML-AZ/credit_card_clients.csv'
df = pd.read_csv(path, header=1)

df['BILL_TOTAL'] = df['BILL_AMT1'] + df['BILL_AMT2'] + df['BILL_AMT3'] + df['BILL_AMT4'] + df['BILL_AMT5'] + df['BILL_AMT6']

X = df.iloc[:,[1,25]].values
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

plt.scatter(X[pred == 0, 0], X[pred == 0, 1], s=100, color='red', label='Cluster 1')
plt.scatter(X[pred == 1, 0], X[pred == 1, 1], s=100, color='orange', label='Cluster 2')
plt.scatter(X[pred == 2, 0], X[pred == 2, 1], s=100, color='green', label='Cluster 3')
plt.scatter(X[pred == 3, 0], X[pred == 3, 1], s=100, color='blue', label='Cluster 4')
plt.xlabel('Limite')
plt.ylabel('Gastos')
plt.legend()
plt.show()

lista_clientes = np.column_stack((df, pred))
lista_clientes = lista_clientes[lista_clientes[:,26].argsort()]