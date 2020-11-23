import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

path = '~/ML-AZ/credit_card_clients.csv'
df = pd.read_csv(path, header=1)

df['BILL_TOTAL'] = df['BILL_AMT1'] + df['BILL_AMT2'] + df['BILL_AMT3'] + df['BILL_AMT4'] + df['BILL_AMT5'] + df['BILL_AMT6']

X = df.iloc[:,[1,25]].values
scaler = StandardScaler()
X = scaler.fit_transform(X)

dendro = dendrogram(linkage(X, method='ward'))

hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
pred = hc.fit_predict(X)
plt.scatter(df[pred == 0, 0], df[pred == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(df[pred == 1, 0], df[pred == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(df[pred == 2, 0], df[pred == 2, 1], s=100, c='green', label='Cluster 3')
plt.xlabel('Limite')
plt.ylabel('Gastos')
plt.legend()
plt.show()