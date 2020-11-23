import numpy as np
import pandas as pd

path = '~/ML-AZ/mercado.csv'
df = pd.read_csv(path, header=None)

transacoes = []
for i in range(0,10):
    transacoes.append([str(df.values[i,j]) for j in range(0,4)])

from apyori import apriori

regras = apriori(transactions=transacoes,
                 min_support=0.3,
                 min_confidence=0.8,
                 min_lift=2,
                 min_length=2)

resultados = list(regras)
resultados2 = [list(x) for x in resultados]

resultado_formatado = []
for j in range(0,3):
    resultado_formatado.append([list(x) for x in resultados[j][2]])