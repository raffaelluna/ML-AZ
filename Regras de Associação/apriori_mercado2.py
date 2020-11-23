import numpy as np
import pandas as pd

path = '~/ML-AZ/mercado2.csv'
df = pd.read_csv(path, header=None)

transacoes = []
for i in range(0,7501):
    transacoes.append([str(df.values[i,j]) for j in range(0,20)])

from apyori import apriori

### DEVE SER FEITA A ALTERAÇÃO DO SUPORTE, DADA A GRANDE VARIABILIDADE DOS PRODUTOS
regras = apriori(transactions=transacoes,
                 min_support=0.003,
                 min_confidence=0.2,
                 min_lift=3.0,
                 min_length=2)

resultados = list(regras)
resultados2 = [list(x) for x in resultados]

resultado_formatado = []
for j in range(0,5):
    resultado_formatado.append([list(x) for x in resultados[j][2]])