import os
import pandas as pd
import numpy as np
from collections import Counter

def dist(x_test, x_train, p):

    """
    Implementa a Distância de Minkowski:
    Recebe dois vetores x_teste e x_train e retorna a distância entre eles.
    """
    
    sum_ = 0
    
    #Soma o módulo diferença elevada a p dos elementos dos vetores
    for i in range(len(x_test)):
        sum_ += abs(x_test[i] - x_train[i])**p
    
    #retorna a raíz p-ésima da soma
    return sum_**(1/p)

class KNearestNeighbours:
    
        def __init__(self, x_train, y_train):

            self.x_train = x_train
            self.y_train = y_train
            
            if not isinstance(x_train, np.ndarray):
                self.x_train = np.array(x_train)
                
            if not isinstance(y_train, np.ndarray):
                self.y_train = np.array(y_train)

            
        def predict(self, x_test, y_test, k, p):
            
            self.x_test = x_test
            self.y_test = y_test
            
            if not isinstance(self.x_test, np.ndarray):
                self.x_test = np.array(x_test)
                
            if not isinstance(self.y_test, np.ndarray):
                self.y_test = np.array(self.y_test)
                
            self.p = p
            self.k = k

            # número de exemplos de teste
            n_test = self.x_test.shape[0]
            
            yhat = []
            t = 0

            for i in range(n_test):
                
                distance_list = [dist(self.x_test[i], self.x_train[j], self.p)
                                for j in range(self.x_train.shape[0])]
                
                distances = pd.DataFrame({'distance': distance_list,
                                        'label': self.y_train})
                
                distances = distances.sort_values(by=['distance'])[:self.k]
                
                #majority vote
                nn_labels_counter = Counter(self.y_train[distances.index])
                pred = nn_labels_counter.most_common()[0][0]
                
                print(t)
                t += 1
                yhat.append(pred)
            
            return yhat