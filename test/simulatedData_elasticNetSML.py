"""
Created on Fri April 21, 2023
@author: Anhui Huang

"""

##
import numpy as np
from sparseSEM import elasticNetSEM, elasticNetSEMcv, elasticNetSEMpoint

##
baseDataDir = './data/'
B = np.loadtxt(baseDataDir + 'B.csv', dtype = 'float64', delimiter = ',', skiprows=1, usecols=range(1,31))
X = np.loadtxt(baseDataDir + 'X.csv', dtype = 'float64', delimiter = ',', skiprows=1, usecols=range(1,201))
Y = np.loadtxt(baseDataDir + 'Y.csv', dtype = 'float64', delimiter = ',', skiprows=1, usecols=range(1,201))
M = np.loadtxt(baseDataDir + 'Missing.csv', dtype = 'float64', delimiter = ',', skiprows=1, usecols=range(1,201))

##
np.random.seed(0)
result = elasticNetSEM(X, Y, M, B, verbose = 0);
print(result.keys() )
print(result['statistics'])
##
# visualize the inferred network:
import networkx as nx
import matplotlib.pyplot as plt

adjacent = result['weight']
r,c = adjacent.shape
G = nx.Graph()
for i in range(r):
 for j in range( c):
   if adjacent[i][j] != 0:
      G.add_edge(i,j)

nx.draw( G,with_labels=True)
plt.show()
