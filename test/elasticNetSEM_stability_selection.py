"""
Created on Fri April 21, 2023
@author: Anhui Huang
"""

##
import numpy as np
from sparseSEM import elasticNetSEM, elasticNetSEMcv, elasticNetSEMpoint, enSEM_stability_selection

##
baseDataDir = './data/'
B = np.loadtxt(baseDataDir + 'B.csv', dtype = 'float64', delimiter = ',', skiprows=1, usecols=range(1,31))
X = np.loadtxt(baseDataDir + 'X.csv', dtype = 'float64', delimiter = ',', skiprows=1, usecols=range(1,201))
Y = np.loadtxt(baseDataDir + 'Y.csv', dtype = 'float64', delimiter = ',', skiprows=1, usecols=range(1,201))
Missing = np.loadtxt(baseDataDir + 'Missing.csv', dtype = 'float64', delimiter = ',', skiprows=1, usecols=range(1,201))

##
np.random.seed(0)
result = enSEM_stability_selection(Y, X,  Missing, B,
                                   alpha_factors=np.arange(1, 0.1, -0.1),
                                   lambda_factors=10 ** np.arange(-0.2, -3, -0.2),
                                   kFold=5, nBootstrap=100, verbose=-1)


print(result.keys() )
fit = result['STS']

print(result['statistics'])
FDRsetS = result["STS data"]["FDRsetS"]
FDRsetS[0]
##
# visualize the inferred network:
import networkx as nx
import matplotlib.pyplot as plt

adjacent = fit
r,c = adjacent.shape
G = nx.Graph()
for i in range(r):
 for j in range( c):
   if adjacent[i][j] != 0:
      G.add_edge(i,j)

nx.draw( G,with_labels=True)
plt.show()





