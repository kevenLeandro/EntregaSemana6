# Regresão por árvore de decisão, pode ser usado para regressão e para outros contextos.
# arvore de decisão está incluida nos problemas np completos. não e sabe se é possivel resolver com complexidade polimonial.
# Vamos escolher variavel de corte usando entropia, quanto maior o valor entropia menos informativo ele é:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv('winequality-red.csv')
ind = dataset.iloc[:, [7,10]].values
dep = dataset.iloc[:, -1].values

# treinamentação
decisionTreeRegressor = DecisionTreeRegressor(random_state=0)
decisionTreeRegressor.fit(ind, dep)

#print (decisionTreeRegressor.predict(  [ [0.9946, 10] ] ) )

plt.scatter(ind[:, 0 ], decisionTreeRegressor.predict(ind), color="red")

plt.xlabel("Alcohol")

plt.ylabel("Density")

plt.title("Alcohol vs Density")

plt.show()


fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(ind[:,0], ind[:,1], decisionTreeRegressor.predict(ind))

plt.show()








