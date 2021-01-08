import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
csv = pd.read_csv('coor.csv')
data = csv[['edge_axis-x', 'edge_axis-y']]
x = data['edge_axis-x']
y = data['edge_axis-y']
plt.scatter(x, y)

#z = np.polynomial.polynomial.polyfit(x, y,)

#plt.plot(x,y)

plt.show()