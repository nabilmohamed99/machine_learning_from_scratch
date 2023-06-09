import numpy as np 
import matplotlib.pyplot as plt 

def sigmoid(z): 
	return 1 / (1 + np.exp( - z)) 

plt.plot(np.arange(-5, 5, 0.1), sigmoid(np.arange(-5, 5, 0.1))) 
plt.title('Sigmoid Function') 
plt.show() 

# Importer les packages 
import numpy as np 
import pandas as pd 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder 

data = pd.read_csv('dataset2.csv', header=None) 
print(data.shape)
print(data.head()) 

# Features 
x_orig = data.iloc[:, 1:-1].values 

# Labels 
y_orig = data.iloc[:, -1:].values 

print("Shape de la matrice des features :", x_orig.shape) 
print("Shape du vecteur Label :", y_orig.shape) 

# Catégorie y=1 Setosa 
Setosa = np.array([x_orig[i] for i in range(len(x_orig)) 
									if y_orig[i] == 1]) 

# Catégorie y=0 Versicolor 
Versi = np.array([x_orig[i] for i in range(len(x_orig)) 
									if y_orig[i] == 0]) 

# La classe setosa 
plt.scatter(Setosa[:, 0], Setosa[:, 1], color = 'blue', label = 'Setosa') 

# La classe Versicolor 
plt.scatter(Versi[:, 0], Versi[:, 1], color = 'red', label = 'Versicolor') 

plt.xlabel('Longueur des sépales') 
plt.ylabel('Largeur des sépales') 
plt.title('Iris : Setosa, Versicolor') 
plt.legend() 

plt.show() 


