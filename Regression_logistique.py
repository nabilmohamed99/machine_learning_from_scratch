#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


data = pd.read_csv("datasets_180_408_data.csv",sep=",") #ouvrir le fichier csv
data.drop(['Unnamed: 32'], axis=1, inplace=True)  #supprimer la colonne unnamed
data.drop(['id'], axis=1, inplace=True)
data.dropna(axis=1,inplace=True) #verifier s il exite des donnes manquantes et les supprimee

# on va encoder la colonne diagnois en  transformant les caractères femme et hommes en des 0 et 1 a l aide d outils labelEncoder

le = preprocessing.LabelEncoder()
y=data.diagnosis=le.fit_transform(data.diagnosis)#y -> target : 
y=y.reshape(y.shape[0],1)#rendre la dimension de y=(569,1) 
x=data.iloc[:,1::].values # x egale a tous les autres colonnes de notre dataset sauf diagnosis

# utilisant une normalisation qui permet a chaque colonne de suivre la loi normale reduite a 0
x= (x- x.mean()) /x.std()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 5)#ensemble d’apprentissage (train-set 80%) et ensemble detest (test-set 20%)


#creer notre matrice de features en ajoutant une colonne de un  la matrice x_train
f=np.ones(x_train.shape[0])
f=f.reshape(f.shape[0],1)
X=np.concatenate((x_train,f),axis=1)

#construire notre parametre thete en utilisant  des nombre aleatoires (des zeros)
theta=np.random.randint(0,1,[X.shape[1],1])

#fonction logit
def sigmoid(X) :
    return 1/(1+np.exp(-X))


#----------------------------------------Modèle----------------------------------------------------
#---------------------------------------------------------------------------------------------------
def model(X,theta) :
     return X.dot(theta)


#---------------------------------------- fonction de cout ----------------------------------------------
#---------------------------------------------------------------------------------------------------
def cost_function(X, y, theta):
     m = len(y)
     f = sigmoid(model(X,theta))
     cost =- (1/m)*np.sum(y.T.dot(np.log(f)) +(1 - y).T.dot(np.log(1 - f)))
     return cost


def gradient(X, y, theta):
     m = len(y)
     f = sigmoid(model(X,theta))
     grad = (1/m)*np.dot(X.T,  f- y)
     return grad


#----------------------------------------Mini batch gradient descent----------------------------------------
#---------------------------------------------------------------------------------------------------------
def gradient_descent(X,y,theta,learning_rate,n_iterations):

    cost_history=np.zeros(n_iterations)

    for i in range(0,n_iterations):
        
        theta=theta-learning_rate*gradient(X,y,theta)
        
        cost_history[i]=cost_function(X,y,theta)
 
    return theta, cost_history


#----------------------------------------stochastic gradient descent----------------------------------------
#---------------------------------------------------------------------------------------------------------
def gradient_descent_stochastique(X,y,theta,learning_rate,n_iterations,batch_size):
    m=len(y)
    cost_history=np.zeros(n_iterations)
    for i in range(n_iterations):
       indices=np.random.permutation(m)
       X1=X[indices]
       y1=y[indices]
       for i in range(0,m,batch_size):
           X_i=X1[i:i+batch_size]
           y_i=y1[i:i+batch_size]
           theta=theta-learning_rate*gradient(X_i,y_i,theta)
       cost_history[i]=cost_function(X_i,y_i,theta)
    return theta, cost_history
theta_final,cost_history=gradient_descent_stochastique(X,y_train,theta,learning_rate=0.001,n_iterations=1000,batch_size=50)



def Predict(X,theta):
    
    return np.where(sigmoid(model(X,theta)) >= 0.5, 1, 0)




#----------------------------------------Entrainement gradient descent ----------------------------------------------
#---------------------------------------------------------------------------------------------------
theta_final,cost_history=gradient_descent(X,y_train,theta,learning_rate=0.01,n_iterations=10000)
plt.figure(figsize=(12, 8))
#la prediction de notre model
y_final=Predict(X,theta_final)
plt.grid()
plt.subplot(1,2,2)
plt.title("comparaison entre notre modele et le dataset   (le caractere  area_mean)")
#tracer notre model pour le caractere  area_mean
plt.scatter(X[:,1],y_train,c="blue",Label="notre data set")
plt.scatter(X[:,1],y_final,c="red")
plt.subplot(1,2,1)
#rracer cost_history
plt.plot(cost_history)
plt.plot(cost_history)
plt.title("cost fonction  gradient descente simpte ")
plt.legend()
plt.show()
score = float(sum(y_final == y_train))/ float(len(y_train))
print("le score de la regression logistique avec la methode de gradient descente"+str(score))


#----------------------------------------Entrainement gradient descent  stochastique ----------------------------------------------
#---------------------------------------------------------------------------------------------------
theta_final1,cost_history1=gradient_descent_stochastique(X,y_train,theta,learning_rate=0.001,n_iterations=1000,batch_size=50)
plt.plot(cost_history1)
y_final1=Predict(X,theta_final1)
score1 = float(sum(y_final1 == y_train))/ float(len(y_train))
print(f"le score de la regression logistique avec la methode de gradient descente stochastique{score1}")





# In[ ]:





# In[ ]:





# In[ ]:




