#!/usr/bin/env python
# coding: utf-8



import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt





dataset=pd.read_csv("Dataset.csv") #extraire les donnes 

dataset.head()

X=np.array(dataset["Year;Land_Annual"]).T

X #un tableau qui contient "year;land_annual" fromat->String

f=[]
Y=[]
e=[]
#convertant le talbeau X a un tableau f qui deux colonne "year;land_annual" ->float
for i in X  :
    y=i.split(sep=";")
    e=[float(i) for i in y ]
    
    f.append(e)
    
v=np.array(f)# convertant f a un tableau numpy pour l utiliser
v.shape  # un tableau numpy qui contient deux ligne


X,y=v[:,1].T,v[:,0].T# v[:,1]->land_annual;v[:,0]->year

# y->target,X#features
# dans notre exemple on  va utiliser un model polynomial de degrees 3 

y=y.reshape(y.shape[0],1)
X=X.reshape(X.shape[0],1)

#la fonction va transformer notre  matrice X -> matrice  polynomiale de degree 3  X[X^3-->1 ere ligne,X^2-> 2 eme,X-->3 eme colonne,1-> 4 eme colonne]
def poly_features(x, degree):
     X_poly = np.zeros(shape=(len(x), degree))
     for i in range(0, degree):
        X_poly[:,(degree-1) -i] = x.squeeze() ** (i + 1);
     z=np.ones((x.shape[0],1))
     X_poly=np.hstack((X_poly,z))
   
     return X_poly

r=poly_features(X,3) 

r.shape #->(139,4)

plt.plot(X,y,'xr')



theta=np.random.randn(4,1)

#----------------------------------------Modèle----------------------------------------------------
#---------------------------------------------------------------------------------------------------
def model(X,theta) :
    
     return X.dot(theta)

#---------------------------------------- function de cout ----------------------------------------------
#---------------------------------------------------------------------------------------------------
def cost_function(X,y,theta):
    m=len(y)
    return (1/2*m)*np.sum((model(X,theta)-y)**2)
  

#----------------------------------------gradient-----------------------------------------------
#---------------------------------------------------------------------------------------------------
def grad(X,y,theta) :
    m=len(y)
    return  1/m*X.T.dot(model(X,theta)-y)
grad(r,y,theta)

#----------------------------------------Mini batch gradient descent----------------------------------------
#---------------------------------------------------------------------------------------------------------
def gradient_descent(X,y,theta,learning_rate,n_iterations):
  
    cost_history=np.zeros(n_iterations)

    for i in range(0,n_iterations):
        
        theta=theta-learning_rate*grad(X,y,theta)
        cost_history[i]=cost_function(X,y,theta)

        
    return theta, cost_history


#----------------------------------------Entrainement----------------------------------------------
#---------------------------------------------------------------------------------------------------
theta_final,cost_history=gradient_descent(r,y,theta,learning_rate=1,n_iterations=1000)

plt.grid()
plt.subplot(2,1,1)
plt.plot(X,y,'Xr')

plt.scatter(X,model(r,theta_final))
plt.subplot(2,1,2)
plt.plot(cost_history)

