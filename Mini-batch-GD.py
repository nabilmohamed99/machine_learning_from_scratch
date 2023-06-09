import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
x, y = make_regression(n_samples=1000, n_features=1, noise=20)
y=y.reshape(y.shape[0],1) 
X=np.hstack((x, np.ones(x.shape))) 
theta=np.random.randn(2,1)
#----------------------------------------Modèle----------------------------------------------------
#---------------------------------------------------------------------------------------------------
def model(X,theta): 
    return X.dot(theta)
#----------------------------------------Cost function----------------------------------------------
#---------------------------------------------------------------------------------------------------
def cost_function(X,y,theta):
    m=len(y) 
    return 1/(2*m)*np.sum(model(X,theta)-y)**2
print (cost_function(X,y,theta)) 

#----------------------------------------Le gradient-----------------------------------------------
#---------------------------------------------------------------------------------------------------
def grad(X,y,theta):
    m=len(y)
    return 1/m*X.T.dot(model(X, theta)-y)


#----------------------------------------Mini batch gradient descent----------------------------------------
#---------------------------------------------------------------------------------------------------------
def minibatch_gradient_descent(X,y,theta,learning_rate, n_iterations, batch_size):
    m = len(y)
    cost_history = np.zeros(n_iterations)
    
    for i in range(n_iterations):
        indices = np.random.permutation(m)
        X = X[indices]
        y = y[indices]
        for i in range(0,m,batch_size):
            X_i = X[i:i+batch_size]
            y_i = y[i:i+batch_size]
            theta=theta-learning_rate*grad(X_i,y_i,theta)
        cost_history[i]=cost_function(X_i,y_i,theta)    
    return theta, cost_history    

#----------------------------------------Entrainement----------------------------------------------
#---------------------------------------------------------------------------------------------------
theta_final,cost_function = minibatch_gradient_descent(X,y,theta,learning_rate=1.5,n_iterations=1000, batch_size=50)
print(theta_final)  
prediction=model(X,theta_final)
plt.scatter(x,y,c='lightblue')
plt.plot (x,prediction,c='r') 
plt.grid()
plt.title("Régression linéaire \n par mini batch gradient descent")

#----------------------------------------Coefficient de détermination----------------------------------------
#-----------------------------------------------------------------------------------------------------------
def coe_det(y, pred) :
    u=((y-pred)**2).sum()
    v=((y-y.mean())**2).sum()
    return 1-u/v
print (coe_det(y, prediction))