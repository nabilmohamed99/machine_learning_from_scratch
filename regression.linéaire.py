#----------------------------------------Régression linéaire----------------------------------------
#---------------------------------------------------------------------------------------------------

import numpy as np
#importer les données de la base de données sklearn
from sklearn.datasets import make_regression

# le modèle matplotlib.pyplot pour créer des graphes et visualiser les données
import matplotlib.pyplot as plt
# 1.Il nous faut un dataset (nbre d'échantillons, nbre de var, un bruit)
x, y = make_regression(n_samples=100, n_features=1, noise=10)
#print (x)
#Affcher le nuage des points
#plt.scatter(x, y) #scatter une des fonctionnalité de matplotlib pour afficher les nuages de points
y=y.reshape(y.shape[0],1) #reshape(100,1) ou mieux (y.shape[0], 1)=(nbre de lignes existant, 1)
#Il nous faut mnt une matrice X avec colonne de biais
X=np.hstack((x, np.ones(x.shape))) #à coté de X une colonne de 1 de même taille de x (x.shape)
#print (X.shape)---->(100,2)
#print (X)
# le vecteur theta qu'on connait pas, alors on va l'initialiser avec des paramètres aléatoires gr^ce au tableau np.random
theta=np.random.randn(2,1)
#print (theta.shape)
#print (theta) 
#2. maintenant on a X,y et theta alors on peut passer au modèle

#----------------------------------------Modèle----------------------------------------
#---------------------------------------------------------------------------------------------------
def model(X,theta): #créer une fonction modèle f(X)=X.theta
    return X.dot(theta)
# print (model(X,theta)) il faut toujours tester
#plt.plot(x, model(X,theta), c='r') #Afficher notre modèle c='la couleur : r,g,b...'
    
#----------------------------------------Cost function----------------------------------------
#---------------------------------------------------------------------------------------------------
#3. La fonction coût
def cost_function(X,y,theta):
    m=len(y) # m ici c'est 100, en général c'est la longieur du vecteur Y
    return 1/(2*m)*np.sum(model(X,theta)-y)**2
print (cost_function(X,y,theta)) #tester l'erreur actuelle-->un coût trés élevé et on voudra qu'il soit proche de 0

#----------------------------------------La descente de gradient----------------------------------------
#---------------------------------------------------------------------------------------------------
#4. Le gradient
def grad(X,y,theta):
    m=len(y)
    return 1/m*X.T.dot(model(X, theta)-y)
#print (grad(X,y,theta))--->Testez
#La descente de gradient
def gradient_descent(X,y,theta,learning_rate,n_iterations):
    cost_history=np.zeros(n_iterations)
    for i in range(0,n_iterations):
        theta=theta-learning_rate*grad(X,y,theta)#on va mettre à jours theta de 0 jusqu'à n itération
        cost_history[i]=cost_function(X,y,theta)
    return theta, cost_history

#----------------------------------------L'entrainement----------------------------------------
#---------------------------------------------------------------------------------------------------
#Machine Learning=Entrainement: on appelle la descente de gradient pour donner le theta optimal
theta_final,cost_function = gradient_descent(X,y,theta,learning_rate=0.1,n_iterations=1000)
#print(theta_final)---->testez  
prediction=model(X,theta_final)
plt.scatter(x,y)
plt.plot (x,prediction,c='r') #on peut faire mieux, soit augmner le nombre d'itérations ou bien augmenter ou diminuer le learning_rate
#Vous avez un trop petit pas, changez learning_rate=0.1
#courbe d'aprentissage : comment on peut voir comment la machine a réussi à apprendre
#print (model(X,theta_final))
#plt.plot (range(1000), cost_function)
#Coefficient de determination R2 (carré) pour montrer la perfermance du modèle, plus qu'il est proche de 1
#plus que le modèle rentre dans le nuage de point
def coe_det(y, pred) :
    u=((y-pred)**2).sum()
    v=((y-y.mean())**2).sum()
    return 1-u/v
print (coe_det(y, prediction))

    

