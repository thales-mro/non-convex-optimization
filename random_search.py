import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import scipy.stats as stats
from sklearn.utils.fixes import loguniform

#Dados de treino
Xtreino = np.load("Xtreino5.npy")
Ytreino = np.load("ytreino5.npy")

#Dados de teste
Xteste = np.load("Xteste5.npy")
Yteste = np.load("yteste5.npy") 

def erro_absoluto_medio(X, Y): #Verificar se isso ta certo
  n = np.size(X)
  return (np.sum(abs(X-Y)))/n

#Random search -- Utilizando o sklearn
parameters = [{#'kernel': ['rbf'], 
              #'gamma': np.logspace(start=-15, stop=3, num=5, endpoint=True, base=2),
              #'gamma': np.linspace(2**(-15),2**3, 5), # version with linspace
              #'C': np.linspace(2**(-5), 2**(15),5), # version with linspace
              #'C': np.logspace(start=-5, stop=15, num=5, endpoint=True, base=2),
              'gamma': loguniform(2e-15, 2e3),
              'C': loguniform(2e-5, 2e15),
              'epsilon': stats.uniform(0.05, 1)}]
              #'epsilon': np.linspace(0.05,1.0, 5)}]
clf = SVR(kernel='rbf')
rsearch = RandomizedSearchCV(clf, parameters, n_jobs=-1, n_iter=125, scoring='neg_mean_absolute_error')
print("whaaaaaaat")
rsearch.fit(Xtreino, Ytreino)
mae = erro_absoluto_medio(rsearch.predict(Xteste),Yteste)
print("Minimum Mean Absolute Error: ", mae)
print(rsearch.best_params_)

def objective():
    clf = SVR(kernel='rbf', )