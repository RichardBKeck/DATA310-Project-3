# DATA310-Project-3
This project was compleated as a part of DATA 310 at William & Mary. The project consisted of one programming assignments. In addition to the README file, this repository contains a folder which has the .ipynb files for all questions.

## Question One

Question One asks the following: 

"For this project, you will apply the three grid search algorithms we studied, such as the Genetic Algorithm, Particle Swarm Optimization, and Simulated Annealing, to determine the best choice of hyperparameters for the Elastic Net and Support Vector Regressor method. Specifically, in the case of the Elastic Net, we look for the best alpha and l1_ratio, and in the case of SVR, we search for the best combination of epsilon and C values. The metric for decision is a 10-fold cross-validated MSE (external validation)."

For this project, I imported a given csv file using the following code.
```Python
df = pd.read_csv('drive/MyDrive/DATA 310/Datasets/AirI.csv',header=None)
Xf = df.values
X = Xf[:,1:]
y = Xf[:,0]
```

From there I wrote two objective functions. They looked similar to each other, with the only difference being in the model line. 

This block is the objective function for the Elastic Net method:
```Python
def objective_ENet(h): # h is a two column matrix
  kf = KFold(n_splits=10,shuffle=True,random_state=123)
  scale = StandardScaler()
  output = [] 
  a = h[0]
  l = h[1]
  model = ElasticNet(alpha=a,l1_ratio=l,max_iter=5000)
  PE = []
  for idxtrain, idxtest in kf.split(X):
    Xtrain = X[idxtrain]
    ytrain = y[idxtrain]
    Xtrain_s = scale.fit_transform(Xtrain)
    Xtest = X[idxtest]
    ytest = y[idxtest]
    Xtest_s = scale.transform(Xtest)
    model.fit(Xtrain_s,ytrain)
    PE.append(MSE(ytest,model.predict(Xtest_s)))
  return np.mean(PE)
```

This block is the objective function for the Support Vector Regressor Method:
```Python
def objective_SVR(h): # h is a two column matrix
  kf = KFold(n_splits=10,shuffle=True,random_state=123)
  scale = StandardScaler()
  output = [] 
  Reg_Param = h[0] # column matrix 
  e = h[1]
  model = SVR(C=Reg_Param,epsilon=e,max_iter=5000)
  PE = []
  for idxtrain, idxtest in kf.split(X):
    Xtrain = X[idxtrain]
    ytrain = y[idxtrain]
    Xtrain_s = scale.fit_transform(Xtrain)
    Xtest = X[idxtest]
    ytest = y[idxtest]
    Xtest_s = scale.transform(Xtest)
    model.fit(Xtrain_s,ytrain)
    PE.append(MSE(ytest,model.predict(Xtest_s)))
  return np.mean(PE)
```
